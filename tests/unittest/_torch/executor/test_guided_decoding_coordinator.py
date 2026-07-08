# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU-only tests for GuidedDecodingCoordinator.

Covers schedule gating on grammar compiles (including the rank-0 broadcast for
non-attention-DP TP) and the park/terminate lifecycle of guided-decoding
failures.
"""

import threading
import time
from collections import deque
from types import SimpleNamespace

import pytest
from guided_test_utils import PARAMS, FakeFactory, make_compiler, wait_for

from tensorrt_llm._torch.pyexecutor.grammar_compiler import CompileState
from tensorrt_llm._torch.pyexecutor.guided_decoder import GuidedDecoder
from tensorrt_llm._torch.pyexecutor.guided_decoding_coordinator import GuidedDecodingCoordinator


class FakeDist:
    """Records tp_broadcast calls; on non-root ranks returns recv_payload."""

    def __init__(self, tp_size=1, tp_rank=0, pp_size=1, recv_payload=None):
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.pp_size = pp_size
        self.recv_payload = recv_payload
        self.broadcasts = []

    def tp_broadcast(self, obj, root=0, **kwargs):
        self.broadcasts.append((obj, root))
        return obj if obj is not None else self.recv_payload


def make_request(req_id, guided=True, last_chunk_context=False, disagg_gen_first=False):
    return SimpleNamespace(
        py_request_id=req_id,
        guided_decoding_params=PARAMS if guided else None,
        is_context_init_state=last_chunk_context,
        is_last_context_chunk=last_chunk_context,
        is_disagg_generation_transmission_complete=disagg_gen_first,
        py_guided_compile_pending=False,
    )


def make_batch(*requests):
    return SimpleNamespace(all_requests=lambda: list(requests))


def make_coordinator(factory=None, dist=None, enable_attention_dp=False):
    """Coordinator over a minimal decoder (no CUDA/tokenizer dependencies)."""
    decoder = GuidedDecoder.__new__(GuidedDecoder)
    decoder.grammar_compiler = make_compiler(factory, allow_sync_fallback=False)
    decoder._async_failures = deque()
    coordinator = GuidedDecodingCoordinator(decoder, dist or FakeDist(), enable_attention_dp)
    coordinator._defer_timeout_s = 8.0  # hermetic against env overrides
    return coordinator, decoder


class TestDisabled:
    def test_all_methods_noop_without_decoder(self):
        coordinator = GuidedDecodingCoordinator(None, FakeDist(), False)
        request = make_request(7, last_chunk_context=True)
        coordinator.start_compiles([request])
        assert request.py_guided_compile_pending is False
        context, generation = coordinator.filter_schedulable([request], [])
        assert context == [request] and generation == []
        coordinator.drain_failures(make_batch(request), [request])
        assert coordinator.take_failures(make_batch(request), [request]) == {}
        coordinator.release(request)


class TestStartCompiles:
    def test_submits_guided_requests_only(self):
        coordinator, decoder = make_coordinator()
        guided = make_request(7)
        unguided = make_request(8, guided=False)
        coordinator.start_compiles([guided, unguided])
        assert guided.py_guided_compile_pending is True
        assert unguided.py_guided_compile_pending is False
        wait_for(lambda: decoder.grammar_compiler.poll(7) == CompileState.READY)
        assert decoder.grammar_compiler.factory.create_calls == 1


class TestScheduleGating:
    def test_defers_pending_compile(self):
        block = threading.Event()
        coordinator, decoder = make_coordinator(FakeFactory(block_event=block))
        request = make_request(7, last_chunk_context=True)
        coordinator.start_compiles([request])
        context, generation = coordinator.filter_schedulable([request], [])
        assert context == [] and generation == []
        assert request.py_guided_compile_pending is True
        block.set()
        wait_for(lambda: decoder.grammar_compiler.poll(7) == CompileState.READY)
        context, _ = coordinator.filter_schedulable([request], [])
        assert context == [request]
        assert request.py_guided_compile_pending is False

    def test_defer_budget_anchored_at_first_deferral(self):
        # Compile time overlapping prefill must not count against the defer
        # budget; only time actually spent deferred does.
        block = threading.Event()
        coordinator, decoder = make_coordinator(FakeFactory(block_event=block))
        coordinator._defer_timeout_s = 0.05
        request = make_request(7, last_chunk_context=True)
        coordinator.start_compiles([request])
        time.sleep(0.1)  # longer than the defer budget, but not yet deferred
        context, _ = coordinator.filter_schedulable([request], [])
        assert context == []  # deferred: the budget starts now
        time.sleep(0.1)
        context, _ = coordinator.filter_schedulable([request], [])
        assert context == [request]  # past the budget: force-failed
        matcher, error_msg = decoder.grammar_compiler.take(7, PARAMS)
        assert matcher is None
        assert "did not finish within" in error_msg
        block.set()

    def test_failed_compile_is_scheduled(self):
        # A failed compile must be scheduled so matcher attach fails and
        # produces the per-request error response.
        coordinator, decoder = make_coordinator(FakeFactory(behavior="raise"))
        request = make_request(7, last_chunk_context=True)
        coordinator.start_compiles([request])
        wait_for(lambda: decoder.grammar_compiler.poll(7) == CompileState.FAILED)
        context, _ = coordinator.filter_schedulable([request], [])
        assert context == [request]
        assert request.py_guided_compile_pending is False

    def test_gates_only_matcher_attach_steps(self):
        # Earlier context chunks and in-progress generation requests do not
        # need the matcher yet and must not be deferred.
        block = threading.Event()
        coordinator, _ = make_coordinator(FakeFactory(block_event=block))
        early_chunk = make_request(7)  # context, not last chunk
        in_progress_gen = make_request(8)  # generation, matcher attached
        disagg_first = make_request(9, disagg_gen_first=True)
        coordinator.start_compiles([early_chunk, in_progress_gen, disagg_first])
        context, generation = coordinator.filter_schedulable(
            [early_chunk], [in_progress_gen, disagg_first]
        )
        assert context == [early_chunk]
        assert generation == [in_progress_gen]
        assert disagg_first.py_guided_compile_pending is True
        block.set()

    def test_rank_local_paths_do_not_broadcast(self):
        for dist, adp in ((FakeDist(tp_size=1), False), (FakeDist(tp_size=4), True)):
            coordinator, _ = make_coordinator(dist=dist, enable_attention_dp=adp)
            request = make_request(7, last_chunk_context=True)
            coordinator.start_compiles([request])
            coordinator.filter_schedulable([request], [])
            assert dist.broadcasts == []

    def test_pp_bails_out_without_gating(self):
        block = threading.Event()
        dist = FakeDist(tp_size=2, pp_size=2)
        coordinator, _ = make_coordinator(FakeFactory(block_event=block), dist=dist)
        request = make_request(7, last_chunk_context=True)
        coordinator.start_compiles([request])
        context, _ = coordinator.filter_schedulable([request], [])
        assert context == [request]
        assert dist.broadcasts == []
        block.set()

    def test_tp_skips_broadcast_without_gated_requests(self):
        # Batch composition is replicated on non-attention-DP TP ranks, so
        # when nothing is at a matcher-attach step every rank skips the
        # broadcast together instead of exchanging empty sets.
        dist = FakeDist(tp_size=2, tp_rank=0)
        coordinator, _ = make_coordinator(dist=dist)
        early_chunk = make_request(7)  # context, not last chunk
        in_progress_gen = make_request(8)  # generation, matcher attached
        context, generation = coordinator.filter_schedulable([early_chunk], [in_progress_gen])
        assert context == [early_chunk] and generation == [in_progress_gen]
        assert dist.broadcasts == []

    def test_tp_rank0_broadcasts_verdict(self):
        block = threading.Event()
        dist = FakeDist(tp_size=2, tp_rank=0)
        coordinator, _ = make_coordinator(FakeFactory(block_event=block), dist=dist)
        request = make_request(7, last_chunk_context=True)
        coordinator.start_compiles([request])
        context, _ = coordinator.filter_schedulable([request], [])
        assert context == []
        assert dist.broadcasts == [(({7}, set()), 0)]
        block.set()

    def test_tp_nonroot_applies_broadcast_verdict(self):
        # The non-root rank polls nothing itself: it defers what rank 0
        # deferred and force-fails what rank 0 failed, so matcher attach
        # fails instantly and identically on every rank.
        block = threading.Event()
        dist = FakeDist(tp_size=2, tp_rank=1, recv_payload=({8}, {7}))
        coordinator, decoder = make_coordinator(FakeFactory(block_event=block), dist=dist)
        failed_req = make_request(7, last_chunk_context=True)
        deferred_req = make_request(8, last_chunk_context=True)
        coordinator.start_compiles([failed_req, deferred_req])
        context, _ = coordinator.filter_schedulable([failed_req, deferred_req], [])
        assert dist.broadcasts == [(None, 0)]
        assert context == [failed_req]
        assert deferred_req.py_guided_compile_pending is True
        assert decoder.grammar_compiler.poll(7) == CompileState.FAILED
        matcher, error_msg = decoder.grammar_compiler.take(7, PARAMS)
        assert matcher is None
        assert "did not finish within 8s" in error_msg
        block.set()


class TestFailureLifecycle:
    def test_drain_parks_then_take_terminates(self):
        coordinator, decoder = make_coordinator()
        request = make_request(7)
        decoder._record_async_failure(7, "boom")
        coordinator.drain_failures(make_batch(request), [request])
        # Not terminated at drain time (the request may be in-flight in the
        # overlap scheduler's current iteration); take_failures pops it at
        # the post-sampling termination point.
        failures = coordinator.take_failures(make_batch(request), [request])
        assert failures == {7: "boom"}
        assert coordinator.take_failures(make_batch(request), [request]) == {}

    def test_drain_fans_out_batch_wide_records_to_guided_only(self):
        coordinator, decoder = make_coordinator()
        guided = make_request(7)
        unguided = make_request(8, guided=False)
        decoder._record_async_failure(None, "hostfunc broke")
        coordinator.drain_failures(make_batch(guided, unguided), [guided, unguided])
        failures = coordinator.take_failures(make_batch(guided, unguided), [guided, unguided])
        assert failures == {7: "hostfunc broke"}

    def test_drain_keeps_first_failure_per_request(self):
        coordinator, decoder = make_coordinator()
        request = make_request(7)
        decoder._record_async_failure(7, "first")
        decoder._record_async_failure(7, "second")
        coordinator.drain_failures(make_batch(request), [request])
        failures = coordinator.take_failures(make_batch(request), [request])
        assert failures == {7: "first"}

    def test_take_keeps_active_requests_outside_batch(self):
        coordinator, decoder = make_coordinator()
        parked = make_request(7)
        other = make_request(8)
        decoder._record_async_failure(7, "boom")
        coordinator.drain_failures(make_batch(parked), [parked, other])
        # A later batch without request 7: the parked failure must survive
        # while 7 is still active, and be dropped once it is gone.
        assert coordinator.take_failures(make_batch(other), [parked, other]) == {}
        assert coordinator.take_failures(make_batch(other), [other]) == {}
        assert coordinator.take_failures(make_batch(parked), [parked, other]) == {}

    def test_drain_without_batch_is_noop(self):
        coordinator, decoder = make_coordinator()
        decoder._record_async_failure(7, "boom")
        coordinator.drain_failures(None, [])
        # The record stays queued in the decoder for a later drain.
        assert len(decoder._async_failures) == 1

    def test_release_discards_compile_state(self):
        coordinator, decoder = make_coordinator()
        request = make_request(7)
        coordinator.start_compiles([request])
        wait_for(lambda: decoder.grammar_compiler.poll(7) == CompileState.READY)
        coordinator.release(request)
        assert 7 not in decoder.grammar_compiler._ready


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
