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
"""CPU-only tests for GuidedDecoder's async-compile integration.

These cover the invariants that make guided decoding safe with one-model
speculative decoding (CapturableGuidedDecoder), where grammar work runs inside
CUDA host callbacks: compilation must happen off the forward path, matcher
attach must be idempotent and non-raising, and all failures must surface
through drain_async_failures instead of exceptions.
"""

from collections import deque

import pytest
import torch
from guided_test_utils import PARAMS, FakeFactory, FakeMatcher, make_compiler, wait_for

from tensorrt_llm._torch.pyexecutor.grammar_compiler import CompileState
from tensorrt_llm._torch.pyexecutor.guided_decoder import (
    CapturableGuidedDecoder,
    GuidedDecoder,
    GuidedRequest,
    GuidedRequests,
)

MAX_NUM_SEQUENCES = 8
VOCAB_SIZE = 128


def make_decoder(decoder_cls=GuidedDecoder, factory=None) -> GuidedDecoder:
    """Construct a decoder without CUDA/tokenizer dependencies."""
    decoder = decoder_cls.__new__(decoder_cls)
    decoder.max_num_sequences = MAX_NUM_SEQUENCES
    decoder.max_num_draft_tokens = 2
    decoder.vocab_size_padded = VOCAB_SIZE
    decoder.rank = 0
    decoder.grammar_matcher_factory = factory or FakeFactory()
    decoder.grammar_matchers = [None] * MAX_NUM_SEQUENCES
    decoder.num_advanced_tokens = [0] * MAX_NUM_SEQUENCES
    decoder.num_guided_tokens = [0] * MAX_NUM_SEQUENCES
    decoder.num_advanced_draft_tokens = [0] * MAX_NUM_SEQUENCES
    decoder.is_draft_terminated = [False] * MAX_NUM_SEQUENCES
    num_bitmask_rows = MAX_NUM_SEQUENCES * (decoder.max_num_draft_tokens + 1)
    decoder.bitmask_host = torch.zeros(
        num_bitmask_rows, decoder.bitmask_size, dtype=decoder.bitmask_dtype
    )
    decoder.token_mask_host = torch.zeros(num_bitmask_rows, dtype=decoder.token_mask_dtype)
    decoder.requests = None
    # Mirrors GuidedDecoder.__init__ (bypassed here for its CUDA state).
    decoder.grammar_compiler = make_compiler(
        decoder.grammar_matcher_factory, allow_sync_fallback=decoder_cls._allow_sync_compile
    )
    decoder._matcher_owner = [None] * MAX_NUM_SEQUENCES
    decoder._async_failures = deque()
    return decoder


def make_guided_request(req_id=1, seq_slot=0, **kwargs) -> GuidedRequest:
    return GuidedRequest(
        guided_decoding_params=PARAMS, request_id=req_id, seq_slot=seq_slot, **kwargs
    )


def submit_and_wait(decoder, req_id=7, state=CompileState.READY):
    decoder.grammar_compiler.submit(req_id, PARAMS)
    wait_for(lambda: decoder.grammar_compiler.poll(req_id) == state)


class TestDisaggGenAttach:
    def _attach_batch(self, decoder, req_id=7, seq_slot=3):
        req = make_guided_request(
            req_id=req_id,
            seq_slot=seq_slot,
            is_generation_only_first_iteration=True,
            is_generation_in_progress_state=True,
            new_token=11,
        )
        requests = GuidedRequests(
            requests=[req], num_contexts=0, num_generations=1, max_num_draft_tokens=2
        )
        decoder._init_disagg_gen_requests(requests)
        return req

    def test_attach_precompiled_matcher(self):
        decoder = make_decoder(CapturableGuidedDecoder)
        submit_and_wait(decoder, req_id=7)
        self._attach_batch(decoder, req_id=7, seq_slot=3)
        assert isinstance(decoder.grammar_matchers[3], FakeMatcher)
        assert decoder._matcher_owner[3] == 7
        assert decoder.drain_async_failures() == []

    def test_attach_is_idempotent_under_graph_replay(self):
        decoder = make_decoder(CapturableGuidedDecoder)
        submit_and_wait(decoder, req_id=7)
        self._attach_batch(decoder, req_id=7, seq_slot=3)
        matcher = decoder.grammar_matchers[3]
        # CUDA-graph warmup/replay re-executes the callback for the same batch.
        self._attach_batch(decoder, req_id=7, seq_slot=3)
        assert decoder.grammar_matchers[3] is matcher
        assert decoder.drain_async_failures() == []

    def test_attach_without_precompile_records_failure(self):
        decoder = make_decoder(CapturableGuidedDecoder)
        self._attach_batch(decoder, req_id=7, seq_slot=3)
        assert decoder.grammar_matchers[3] is None
        failures = decoder.drain_async_failures()
        assert len(failures) == 1
        assert failures[0][0] == 7
        assert "no precompiled grammar" in failures[0][1]
        # A failed attach must not re-fail on graph replay of the same batch:
        # the slot owner is recorded even for a failed attach.
        assert decoder._matcher_owner[3] == 7
        self._attach_batch(decoder, req_id=7, seq_slot=3)
        assert decoder.drain_async_failures() == []

    def test_attach_failed_compile_records_failure(self):
        decoder = make_decoder(CapturableGuidedDecoder, factory=FakeFactory(behavior="raise"))
        submit_and_wait(decoder, req_id=7, state=CompileState.FAILED)
        self._attach_batch(decoder, req_id=7, seq_slot=3)
        assert decoder.grammar_matchers[3] is None
        failures = decoder.drain_async_failures()
        assert len(failures) == 1
        assert failures[0][0] == 7
        assert "unsatisfiable schema" in failures[0][1]

    def test_drain_filters_by_synced_batch(self):
        # Rank-consistency contract: only records for the just-synchronized
        # batch are drained; records for still-active requests from a later,
        # unsynced batch are requeued; records for dead requests are dropped;
        # None-keyed (batch-wide) records always drain.
        decoder = make_decoder(CapturableGuidedDecoder)
        decoder._record_async_failure(7, "in batch")
        decoder._record_async_failure(8, "later batch, still active")
        decoder._record_async_failure(9, "already terminated")
        decoder._record_async_failure(None, "batch-wide")
        drained = decoder.drain_async_failures(allowed_req_ids={7}, active_req_ids={7, 8})
        assert sorted(str(r[0]) for r in drained) == ["7", "None"]
        # The requeued record for request 8 drains once its batch syncs.
        drained = decoder.drain_async_failures(allowed_req_ids={8}, active_req_ids={8})
        assert [r[0] for r in drained] == [8]
        assert decoder.drain_async_failures() == []


class TestHostfuncGuard:
    def test_hostfunc_on_error_records_and_swallows(self, monkeypatch):
        # Run hostfuncs synchronously on this thread, the way the CUDA
        # callback thread would; exceptions must be recorded, never raised.
        from tensorrt_llm._torch import hostfunc as hostfunc_module

        monkeypatch.setattr(
            hostfunc_module, "launch_hostfunc", lambda fn, *args, **kwargs: fn(*args, **kwargs)
        )
        decoder = make_decoder(CapturableGuidedDecoder)
        decoder.requests_hostfunc = None  # raises AttributeError on use
        decoder.build()
        failures = decoder.drain_async_failures()
        assert len(failures) == 1
        assert failures[0][0] is None
        assert "internal error in build" in failures[0][1]

    def test_record_hostfunc_error_message_excludes_traceback(self):
        # The recorded message reaches client responses; the traceback must
        # stay in the server log only.
        decoder = make_decoder(CapturableGuidedDecoder)
        try:
            raise ValueError("boom")
        except ValueError as e:
            decoder._record_hostfunc_error("build", e)
        failures = decoder.drain_async_failures()
        assert len(failures) == 1
        assert failures[0][0] is None
        assert "boom" in failures[0][1]
        assert "Traceback" not in failures[0][1]


class TestBuildAndRollback:
    def _build_batch(self, decoder, req_id=7, seq_slot=2):
        req = make_guided_request(
            req_id=req_id,
            seq_slot=seq_slot,
            is_context_init_state=True,
            is_last_context_chunk=True,
            new_token=5,
            draft_tokens=[],
        )
        requests = GuidedRequests(
            requests=[req], num_contexts=1, num_generations=0, max_num_draft_tokens=2
        )
        return decoder._build(requests)

    def test_build_matcher_init_uses_precompiled(self):
        decoder = make_decoder()
        submit_and_wait(decoder, req_id=7)
        failed = self._build_batch(decoder, req_id=7, seq_slot=2)
        assert failed == []
        assert isinstance(decoder.grammar_matchers[2], FakeMatcher)
        assert decoder.grammar_matcher_factory.create_calls == 1

    def test_build_sync_fallback_on_executor_thread_decoder(self):
        # Plain GuidedDecoder may compile synchronously when nothing was
        # precompiled (status quo); CapturableGuidedDecoder must not, since
        # its attach paths run inside CUDA host callbacks.
        decoder = make_decoder(GuidedDecoder)
        failed = self._build_batch(decoder, req_id=7, seq_slot=2)
        assert failed == []
        assert isinstance(decoder.grammar_matchers[2], FakeMatcher)

        capturable = make_decoder(CapturableGuidedDecoder)
        failed = self._build_batch(capturable, req_id=7, seq_slot=2)
        assert len(failed) == 1
        assert "no precompiled grammar" in failed[0][1]
        assert capturable.grammar_matcher_factory.create_calls == 0
        # The failed attach still claims the slot so a later matcher_advance
        # sees None rather than a previous occupant's matcher.
        assert capturable.grammar_matchers[2] is None
        assert capturable._matcher_owner[2] == 7

    def test_build_reports_failed_compile_per_request(self):
        decoder = make_decoder(factory=FakeFactory(behavior="raise"))
        submit_and_wait(decoder, req_id=7, state=CompileState.FAILED)
        failed = self._build_batch(decoder, req_id=7, seq_slot=2)
        assert len(failed) == 1
        assert failed[0][0] == 7
        assert "unsatisfiable schema" in failed[0][1]

    def test_rollback_error_recorded_not_raised(self):
        decoder = make_decoder()
        req = make_guided_request(
            req_id=7, seq_slot=2, is_generation_in_progress_state=True, num_accepted_draft_tokens=5
        )
        requests = GuidedRequests(
            requests=[req], num_contexts=0, num_generations=1, max_num_draft_tokens=2
        )
        decoder.grammar_matchers[2] = FakeMatcher()
        # num_advanced < 1 + num_accepted -> negative rollback -> must be
        # recorded as a failure, never raised (may run in a CUDA callback).
        decoder.num_advanced_tokens[2] = 1
        decoder._rollback_rejected_tokens(requests)
        failures = decoder.drain_async_failures()
        assert len(failures) == 1
        assert failures[0][0] == 7
        assert "Failed to rollback" in failures[0][1]

    def test_rollback_with_none_matcher_is_noop(self):
        decoder = make_decoder()
        req = make_guided_request(
            req_id=7, seq_slot=2, is_generation_in_progress_state=True, num_accepted_draft_tokens=0
        )
        requests = GuidedRequests(
            requests=[req], num_contexts=0, num_generations=1, max_num_draft_tokens=2
        )
        decoder.grammar_matchers[2] = None
        decoder.num_advanced_tokens[2] = 2
        decoder._rollback_rejected_tokens(requests)
        decoder._rollback_draft_tokens(requests)
        assert decoder.drain_async_failures() == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
