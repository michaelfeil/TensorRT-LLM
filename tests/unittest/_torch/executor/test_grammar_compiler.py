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
"""CPU-only tests for AsyncGrammarCompiler.

Covers the request-keyed async grammar-compile state machine used by guided
decoding.
"""

import threading
import time

import pytest
from guided_test_utils import PARAMS, FakeFactory, FakeMatcher, make_compiler, wait_for

from tensorrt_llm._torch.pyexecutor.grammar_compiler import CompileState, format_guided_error


class TestCompileStateMachine:
    def test_submit_then_ready(self):
        compiler = make_compiler()
        compiler.submit(7, PARAMS)
        wait_for(lambda: compiler.poll(7) == CompileState.READY)
        matcher, error_msg = compiler.take(7, PARAMS)
        assert isinstance(matcher, FakeMatcher)
        assert error_msg == ""
        # State is consumed on take.
        assert 7 not in compiler._ready

    def test_submit_is_idempotent(self):
        compiler = make_compiler()
        compiler.submit(7, PARAMS)
        compiler.submit(7, PARAMS)
        wait_for(lambda: compiler.poll(7) == CompileState.READY)
        compiler.submit(7, PARAMS)
        assert compiler.factory.create_calls == 1

    def test_unguided_submit_is_noop_and_polls_ready(self):
        compiler = make_compiler()
        compiler.submit(7, None)
        assert compiler.poll(7) == CompileState.READY
        assert compiler.factory.create_calls == 0

    def test_compile_failure_is_reported_not_raised(self):
        compiler = make_compiler(FakeFactory(behavior="raise"))
        compiler.submit(7, PARAMS)
        wait_for(lambda: compiler.poll(7) == CompileState.FAILED)
        matcher, error_msg = compiler.take(7, PARAMS)
        assert matcher is None
        assert "unsatisfiable schema" in error_msg

    def test_slow_compile_is_pending_then_ready(self):
        block = threading.Event()
        compiler = make_compiler(FakeFactory(block_event=block))
        compiler.submit(7, PARAMS)
        assert compiler.poll(7) == CompileState.PENDING
        block.set()
        wait_for(lambda: compiler.poll(7) == CompileState.READY)

    def test_compile_timeout_fails_request(self):
        block = threading.Event()
        compiler = make_compiler(FakeFactory(block_event=block))
        compiler._compile_timeout_s = 0.0
        compiler.submit(7, PARAMS)
        time.sleep(0.01)
        assert compiler.poll(7) == CompileState.FAILED
        matcher, error_msg = compiler.take(7, PARAMS)
        assert matcher is None
        assert "did not finish within" in error_msg
        block.set()

    def test_take_timeout_is_cached_as_failed(self):
        # A timed-out take() must leave consistent state: later polls report
        # FAILED rather than "never submitted" READY, and discard() cleans up.
        block = threading.Event()
        compiler = make_compiler(FakeFactory(block_event=block))
        compiler._take_wait_s = 0.05
        compiler.submit(7, PARAMS)
        matcher, error_msg = compiler.take(7, PARAMS)
        assert matcher is None
        assert "still pending" in error_msg
        assert compiler.poll(7) == CompileState.FAILED
        matcher, second_msg = compiler.take(7, PARAMS)
        assert matcher is None
        assert second_msg == error_msg
        compiler.discard(7)
        assert 7 not in compiler._failed
        block.set()

    def test_take_exception_is_cached_as_failed(self):
        block = threading.Event()
        compiler = make_compiler(FakeFactory(behavior="raise", block_event=block))
        compiler.submit(7, PARAMS)
        threading.Timer(0.02, block.set).start()
        matcher, error_msg = compiler.take(7, PARAMS)
        assert matcher is None
        assert "unsatisfiable schema" in error_msg
        assert compiler.poll(7) == CompileState.FAILED

    def test_shutdown_recreates_pool_on_next_submit(self):
        # The KV-estimation probe executor shuts the shared compiler down
        # mid-init; the final executor must still be able to compile.
        compiler = make_compiler()
        compiler.submit(7, PARAMS)
        wait_for(lambda: compiler.poll(7) == CompileState.READY)
        compiler.shutdown()
        compiler.shutdown()  # idempotent
        compiler.submit(8, PARAMS)
        wait_for(lambda: compiler.poll(8) == CompileState.READY)

    def test_discard_clears_all_state(self):
        compiler = make_compiler()
        compiler.submit(7, PARAMS)
        wait_for(lambda: compiler.poll(7) == CompileState.READY)
        compiler.discard(7)
        assert 7 not in compiler._ready
        assert 7 not in compiler._pending
        assert 7 not in compiler._failed

    def test_mark_failed_pending_fails_fast(self):
        # Broadcast verdict from rank 0: a still-pending local compile must
        # fail instantly at take instead of blocking for the bounded wait
        # inside the CUDA callback.
        block = threading.Event()
        compiler = make_compiler(FakeFactory(block_event=block))
        compiler.submit(7, PARAMS)
        assert compiler.poll(7) == CompileState.PENDING
        compiler.mark_failed(7, "Guided decoding error: rank-0 budget.")
        assert compiler.poll(7) == CompileState.FAILED
        start = time.monotonic()
        matcher, error_msg = compiler.take(7, PARAMS)
        assert time.monotonic() - start < 1.0
        assert matcher is None
        assert "rank-0 budget" in error_msg
        block.set()

    def test_mark_failed_drops_ready_matcher(self):
        # If the local compile already finished, the matcher must still be
        # dropped: rank 0 failed the request and will terminate it, so
        # attaching here would diverge batch composition across ranks.
        compiler = make_compiler()
        compiler.submit(7, PARAMS)
        wait_for(lambda: compiler.poll(7) == CompileState.READY)
        compiler.mark_failed(7, "Guided decoding error: rank-0 budget.")
        assert 7 not in compiler._ready
        matcher, error_msg = compiler.take(7, PARAMS)
        assert matcher is None
        assert "rank-0 budget" in error_msg

    def test_mark_failed_keeps_local_message(self):
        # A locally recorded failure (e.g. the real compile exception) is
        # more specific than the broadcast fallback text; keep it.
        compiler = make_compiler(FakeFactory(behavior="raise"))
        compiler.submit(7, PARAMS)
        wait_for(lambda: compiler.poll(7) == CompileState.FAILED)
        compiler.mark_failed(7, "Guided decoding error: rank-0 budget.")
        matcher, error_msg = compiler.take(7, PARAMS)
        assert matcher is None
        assert "unsatisfiable schema" in error_msg

    def test_take_sync_fallback_gated(self):
        # With sync fallback (executor-thread decoder), an unsubmitted request
        # compiles on the spot; without it (CUDA-callback attach), it fails.
        with_fallback = make_compiler(allow_sync_fallback=True)
        matcher, error_msg = with_fallback.take(7, PARAMS)
        assert isinstance(matcher, FakeMatcher)

        without_fallback = make_compiler(allow_sync_fallback=False)
        matcher, error_msg = without_fallback.take(7, PARAMS)
        assert matcher is None
        assert "no precompiled grammar" in error_msg
        assert without_fallback.factory.create_calls == 0

    def test_take_waits_bounded_for_pending_compile(self):
        block = threading.Event()
        compiler = make_compiler(FakeFactory(block_event=block))
        compiler._take_wait_s = 0.05
        compiler.submit(7, PARAMS)
        matcher, error_msg = compiler.take(7, PARAMS)
        assert matcher is None
        assert "still pending" in error_msg
        block.set()


def test_format_guided_error_is_idempotent():
    assert format_guided_error(ValueError("boom")) == "Guided decoding error: boom"
    assert (
        format_guided_error(ValueError("Guided decoding error: boom"))
        == "Guided decoding error: boom"
    )
    assert format_guided_error(ValueError()) == "Guided decoding error: ValueError"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
