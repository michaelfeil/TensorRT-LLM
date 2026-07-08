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
"""Shared CPU-only fakes for the guided-decoding test suite."""

import threading
import time

from tensorrt_llm._torch.pyexecutor.grammar_compiler import AsyncGrammarCompiler

PARAMS = object()


class FakeMatcher:
    def __init__(self):
        self.accepted_tokens = []
        self.rollback_calls = []
        self.terminated = False

    def accept_token(self, token_id: int) -> bool:
        self.accepted_tokens.append(token_id)
        return True

    def is_terminated(self) -> bool:
        return self.terminated

    def fill_next_token_bitmask(self, bitmask, offset) -> None:
        pass

    def rollback(self, num_tokens: int) -> None:
        self.rollback_calls.append(num_tokens)


class FakeFactory:
    """Grammar matcher factory with scriptable behavior per create() call."""

    def __init__(self, behavior="ok", block_event: threading.Event = None):
        self.behavior = behavior
        self.block_event = block_event
        self.create_calls = 0

    def create(self, guided_decoding_params):
        self.create_calls += 1
        if self.block_event is not None:
            self.block_event.wait()
        if self.behavior == "raise":
            raise ValueError("unsatisfiable schema")
        return FakeMatcher()


def make_compiler(factory=None, allow_sync_fallback=True) -> AsyncGrammarCompiler:
    compiler = AsyncGrammarCompiler(
        factory or FakeFactory(), allow_sync_fallback=allow_sync_fallback
    )
    # Hermetic against TRTLLM_GUIDED_* env overrides in the test environment.
    compiler._compile_timeout_s = 60.0
    compiler._take_wait_s = 10.0
    return compiler


def wait_for(predicate, timeout_s=10.0):
    deadline = time.monotonic() + timeout_s
    while not predicate():
        assert time.monotonic() < deadline, "timed out waiting for condition"
        time.sleep(0.01)
