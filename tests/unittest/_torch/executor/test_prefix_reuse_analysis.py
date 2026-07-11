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
"""End-to-end tests for prefix-reuse analysis against the real nanobind KV cache manager.

Unlike the scheduler/router unit tests (which mock the manager), these tests
construct a real ``KVCacheManager`` with GPU pools and exercise:

* the request-based ``analyze_prefix_reuse(llm_request, beam)`` overload and its
  equivalence with the list-based ``analyze_prefix_reuse(unique_tokens, llm_request)``,
* ``estimate_reusable_prompt_len_with_summary`` (previously untested), and
* ``probe_prefix_match_length``,

all on top of a real store-and-release reuse tree, which also drives
``findReusableBlockMatches`` through actual C++.
"""

import pytest

import tensorrt_llm
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._utils import str_dtype_to_binding
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.sampling_params import SamplingParams

CacheTypeCpp = tensorrt_llm.bindings.internal.batch_manager.CacheType

TOKENS_PER_BLOCK = 4


@pytest.fixture
def kv_cache_manager():
    manager = KVCacheManager(
        KvCacheConfig(
            max_tokens=512,
            enable_block_reuse=True,
            enable_partial_reuse=False,
        ),
        CacheTypeCpp.SELF,
        num_layers=1,
        num_kv_heads=1,
        head_dim=64,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=64,
        max_batch_size=4,
        mapping=Mapping(),
        dtype=str_dtype_to_binding("float16"),
    )
    yield manager
    manager.shutdown()


def _make_request(request_id: int, tokens: list[int]) -> LlmRequest:
    return LlmRequest(
        request_id=request_id,
        max_new_tokens=1,
        input_tokens=tokens,
        sampling_config=SamplingConfig(SamplingParams()._get_sampling_config()),
        is_streaming=False,
    )


def _store_prefix(manager: KVCacheManager, request: LlmRequest) -> None:
    """Add a sequence and release it so its full context blocks land in the reuse tree."""
    manager.impl.add_sequence_batch([(request.py_request_id, request.py_prompt_len, 1)], [request])
    manager.free_resources(request)


def _block_token_ids(block_key) -> list[int]:
    return [t.token_id for t in block_key.unique_tokens]


# 17 tokens = 4 full blocks (of 4) + 1; only the 4 full blocks are storable/reusable.
PROMPT_A = list(range(100, 117))


def test_request_overload_matches_list_overload(kv_cache_manager):
    """Both analyze_prefix_reuse overloads must return identical summaries."""
    req_a = _make_request(0, PROMPT_A)
    _store_prefix(kv_cache_manager, req_a)

    # Shares the first 2 blocks with PROMPT_A, then diverges.
    prompt_b = PROMPT_A[:8] + list(range(500, 509))
    req_b = _make_request(1, prompt_b)

    summary_list = kv_cache_manager.impl.analyze_prefix_reuse(req_b.get_unique_tokens(0), req_b)
    summary_request = kv_cache_manager.impl.analyze_prefix_reuse(req_b)

    assert summary_request.reusable_blocks_all == summary_list.reusable_blocks_all == 2
    assert (
        summary_request.reusable_blocks_allocated
        == summary_list.reusable_blocks_allocated
        == 0  # req_a was released, so its blocks are cached but unreferenced
    )
    assert summary_list.first_new_block is not None
    assert summary_request.first_new_block is not None
    assert _block_token_ids(summary_request.first_new_block) == _block_token_ids(
        summary_list.first_new_block
    )
    # The first uncached block is the third block of prompt_b.
    assert _block_token_ids(summary_request.first_new_block) == prompt_b[8:12]


def test_full_prefix_match_and_no_match(kv_cache_manager):
    req_a = _make_request(0, PROMPT_A)
    _store_prefix(kv_cache_manager, req_a)

    # Identical prompt: every stored block matches.
    req_same = _make_request(1, list(PROMPT_A))
    summary = kv_cache_manager.impl.analyze_prefix_reuse(req_same)
    assert summary.reusable_blocks_all == 4

    # Disjoint prompt: nothing matches; the first block is immediately new.
    req_other = _make_request(2, list(range(900, 917)))
    summary = kv_cache_manager.impl.analyze_prefix_reuse(req_other)
    assert summary.reusable_blocks_all == 0
    assert summary.first_new_block is not None


def test_estimate_reusable_prompt_len_with_summary(kv_cache_manager):
    req_a = _make_request(0, PROMPT_A)
    _store_prefix(kv_cache_manager, req_a)

    # Full match: all 4 stored blocks are reusable and within the
    # (prompt_len - 1) recoverable cap of 16 tokens.
    req_same = _make_request(1, list(PROMPT_A))
    reusable_len, summary = kv_cache_manager.estimate_reusable_prompt_len_with_summary(req_same)
    assert reusable_len == 16
    assert summary is not None and summary.reusable_blocks_all == 4

    # No match: zero reusable tokens but a summary is still produced.
    req_other = _make_request(2, list(range(900, 917)))
    reusable_len, summary = kv_cache_manager.estimate_reusable_prompt_len_with_summary(req_other)
    assert reusable_len == 0
    assert summary is not None and summary.reusable_blocks_all == 0

    # Prompts of at most one full block have no recoverable prefix
    # (the last token's KV cannot be recovered): early-out with no summary.
    req_short = _make_request(3, PROMPT_A[:TOKENS_PER_BLOCK])
    reusable_len, summary = kv_cache_manager.estimate_reusable_prompt_len_with_summary(req_short)
    assert reusable_len == 0
    assert summary is None


def test_probe_prefix_match_length(kv_cache_manager):
    assert kv_cache_manager.probe_prefix_match_length(PROMPT_A) == 0

    req_a = _make_request(0, PROMPT_A)
    _store_prefix(kv_cache_manager, req_a)

    assert kv_cache_manager.probe_prefix_match_length(PROMPT_A) == 16
    assert kv_cache_manager.probe_prefix_match_length(PROMPT_A[:8]) == 8
    assert kv_cache_manager.probe_prefix_match_length(list(range(900, 917))) == 0


def test_reusable_blocks_allocated_tracks_live_references(kv_cache_manager):
    """reusable_blocks_allocated counts only tree blocks currently referenced by a sequence."""
    req_a = _make_request(0, PROMPT_A)
    _store_prefix(kv_cache_manager, req_a)

    # Shares the first 2 blocks with the stored prefix, then diverges.
    prompt_live = PROMPT_A[:8] + list(range(600, 609))
    req_live = _make_request(1, prompt_live)
    kv_cache_manager.impl.add_sequence_batch(
        [(req_live.py_request_id, req_live.py_prompt_len, 1)], [req_live]
    )

    req_query = _make_request(2, list(PROMPT_A))
    summary = kv_cache_manager.impl.analyze_prefix_reuse(req_query)
    assert summary.reusable_blocks_all == 4
    assert summary.reusable_blocks_allocated == 2  # blocks 0-1 claimed by req_live

    # After release nothing holds references. reusable_blocks_all is deliberately
    # not asserted here: releasing a sequence whose context never advanced churns
    # the tail of its claimed prefix in the reuse tree (pre-existing engine
    # behavior, reproduced on builds predating this test).
    kv_cache_manager.free_resources(req_live)
    summary = kv_cache_manager.impl.analyze_prefix_reuse(req_query)
    assert summary.reusable_blocks_allocated == 0
