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

import unittest

from tensorrt_llm._torch.pyexecutor.model_engine import (
    _get_cuda_graph_warmup_seq_lens,
)
from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig


class TestCudaGraphWarmupSeqLens(unittest.TestCase):

    def test_dsa_uses_minimal_long_sequence_for_indexer_warmup(self):
        sparse_config = DeepSeekSparseAttentionConfig(index_topk=2048)

        seq_lens = _get_cuda_graph_warmup_seq_lens(
            sparse_config=sparse_config,
            effective_max_seq_len=202752,
            max_draft_len=4,
            kv_reserve_draft_tokens=4,
            num_extra_decoding_steps=0,
        )

        self.assertEqual(seq_lens, [2050, 2043])

    def test_dsa_long_sequence_accounts_for_extra_reserved_draft_slots(self):
        sparse_config = DeepSeekSparseAttentionConfig(index_topk=2048)

        seq_lens = _get_cuda_graph_warmup_seq_lens(
            sparse_config=sparse_config,
            effective_max_seq_len=202752,
            max_draft_len=4,
            kv_reserve_draft_tokens=16,
            num_extra_decoding_steps=0,
        )

        self.assertEqual(seq_lens, [2062, 2043])

    def test_dsa_long_sequence_accounts_for_extra_decoding_steps(self):
        sparse_config = DeepSeekSparseAttentionConfig(index_topk=2048)

        seq_lens = _get_cuda_graph_warmup_seq_lens(
            sparse_config=sparse_config,
            effective_max_seq_len=202752,
            max_draft_len=4,
            kv_reserve_draft_tokens=4,
            num_extra_decoding_steps=8,
        )

        self.assertEqual(seq_lens, [2058, 2043])

    def test_dsa_skips_extra_short_long_capture_when_long_mode_unreachable(
            self):
        sparse_config = DeepSeekSparseAttentionConfig(index_topk=2048)

        seq_lens = _get_cuda_graph_warmup_seq_lens(
            sparse_config=sparse_config,
            effective_max_seq_len=2049,
            max_draft_len=4,
            kv_reserve_draft_tokens=4,
            num_extra_decoding_steps=0,
        )

        self.assertEqual(seq_lens, [2049])

    def test_dsa_without_short_sequence_skip_uses_effective_max(self):
        sparse_config = DeepSeekSparseAttentionConfig(
            index_topk=2048, skip_indexer_for_short_seqs=False)

        seq_lens = _get_cuda_graph_warmup_seq_lens(
            sparse_config=sparse_config,
            effective_max_seq_len=202752,
            max_draft_len=4,
            kv_reserve_draft_tokens=4,
            num_extra_decoding_steps=0,
        )

        self.assertEqual(seq_lens, [202752])


if __name__ == "__main__":
    unittest.main()
