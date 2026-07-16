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

from types import SimpleNamespace
from unittest import mock

import torch

from tensorrt_llm._torch.speculative.eagle3 import Eagle3OneModelWorker
from tensorrt_llm.llmapi import MTPDecodingConfig


def test_mtp_eagle_uses_greedy_draft_tokens_by_default():
    config = MTPDecodingConfig(max_draft_len=3)
    worker = Eagle3OneModelWorker(config)

    assert config.use_greedy_draft_tokens is True
    assert config.model_dump()["use_greedy_draft_tokens"] is True
    assert worker._use_greedy_draft_tokens is True

    expected = torch.tensor([7, 11], dtype=torch.int32)
    worker.draft_sampler = mock.Mock(return_value=expected)
    worker._draft_sampler_advanced = mock.Mock(
        side_effect=AssertionError("advanced sampler must be bypassed")
    )
    spec_metadata = SimpleNamespace(is_all_greedy_sample=False)

    actual = worker.draft_decoder(
        logits=torch.zeros(2, 8),
        draft_model=SimpleNamespace(model=None),
        spec_metadata=spec_metadata,
        batch_size=2,
        draft_step=0,
    )

    assert actual is expected
    worker.draft_sampler.assert_called_once()
    worker._draft_sampler_advanced.assert_not_called()
    assert worker._can_use_rejection_sampling(spec_metadata) is False


def test_mtp_eagle_can_use_request_sampling_for_draft_tokens():
    config = MTPDecodingConfig(max_draft_len=3, use_greedy_draft_tokens=False)
    worker = Eagle3OneModelWorker(config)

    assert worker._use_greedy_draft_tokens is False

    expected = torch.tensor([7, 11], dtype=torch.int32)
    worker.draft_sampler = mock.Mock(side_effect=AssertionError("greedy sampler must be bypassed"))
    worker._draft_sampler_advanced = mock.Mock(return_value=expected)
    spec_metadata = SimpleNamespace(is_all_greedy_sample=False, use_rejection_sampling=False)

    actual = worker.draft_decoder(
        logits=torch.zeros(2, 8),
        draft_model=SimpleNamespace(model=None),
        spec_metadata=spec_metadata,
        batch_size=2,
        draft_step=0,
    )

    assert actual is expected
    worker.draft_sampler.assert_not_called()
    worker._draft_sampler_advanced.assert_called_once()
