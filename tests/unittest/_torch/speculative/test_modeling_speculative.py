# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from tensorrt_llm._torch.models.modeling_speculative import (
    _merge_draft_attention_extra_attrs,
)


def test_merge_draft_attention_extra_attrs_ignores_non_layer_metadata():
    model_extra_attrs = {
        "attn_layers": {"0": "target_attn"},
        "mla_layers": {"1": "target_mla"},
        "allreduce_hidden_size": 4096,
    }
    draft_extra_attrs = {
        "attn_layers": {"2": "draft_attn"},
        "mla_layers": {"3": "draft_mla"},
        "allreduce_max_num_tokens": 8192,
        "nvfp4_gemm_allowed_backends": ["TRTLLM"],
    }

    _merge_draft_attention_extra_attrs(
        model_extra_attrs,
        draft_extra_attrs,
        use_separate_draft_kv_cache=True,
    )

    assert model_extra_attrs["attn_layers"] == {
        "0": "target_attn",
        "2": "draft_attn",
    }
    assert model_extra_attrs["mla_layers"] == {
        "1": "target_mla",
        "3": "draft_mla",
    }
    assert model_extra_attrs["allreduce_hidden_size"] == 4096
    assert "allreduce_max_num_tokens" not in model_extra_attrs
    assert "nvfp4_gemm_allowed_backends" not in model_extra_attrs


def test_merge_draft_attention_extra_attrs_requires_shared_target_layer_map():
    model_extra_attrs = {}
    draft_extra_attrs = {"attn_layers": {"0": "draft_attn"}}

    with pytest.raises(ValueError, match="Draft extra_attrs key 'attn_layers'"):
        _merge_draft_attention_extra_attrs(
            model_extra_attrs,
            draft_extra_attrs,
            use_separate_draft_kv_cache=False,
        )
