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

from pathlib import Path


LLM_PATH = Path(__file__).parents[3] / "tensorrt_llm" / "llmapi" / "llm.py"
MODEL_ENGINE_PATH = (Path(__file__).parents[3] / "tensorrt_llm" / "_torch" /
                     "pyexecutor" / "model_engine.py")


def test_llm_inputs_embeds_branch_uses_span_metadata_without_cumsum() -> None:
    source = LLM_PATH.read_text()
    inputs_embeds_branch = source[
        source.index('if inputs.get("inputs_embeds") is not None:'):
        source.index("# NOTE: when running in `generation_only` for disagg")
    ]

    assert "_multimodal_embed_mask_cumsum_from_spans" not in source
    assert '"multimodal_embed_mask_cumsum"' not in inputs_embeds_branch
    assert "MultimodalInput.from_components" in inputs_embeds_branch


def test_full_prefill_inputs_embeds_skips_multimodal_runtime() -> None:
    source = MODEL_ENGINE_PATH.read_text()
    runtime_block = source[
        source.index("needs_mm_runtime = "):
        source.index("multimodal_params = MultimodalParams(")
    ]

    assert "past_seen_token_num > 0" in runtime_block
    assert "end_compute < len(all_prompt_tokens)" in runtime_block
    assert "if needs_mm_runtime and cumsum is not None:" in runtime_block
    assert "elif needs_mm_runtime and has_span_metadata:" in runtime_block
