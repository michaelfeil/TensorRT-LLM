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


MODELS_DIR = Path(__file__).parents[4] / "tensorrt_llm" / "_torch" / "models"


def test_kimi_k25_uses_external_encoder_wrapper_only() -> None:
    models_init = (MODELS_DIR / "__init__.py").read_text()

    assert "modeling_kimik25vl" in models_init
    assert "modeling_kimi_k25" not in models_init
    assert not (MODELS_DIR / "modeling_kimi_k25.py").exists()


def test_kimi_k25_does_not_eagerly_move_external_embeddings() -> None:
    source = (MODELS_DIR / "modeling_kimik25vl.py").read_text()

    assert "def multimodal_data_device_paths" in source
    assert "return []" in source


def test_kimi_k25_reads_special_offsets_from_multimodal_data() -> None:
    source = (MODELS_DIR / "modeling_kimik25vl.py").read_text()

    assert "runtime.special_token_offsets" not in source
    assert "runtime.mm_token_positions" not in source
    assert "runtime.mm_token_lengths" not in source
    assert '"special_token_offsets"' in source
