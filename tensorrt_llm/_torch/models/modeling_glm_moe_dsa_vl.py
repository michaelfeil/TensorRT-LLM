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

"""Thin VLM wrapper for GLM MoE DSA with an external vision encoder.

Reuses the Kimi K2.5 embedding fuse for the shared DeepseekV3 backbone;
only the media placeholder token default differs.
"""

from .modeling_kimik25vl import KimiK25VLModel
from .modeling_utils import register_auto_model

# <|image|> in the GLM vocabulary; GLM text-config checkpoints do not carry
# media_placeholder_token_id in config.json.
_GLM_MEDIA_PLACEHOLDER_TOKEN_ID = 154854


@register_auto_model("GlmMoeDsaForCausalLM")
class GlmMoeDsaVLModel(KimiK25VLModel):
    def __init__(self, model_config, *args, **kwargs):
        super().__init__(model_config, *args, **kwargs)
        self._media_token_id = getattr(
            model_config.pretrained_config,
            "media_placeholder_token_id",
            _GLM_MEDIA_PLACEHOLDER_TOKEN_ID,
        )
