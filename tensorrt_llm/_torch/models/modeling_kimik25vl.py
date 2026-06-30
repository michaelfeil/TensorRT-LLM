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

"""Thin VLM wrapper for Kimi K2.5 with external vision encoder.

The vision encoder runs as a separate service.  This model receives
pre-computed multimodal embeddings via ``multimodal_data["inputs_embeds"]``
and fuses them with text embeddings before forwarding to the DeepseekV3
LLM backbone.
"""

import torch

from tensorrt_llm.inputs import multimodal

from ..attention_backend import AttentionMetadata
from .modeling_deepseekv3 import DeepseekV3ForCausalLM
from .modeling_multimodal_utils import find_input_mm_embeds, fuse_input_embeds
from .modeling_utils import register_auto_model


def _cat_kimi_mm_embeds(mm_embeds: list[torch.Tensor]) -> torch.Tensor:
    if len(mm_embeds) == 1:
        return mm_embeds[0]
    return torch.cat(mm_embeds, dim=0)


def _get_kimi_mm_token_indices(
    multimodal_params: list[multimodal.MultimodalParams],
    device: torch.device,
) -> torch.Tensor | None:
    if not multimodal_params:
        return None

    indices = []
    for param in multimodal_params:
        runtime = param.multimodal_runtime
        if runtime is None:
            return None

        flat_base = param.input_ids_start_offset
        chunk_start = runtime.past_seen_token_num
        chunk_end = runtime.chunk_end_pos
        if chunk_end < chunk_start:
            raise ValueError(
                f"Kimi K2.5 multimodal chunk end ({chunk_end}) is before "
                f"chunk start ({chunk_start})"
            )

        if runtime.embed_mask_cumsum is not None:
            cumsum = runtime.embed_mask_cumsum
            segment = cumsum[chunk_start:chunk_end]
            if segment.numel() > 0:
                prev_value = (cumsum[chunk_start - 1]
                              if chunk_start > 0 else segment.new_zeros(()))
                prev = torch.cat((prev_value.reshape(1), segment[:-1]))
                chunk_indices = torch.nonzero(segment > prev,
                                              as_tuple=False).flatten()
                if chunk_indices.numel() > 0:
                    indices.extend(
                        (chunk_indices + flat_base).tolist())
            continue

        if (runtime.multimodal_positions is None
                or runtime.multimodal_lengths is None):
            return None

        multimodal_data = param.multimodal_data or {}
        special_offsets = set(multimodal_data.get("special_token_offsets") or [])
        mm_offset = 0
        for pos, length in zip(runtime.multimodal_positions,
                               runtime.multimodal_lengths):
            span_end = pos + length
            overlap_start = max(pos, chunk_start)
            overlap_end = min(span_end, chunk_end)
            for token_pos in range(overlap_start, overlap_end):
                token_mm_offset = mm_offset + token_pos - pos
                if token_mm_offset not in special_offsets:
                    indices.append(flat_base + token_pos - chunk_start)
            mm_offset += length

    if not indices:
        return None
    return torch.tensor(indices, dtype=torch.long, device=device)


def _fuse_kimi_external_input_embeds(
    embedding_layer,
    input_ids: torch.IntTensor,
    mm_embeds: list[torch.Tensor],
    mm_token_indices: torch.IntTensor,
) -> tuple[torch.IntTensor | None, torch.FloatTensor | None]:
    mm_embed = _cat_kimi_mm_embeds(mm_embeds)
    mm_token_indices = mm_token_indices.to(input_ids.device, dtype=torch.long)
    if mm_token_indices.shape[0] != mm_embed.shape[0]:
        raise ValueError(
            "Kimi K2.5 multimodal token count mismatch: found "
            f"{len(mm_token_indices)} media tokens in input_ids "
            f"but received {mm_embed.shape[0]} media embeddings.")

    safe_input_ids = input_ids.clamp(max=embedding_layer.num_embeddings - 1)
    input_embeds = embedding_layer(safe_input_ids)
    input_embeds[mm_token_indices, :] = mm_embed.to(
        device=input_embeds.device,
        dtype=input_embeds.dtype,
    )
    return None, input_embeds


def _num_kimi_mm_embeds(mm_embeds: list[torch.Tensor]) -> int:
    return sum(mm_embed.shape[0] for mm_embed in mm_embeds)


@register_auto_model("KimiK25ForConditionalGeneration")
@register_auto_model("KimiK25ForCausalLM")
class KimiK25VLModel(DeepseekV3ForCausalLM):
    def __init__(self, model_config, *args, **kwargs):
        super().__init__(model_config, *args, **kwargs)
        self._media_token_id = getattr(
            model_config.pretrained_config,
            "media_placeholder_token_id",
            163605,
        )

    @property
    def multimodal_data_device_paths(self) -> list[str]:
        return []

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor | None = None,
        position_ids: torch.IntTensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        return_context_logits: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        mm_params_list = kwargs.get("multimodal_params", [])
        # PyExecutor cannot derive Kimi media indices from model metadata, so
        # remove its generic indices before either using our exact indices or
        # falling back to media-token-id filtering.
        kwargs.pop("text_token_indices", None)
        kwargs.pop("mm_token_indices", None)
        mm_embeds: list[torch.Tensor] = []
        for mp in mm_params_list:
            ie = mp.multimodal_data.get("inputs_embeds") if mp.multimodal_data else None
            if ie is not None:
                mm_embeds.append(ie)

        if mm_embeds:
            mm_embeds = find_input_mm_embeds(mm_embeds, mm_params_list)

        if mm_embeds and input_ids is not None:
            mm_token_indices = _get_kimi_mm_token_indices(
                mm_params_list, device=input_ids.device)
        else:
            mm_token_indices = None

        if (mm_embeds and input_ids is not None and mm_token_indices is not None
                and mm_token_indices.shape[0] == _num_kimi_mm_embeds(mm_embeds)):
            fused_ids, fused_embeds = _fuse_kimi_external_input_embeds(
                self.model.embed_tokens,
                input_ids,
                mm_embeds,
                mm_token_indices,
            )
        else:
            fused_ids, fused_embeds = fuse_input_embeds(
                self.model.embed_tokens,
                input_ids,
                mm_embeds,
                mm_token_ids=torch.tensor([self._media_token_id]),
                **kwargs,
            )
        if fused_ids is not None:
            input_ids = fused_ids
        if fused_embeds is not None:
            inputs_embeds = fused_embeds
        if input_ids is not None:
            input_ids = input_ids.clamp(max=self.model.embed_tokens.num_embeddings - 1)

        kwargs.pop("multimodal_params", None)
        return super().forward(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_context_logits=return_context_logits,
            **kwargs,
        )
