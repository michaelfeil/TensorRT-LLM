"""Thin VLM wrapper for Kimi K2.5 with external vision encoder.

The vision encoder runs as a separate service.  This model receives
pre-computed multimodal embeddings via ``multimodal_data["inputs_embeds"]``
and fuses them with text embeddings before forwarding to the DeepseekV3
LLM backbone.
"""

import torch

from ..attention_backend import AttentionMetadata
from .modeling_deepseekv3 import DeepseekV3ForCausalLM
from .modeling_multimodal_utils import find_input_mm_embeds, fuse_input_embeds
from .modeling_utils import register_auto_model


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
        mm_embeds: list[torch.Tensor] = []
        for mp in mm_params_list:
            ie = mp.multimodal_data.get("inputs_embeds") if mp.multimodal_data else None
            if ie is not None:
                mm_embeds.append(ie)

        if mm_embeds:
            mm_embeds = find_input_mm_embeds(mm_embeds, mm_params_list)

        if mm_embeds and input_ids is not None:
            # The model engine computes mm_token_indices via the mm_token_ids
            # attribute, which we intentionally omit to avoid CUDA graph warmup
            # issues.  Its indices will be wrong; drop them so fuse_input_embeds
            # recomputes using our media token ID.
            kwargs.pop("text_token_indices", None)
            kwargs.pop("mm_token_indices", None)

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
