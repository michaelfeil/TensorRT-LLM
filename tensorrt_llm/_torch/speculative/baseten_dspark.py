# SPDX-FileCopyrightText: Copyright (c) 2026 Baseten. All rights reserved.
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
#
# DSpark speculative decoding (Cheng et al., 2026) on the BasetenDFlash
# backbone. The parallel denoising pass is unchanged; drafting swaps DFlash's
# single parallel argmax for a semi-autoregressive stage:
#
#   * Markov head: each draft position's logits get a low-rank transition bias
#     W2 @ W1[x_prev] from the previously *sampled* token, so positions are
#     sampled left-to-right instead of independently. This mitigates the
#     suffix acceptance decay of pure parallel drafters.
#   * Confidence head (opt-in via enable_confidence_head): per-position
#     conditional acceptance estimates c_k = sigmoid(w . [h_k ; W1[x_{k-1}]]);
#     prefix survival cumprods are written to a persistent buffer for
#     telemetry and future confidence-scheduled draft-length selection.
#
# The whole loop runs inside the captured CUDA graph: it is unrolled over
# runtime_draft_len (fixed per graph key), uses only fixed-shape device ops,
# and never syncs with the host.

from typing import TYPE_CHECKING, Optional

import torch
from torch import nn

from tensorrt_llm.mapping import Mapping

from ..attention_backend import AttentionMetadata
from .baseten_dflash import BasetenDFlashOneModelWorker
from .eagle3 import Eagle3OneModelSpecMetadata

if TYPE_CHECKING:
    from ...llmapi.llm_args import BasetenDSparkDecodingConfig


class BasetenDSparkOneModelWorker(BasetenDFlashOneModelWorker):
    def __init__(
        self,
        spec_config: "BasetenDSparkDecodingConfig",
        mapping: Mapping,
        use_separate_draft_kv_cache: bool = False,
    ):
        super().__init__(
            spec_config=spec_config,
            mapping=mapping,
            use_separate_draft_kv_cache=use_separate_draft_kv_cache,
        )
        # Prefix survival probabilities of the drafts proposed in the most
        # recent step, [max_num_requests, max_draft_len]. Fixed address so
        # CUDA-graph replays keep it current; rows beyond the live batch and
        # columns beyond runtime_draft_len are stale.
        self.draft_survival: Optional[torch.Tensor] = None

    def _store_draft_survival(
        self, survival: torch.Tensor, attn_metadata: AttentionMetadata
    ) -> None:
        if self.draft_survival is None:
            self.draft_survival = torch.zeros(
                (attn_metadata.max_num_requests, self.max_draft_len),
                dtype=torch.float32,
                device=survival.device,
            )
        batch_size, runtime_draft_len = survival.shape
        self.draft_survival[:batch_size, :runtime_draft_len].copy_(survival)

    def process_draft_logits(
        self,
        *,
        draft_model: nn.Module,
        hidden_out: torch.Tensor,
        batch_size: int,
        attn_metadata: AttentionMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project the full block, treating the anchor as draft position 0."""
        draft_hidden = hidden_out.reshape(batch_size, self.block_size, -1)
        logits = draft_model.logits_processor(
            draft_hidden.reshape(batch_size * self.block_size, -1),
            draft_model.lm_head,
            attn_metadata,
            True,
        )
        draft_logits = logits.reshape(batch_size, self.block_size, -1)
        return draft_hidden, draft_logits

    def _sample_block_draft_tokens(
        self,
        *,
        draft_model: nn.Module,
        draft_hidden: torch.Tensor,
        draft_logits: torch.Tensor,
        block_input_ids: torch.Tensor,
        batch_size: int,
        runtime_draft_len: int,
        attn_metadata: AttentionMetadata,
        spec_metadata: Eagle3OneModelSpecMetadata,
    ) -> torch.Tensor:
        model = draft_model.model
        markov_head = model.markov_head
        confidence_head = model.confidence_head
        d2t = getattr(model, "d2t", None)

        hidden = draft_hidden
        base_logits = draft_logits

        # Anchor = last accepted token in block slot 0; seeds the Markov chain.
        prev_tokens = block_input_ids[:, 0]

        draft_tokens = []
        confidences = [] if confidence_head is not None else None
        prev_embedding: Optional[torch.Tensor] = None
        use_advanced_sampling = not spec_metadata.is_all_greedy_sample
        use_rejection_sampling = spec_metadata.use_rejection_sampling and use_advanced_sampling
        for step in range(runtime_draft_len):
            step_logits = base_logits[:, step]
            if markov_head is not None:
                prev_embedding = markov_head.prev_embedding(prev_tokens)
                step_logits = step_logits + markov_head.bias(prev_embedding)
            if confidence_head is not None:
                if model.confidence_with_markov:
                    features = torch.cat([hidden[:, step], prev_embedding], dim=-1)
                else:
                    features = hidden[:, step]
                confidences.append(confidence_head(features))
            # The sampled target-vocab token feeds the next step's Markov bias
            # in both modes, preserving the checkpoint's recurrence exactly.
            # Rejection sampling additionally stores the exact filtered draft
            # distribution used for this draw in the stable request-slot row.
            if use_rejection_sampling:
                step_tokens = self._draft_sampler_advanced_for_rejection(
                    step_logits,
                    spec_metadata,
                    batch_size,
                    d2t,
                    draft_step=step,
                )
            elif use_advanced_sampling:
                step_tokens = self._draft_sampler_advanced(
                    step_logits,
                    spec_metadata,
                    batch_size,
                    d2t,
                )
            else:
                step_tokens = self._draft_sampler_greedy(step_logits, d2t)
            draft_tokens.append(step_tokens)
            prev_tokens = step_tokens

        next_draft_tokens = torch.stack(draft_tokens, dim=1)
        if spec_metadata.use_rejection_sampling:
            if use_rejection_sampling:
                spec_metadata.d2t = d2t.data if d2t is not None else None
                spec_metadata.draft_probs_valid = True
            else:
                # All-greedy batches do not populate draft_probs. Prevent a
                # later acceptance pass from consuming a stale distribution.
                spec_metadata.draft_probs_valid = False
        if confidences is not None:
            survival = torch.cumprod(torch.stack(confidences, dim=1), dim=1)
            self._store_draft_survival(survival, attn_metadata)
        return next_draft_tokens
