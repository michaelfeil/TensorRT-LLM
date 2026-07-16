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
# This is Baseten's custom DFlash implementation (originally named "Dflash").
# It is kept alongside the upstream ``DFlash`` implementation in ``dflash.py``
# and is intentionally named ``BasetenDFlash`` to avoid conflicts.

from contextlib import contextmanager
from typing import TYPE_CHECKING, Dict, Optional

import torch
from torch import nn

from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.mapping import Mapping

from ..attention_backend import AttentionMetadata
from ..pyexecutor.guided_decoder import CapturableGuidedDecoder
from .eagle3 import Eagle3OneModelSpecMetadata, Eagle3OneModelWorker

if TYPE_CHECKING:
    from ...llmapi.llm_args import BasetenDFlashDecodingConfig


class BasetenDFlashCapturableGuidedDecoder(CapturableGuidedDecoder):
    """
    DFlash produces all draft tokens in a single draft step.
    In order to run guided decoding similar to Eagle3/MTP we need
    to run a for-loop at the end and run guided decoding there.
    However, this is causing some kind of deadlock due to us running
    the add_batch and execute_draft_batch successively. Trying to add
    synchronization is problematic because we break CUDA graph compatibility
    and it doesn't even work when we exclude the spec dec worker.

    My idea is, for correctness, all we care about is that the logits are
    correctly masked at the time of acceptance.

    The idea of this Guided Decoder is hence to simplify the code by removing
    some unnecessary steps and just ensure that the grammar state is correctly
    rolled back.

    What happens per iteration:
    1. execute(logits): advance grammar through prev iteration's tokens,
       apply bitmask to target logits
    2. Sample and accept: get back accepted_tokens, num_accepted_tokens
    3. add_draft_batch: copy last accepted token + num_accepted to pinned mem
    4. execute_draft_batch(draft_step=0): rollback rejected tokens, build
       (advances by 1), then ALWAYS rollback_draft_tokens (undoes that 1)
       and then net grammar state = num_accepted tokens which is correct for the
       next iteration
    """

    def execute_draft_batch(
        self, logits: torch.Tensor, d2t: Optional[torch.Tensor] = None, draft_step: int = 0
    ):
        with torch.cuda.stream(self.stream):
            torch.cuda.current_stream().wait_event(self.token_event)
            self.fetch_draft_batch(draft_step=draft_step)
            if draft_step == 0:
                self.rollback_rejected_tokens()
            failed_requests = self.build()
            # --- DFlash change: always rollback, not just at the last step ---
            self.rollback_draft_tokens()
            self.copy_bitmask(num_bitmask_tokens=len(self.requests))
            self.bitmask_event.record()

        torch.cuda.current_stream().wait_event(self.bitmask_event)
        self.apply_bitmask(logits, d2t=d2t, num_bitmask_tokens=len(self.requests))

        return failed_requests


class BasetenDFlashOneModelWorker(Eagle3OneModelWorker):
    def __init__(
        self,
        spec_config: "BasetenDFlashDecodingConfig",
        mapping: Mapping,
        use_separate_draft_kv_cache: bool = False,
    ):
        super().__init__(
            spec_config=spec_config,
            mapping=mapping,
            use_separate_draft_kv_cache=use_separate_draft_kv_cache,
        )
        self.block_size = spec_config.block_size
        # In the overlap scheduler, there is a race condition between the diffusion
        # forward and the attn_metadata.prepare() call. To avoid this condition,
        # we use internal buffers to store the values of the attn_metadata fields
        # of both host and device attention metadata fields.
        self._internal_buffers_initialized: bool = False
        self._internal_buffers: Dict[str, torch.Tensor] = {}

        # This is used for non-cuda graph mode. It is recorded right after
        # the diffusion FMHA enqueue so the next iter can CPU-wait on this single event
        # before reusing the internal buffers. This is for additional safety.
        self._diffusion_done_event: Optional[torch.cuda.Event] = None

        self.use_mla: bool = spec_config.use_mla

    def set_guided_decoder(self, guided_decoder: CapturableGuidedDecoder) -> bool:
        """Upgrade the engine's CapturableGuidedDecoder to DFlash variant.

        Instead of constructing a brand-new object (which would require the
        GuidedDecodingConfig that the matcher factories don't store), we
        simply swap the class of the existing, fully-initialized instance.

        This is safe because BasetenDFlashCapturableGuidedDecoder only overrides
        ``execute_draft_batch`` and adds no new ``__init__`` state.  The
        object keeps all its existing buffers, streams, events, and matcher
        factory -- the engine's add_batch / token_event.record pipeline
        works unchanged.
        """
        guided_decoder.__class__ = BasetenDFlashCapturableGuidedDecoder
        self.guided_decoder = guided_decoder
        return True

    def _prepare_attn_metadata_for_spec_dec(self, attn_metadata):
        """
        We do not run this for DFlash.

        Previously, we cloned some attn_metadata buffers and restored them
        after the diffusion forward pass mutated them in place. However,
        this mutate and restore caused a race condition in the overlap scheduler
        which also modifies the same shared host-pinned buffers.

        """
        return

    def _restore_attn_metadata_from_spec_dec(
        self,
        attn_metadata,
        num_accepted_tokens,
        original_num_contexts,
    ):
        """
        We do not run this for DFlash. See _prepare_attn_metadata_for_spec_dec.
        """
        return

    def _maybe_init_internal_buffers(self, attn_metadata):
        """
        We allocate internal buffers on first call (warmup).

        The sizes are based on attn_metadata.max_num_requests to
        cover all batch sizes for CUDA-graph capture/replay.
        """
        if self._internal_buffers_initialized:
            return

        max_n = attn_metadata.max_num_requests
        device = attn_metadata.kv_lens_cuda.device
        pinned = prefer_pinned()

        # Static fields

        # _seq_lens / _seq_lens_cuda we always set these to block_size
        self._internal_buffers["_seq_lens"] = torch.full(
            (max_n,), self.block_size, dtype=attn_metadata._seq_lens.dtype, pin_memory=pinned
        )
        self._internal_buffers["_seq_lens_cuda"] = torch.full(
            (max_n,), self.block_size, dtype=attn_metadata._seq_lens_cuda.dtype, device=device
        )

        # prompt_lens_cuda / prompt_lens_cpu we always set these to block_size
        # because we run a context length of block_size tokens per request
        self._internal_buffers["prompt_lens_cuda"] = torch.full(
            (max_n,), self.block_size, dtype=attn_metadata.prompt_lens_cuda.dtype, device=device
        )
        self._internal_buffers["prompt_lens_cpu"] = torch.full(
            (max_n,), self.block_size, dtype=attn_metadata.prompt_lens_cpu.dtype, pin_memory=pinned
        )

        # For host_request_types, 0 = context, 1 = generation. The diffusion
        # pass treats everything as context in MHA, so we always set these to 0
        # in MLA, we set these to 1 for generation since we use Causal mask there
        self._internal_buffers["host_request_types"] = None
        if self.use_mla:
            self._internal_buffers["host_request_types"] = torch.ones(
                max_n, dtype=attn_metadata.host_request_types.dtype, pin_memory=pinned
            )
        else:
            self._internal_buffers["host_request_types"] = torch.zeros(
                max_n, dtype=attn_metadata.host_request_types.dtype, pin_memory=pinned
            )

        # Dynamic fields
        self._internal_buffers["kv_lens_cuda"] = torch.zeros(
            max_n, dtype=attn_metadata.kv_lens_cuda.dtype, device=device
        )
        self._internal_buffers["host_total_kv_lens"] = torch.empty(2, device="cpu", dtype=torch.int)
        self._internal_buffers["host_total_kv_lens_total"] = torch.empty(
            2, device="cpu", dtype=torch.int
        )
        self._internal_buffers["kv_lens_runtime"] = torch.zeros(
            max_n, dtype=torch.int, pin_memory=pinned
        )

        self._internal_buffers_initialized = True

    @contextmanager
    def _diffusion_attn_metadata(
        self,
        attn_metadata,
        batch_size,
        num_contexts,
        num_gens,
        num_accepted_tokens,
        runtime_draft_len,
        original_all_rank_num_tokens,
        all_rank_num_seqs,
    ):
        """
        Run the diffusion forward against internal buffers.

        The diffusion forward needs to be a context call in order to work
        with bidirectional attention. Here, we use internal buffers to store the
        values of the attn_metadata fields.

        However, instead of modifying the attn_metadata fields in place, we modify
        the internal buffers and replace the references of the attn_metadata fields
        to these internal buffers.

        This supports the overlap scheduler which edits the original pinned
        attn_metadata fields while the diffusion forward is running.

        The buffers work because at capture time, we make it read from internal buffers
        so that during cuda graph replay, since the address is fixed, it will always
        read from these buffers which are correct for us to use.
        """
        self._maybe_init_internal_buffers(attn_metadata)

        # The CUDA-event safety net is only meaningful in eager mode.
        # If we capture the graph, we do not want to synchronize.
        is_capturing = torch.cuda.is_current_stream_capturing()

        # For non cuda graph mode, we synchronize on the diffusion done event.
        if not is_capturing and self._diffusion_done_event is not None:
            self._diffusion_done_event.synchronize()

        # Track original references and rebind them in the finally branch.
        saved = {
            "_seq_lens": attn_metadata._seq_lens,
            "_seq_lens_cuda": attn_metadata._seq_lens_cuda,
            "prompt_lens_cuda": attn_metadata.prompt_lens_cuda,
            "prompt_lens_cpu": attn_metadata.prompt_lens_cpu,
            "kv_lens_cuda": attn_metadata.kv_lens_cuda,
            "host_request_types": attn_metadata.host_request_types,
            "host_total_kv_lens": attn_metadata.host_total_kv_lens,
            "kv_lens_cuda_runtime": attn_metadata.kv_lens_cuda_runtime,
            "kv_lens_runtime": attn_metadata.kv_lens_runtime,
            "prompt_lens_cuda_runtime": attn_metadata.prompt_lens_cuda_runtime,
            "prompt_lens_cpu_runtime": attn_metadata.prompt_lens_cpu_runtime,
            "host_request_types_runtime": attn_metadata.host_request_types_runtime,
            "_num_contexts": attn_metadata._num_contexts,
            "_num_generations": attn_metadata._num_generations,
            "_num_tokens": attn_metadata._num_tokens,
            "_num_ctx_tokens": attn_metadata._num_ctx_tokens,
            "all_rank_num_tokens": attn_metadata.all_rank_num_tokens,
        }

        buffers = self._internal_buffers

        # kv_lens_cuda is the amount of valid tokens in the kv cache
        # including the ones that have not been filled/accepted yet.
        # It's just the amount in that batch. Since we're predicting
        # block_size tokens per request, we add block_size and
        # adjust for the amount of accepted tokens and runtime draft length.
        buffers["kv_lens_cuda"][:batch_size].copy_(
            saved["kv_lens_cuda"][:batch_size], non_blocking=True
        )
        buffers["kv_lens_cuda"][num_contexts:batch_size] += num_accepted_tokens[
            num_contexts:batch_size
        ] + (self.block_size - runtime_draft_len - 1)
        buffers["kv_lens_cuda"][:num_contexts] += self.block_size

        # host_total_kv_lens is the total amount of valid tokens in the kv cache
        # for context ([0]) and generation ([1]) requests, without extra tokens.
        # The diffusion pass folds both into a single bucket: everything is
        # treated as context in MHA (bucket 0) or as generation in MLA
        # (bucket 1, causal mask). Since we're predicting block_size tokens per
        # request, we add block_size for each context and generation request.
        base_total = saved["host_total_kv_lens"][0].item() + saved["host_total_kv_lens"][1].item()
        combined = base_total + self.block_size * (num_gens + num_contexts)
        if not self.use_mla:
            buffers["host_total_kv_lens_total"][0] = combined
            buffers["host_total_kv_lens_total"][1] = 0
        else:
            buffers["host_total_kv_lens_total"][1] = combined
            buffers["host_total_kv_lens_total"][0] = 0

        buffers["host_total_kv_lens"].copy_(buffers["host_total_kv_lens_total"])

        # kv_lens_runtime has same storage role as kv_lens_cuda but on host
        buffers["kv_lens_runtime"][:batch_size].copy_(saved["kv_lens_runtime"][:batch_size])
        buffers["kv_lens_runtime"][:batch_size] += self.block_size

        attn_metadata._seq_lens = buffers["_seq_lens"][:batch_size]
        attn_metadata._seq_lens_cuda = buffers["_seq_lens_cuda"][:batch_size]
        attn_metadata.prompt_lens_cuda = buffers["prompt_lens_cuda"]
        attn_metadata.prompt_lens_cpu = buffers["prompt_lens_cpu"]
        attn_metadata.kv_lens_cuda = buffers["kv_lens_cuda"]
        attn_metadata.host_request_types = buffers["host_request_types"]
        attn_metadata.host_total_kv_lens = buffers["host_total_kv_lens"]
        attn_metadata.kv_lens_runtime = buffers["kv_lens_runtime"][:batch_size]
        attn_metadata.kv_lens_cuda_runtime = buffers["kv_lens_cuda"][:batch_size]
        attn_metadata.prompt_lens_cuda_runtime = buffers["prompt_lens_cuda"][:batch_size]
        attn_metadata.prompt_lens_cpu_runtime = buffers["prompt_lens_cpu"][:batch_size]
        attn_metadata.host_request_types_runtime = buffers["host_request_types"][:batch_size]

        if not self.use_mla:
            attn_metadata._num_contexts = batch_size
            attn_metadata._num_generations = 0
            attn_metadata._num_ctx_tokens = batch_size * self.block_size
        else:
            attn_metadata._num_contexts = 0
            attn_metadata._num_generations = batch_size
            attn_metadata._num_ctx_tokens = 0
        attn_metadata._num_tokens = batch_size * self.block_size

        if original_all_rank_num_tokens is not None and all_rank_num_seqs is not None:
            attn_metadata.all_rank_num_tokens = [n * self.block_size for n in all_rank_num_seqs]

        try:
            yield
        finally:
            # non cuda graph mode: record the diffusion done event
            if not is_capturing:
                if self._diffusion_done_event is None:
                    self._diffusion_done_event = torch.cuda.Event()
                self._diffusion_done_event.record()

            # Restore the original references of the attn_metadata fields.
            for k, v in saved.items():
                setattr(attn_metadata, k, v)

            # Apply kv_lens_cuda increment to the original kv_lens_cuda.
            attn_metadata.kv_lens_cuda[num_contexts:batch_size] += num_accepted_tokens[
                num_contexts:batch_size
            ]
            attn_metadata.kv_lens_cuda[:num_contexts] += 1

    def forward(
        self,
        input_ids,
        position_ids,
        hidden_states,
        logits,
        attn_metadata,
        spec_metadata,
        draft_model,
        resource_manager=None,
    ):
        """
        1. Generate kv cache for the tokens
        2. Initialize Noise Tokens in Input IDs
        3. Update position ids
        4. Set first noise token as last ctx token (or last accepted token)
        5. Run draft Model forward
        """

        runtime_draft_len = spec_metadata.runtime_draft_len

        if (off_input_ids := getattr(spec_metadata, "offloader_input_ids", None)) is not None:
            n = input_ids.shape[0]
            off_input_ids[:, :n].index_copy_(
                0, spec_metadata.hidden_idx_cuda, input_ids.unsqueeze(0)
            )

        position_ids_1d = position_ids.squeeze(0)

        if (off_position_ids := getattr(spec_metadata, "offloader_position_ids", None)) is not None:
            n = position_ids_1d.shape[0]
            off_position_ids[:, :n].index_copy_(
                0, spec_metadata.hidden_idx_cuda, position_ids_1d.unsqueeze(0)
            )

        if (
            off_target_hidden_states := getattr(
                spec_metadata, "offloader_target_hidden_states", None
            )
        ) is not None:
            n = hidden_states.shape[0]
            off_target_hidden_states[:, :n].index_copy_(
                0, spec_metadata.hidden_idx_cuda, hidden_states.unsqueeze(0)
            )

        if runtime_draft_len == 0:
            return self.skip_drafting(
                input_ids,
                position_ids,
                hidden_states,
                logits,
                attn_metadata,
                spec_metadata,
                draft_model,
            )

        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts

        raw_logits = logits

        self._execute_guided_decoder_if_present(logits)

        accepted_tokens, num_accepted_tokens, sampled_log_probs = (
            self.sample_and_accept_draft_tokens(input_ids, logits, attn_metadata, spec_metadata)
        )

        # Roll back grammar to only the truly-accepted tokens.
        if self.guided_decoder is not None:
            new_tokens = accepted_tokens[
                spec_metadata.batch_indices_cuda[:batch_size], num_accepted_tokens - 1
            ]
            self.guided_decoder.add_draft_batch(new_tokens, num_accepted_tokens, draft_step=0)

        attn_metadata.use_spec_decoding = True

        # A dense MHA draft needs paged-context FMHA: its prefill (gen
        # requests append onto cached draft KV) and block-denoise (a
        # block_size query attending over the full cached context) are
        # context-with-KV-reuse calls. An MLA target clobbers this shared
        # flag every step (context MLA uses separate qkv), so set it for the
        # draft passes and restore afterwards — a dense TARGET would
        # otherwise inherit the leaked True on its later context passes.
        saved_use_paged_context_fmha = attn_metadata.use_paged_context_fmha
        if not self.use_mla:
            attn_metadata.use_paged_context_fmha = True

        # Prepare inputs for generating KV Cache. The prefill pass uses the
        # ORIGINAL attn_metadata (correct target-post-forward state); only
        # the diffusion pass below uses shadow buffers.
        inputs = self.prepare_drafter_inputs_prefill(
            input_ids=input_ids,
            position_ids=position_ids_1d,
            hidden_states=hidden_states,
            accepted_tokens=accepted_tokens,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            draft_model=draft_model,
        )

        original_all_rank_num_tokens = attn_metadata.all_rank_num_tokens
        all_rank_num_seqs = spec_metadata.all_rank_num_seqs

        draft_kv_cache_manager = self.get_draft_kv_cache_manager(resource_manager)

        with self.draft_kv_cache_context(attn_metadata, draft_kv_cache_manager):
            # Create KV Cache for the accepted tokens
            draft_model.model.prefill_forward(**inputs)

            # Put this a bit later to keep a gap for stability
            if self.guided_decoder is not None:
                self.guided_decoder.execute_draft_batch(raw_logits, d2t=None, draft_step=0)

            # Use the internal buffers for the diffusion forward.
            with self._diffusion_attn_metadata(
                attn_metadata,
                batch_size=batch_size,
                num_contexts=num_contexts,
                num_gens=num_gens,
                num_accepted_tokens=num_accepted_tokens,
                runtime_draft_len=runtime_draft_len,
                original_all_rank_num_tokens=original_all_rank_num_tokens,
                all_rank_num_seqs=all_rank_num_seqs,
            ):
                inputs = self.prepare_drafter_inputs_diffusion(
                    input_ids=input_ids,
                    position_ids=position_ids_1d,
                    accepted_tokens=accepted_tokens,
                    num_accepted_tokens=num_accepted_tokens,
                    attn_metadata=attn_metadata,
                    spec_metadata=spec_metadata,
                    draft_model=draft_model,
                )

                # Denoise
                hidden_out = draft_model.model(**inputs)

        attn_metadata.use_paged_context_fmha = saved_use_paged_context_fmha

        draft_hidden, draft_logits = self.process_draft_logits(
            draft_model=draft_model,
            hidden_out=hidden_out,
            batch_size=batch_size,
            attn_metadata=attn_metadata,
        )
        # The backbone exposes every candidate position. The runtime draft
        # length may be shorter, so trim both tensors before sampling.
        draft_hidden = draft_hidden[:, :runtime_draft_len]
        draft_logits = draft_logits[:, :runtime_draft_len]
        block_input_ids = inputs["input_ids"].reshape(batch_size, self.block_size)
        next_draft_tokens = self._sample_block_draft_tokens(
            draft_model=draft_model,
            draft_hidden=draft_hidden,
            draft_logits=draft_logits,
            block_input_ids=block_input_ids,
            batch_size=batch_size,
            runtime_draft_len=runtime_draft_len,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
        )

        # Restore all_rank_num_tokens for attention DP.
        if original_all_rank_num_tokens is not None:
            attn_metadata.all_rank_num_tokens = original_all_rank_num_tokens

        attn_metadata.num_contexts = 0
        attn_metadata._num_contexts = 0
        attn_metadata.host_request_types[:batch_size].fill_(1)

        attn_metadata._num_ctx_tokens = 0
        attn_metadata._num_generations = attn_metadata._seq_lens.shape[0]

        next_new_tokens = self._prepare_next_new_tokens(
            accepted_tokens,
            next_draft_tokens,
            spec_metadata.batch_indices_cuda,
            batch_size,
            num_accepted_tokens,
        )

        attn_metadata.use_spec_decoding = True

        return self._build_forward_outputs(
            logits=raw_logits,
            new_tokens=accepted_tokens,
            new_tokens_lens=num_accepted_tokens,
            next_draft_tokens=next_draft_tokens,
            next_new_tokens=next_new_tokens,
            sampled_log_probs=sampled_log_probs,
        )

    def process_draft_logits(
        self,
        *,
        draft_model: nn.Module,
        hidden_out: torch.Tensor,
        batch_size: int,
        attn_metadata: AttentionMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project all DFlash mask positions, excluding the anchor in slot 0."""
        block_hidden = hidden_out.reshape(batch_size, self.block_size, -1)
        draft_hidden = block_hidden[:, 1:]
        logits = draft_model.logits_processor(
            draft_hidden.reshape(batch_size * (self.block_size - 1), -1),
            draft_model.lm_head,
            attn_metadata,
            True,
        )
        draft_logits = logits.reshape(batch_size, self.block_size - 1, -1)
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
        """Sample draft tokens from the denoised draft-position hidden states.

        ``draft_hidden`` and ``draft_logits`` contain the runtime-trimmed
        candidate positions. ``block_input_ids`` is [batch, block_size] (a
        reshape view of the block token ids; unused here — the BasetenDSpark
        override reads the anchor from slot 0 for its sequential stage).
        """
        logits = draft_logits.reshape(batch_size * runtime_draft_len, -1)
        # DFlash denoises a whole block per step and samples every draft
        # position greedily (one argmax per row). The base draft_decoder's
        # advanced (per-request) sampler slices its tensors by request-count
        # batch_size, which does not match our (batch_size * runtime_draft_len)
        # flattened layout, so sample greedily here as the original Dflash did.
        d2t = getattr(getattr(draft_model, "model", None), "d2t", None)
        next_draft_tokens = self._draft_sampler_greedy(logits, d2t)
        return torch.reshape(next_draft_tokens, (batch_size, runtime_draft_len))

    def prepare_drafter_inputs_prefill(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        hidden_states: torch.Tensor,
        accepted_tokens: torch.Tensor,
        attn_metadata: AttentionMetadata,
        spec_metadata: Eagle3OneModelSpecMetadata,
        draft_model: nn.Module,
    ):
        num_contexts = attn_metadata.num_contexts
        num_tokens = input_ids.shape[0]

        hidden_states = spec_metadata.hidden_states[:num_tokens]
        hidden_states = draft_model.apply_dflash_fc(hidden_states)

        # context
        input_ctx_ids = input_ids[: attn_metadata.num_ctx_tokens]
        input_ids_ctx = torch.empty_like(input_ctx_ids, dtype=torch.int32, device="cuda")
        input_ids_ctx[:-1].copy_(input_ctx_ids[1:])
        input_ids_ctx[spec_metadata.gather_ids[:num_contexts]] = accepted_tokens[:num_contexts, 0]

        # generation
        input_ids_gen = accepted_tokens[num_contexts:, :].flatten()

        input_ids = torch.concat([input_ids_ctx, input_ids_gen], dim=0)

        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "hidden_states": hidden_states,
            "attn_metadata": attn_metadata,
            "spec_metadata": spec_metadata,
        }

    def prepare_drafter_inputs_diffusion(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        accepted_tokens: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        attn_metadata: AttentionMetadata,
        spec_metadata: Eagle3OneModelSpecMetadata,
        draft_model: nn.Module,
    ):
        num_seqs = attn_metadata.num_seqs

        input_ids, hidden_states = draft_model.initialize_blocks(
            self.block_size,
            accepted_tokens,
            num_accepted_tokens,
        )

        past_kv_lens = attn_metadata.kv_lens_cuda[:num_seqs] - self.block_size
        offsets = torch.arange(self.block_size, device=position_ids.device, dtype=torch.int32)
        position_ids = (past_kv_lens.unsqueeze(1) + offsets).reshape(-1)

        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "hidden_states": hidden_states,
            "attn_metadata": attn_metadata,
            "spec_metadata": spec_metadata,
        }
