from typing import Optional, Tuple

import torch

from ..flashinfer_utils import IS_FLASHINFER_AVAILABLE

if IS_FLASHINFER_AVAILABLE:
    from flashinfer.sampling import (
        chain_speculative_sampling,
        sampling_from_probs,
        top_k_mask_logits,
        top_k_top_p_sampling_from_logits,
        top_p_renorm_probs,
    )
    from flashinfer.sampling import softmax as flashinfer_softmax
else:
    chain_speculative_sampling = None
    sampling_from_probs = None
    flashinfer_softmax = None
    top_k_mask_logits = None
    top_p_renorm_probs = None
    top_k_top_p_sampling_from_logits = None


def forward_native(
    logits: torch.Tensor,
    k: Optional[torch.Tensor],
    p: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    PyTorch-native implementation of top-k and top-p sampling.

    The logits tensor may be updated in-place.
    """
    logits = apply_top_k_top_p(logits, k, p)
    return random_sample(logits)


def random_sample(
    logits: torch.Tensor,
) -> torch.Tensor:
    """Randomly sample from unnormalized logits.

    This uses the Gumbel-max trick and avoids CPU-GPU synchronization from
    torch.multinomial.
    """
    q = torch.empty_like(logits).exponential_()
    return logits.sub(q.log_()).argmax(dim=-1).view(-1)


@torch.compiler.disable
def apply_top_k_top_p(
    logits: torch.Tensor,
    k: Optional[torch.Tensor],
    p: Optional[torch.Tensor],
) -> torch.Tensor:
    """Apply top-k and top-p masks to the logits.

    If a top-p is used, this function will sort the logits tensor,
    which can be slow for large batches.

    The logits tensor may be updated in-place.
    """
    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)
    if k is not None:
        # Apply top-k.
        top_k_mask = logits_sort.size(1) - k.to(torch.long)  # shape: B
        top_k_mask = top_k_mask.clamp(min=0, max=logits_sort.size(1) - 1)
        # Get all the top_k values.
        top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
        top_k_mask = logits_sort < top_k_mask
        logits_sort.masked_fill_(top_k_mask, -float("inf"))

    if p is not None:
        # Apply top-p.
        max_logits = logits_sort[:, -1:].clone()
        shifted_exp = (logits_sort - max_logits).exp()
        probs_sum = torch.cumsum(shifted_exp, dim=-1, out=shifted_exp)
        top_p_threshold = (1 - p.unsqueeze(dim=1)) * probs_sum[:, -1:]
        top_p_mask = probs_sum <= top_p_threshold
        # at least one
        top_p_mask[:, -1] = False
        logits_sort.masked_fill_(top_p_mask, -float("inf"))
    # Re-sort the probabilities.
    logits = logits_sort.scatter(dim=-1, index=logits_idx, src=logits_sort)
    return logits


def apply_temperature(
    logits: torch.Tensor,
    temp: torch.Tensor,
) -> torch.Tensor:
    return logits.div_(temp.unsqueeze(dim=1))


@torch.compile(options={"max-autotune": True})
def _sampling_batch_native(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sort-based fallback sampling path.

    Returned logprobs are over the post-temperature, post-mask distribution
    (matching the previous behavior of this sampler).
    """
    safe_temperatures = torch.clamp(temperatures, min=1e-5)
    scaled_logits = apply_temperature(logits, safe_temperatures)
    masked_logits = apply_top_k_top_p(scaled_logits, top_k, top_p)

    greedy_mask = temperatures == 0
    greedy_sampled = masked_logits.argmax(dim=-1)
    random_sampled = random_sample(masked_logits)
    sampled_tokens = torch.where(greedy_mask, greedy_sampled, random_sampled).to(torch.int32)

    sampled_log_probs = (
        torch.gather(masked_logits, dim=-1, index=sampled_tokens.long().unsqueeze(-1)).squeeze(-1)
        - torch.logsumexp(masked_logits, dim=-1)
    ).to(torch.float32)
    return sampled_tokens, sampled_log_probs


def _sampling_batch_flashinfer(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    seed: Optional[torch.Tensor] = None,
    offset: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FlashInfer fast path: fused filter + sample, RAW logprobs.

    Uses ``top_k_top_p_sampling_from_logits`` — one fused kernel does top-k
    mask, softmax, top-p renorm, and sampling. The sampler returns only the
    sampled token ids (not the filtered distribution), so logprobs are
    computed RAW — over the post-temperature, pre-filter distribution — via
    ``logit[sampled] - logsumexp(logits)``. This avoids materializing a full
    N*V ``log_softmax`` tensor; ``logsumexp`` reduces to N scalars.

    Semantic note: RAW != PROCESSED. For rows with meaningful top-k/top-p
    filtering, RAW logprob is smaller (less concentrated) than the logprob
    over the filtered distribution would be. For greedy-equivalent rows
    (``top_k=iinfo(int32).max``, ``top_p=1.0``), RAW == PROCESSED.

    Greedy rows arrive as ``temp=0``. Temperature is clamped for division,
    then sampled tokens for those rows are overridden with argmax. ``logits``
    is mutated in place by the temperature scaling.
    """
    # Clamp only for the division; keep the original temperature tensor so
    # temperature=0 rows can force argmax after FlashInfer sampling.
    safe_temperatures = torch.clamp(temperatures, min=1e-5)
    scaled_logits = logits.div_(safe_temperatures.unsqueeze(1))
    greedy_mask = temperatures == 0
    greedy_sampled = scaled_logits.argmax(dim=-1)

    # Fused filter + sample (top_k mask -> softmax -> top_p renorm -> sample
    # in one kernel set). Returns int64 sampled tokens.
    # Use FlashInfer's default ``deterministic=True`` path (prefix-sum based)
    # to match upstream TRT-LLM and avoid the less-exercised curand-rejection
    # path. ``seed``/``offset`` are forwarded so a managed RNG stream
    # (per-rank seeding from SpecWorkerBase) can produce a stable draw
    # sequence within a run.
    if top_k_top_p_sampling_from_logits is None:
        raise RuntimeError(
            "FlashInfer sampling requested but flashinfer is unavailable")
    sampled_long = top_k_top_p_sampling_from_logits(scaled_logits,
                                                    top_k,
                                                    top_p,
                                                    seed=seed,
                                                    offset=offset)
    sampled_long = torch.where(greedy_mask, greedy_sampled, sampled_long)

    # RAW logprob: logit[sampled] - logsumexp(logits). Avoids a full N*V
    # log_softmax tensor by reducing directly to N scalars.
    sampled_logit_val = torch.gather(
        scaled_logits, dim=-1, index=sampled_long.unsqueeze(-1)
    ).squeeze(-1)
    sampled_log_probs = sampled_logit_val - torch.logsumexp(scaled_logits, dim=-1)
    return sampled_long.to(torch.int32), sampled_log_probs


def sampling_batch_spec_dec_one_model(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    use_flashinfer: bool = False,
    seed: Optional[torch.Tensor] = None,
    offset: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    CUDA-graph compatible sampling. Supports mixed sampling params.

    When FlashInfer is available and ``use_flashinfer`` is True, sampling
    routes through a fused kernel and returns RAW logprobs (over the
    post-temperature, pre-filter distribution). The native fallback returns
    logprobs over the post-temperature, post-top-k, post-top-p distribution.

    We can't do dynamic kernel selection inside graphs, so the native path
    may be slower than a torch.argmax for greedy requests. This is why
    advanced sampling is opt-in for now.
    """
    if use_flashinfer:
        return _sampling_batch_flashinfer(
            logits, temperatures, top_k, top_p, seed=seed, offset=offset)
    # Native path does not currently honor seed/offset; those are forwarded
    # only to the FlashInfer fast path.
    del seed, offset
    return _sampling_batch_native(logits, temperatures, top_k, top_p)


def compute_probs_from_logits(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: Optional[torch.Tensor],
    top_p: Optional[torch.Tensor],
    skip_temperature: bool = False,
) -> torch.Tensor:
    """
    Compute filtered + normalized probs from logits (temperature + top_k +
    top_p + softmax). Picks the fastest path for the input device:

    1. CUDA + flashinfer: ``top_k_mask_logits`` → fused ``softmax+temp`` →
       ``top_p_renorm_probs`` (all O(N) radix). ``skip_temperature`` ignored.
    2. CUDA, no flashinfer: ``compute_probs_from_logits_op`` (sort-based,
       O(N log N)).
    3. CPU: manual PyTorch fallback.
    """
    if logits.is_cuda and IS_FLASHINFER_AVAILABLE:
        # Fast path: flashinfer composition (O(N) per row, friendly to small
        # batch sizes). skip_temperature is ignored — flashinfer's softmax
        # always applies the temperature tensor.
        if top_k is not None:
            logits = top_k_mask_logits(logits, top_k)
        probs = flashinfer_softmax(logits, temperatures)
        if top_p is not None:
            probs = top_p_renorm_probs(probs, top_p)
        return probs

    if logits.is_cuda:
        # CUDA without flashinfer: fall back to the C++ op (slower sort-based
        # top-k path, but works without flashinfer).
        return torch.ops.trtllm.compute_probs_from_logits_op(
            logits, temperatures, top_k, top_p, skip_temperature
        )

    if not skip_temperature:
        logits = apply_temperature(logits, temperatures)
    logits = apply_top_k_top_p(logits, top_k, top_p)
    probs = logits.softmax(dim=-1, dtype=torch.float32)
    # Greedy rows should remain exactly one-hot so rejection sampling does not
    # spuriously reject numerically-near argmax tokens.
    greedy_temp_threshold = 1e-4
    is_greedy = temperatures <= greedy_temp_threshold
    argmax_ids = logits.argmax(dim=-1, keepdim=True)
    one_hot = torch.zeros_like(probs).scatter_(1, argmax_ids, 1.0)
    return torch.where(is_greedy.unsqueeze(1), one_hot, probs)


def sampling_batch_spec_dec_one_model_for_rejection(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    seed: Optional[torch.Tensor] = None,
    offset: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rejection-sampling-aware draft sampler: returns BOTH the sampled tokens
    AND the prob distribution they were sampled from, so the downstream
    rejection-sampling path can reuse the probs without a second softmax +
    temp/top_k/top_p pass.
    """
    if sampling_from_probs is None:
        raise RuntimeError(
            "Rejection sampling for one-model speculative decoding requires flashinfer"
        )
    probs = compute_probs_from_logits(logits, temperatures, top_k, top_p)
    tokens = sampling_from_probs(probs, deterministic=True, seed=seed, offset=offset)
    return tokens, probs


def rejection_sampling_one_model(
    draft_probs: torch.Tensor,
    draft_token_ids: torch.Tensor,
    target_probs: torch.Tensor,
    deterministic: bool = True,
    seed: Optional[int] = None,
    offset: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    CUDA-graph compatible rejection sampling for one-model speculative decoding.
    """
    batch_size = draft_token_ids.shape[0]
    device = draft_token_ids.device

    output_accepted_token_num = torch.zeros(batch_size, dtype=torch.int32, device=device)
    output_emitted_draft_token_num = torch.zeros(batch_size, dtype=torch.int32, device=device)

    if chain_speculative_sampling is None:
        raise RuntimeError(
            "Rejection sampling for one-model speculative decoding requires flashinfer"
        )

    accepted_tokens, _, output_emitted_draft_token_num = chain_speculative_sampling(
        draft_probs,
        draft_token_ids,
        target_probs,
        maybe_output_accepted_token_num=output_accepted_token_num,
        maybe_output_emitted_draft_token_num=output_emitted_draft_token_num,
        deterministic=deterministic,
        generator=None,
        seed=seed,
        offset=offset,
    )

    return accepted_tokens, output_emitted_draft_token_num + 1
