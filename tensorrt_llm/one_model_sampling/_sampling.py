# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright 2025 Baseten
# SPDX-License-Identifier: Apache-2.0

import torch
from typing import Optional


def _random_sample(logits: torch.Tensor) -> torch.Tensor:
    """Randomly sample from unnormalized logits via the Gumbel-max trick:
    argmax(logits - log(E)), E ~ Exp(1).
    Avoids materializing normalized probabilities and avoids CPU-GPU
    synchronization from torch.multinomial."""
    q = torch.empty_like(logits).exponential_()
    return logits.sub(q.log_()).argmax(dim=-1).view(-1)


def resample(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    top_k: torch.Tensor,
) -> torch.Tensor:
    B, V = logits.shape

    temperature = temperature.to(logits.dtype).view(B, 1)
    greedy_mask = (temperature == 0.0).squeeze(-1)
    scaled_logits = logits / torch.clamp(temperature, min=1e-5)

    logits_sort, logits_idx = scaled_logits.sort(dim=-1, descending=False)

    # top-k: position-based mask
    cutoff = (V - top_k.to(torch.long)).clamp(min=0, max=V - 1)
    positions = torch.arange(V, device=logits.device).unsqueeze(0)
    logits_sort = logits_sort.masked_fill(
        positions < cutoff.unsqueeze(1), -float("inf"))

    # top-p: nucleus filtering via shifted-exp cumulative sum.
    # Graph break prevents Inductor from fusing exp() into cumsum's SplitScan,
    # which triggers a Triton codegen bug on large vocab dimensions.
    max_logits = logits_sort[:, -1:].clone()
    shifted_exp = (logits_sort - max_logits).exp()
    torch._dynamo.graph_break()
    probs_sum = torch.cumsum(shifted_exp, dim=-1)
    top_p_threshold = (1 - top_p.unsqueeze(dim=1)) * probs_sum[:, -1:]
    top_p_mask = probs_sum <= top_p_threshold
    top_p_mask[:, -1] = False
    logits_sort = logits_sort.masked_fill(top_p_mask, -float("inf"))

    scaled_logits = logits_sort.scatter(
        dim=-1, index=logits_idx, src=logits_sort)

    sampled_tokens = torch.where(
        greedy_mask,
        torch.argmax(logits, dim=-1),
        _random_sample(scaled_logits),
    )
    return sampled_tokens.to(torch.int32)


def temperature_sampling(
    logits: torch.Tensor,  # [B, V]
    temperature: torch.Tensor,  # [B]
    top_p: Optional[torch.Tensor] = None,  # [B]
    top_k: Optional[torch.Tensor] = None,  # [B]
) -> torch.Tensor:
    """
    Sample one token per batch row from the categorical distribution induced by
    logits / temperature, with optional top-p and top-k filtering.

    Returns:
        sampled token indices, shape [B], dtype torch.int32
    """
    assert logits.dim() == 2, "logits must be [B, V]"
    B = logits.shape[0]
    device = logits.device

    if isinstance(temperature, float):
        temperature = torch.full((B,), temperature, dtype=torch.float32,
                                 device=device)
    if top_k is None:
        top_k = torch.zeros(B, dtype=torch.int32, device=device)
    elif isinstance(top_k, int):
        top_k = torch.full((B,), top_k, dtype=torch.int32, device=device)
    if top_p is None:
        top_p = torch.ones(B, dtype=torch.float32, device=device)
    elif isinstance(top_p, float):
        top_p = torch.full((B,), top_p, dtype=torch.float32, device=device)

    return resample(logits, temperature, top_p, top_k)
