# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright 2025 Baseten
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict

import torch

from tensorrt_llm._utils import prefer_pinned

try:
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
except ImportError:
    # it's only needed for type checking for store_sampling_metadata and get_sampling_metadata
    pass

from ._sampling import resample


def singleton(cls):
    """Decorator that makes a class a singleton using a metaclass.

    Usage:
        @singleton
        class B:
            def some_func(self):
                return "result"

        # Both work the same way:
        b = B()
        result = B().some_func()  # calls the same function everywhere
    """

    class SingletonMeta(type):
        _instances = {}

        def __call__(cls, *args, **kwargs):
            if cls not in cls._instances:
                cls._instances[cls] = super().__call__(*args, **kwargs)
            return cls._instances[cls]

        def __getattr__(cls, name):
            if cls not in cls._instances:
                cls._instances[cls] = super().__call__()
            return getattr(cls._instances[cls], name)

    return SingletonMeta(cls.__name__, cls.__bases__, dict(cls.__dict__))


@singleton
class _MetadataStore:
    def __init__(self):
        self.metadata = OrderedDict()

        torch.manual_seed(42)

        self.temp_pinned = torch.empty(0, device="cpu", pin_memory=prefer_pinned())
        self.top_p_pinned = torch.empty(0, device="cpu", pin_memory=prefer_pinned())
        self.top_k_pinned = torch.empty(0, device="cpu", pin_memory=prefer_pinned())

    def get(self, request_ids: list[int]):
        temp, top_p, top_k = zip(*[self.metadata[request_id] for request_id in request_ids])

        if len(temp) > len(self.temp_pinned):
            self.temp_pinned = torch.empty(len(temp), device="cpu", pin_memory=prefer_pinned())
        if len(top_p) > len(self.top_p_pinned):
            self.top_p_pinned = torch.empty(len(top_p), device="cpu", pin_memory=prefer_pinned())
        if len(top_k) > len(self.top_k_pinned):
            self.top_k_pinned = torch.empty(len(top_k), device="cpu", pin_memory=prefer_pinned())

        self.temp_pinned[:len(temp)].copy_(torch.tensor(temp, dtype=torch.float32, device="cpu"), non_blocking=True)
        self.top_p_pinned[:len(top_p)].copy_(torch.tensor(top_p, dtype=torch.float32, device="cpu"), non_blocking=True)
        self.top_k_pinned[:len(top_k)].copy_(torch.tensor(top_k, dtype=torch.int32, device="cpu"), non_blocking=True)

        return self.temp_pinned[:len(temp)], self.top_p_pinned[:len(top_p)], self.top_k_pinned[:len(top_k)]

    def add(self, request_id: int, temp: float, top_p: float, top_k: int):
        self.metadata[request_id] = (temp, top_p, top_k)

        if len(self.metadata) > 8 * 1024:
            self.metadata.popitem(last=False)

    def remove(self, request_id: int):
        del self.metadata[request_id]

    def has(self, request_id: int):
        return request_id in self.metadata


def _get_temperature(request):
    if not request.sampling_config:
        return 0

    if not request.sampling_config.temperature:
        return 0

    if isinstance(request.sampling_config.temperature, float):
        return request.sampling_config.temperature

    if isinstance(request.sampling_config.temperature, list):
        assert len(request.sampling_config.temperature) == 1
        return request.sampling_config.temperature[0]

    assert False, "Invalid temperature type"


def _get_top_p(request):
    if not request.sampling_config:
        return 1

    if not request.sampling_config.top_p:
        return 1

    if isinstance(request.sampling_config.top_p, float):
        return request.sampling_config.top_p

    if isinstance(request.sampling_config.top_p, list):
        assert len(request.sampling_config.top_p) == 1
        return request.sampling_config.top_p[0]

    assert False, "Invalid top_p type"


def _get_top_k(request):
    if not request.sampling_config:
        return 50

    if not request.sampling_config.top_k:
        return 50

    if isinstance(request.sampling_config.top_k, int):
        return request.sampling_config.top_k

    if isinstance(request.sampling_config.top_k, list):
        assert len(request.sampling_config.top_k) == 1
        return request.sampling_config.top_k[0]

    assert False, "Invalid top_k type"


def store_sampling_metadata(request: LlmRequest):
    if not _MetadataStore().has(request.py_request_id):
        _MetadataStore().add(
            request_id=request.py_request_id,
            temp=float(_get_temperature(request)),
            top_p=float(_get_top_p(request)) or 1,
            top_k=int(_get_top_k(request)) or 50,
        )


def get_sampling_metadata(request_ids: list[int]):
    return _MetadataStore().get(request_ids)


@torch.compile(options={"max-autotune": True})
def apply_resampling(
    accepted_tokens: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    num_seqs: int,
    num_contexts: int,
    draft_len: int,
    logits: torch.Tensor,
    extern_temperature: torch.Tensor,
    extern_top_p: torch.Tensor,
    extern_top_k: torch.Tensor,
    batch_indices_cuda: torch.Tensor,
):
    """Baseten fast rejection sampling.

    Resample the first generation token for context requests (first token after
    context) and the first rejected draft token for generation requests.
    Writes accepted_tokens and num_accepted_tokens in place.
    """
    num_gens = num_seqs - num_contexts

    if num_contexts > 0:
        # sample the first generation token

        first_token_logits = logits[:num_contexts]
        accepted_tokens[:num_contexts, 0] = resample(
            first_token_logits,
            extern_temperature[:num_contexts],
            extern_top_p[:num_contexts],
            extern_top_k[:num_contexts],
        )

    if num_gens > 0:
        gen_num_accepted = num_accepted_tokens[num_contexts:num_seqs]

        logits_table = logits[num_contexts:].reshape(
            num_gens, draft_len + 1, logits.shape[1]
        )

        # resample the first rejected draft token
        rejected_logits = logits_table[
            batch_indices_cuda[:num_gens], gen_num_accepted - 1
        ]
        new_tokens = resample(
            rejected_logits,
            extern_temperature[num_contexts:num_seqs],
            extern_top_p[num_contexts:num_seqs],
            extern_top_k[num_contexts:num_seqs],
        )
        accepted_tokens[batch_indices_cuda[num_contexts:num_seqs], gen_num_accepted - 1] = new_tokens
