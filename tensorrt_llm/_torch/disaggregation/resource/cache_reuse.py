# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Union

import numpy as np
import torch

from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager

from .page import AttentionLayerGroup
from .utils import get_global_layer_ids


class CacheReuseAdapter(ABC):
    """Uniform prefix-reuse API over KVCacheManager V1/V2."""

    @property
    @abstractmethod
    def enable_block_reuse(self) -> bool: ...

    @property
    @abstractmethod
    def tokens_per_block(self) -> int: ...

    @abstractmethod
    def _global_cached_token_count(self, req: LlmRequest) -> int:
        """Block-aligned cached prefix length reported by the cache manager."""

    def get_cached_token_count_per_layer_group(
        self,
        req: LlmRequest,
        layer_groups: Sequence[AttentionLayerGroup],
    ) -> List[int]:
        """Per-layer-group cached prefix in tokens (block-aligned).

        Returns the reuse-hit prefix only; SWA stale-region handling lives at
        the transfer call site (it is a transport concern, not a cache one).
        """
        if not self.enable_block_reuse:
            return [0] * len(layer_groups)
        scalar = max(0, self._global_cached_token_count(req))
        return [scalar] * len(layer_groups)

    def begin_kv_slice(self, req: LlmRequest) -> None:
        pass

    def end_kv_slice(self, req: LlmRequest) -> None:
        pass

    @abstractmethod
    def get_block_ids(
        self,
        req: LlmRequest,
        group_idx: int,
        lg: AttentionLayerGroup,
    ) -> np.ndarray:
        """Physical pool block IDs for *req* in layer group *lg*."""

    @abstractmethod
    def commit_blocks_for_reuse(self, req: LlmRequest) -> None:
        """Commit KV blocks to radix tree for future prefix reuse.

        Must be called after ``req.context_current_position = req.prompt_len``.
        """


class _CacheReuseAdapterV1(CacheReuseAdapter):
    """C++-backed KVCacheManager."""

    def __init__(self, mgr: KVCacheManager) -> None:
        self._mgr = mgr
        self._host_offsets_request_id: Optional[int] = None
        self._slice_host_offsets: Optional[torch.Tensor] = None

    @property
    def enable_block_reuse(self) -> bool:
        return self._mgr.enable_block_reuse

    @property
    def tokens_per_block(self) -> int:
        return self._mgr.tokens_per_block

    def _global_cached_token_count(self, req: LlmRequest) -> int:
        if not self.enable_block_reuse:
            return 0
        tpb = self.tokens_per_block
        return (req.prepopulated_prompt_len // tpb) * tpb

    def get_block_ids(self, req, group_idx, lg):  # noqa: ARG002
        first_layer = get_global_layer_ids(lg)[0]
        beam_width = req.py_beam_width
        block_ids = np.asarray(
            self._mgr.get_batch_cache_indices(
                [req.py_request_id], layer_idx=first_layer, beam_width=beam_width
            )[0],
            dtype=np.int64,
        )
        if beam_width != 1:
            return block_ids
        return self._decode_v1_physical_block_ids(req, lg, block_ids)

    def begin_kv_slice(self, req: LlmRequest) -> None:
        self._host_offsets_request_id = None
        if getattr(req, "py_beam_width", 1) == 1:
            self._refresh_v1_physical_block_offsets(req)

    def end_kv_slice(self, req: LlmRequest) -> None:  # noqa: ARG002
        self._host_offsets_request_id = None

    def _has_secondary_pool(self) -> bool:
        blocks_per_window = getattr(self._mgr, "blocks_per_window", None)
        if blocks_per_window is not None:
            return any(secondary > 0 for _, secondary in blocks_per_window.values())
        return getattr(self._mgr, "blocks_in_secondary_pool", 0) > 0

    def _refresh_v1_physical_block_offsets(self, req: LlmRequest) -> bool:
        mgr_host_offsets = getattr(self._mgr, "host_kv_cache_block_offsets", None)
        copy_offsets = getattr(getattr(self._mgr, "impl", None), "copy_batch_block_offsets", None)
        if mgr_host_offsets is None or copy_offsets is None:
            return False
        if not self._has_secondary_pool():
            return False

        # Stage into an adapter-owned buffer, never into the manager's shared
        # pinned host_kv_cache_block_offsets: the overlap executor loop may
        # still have a pending non_blocking H2D copy reading that tensor for
        # the in-flight batch (TrtllmAttentionMetadata.prepare), and rewriting
        # it here would retarget the live batch's device block-offset table.
        if self._slice_host_offsets is None:
            self._slice_host_offsets = torch.zeros(
                (
                    mgr_host_offsets.shape[0],
                    1,
                    mgr_host_offsets.shape[2],
                    mgr_host_offsets.shape[3],
                ),
                dtype=mgr_host_offsets.dtype,
                device="cpu",
            )
        copy_offsets(self._slice_host_offsets, [req.py_request_id], 1, 0)
        self._host_offsets_request_id = int(req.py_request_id)
        return True

    def _decode_v1_physical_block_ids(
        self,
        req: LlmRequest,
        lg: AttentionLayerGroup,
        block_ids: np.ndarray,
    ) -> np.ndarray:
        if block_ids.size == 0:
            return block_ids
        if self._host_offsets_request_id != int(req.py_request_id):
            if not self._refresh_v1_physical_block_offsets(req):
                return block_ids
        host_offsets = self._slice_host_offsets
        if host_offsets is None:
            return block_ids
        # A wrong decode here silently transfers the wrong physical blocks,
        # so metadata inconsistencies must raise instead of falling back to
        # logical block ids.
        local_layer_id = lg.local_layers[0].local_layer_id
        pool_idx = int(self._mgr.kv_cache_pool_mapping[local_layer_id][0].item())

        # C++ WindowBlockManager::setOffsets encodes block-first pools as:
        # memoryPoolBlockIndex * pool.numLayers * kvFactor + layer/kv offset.
        # B10 page-table slots use that same block-first pool layout.
        encoded_offsets = host_offsets[pool_idx, 0, 0, : block_ids.size]
        if encoded_offsets.numel() != block_ids.size:
            raise RuntimeError(
                f"V1 host block-offset snapshot holds "
                f"{encoded_offsets.numel()} entries for request "
                f"{req.py_request_id} but {block_ids.size} block ids need "
                f"physical decoding"
            )

        slot_stride = len(lg.local_layers) * self._mgr.kv_factor
        encoded_np = encoded_offsets.cpu().numpy().astype(np.int64, copy=False)
        physical = block_ids.copy()
        valid = encoded_np >= 0
        physical[valid] = encoded_np[valid] // slot_stride
        physical[~valid] = -1
        return physical

    def commit_blocks_for_reuse(self, req: LlmRequest) -> None:
        if not self.enable_block_reuse:
            return
        self._mgr.store_blocks_for_reuse(req, pin_blocks=False)


class _CacheReuseAdapterV2(CacheReuseAdapter):
    """Python-based KVCacheManagerV2."""

    def __init__(self, mgr: KVCacheManagerV2) -> None:
        self._mgr = mgr

    @property
    def enable_block_reuse(self) -> bool:
        return self._mgr.enable_block_reuse

    @property
    def tokens_per_block(self) -> int:
        return self._mgr.tokens_per_block

    def _global_cached_token_count(self, req: LlmRequest) -> int:
        if not self.enable_block_reuse:
            return 0
        kv_cache = self._mgr.kv_cache_map.get(req.py_request_id)
        if kv_cache is None:
            return 0
        tpb = self.tokens_per_block
        return (kv_cache.num_committed_tokens // tpb) * tpb

    def get_block_ids(self, req, group_idx, lg):  # noqa: ARG002
        return np.fromiter(
            self._mgr.kv_cache_map[req.py_request_id].get_aggregated_page_indices(
                group_idx, valid_only=True
            ),
            dtype=np.int64,
        )

    def commit_blocks_for_reuse(self, req: LlmRequest) -> None:
        if not self.enable_block_reuse:
            return
        kv_cache = self._mgr.kv_cache_map.get(req.py_request_id)
        if kv_cache is None:
            return
        self._mgr.try_commit_blocks_for_reuse(req, kv_cache)


def create_cache_reuse_adapter(
    mgr: Union[KVCacheManager, KVCacheManagerV2],
) -> CacheReuseAdapter:
    """Factory — pick the right adapter for the concrete manager type."""
    if isinstance(mgr, KVCacheManagerV2):
        return _CacheReuseAdapterV2(mgr)
    return _CacheReuseAdapterV1(mgr)
