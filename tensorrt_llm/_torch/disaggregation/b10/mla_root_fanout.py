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
import os
import time
from dataclasses import replace
from typing import Optional

import numpy as np
import torch

from tensorrt_llm import logger
from tensorrt_llm._torch.disaggregation.b10.kernels import (
    _scatter_cuda_buffer_to_vram_spans,
    _span_view,
)
from tensorrt_llm._torch.disaggregation.b10.memory import _DescArrayView
from tensorrt_llm._torch.disaggregation.b10.planning import (
    _coalesce_memory_descs,
    _contiguous_desc_spans,
)
from tensorrt_llm._torch.disaggregation.base.transfer import KVSlice, get_unique_rid
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import KVRegionExtractorV1
from tensorrt_llm._torch.disaggregation.resource.page import AttentionLayerGroup, KVCachePageTable
from tensorrt_llm._torch.distributed.communicator import Distributed
from tensorrt_llm._torch.distributed.ops import _allgather
from tensorrt_llm._torch.distributed.pg_utils import split
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._utils import mpi_disabled
from tensorrt_llm.bindings import LlmRequestState
from tensorrt_llm.mapping import Mapping

_MLA_ROOT_FANOUT_ENV = "TRTLLM_MLA_KVCACHE_ROOT_FANOUT_FORCE"
_MLA_ROOT_FANOUT_CHUNK_SIZE = 64 * 1024 * 1024
_MLA_ROOT_FANOUT_GROUP_SIZE = 4


def _broadcast_tensor(
    input: torch.Tensor,
    group: list[int],
    rank: int,
    root: int,
    group_boxed: Optional[object],
) -> torch.Tensor:
    if len(group) == 1:
        return input

    if mpi_disabled():
        if group_boxed is None:
            raise RuntimeError("B10 MLA root fanout subgroup is not initialized")
        process_group = torch.distributed.ProcessGroup.unbox(group_boxed)
        torch.distributed.broadcast(input, src=group[root], group=process_group)
        return input

    flat_input = input.contiguous().view(-1)
    sizes = [0] * len(group)
    sizes[root] = flat_input.numel()
    local_input = flat_input if rank == root else flat_input[:0]
    output = _allgather(
        local_input,
        group,
        rank,
        group_boxed,
        dim=0,
        sizes=sizes,
    )
    return output.view_as(input)


class MLARootFanout:
    """Pull disagg MLA KV on rotating TP leaders, then broadcast to peers.

    Opt-in via TRTLLM_MLA_KVCACHE_ROOT_FANOUT_FORCE=1,
    Only TP, no PP/CP/attention-DP, kv_factor 1 (MLA).
    """

    @classmethod
    def create_if_enabled(
        cls,
        mapping: Mapping,
        dist: Distributed,
        kv_cache_manager: KVCacheManager,
        page_table: KVCachePageTable,
        device_id: int,
    ) -> Optional["MLARootFanout"]:
        enabled = os.getenv(_MLA_ROOT_FANOUT_ENV) == "1"

        valid_topology = (
            mapping.tp_size > 1
            and mapping.pp_size == 1
            and mapping.cp_size == 1
            and not mapping.enable_attention_dp
            and kv_cache_manager.kv_factor == 1
        )
        if not enabled:
            return None
        elif enabled and not valid_topology:
            raise RuntimeError(
                "B10 MLA root fanout requires valid topology: "
                f"tp_size={mapping.tp_size} pp_size={mapping.pp_size} cp_size={mapping.cp_size} "
                f"enable_attention_dp={mapping.enable_attention_dp} kv_factor={kv_cache_manager.kv_factor}\n"
                f"Only TP kv_factor 1 (MLA) is supported; no PP/CP/attention-DP."
            )
        return cls(mapping, dist, page_table, device_id)

    def __init__(
        self,
        mapping: Mapping,
        dist: Distributed,
        page_table: KVCachePageTable,
        device_id: int,
    ) -> None:
        self._mapping = mapping
        self._dist = dist
        self._page_table = page_table
        self._device_id = device_id
        self._extractor = KVRegionExtractorV1(page_table)
        self._pending_slices: dict[int, KVSlice] = {}
        group_start = (mapping.tp_rank // _MLA_ROOT_FANOUT_GROUP_SIZE) * (
            _MLA_ROOT_FANOUT_GROUP_SIZE
        )
        group_end = min(group_start + _MLA_ROOT_FANOUT_GROUP_SIZE, mapping.tp_size)
        self._fanout_group = mapping.tp_group[group_start:group_end]
        self._fanout_group_rank = mapping.tp_rank - group_start
        self._fanout_group_boxed = None
        if mpi_disabled():
            tp_group_boxed = mapping.tp_group_pg.boxed()
            self._fanout_group_boxed = (
                tp_group_boxed
                if len(self._fanout_group) == mapping.tp_size
                else split(
                    color=group_start // _MLA_ROOT_FANOUT_GROUP_SIZE,
                    key=self._fanout_group_rank,
                    pg_boxed=tp_group_boxed,
                )
            )
        logger.debug(
            "B10 MLA root fanout enabled: "
            f"tp_rank={mapping.tp_rank} tp_size={mapping.tp_size} "
            f"fanout_group={self._fanout_group} "
            f"chunk_size={_MLA_ROOT_FANOUT_CHUNK_SIZE}"
        )

    def _root_group_rank(self, request_id: int) -> int:
        # Derived from the rid so group members agree without coordinating a
        # mutable counter.
        return request_id % len(self._fanout_group)

    def _root_tp_rank(self, request_id: int) -> int:
        root_group_rank = self._root_group_rank(request_id)
        return self._mapping.tp_group.index(self._fanout_group[root_group_rank])

    def register_receive_slice(self, req: LlmRequest, kv_slice: KVSlice) -> KVSlice:
        """Save the full slice for fanout; return the slice the session should request.

        Non-leader ranks get a zero-block copy. Each group leader pulls the
        full remote payload, while the sender transfers nothing to its peers.
        """
        request_id = get_unique_rid(req)
        if request_id is None:
            raise RuntimeError("B10 MLA root fanout requires request sync metadata")
        if request_id in self._pending_slices:
            raise RuntimeError(
                f"B10 MLA root fanout requires exactly one receive slice per request: "
                f"rid={request_id}"
            )
        self._pending_slices[request_id] = kv_slice
        root_rank = self._root_tp_rank(request_id)
        is_root = self._mapping.tp_rank == root_rank
        logger.debug(
            "B10 MLA root fanout receive request: "
            f"rid={request_id} tp_rank={self._mapping.tp_rank} "
            f"root_rank={root_rank} "
            f"requests_remote_payload={is_root}"
        )
        if is_root:
            return kv_slice
        return replace(
            kv_slice,
            block_ids_per_layer_groups=[
                np.empty(0, dtype=np.int64) for _ in kv_slice.block_ids_per_layer_groups
            ],
        )

    def finish_sync_receive(self, req: LlmRequest, local_error: Optional[Exception] = None) -> None:
        request_id = get_unique_rid(req)
        kv_slice = self._pending_slices.pop(request_id, None)
        local_outcome = (
            request_id,
            req.state == LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE,
            kv_slice is not None,
            None if local_error is None else f"{type(local_error).__name__}: {local_error}",
        )
        outcomes = self._dist.tp_allgather(local_outcome)
        expected_request_id = outcomes[0][0]
        succeeded = (
            expected_request_id is not None
            and all(outcome[0] == expected_request_id for outcome in outcomes)
            and all(outcome[1] and outcome[2] and outcome[3] is None for outcome in outcomes)
        )
        if not succeeded:
            req.state = LlmRequestState.DISAGG_TRANS_ERROR
            message = (
                "B10 MLA root fanout sync receive failed before fanout: "
                f"rid={request_id} outcomes={outcomes}"
            )
            if local_error is not None:
                raise RuntimeError(message) from local_error
            raise RuntimeError(message)

        assert request_id is not None
        assert kv_slice is not None
        self._fanout(request_id, kv_slice)

    def finish_async_receives(self, completed_rids: list[int], failed_rids: list[int]) -> None:
        for rid in completed_rids:
            kv_slice = self._pending_slices.pop(rid, None)
            slice_availability = self._dist.tp_allgather(kv_slice is not None)
            if not all(slice_availability):
                raise RuntimeError(
                    "B10 MLA root fanout is missing a completed receive slice: "
                    f"rid={rid} availability={slice_availability}"
                )
            assert kv_slice is not None
            self._fanout(rid, kv_slice)
        for rid in failed_rids:
            self._pending_slices.pop(rid, None)

    def _extract_desc_sets(self, kv_slice: KVSlice) -> list[_DescArrayView]:
        descs = []
        for layer_group_id, layer_group in enumerate(self._page_table.layer_groups):
            if not isinstance(layer_group, AttentionLayerGroup):
                continue
            block_ids = kv_slice.block_ids_per_layer_groups[layer_group_id]
            for pool_idx in range(len(layer_group.pool_views)):
                region = self._extractor.extract(block_ids, layer_group_id, pool_idx)
                ptrs = np.asarray(region.memory.ptrs, dtype=np.int64)
                sizes = np.full(ptrs.shape, region.memory.bytes_per_region, dtype=np.int64)
                device_ids = np.full(ptrs.shape, self._device_id, dtype=np.int64)
                descs.append(_DescArrayView(ptrs, sizes, device_ids))
        return descs

    def _fanout(self, request_id: int, kv_slice: KVSlice) -> None:
        desc_sets = self._extract_desc_sets(kv_slice)
        signature = [(len(descs), int(descs.sizes.sum())) for descs in desc_sets]
        signatures = self._dist.tp_allgather(signature)
        if any(peer_signature != signature for peer_signature in signatures):
            raise RuntimeError(
                "B10 MLA root fanout requires identical descriptor sizes across decode TP ranks: "
                f"{signatures}"
            )

        chunk_sets = [
            _coalesce_memory_descs(descs, _MLA_ROOT_FANOUT_CHUNK_SIZE) for descs in desc_sets
        ]
        max_chunk_size = max(
            (chunk.size for chunks in chunk_sets for chunk in chunks),
            default=0,
        )
        if max_chunk_size == 0:
            return

        scratch = torch.empty(max_chunk_size, dtype=torch.uint8, device=self._device_id)
        copy_stream = torch.cuda.current_stream(self._device_id)
        root_group_rank = self._root_group_rank(request_id)
        root_rank = self._root_tp_rank(request_id)
        is_root = self._fanout_group_rank == root_group_rank
        total_bytes = 0
        chunk_count = 0
        started = time.perf_counter()
        for descs, chunks in zip(desc_sets, chunk_sets):
            for chunk in chunks:
                spans = _contiguous_desc_spans(descs, chunk)
                scratch_view = scratch[: chunk.size]
                lifetime_refs = []
                if is_root:
                    with torch.cuda.stream(copy_stream):
                        for span in spans:
                            src_view = _span_view(descs, span, "VRAM")
                            scratch_view[span.chunk_offset : span.chunk_offset + span.size].copy_(
                                src_view.buffer
                            )
                            lifetime_refs.append(src_view)
                copy_stream.synchronize()

                fanout_view = _broadcast_tensor(
                    scratch_view,
                    self._fanout_group,
                    self._fanout_group_rank,
                    root_group_rank,
                    self._fanout_group_boxed,
                )

                if not is_root:
                    _scatter_cuda_buffer_to_vram_spans(
                        fanout_view,
                        descs,
                        spans,
                        copy_stream,
                        lifetime_refs=lifetime_refs,
                    )
                    copy_stream.synchronize()
                lifetime_refs.clear()
                total_bytes += chunk.size
                chunk_count += 1
        logger.debug(
            "B10 MLA root fanout complete: "
            f"rid={request_id} tp_rank={self._mapping.tp_rank} "
            f"root_rank={root_rank} "
            f"bytes={total_bytes} chunks={chunk_count} "
            f"duration_ms={(time.perf_counter() - started) * 1000:.3f}"
        )
