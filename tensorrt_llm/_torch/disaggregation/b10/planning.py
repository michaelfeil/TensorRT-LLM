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
from __future__ import annotations

import operator
from typing import Any, Optional

import numpy as np

from tensorrt_llm._torch.disaggregation.b10.memory import (
    _DescArrayView,
    _DescPairOrder,
    _DestinationScatterPlan,
    _SpanArrays,
    _TransferChunk,
)
from tensorrt_llm._torch.disaggregation.base.agent import MemoryDescs


def _memory_desc_stats(descs: _DescArrayView) -> tuple[int, int, int]:
    if not descs:
        return 0, 0, 0
    _, sizes, _ = _desc_arrays(descs)
    return len(descs), int(sizes.sum()), int(sizes.max())


def _transfer_chunk_stats(chunks: list[_TransferChunk]) -> tuple[int, int, int]:
    if not chunks:
        return 0, 0, 0
    sizes = [chunk.size for chunk in chunks]
    return len(chunks), sum(sizes), max(sizes)


def _coalesce_memory_descs(descs: _DescArrayView, max_chunk_size: int) -> list[_TransferChunk]:
    """Greedily coalesce adjacent descriptors into transfer chunks.

    Vectorized reformulation (prefix sums + one searchsorted per chunk) of
    the original per-descriptor greedy loop with identical boundaries: a
    chunk flushes the moment its running size hits max_chunk_size exactly,
    flushes before a descriptor that would push a positive running size
    past max_chunk_size, and always accepts the first positive-size
    descriptor (even oversize, and never split from leading zero-size
    descriptors, which alone never flush).
    """
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be positive")
    _, sizes, _ = _desc_arrays(descs)
    total_count = int(sizes.shape[0])
    if total_count == 0:
        return []
    if bool((sizes < 0).any()):
        raise ValueError("memory descriptor size must be non-negative")
    # prefix[j] = total size of descs[:j].
    prefix = np.concatenate(([0], np.cumsum(sizes)))
    positive = np.flatnonzero(sizes > 0)
    chunks: list[_TransferChunk] = []
    start = 0
    while start < total_count:
        target = int(prefix[start]) + max_chunk_size
        # Smallest j with prefix[j] >= target: the exact-fit flush point
        # when equal, else one past the last descriptor that still fits.
        # j may be len(prefix) when no prefix entry reaches target, hence
        # the j <= total_count guard before indexing prefix[j].
        # Loop progress (end > start) holds in every branch: a zero-size
        # head keeps prefix flat so j - 1 >= start + 1, and an oversize
        # positive descriptor (j == start + 1, so end == start) is rescued
        # by the positive[k] + 1 override below.
        j = int(np.searchsorted(prefix, target, side="left"))
        if j <= total_count and int(prefix[j]) == target:
            end = j
        else:
            end = min(j - 1, total_count)
        k = int(np.searchsorted(positive, start))
        if k < positive.shape[0]:
            # First positive-size descriptor after start is always accepted.
            end = max(end, int(positive[k]) + 1)
        else:
            # All-zero tail forms one final chunk.
            end = total_count
        chunks.append(
            _TransferChunk(start=start, count=end - start, size=int(prefix[end] - prefix[start]))
        )
        start = end
    return chunks


def _transfer_chunks_to_control(chunks: list[_TransferChunk]) -> list[dict[str, int]]:
    return [
        {
            "start": chunk.start,
            "count": chunk.count,
            "size": chunk.size,
        }
        for chunk in chunks
    ]


def _transfer_chunks_from_control(
    control: dict[str, Any], descs: _DescArrayView
) -> list[_TransferChunk]:
    raw_chunks = control.get("transfer_chunks")
    _, sizes, _ = _desc_arrays(descs)
    if raw_chunks is None:
        # Cold path (production senders always emit transfer_chunks): one
        # chunk per descriptor.
        return [
            _TransferChunk(start=idx, count=1, size=size) for idx, size in enumerate(sizes.tolist())
        ]

    chunks = [
        _TransferChunk(start=int(item["start"]), count=int(item["count"]), size=int(item["size"]))
        for item in raw_chunks
    ]
    # Cumulative-sum lookup replaces the per-chunk sum() generator: one
    # vectorized pass over the sizes instead of O(descs) Python-object work
    # for every chunk.
    cumulative_sizes = np.concatenate(([0], np.cumsum(sizes)))
    expected_start = 0
    for chunk in chunks:
        if chunk.start != expected_start:
            raise ValueError(
                f"B10 transfer chunk starts at {chunk.start}, expected {expected_start}"
            )
        if chunk.count <= 0:
            raise ValueError("B10 transfer chunk count must be positive")
        if chunk.start + chunk.count > len(descs):
            raise ValueError("B10 transfer chunk exceeds descriptor list")
        expected_size = int(
            cumulative_sizes[chunk.start + chunk.count] - cumulative_sizes[chunk.start]
        )
        if chunk.size != expected_size:
            raise ValueError(
                f"B10 transfer chunk size mismatch: expected {expected_size}, got {chunk.size}"
            )
        expected_start += chunk.count
    if expected_start != len(descs):
        raise ValueError(
            f"B10 transfer chunks cover {expected_start} descriptors, expected {len(descs)}"
        )
    return chunks


def _desc_arrays(
    descs: _DescArrayView,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Named accessor for a view's three descriptor columns."""
    return descs.ptrs, descs.sizes, descs.device_ids


def _arrays_have_overlap(ptrs: np.ndarray, sizes: np.ndarray, devices: np.ndarray) -> bool:
    mask = sizes > 0
    if not bool(mask.any()):
        return False
    ptrs, sizes, devices = ptrs[mask], sizes[mask], devices[mask]
    order = np.lexsort((ptrs, devices))
    ptrs, sizes, devices = ptrs[order], sizes[order], devices[order]
    # Running-max accumulation must not leak across device boundaries.
    device_boundaries = np.flatnonzero(devices[1:] != devices[:-1]) + 1
    ends = ptrs + sizes
    for group_ptrs, group_ends in zip(
        np.split(ptrs, device_boundaries), np.split(ends, device_boundaries)
    ):
        if group_ptrs.size < 2:
            continue
        running_end = np.maximum.accumulate(group_ends[:-1])
        if bool(np.any(group_ptrs[1:] < running_end)):
            return True
    return False


def _has_overlapping_descs(descs: _DescArrayView) -> bool:
    return _arrays_have_overlap(*_desc_arrays(descs))


def _desc_break_mask(ptrs: np.ndarray, sizes: np.ndarray, devices: np.ndarray) -> np.ndarray:
    """True at i where descriptor i+1 starts a new contiguous span: the
    device changes or the next ptr is not exactly prev ptr + prev size."""
    return (devices[1:] != devices[:-1]) | (ptrs[1:] != ptrs[:-1] + sizes[:-1])


def _span_count_from_arrays(
    ptrs: np.ndarray, sizes: np.ndarray, devices: np.ndarray, chunks: list[_TransferChunk]
) -> int:
    # Mirrors len(_contiguous_desc_spans(descs, chunk)) summed over chunks.
    if len(ptrs) == 0:
        return 0
    breaks = _desc_break_mask(ptrs, sizes, devices)
    cumulative_breaks = np.concatenate(([0], np.cumsum(breaks)))
    total = 0
    for chunk in chunks:
        if chunk.count <= 0:
            continue
        total += 1 + int(
            cumulative_breaks[chunk.start + chunk.count - 1] - cumulative_breaks[chunk.start]
        )
    return total


def _reorder_desc_pairs_for_contiguity(
    src_descs: _DescArrayView,
    dst_descs: _DescArrayView,
    max_chunk_size: int,
) -> _DescPairOrder:
    if len(src_descs) != len(dst_descs):
        raise ValueError(
            f"B10 source/destination descriptor counts differ: {len(src_descs)} != {len(dst_descs)}"
        )

    src_arrays = _desc_arrays(src_descs)
    dst_arrays = _desc_arrays(dst_descs)

    def make_result(order: Optional[np.ndarray], strategy: str) -> _DescPairOrder:
        if order is None:
            src_ptrs, src_sizes, src_devices = src_arrays
            dst_ptrs, dst_sizes, dst_devices = dst_arrays
        else:
            src_ptrs, src_sizes, src_devices = (a[order] for a in src_arrays)
            dst_ptrs, dst_sizes, dst_devices = (a[order] for a in dst_arrays)
        # Array-backed descs for both sides: every downstream consumer
        # (chunking, spans, control packing) reuses the columns directly.
        ordered_src = _DescArrayView(src_ptrs, src_sizes, src_devices)
        ordered_dst = _DescArrayView(dst_ptrs, dst_sizes, dst_devices)
        chunks = _coalesce_memory_descs(ordered_src, max_chunk_size)
        return _DescPairOrder(
            src_descs=ordered_src,
            dst_descs=ordered_dst,
            strategy=strategy,
            transfer_chunks=chunks,
            src_span_count=_span_count_from_arrays(src_ptrs, src_sizes, src_devices, chunks),
            dst_span_count=_span_count_from_arrays(dst_ptrs, dst_sizes, dst_devices, chunks),
        )

    if _arrays_have_overlap(*src_arrays) or _arrays_have_overlap(*dst_arrays):
        return make_result(None, "original")

    # Stable sort by (device_id, ptr); ties keep original index order,
    # matching the previous (device_id, ptr, idx) key.
    return make_result(np.lexsort((src_arrays[0], src_arrays[2])), "src_ptr")


def _validate_matching_desc_sizes(src_descs: _DescArrayView, dst_descs: _DescArrayView) -> None:
    if len(src_descs) != len(dst_descs):
        raise ValueError(
            f"B10 source/destination descriptor counts differ: {len(src_descs)} != {len(dst_descs)}"
        )
    _, src_sizes, _ = _desc_arrays(src_descs)
    _, dst_sizes, _ = _desc_arrays(dst_descs)
    mismatch = src_sizes != dst_sizes
    if bool(mismatch.any()):
        idx = int(np.flatnonzero(mismatch)[0])
        raise ValueError(
            f"B10 source/destination descriptor size mismatch at {idx}: "
            f"{int(src_sizes[idx])} != {int(dst_sizes[idx])}"
        )


def _contiguous_desc_spans(descs: _DescArrayView, chunk: _TransferChunk) -> _SpanArrays:
    """Contiguous spans of a chunk's descriptors as `_SpanArrays` columns.

    Break indices come from `_desc_break_mask`, restricted to the chunk's
    [start, start + count) range; O(1) Python-object work regardless of
    descriptor or span count.
    """
    stop = chunk.start + chunk.count
    ptrs = descs.ptrs[chunk.start : stop]
    sizes = descs.sizes[chunk.start : stop]
    devices = descs.device_ids[chunk.start : stop]
    count = int(ptrs.shape[0])
    if count == 0:
        empty = np.empty(0, dtype=np.int64)
        return _SpanArrays(empty, empty, empty)
    # offsets[i] = chunk offset of descriptor i; last entry = chunk size.
    offsets = np.concatenate(([0], np.cumsum(sizes)))
    breaks = _desc_break_mask(ptrs, sizes, devices)
    span_starts = np.concatenate(([0], np.flatnonzero(breaks) + 1))
    span_stops = np.concatenate((span_starts[1:], [count]))
    return _SpanArrays(
        starts=chunk.start + span_starts,
        sizes=offsets[span_stops] - offsets[span_starts],
        chunk_offsets=offsets[span_starts],
    )


def _normalize_memory_descs(memory_descs: MemoryDescs) -> _DescArrayView:
    """Ingest a MemoryDescs into a `_DescArrayView` in one bulk pass.

    The C++ MemoryDescs binding exposes only per-item descriptors
    (addr/len/device_id properties), so ingestion cannot avoid touching
    each item once; it CAN avoid building a per-descriptor Python object
    that every downstream consumer would re-vectorize. Tuple descs (the
    pure-Python MemoryDescs) convert in one C-level np.asarray pass;
    attribute descs take three per-column fromiter passes (measurably
    faster on nanobind objects than one flat 3N-yield pass + reshape).
    Descriptor kinds are homogeneous within one MemoryDescs (all items
    come from the same constructor), so the first item picks the field
    names.
    """
    descs = memory_descs.descs
    count = len(descs)
    if count == 0:
        empty = np.empty(0, dtype=np.int64)
        return _DescArrayView(empty, empty, empty)
    first = descs[0]
    if isinstance(first, (tuple, list)):
        flat = np.asarray(descs, dtype=np.int64)
        return _DescArrayView(flat[:, 0], flat[:, 1], flat[:, 2])
    ptr_name = "ptr" if hasattr(first, "ptr") else "addr"
    size_name = "size" if hasattr(first, "size") else "len"
    if not (hasattr(first, ptr_name) and hasattr(first, size_name) and hasattr(first, "device_id")):
        raise TypeError(f"Unsupported memory descriptor type: {type(first)!r}")
    get_ptr = operator.attrgetter(ptr_name)
    get_size = operator.attrgetter(size_name)
    get_device = operator.attrgetter("device_id")
    return _DescArrayView(
        np.fromiter((get_ptr(desc) for desc in descs), np.int64, count=count),
        np.fromiter((get_size(desc) for desc in descs), np.int64, count=count),
        np.fromiter((get_device(desc) for desc in descs), np.int64, count=count),
    )


def _desc_view_from_arrays(
    ptrs: np.ndarray, sizes: np.ndarray, device_ids: int | np.ndarray
) -> _DescArrayView:
    """Build a `_DescArrayView` from producer-supplied parallel arrays.

    The side-channel ingestion boundary: a producer that already holds the
    descriptor columns (the native Sender builds its C++ MemoryDescs from
    these exact arrays) hands them over directly instead of paying the
    per-item nanobind walk in `_normalize_memory_descs`. `device_ids` may
    be a scalar (the uniform-device case; expanded here) or a per-desc
    array. The arrays are aliased, not copied, so they must describe the
    request they accompany and must never be mutated afterwards (same
    read-only contract as every `_DescArrayView` column). Shape errors
    raise loudly: a mismatch is a producer bug.
    """
    ptrs = np.asarray(ptrs, dtype=np.int64)
    sizes = np.asarray(sizes, dtype=np.int64)
    if ptrs.ndim != 1 or ptrs.shape != sizes.shape:
        raise ValueError(
            f"B10 desc arrays malformed: ptrs shape {ptrs.shape} vs sizes shape {sizes.shape}"
        )
    if np.ndim(device_ids) == 0:
        device_ids = np.full(ptrs.shape[0], int(device_ids), dtype=np.int64)
    else:
        device_ids = np.asarray(device_ids, dtype=np.int64)
        if device_ids.shape != ptrs.shape:
            raise ValueError(
                f"B10 desc arrays malformed: device_ids shape "
                f"{device_ids.shape} vs ptrs shape {ptrs.shape}"
            )
    return _DescArrayView(ptrs, sizes, device_ids)


def _destination_scatter_plan_for_chunks(
    descs: _DescArrayView,
    chunk_sources: list[tuple[_TransferChunk, int]],
) -> _DestinationScatterPlan:
    desc_ptrs, desc_sizes, desc_devices = _desc_arrays(descs)
    src_parts: list[np.ndarray] = []
    dst_parts: list[np.ndarray] = []
    size_parts: list[np.ndarray] = []
    device_parts: list[np.ndarray] = []
    for chunk, src_base_ptr in chunk_sources:
        stop = chunk.start + chunk.count
        chunk_sizes = desc_sizes[chunk.start : stop]
        # Exclusive prefix sum of desc sizes = per-desc offset into the chunk.
        chunk_offsets = np.zeros(chunk_sizes.shape[0], dtype=np.int64)
        np.cumsum(chunk_sizes[:-1], out=chunk_offsets[1:])
        src_parts.append(src_base_ptr + chunk_offsets)
        dst_parts.append(desc_ptrs[chunk.start : stop])
        size_parts.append(chunk_sizes)
        device_parts.append(desc_devices[chunk.start : stop])
    empty = np.empty(0, dtype=np.int64)
    src_ptrs = np.concatenate(src_parts) if src_parts else empty
    dst_ptrs = np.concatenate(dst_parts) if dst_parts else empty
    sizes = np.concatenate(size_parts) if size_parts else empty
    device_ids = np.concatenate(device_parts) if device_parts else empty

    nonzero = sizes > 0
    if not bool(nonzero.all()):
        src_ptrs = src_ptrs[nonzero]
        dst_ptrs = dst_ptrs[nonzero]
        sizes = sizes[nonzero]
        device_ids = device_ids[nonzero]
    total_bytes = int(sizes.sum())

    # Stable sort by (device_id, dst_ptr, src_ptr), matching the previous
    # fragments.sort key.
    order = np.lexsort((src_ptrs, dst_ptrs, device_ids))
    src_ptrs = src_ptrs[order]
    dst_ptrs = dst_ptrs[order]
    sizes = sizes[order]
    device_ids = device_ids[order]

    # Coalesce fragments contiguous in both source and destination. A run's
    # end always equals its last fragment's (ptr + size), so the previous
    # sequential merge reduces to a pairwise adjacency test.
    if src_ptrs.shape[0] > 1:
        contiguous = (
            (device_ids[1:] == device_ids[:-1])
            & (dst_ptrs[1:] == dst_ptrs[:-1] + sizes[:-1])
            & (src_ptrs[1:] == src_ptrs[:-1] + sizes[:-1])
        )
        if bool(contiguous.any()):
            starts = np.concatenate(([0], np.flatnonzero(~contiguous) + 1))
            # Sums sizes over each contiguous run [starts[i], starts[i+1]).
            coalesced_sizes = np.add.reduceat(sizes, starts)
            src_ptrs = src_ptrs[starts]
            dst_ptrs = dst_ptrs[starts]
            sizes = coalesced_sizes
            device_ids = device_ids[starts]
    return _DestinationScatterPlan(
        src_ptrs=src_ptrs,
        dst_ptrs=dst_ptrs,
        sizes=sizes,
        device_ids=device_ids,
        total_bytes=total_bytes,
    )


def _scatter_program_count_for_spans(spans: _SpanArrays, block_size: int) -> int:
    # ceil(sizes / block_size) summed, as array math.
    return int((-(-spans.sizes // block_size)).sum())


def _scatter_programs_for_fragment_sizes(
    sizes: np.ndarray,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized one-program-per-block expansion over fragment sizes.

    Equivalent to emitting (fragment_idx, block_offset) for every
    block_offset in range(0, size, block_size), fragment by fragment.
    """
    programs_per_fragment = -(-sizes // block_size)  # ceil(sizes / block_size)
    program_count = int(programs_per_fragment.sum())
    program_fragment_indices = np.repeat(
        np.arange(sizes.shape[0], dtype=np.int64), programs_per_fragment
    )
    first_program_of_fragment = np.repeat(
        np.cumsum(programs_per_fragment) - programs_per_fragment, programs_per_fragment
    )
    program_offsets = (
        np.arange(program_count, dtype=np.int64) - first_program_of_fragment
    ) * block_size
    return program_fragment_indices, program_offsets


def _gather_plan_for_vram_spans(
    descs: _DescArrayView,
    spans: _SpanArrays,
    staging_base_ptr: int,
) -> _DestinationScatterPlan:
    """Absolute-pointer copy plan gathering VRAM spans into a staging buffer.

    Source pointers are the spans' VRAM addresses; destination pointers are
    staging_base_ptr + span.chunk_offset — the same staging offsets the
    per-span fallback loop uses (the chunk's spans are contiguous in staging
    in span order). Zero-size spans move no bytes and
    are dropped. No sort or coalesce pass is needed: spans are already
    maximal contiguous runs, so no two spans are adjacent in both source and
    staging.
    """
    src_ptrs = descs.ptrs[spans.starts]
    device_ids = descs.device_ids[spans.starts]
    sizes = spans.sizes
    dst_ptrs = staging_base_ptr + spans.chunk_offsets
    nonzero = sizes > 0
    if not bool(nonzero.all()):
        src_ptrs = src_ptrs[nonzero]
        dst_ptrs = dst_ptrs[nonzero]
        sizes = sizes[nonzero]
        device_ids = device_ids[nonzero]
    return _DestinationScatterPlan(
        src_ptrs=src_ptrs,
        dst_ptrs=dst_ptrs,
        sizes=sizes,
        device_ids=device_ids,
        total_bytes=int(sizes.sum()),
    )
