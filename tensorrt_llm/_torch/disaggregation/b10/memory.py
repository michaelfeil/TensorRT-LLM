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

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

# --------------------------------------------------------------------------
# Descriptor / span / fragment contract: arrays end-to-end.
#
# Bulk per-descriptor, per-span, and per-fragment metadata travels as
# parallel int64 numpy arrays from the ingestion boundary (MemoryDescs in
# _normalize_memory_descs, wire bytes in recv._dst_descs_from_control) all
# the way to the kernel launch. Five sequential optimizations each re-fixed
# the same disease — arrays exploded into Python objects that every
# downstream consumer re-vectorized with fromiter — so object
# materialization (_NormalizedMemoryDesc, _ContiguousSpan) is now allowed
# ONLY through __getitem__ on the two array-backed views (_DescArrayView,
# _SpanArrays), and only on fallback/cold paths doing O(small) random
# access; _DestinationScatterPlan is arrays-only, with no per-fragment
# object form at all. Every hot-path addition must consume and return
# arrays or array-backed containers, and must be locked against the
# object implementation with a differential harness before landing.
# See DESIGN.md "Array-native hot path".
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _NormalizedMemoryDesc:
    ptr: int
    size: int
    device_id: int


class _DescArrayView:
    """Column-oriented, array-backed view over normalized memory descriptors.

    The universal descriptor container of the array-native contract above:
    every production path hands descriptors around as one of these. Six
    producers construct it — `_normalize_memory_descs` (tuple and attribute
    branches), `_desc_view_from_arrays` (the producer-supplied side-channel),
    `_reorder_desc_pairs_for_contiguity`, recv's
    `_dst_descs_from_control` (the packed wire decode), and the
    kernel warmups. ptrs, sizes, and device_ids are equal-length 1-D int64
    arrays (possibly read-only np.frombuffer views over wire bytes, so they
    must never be mutated in place). Hot consumers take the `_desc_arrays`
    fast path and stay loop-free; `__getitem__` materializes a
    `_NormalizedMemoryDesc` on demand for O(spans) random access.
    """

    __slots__ = ("ptrs", "sizes", "device_ids")

    def __init__(self, ptrs: np.ndarray, sizes: np.ndarray, device_ids: np.ndarray):
        # Shape/dtype validation is skipped as structural: every producer
        # (see the class docstring and the contract block above) constructs
        # the three equal-length int64 columns together.
        self.ptrs = ptrs
        self.sizes = sizes
        self.device_ids = device_ids

    def __len__(self) -> int:
        return int(self.ptrs.shape[0])

    def __getitem__(self, index: int) -> _NormalizedMemoryDesc:
        return _NormalizedMemoryDesc(
            ptr=int(self.ptrs[index]),
            size=int(self.sizes[index]),
            device_id=int(self.device_ids[index]),
        )


@dataclass(frozen=True)
class _TransferChunk:
    start: int
    count: int
    size: int


@dataclass(frozen=True)
class _ContiguousSpan:
    start: int
    size: int
    chunk_offset: int


class _SpanArrays:
    """Column-oriented, array-backed view over a chunk's contiguous spans.

    Sole hot-path producer is `_contiguous_desc_spans` (the kernel warmups
    construct one directly): starts (span-start descriptor indices), sizes,
    and chunk_offsets are equal-length 1-D int64 arrays. Hot consumers use the columns directly and stay
    loop-free; `__getitem__` materializes a `_ContiguousSpan` on demand
    for O(spans) random access on fallback paths.
    """

    __slots__ = ("starts", "sizes", "chunk_offsets")

    def __init__(self, starts: np.ndarray, sizes: np.ndarray, chunk_offsets: np.ndarray):
        self.starts = starts
        self.sizes = sizes
        self.chunk_offsets = chunk_offsets

    def __len__(self) -> int:
        return int(self.starts.shape[0])

    def __getitem__(self, index: int) -> _ContiguousSpan:
        return _ContiguousSpan(
            start=int(self.starts[index]),
            size=int(self.sizes[index]),
            chunk_offset=int(self.chunk_offsets[index]),
        )


@dataclass(frozen=True)
class _DescPairOrder:
    src_descs: "_DescArrayView"
    dst_descs: "_DescArrayView"
    strategy: str
    transfer_chunks: list["_TransferChunk"]
    src_span_count: int
    dst_span_count: int

    @property
    def wire_chunk_count(self) -> int:
        return len(self.transfer_chunks)


@dataclass(frozen=True)
class _DestinationScatterPlan:
    """Destination-ordered scatter fragments as parallel int64 arrays.

    The hot path (thousands of fragments per request-level scatter) stores
    fragments as numpy arrays so metadata assembly never walks per-fragment
    Python objects.
    """

    src_ptrs: np.ndarray
    dst_ptrs: np.ndarray
    sizes: np.ndarray
    device_ids: np.ndarray
    total_bytes: int

    @property
    def fragment_count(self) -> int:
        return int(self.src_ptrs.shape[0])


@dataclass
class _BufferView:
    buffer: Any
    owner: Any
    pool: Optional[Any] = None
    ready_event: Optional[Any] = None
    metadata: Optional[dict[str, Any]] = None
    # Set by the owning pool the first time this view goes back, so a second
    # return is refused instead of putting one buffer into circulation twice.
    # A view reaches a pool at most once by construction; a repeat means some
    # path released it and then a failure path released or quarantined it
    # again, which would hand the same memory to two transfers at once.
    returned: bool = False
