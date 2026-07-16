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

import math
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch

from tensorrt_llm import logger
from tensorrt_llm._torch.disaggregation.b10.kernels import (
    _SCATTER_KERNEL_BLOCK_SIZE,
    _new_scatter_metadata,
)
from tensorrt_llm._torch.disaggregation.b10.memory import _BufferView, _SpanArrays
from tensorrt_llm._torch.disaggregation.b10.planning import _scatter_program_count_for_spans
from tensorrt_llm._utils import prefer_pinned

_DEFAULT_STAGING_POOL_NUM_BUFFERS = 32
_DEFAULT_STAGING_POOL_BUFFER_SIZE = 512 * 1024 * 1024
_DEFAULT_RECV_SCRATCH_MIN_SPANS = 8
# Below this span count the send-side per-span copy loop is cheap enough;
# at or above it one gather kernel launch replaces the per-span dispatches.
_DEFAULT_SEND_GATHER_MIN_SPANS = _DEFAULT_RECV_SCRATCH_MIN_SPANS
_DEFAULT_RECV_SCRATCH_METADATA_BYTES_PER_SPAN = 1024 * 1024


def _default_recv_scratch_metadata_max_spans(buffer_size: int) -> int:
    return max(
        _DEFAULT_RECV_SCRATCH_MIN_SPANS,
        math.ceil(buffer_size / _DEFAULT_RECV_SCRATCH_METADATA_BYTES_PER_SPAN),
    )


class _NoStagingBufferAvailableError(RuntimeError):
    pass


@dataclass
class _QuarantinedBufferViews:
    """Staging views parked after a failed transfer until reuse is safe."""

    views: list[_BufferView]
    quarantined_at: float


class _CombinedEvents:
    def __init__(self, events: list[Any]):
        self._events = events

    def query(self) -> bool:
        return all(event.query() for event in self._events)

    def synchronize(self) -> None:
        for event in self._events:
            event.synchronize()


class _CudaCopyStreamPool:
    """One dedicated CUDA copy stream per device, created lazily.

    Staging<->VRAM copies run on these side streams so transfers never
    serialize against the model's compute streams.
    """

    def __init__(self):
        self._streams: dict[tuple[str, int], Any] = {}
        self._lock = threading.Lock()

    def stream_for(self, device: torch.device) -> Any:
        key = _device_key(device)
        with self._lock:
            stream = self._streams.get(key)
            if stream is None:
                stream = torch.cuda.Stream(device=device)
                self._streams[key] = stream
            return stream


class _CudaScratchBufferPool:
    """Per-device pool of fixed-size VRAM scratch buffers for recv scatter.

    Received chunks land here before the scatter kernel fans them out to the
    destination KV spans; each buffer also carries reusable scatter-metadata
    tensors (sized by `metadata_max_spans`) so kernel launches skip per-call
    allocation. Buffers are recycled via ready-events (`_pending`) rather
    than synchronously, so checkout never blocks on an in-flight scatter.
    """

    def __init__(
        self, num_buffers: int, buffer_size: int, metadata_max_spans: Optional[int] = None
    ):
        if num_buffers < 0:
            raise ValueError("num_buffers must be non-negative")
        if buffer_size < 0:
            raise ValueError("buffer_size must be non-negative")
        if metadata_max_spans is not None and metadata_max_spans <= 0:
            raise ValueError("metadata_max_spans must be positive")
        self._num_buffers = num_buffers
        self._buffer_size = buffer_size
        self._metadata_max_spans = (
            _default_recv_scratch_metadata_max_spans(buffer_size)
            if metadata_max_spans is None
            else metadata_max_spans
        )
        self._metadata_max_programs = self._metadata_max_spans + math.ceil(
            buffer_size / _SCATTER_KERNEL_BLOCK_SIZE
        )
        self._available: dict[tuple[str, int], list[torch.Tensor]] = {}
        self._pending: dict[tuple[str, int], list[tuple[torch.Tensor, Any]]] = {}
        self._checked_out: dict[tuple[str, int], int] = {}
        self._allocated: dict[tuple[str, int], int] = {}
        self._metadata: dict[int, dict[str, Any]] = {}
        self._lock = threading.Lock()

    @property
    def buffer_size(self) -> int:
        return self._buffer_size

    @property
    def num_buffers(self) -> int:
        return self._num_buffers

    @property
    def metadata_max_spans(self) -> int:
        return self._metadata_max_spans

    @property
    def metadata_max_programs(self) -> int:
        return self._metadata_max_programs

    def metadata_fits(self, spans: "_SpanArrays") -> bool:
        return (
            self._num_buffers > 0
            and len(spans) <= self._metadata_max_spans
            and _scatter_program_count_for_spans(spans, _SCATTER_KERNEL_BLOCK_SIZE)
            <= self._metadata_max_programs
        )

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            for key in list(self._allocated):
                self._collect_ready_locked(key)
            snapshot = self._snapshot_locked()
        return snapshot

    def preallocate(self, device: torch.device) -> None:
        key = _device_key(device)
        with self._lock:
            self._collect_ready_locked(key)
            self._pending.setdefault(key, [])
        self._preallocate_sync(device)

    def acquire(self, size: int, device: torch.device) -> _BufferView:
        if size < 0:
            raise ValueError("size must be non-negative")
        if size > self._buffer_size:
            raise RuntimeError(
                f"B10 recv scratch chunk size {size} exceeds B10 recv scratch "
                f"buffer size {self._buffer_size}; increase "
                "TRTLLM_B10_UCXX_STAGING_POOL_BUFFER_SIZE_BYTES"
            )
        key = _device_key(device)
        with self._lock:
            self._collect_ready_locked(key)
            available = self._available.setdefault(key, [])
            if not available:
                raise _NoStagingBufferAvailableError(
                    "No B10 recv GPU scratch buffers available; increase "
                    "TRTLLM_B10_UCXX_RECV_SCRATCH_POOL_NUM_BUFFERS"
                )
            owner = available.pop()
            self._checked_out[key] = self._checked_out.get(key, 0) + 1
            return _BufferView(
                owner[:size],
                owner,
                self,
                metadata=self._metadata_for_owner_locked(owner),
            )

    def release(self, views: list[_BufferView]) -> None:
        with self._lock:
            for view in views:
                if view.pool is not self:
                    continue
                key = _device_key(view.owner.device)
                self._checked_out[key] = max(0, self._checked_out.get(key, 0) - 1)
                if view.ready_event is None:
                    self._available.setdefault(key, []).append(view.owner)
                    continue
                self._pending.setdefault(key, []).append((view.owner, view.ready_event))

    def _preallocate_sync(self, device: torch.device) -> None:
        key = _device_key(device)
        while True:
            with self._lock:
                if self._allocated_count_locked() >= self._num_buffers:
                    return
            owner = self._allocate(self._buffer_size, device)
            with self._lock:
                if self._allocated_count_locked() < self._num_buffers:
                    self._available.setdefault(key, []).append(owner)
                    self._allocated[key] = self._allocated.get(key, 0) + 1
                    if isinstance(owner, torch.Tensor) and owner.is_cuda:
                        self._metadata[id(owner)] = _new_scatter_metadata(
                            self._metadata_max_spans,
                            self._metadata_max_programs,
                            device,
                        )

    def _collect_ready_locked(self, key: tuple[str, int]) -> None:
        pending: list[tuple[torch.Tensor, Any]] = []
        for owner, event in self._pending.get(key, []):
            try:
                ready = event.query()
            except Exception as exc:
                logger.warning(
                    f"B10 recv scratch buffer pending event query failed; "
                    f"keeping buffer pending: "
                    f"error={type(exc).__name__}: {exc}"
                )
                pending.append((owner, event))
                continue
            if ready:
                self._available.setdefault(key, []).append(owner)
            else:
                pending.append((owner, event))
        self._pending[key] = pending

    def _snapshot_locked(self) -> dict[str, int]:
        return {
            "num_buffers": self._num_buffers,
            "buffer_size": self._buffer_size,
            "available": sum(len(items) for items in self._available.values()),
            "pending": sum(len(items) for items in self._pending.values()),
            "checked_out": sum(self._checked_out.values()),
            "allocated": sum(self._allocated.values()),
            "metadata_max_spans": self._metadata_max_spans,
            "metadata_max_programs": self._metadata_max_programs,
        }

    def _allocated_count_locked(self) -> int:
        return sum(self._allocated.values())

    def _metadata_for_owner_locked(self, owner: torch.Tensor) -> dict[str, Any]:
        return self._metadata.setdefault(id(owner), {})

    @staticmethod
    def _allocate(size: int, device: torch.device) -> torch.Tensor:
        return torch.empty((size,), dtype=torch.uint8, device=device)


class _PinnedStagingBufferPool:
    """Fixed pool of pinned-host staging buffers — the heart of host staging.

    Every transfer stages through these DRAM buffers (bounding failure blast
    radius to host memory), so the pool size caps in-flight transfer bytes.
    Like the scratch pool, buffers recycle via ready-events; views handed to
    a failed transfer are quarantined until the peer can no longer write
    into them, then returned. See DESIGN.md "Buffer ownership" and
    "Fault tolerance model".
    """

    def __init__(
        self,
        num_buffers: int = _DEFAULT_STAGING_POOL_NUM_BUFFERS,
        buffer_size: int = _DEFAULT_STAGING_POOL_BUFFER_SIZE,
    ):
        if num_buffers <= 0:
            raise ValueError("num_buffers must be positive")
        if buffer_size < 0:
            raise ValueError("buffer_size must be non-negative")
        self._num_buffers = num_buffers
        self._buffer_size = buffer_size
        self._available: list[torch.Tensor] = []
        self._pending: list[tuple[torch.Tensor, Any]] = []
        self._checked_out = 0
        self._refill_in_progress = False
        self._lock = threading.Lock()
        self._refill_sync()

    @property
    def buffer_size(self) -> int:
        return self._buffer_size

    @property
    def num_buffers(self) -> int:
        return self._num_buffers

    def snapshot(self) -> dict[str, int | bool]:
        with self._lock:
            self._collect_ready_locked()
            return self._snapshot_locked()

    def acquire(self, size: int) -> _BufferView:
        if size < 0:
            raise ValueError("size must be non-negative")
        if size > self._buffer_size:
            raise RuntimeError(
                f"B10 staging chunk size {size} exceeds B10 staging buffer size "
                f"{self._buffer_size}; increase "
                "TRTLLM_B10_UCXX_STAGING_POOL_BUFFER_SIZE_BYTES"
            )
        with self._lock:
            self._collect_ready_locked()
            if not self._available:
                raise _NoStagingBufferAvailableError(
                    "No B10 pinned staging buffers available; increase "
                    "TRTLLM_B10_UCXX_STAGING_POOL_NUM_BUFFERS"
                )
            owner = self._available.pop()
            self._checked_out += 1
            return _BufferView(owner[:size], owner, self)

    def release(self, views: list[_BufferView]) -> None:
        with self._lock:
            for view in views:
                if view.pool is not self:
                    continue
                self._checked_out = max(0, self._checked_out - 1)
                if view.ready_event is not None and not view.ready_event.query():
                    self._pending.append((view.owner, view.ready_event))
                else:
                    self._available.append(view.owner)

    def quarantine(self, views: list[_BufferView]) -> None:
        replaced = False
        with self._lock:
            for view in views:
                if view.pool is not self:
                    continue
                self._checked_out = max(0, self._checked_out - 1)
                replaced = True
        if replaced:
            self._start_background_refill()

    def _start_background_refill(self) -> None:
        with self._lock:
            if self._refill_in_progress:
                return
            self._refill_in_progress = True
            missing = (
                self._num_buffers - len(self._available) - len(self._pending) - self._checked_out
            )
            state = self._snapshot_locked()
        logger.info(
            f"B10 staging pool refill started: missing={missing} "
            f"{_format_staging_pool_snapshot(state)}"
        )
        threading.Thread(
            target=self._refill_background, name="b10-staging-pool-refill", daemon=True
        ).start()

    def _refill_sync(self) -> None:
        while len(self._available) + self._checked_out < self._num_buffers:
            self._available.append(self._allocate(self._buffer_size))

    def _refill_background(self) -> None:
        try:
            while True:
                complete_state: Optional[dict[str, int | bool]] = None
                with self._lock:
                    self._collect_ready_locked()
                    missing = (
                        self._num_buffers
                        - len(self._available)
                        - len(self._pending)
                        - self._checked_out
                    )
                    if missing <= 0:
                        self._refill_in_progress = False
                        complete_state = self._snapshot_locked()
                if complete_state is not None:
                    logger.info(
                        f"B10 staging pool refill complete: "
                        f"{_format_staging_pool_snapshot(complete_state)}"
                    )
                    return
                owner = self._allocate(self._buffer_size)
                with self._lock:
                    current = len(self._available) + len(self._pending) + self._checked_out
                    if current < self._num_buffers:
                        self._available.append(owner)
        except Exception as exc:
            with self._lock:
                self._refill_in_progress = False
                state = self._snapshot_locked()
            logger.warning(
                f"B10 staging pool refill failed: "
                f"error={type(exc).__name__}: {exc} "
                f"{_format_staging_pool_snapshot(state)}"
            )
            raise

    def _collect_ready_locked(self) -> None:
        pending: list[tuple[torch.Tensor, Any]] = []
        for owner, event in self._pending:
            try:
                ready = event.query()
            except Exception as exc:
                logger.warning(
                    f"B10 staging pool pending event query failed; "
                    f"keeping buffer pending: "
                    f"error={type(exc).__name__}: {exc}"
                )
                pending.append((owner, event))
                continue
            if ready:
                self._available.append(owner)
            else:
                pending.append((owner, event))
        self._pending = pending

    def _snapshot_locked(self) -> dict[str, int | bool]:
        return {
            "num_buffers": self._num_buffers,
            "buffer_size": self._buffer_size,
            "available": len(self._available),
            "pending": len(self._pending),
            "checked_out": self._checked_out,
            "refill_in_progress": self._refill_in_progress,
        }

    @staticmethod
    def _allocate(size: int) -> torch.Tensor:
        return torch.empty((size,), dtype=torch.uint8, device="cpu", pin_memory=prefer_pinned())


def _ready_event_for_copy_events(events: list[Any]) -> Optional[Any]:
    if not events:
        return None
    if len(events) == 1:
        return events[0]
    return _CombinedEvents(events)


def _format_staging_pool_snapshot(state: dict[str, int | bool]) -> str:
    return (
        f"staging_pool_buffers={state['num_buffers']} "
        f"staging_pool_buffer_size={state['buffer_size']} "
        f"staging_pool_available={state['available']} "
        f"staging_pool_pending={state['pending']} "
        f"staging_pool_checked_out={state['checked_out']} "
        f"staging_pool_refill_in_progress={state['refill_in_progress']}"
    )


def _format_staging_pool_state(pool: _PinnedStagingBufferPool) -> str:
    return _format_staging_pool_snapshot(pool.snapshot())


def _format_cuda_scratch_pool_snapshot(state: dict[str, int]) -> str:
    return (
        f"recv_scratch_pool_buffers={state['num_buffers']} "
        f"recv_scratch_pool_buffer_size={state['buffer_size']} "
        f"recv_scratch_pool_available={state['available']} "
        f"recv_scratch_pool_pending={state['pending']} "
        f"recv_scratch_pool_checked_out={state['checked_out']} "
        f"recv_scratch_pool_allocated={state['allocated']} "
        f"recv_scratch_metadata_max_spans={state['metadata_max_spans']} "
        f"recv_scratch_metadata_max_programs={state['metadata_max_programs']}"
    )


def _format_cuda_scratch_pool_state(pool: _CudaScratchBufferPool) -> str:
    return _format_cuda_scratch_pool_snapshot(pool.snapshot())


def _device_key(device: torch.device) -> tuple[str, int]:
    index = device.index
    if index is None and device.type == "cuda":
        index = torch.cuda.current_device()
    return device.type, int(index or 0)


def _record_cuda_copy_events(
    devices: list[torch.device], stream_getter: Optional[Callable[[torch.device], Any]] = None
) -> list[Any]:
    events: list[Any] = []
    seen_devices = set()
    for device in devices:
        key = _device_key(device)
        if key in seen_devices:
            continue
        seen_devices.add(key)
        event = torch.cuda.Event()
        stream = (
            torch.cuda.current_stream(device) if stream_getter is None else stream_getter(device)
        )
        event.record(stream)
        events.append(event)
    return events
