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
"""Receive side of the B10 UCXX transfer agent: incoming WRITEs.

`RecvPipeline` is the recv collaborator constructed by
`B10CacheTransferAgent.__init__`; the UCXX listener callback
(`_on_endpoint`) runs here.

Constructor-injected:

- ``core`` (`_AgentCore`: loop, tag registry, staging pool and its
  acquire/release helpers, recv scratch pool, copy streams, shutdown flag,
  config-derived knobs)
- ``trace`` (`TransferTrace`: transfer traces and recv timing logs)
- ``staging_quarantine_ttl_s`` (quarantine TTL from B10AgentConfig)
- ``run_limited`` / ``reserve_message_tags`` /
  ``request_id_from_sync_message`` (agent-shell helpers; injected
  references)
- ``send_gather_device_for_chunk`` (send-pipeline gather gate for the
  shared staging<->descs copy helper; injected as a callable so the two
  pipelines never hold each other)

Own state:

- ``_active_recv_request_counts`` guarded by
  ``_active_recv_request_counts_lock``
- ``_active_recv_copy_request_counts`` and
  ``_cancelled_recv_request_expiries`` guarded by
  ``_cancelled_recv_request_ids_lock`` (TTL in
  ``_cancelled_recv_request_ttl_s``)
- ``_incoming_write_listener``
- ``_recv_scratch_buffer_slots`` / ``_recv_scratch_device`` /
  ``_retired_recv_scratch_views``
- ``_request_level_recv_scatter_admissions`` guarded by
  ``_request_level_recv_scatter_reserve_lock``
- ``_quarantined_buffer_views`` (TTL in ``_staging_quarantine_ttl_s``)

Per-transfer state for one incoming WRITE lives on ``_RecvTransfer``, a
plain attribute bag built by ``_validate_and_prepare`` and threaded
through the ``_handle_incoming_write`` phase methods (see the phase-flow
comment on the orchestrator).
"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import TimeoutError
from typing import Any, Callable, Optional

import numpy as np
import torch

from tensorrt_llm import logger
from tensorrt_llm._torch.disaggregation.b10 import memory as b10_memory
from tensorrt_llm._torch.disaggregation.b10 import protocol as b10_protocol
from tensorrt_llm._torch.disaggregation.b10.async_utils import (
    _abort_endpoint_background,
    _await_with_timeout,
    _gather_cancel_on_failure,
    _TransferDeadline,
)
from tensorrt_llm._torch.disaggregation.b10.core import _AgentCore
from tensorrt_llm._torch.disaggregation.b10.kernels import (
    _copy_buffer,
    _cuda_copy_device,
    _gather_vram_spans_to_pinned_staging,
    _scatter_cuda_buffer_to_vram_spans,
    _scatter_cuda_buffers_to_vram_destination_order,
    _span_view,
)
from tensorrt_llm._torch.disaggregation.b10.memory import (
    _BufferView,
    _DescArrayView,
    _TransferChunk,
)
from tensorrt_llm._torch.disaggregation.b10.planning import (
    _contiguous_desc_spans,
    _desc_arrays,
    _destination_scatter_plan_for_chunks,
    _has_overlapping_descs,
    _memory_desc_stats,
    _transfer_chunk_stats,
    _transfer_chunks_from_control,
)
from tensorrt_llm._torch.disaggregation.b10.pools import (
    _DEFAULT_RECV_SCRATCH_MIN_SPANS,
    _device_key,
    _format_cuda_scratch_pool_state,
    _format_staging_pool_state,
    _NoStagingBufferAvailableError,
    _QuarantinedBufferViews,
    _ready_event_for_copy_events,
    _record_cuda_copy_events,
)
from tensorrt_llm._torch.disaggregation.b10.protocol import (
    _B10_PROTOCOL,
    _B10_PROTOCOL_VERSION,
    _BOOTSTRAP_CONTROL_TAG,
    _DEFAULT_TAG_QUARANTINE_TTL_S,
    _data_tag,
    _ready_tag,
    _recv_obj,
    _result_tag,
    _send_reply,
)
from tensorrt_llm._torch.disaggregation.b10.state import (
    _ReceivedTransferChunk,
    _RequestScatterChunk,
    _SourceReadyEvents,
    _StagingCheckoutTracker,
)
from tensorrt_llm._torch.disaggregation.b10.timings import TransferTrace

_REQUEST_LEVEL_RECV_SCATTER_MIN_SPANS_PER_CHUNK = _DEFAULT_RECV_SCRATCH_MIN_SPANS


def _dst_descs_from_control(
    control: dict[str, Any],
) -> _DescArrayView:
    """Decode destination descriptors from a B10 control message.

    `dst_descs_packed` is the only supported encoding: a flat little-endian
    int64 (3, n) array of (ptrs, sizes, device_ids); decoding it is O(1)
    Python-object work. np.frombuffer arrays are read-only, so downstream
    consumers must never mutate them in place (lexsort/cumsum/comparisons
    only read). Controls without the key (senders predating the
    `_FEATURE_PACKED_DESCS` builds) are rejected loudly.
    """
    packed = control.get("dst_descs_packed")
    if packed is None:
        raise ValueError(
            "B10 control message uses the legacy descriptor encoding; this "
            "build requires packed_descs — upgrade the sender"
        )
    flat = np.frombuffer(packed, dtype="<i8")
    if flat.size % 3 != 0:
        raise ValueError(
            f"B10 packed destination descriptors have {flat.size} int64 "
            f"words, expected a multiple of 3"
        )
    arr = flat.reshape(3, -1)
    return _DescArrayView(arr[0], arr[1], arr[2])


class _RecvTransfer:
    """Per-transfer state for one incoming WRITE.

    Plain attribute bag shared by the ``RecvPipeline._handle_incoming_write``
    phase methods; ``_validate_and_prepare`` assigns every field except
    ``chunk_dst_spans``, which is planned later in
    ``_reserve_transfer_resources``. These previously lived as
    closure-captured locals of the monolithic handler.

    ``__slots__`` (kept in sync with the annotations below) turns a typo in
    a phase method's attribute assignment into an immediate AttributeError
    instead of a silently ignored stray attribute.
    """

    __slots__ = (
        "endpoint",
        "recv_total_start_s",
        "recv_timings",
        "recv_span_count",
        "recv_status",
        "recv_in_flight",
        "recv_copy_in_flight",
        "deadline",
        "transfer_id",
        "request_id",
        "endpoint_generation",
        "tag_domain",
        "dst_type",
        "dst_descs",
        "dst_descs_have_overlap",
        "transfer_chunks",
        "desc_count",
        "total_bytes",
        "max_desc_size",
        "wire_chunk_count",
        "max_wire_chunk_size",
        "pool_buffers",
        "dst_view_lifetime_refs",
        "staging_tracker",
        "recv_scratch_tracker",
        "copy_events",
        "request_scatter_chunks",
        "recv_copy_request_marked",
        "recv_request_marked",
        "received_chunks",
        "use_request_level_scatter",
        "request_level_scatter_chunk_indices",
        "request_scatter_scratch_views",
        "request_scatter_admission_marked",
        "ready_tag",
        "result_tag",
        "tag_owner",
        "chunk_dst_spans",
    )

    endpoint: Any
    recv_total_start_s: float
    recv_timings: dict[str, float]
    recv_span_count: int
    recv_status: str
    recv_in_flight: int
    recv_copy_in_flight: int
    deadline: _TransferDeadline
    transfer_id: int
    request_id: Optional[int]
    endpoint_generation: int
    tag_domain: int
    dst_type: str
    dst_descs: _DescArrayView
    dst_descs_have_overlap: bool
    transfer_chunks: list[_TransferChunk]
    desc_count: int
    total_bytes: int
    max_desc_size: int
    wire_chunk_count: int
    max_wire_chunk_size: int
    pool_buffers: int
    dst_view_lifetime_refs: list[_BufferView]
    staging_tracker: _StagingCheckoutTracker
    recv_scratch_tracker: _StagingCheckoutTracker
    copy_events: list[Any]
    request_scatter_chunks: list[_RequestScatterChunk]
    recv_copy_request_marked: bool
    recv_request_marked: bool
    received_chunks: asyncio.Queue[Optional[_ReceivedTransferChunk]]
    use_request_level_scatter: bool
    request_level_scatter_chunk_indices: set[int]
    request_scatter_scratch_views: dict[int, _BufferView]
    request_scatter_admission_marked: bool
    ready_tag: int
    result_tag: int
    tag_owner: tuple[str, int, int]
    chunk_dst_spans: list[b10_memory._SpanArrays]


class RecvPipeline:
    def __init__(
        self,
        core: _AgentCore,
        trace: TransferTrace,
        *,
        staging_quarantine_ttl_s: float,
        run_limited: Callable[..., Any],
        reserve_message_tags: Callable[..., None],
        request_id_from_sync_message: Callable[[Optional[str]], Optional[int]],
        send_gather_device_for_chunk: Callable[..., Optional[torch.device]],
    ):
        self._core = core
        self._trace = trace
        self._staging_quarantine_ttl_s = staging_quarantine_ttl_s
        self._run_limited = run_limited
        self._reserve_message_tags = reserve_message_tags
        self._request_id_from_sync_message = request_id_from_sync_message
        self._send_gather_device_for_chunk = send_gather_device_for_chunk
        self._incoming_write_listener: Optional[Callable[[int, bool], None]] = None
        self._quarantined_buffer_views: list[_QuarantinedBufferViews] = []
        self._recv_scratch_buffer_slots = asyncio.BoundedSemaphore(
            core.recv_scratch_buffer_pool.num_buffers
        )
        self._recv_scratch_device: Optional[torch.device] = None
        self._retired_recv_scratch_views: list[_BufferView] = []
        self._request_level_recv_scatter_reserve_lock = asyncio.Lock()
        self._request_level_recv_scatter_admissions = 0
        self._active_recv_request_counts: dict[int, int] = {}
        self._active_recv_request_counts_lock = threading.Lock()
        self._active_recv_copy_request_counts: dict[int, int] = {}
        # rid -> monotonic expiry. Fences late incoming writes for cancelled
        # requests; entries expire once the sender's transfer can no longer
        # arrive so recycled request ids are not rejected forever.
        self._cancelled_recv_request_expiries: dict[int, float] = {}
        cancelled_recv_ttl_s = _DEFAULT_TAG_QUARANTINE_TTL_S
        if core.transfer_timeout_s is not None:
            cancelled_recv_ttl_s = max(cancelled_recv_ttl_s, 2.0 * core.transfer_timeout_s)
        self._cancelled_recv_request_ttl_s = cancelled_recv_ttl_s
        self._cancelled_recv_request_ids_lock = threading.Lock()
        self._preallocate_recv_scratch_buffers()

    def has_active_recv_request(self, request_id: int) -> bool:
        with self._active_recv_request_counts_lock:
            return self._active_recv_request_counts.get(int(request_id), 0) > 0

    def has_active_recv_copy_request(self, request_id: int) -> bool:
        with self._cancelled_recv_request_ids_lock:
            return self._active_recv_copy_request_counts.get(int(request_id), 0) > 0

    def cancel_recv_request(self, request_id: int) -> None:
        now = time.monotonic()
        with self._cancelled_recv_request_ids_lock:
            expired = [
                rid
                for rid, expiry in self._cancelled_recv_request_expiries.items()
                if expiry <= now
            ]
            for rid in expired:
                del self._cancelled_recv_request_expiries[rid]
            self._cancelled_recv_request_expiries[int(request_id)] = (
                now + self._cancelled_recv_request_ttl_s
            )

    def _is_recv_request_cancelled_locked(self, request_id: int) -> bool:
        expiry = self._cancelled_recv_request_expiries.get(int(request_id))
        if expiry is None:
            return False
        if expiry <= time.monotonic():
            del self._cancelled_recv_request_expiries[int(request_id)]
            return False
        return True

    def _is_recv_request_cancelled(self, request_id: int) -> bool:
        with self._cancelled_recv_request_ids_lock:
            return self._is_recv_request_cancelled_locked(request_id)

    def _raise_if_recv_request_cancelled(self, request_id: Optional[int]) -> None:
        if request_id is None:
            return
        if self._is_recv_request_cancelled(request_id):
            raise RuntimeError(f"B10 recv request {request_id} was cancelled")

    def _mark_active_recv_request(self, request_id: int) -> None:
        with self._active_recv_request_counts_lock:
            request_id = int(request_id)
            self._active_recv_request_counts[request_id] = (
                self._active_recv_request_counts.get(request_id, 0) + 1
            )

    def _clear_active_recv_request(self, request_id: int) -> None:
        with self._active_recv_request_counts_lock:
            request_id = int(request_id)
            count = self._active_recv_request_counts.get(request_id, 0)
            if count <= 1:
                self._active_recv_request_counts.pop(request_id, None)
            else:
                self._active_recv_request_counts[request_id] = count - 1

    def _mark_active_recv_copy_request(self, request_id: int) -> None:
        with self._cancelled_recv_request_ids_lock:
            request_id = int(request_id)
            if self._is_recv_request_cancelled_locked(request_id):
                raise RuntimeError(f"B10 recv request {request_id} was cancelled")
            self._active_recv_copy_request_counts[request_id] = (
                self._active_recv_copy_request_counts.get(request_id, 0) + 1
            )

    def _clear_active_recv_copy_request(self, request_id: int) -> None:
        with self._cancelled_recv_request_ids_lock:
            request_id = int(request_id)
            count = self._active_recv_copy_request_counts.get(request_id, 0)
            if count <= 1:
                self._active_recv_copy_request_counts.pop(request_id, None)
            else:
                self._active_recv_copy_request_counts[request_id] = count - 1

    def set_incoming_write_listener(self, listener: Callable[[int, bool], None]) -> None:
        """Register listener(request_id, success), invoked when an incoming
        WRITE reaches a terminal state on this (receiver) agent. Lets the
        session layer complete receive tasks locally instead of depending on
        the sender's KV_AGENT_RESULT notification. Runs on the agent's event
        loop thread; keep it brief and non-blocking."""
        self._incoming_write_listener = listener

    def _notify_incoming_write_listener(self, request_id: Optional[int], recv_status: str) -> None:
        listener = self._incoming_write_listener
        if listener is None or request_id is None:
            return
        if recv_status == "success":
            success = True
        elif recv_status == "failed":
            success = False
        else:
            # Cancelled receives are receiver-initiated; the session layer is
            # already resolving them through the cancel path.
            return
        try:
            listener(request_id, success)
        except Exception as exc:
            logger.warning(
                f"B10 incoming-write listener failed for request "
                f"{request_id}: {type(exc).__name__}: {exc}"
            )

    async def _recv_buffer(
        self, endpoint: Any, buffer: Any, tag: int, deadline: _TransferDeadline
    ) -> None:
        if isinstance(buffer, torch.Tensor):
            if buffer.is_cuda and self._core.sync_cuda_before_transfer:
                torch.cuda.synchronize(buffer.device)
            elif not buffer.is_cuda:
                buffer = buffer.numpy()
        await _await_with_timeout(
            endpoint.recv(buffer, tag=tag), deadline.remaining_s(), cancel_on_timeout=False
        )

    def _preallocate_recv_scratch_buffers(self) -> None:
        if self._core.recv_scratch_buffer_pool.num_buffers == 0:
            return
        if not torch.cuda.is_available():
            return
        self._recv_scratch_device = torch.device("cuda", torch.cuda.current_device())
        self._core.recv_scratch_buffer_pool.preallocate(self._recv_scratch_device)

    async def _acquire_recv_scratch_buffer(
        self, size: int, device: torch.device, deadline: _TransferDeadline
    ) -> _BufferView:
        while True:
            await _await_with_timeout(
                self._recv_scratch_buffer_slots.acquire(), deadline.remaining_s()
            )
            try:
                return self._core.recv_scratch_buffer_pool.acquire(size, device)
            except _NoStagingBufferAvailableError:
                self._recv_scratch_buffer_slots.release()
                await _await_with_timeout(asyncio.sleep(0.001), deadline.remaining_s())
            except Exception:
                self._recv_scratch_buffer_slots.release()
                raise

    async def _release_recv_scratch_slot_after_event(self, event: Any) -> None:
        # Same contract as _release_staging_slot_after_event: the scratch pool
        # quarantines the buffer while its event is pending; the permit must
        # come back even when the event wait fails.
        try:
            await asyncio.to_thread(event.synchronize)
        except Exception as exc:
            logger.warning(
                f"B10 recv scratch slot event wait failed before slot release: "
                f"error={type(exc).__name__}: {exc}"
            )
        finally:
            self._recv_scratch_buffer_slots.release()

    def _release_recv_scratch_slots_for_views(self, views: list[_BufferView]) -> None:
        for view in views:
            if view.pool is not self._core.recv_scratch_buffer_pool:
                continue
            event = view.ready_event
            if event is not None and not self._core._event_ready_for_slot_release(
                event, "recv scratch"
            ):
                asyncio.create_task(self._release_recv_scratch_slot_after_event(event))
            else:
                self._recv_scratch_buffer_slots.release()

    def _release_recv_scratch_buffers(self, views: list[_BufferView]) -> None:
        self._core.recv_scratch_buffer_pool.release(views)
        self._release_recv_scratch_slots_for_views(views)

    @staticmethod
    def _single_cuda_device_for_spans(
        descs: _DescArrayView,
        spans: b10_memory._SpanArrays,
    ) -> Optional[torch.device]:
        if not spans:
            return None
        devices = descs.device_ids[spans.starts]
        if not bool((devices == devices[0]).all()):
            return None
        return torch.device("cuda", int(devices[0]))

    def _should_use_recv_scratch(
        self,
        descs: _DescArrayView,
        memory_type: str,
        spans: b10_memory._SpanArrays,
        *,
        descs_have_overlap: Optional[bool] = None,
    ) -> bool:
        scratch_device = self._recv_scratch_device
        scratch_pool = self._core.recv_scratch_buffer_pool
        span_device = RecvPipeline._single_cuda_device_for_spans(descs, spans)
        has_overlap = (
            _has_overlapping_descs(descs) if descs_have_overlap is None else descs_have_overlap
        )
        return (
            memory_type == "VRAM"
            and self._request_level_recv_scatter_admissions == 0
            and len(spans) >= _DEFAULT_RECV_SCRATCH_MIN_SPANS
            and scratch_pool is not None
            and scratch_pool.metadata_fits(spans)
            and len(self._retired_recv_scratch_views) < scratch_pool.num_buffers
            and scratch_device is not None
            and span_device == scratch_device
            and not has_overlap
        )

    def _request_level_recv_scatter_chunk_indices(
        self,
        descs: _DescArrayView,
        memory_type: str,
        transfer_chunks: list[_TransferChunk],
        chunk_spans: list[b10_memory._SpanArrays],
        *,
        descs_have_overlap: bool,
    ) -> set[int]:
        if memory_type != "VRAM" or descs_have_overlap:
            return set()
        if len(transfer_chunks) <= 1:
            return set()
        scratch_device = self._recv_scratch_device
        scratch_pool = self._core.recv_scratch_buffer_pool
        if scratch_device is None or scratch_pool is None:
            return set()
        # Vectorized single-device check; the previous per-desc set
        # comprehension was an O(descs) Python pass on the recv hot path.
        _, desc_sizes, desc_devices = _desc_arrays(descs)
        nonzero_devices = desc_devices[desc_sizes > 0]
        if nonzero_devices.shape[0] == 0:
            return set()
        device_id = int(nonzero_devices[0])
        if bool((nonzero_devices != device_id).any()):
            return set()
        if torch.device("cuda", device_id) != scratch_device:
            return set()

        candidate_indices = {
            idx
            for idx, chunk in enumerate(transfer_chunks)
            if (
                chunk.size <= scratch_pool.buffer_size
                and len(chunk_spans[idx]) >= _REQUEST_LEVEL_RECV_SCATTER_MIN_SPANS_PER_CHUNK
            )
        }
        if len(candidate_indices) <= 1:
            return set()

        usable_scratch_buffers = max(
            0, scratch_pool.num_buffers - len(self._retired_recv_scratch_views)
        )
        if len(candidate_indices) > usable_scratch_buffers:
            return set()
        return candidate_indices

    def _copy_stream_for_device(
        self,
        copy_device: torch.device,
        prepared_copy_streams: dict[tuple[str, int], Any],
        *,
        wait_current_stream: bool,
        source_ready_events: Optional[_SourceReadyEvents] = None,
        source_ready_waited_keys: Optional[set[tuple[str, int]]] = None,
    ) -> Any:
        key = _device_key(copy_device)
        copy_stream = prepared_copy_streams.get(key)
        if copy_stream is None:
            copy_stream = self._core.cuda_copy_streams.stream_for(copy_device)
            waited_on_source = (
                source_ready_waited_keys is not None and key in source_ready_waited_keys
            )
            if source_ready_events and not waited_on_source:
                for event_device, event in source_ready_events:
                    if _device_key(event_device) == key:
                        copy_stream.wait_event(event)
                        waited_on_source = True
                if waited_on_source and source_ready_waited_keys is not None:
                    source_ready_waited_keys.add(key)
            if source_ready_events is not None and not waited_on_source:
                # Explicit source-ready events are the only valid producer
                # boundary for B10 VRAM sends; sender-thread streams are unrelated.
                raise RuntimeError(f"B10 missing source-ready event for copy device {copy_device}")
            if not waited_on_source and wait_current_stream:
                current_stream = torch.cuda.current_stream(copy_device)
                if current_stream is not copy_stream:
                    copy_stream.wait_stream(current_stream)
            prepared_copy_streams[key] = copy_stream
        return copy_stream

    def _record_recv_scratch_ready_events(
        self,
        transfer_id: int,
        chunk_index: int,
        scratch_view: _BufferView,
    ) -> Optional[list[Any]]:
        try:
            scratch_events = _record_cuda_copy_events(
                [scratch_view.buffer.device], self._core.cuda_copy_streams.stream_for
            )
        except Exception as exc:
            logger.warning(
                f"B10 recv transfer {transfer_id} failed to record scratch "
                f"ready event after DATA chunk failure: "
                f"chunk_index={chunk_index} "
                f"error={type(exc).__name__}: {exc}; "
                f"leaving scratch buffer checked out"
            )
            self._retire_recv_scratch_buffer(scratch_view)
            return None
        scratch_view.ready_event = _ready_event_for_copy_events(scratch_events)
        return scratch_events

    def _retire_recv_scratch_buffer(self, scratch_view: _BufferView) -> None:
        self._retired_recv_scratch_views.append(scratch_view)
        if len(self._retired_recv_scratch_views) >= self._core.recv_scratch_buffer_pool.num_buffers:
            self._recv_scratch_device = None

    def _scatter_staging_to_vram_spans(
        self,
        staging_buffer: Any,
        descs: _DescArrayView,
        chunk: _TransferChunk,
        spans: b10_memory._SpanArrays,
        scratch_view: _BufferView,
        copy_stream: Any,
        lifetime_refs: Optional[list[_BufferView]],
    ) -> list[torch.device]:
        scratch_buffer = scratch_view.buffer[: chunk.size]
        _copy_buffer(staging_buffer[: chunk.size], scratch_buffer, copy_stream=copy_stream)
        _scatter_cuda_buffer_to_vram_spans(
            scratch_buffer, descs, spans, copy_stream, lifetime_refs, scratch_view.metadata
        )
        return [scratch_buffer.device]

    def _copy_chunk_between_staging_and_descs(
        self,
        descs: _DescArrayView,
        chunk: _TransferChunk,
        memory_type: str,
        staging_buffer: Any,
        *,
        copy_from_staging: bool,
        lifetime_refs: Optional[list[_BufferView]] = None,
        spans: Optional[b10_memory._SpanArrays] = None,
        scratch_view: Optional[_BufferView] = None,
        staging_view: Optional[_BufferView] = None,
        source_ready_events: Optional[_SourceReadyEvents] = None,
        source_ready_waited_keys: Optional[set[tuple[str, int]]] = None,
    ) -> tuple[int, list[torch.device]]:
        copy_devices: list[torch.device] = []
        prepared_copy_streams: dict[tuple[str, int], Any] = {}
        spans = _contiguous_desc_spans(descs, chunk) if spans is None else spans
        if scratch_view is not None:
            if not copy_from_staging:
                raise RuntimeError("B10 recv scratch copy only supports staging-to-VRAM copies")
            if memory_type != "VRAM":
                raise RuntimeError("B10 recv scratch copy only supports VRAM destinations")
            copy_device = _cuda_copy_device(staging_buffer, scratch_view.buffer)
            if copy_device is None:
                raise RuntimeError("B10 recv scratch copy requires a CUDA scratch buffer")
            copy_stream = self._copy_stream_for_device(
                copy_device, prepared_copy_streams, wait_current_stream=False
            )
            copy_devices.extend(
                self._scatter_staging_to_vram_spans(
                    staging_buffer, descs, chunk, spans, scratch_view, copy_stream, lifetime_refs
                )
            )
            return len(spans), copy_devices
        if not copy_from_staging:
            gather_device = self._send_gather_device_for_chunk(
                descs, memory_type, staging_buffer, spans
            )
            if gather_device is not None:
                copy_stream = self._copy_stream_for_device(
                    gather_device,
                    prepared_copy_streams,
                    wait_current_stream=True,
                    source_ready_events=source_ready_events,
                    source_ready_waited_keys=source_ready_waited_keys,
                )
                if _gather_vram_spans_to_pinned_staging(
                    descs, spans, staging_buffer, copy_stream, lifetime_refs
                ):
                    copy_devices.append(gather_device)
                    if staging_view is not None:
                        # The gather kernel's stores into pinned staging are
                        # invisible to torch's CachingHostAllocator (HEAD's
                        # per-span copy_ registered allocator events); this
                        # event restores the reuse barrier the quarantine
                        # prune relies on.
                        event = torch.cuda.Event()
                        event.record(copy_stream)
                        staging_view.ready_event = event
                return len(spans), copy_devices
        staging_tensor = staging_buffer if isinstance(staging_buffer, torch.Tensor) else None
        if (
            memory_type == "VRAM"
            and staging_tensor is not None
            and staging_tensor.device.type == "cpu"
        ):
            # Per-span loop for host-staging chunks the gather kernel does
            # not take (recv staging-to-VRAM fallback, small span counts,
            # multi-device sources); CUDA staging falls through to the
            # generic loop below. The staging buffer is fixed for the whole
            # chunk, so _copy_buffer's per-call direction/pinned probes and
            # the per-copy torch.cuda.stream context (both GIL-held torch
            # dispatches) are hoisted out of the span loop; only the span
            # views change per iteration.
            non_blocking = staging_tensor.is_pinned()
            active_stream = None
            stream_context = None
            try:
                for span in spans:
                    if span.size == 0:
                        # Zero-size spans move no bytes; skip the degenerate
                        # 0-element copy.
                        continue
                    desc_view = _span_view(descs, span, memory_type)
                    if lifetime_refs is not None:
                        lifetime_refs.append(desc_view)
                    desc_buffer = desc_view.buffer
                    staging_slice = staging_tensor[
                        span.chunk_offset : span.chunk_offset + span.size
                    ]
                    # Mirrors _cuda_copy_device: VRAM span views are always
                    # CUDA and staging is host, so the span side is the copy
                    # device.
                    copy_device = desc_buffer.device
                    copy_devices.append(copy_device)
                    copy_stream = self._copy_stream_for_device(
                        copy_device,
                        prepared_copy_streams,
                        wait_current_stream=not copy_from_staging,
                        source_ready_events=source_ready_events,
                        source_ready_waited_keys=source_ready_waited_keys,
                    )
                    if copy_stream is not active_stream:
                        if stream_context is not None:
                            stream_context.__exit__(None, None, None)
                        stream_context = torch.cuda.stream(copy_stream)
                        stream_context.__enter__()
                        active_stream = copy_stream
                    if copy_from_staging:
                        desc_buffer.copy_(staging_slice, non_blocking=non_blocking)
                    else:
                        staging_slice.copy_(desc_buffer, non_blocking=non_blocking)
            finally:
                if stream_context is not None:
                    stream_context.__exit__(None, None, None)
            return len(spans), copy_devices
        for span in spans:
            desc_view = _span_view(descs, span, memory_type)
            if lifetime_refs is not None:
                lifetime_refs.append(desc_view)
            staging_slice = staging_buffer[span.chunk_offset : span.chunk_offset + span.size]
            if copy_from_staging:
                src_buffer = staging_slice
                dst_buffer = desc_view.buffer
            else:
                src_buffer = desc_view.buffer
                dst_buffer = staging_slice
            copy_device = _cuda_copy_device(src_buffer, dst_buffer)
            copy_stream = None
            if copy_device is not None:
                copy_devices.append(copy_device)
                copy_stream = self._copy_stream_for_device(
                    copy_device,
                    prepared_copy_streams,
                    wait_current_stream=not copy_from_staging,
                    source_ready_events=source_ready_events,
                    source_ready_waited_keys=source_ready_waited_keys,
                )
            if copy_stream is None:
                _copy_buffer(src_buffer, dst_buffer)
            else:
                _copy_buffer(src_buffer, dst_buffer, copy_stream=copy_stream)
        return len(spans), copy_devices

    async def _on_endpoint(self, endpoint: Any) -> None:
        transfer_id: Optional[int] = None
        abort_endpoint = True
        try:
            while not self._core.shutdown:
                control = await _recv_obj(endpoint, _BOOTSTRAP_CONTROL_TAG, None)
                transfer_id = int(control.get("transfer_id", -1))
                await self._handle_incoming_write(endpoint, control)
                transfer_id = None
        except b10_protocol.B10TagCollisionError as exc:
            if not self._core.shutdown:
                logger.warning(
                    f"B10 endpoint listener closed after tag collision: "
                    f"transfer_id={transfer_id} "
                    f"error={type(exc).__name__}: {exc} "
                    f"{_format_staging_pool_state(self._core.staging_buffer_pool)}"
                )
        except TimeoutError as exc:
            abort_endpoint = False
            if not self._core.shutdown:
                logger.warning(
                    f"B10 endpoint listener closed after transfer timeout: "
                    f"transfer_id={transfer_id} "
                    f"error={type(exc).__name__}: {exc} "
                    f"{_format_staging_pool_state(self._core.staging_buffer_pool)}"
                )
        except Exception as exc:
            if not self._core.shutdown:
                logger.warning(
                    f"B10 endpoint listener closed: transfer_id={transfer_id} "
                    f"error={type(exc).__name__}: {exc} "
                    f"{_format_staging_pool_state(self._core.staging_buffer_pool)}"
                )
        finally:
            if abort_endpoint:
                _abort_endpoint_background(endpoint)

    async def _handle_incoming_write(self, endpoint: Any, control: dict[str, Any]) -> None:
        # Phase flow for one incoming WRITE (state shared via _RecvTransfer):
        #   _validate_and_prepare  decode/validate the control message, check
        #                          the cancel tombstone, build the context
        #   _reserve_transfer_resources
        #                          mark the request active, reserve message
        #                          tags, plan chunk spans and request-level
        #                          scatter, reserve scratch, size the
        #                          in-flight bounds
        #   _send_ready            READY reply unblocks the sender
        #   _receive_chunks        bounded receive/copy pipeline over the
        #                          DATA chunks, then the deferred
        #                          request-level scatter
        #   _finalize_success      wait for copy events, then RESULT (RESULT
        #                          implies the KV physically landed)
        # Failures before/at READY take _fail_transfer_setup; failures after
        # take _fail_transfer; _finish_transfer always runs last. The phases
        # themselves never release resources on error: the _fail_* funnels
        # sweep still-tracked staging views into quarantine, return scratch
        # views (local-only, no peer-write hazard), and quarantine the
        # transfer's tags. See DESIGN.md "Anatomy of a WRITE".
        ctx = self._validate_and_prepare(endpoint, control)
        try:
            await self._reserve_transfer_resources(ctx)
            await self._send_ready(ctx)
        except BaseException as exc:
            # Setup runs before the data-phase try/finally, so it must clean
            # up after itself. BaseException: asyncio.CancelledError must take
            # this path too.
            self._fail_transfer_setup(ctx, exc)
            raise
        try:
            await self._receive_chunks(ctx)
            await self._finalize_success(ctx)
        except BaseException as exc:
            await self._fail_transfer(ctx, exc)
            raise
        finally:
            self._finish_transfer(ctx)

    def _validate_and_prepare(self, endpoint: Any, control: dict[str, Any]) -> _RecvTransfer:
        ctx = _RecvTransfer()
        ctx.endpoint = endpoint
        ctx.recv_total_start_s = time.perf_counter()
        ctx.recv_timings = {}
        ctx.recv_span_count = 0
        ctx.recv_status = "unknown"
        ctx.recv_in_flight = 0
        ctx.deadline = _TransferDeadline(self._core.transfer_timeout_s)
        if control.get("protocol") != _B10_PROTOCOL:
            raise ValueError(f"Unexpected B10 control protocol: {control.get('protocol')}")
        if int(control.get("version", -1)) != _B10_PROTOCOL_VERSION:
            raise ValueError(f"Unexpected B10 control version: {control.get('version')}")
        ctx.transfer_id = int(control["transfer_id"])
        ctx.request_id = self._request_id_from_sync_message(control.get("sync_message"))
        if ctx.request_id is not None and self._is_recv_request_cancelled(ctx.request_id):
            raise RuntimeError(f"B10 recv request {ctx.request_id} was already cancelled")
        ctx.endpoint_generation = int(control["endpoint_generation"])
        ctx.tag_domain = int(control.get("tag_domain", 0))
        ctx.dst_type = str(control["dst_type"])
        ctx.dst_descs = _dst_descs_from_control(control)
        ctx.dst_descs_have_overlap = _has_overlapping_descs(ctx.dst_descs)
        ctx.transfer_chunks = _transfer_chunks_from_control(control, ctx.dst_descs)
        ctx.desc_count, ctx.total_bytes, ctx.max_desc_size = _memory_desc_stats(ctx.dst_descs)
        ctx.wire_chunk_count, _, ctx.max_wire_chunk_size = _transfer_chunk_stats(
            ctx.transfer_chunks
        )
        self._trace._trace_transfer(
            lambda: f"B10 recv transfer {ctx.transfer_id} control received: "
            f"endpoint_generation={ctx.endpoint_generation} "
            f"tag_domain={ctx.tag_domain} dst_type={ctx.dst_type} "
            f"descs={ctx.desc_count} data_chunks={ctx.wire_chunk_count} "
            f"total_bytes={ctx.total_bytes} max_desc_size={ctx.max_desc_size} "
            f"max_data_chunk_size={ctx.max_wire_chunk_size} "
            f"max_in_flight_ops={self._core.max_in_flight_ops} "
            f"{_format_staging_pool_state(self._core.staging_buffer_pool)}"
        )
        ctx.pool_buffers = self._core.staging_buffer_pool.num_buffers
        self._prune_quarantined_buffer_views()
        # Keep destination view owners alive until recorded copy events drain.
        ctx.dst_view_lifetime_refs = []
        ctx.staging_tracker = _StagingCheckoutTracker()
        ctx.recv_scratch_tracker = _StagingCheckoutTracker()
        ctx.copy_events = []
        ctx.request_scatter_chunks = []
        ctx.recv_copy_request_marked = False
        ctx.recv_request_marked = False
        ctx.received_chunks = asyncio.Queue()
        ctx.use_request_level_scatter = False
        ctx.request_level_scatter_chunk_indices = set()
        ctx.request_scatter_scratch_views = {}
        ctx.request_scatter_admission_marked = False
        ctx.recv_copy_in_flight = 0

        ctx.ready_tag = _ready_tag(ctx.transfer_id, ctx.endpoint_generation, ctx.tag_domain)
        ctx.result_tag = _result_tag(ctx.transfer_id, ctx.endpoint_generation, ctx.tag_domain)
        ctx.tag_owner = ("recv", id(endpoint), ctx.transfer_id)
        return ctx

    def _clear_request_scatter_admission(self, ctx: _RecvTransfer) -> None:
        if not ctx.request_scatter_admission_marked:
            return
        ctx.request_scatter_admission_marked = False
        if self._request_level_recv_scatter_admissions <= 0:
            # Accounting bug. Log instead of raising: this also runs in
            # the terminal finally block, where a raise would mask the
            # original transfer error and skip tag release/quarantine.
            logger.error("B10 request-level recv scatter admission underflow")
            return
        self._request_level_recv_scatter_admissions -= 1

    async def _reserve_request_scatter_scratch_views(self, ctx: _RecvTransfer) -> None:
        if not ctx.request_level_scatter_chunk_indices:
            return
        scratch_device = self._recv_scratch_device
        if scratch_device is None:
            raise RuntimeError("B10 request-level recv scatter scratch device disappeared")
        acquire_start_s = time.perf_counter()
        acquired: list[_BufferView] = []
        lock_acquired = False
        try:
            reserve_lock = self._request_level_recv_scatter_reserve_lock
            await _await_with_timeout(reserve_lock.acquire(), ctx.deadline.remaining_s())
            lock_acquired = True
            self._raise_if_recv_request_cancelled(ctx.request_id)

            for idx in sorted(ctx.request_level_scatter_chunk_indices):
                self._raise_if_recv_request_cancelled(ctx.request_id)
                scratch_view = await self._acquire_recv_scratch_buffer(
                    ctx.transfer_chunks[idx].size, scratch_device, ctx.deadline
                )
                ctx.request_scatter_scratch_views[idx] = scratch_view
                acquired.append(scratch_view)
                ctx.recv_scratch_tracker.track(scratch_view)
                self._raise_if_recv_request_cancelled(ctx.request_id)
        except Exception:
            if acquired:
                self._release_recv_scratch_buffers(acquired)
                for scratch_view in acquired:
                    ctx.recv_scratch_tracker.untrack(scratch_view)
                ctx.request_scatter_scratch_views.clear()
            raise
        finally:
            if lock_acquired:
                reserve_lock.release()
            self._trace._add_elapsed_ms(
                ctx.recv_timings, "recv_scratch_acquire_ms", acquire_start_s
            )

    async def _reserve_transfer_resources(self, ctx: _RecvTransfer) -> None:
        if ctx.request_id is not None:
            self._mark_active_recv_request(ctx.request_id)
            ctx.recv_request_marked = True
        self._reserve_message_tags(
            ctx.tag_owner,
            ctx.transfer_id,
            ctx.endpoint_generation,
            ctx.tag_domain,
            ctx.wire_chunk_count,
        )
        ctx.chunk_dst_spans = [
            _contiguous_desc_spans(ctx.dst_descs, chunk) for chunk in ctx.transfer_chunks
        ]
        ctx.request_level_scatter_chunk_indices = self._request_level_recv_scatter_chunk_indices(
            ctx.dst_descs,
            ctx.dst_type,
            ctx.transfer_chunks,
            ctx.chunk_dst_spans,
            descs_have_overlap=ctx.dst_descs_have_overlap,
        )
        ctx.use_request_level_scatter = bool(ctx.request_level_scatter_chunk_indices)
        if ctx.use_request_level_scatter:
            self._request_level_recv_scatter_admissions += 1
            ctx.request_scatter_admission_marked = True
        uses_recv_scratch = ctx.use_request_level_scatter or any(
            self._should_use_recv_scratch(
                ctx.dst_descs, ctx.dst_type, spans, descs_have_overlap=ctx.dst_descs_have_overlap
            )
            for spans in ctx.chunk_dst_spans
        )
        scratch_buffers = (
            self._core.recv_scratch_buffer_pool.num_buffers
            if uses_recv_scratch
            else ctx.pool_buffers
        )
        if ctx.dst_descs_have_overlap:
            ctx.recv_in_flight = 1
            ctx.recv_copy_in_flight = 1
        else:
            ctx.recv_in_flight = max(1, min(self._core.max_in_flight_ops, ctx.pool_buffers))
            ctx.recv_copy_in_flight = max(
                1, min(self._core.max_in_flight_ops, ctx.pool_buffers, scratch_buffers)
            )
        await self._reserve_request_scatter_scratch_views(ctx)
        self._clear_request_scatter_admission(ctx)
        self._raise_if_recv_request_cancelled(ctx.request_id)

    async def _send_ready(self, ctx: _RecvTransfer) -> None:
        with self._trace.span(ctx.recv_timings, "ready_send_ms"):
            await _send_reply(
                ctx.endpoint,
                ctx.transfer_id,
                ctx.ready_tag,
                {"ok": True},
                ctx.deadline.remaining_s(),
            )
        self._trace._trace_transfer(
            lambda: f"B10 recv transfer {ctx.transfer_id} READY sent: "
            f"ready_tag={ctx.ready_tag} result_tag={ctx.result_tag}"
        )

    def _fail_transfer_setup(self, ctx: _RecvTransfer, exc: BaseException) -> None:
        """Cleanup ladder for failures before/at READY (setup phase)."""
        ctx.recv_status = "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed"
        self._trace._log_recv_transfer_timings(
            ctx.transfer_id,
            ctx.recv_status,
            ctx.recv_timings,
            ctx.recv_total_start_s,
            request_id=ctx.request_id,
            desc_count=ctx.desc_count,
            total_bytes=ctx.total_bytes,
            wire_chunk_count=ctx.wire_chunk_count,
            max_wire_chunk_size=ctx.max_wire_chunk_size,
            copy_events=0,
            span_count=0,
            recv_in_flight=0,
        )
        self._core.tag_registry.quarantine(ctx.tag_owner)
        self._release_recv_scratch_buffers(ctx.recv_scratch_tracker.take_all())
        self._clear_request_scatter_admission(ctx)
        if ctx.recv_request_marked:
            self._clear_active_recv_request(ctx.request_id)
        self._notify_incoming_write_listener(ctx.request_id, ctx.recv_status)

    async def _receive_one(self, ctx: _RecvTransfer, chunk: _TransferChunk, idx: int) -> None:
        data_tag = _data_tag(ctx.transfer_id, idx, ctx.endpoint_generation, ctx.tag_domain)
        try:
            try:
                with self._trace.span(ctx.recv_timings, "staging_acquire_ms"):
                    staging_view = await self._core._acquire_staging_buffer(
                        chunk.size, ctx.deadline
                    )
            except Exception as exc:
                logger.warning(
                    f"B10 recv transfer {ctx.transfer_id} failed to acquire "
                    f"staging buffer: chunk_index={idx} "
                    f"chunk_size={chunk.size} "
                    f"data_chunks={ctx.wire_chunk_count} "
                    f"error={type(exc).__name__}: {exc} "
                    f"{_format_staging_pool_state(self._core.staging_buffer_pool)}"
                )
                raise
            ctx.staging_tracker.track(staging_view)
            spans = ctx.chunk_dst_spans[idx]
            use_request_level_chunk = idx in ctx.request_level_scatter_chunk_indices
            use_recv_scratch = use_request_level_chunk or self._should_use_recv_scratch(
                ctx.dst_descs, ctx.dst_type, spans, descs_have_overlap=ctx.dst_descs_have_overlap
            )
            with self._trace.span(ctx.recv_timings, "ucxx_recv_ms"):
                await self._recv_buffer(ctx.endpoint, staging_view.buffer, data_tag, ctx.deadline)
            await ctx.received_chunks.put(
                _ReceivedTransferChunk(
                    chunk=chunk,
                    index=idx,
                    staging_view=staging_view,
                    spans=spans,
                    use_recv_scratch=use_recv_scratch,
                )
            )
        except Exception as exc:
            logger.warning(
                f"B10 recv transfer {ctx.transfer_id} chunk failed: "
                f"chunk_index={idx} chunk_size={chunk.size} "
                f"data_chunks={ctx.wire_chunk_count} data_tag={data_tag} "
                f"error={type(exc).__name__}: {exc} "
                f"{_format_staging_pool_state(self._core.staging_buffer_pool)}"
            )
            raise

    async def _copy_received(self, ctx: _RecvTransfer, received: _ReceivedTransferChunk) -> None:
        chunk = received.chunk
        idx = received.index
        staging_view = received.staging_view
        spans = received.spans
        scratch_view: Optional[_BufferView] = None
        scratch_gpu_work_started = False
        try:
            if ctx.request_id is not None and not ctx.recv_copy_request_marked:
                self._mark_active_recv_copy_request(ctx.request_id)
                ctx.recv_copy_request_marked = True
            else:
                self._raise_if_recv_request_cancelled(ctx.request_id)
            if idx in ctx.request_level_scatter_chunk_indices:
                # Request-level scatter reserves all scratch before READY, so
                # a missing entry is an accounting bug rather than a fallback.
                scratch_view = ctx.request_scatter_scratch_views.pop(idx)
            elif received.use_recv_scratch and self._request_level_recv_scatter_admissions == 0:
                scratch_device = self._single_cuda_device_for_spans(ctx.dst_descs, spans)
                if scratch_device is None:
                    raise RuntimeError("B10 recv scratch device disappeared")
                with self._trace.span(ctx.recv_timings, "recv_scratch_acquire_ms"):
                    scratch_view = await self._acquire_recv_scratch_buffer(
                        chunk.size, scratch_device, ctx.deadline
                    )
                if self._request_level_recv_scatter_admissions == 0:
                    ctx.recv_scratch_tracker.track(scratch_view)
                    self._raise_if_recv_request_cancelled(ctx.request_id)
                else:
                    self._release_recv_scratch_buffers([scratch_view])
                    scratch_view = None
            copy_start_s = time.perf_counter()
            scratch_gpu_work_started = scratch_view is not None
            if idx in ctx.request_level_scatter_chunk_indices:
                scratch_buffer = scratch_view.buffer[: chunk.size]
                copy_device = _cuda_copy_device(staging_view.buffer, scratch_buffer)
                if copy_device is None:
                    raise RuntimeError("B10 request-level recv scatter requires CUDA scratch")
                copy_stream = self._copy_stream_for_device(
                    copy_device, {}, wait_current_stream=False
                )
                _copy_buffer(
                    staging_view.buffer[: chunk.size], scratch_buffer, copy_stream=copy_stream
                )
                with self._trace.span(ctx.recv_timings, "cuda_event_record_ms"):
                    h2d_events = _record_cuda_copy_events(
                        [scratch_buffer.device], self._core.cuda_copy_streams.stream_for
                    )
                staging_view.ready_event = _ready_event_for_copy_events(h2d_events)
                scratch_view.ready_event = staging_view.ready_event
                ctx.request_scatter_chunks.append(
                    _RequestScatterChunk(chunk=chunk, scratch_view=scratch_view, spans=spans)
                )
                ctx.recv_span_count += len(spans)
                ctx.recv_timings["request_scatter_chunks"] = (
                    ctx.recv_timings.get("request_scatter_chunks", 0.0) + 1.0
                )
                self._trace._add_elapsed_ms(ctx.recv_timings, "h2scratch_ms", copy_start_s)
                copy_devices = []
            else:
                span_count, copy_devices = self._copy_chunk_between_staging_and_descs(
                    ctx.dst_descs,
                    chunk,
                    ctx.dst_type,
                    staging_view.buffer,
                    copy_from_staging=True,
                    lifetime_refs=ctx.dst_view_lifetime_refs,
                    spans=spans,
                    scratch_view=scratch_view,
                )
                ctx.recv_span_count += span_count
            self._trace._add_elapsed_ms(ctx.recv_timings, "dst_copy_ms", copy_start_s)
            with self._trace.span(ctx.recv_timings, "cuda_event_record_ms"):
                chunk_copy_events = _record_cuda_copy_events(
                    copy_devices, self._core.cuda_copy_streams.stream_for
                )
            if chunk_copy_events:
                ctx.copy_events.extend(chunk_copy_events)
                staging_view.ready_event = _ready_event_for_copy_events(chunk_copy_events)
                if scratch_view is not None:
                    scratch_view.ready_event = staging_view.ready_event
            with self._trace.span(ctx.recv_timings, "staging_release_ms"):
                self._core._release_staging_buffers([staging_view])
                if scratch_view is not None and idx not in ctx.request_level_scatter_chunk_indices:
                    self._release_recv_scratch_buffers([scratch_view])
            ctx.staging_tracker.untrack(staging_view)
            if scratch_view is not None and idx not in ctx.request_level_scatter_chunk_indices:
                ctx.recv_scratch_tracker.untrack(scratch_view)
        except Exception as exc:
            if scratch_view is not None:
                release_scratch = True
                if scratch_gpu_work_started and scratch_view.ready_event is None:
                    scratch_events = self._record_recv_scratch_ready_events(
                        ctx.transfer_id, idx, scratch_view
                    )
                    if scratch_events is None:
                        release_scratch = False
                    else:
                        ctx.copy_events.extend(scratch_events)
                if release_scratch:
                    self._release_recv_scratch_buffers([scratch_view])
                ctx.recv_scratch_tracker.untrack(scratch_view)
            data_tag = _data_tag(ctx.transfer_id, idx, ctx.endpoint_generation, ctx.tag_domain)
            logger.warning(
                f"B10 recv transfer {ctx.transfer_id} chunk failed: "
                f"chunk_index={idx} chunk_size={chunk.size} "
                f"data_chunks={ctx.wire_chunk_count} data_tag={data_tag} "
                f"error={type(exc).__name__}: {exc} "
                f"{_format_staging_pool_state(self._core.staging_buffer_pool)}"
            )
            raise

    async def _copy_worker(self, ctx: _RecvTransfer) -> None:
        while True:
            received = await ctx.received_chunks.get()
            if received is None:
                return
            await self._copy_received(ctx, received)

    async def _receive_stage(self, ctx: _RecvTransfer, copy_worker_count: int) -> None:
        try:
            await self._run_limited(
                [
                    (lambda chunk=chunk, idx=idx: self._receive_one(ctx, chunk, idx))
                    for idx, chunk in enumerate(ctx.transfer_chunks)
                ],
                transfer_id=ctx.transfer_id,
                phase="recv DATA receive",
                max_in_flight=ctx.recv_in_flight,
            )
        finally:
            for _ in range(copy_worker_count):
                await ctx.received_chunks.put(None)

    async def _run_receive_copy_pipeline(self, ctx: _RecvTransfer) -> None:
        copy_worker_count = max(1, min(ctx.recv_copy_in_flight, len(ctx.transfer_chunks)))
        producer_task = asyncio.create_task(self._receive_stage(ctx, copy_worker_count))
        worker_tasks = [
            asyncio.create_task(self._copy_worker(ctx)) for _ in range(copy_worker_count)
        ]
        await _gather_cancel_on_failure([producer_task] + worker_tasks)

    def _scatter_request_chunks(self, ctx: _RecvTransfer) -> None:
        if not ctx.use_request_level_scatter:
            return
        if not ctx.request_scatter_chunks:
            return
        scatter_start_s = time.perf_counter()
        chunk_sources = [
            (item.chunk, int(item.scratch_view.buffer.data_ptr()))
            for item in ctx.request_scatter_chunks
        ]
        plan = _destination_scatter_plan_for_chunks(ctx.dst_descs, chunk_sources)
        ctx.recv_timings["request_scatter_fragments"] = float(plan.fragment_count)
        if plan.fragment_count:
            copy_device = torch.device("cuda", int(plan.device_ids[0]))
        else:
            copy_device = ctx.request_scatter_chunks[0].scratch_view.buffer.device
        copy_stream = self._copy_stream_for_device(copy_device, {}, wait_current_stream=False)
        scatter_may_have_started = False
        scatter_timing_recorded = False

        def mark_scatter_started() -> None:
            nonlocal scatter_may_have_started
            scatter_may_have_started = True

        try:
            if plan.fragment_count:
                used_aligned_scatter = _scatter_cuda_buffers_to_vram_destination_order(
                    plan,
                    copy_stream,
                    lifetime_refs=ctx.dst_view_lifetime_refs,
                    mark_launch_started=mark_scatter_started,
                )
                ctx.recv_timings["request_scatter_kernels"] = 1.0
                if used_aligned_scatter:
                    ctx.recv_timings["request_scatter_aligned_kernels"] = 1.0
            else:
                for item in ctx.request_scatter_chunks:
                    scatter_may_have_started = True
                    _scatter_cuda_buffer_to_vram_spans(
                        item.scratch_view.buffer[: item.chunk.size],
                        ctx.dst_descs,
                        item.spans,
                        copy_stream,
                        ctx.dst_view_lifetime_refs,
                        item.scratch_view.metadata,
                    )
                    ctx.recv_span_count += len(item.spans)
            self._trace._add_elapsed_ms(ctx.recv_timings, "dst_copy_ms", scatter_start_s)
            self._trace._add_elapsed_ms(ctx.recv_timings, "request_scatter_ms", scatter_start_s)
            scatter_timing_recorded = True
            with self._trace.span(ctx.recv_timings, "cuda_event_record_ms"):
                request_copy_events = _record_cuda_copy_events(
                    [copy_device], self._core.cuda_copy_streams.stream_for
                )
            ctx.copy_events.extend(request_copy_events)
            ready_event = _ready_event_for_copy_events(request_copy_events)
            for item in ctx.request_scatter_chunks:
                item.scratch_view.ready_event = ready_event
            with self._trace.span(ctx.recv_timings, "scratch_release_ms"):
                self._release_recv_scratch_buffers(
                    [item.scratch_view for item in ctx.request_scatter_chunks]
                )
            for item in ctx.request_scatter_chunks:
                ctx.recv_scratch_tracker.untrack(item.scratch_view)
            ctx.request_scatter_chunks.clear()
        except Exception:
            if not scatter_timing_recorded:
                self._trace._add_elapsed_ms(ctx.recv_timings, "dst_copy_ms", scatter_start_s)
                self._trace._add_elapsed_ms(ctx.recv_timings, "request_scatter_ms", scatter_start_s)
            if scatter_may_have_started:
                sync_start_s = time.perf_counter()
                sync_ok = False
                try:
                    copy_stream.synchronize()
                    sync_ok = True
                except Exception as stream_sync_exc:
                    logger.warning(
                        f"B10 recv transfer {ctx.transfer_id} request-level "
                        f"scatter stream sync failed after scatter "
                        f"may have started: "
                        f"error={type(stream_sync_exc).__name__}: "
                        f"{stream_sync_exc}"
                    )
                    try:
                        torch.cuda.synchronize(copy_device)
                        sync_ok = True
                    except Exception as device_sync_exc:
                        logger.warning(
                            f"B10 recv transfer {ctx.transfer_id} "
                            f"request-level scatter device sync failed "
                            f"after scatter may have started: "
                            f"error={type(device_sync_exc).__name__}: "
                            f"{device_sync_exc}"
                        )
                self._trace._add_elapsed_ms(
                    ctx.recv_timings, "request_scatter_recovery_sync_ms", sync_start_s
                )
                logger.warning(
                    f"B10 recv transfer {ctx.transfer_id} failed after "
                    "request-level scatter may have started; retiring "
                    f"{len(ctx.request_scatter_chunks)} recv scratch buffers "
                    f"recovery_sync_ok={sync_ok}"
                )
                for item in ctx.request_scatter_chunks:
                    self._retire_recv_scratch_buffer(item.scratch_view)
                    ctx.recv_scratch_tracker.untrack(item.scratch_view)
                ctx.request_scatter_chunks.clear()
            raise

    async def _receive_chunks(self, ctx: _RecvTransfer) -> None:
        data_phase_start_s = time.perf_counter()
        try:
            await self._run_receive_copy_pipeline(ctx)
            self._raise_if_recv_request_cancelled(ctx.request_id)
            self._scatter_request_chunks(ctx)
        finally:
            self._trace._add_elapsed_ms(ctx.recv_timings, "data_phase_wall_ms", data_phase_start_s)
        self._trace._trace_transfer(
            lambda: f"B10 recv transfer {ctx.transfer_id} DATA complete: "
            f"copy_events={len(ctx.copy_events)} "
            f"{_format_staging_pool_state(self._core.staging_buffer_pool)} "
            f"{_format_cuda_scratch_pool_state(self._core.recv_scratch_buffer_pool)}"
        )

    async def _finalize_success(self, ctx: _RecvTransfer) -> None:
        with self._trace.span(ctx.recv_timings, "copy_event_wait_ms"):
            await self._core._wait_copy_events_async(ctx.copy_events, ctx.deadline)
        self._raise_if_recv_request_cancelled(ctx.request_id)
        with self._trace.span(ctx.recv_timings, "result_send_ms"):
            await _send_reply(
                ctx.endpoint,
                ctx.transfer_id,
                ctx.result_tag,
                {"ok": True},
                ctx.deadline.remaining_s(),
            )
        self._trace._trace_transfer(
            lambda: f"B10 recv transfer {ctx.transfer_id} RESULT sent: ok=True"
        )
        ctx.recv_status = "success"

    async def _fail_transfer(self, ctx: _RecvTransfer, exc: BaseException) -> None:
        """Cleanup ladder for failures after READY (data/finalize phases)."""
        cancelled = isinstance(exc, asyncio.CancelledError)
        ctx.recv_status = "cancelled" if cancelled else "failed"
        if cancelled:
            try:
                await self._drain_recv_copy_events_for_terminal_status(
                    ctx.transfer_id, ctx.copy_events, ctx.recv_timings
                )
            except Exception:
                pass
        checked_out_views = ctx.staging_tracker.take_all()
        checked_out_scratch_views = ctx.recv_scratch_tracker.take_all()
        self._quarantine_staging_buffers(ctx.transfer_id, checked_out_views)
        self._release_recv_scratch_buffers(checked_out_scratch_views)
        failure_detail = (
            ""
            if cancelled
            else f"total_bytes={ctx.total_bytes} max_desc_size={ctx.max_desc_size} "
            f"max_data_chunk_size={ctx.max_wire_chunk_size} "
        )
        error_detail = "" if cancelled else f"error={type(exc).__name__}: {exc} "
        logger.warning(
            f"B10 recv transfer {ctx.transfer_id} {ctx.recv_status}: "
            f"descs={ctx.desc_count} data_chunks={ctx.wire_chunk_count} "
            f"{failure_detail}"
            f"checked_out_staging_buffers={len(checked_out_views)} "
            f"released_recv_scratch_buffers="
            f"{len(checked_out_scratch_views)} "
            f"{error_detail}"
            f"{_format_staging_pool_state(self._core.staging_buffer_pool)}"
        )
        if not cancelled:
            # Copies must drain before the failure RESULT so the sender
            # cannot observe RESULT while writes are still landing.
            try:
                await self._drain_recv_copy_events_for_terminal_status(
                    ctx.transfer_id, ctx.copy_events, ctx.recv_timings
                )
                await _send_reply(
                    ctx.endpoint,
                    ctx.transfer_id,
                    ctx.result_tag,
                    {"ok": False, "error": str(exc)},
                    ctx.deadline.remaining_s(),
                )
            except Exception as reply_exc:
                logger.warning(
                    f"B10 failed to send failure RESULT for transfer {ctx.transfer_id}: {reply_exc}"
                )

    def _finish_transfer(self, ctx: _RecvTransfer) -> None:
        if ctx.request_id is not None and ctx.recv_copy_request_marked:
            self._clear_active_recv_copy_request(ctx.request_id)
        if ctx.request_id is not None and ctx.recv_request_marked:
            self._clear_active_recv_request(ctx.request_id)
        self._clear_request_scatter_admission(ctx)
        if ctx.recv_status == "success":
            self._core.tag_registry.release(ctx.tag_owner)
        elif ctx.recv_status in ("failed", "cancelled"):
            self._core.tag_registry.quarantine(ctx.tag_owner)
        self._trace._log_recv_transfer_timings(
            ctx.transfer_id,
            ctx.recv_status,
            ctx.recv_timings,
            ctx.recv_total_start_s,
            request_id=ctx.request_id,
            desc_count=ctx.desc_count,
            total_bytes=ctx.total_bytes,
            wire_chunk_count=ctx.wire_chunk_count,
            max_wire_chunk_size=ctx.max_wire_chunk_size,
            copy_events=len(ctx.copy_events),
            span_count=ctx.recv_span_count,
            recv_in_flight=ctx.recv_in_flight,
        )
        self._notify_incoming_write_listener(ctx.request_id, ctx.recv_status)

    def _prune_quarantined_buffer_views(self) -> None:
        now = time.monotonic()
        retained: list[_QuarantinedBufferViews] = []
        for entry in self._quarantined_buffer_views:
            if now - entry.quarantined_at < self._staging_quarantine_ttl_s:
                retained.append(entry)
            elif any(
                view.ready_event is not None and not view.ready_event.query()
                for view in entry.views
            ):
                # Gather-kernel writes into pinned staging are not
                # CachingHostAllocator-tracked (HEAD's per-span copy_ was),
                # so the ready_event is the only reuse barrier: keep the
                # entry past the TTL until the event fires; a later pass
                # prunes it.
                retained.append(entry)
        self._quarantined_buffer_views = retained

    def _quarantine_staging_buffers(
        self, transfer_id: int, views: list[_BufferView], direction: str = "recv"
    ) -> None:
        self._prune_quarantined_buffer_views()
        if not views:
            return
        quarantined_count = len(views)
        self._core.staging_buffer_pool.quarantine(views)
        self._core._release_staging_slots_for_views(views)
        self._quarantined_buffer_views.append(
            _QuarantinedBufferViews(views=views, quarantined_at=time.monotonic())
        )
        logger.warning(
            f"B10 {direction} transfer {transfer_id} quarantined staging buffers: "
            f"count={quarantined_count} "
            f"active_quarantines={len(self._quarantined_buffer_views)} "
            f"{_format_staging_pool_state(self._core.staging_buffer_pool)}"
        )

    async def _drain_recv_copy_events_for_terminal_status(
        self, transfer_id: int, events: list[Any], timings: dict[str, float]
    ) -> None:
        if not events:
            return
        wait_start_s = time.perf_counter()
        try:
            await asyncio.to_thread(self._core._wait_copy_events, events)
        except Exception as exc:
            logger.warning(
                f"B10 recv transfer {transfer_id} failed while draining "
                f"copy events before terminal RESULT: "
                f"error={type(exc).__name__}: {exc}"
            )
            raise
        finally:
            self._trace._add_elapsed_ms(timings, "copy_event_wait_ms", wait_start_s)
