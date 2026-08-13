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
"""Receive side of the B10 UCXX transfer agent.

``RecvPipeline`` handles incoming WRITEs from control validation through
destination-copy completion. ``_RecvTransfer`` carries one WRITE across those
phases; ``_RecvRequestRegistry`` alone owns cross-thread request cancellation
and scheduler-visible receive activity. See DESIGN.md "Anatomy of a WRITE".
"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import TimeoutError
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional

import numpy as np
import torch

from tensorrt_llm import logger
from tensorrt_llm._torch.disaggregation.b10 import memory as b10_memory
from tensorrt_llm._torch.disaggregation.b10.am import B10AmDispatcher
from tensorrt_llm._torch.disaggregation.b10.async_utils import (
    _abort_endpoint_background,
    _await_detached_with_timeout,
    _gather_cancel_on_failure,
    _lock_with_timeout,
    _run_limited,
    _TransferDeadline,
)
from tensorrt_llm._torch.disaggregation.b10.copy_engine import _CopyEngine
from tensorrt_llm._torch.disaggregation.b10.core import _AgentCore
from tensorrt_llm._torch.disaggregation.b10.kernels import (
    _copy_buffer,
    _cuda_copy_device,
    _scatter_cuda_buffer_to_vram_spans,
    _scatter_cuda_buffers_to_vram_destination_order,
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
    _STAGING_VIEW_AM_DIRECT,
    _AmStagingAllocator,
    _format_cuda_scratch_pool_state,
    _format_staging_pool_state,
    _ready_event_for_copy_events,
    _record_cuda_copy_events,
)
from tensorrt_llm._torch.disaggregation.b10.protocol import (
    _AM_HEADER_SIZE,
    _AM_KIND_READY,
    _AM_KIND_RESULT,
    _B10_PROTOCOL,
    _B10_PROTOCOL_VERSION,
    _DEFAULT_TAG_QUARANTINE_TTL_S,
    _ENDPOINT_GENERATION_RING,
    _am_send_message,
    _AmHeader,
    _request_id_from_sync_message,
)
from tensorrt_llm._torch.disaggregation.b10.state import (
    _BufferCheckoutTracker,
    _ReceivedTransferChunk,
    _RequestScatterChunk,
)
from tensorrt_llm._torch.disaggregation.b10.timings import TransferTrace, TransferTracer

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


# Identity of one logical incoming transfer: (sender worker address, transfer
# id). The sender's worker address is stable across its endpoint rebuilds -
# unlike the inbound endpoint handle, which changes on exactly the retry this
# has to recognise - and the transfer id is unchanged by a retry, so together
# they name the same logical transfer across attempts while separating
# concurrent senders whose id rings are independent. What a key maps to is the
# endpoint generation of the attempt currently receiving it, i.e. the live one.
_AttemptKey = tuple[bytes, int]


def _generation_is_newer(candidate: int, live: int) -> bool:
    """Whether `candidate` is a later endpoint generation than `live`.

    Generations wrap (see _next_endpoint_generation), and plain ordering would
    read a wrap as a huge regression and reject a real retry as stale. Compare
    by forward distance instead, treating less than half the ring as "ahead" -
    the serial-number arithmetic of RFC 1982, which is also how TCP orders
    wrapping sequence numbers (RFC 7323 PAWS). At the full uint32 the counter
    now spans, wrapping is unreachable in practice; this keeps the ordering
    correct by construction rather than by argument.
    """
    forward = (candidate - live) % _ENDPOINT_GENERATION_RING
    return 0 < forward < _ENDPOINT_GENERATION_RING // 2


class _RecvRequestRegistry:
    """Request cancellation and scheduler-visible receive activity.

    Tombstones and activity counts are shared with executor threads and use
    ``_lock``. Task bindings are owned by the agent event loop; ``cancel``
    schedules their one-shot cancellation onto that loop.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, cancellation_ttl_s: float):
        self._loop = loop
        self._cancellation_ttl_s = cancellation_ttl_s
        self._lock = threading.Lock()
        self._active_transfer_counts: dict[int, int] = {}
        self._active_copy_counts: dict[int, int] = {}
        self._cancelled_until: dict[int, float] = {}
        self._tasks_by_request_id: dict[int, set[asyncio.Task]] = {}
        # Attempt key -> endpoint_generation of the attempt currently being
        # received, and the handler task receiving it. A sender that loses a NIC
        # retires its endpoint and retries on a fresh one, so generations rise
        # for a given attempt and identify which one is live; see
        # _classify_incoming_attempt(). Keyed per SENDER, never per request: a
        # request whose peer layout overlaps several sender ranks is written by
        # all of them concurrently, and their transfer ids and generations are
        # sender-local (see _RecvTransfer.src_ep_handle).
        self._attempt_generations: dict[_AttemptKey, int] = {}
        self._tasks_by_attempt: dict[_AttemptKey, asyncio.Task] = {}

    def has_active_transfer(self, request_id: int) -> bool:
        with self._lock:
            return self._active_transfer_counts.get(int(request_id), 0) > 0

    def has_active_copy(self, request_id: int) -> bool:
        with self._lock:
            return self._active_copy_counts.get(int(request_id), 0) > 0

    def cancel(self, request_id: int) -> None:
        request_id = int(request_id)
        now = time.monotonic()
        with self._lock:
            for rid, expiry in list(self._cancelled_until.items()):
                if expiry <= now:
                    del self._cancelled_until[rid]
            self._cancelled_until[request_id] = now + self._cancellation_ttl_s
        self._loop.call_soon_threadsafe(self._cancel_tasks, request_id)

    def is_cancelled(self, request_id: int) -> bool:
        with self._lock:
            return self._is_cancelled_locked(int(request_id))

    def new_activity(self, request_id: Optional[int]) -> _RecvRequestActivity:
        return _RecvRequestActivity(self, request_id)

    def _is_cancelled_locked(self, request_id: int) -> bool:
        expiry = self._cancelled_until.get(request_id)
        if expiry is None:
            return False
        if expiry <= time.monotonic():
            del self._cancelled_until[request_id]
            return False
        return True

    @contextmanager
    def track_task(
        self, request_id: Optional[int], attempt_key: Optional[_AttemptKey] = None
    ) -> Iterator[None]:
        """Make the current endpoint handler cancellable.

        By request ID, which cancels every transfer for the request (what an
        aborted request wants), and by attempt key, which cancels only this one
        (what superseding wants - concurrent senders each write part of the
        same request and must not cancel one another).
        """

        task = asyncio.current_task()
        assert task is not None
        tasks = None
        if request_id is not None:
            request_id = int(request_id)
            tasks = self._tasks_by_request_id.setdefault(request_id, set())
            tasks.add(task)
        if attempt_key is not None:
            self._tasks_by_attempt[attempt_key] = task
        try:
            yield
        finally:
            if tasks is not None:
                tasks.discard(task)
                if not tasks and self._tasks_by_request_id.get(request_id) is tasks:
                    del self._tasks_by_request_id[request_id]
            if attempt_key is not None and self._tasks_by_attempt.get(attempt_key) is task:
                del self._tasks_by_attempt[attempt_key]

    def current_attempt_generation(self, attempt_key: _AttemptKey) -> Optional[int]:
        with self._lock:
            return self._attempt_generations.get(attempt_key)

    def begin_attempt(self, attempt_key: _AttemptKey, endpoint_generation: int) -> None:
        with self._lock:
            self._attempt_generations[attempt_key] = int(endpoint_generation)

    def end_attempt(self, attempt_key: _AttemptKey, endpoint_generation: int) -> None:
        """Forget an attempt, unless a newer one already superseded it."""
        with self._lock:
            if self._attempt_generations.get(attempt_key) == int(endpoint_generation):
                del self._attempt_generations[attempt_key]

    def supersede_attempt(self, attempt_key: _AttemptKey) -> None:
        """Cancel one attempt's handler without tombstoning the request.

        Unlike cancel(), the request itself stays receivable and its other
        senders keep transferring: only the superseded attempt is unwound, and
        a newer attempt of it is about to take over. Cancellation runs the old
        handler through its normal failure path, which releases its staging
        buffers into quarantine and unregisters its data sink.
        """
        self._loop.call_soon_threadsafe(self._cancel_attempt_task, attempt_key)

    def _cancel_tasks(self, request_id: int) -> None:
        # Consume on first cancellation so retry polling cannot re-cancel the
        # same task while it drains exceptional-lifetime work.
        for task in self._tasks_by_request_id.pop(request_id, ()):
            if not task.done():
                task.cancel()

    def _cancel_attempt_task(self, attempt_key: _AttemptKey) -> None:
        # Consumed like _cancel_tasks, so a repeated supersede cannot re-cancel
        # a handler that is already unwinding.
        task = self._tasks_by_attempt.pop(attempt_key, None)
        if task is not None and not task.done():
            task.cancel()

    def _start_transfer(self, request_id: int) -> None:
        with self._lock:
            self._increment(self._active_transfer_counts, request_id)

    def _start_copy(self, request_id: int) -> None:
        with self._lock:
            if self._is_cancelled_locked(request_id):
                raise asyncio.CancelledError
            self._increment(self._active_copy_counts, request_id)

    def _finish_transfer(self, request_id: int) -> None:
        with self._lock:
            self._decrement(self._active_transfer_counts, request_id)

    def _finish_copy(self, request_id: int) -> None:
        with self._lock:
            self._decrement(self._active_copy_counts, request_id)

    @staticmethod
    def _increment(counts: dict[int, int], request_id: int) -> None:
        counts[request_id] = counts.get(request_id, 0) + 1

    @staticmethod
    def _decrement(counts: dict[int, int], request_id: int) -> None:
        count = counts.get(request_id, 0)
        if count <= 1:
            counts.pop(request_id, None)
        else:
            counts[request_id] = count - 1


class _RecvRequestActivity:
    """Activity owned by one incoming WRITE for an optional request ID."""

    def __init__(self, registry: _RecvRequestRegistry, request_id: Optional[int]):
        self._registry = registry
        self._request_id = None if request_id is None else int(request_id)
        self._transfer_active = False
        self._copy_active = False

    def start_transfer(self) -> None:
        if self._request_id is None:
            return
        self._registry._start_transfer(self._request_id)
        self._transfer_active = True

    def start_copy(self) -> None:
        """Atomically fence cancellation before READY commits to DATA."""

        if self._request_id is None:
            return
        self._registry._start_copy(self._request_id)
        self._copy_active = True

    def finish(self) -> None:
        if self._request_id is None:
            return
        if self._copy_active:
            self._copy_active = False
            self._registry._finish_copy(self._request_id)
        if self._transfer_active:
            self._transfer_active = False
            self._registry._finish_transfer(self._request_id)


@dataclass(slots=True)
class _RecvTransfer:
    """Mutable state shared by the phases of one incoming WRITE."""

    deadline: _TransferDeadline
    transfer_id: int
    request_id: Optional[int]
    endpoint_generation: int
    # UCX endpoint handle on which CONTROL and DATA arrive. Transfer ids and
    # endpoint generations are sender-local, so the source endpoint is part
    # of the dispatcher key when several peers send concurrently.
    src_ep_handle: int
    # Sender's worker address from the control message; keys the cached
    # reverse endpoint (`endpoint`, assigned right after validation) used
    # for READY/RESULT.
    src_worker_address: bytes
    dst_type: str
    dst_descs: _DescArrayView
    transfer_chunks: list[_TransferChunk]
    pool_buffers: int
    request_activity: _RecvRequestActivity
    trace: TransferTrace
    recv_span_count: int = 0
    recv_status: str = "unknown"
    recv_in_flight: int = 0
    recv_copy_in_flight: int = 0
    dst_descs_have_overlap: bool = field(init=False)
    desc_count: int = field(init=False)
    total_bytes: int = field(init=False)
    max_desc_size: int = field(init=False)
    wire_chunk_count: int = field(init=False)
    max_wire_chunk_size: int = field(init=False)
    dst_view_lifetime_refs: list[_BufferView] = field(default_factory=list)
    staging_tracker: _BufferCheckoutTracker = field(default_factory=_BufferCheckoutTracker)
    recv_scratch_tracker: _BufferCheckoutTracker = field(default_factory=_BufferCheckoutTracker)
    copy_events: list[Any] = field(default_factory=list)
    request_scatter_chunks: list[_RequestScatterChunk] = field(default_factory=list)
    received_chunks: asyncio.Queue[Optional[_ReceivedTransferChunk]] = field(
        default_factory=asyncio.Queue
    )
    use_request_level_scatter: bool = False
    request_level_scatter_chunk_indices: set[int] = field(default_factory=set)
    request_scatter_scratch_views: dict[int, _BufferView] = field(default_factory=dict)
    request_scatter_admission_marked: bool = False
    chunk_dst_spans: list[b10_memory._SpanArrays] = field(default_factory=list)
    # Reverse endpoint for READY/RESULT (assigned after validation).
    endpoint: Any = None

    # Per-chunk delivery futures fed by the AM dispatcher's data sink;
    # `_receive_one` awaits its own index. Created when the sink registers
    # (before READY, so no DATA can precede them).
    am_chunks: dict[int, asyncio.Future] = field(default_factory=dict)
    am_sink_registered: bool = False

    @property
    def attempt_key(self) -> _AttemptKey:
        return (self.src_worker_address, self.transfer_id)

    def __post_init__(self) -> None:
        self.dst_descs_have_overlap = _has_overlapping_descs(self.dst_descs)
        self.desc_count, self.total_bytes, self.max_desc_size = _memory_desc_stats(self.dst_descs)
        self.wire_chunk_count, _, self.max_wire_chunk_size = _transfer_chunk_stats(
            self.transfer_chunks
        )


class RecvPipeline:
    def __init__(
        self,
        core: _AgentCore,
        copies: _CopyEngine,
        tracer: TransferTracer,
        dispatcher: B10AmDispatcher,
        ucxx: Any,
        am_staging_allocator: Optional[_AmStagingAllocator] = None,
    ):
        self._core = core
        self._copies = copies
        self._tracer = tracer
        self._dispatcher = dispatcher
        self._ucxx = ucxx
        self._am_staging_allocator = am_staging_allocator
        # Reverse (READY/RESULT) endpoints toward senders, keyed by the
        # sender's worker address blob carried in each control message.
        # Worker-address endpoints, same as the send side, so replies are
        # failover-capable too. An entry is dropped when a reply send fails;
        # the next transfer from that sender recreates it.
        self._reply_endpoints: dict[bytes, Any] = {}
        dispatcher.set_control_handler(self._on_am_control)
        self._incoming_write_listener: Optional[Callable[[int, bool], None]] = None
        # Predicate answering "is this request still expecting data?". Installed
        # by the owner of the receive-session table; absent for direct agent
        # users, in which case every request counts as expected.
        self._expectation_check: Optional[Callable[[int], bool]] = None
        self._recv_scratch_buffer_slots = asyncio.BoundedSemaphore(
            core.recv_scratch_buffer_pool.num_buffers
        )
        self._recv_scratch_device: Optional[torch.device] = None
        self._retired_recv_scratch_views: list[_BufferView] = []
        self._request_scatter_reservation_lock = asyncio.Lock()
        self._request_level_recv_scatter_admissions = 0
        # Attempt-classification counters (see _classify_incoming_attempt).
        self._duplicate_attempts_dropped = 0
        self._stale_attempts_rejected = 0
        self._attempts_superseded = 0
        self._unexpected_attempts_rejected = 0
        cancelled_recv_ttl_s = _DEFAULT_TAG_QUARANTINE_TTL_S
        if core.transfer_timeout_s is not None:
            cancelled_recv_ttl_s = max(cancelled_recv_ttl_s, 2.0 * core.transfer_timeout_s)
        self._requests = _RecvRequestRegistry(core.loop, cancelled_recv_ttl_s)
        self._preallocate_recv_scratch_buffers()

    def has_active_recv_request(self, request_id: int) -> bool:
        return self._requests.has_active_transfer(request_id)

    def has_active_recv_copy_request(self, request_id: int) -> bool:
        return self._requests.has_active_copy(request_id)

    def cancel_recv_request(self, request_id: int) -> None:
        self._requests.cancel(request_id)

    def set_incoming_write_listener(self, listener: Callable[[int, bool], None]) -> None:
        """Register listener(request_id, success), invoked when an incoming
        WRITE reaches a terminal state on this (receiver) agent. Lets the
        session layer complete receive tasks locally instead of depending on
        the sender's KV_AGENT_RESULT notification. Runs on the agent's event
        loop thread; keep it brief and non-blocking."""
        self._incoming_write_listener = listener

    def set_transfer_expectation_check(self, check: Callable[[int], bool]) -> None:
        """Register check(request_id) -> bool, answering whether the request is
        still expecting data to land. Without it, an attempt for a request with
        no in-flight attempt is indistinguishable from a first attempt, so a
        retry arriving after the transfer already completed would be accepted
        and would write a destination the session layer has since released.

        Called from the agent's event loop thread on every incoming control
        message; keep it to a membership test."""
        self._expectation_check = check

    def _is_expected_transfer(self, request_id: int) -> bool:
        check = self._expectation_check
        if check is None:
            # No session layer to ask; preserve the pre-existing behaviour of
            # treating any uncorrelated attempt as a first attempt.
            return True
        try:
            return bool(check(request_id))
        except Exception as exc:
            # A broken predicate must not reject live transfers.
            logger.warning(f"B10 recv expectation check failed for request {request_id}: {exc}")
            return True

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
        return await self._core._acquire_pooled_buffer(
            self._recv_scratch_buffer_slots,
            lambda: self._core.recv_scratch_buffer_pool.acquire(size, device),
            deadline,
        )

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
        # Permits follow the pool's decision, as on the staging side: a
        # refused view already returned its permit the first time.
        self._release_recv_scratch_slots_for_views(
            self._core.recv_scratch_buffer_pool.release(views)
        )

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
        span_device = self._copies.single_cuda_device(descs, spans)
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

    def _on_am_control(
        self, src_ep_handle: int, header: _AmHeader, control: dict[str, Any]
    ) -> None:
        """AM dispatcher control handler: runs on the agent loop; spawns one
        incoming-write task per transfer (the AM analog of the per-endpoint
        listener loop)."""
        if self._core.shutdown:
            return
        if (
            control.get("transfer_id") != header.transfer_id
            or control.get("endpoint_generation") != header.endpoint_generation
        ):
            logger.warning(
                f"B10 CONTROL header/payload identity mismatch: "
                f"header_transfer_id={header.transfer_id} "
                f"payload_transfer_id={control.get('transfer_id')} "
                f"header_endpoint_generation={header.endpoint_generation} "
                f"payload_endpoint_generation={control.get('endpoint_generation')}"
            )
            return
        self._core.loop.create_task(self._handle_incoming_write_safe(src_ep_handle, control))

    async def _handle_incoming_write_safe(
        self, src_ep_handle: int, control: dict[str, Any]
    ) -> None:
        transfer_id = int(control.get("transfer_id", -1))
        try:
            await self._handle_incoming_write(src_ep_handle, control)
        except TimeoutError as exc:
            if not self._core.shutdown:
                logger.warning(
                    f"B10 incoming write failed after transfer timeout: "
                    f"transfer_id={transfer_id} "
                    f"error={type(exc).__name__}: {exc} "
                    f"{_format_staging_pool_state(self._core.staging_buffer_pool)}"
                )
        except Exception as exc:
            if not self._core.shutdown:
                logger.warning(
                    f"B10 incoming write failed: transfer_id={transfer_id} "
                    f"error={type(exc).__name__}: {exc} "
                    f"{_format_staging_pool_state(self._core.staging_buffer_pool)}"
                )

    async def _get_reply_endpoint(self, src_worker_address: bytes, deadline: Any) -> Any:
        endpoint = self._reply_endpoints.get(src_worker_address)
        if endpoint is None:
            address = self._ucxx.get_ucx_address_from_buffer(src_worker_address)
            endpoint = await _await_detached_with_timeout(
                self._ucxx.create_endpoint_from_worker_address(address),
                deadline.remaining_s(),
                on_late_result=_abort_endpoint_background,
            )
            self._reply_endpoints[src_worker_address] = endpoint
        return endpoint

    def _drop_reply_endpoint(self, src_worker_address: bytes, endpoint: Any) -> None:
        if self._reply_endpoints.get(src_worker_address) is endpoint:
            self._reply_endpoints.pop(src_worker_address, None)
        _abort_endpoint_background(endpoint)

    async def _handle_incoming_write(self, src_ep_handle: int, control: dict[str, Any]) -> None:
        ctx = self._validate_and_prepare(src_ep_handle, control)
        # Classify before reserving anything: a duplicate must cost a dictionary
        # lookup, not staging buffers and a second destination write.
        decision = self._classify_incoming_attempt(ctx)
        if decision == "drop_duplicate":
            self._duplicate_attempts_dropped += 1
            logger.info(
                f"B10 recv dropped duplicate transfer {ctx.transfer_id}: "
                f"request={ctx.request_id} "
                f"endpoint_generation={ctx.endpoint_generation} already in "
                f"flight from this sender "
                f"(dropped={self._duplicate_attempts_dropped})"
            )
            return
        if decision == "reject_unexpected":
            self._unexpected_attempts_rejected += 1
            logger.warning(
                f"B10 recv rejected unexpected transfer {ctx.transfer_id}: "
                f"request={ctx.request_id} is not awaiting data, so this "
                f"endpoint_generation={ctx.endpoint_generation} attempt arrived "
                f"after the request already reached a terminal state "
                f"(rejected={self._unexpected_attempts_rejected})"
            )
            return
        if decision == "reject_stale":
            self._stale_attempts_rejected += 1
            logger.warning(
                f"B10 recv rejected stale transfer {ctx.transfer_id}: "
                f"request={ctx.request_id} "
                f"endpoint_generation={ctx.endpoint_generation} is older than "
                f"the in-flight "
                f"{self._requests.current_attempt_generation(ctx.attempt_key)} "
                f"(rejected={self._stale_attempts_rejected})"
            )
            return
        if decision == "supersede":
            self._attempts_superseded += 1
            logger.warning(
                f"B10 recv superseding transfer {ctx.transfer_id}: "
                f"request={ctx.request_id} endpoint_generation "
                f"{self._requests.current_attempt_generation(ctx.attempt_key)} "
                f"-> {ctx.endpoint_generation}; the older attempt cannot "
                f"complete (superseded={self._attempts_superseded})"
            )
            self._requests.supersede_attempt(ctx.attempt_key)
        self._requests.begin_attempt(ctx.attempt_key, ctx.endpoint_generation)
        # Reverse endpoint for READY/RESULT, from the sender's worker address
        # in the control message (self-contained; no registration-plane
        # dependency). DATA needs no endpoint object at all — it arrives via
        # the worker-scoped dispatcher.
        #
        # Connecting back to the sender is the one step that owns an attempt
        # slot without yet being cancellable, since track_task() below is what
        # registers this handler. So release the slot by hand if the connect
        # fails - the sender just lost a NIC, which is exactly when connecting
        # back to it times out, and a slot left behind would make every later
        # attempt of this transfer look like a duplicate forever.
        try:
            ctx.endpoint = await self._get_reply_endpoint(ctx.src_worker_address, ctx.deadline)
        except BaseException:
            self._requests.end_attempt(ctx.attempt_key, ctx.endpoint_generation)
            raise
        # For the same reason, a newer attempt that arrived while we were
        # connecting could not cancel this handler; it only took the slot. Yield
        # to it here, before reserving resources or writing the destination it
        # now owns.
        if self._requests.current_attempt_generation(ctx.attempt_key) != ctx.endpoint_generation:
            self._attempts_superseded += 1
            logger.warning(
                f"B10 recv abandoning superseded transfer {ctx.transfer_id}: "
                f"request={ctx.request_id} endpoint_generation "
                f"{ctx.endpoint_generation} lost the attempt to "
                f"{self._requests.current_attempt_generation(ctx.attempt_key)} "
                f"while connecting back to the sender "
                f"(superseded={self._attempts_superseded})"
            )
            return
        # No await separates tombstone validation from task registration, so
        # cancellation is caught by one or the other without a race window.
        with self._requests.track_task(ctx.request_id, ctx.attempt_key):
            ready_sent = False
            try:
                await self._reserve_transfer_resources(ctx)
                await self._send_ready(ctx)
                ready_sent = True
                await self._receive_chunks(ctx)
                await self._finalize_success(ctx)
            except BaseException as exc:
                if ready_sent:
                    await self._fail_after_ready(ctx, exc)
                else:
                    await self._fail_before_ready(ctx, exc)
                raise
            finally:
                self._finish_transfer(ctx)

    def _classify_incoming_attempt(self, ctx: _RecvTransfer) -> str:
        """Decide what to do with an incoming transfer.

        A control message can arrive more than once for the same transfer: the
        transport may retransmit it after a lane failure, and a sender that lost
        a NIC retires its endpoint and retries on a fresh one. Deciding by
        ``endpoint_generation`` separates those cases, because a retry that
        followed an endpoint rebuild necessarily carries a later generation while
        a pure retransmission carries the same one.

        Generations are only comparable **within one sender**, since each
        sender's counter lives on its own endpoint slot, so the comparison is
        scoped to ``ctx.attempt_key``. Whether the request still wants data is a
        separate, request-scoped question, and only the session layer can answer
        it.

        Returns "accept", "drop_duplicate", "supersede", "reject_stale" or
        "reject_unexpected".
        """
        live = self._requests.current_attempt_generation(ctx.attempt_key)
        if live is None:
            # Nothing in flight for this transfer. Either it is the first
            # attempt, or every earlier one already reached a terminal state -
            # including success, after which the destination belongs to the
            # session layer again and must not be written.
            if ctx.request_id is None:
                # Nothing to ask the session layer about; behave as before.
                return "accept"
            if not self._is_expected_transfer(ctx.request_id):
                return "reject_unexpected"
            return "accept"
        if ctx.endpoint_generation == live:
            # Same endpoint, so this is the same attempt arriving twice. The
            # in-flight handler owns the single READY/RESULT pair the sender is
            # waiting for; answering again would double-reserve resources and
            # write the destination twice.
            return "drop_duplicate"
        if _generation_is_newer(ctx.endpoint_generation, live):
            # The sender rebuilt its endpoint, so the attempt we are still
            # holding cannot complete - its data will never arrive. Hand over
            # rather than making the sender wait out a timeout.
            return "supersede"
        return "reject_stale"

    def _validate_and_prepare(self, src_ep_handle: int, control: dict[str, Any]) -> _RecvTransfer:
        deadline = _TransferDeadline(self._core.transfer_timeout_s)
        if control.get("protocol") != _B10_PROTOCOL:
            raise ValueError(f"Unexpected B10 control protocol: {control.get('protocol')}")
        if int(control.get("version", -1)) != _B10_PROTOCOL_VERSION:
            raise ValueError(f"Unexpected B10 control version: {control.get('version')}")
        transfer_id = int(control["transfer_id"])
        trace = self._tracer.start("recv", transfer_id)
        request_id = _request_id_from_sync_message(control.get("sync_message"))
        if request_id is not None and self._requests.is_cancelled(request_id):
            raise RuntimeError(f"B10 recv request {request_id} was already cancelled")
        endpoint_generation = int(control["endpoint_generation"])
        src_worker_address = control.get("src_worker_address")
        if not src_worker_address:
            raise ValueError(
                f"B10 control for transfer {transfer_id} carries no "
                "src_worker_address (peer running a pre-AM build?)"
            )
        dst_type = str(control["dst_type"])
        dst_descs = _dst_descs_from_control(control)
        ctx = _RecvTransfer(
            deadline=deadline,
            transfer_id=transfer_id,
            request_id=request_id,
            endpoint_generation=endpoint_generation,
            src_ep_handle=src_ep_handle,
            src_worker_address=bytes(src_worker_address),
            dst_type=dst_type,
            dst_descs=dst_descs,
            transfer_chunks=_transfer_chunks_from_control(control, dst_descs),
            pool_buffers=self._core.staging_buffer_pool.num_buffers,
            request_activity=self._requests.new_activity(request_id),
            trace=trace,
        )
        ctx.trace.debug(
            lambda: (
                f"control received: "
                f"endpoint_generation={ctx.endpoint_generation} "
                f"dst_type={ctx.dst_type} "
                f"descs={ctx.desc_count} data_chunks={ctx.wire_chunk_count} "
                f"total_bytes={ctx.total_bytes} max_desc_size={ctx.max_desc_size} "
                f"max_data_chunk_size={ctx.max_wire_chunk_size} "
                f"max_in_flight_ops={self._core.max_in_flight_ops} "
                f"{_format_staging_pool_state(self._core.staging_buffer_pool)}"
            )
        )
        self._core._prune_quarantined_staging_buffers()
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
        acquired: list[_BufferView] = []
        with ctx.trace.measure_attempt("recv_scratch_acquire"):
            try:
                async with _lock_with_timeout(
                    self._request_scatter_reservation_lock,
                    ctx.deadline.remaining_s(),
                ):
                    for idx in sorted(ctx.request_level_scatter_chunk_indices):
                        scratch_view = await self._acquire_recv_scratch_buffer(
                            ctx.transfer_chunks[idx].size, scratch_device, ctx.deadline
                        )
                        ctx.request_scatter_scratch_views[idx] = scratch_view
                        acquired.append(scratch_view)
                        ctx.recv_scratch_tracker.track(scratch_view)
            except Exception:
                if acquired:
                    self._release_recv_scratch_buffers(acquired)
                    for scratch_view in acquired:
                        ctx.recv_scratch_tracker.untrack(scratch_view)
                    ctx.request_scatter_scratch_views.clear()
                raise

    def _register_am_data_sink(self, ctx: _RecvTransfer) -> None:
        """Route this transfer's DATA messages into per-chunk futures.

        Registered before READY is sent, so no DATA can precede it (the
        sender only ships DATA after READY). The sink runs on the agent loop
        (dispatcher dispatch context); each future holds a zero-copy view
        over the ucxx-allocated receive buffer, which the view keeps alive
        until `_receive_one` copies it into pinned staging.
        """
        loop = self._core.loop
        ctx.am_chunks = {idx: loop.create_future() for idx in range(ctx.wire_chunk_count)}

        def sink(chunk_index: int, payload: Any) -> None:
            future = ctx.am_chunks.get(chunk_index)
            if future is None or future.done():
                logger.warning(
                    f"B10 recv transfer {ctx.transfer_id} dropped unexpected "
                    f"DATA chunk_index={chunk_index} "
                    f"(chunks={ctx.wire_chunk_count}, duplicate or out of range)"
                )
                return
            future.set_result(payload)

        self._dispatcher.register_data_sink(
            ctx.src_ep_handle,
            ctx.transfer_id,
            ctx.endpoint_generation,
            sink,
        )
        ctx.am_sink_registered = True

    def _unregister_am_data_sink(self, ctx: _RecvTransfer) -> None:
        if not ctx.am_sink_registered:
            return
        ctx.am_sink_registered = False
        self._dispatcher.unregister_data_sink(
            ctx.src_ep_handle, ctx.transfer_id, ctx.endpoint_generation
        )
        ctx.am_chunks.clear()

    async def _reserve_transfer_resources(self, ctx: _RecvTransfer) -> None:
        ctx.request_activity.start_transfer()
        self._register_am_data_sink(ctx)
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
        ctx.request_activity.start_copy()

    async def _send_reply_am(self, ctx: _RecvTransfer, kind: int, payload: dict[str, Any]) -> None:
        try:
            await _am_send_message(
                ctx.endpoint,
                kind,
                ctx.transfer_id,
                ctx.endpoint_generation,
                {**payload, "transfer_id": ctx.transfer_id},
                ctx.deadline.remaining_s(),
            )
        except BaseException:
            # A failed reply leaves the reverse endpoint in unknown state;
            # drop it so the next transfer from this sender gets a fresh one.
            self._drop_reply_endpoint(ctx.src_worker_address, ctx.endpoint)
            raise

    async def _send_ready(self, ctx: _RecvTransfer) -> None:
        with ctx.trace.measure("ready_send"):
            await self._send_reply_am(ctx, _AM_KIND_READY, {"ok": True})
        ctx.trace.debug(lambda: "READY sent")

    async def _fail_before_ready(self, ctx: _RecvTransfer, exc: BaseException) -> None:
        ctx.recv_status = "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed"
        self._release_recv_scratch_buffers(ctx.recv_scratch_tracker.take_all())
        if ctx.recv_status == "cancelled":
            return
        try:
            await self._send_reply_am(ctx, _AM_KIND_READY, {"ok": False, "error": str(exc)})
        except Exception as reply_exc:
            logger.warning(
                f"B10 failed to send failure READY for transfer {ctx.transfer_id}: {reply_exc}"
            )

    def _claim_am_staging(self, payload: memoryview) -> Optional[_BufferView]:
        """Claim the staging-pool view an AM message was received into, if
        the AM staging allocator served that receive. The allocation starts
        at the in-band header, `_AM_HEADER_SIZE` bytes before the payload."""
        if self._am_staging_allocator is None or len(payload) == 0:
            return None
        payload_ptr = torch.frombuffer(payload, dtype=torch.uint8).data_ptr()
        return self._am_staging_allocator.claim(payload_ptr - _AM_HEADER_SIZE)

    async def _receive_one(self, ctx: _RecvTransfer, chunk: _TransferChunk, idx: int) -> None:
        try:
            # Wait for this chunk's AM delivery before taking a staging
            # buffer, so staging is never held hostage to the network. The
            # payload is a view over the ucxx-allocated host buffer.
            with ctx.trace.measure("ucxx_recv"):
                payload = await _await_detached_with_timeout(
                    ctx.am_chunks[idx], ctx.deadline.remaining_s()
                )
            if len(payload) != chunk.size:
                raise RuntimeError(
                    f"B10 recv transfer {ctx.transfer_id} chunk {idx} size "
                    f"mismatch: header/payload {len(payload)} B, plan {chunk.size} B"
                )
            claimed_view = self._claim_am_staging(payload)
            if claimed_view is not None:
                # The AM allocator already landed this message in a pinned
                # staging buffer: slice the payload region out (the in-band
                # header occupies the first _AM_HEADER_SIZE bytes) and skip
                # both the staging acquire and the copy-in below.
                staging_view = _BufferView(
                    buffer=claimed_view.buffer[_AM_HEADER_SIZE : _AM_HEADER_SIZE + chunk.size],
                    owner=claimed_view.owner,
                    pool=claimed_view.pool,
                    metadata={_STAGING_VIEW_AM_DIRECT: True},
                )
            else:
                try:
                    with ctx.trace.measure("staging_acquire"):
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
            if claimed_view is None:
                # Host-to-pinned-host copy out of the ucxx AM buffer; the
                # copy engine's async H2D path requires pinned staging, which
                # the ucxx-internal buffer is not. Dropping the future's
                # payload ref afterwards releases the ucxx buffer promptly.
                with ctx.trace.measure("h2scratch"):
                    staging_view.buffer[: chunk.size].copy_(
                        torch.frombuffer(payload, dtype=torch.uint8)
                    )
            ctx.am_chunks.pop(idx, None)
            del payload
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
                f"data_chunks={ctx.wire_chunk_count} "
                f"error={type(exc).__name__}: {exc} "
                f"{_format_staging_pool_state(self._core.staging_buffer_pool)}"
            )
            raise

    async def _copy_received(self, ctx: _RecvTransfer, received: _ReceivedTransferChunk) -> None:
        chunk = received.chunk
        idx = received.index
        staging_view = received.staging_view
        spans = received.spans
        uses_request_scatter = idx in ctx.request_level_scatter_chunk_indices
        scratch_view: Optional[_BufferView] = None
        scratch_gpu_work_started = False
        try:
            if uses_request_scatter:
                # Request-level scatter reserves all scratch before READY, so
                # a missing entry is an accounting bug rather than a fallback.
                scratch_view = ctx.request_scatter_scratch_views.pop(idx)
            elif received.use_recv_scratch and self._request_level_recv_scatter_admissions == 0:
                scratch_device = self._copies.single_cuda_device(ctx.dst_descs, spans)
                if scratch_device is None:
                    raise RuntimeError("B10 recv scratch device disappeared")
                with ctx.trace.measure("recv_scratch_acquire"):
                    scratch_view = await self._acquire_recv_scratch_buffer(
                        chunk.size, scratch_device, ctx.deadline
                    )
                if self._request_level_recv_scatter_admissions == 0:
                    ctx.recv_scratch_tracker.track(scratch_view)
                else:
                    self._release_recv_scratch_buffers([scratch_view])
                    scratch_view = None
            copy_metrics = ("dst_copy", "h2scratch") if uses_request_scatter else ("dst_copy",)
            copy_timer = ctx.trace.timer(*copy_metrics)
            scratch_gpu_work_started = scratch_view is not None
            if uses_request_scatter:
                scratch_buffer = scratch_view.buffer[: chunk.size]
                copy_device = _cuda_copy_device(staging_view.buffer, scratch_buffer)
                if copy_device is None:
                    raise RuntimeError("B10 request-level recv scatter requires CUDA scratch")
                copy_stream = self._core.cuda_copy_streams.stream_for(copy_device)
                _copy_buffer(
                    staging_view.buffer[: chunk.size],
                    scratch_buffer,
                    copy_stream=copy_stream,
                )
                with ctx.trace.measure("cuda_event_record"):
                    h2d_events = _record_cuda_copy_events(
                        [scratch_buffer.device], self._core.cuda_copy_streams.stream_for
                    )
                staging_view.ready_event = _ready_event_for_copy_events(h2d_events)
                scratch_view.ready_event = staging_view.ready_event
                ctx.request_scatter_chunks.append(
                    _RequestScatterChunk(chunk=chunk, scratch_view=scratch_view, spans=spans)
                )
                ctx.recv_span_count += len(spans)
                ctx.trace.increment("request_scatter_chunks")
                copy_devices = []
            else:
                span_count, copy_devices = self._copies.copy_chunk(
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
            copy_timer.stop()
            with ctx.trace.measure("cuda_event_record"):
                chunk_copy_events = _record_cuda_copy_events(
                    copy_devices, self._core.cuda_copy_streams.stream_for
                )
            if chunk_copy_events:
                ctx.copy_events.extend(chunk_copy_events)
                staging_view.ready_event = _ready_event_for_copy_events(chunk_copy_events)
                if scratch_view is not None:
                    scratch_view.ready_event = staging_view.ready_event
            with ctx.trace.measure("staging_release"):
                self._core._release_staging_buffers([staging_view])
                if scratch_view is not None and not uses_request_scatter:
                    self._release_recv_scratch_buffers([scratch_view])
            ctx.staging_tracker.untrack(staging_view)
            if scratch_view is not None and not uses_request_scatter:
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
            logger.warning(
                f"B10 recv transfer {ctx.transfer_id} chunk failed: "
                f"chunk_index={idx} chunk_size={chunk.size} "
                f"data_chunks={ctx.wire_chunk_count} "
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
            # Queue.get() does not suspend while DATA is backlogged. Yield so
            # request cancellation reaches the pipeline between copy chunks.
            await asyncio.sleep(0)

    async def _receive_stage(self, ctx: _RecvTransfer, copy_worker_count: int) -> None:
        try:
            await _run_limited(
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
        scatter_timer = ctx.trace.timer("dst_copy", "request_scatter")
        chunk_sources = [
            (item.chunk, int(item.scratch_view.buffer.data_ptr()))
            for item in ctx.request_scatter_chunks
        ]
        plan = _destination_scatter_plan_for_chunks(ctx.dst_descs, chunk_sources)
        ctx.trace.set("request_scatter_fragments", plan.fragment_count)
        if plan.fragment_count:
            copy_device = torch.device("cuda", int(plan.device_ids[0]))
        else:
            copy_device = ctx.request_scatter_chunks[0].scratch_view.buffer.device
        copy_stream = self._core.cuda_copy_streams.stream_for(copy_device)
        scatter_may_have_started = False

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
                ctx.trace.set("request_scatter_kernels", 1)
                if used_aligned_scatter:
                    ctx.trace.set("request_scatter_aligned_kernels", 1)
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
            scatter_timer.stop()
            with ctx.trace.measure("cuda_event_record"):
                request_copy_events = _record_cuda_copy_events(
                    [copy_device], self._core.cuda_copy_streams.stream_for
                )
            ctx.copy_events.extend(request_copy_events)
            ready_event = _ready_event_for_copy_events(request_copy_events)
            for item in ctx.request_scatter_chunks:
                item.scratch_view.ready_event = ready_event
            with ctx.trace.measure("scratch_release"):
                self._release_recv_scratch_buffers(
                    [item.scratch_view for item in ctx.request_scatter_chunks]
                )
            for item in ctx.request_scatter_chunks:
                ctx.recv_scratch_tracker.untrack(item.scratch_view)
            ctx.request_scatter_chunks.clear()
        except Exception:
            scatter_timer.stop()
            if scatter_may_have_started:
                sync_ok = False
                with ctx.trace.measure("request_scatter_recovery_sync"):
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
        with ctx.trace.measure_attempt("data_phase_wall"):
            await self._run_receive_copy_pipeline(ctx)
            self._scatter_request_chunks(ctx)
        ctx.trace.debug(
            lambda: (
                f"DATA complete: "
                f"copy_events={len(ctx.copy_events)} "
                f"{_format_staging_pool_state(self._core.staging_buffer_pool)} "
                f"{_format_cuda_scratch_pool_state(self._core.recv_scratch_buffer_pool)}"
            )
        )

    async def _finalize_success(self, ctx: _RecvTransfer) -> None:
        with ctx.trace.measure("copy_event_wait"):
            await self._core._wait_copy_events_async(ctx.copy_events, ctx.deadline)
        with ctx.trace.measure("result_send"):
            await self._send_reply_am(ctx, _AM_KIND_RESULT, {"ok": True})
        ctx.trace.debug(lambda: "RESULT sent: ok=True")
        ctx.recv_status = "success"

    async def _fail_after_ready(self, ctx: _RecvTransfer, exc: BaseException) -> None:
        cancelled = isinstance(exc, asyncio.CancelledError)
        ctx.recv_status = "cancelled" if cancelled else "failed"
        checked_out_views = ctx.staging_tracker.take_all()
        checked_out_scratch_views = ctx.recv_scratch_tracker.take_all()
        self._core._quarantine_staging_buffers(ctx.transfer_id, checked_out_views, "recv")
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
        try:
            await self._drain_recv_copy_events_for_terminal_status(ctx)
        except Exception:
            return
        if cancelled:
            return
        # RESULT must not overtake destination copies.
        try:
            await self._send_reply_am(ctx, _AM_KIND_RESULT, {"ok": False, "error": str(exc)})
        except Exception as reply_exc:
            logger.warning(
                f"B10 failed to send failure RESULT for transfer {ctx.transfer_id}: {reply_exc}"
            )

    def _finish_transfer(self, ctx: _RecvTransfer) -> None:
        # Release the attempt slot first: a retry of this transfer may already
        # be waiting, and must not be mistaken for a duplicate of this one.
        self._requests.end_attempt(ctx.attempt_key, ctx.endpoint_generation)
        ctx.request_activity.finish()
        self._clear_request_scatter_admission(ctx)
        # After this, a late DATA for this transfer has no sink and is
        # dropped by the dispatcher as stale (the tag-quarantine analog).
        self._unregister_am_data_sink(ctx)
        self._tracer.log_recv(
            ctx.trace,
            ctx.recv_status,
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

    async def _drain_recv_copy_events_for_terminal_status(self, ctx: _RecvTransfer) -> None:
        if not ctx.copy_events:
            return
        with ctx.trace.measure_attempt("copy_event_wait"):
            try:
                await asyncio.to_thread(self._core._wait_copy_events, ctx.copy_events)
            except Exception as exc:
                logger.warning(
                    f"B10 recv transfer {ctx.transfer_id} failed while draining "
                    f"copy events before terminal RESULT: "
                    f"error={type(exc).__name__}: {exc}"
                )
                raise
