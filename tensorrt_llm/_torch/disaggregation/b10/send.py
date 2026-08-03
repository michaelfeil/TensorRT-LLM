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
"""Send side of the B10 UCXX transfer agent.

``SendPipeline`` plans and executes outgoing WRITEs over a leased persistent
endpoint. A transfer owns its admission permit, endpoint slot, and staging
views until its single terminal cleanup boundary. See DESIGN.md "Anatomy
of a WRITE" and "Timeout and failure handling".
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch

from tensorrt_llm import logger
from tensorrt_llm._torch.disaggregation.b10 import net as b10_net
from tensorrt_llm._torch.disaggregation.b10.am import B10AmDispatcher
from tensorrt_llm._torch.disaggregation.b10.async_utils import (
    _acquire_with_timeout,
    _await_detached_with_timeout,
    _is_retryable_endpoint_error,
    _lock_with_timeout,
    _run_limited,
    _TransferDeadline,
)
from tensorrt_llm._torch.disaggregation.b10.copy_engine import _CopyEngine
from tensorrt_llm._torch.disaggregation.b10.core import _AgentCore
from tensorrt_llm._torch.disaggregation.b10.endpoints import EndpointPool
from tensorrt_llm._torch.disaggregation.b10.kernels import _span_view
from tensorrt_llm._torch.disaggregation.b10.memory import (
    _BufferView,
    _DescArrayView,
    _TransferChunk,
)
from tensorrt_llm._torch.disaggregation.b10.planning import (
    _contiguous_desc_spans,
    _desc_arrays,
    _desc_view_from_arrays,
    _memory_desc_stats,
    _normalize_memory_descs,
    _reorder_desc_pairs_for_contiguity,
    _transfer_chunk_stats,
    _transfer_chunks_to_control,
    _validate_matching_desc_sizes,
)
from tensorrt_llm._torch.disaggregation.b10.pools import (
    _format_staging_pool_state,
    _ready_event_for_copy_events,
    _record_cuda_copy_events,
)
from tensorrt_llm._torch.disaggregation.b10.protocol import (
    _AM_HEADER_SIZE,
    _AM_KIND_CONTROL,
    _AM_KIND_DATA,
    _AM_KIND_READY,
    _AM_KIND_RESULT,
    _AM_RECEIVER_CALLBACK_INFO,
    _B10_PROTOCOL,
    _B10_PROTOCOL_VERSION,
    _FEATURE_PACKED_DESCS,
    _MAX_CHUNK_INDEX,
    _am_send_message,
    _pack_am_header,
    _request_id_from_sync_message,
)
from tensorrt_llm._torch.disaggregation.b10.state import (
    B10TransferStatus,
    _BufferCheckoutTracker,
    _CompletedTransferStatus,
    _FailedTransferStatus,
    _SendEndpointLease,
    _SendTransferPlan,
    _SourceReadyEvents,
    _TransferAbortHandle,
)
from tensorrt_llm._torch.disaggregation.b10.timings import TransferTrace, TransferTracer
from tensorrt_llm._torch.disaggregation.base.agent import TransferRequest, TransferStatus


def _enum_name(value: Any) -> str:
    if isinstance(value, str):
        return value
    name = getattr(value, "name", None)
    if name is not None:
        return str(name)
    text = str(value).rsplit(".", maxsplit=1)[-1]
    return text.split(":", maxsplit=1)[0].strip("<> ")


@dataclass(slots=True)
class _SendTransfer:
    """Mutable state owned by one outgoing WRITE."""

    plan: _SendTransferPlan
    deadline: _TransferDeadline
    abort_handle: _TransferAbortHandle
    cleanup_event: Optional[threading.Event]
    source_ready_events: Optional[_SourceReadyEvents]
    trace: TransferTrace
    lease: Optional[_SendEndpointLease] = None
    staging_tracker: _BufferCheckoutTracker = field(default_factory=_BufferCheckoutTracker)
    status: str = "unknown"
    admission_acquired: bool = False
    span_count: int = 0
    send_in_flight: int = 0
    # READY/RESULT arrive via the AM dispatcher; both futures are registered
    # before the control message is sent so a reply can never race its
    # waiter. Keyed by (transfer_id, endpoint_generation, kind) — a lease
    # refresh (new generation) re-registers them.
    ready_future: Optional[asyncio.Future] = None
    result_future: Optional[asyncio.Future] = None


class SendPipeline:
    def __init__(
        self,
        core: _AgentCore,
        copies: _CopyEngine,
        endpoints: EndpointPool,
        tracer: TransferTracer,
        dispatcher: B10AmDispatcher,
        *,
        validate_send_source: bool,
        send_admission_limit: int,
        send_admission_bypass_bytes: int,
    ):
        self._core = core
        self._copies = copies
        self._endpoints = endpoints
        self._tracer = tracer
        self._dispatcher = dispatcher
        self._local_worker_address: Optional[bytes] = None
        self._validate_send_source = validate_send_source
        self._send_admission = (
            asyncio.Semaphore(send_admission_limit) if send_admission_limit > 0 else None
        )
        self._send_admission_bypass_bytes = send_admission_bypass_bytes
        self._source_ready_events: dict[int, _SourceReadyEvents] = {}
        self._source_ready_events_lock = threading.Lock()

    def submit_transfer_requests(self, request: TransferRequest) -> TransferStatus:
        return self._submit_transfer_request(
            request,
            _normalize_memory_descs(request.src_descs),
            _normalize_memory_descs(request.dst_descs),
        )

    def submit_transfer_requests_with_desc_arrays(
        self,
        request: TransferRequest,
        src_arrays: tuple[np.ndarray, np.ndarray, int | np.ndarray],
        dst_arrays: tuple[np.ndarray, np.ndarray, int | np.ndarray],
    ) -> TransferStatus:
        """Submit with producer-supplied descriptor arrays (side-channel).

        `src_arrays` / `dst_arrays` are `(ptrs, sizes, device_ids)` tuples —
        parallel int64 arrays with a scalar or per-desc device — describing
        exactly the descriptors already inside `request.src_descs` /
        `request.dst_descs`. A producer that built the C++ MemoryDescs from
        these arrays (the native Sender) hands them over so ingestion skips
        the per-item nanobind property walk of `_normalize_memory_descs`
        (~1 ms/call at 3.6k descs). The arrays are trusted to match the
        request — even reading `len(MemoryDescs.descs)` materializes every
        nanobind wrapper — but src/dst column consistency is still
        validated (shape checks here, src-vs-dst count below).

        Discovered by callers via `getattr(agent, ..., None)`, mirroring
        `set_incoming_write_listener`; agents without this method (NIXL)
        take the `submit_transfer_requests` path unchanged.
        """
        return self._submit_transfer_request(
            request, _desc_view_from_arrays(*src_arrays), _desc_view_from_arrays(*dst_arrays)
        )

    def _submit_transfer_request(
        self, request: TransferRequest, src_descs: _DescArrayView, dst_descs: _DescArrayView
    ) -> TransferStatus:
        op_name = _enum_name(request.op)
        if op_name != "WRITE":
            raise NotImplementedError(f"B10 only supports WRITE, got {op_name}")

        if len(src_descs) != len(dst_descs):
            raise ValueError(
                f"B10 source/destination descriptor count mismatch: "
                f"{len(src_descs)} != {len(dst_descs)}"
            )
        if not src_descs:
            return _CompletedTransferStatus()
        src_type = _enum_name(request.src_descs.type)
        dst_type = _enum_name(request.dst_descs.type)
        request_id = _request_id_from_sync_message(request.sync_message)
        source_ready_events = None
        if src_type == "VRAM":
            if request_id is None:
                return _FailedTransferStatus("B10 missing request sync metadata for VRAM send")
            source_ready_events = self._get_source_ready_events(request_id)
            if source_ready_events is None:
                return _FailedTransferStatus(
                    f"B10 missing source-ready event for VRAM send request {request_id}"
                )
        transfer_id = self._core.transfer_ids.allocate()
        desc_count, total_bytes, max_desc_size = _memory_desc_stats(src_descs)
        self._tracer.debug(
            lambda: f"B10 transfer {transfer_id} submitted: "
            f"remote={request.remote_name} "
            f"src_type={src_type} "
            f"dst_type={dst_type} "
            f"descs={desc_count} total_bytes={total_bytes} "
            f"max_desc_size={max_desc_size}"
        )
        abort_handle = _TransferAbortHandle()
        cleanup_event = threading.Event()
        future = asyncio.run_coroutine_threadsafe(
            self._submit_write(
                request.remote_name,
                transfer_id,
                src_type,
                src_descs,
                dst_type,
                dst_descs,
                abort_handle,
                cleanup_event,
                request.sync_message,
                source_ready_events,
            ),
            self._core.loop,
        )
        return B10TransferStatus(
            future,
            transfer_id,
            self._core.transfer_ids,
            abort_handle,
            b10_net._status_wait_timeout_ms(self._core.transfer_timeout_s),
            cleanup_event=cleanup_event,
        )

    def record_source_ready_event(self, request_id: int, *, stream: Optional[Any] = None) -> None:
        stream_device = getattr(stream, "device", None)
        if stream_device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        elif isinstance(stream_device, int):
            device = torch.device("cuda", stream_device)
        else:
            device = torch.device(stream_device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if stream is None:
            stream = torch.cuda.current_stream(device)
        event = torch.cuda.Event()
        event.record(stream)
        with self._source_ready_events_lock:
            self._source_ready_events[int(request_id)] = [(device, event)]

    def discard_source_ready_event(self, request_id: int) -> None:
        with self._source_ready_events_lock:
            self._source_ready_events.pop(int(request_id), None)

    def _get_source_ready_events(self, request_id: int) -> Optional[_SourceReadyEvents]:
        with self._source_ready_events_lock:
            return self._source_ready_events.get(int(request_id))

    async def _send_buffer(self, endpoint: Any, buffer: Any, deadline: _TransferDeadline) -> None:
        # One DATA active message. `buffer` already carries the 32-byte AM
        # header written into its first bytes by `send_one`; identity rides
        # in-band, so there is no tag.
        if isinstance(buffer, torch.Tensor):
            if buffer.is_cuda and self._core.sync_cuda_before_transfer:
                torch.cuda.synchronize(buffer.device)
            elif not buffer.is_cuda:
                buffer = buffer.numpy()
        await _await_detached_with_timeout(
            endpoint.am_send(buffer, receiver_callback_info=_AM_RECEIVER_CALLBACK_INFO),
            deadline.remaining_s(),
        )

    def _build_send_transfer_plan(
        self,
        remote_name: str,
        transfer_id: int,
        src_type: str,
        src_descs: _DescArrayView,
        dst_type: str,
        dst_descs: _DescArrayView,
        sync_message: Optional[str] = None,
    ) -> _SendTransferPlan:
        if src_type != dst_type:
            raise ValueError(
                f"B10 source/destination memory types differ: {src_type} != {dst_type}"
            )
        _validate_matching_desc_sizes(src_descs, dst_descs)
        remote = self._endpoints._remote_agents.get(remote_name)
        if remote is None:
            raise KeyError(f"B10 remote agent is not loaded: {remote_name}")
        # Validated once per plan (not per control-send retry): the packed
        # int64 descriptor encoding is the only wire format, and the
        # descriptor feature list is the version check for it.
        if _FEATURE_PACKED_DESCS not in remote.features:
            raise RuntimeError(
                f"B10 peer '{remote_name}' does not advertise "
                f"{_FEATURE_PACKED_DESCS}; B10 no longer supports the legacy "
                f"descriptor encoding — upgrade the peer build"
            )

        # DATA payload shares its staging buffer with the 32-byte in-band AM
        # header, so chunks coalesce up to header-size less than the buffer.
        desc_order = _reorder_desc_pairs_for_contiguity(
            src_descs,
            dst_descs,
            self._core.staging_buffer_pool.buffer_size - _AM_HEADER_SIZE,
        )
        src_descs = desc_order.src_descs
        dst_descs = desc_order.dst_descs
        desc_count, total_bytes, max_desc_size = _memory_desc_stats(src_descs)
        transfer_chunks = desc_order.transfer_chunks
        wire_chunk_count, _, max_wire_chunk_size = _transfer_chunk_stats(transfer_chunks)
        if wire_chunk_count > _MAX_CHUNK_INDEX + 1:
            raise ValueError(f"B10 transfer has too many coalesced DATA chunks: {wire_chunk_count}")
        return _SendTransferPlan(
            remote_name=remote_name,
            remote=remote,
            transfer_id=transfer_id,
            src_type=src_type,
            src_descs=src_descs,
            dst_type=dst_type,
            dst_descs=dst_descs,
            transfer_chunks=transfer_chunks,
            desc_count=desc_count,
            total_bytes=total_bytes,
            max_desc_size=max_desc_size,
            wire_chunk_count=wire_chunk_count,
            max_wire_chunk_size=max_wire_chunk_size,
            desc_order_strategy=desc_order.strategy,
            src_span_count=desc_order.src_span_count,
            dst_span_count=desc_order.dst_span_count,
            sync_message=sync_message,
        )

    @staticmethod
    def _validate_send_source_residency(plan: _SendTransferPlan) -> None:
        if plan.src_type != "VRAM":
            return
        try:
            for chunk in plan.transfer_chunks:
                for span in _contiguous_desc_spans(plan.src_descs, chunk):
                    _span_view(plan.src_descs, span, plan.src_type)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"B10 source descriptor validation failed before DATA: "
                f"transfer_id={plan.transfer_id} src_type={plan.src_type} "
                f"error={type(exc).__name__}: {exc}"
            ) from exc

    def _get_local_worker_address(self) -> bytes:
        if self._local_worker_address is None:
            self._local_worker_address = bytes(self._endpoints._ucxx.get_worker_address())
        return self._local_worker_address

    def _make_send_control(
        self, plan: _SendTransferPlan, lease: _SendEndpointLease
    ) -> dict[str, Any]:
        control = {
            "protocol": _B10_PROTOCOL,
            "version": _B10_PROTOCOL_VERSION,
            "transfer_id": plan.transfer_id,
            "endpoint_generation": lease.endpoint_generation,
            "src_type": plan.src_type,
            "dst_type": plan.dst_type,
            # The receiver creates its reverse (READY/RESULT) endpoint from
            # this; self-contained so replies never depend on registration-
            # plane state. Stable per process, computed once.
            "src_worker_address": self._get_local_worker_address(),
        }
        # plan.dst_descs is the array-backed view carried from
        # _reorder_desc_pairs_for_contiguity, so the encoding below is
        # column reuse, not a per-desc walk. Packed encoding: one flat
        # (3, n) int64 buffer instead of n string-keyed dicts, so the
        # receiver does zero per-descriptor Python-object work. Both ends
        # run little-endian (x86_64/aarch64); '<i8' makes that explicit so
        # the bytes always match the receiver's frombuffer dtype. The peer's
        # packed_descs capability was validated at plan build.
        ptrs, sizes, device_ids = _desc_arrays(plan.dst_descs)
        control["dst_descs_packed"] = (
            np.stack((ptrs, sizes, device_ids)).astype("<i8", copy=False).tobytes()
        )
        control["transfer_chunks"] = _transfer_chunks_to_control(plan.transfer_chunks)
        if plan.sync_message is not None:
            control["sync_message"] = plan.sync_message
        return control

    def _discard_reply_futures(self, ctx: _SendTransfer, lease: _SendEndpointLease) -> None:
        self._dispatcher.discard_reply_future(
            ctx.plan.transfer_id, lease.endpoint_generation, _AM_KIND_READY
        )
        self._dispatcher.discard_reply_future(
            ctx.plan.transfer_id, lease.endpoint_generation, _AM_KIND_RESULT
        )
        ctx.ready_future = None
        ctx.result_future = None

    @staticmethod
    def _check_reply(payload: dict[str, Any], transfer_id: int, message_name: str) -> None:
        if int(payload.get("transfer_id", -1)) != transfer_id:
            raise RuntimeError(
                f"B10 {message_name} transfer_id mismatch: expected {transfer_id}, got {payload}"
            )
        if not payload.get("ok"):
            raise RuntimeError(
                f"B10 {message_name} failed for transfer {transfer_id}: {payload.get('error')}"
            )

    async def _send_control_and_wait_ready(
        self,
        ctx: _SendTransfer,
        lease: _SendEndpointLease,
    ) -> None:
        plan = ctx.plan
        # Register both reply futures before control leaves so neither reply
        # can race its waiter; RESULT is awaited later in _execute_write.
        ctx.ready_future = self._dispatcher.register_reply_future(
            plan.transfer_id, lease.endpoint_generation, _AM_KIND_READY
        )
        ctx.result_future = self._dispatcher.register_reply_future(
            plan.transfer_id, lease.endpoint_generation, _AM_KIND_RESULT
        )
        try:
            await _am_send_message(
                lease.endpoint,
                _AM_KIND_CONTROL,
                plan.transfer_id,
                lease.endpoint_generation,
                self._make_send_control(plan, lease),
                ctx.deadline.remaining_s(),
            )
            ctx.trace.debug(lambda: "control sent")
            ready = await _await_detached_with_timeout(ctx.ready_future, ctx.deadline.remaining_s())
            self._check_reply(ready, plan.transfer_id, "READY")
        except BaseException:
            self._discard_reply_futures(ctx, lease)
            raise
        ctx.trace.debug(lambda: "READY received")

    async def _send_control_and_wait_ready_with_retry(
        self,
        ctx: _SendTransfer,
        lease: _SendEndpointLease,
    ) -> _SendEndpointLease:
        plan = ctx.plan

        async def timed_control_ready(current_lease: _SendEndpointLease) -> None:
            # Sending CONTROL and awaiting READY needs no per-transfer setup:
            # message identity travels in the in-band header (endpoint-scoped)
            # and replies resolve through dispatcher futures keyed by
            # (transfer_id, generation, kind).
            with ctx.trace.measure_attempt("control_ready"):
                await self._send_control_and_wait_ready(ctx, current_lease)

        async def refresh_and_ready(current_lease: _SendEndpointLease) -> _SendEndpointLease:
            refreshed_lease = await self._endpoints._refresh_stale_send_endpoint(
                plan, current_lease, ctx.deadline, ctx.abort_handle
            )
            try:
                await timed_control_ready(refreshed_lease)
            except BaseException:
                self._endpoints._retire_send_endpoint(refreshed_lease, ctx.abort_handle)
                raise
            return refreshed_lease

        try:
            await timed_control_ready(lease)
        except Exception as exc:
            if not _is_retryable_endpoint_error(exc):
                raise
            logger.warning(
                f"B10 send transfer {plan.transfer_id} retrying stale "
                f"endpoint before DATA: remote={plan.remote_name} "
                f"slot_index={lease.slot_index} "
                f"endpoint_generation={lease.endpoint_generation} "
                f"error={type(exc).__name__}: {exc}"
            )
            lease = await refresh_and_ready(lease)
        return lease

    async def _send_transfer_chunks(self, ctx: _SendTransfer) -> None:
        plan = ctx.plan
        lease = ctx.lease
        assert lease is not None
        source_ready_waited_keys: set[tuple[str, int]] = set()

        async def send_one(chunk: _TransferChunk, chunk_idx: int) -> None:
            try:
                with ctx.trace.measure("staging_acquire"):
                    # The wire message is header + payload in one contiguous
                    # pinned buffer: 32-byte in-band AM header first, chunk
                    # payload after it (plan build already caps chunk size at
                    # buffer_size - header).
                    staging_view = await self._core._acquire_staging_buffer(
                        chunk.size + _AM_HEADER_SIZE, ctx.deadline
                    )
            except Exception as exc:
                logger.warning(
                    f"B10 send transfer {plan.transfer_id} failed to acquire "
                    f"staging buffer: remote={plan.remote_name} "
                    f"chunk_index={chunk_idx} chunk_size={chunk.size} "
                    f"data_chunks={plan.wire_chunk_count} "
                    f"error={type(exc).__name__}: {exc} "
                    f"{_format_staging_pool_state(self._core.staging_buffer_pool)}"
                )
                raise
            ctx.staging_tracker.track(staging_view)
            header = _pack_am_header(
                _AM_KIND_DATA, plan.transfer_id, chunk_idx, lease.endpoint_generation, chunk.size
            )
            # CPU write into pinned host memory; trivially cheap vs the chunk.
            staging_view.buffer[:_AM_HEADER_SIZE].copy_(
                torch.frombuffer(bytearray(header), dtype=torch.uint8)
            )
            payload_view = staging_view.buffer[_AM_HEADER_SIZE:]
            try:
                with ctx.trace.measure("src_copy"):
                    # Keep fallback source span views alive until their D2H
                    # copies complete.
                    copy_lifetime_refs: list[_BufferView] = []
                    span_count, copy_devices = self._copies.copy_chunk(
                        plan.src_descs,
                        chunk,
                        plan.src_type,
                        payload_view,
                        copy_from_staging=False,
                        lifetime_refs=copy_lifetime_refs,
                        source_ready_events=ctx.source_ready_events,
                        source_ready_waited_keys=source_ready_waited_keys,
                    )
                    ctx.span_count += span_count
                with ctx.trace.measure("cuda_event_record"):
                    copy_events = _record_cuda_copy_events(
                        copy_devices, self._core.cuda_copy_streams.stream_for
                    )
                    staging_view.ready_event = _ready_event_for_copy_events(copy_events)
                if copy_events:
                    with ctx.trace.measure("copy_event_wait"):
                        await self._core._wait_copy_events_async(copy_events, ctx.deadline)
                copy_lifetime_refs.clear()

                # The wire send: everything above staged this chunk into
                # pinned host memory after the in-band header; this ships
                # header+payload as one active message.
                with ctx.trace.measure("ucxx_send"):
                    await self._send_buffer(lease.endpoint, staging_view.buffer, ctx.deadline)

                with ctx.trace.measure("staging_release"):
                    self._core._release_staging_buffers([staging_view])
                ctx.staging_tracker.untrack(staging_view)
            except Exception as exc:
                logger.warning(
                    f"B10 send transfer {plan.transfer_id} chunk failed: "
                    f"remote={plan.remote_name} chunk_index={chunk_idx} "
                    f"chunk_size={chunk.size} "
                    f"data_chunks={plan.wire_chunk_count} "
                    f"error={type(exc).__name__}: {exc}"
                )
                raise

        send_pool_buffers = self._core.staging_buffer_pool.num_buffers
        send_in_flight = max(1, min(self._core.max_in_flight_ops, send_pool_buffers))
        ctx.send_in_flight = send_in_flight
        self._endpoints._bind_send_abort_handle(ctx.abort_handle, lease, abort_endpoint=False)
        with ctx.trace.measure("data_phase_wall"):
            await _run_limited(
                [
                    (lambda chunk=chunk, chunk_idx=chunk_idx: send_one(chunk, chunk_idx))
                    for chunk_idx, chunk in enumerate(plan.transfer_chunks)
                ],
                transfer_id=plan.transfer_id,
                phase="send DATA",
                max_in_flight=send_in_flight,
            )
        ctx.trace.debug(lambda: "DATA complete")

    async def _submit_write(
        self,
        remote_name: str,
        transfer_id: int,
        src_type: str,
        src_descs: _DescArrayView,
        dst_type: str,
        dst_descs: _DescArrayView,
        abort_handle: _TransferAbortHandle,
        cleanup_event: Optional[threading.Event] = None,
        sync_message: Optional[str] = None,
        source_ready_events: Optional[_SourceReadyEvents] = None,
    ) -> bool:
        trace = self._tracer.start("send", transfer_id)
        try:
            abort_handle.bind_current_task()
            with trace.measure("plan_build"):
                plan = self._build_send_transfer_plan(
                    remote_name, transfer_id, src_type, src_descs, dst_type, dst_descs, sync_message
                )
                if self._validate_send_source:
                    self._validate_send_source_residency(plan)
        except BaseException:
            # The transfer id itself is handled by B10TransferStatus: the
            # raise resolves the submit future, and _on_done quarantines the
            # id on any failure.
            if cleanup_event is not None:
                cleanup_event.set()
            raise

        ctx = _SendTransfer(
            plan=plan,
            deadline=_TransferDeadline(self._core.transfer_timeout_s),
            abort_handle=abort_handle,
            cleanup_event=cleanup_event,
            source_ready_events=source_ready_events,
            trace=trace,
        )
        try:
            await self._execute_write(ctx)
            ctx.status = "success"
            return True
        except BaseException as exc:
            self._fail_write(ctx, exc)
            raise
        finally:
            self._finish_write(ctx)

    async def _execute_write(self, ctx: _SendTransfer) -> None:
        plan = ctx.plan
        if (
            self._send_admission is not None
            and plan.total_bytes >= self._send_admission_bypass_bytes
        ):
            with ctx.trace.measure("admission_wait"):
                await _acquire_with_timeout(self._send_admission, ctx.deadline.remaining_s())
                ctx.admission_acquired = True

        slot_index, slot = self._endpoints._get_endpoint_slot(plan.remote_name, plan.transfer_id)
        slot_lock_wait = ctx.trace.timer("slot_lock_wait")
        async with _lock_with_timeout(slot.transfer_lock, ctx.deadline.remaining_s()):
            slot_lock_wait.stop()
            with ctx.trace.measure("lease"):
                ctx.lease = await self._endpoints._lease_send_endpoint(
                    plan, slot_index, slot, ctx.deadline, ctx.abort_handle
                )
            lease = ctx.lease
            ctx.trace.debug(
                lambda: f"begin: "
                f"remote={plan.remote_name} remote_host={plan.remote.host} "
                f"remote_port={plan.remote.port} slot_index={slot_index} "
                f"endpoint_generation={lease.endpoint_generation} "
                f"dst_type={plan.dst_type} descs={plan.desc_count} "
                f"data_chunks={plan.wire_chunk_count} total_bytes={plan.total_bytes} "
                f"max_desc_size={plan.max_desc_size} "
                f"max_data_chunk_size={plan.max_wire_chunk_size} "
                f"desc_order={plan.desc_order_strategy} "
                f"src_spans={plan.src_span_count} dst_spans={plan.dst_span_count} "
                f"max_in_flight_ops={self._core.max_in_flight_ops}"
            )
            ctx.lease = await self._send_control_and_wait_ready_with_retry(ctx, lease)
            await self._send_transfer_chunks(ctx)
            lease = ctx.lease
            with ctx.trace.measure("result_recv"):
                result = await _await_detached_with_timeout(
                    ctx.result_future, ctx.deadline.remaining_s()
                )
                self._check_reply(result, plan.transfer_id, "RESULT")
            ctx.trace.debug(lambda: "RESULT received: ok=True")

    def _fail_write(self, ctx: _SendTransfer, exc: BaseException) -> None:
        cancelled = isinstance(exc, asyncio.CancelledError)
        ctx.status = "cancelled" if cancelled else "failed"
        checked_out_views = ctx.staging_tracker.take_all()
        self._core._quarantine_staging_buffers(ctx.plan.transfer_id, checked_out_views, "send")
        lease = ctx.lease
        if lease is None:
            return
        plan = ctx.plan
        failure_detail = (
            ""
            if cancelled
            else f"max_desc_size={plan.max_desc_size} "
            f"max_data_chunk_size={plan.max_wire_chunk_size} "
        )
        error_detail = "" if cancelled else f" error={type(exc).__name__}: {exc}"
        # Say which kind of failure this is: a local NIC that went down (what
        # NIC failover handles) or a peer that stopped answering (which it
        # cannot). Both otherwise surface as the same timeout/endpoint error.
        cause_detail = "" if cancelled else f" {b10_net.classify_transfer_failure_cause()}"
        logger.warning(
            f"B10 send transfer {plan.transfer_id} {ctx.status}: "
            f"remote={plan.remote_name} slot_index={lease.slot_index} "
            f"endpoint_generation={lease.endpoint_generation} "
            f"descs={plan.desc_count} data_chunks={plan.wire_chunk_count} "
            f"total_bytes={plan.total_bytes} {failure_detail}"
            f"checked_out_staging_buffers={len(checked_out_views)}"
            f"{error_detail}{cause_detail}"
        )
        self._endpoints._retire_send_endpoint(
            lease,
            ctx.abort_handle,
            abort_endpoint=(not cancelled and not isinstance(exc, TimeoutError)),
        )

    def _finish_write(self, ctx: _SendTransfer) -> None:
        if ctx.admission_acquired:
            self._send_admission.release()
        # Drop any reply future still registered (no-op after clean RESULT,
        # which pops its future on delivery); a reply landing later is then
        # dropped by the dispatcher as stale.
        if ctx.lease is not None:
            self._discard_reply_futures(ctx, ctx.lease)
            ctx.abort_handle.unbind_endpoint(ctx.lease.endpoint)
        self._tracer.log_send(
            ctx.trace,
            ctx.plan,
            ctx.status,
            span_count=ctx.span_count,
            send_in_flight=ctx.send_in_flight,
        )
        if ctx.cleanup_event is not None:
            ctx.cleanup_event.set()
