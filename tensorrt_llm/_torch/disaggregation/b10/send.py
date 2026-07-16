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
"""Send side of the B10 UCXX transfer agent: outgoing WRITEs.

`SendPipeline` is the send collaborator constructed by
`B10CacheTransferAgent.__init__`.

Constructor-injected:

- ``core`` (`_AgentCore`: loop, transfer ids, tag registry, staging pool
  and its acquire/release helpers, copy streams, config-derived knobs)
- ``endpoints`` (`EndpointPool`: slot lookup, leasing, retirement, stale
  refresh, abort arming; peer descriptor registry)
- ``trace`` (`TransferTrace`: transfer traces and send timing logs)
- ``validate_send_source`` / ``send_admission_limit`` /
  ``send_admission_bypass_bytes`` (send knobs from B10AgentConfig; the
  admission semaphore is built here from the limit)
- ``run_limited`` / ``reserve_message_tags`` /
  ``request_id_from_sync_message`` (agent-shell helpers; injected
  references)
- ``copy_chunk_between_staging_and_descs`` /
  ``quarantine_staging_buffers`` / ``single_cuda_device_for_spans``
  (recv-pipeline helpers shared with the send path; injected so the two
  pipelines never hold each other)

Own state:

- ``_source_ready_events`` guarded by ``_source_ready_events_lock``
  (request id -> recorded CUDA events gating VRAM sends)
- ``_send_admission`` (``None`` when admission gating is disabled)
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, Callable, Hashable, Optional

import numpy as np
import torch

from tensorrt_llm import logger
from tensorrt_llm._torch.disaggregation.b10 import memory as b10_memory
from tensorrt_llm._torch.disaggregation.b10 import net as b10_net
from tensorrt_llm._torch.disaggregation.b10 import protocol as b10_protocol
from tensorrt_llm._torch.disaggregation.b10.async_utils import (
    _await_with_timeout,
    _is_retryable_endpoint_error,
    _TransferDeadline,
)
from tensorrt_llm._torch.disaggregation.b10.core import _AgentCore
from tensorrt_llm._torch.disaggregation.b10.endpoints import EndpointPool
from tensorrt_llm._torch.disaggregation.b10.kernels import _scatter_kernels_available, _span_view
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
    _DEFAULT_SEND_GATHER_MIN_SPANS,
    _format_staging_pool_state,
    _record_cuda_copy_events,
)
from tensorrt_llm._torch.disaggregation.b10.protocol import (
    _B10_PROTOCOL,
    _B10_PROTOCOL_VERSION,
    _BOOTSTRAP_CONTROL_TAG,
    _FEATURE_PACKED_DESCS,
    _MAX_CHUNK_INDEX,
    _data_tag,
    _ready_tag,
    _recv_reply,
    _result_tag,
    _send_obj,
)
from tensorrt_llm._torch.disaggregation.b10.state import (
    B10TransferStatus,
    _CompletedTransferStatus,
    _FailedTransferStatus,
    _SendEndpointLease,
    _SendTransferPlan,
    _SourceReadyEvents,
    _StagingCheckoutTracker,
    _TransferAbortHandle,
)
from tensorrt_llm._torch.disaggregation.b10.timings import TransferTrace
from tensorrt_llm._torch.disaggregation.base.agent import TransferRequest, TransferStatus


def _enum_name(value: Any) -> str:
    if isinstance(value, str):
        return value
    name = getattr(value, "name", None)
    if name is not None:
        return str(name)
    text = str(value).rsplit(".", maxsplit=1)[-1]
    return text.split(":", maxsplit=1)[0].strip("<> ")


def _raise_if_cancel_requested(abort_handle: _TransferAbortHandle) -> None:
    # CancelledError is a BaseException: cancellation deliberately bypasses
    # the per-chunk `except Exception` warning blocks and reaches the
    # transfer-level failure funnel without error-log noise.
    if abort_handle.is_cancel_requested():
        raise asyncio.CancelledError


class SendPipeline:
    def __init__(
        self,
        core: _AgentCore,
        endpoints: EndpointPool,
        trace: TransferTrace,
        *,
        validate_send_source: bool,
        send_admission_limit: int,
        send_admission_bypass_bytes: int,
        run_limited: Callable[..., Any],
        reserve_message_tags: Callable[..., None],
        request_id_from_sync_message: Callable[[Optional[str]], Optional[int]],
        copy_chunk_between_staging_and_descs: Callable[..., Any],
        quarantine_staging_buffers: Callable[..., None],
        single_cuda_device_for_spans: Callable[..., Optional[torch.device]],
    ):
        self._core = core
        self._endpoints = endpoints
        self._trace = trace
        self._validate_send_source = validate_send_source
        self._send_admission = (
            asyncio.Semaphore(send_admission_limit) if send_admission_limit > 0 else None
        )
        self._send_admission_bypass_bytes = send_admission_bypass_bytes
        self._run_limited = run_limited
        self._reserve_message_tags = reserve_message_tags
        self._request_id_from_sync_message = request_id_from_sync_message
        self._copy_chunk_between_staging_and_descs = copy_chunk_between_staging_and_descs
        self._quarantine_staging_buffers = quarantine_staging_buffers
        self._single_cuda_device_for_spans = single_cuda_device_for_spans
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
        request_id = self._request_id_from_sync_message(request.sync_message)
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
        self._trace._trace_transfer(
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
            tag_registry=self._core.tag_registry,
            tag_owner=("send", transfer_id),
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

    async def _send_buffer(
        self, endpoint: Any, buffer: Any, tag: int, deadline: _TransferDeadline
    ) -> None:
        if isinstance(buffer, torch.Tensor):
            if buffer.is_cuda and self._core.sync_cuda_before_transfer:
                torch.cuda.synchronize(buffer.device)
            elif not buffer.is_cuda:
                buffer = buffer.numpy()
        await _await_with_timeout(
            endpoint.send(buffer, tag=tag), deadline.remaining_s(), cancel_on_timeout=False
        )

    def _send_gather_device_for_chunk(
        self,
        descs: _DescArrayView,
        memory_type: str,
        staging_buffer: Any,
        spans: b10_memory._SpanArrays,
    ) -> Optional[torch.device]:
        """Device for the send-side single-kernel gather, or None for the loop.

        Eligible when the chunk's sources are VRAM spans on one CUDA device,
        the chunk is fragmented enough for per-span dispatch overhead to
        matter, and the staging buffer is a pinned host tensor (UVA
        device-addressable, so the gather kernel stores into it directly).
        Sources are read-only, so unlike the recv scratch path no overlap
        check is needed.
        """
        if memory_type != "VRAM":
            return None
        if len(spans) < _DEFAULT_SEND_GATHER_MIN_SPANS:
            return None
        if (
            not isinstance(staging_buffer, torch.Tensor)
            or staging_buffer.device.type != "cpu"
            or not staging_buffer.is_pinned()
        ):
            return None
        if not _scatter_kernels_available():
            # Triton missing/broken: route to the per-span copy loop (which
            # is what HEAD ran) instead of failing the send.
            return None
        return self._single_cuda_device_for_spans(descs, spans)

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

        desc_order = _reorder_desc_pairs_for_contiguity(
            src_descs, dst_descs, self._core.staging_buffer_pool.buffer_size
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

    @staticmethod
    def _make_send_control(plan: _SendTransferPlan, lease: _SendEndpointLease) -> dict[str, Any]:
        control = {
            "protocol": _B10_PROTOCOL,
            "version": _B10_PROTOCOL_VERSION,
            "transfer_id": plan.transfer_id,
            "endpoint_generation": lease.endpoint_generation,
            "tag_domain": lease.tag_domain,
            "src_type": plan.src_type,
            "dst_type": plan.dst_type,
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

    async def _send_control_and_wait_ready(
        self,
        plan: _SendTransferPlan,
        lease: _SendEndpointLease,
        deadline: _TransferDeadline,
    ) -> None:
        ready_tag = _ready_tag(plan.transfer_id, lease.endpoint_generation, lease.tag_domain)
        result_tag = _result_tag(plan.transfer_id, lease.endpoint_generation, lease.tag_domain)
        await _send_obj(
            lease.endpoint,
            self._make_send_control(plan, lease),
            _BOOTSTRAP_CONTROL_TAG,
            deadline.remaining_s(),
        )
        self._trace._trace_transfer(
            lambda: f"B10 send transfer {plan.transfer_id} control sent: "
            f"ready_tag={ready_tag} result_tag={result_tag}"
        )
        await _recv_reply(
            lease.endpoint, plan.transfer_id, ready_tag, "READY", deadline.remaining_s()
        )
        self._trace._trace_transfer(lambda: f"B10 send transfer {plan.transfer_id} READY received")

    async def _send_control_and_wait_ready_with_retry(
        self,
        plan: _SendTransferPlan,
        lease: _SendEndpointLease,
        deadline: _TransferDeadline,
        abort_handle: _TransferAbortHandle,
        timings: dict[str, float],
        tag_owner: Hashable,
    ) -> _SendEndpointLease:
        async def timed_control_ready(current_lease: _SendEndpointLease) -> None:
            control_ready_start_s = time.perf_counter()
            reserved_tags = False
            try:
                self._reserve_message_tags(
                    tag_owner,
                    plan.transfer_id,
                    current_lease.endpoint_generation,
                    current_lease.tag_domain,
                    plan.wire_chunk_count,
                )
                reserved_tags = True
                await self._send_control_and_wait_ready(plan, current_lease, deadline)
            except Exception:
                if reserved_tags:
                    self._core.tag_registry.quarantine(tag_owner)
                raise
            finally:
                self._trace._add_elapsed_ms(timings, "control_ready_ms", control_ready_start_s)

        async def refresh_and_ready(current_lease: _SendEndpointLease) -> _SendEndpointLease:
            refreshed_lease = await self._endpoints._refresh_stale_send_endpoint(
                plan, current_lease, deadline, abort_handle
            )
            try:
                await timed_control_ready(refreshed_lease)
            except BaseException:
                self._endpoints._retire_send_endpoint(refreshed_lease, abort_handle)
                raise
            return refreshed_lease

        try:
            await timed_control_ready(lease)
        except b10_protocol.B10TagCollisionError as exc:
            logger.warning(
                f"B10 send transfer {plan.transfer_id} retrying tag "
                f"collision before DATA: remote={plan.remote_name} "
                f"slot_index={lease.slot_index} "
                f"endpoint_generation={lease.endpoint_generation} "
                f"error={type(exc).__name__}: {exc}"
            )
            lease = await refresh_and_ready(lease)
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

    async def _send_transfer_chunks(
        self,
        plan: _SendTransferPlan,
        lease: _SendEndpointLease,
        deadline: _TransferDeadline,
        staging_tracker: _StagingCheckoutTracker,
        abort_handle: _TransferAbortHandle,
        timings: dict[str, float],
        counts: dict[str, int],
        source_ready_events: Optional[_SourceReadyEvents],
    ) -> None:
        source_ready_waited_keys: set[tuple[str, int]] = set()

        async def send_one(chunk: _TransferChunk, idx: int) -> None:
            _raise_if_cancel_requested(abort_handle)
            try:
                with self._trace.span(timings, "staging_acquire_ms"):
                    staging_view = await self._core._acquire_staging_buffer(chunk.size, deadline)
            except Exception as exc:
                logger.warning(
                    f"B10 send transfer {plan.transfer_id} failed to acquire "
                    f"staging buffer: remote={plan.remote_name} "
                    f"chunk_index={idx} chunk_size={chunk.size} "
                    f"data_chunks={plan.wire_chunk_count} "
                    f"error={type(exc).__name__}: {exc} "
                    f"{_format_staging_pool_state(self._core.staging_buffer_pool)}"
                )
                raise
            staging_tracker.track(staging_view)
            _raise_if_cancel_requested(abort_handle)
            data_tag = _data_tag(plan.transfer_id, idx, lease.endpoint_generation, lease.tag_domain)
            try:
                with self._trace.span(timings, "src_copy_ms"):
                    # Holds the gather kernel's metadata tensors (and any span
                    # views) until the copy events recorded on the copy stream
                    # are awaited below; on failure the refs drop early, which
                    # record_stream on the device metadata and the pinned-host
                    # CachingHostAllocator events backstop (see
                    # _upload_scatter_metadata_adhoc).
                    copy_lifetime_refs: list[_BufferView] = []
                    span_count, copy_devices = self._copy_chunk_between_staging_and_descs(
                        plan.src_descs,
                        chunk,
                        plan.src_type,
                        staging_view.buffer,
                        copy_from_staging=False,
                        lifetime_refs=copy_lifetime_refs,
                        staging_view=staging_view,
                        source_ready_events=source_ready_events,
                        source_ready_waited_keys=source_ready_waited_keys,
                    )
                    counts["spans"] += span_count
                with self._trace.span(timings, "cuda_event_record_ms"):
                    copy_events = _record_cuda_copy_events(
                        copy_devices, self._core.cuda_copy_streams.stream_for
                    )
                if copy_events:
                    with self._trace.span(timings, "copy_event_wait_ms"):
                        await self._core._wait_copy_events_async(copy_events, deadline)
                copy_lifetime_refs.clear()
                _raise_if_cancel_requested(abort_handle)

                # The wire send: everything above staged this chunk into
                # pinned host memory; this ships it to the peer.
                with self._trace.span(timings, "ucxx_send_ms"):
                    await self._send_buffer(lease.endpoint, staging_view.buffer, data_tag, deadline)

                _raise_if_cancel_requested(abort_handle)
                with self._trace.span(timings, "staging_release_ms"):
                    self._core._release_staging_buffers([staging_view])
                staging_tracker.untrack(staging_view)
            except Exception as exc:
                logger.warning(
                    f"B10 send transfer {plan.transfer_id} chunk failed: "
                    f"remote={plan.remote_name} chunk_index={idx} "
                    f"chunk_size={chunk.size} "
                    f"data_chunks={plan.wire_chunk_count} data_tag={data_tag} "
                    f"error={type(exc).__name__}: {exc}"
                )
                raise

        send_pool_buffers = self._core.staging_buffer_pool.num_buffers
        send_in_flight = max(1, min(self._core.max_in_flight_ops, send_pool_buffers))
        counts["send_in_flight"] = send_in_flight
        data_phase_start_s = time.perf_counter()
        self._endpoints._arm_send_abort_handle(abort_handle, lease, abort_endpoint=False)
        await self._run_limited(
            [
                (lambda chunk=chunk, idx=idx: send_one(chunk, idx))
                for idx, chunk in enumerate(plan.transfer_chunks)
            ],
            transfer_id=plan.transfer_id,
            phase="send DATA",
            max_in_flight=send_in_flight,
        )
        self._trace._add_elapsed_ms(timings, "data_phase_wall_ms", data_phase_start_s)
        self._trace._trace_transfer(lambda: f"B10 send transfer {plan.transfer_id} DATA complete")

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
        # Failure funnel: nothing in this body releases resources on error.
        # The BaseException handler in the data phase sweeps every staging
        # view still checked out (staging_tracker.take_all) into quarantine
        # and retires the endpoint lease; the outer finally settles the
        # admission permit and releases (success) or quarantines (failure)
        # the transfer's message tags. See DESIGN.md "Timeout and failure
        # handling".
        lease: Optional[_SendEndpointLease] = None
        send_total_start_s = time.perf_counter()
        send_timings: dict[str, float] = {}
        send_counts = {"spans": 0, "send_in_flight": 0}
        send_status = "unknown"
        send_admission_acquired = False
        tag_owner = ("send", transfer_id)
        try:
            with self._trace.span(send_timings, "plan_build_ms"):
                plan = self._build_send_transfer_plan(
                    remote_name, transfer_id, src_type, src_descs, dst_type, dst_descs, sync_message
                )
                if self._validate_send_source:
                    self._validate_send_source_residency(plan)
            deadline = _TransferDeadline(self._core.transfer_timeout_s)
            if (
                self._send_admission is not None
                and plan.total_bytes >= self._send_admission_bypass_bytes
            ):
                with self._trace.span(send_timings, "admission_wait_ms"):
                    await _await_with_timeout(
                        self._send_admission.acquire(), deadline.remaining_s()
                    )
                    send_admission_acquired = True
            slot_index, slot = self._endpoints._get_endpoint_slot(remote_name, transfer_id)
            slot_lock_wait_start_s = time.perf_counter()
            async with slot.lock:
                self._trace._add_elapsed_ms(
                    send_timings, "slot_lock_wait_ms", slot_lock_wait_start_s
                )
                with self._trace.span(send_timings, "lease_ms"):
                    lease = await self._endpoints._lease_send_endpoint(
                        plan, slot_index, slot, deadline, abort_handle
                    )
                self._trace._trace_transfer(
                    lambda: f"B10 send transfer {transfer_id} begin: "
                    f"remote={remote_name} remote_host={plan.remote.host} "
                    f"remote_port={plan.remote.port} slot_index={slot_index} "
                    f"endpoint_generation={lease.endpoint_generation} "
                    f"tag_domain={lease.tag_domain} src_type={src_type} "
                    f"dst_type={dst_type} descs={plan.desc_count} "
                    f"data_chunks={plan.wire_chunk_count} "
                    f"total_bytes={plan.total_bytes} "
                    f"max_desc_size={plan.max_desc_size} "
                    f"max_data_chunk_size={plan.max_wire_chunk_size} "
                    f"desc_order={plan.desc_order_strategy} "
                    f"src_spans={plan.src_span_count} "
                    f"dst_spans={plan.dst_span_count} "
                    f"max_in_flight_ops={self._core.max_in_flight_ops}"
                )
                staging_tracker = _StagingCheckoutTracker()
                try:
                    lease = await self._send_control_and_wait_ready_with_retry(
                        plan, lease, deadline, abort_handle, send_timings, tag_owner
                    )
                    _raise_if_cancel_requested(abort_handle)
                    await self._send_transfer_chunks(
                        plan,
                        lease,
                        deadline,
                        staging_tracker,
                        abort_handle,
                        send_timings,
                        send_counts,
                        source_ready_events,
                    )
                    _raise_if_cancel_requested(abort_handle)
                    with self._trace.span(send_timings, "result_recv_ms"):
                        await _recv_reply(
                            lease.endpoint,
                            transfer_id,
                            _result_tag(transfer_id, lease.endpoint_generation, lease.tag_domain),
                            "RESULT",
                            deadline.remaining_s(),
                        )
                    _raise_if_cancel_requested(abort_handle)
                    self._trace._trace_transfer(
                        lambda: f"B10 send transfer {transfer_id} RESULT received: ok=True"
                    )
                    send_status = "success"
                    return True
                except BaseException as exc:
                    cancelled = isinstance(exc, asyncio.CancelledError)
                    send_status = "cancelled" if cancelled else "failed"
                    checked_out_views = staging_tracker.take_all()
                    self._quarantine_staging_buffers(transfer_id, checked_out_views, "send")
                    failure_detail = (
                        ""
                        if cancelled
                        else f"max_desc_size={plan.max_desc_size} "
                        f"max_data_chunk_size={plan.max_wire_chunk_size} "
                    )
                    error_detail = "" if cancelled else f" error={type(exc).__name__}: {exc}"
                    logger.warning(
                        f"B10 send transfer {transfer_id} {send_status}: "
                        f"remote={remote_name} slot_index={slot_index} "
                        f"endpoint_generation={lease.endpoint_generation} "
                        f"descs={plan.desc_count} "
                        f"data_chunks={plan.wire_chunk_count} "
                        f"total_bytes={plan.total_bytes} "
                        f"{failure_detail}"
                        f"checked_out_staging_buffers="
                        f"{len(checked_out_views)}"
                        f"{error_detail}"
                    )
                    self._endpoints._retire_send_endpoint(
                        lease,
                        abort_handle,
                        abort_endpoint=(not cancelled and not isinstance(exc, TimeoutError)),
                    )
                    raise
                finally:
                    self._trace._log_send_transfer_timings(
                        plan,
                        send_status,
                        send_timings,
                        send_total_start_s,
                        span_count=send_counts["spans"],
                        send_in_flight=send_counts["send_in_flight"],
                    )
        finally:
            if send_admission_acquired:
                self._send_admission.release()
            if send_status == "success":
                self._core.tag_registry.release(tag_owner)
            elif send_status in ("failed", "cancelled"):
                self._core.tag_registry.quarantine(tag_owner)
            if lease is not None:
                abort_handle.clear_endpoint(lease.endpoint)
            if cleanup_event is not None:
                cleanup_event.set()
