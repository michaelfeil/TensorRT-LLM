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
"""Worker-scoped active message dispatcher for the B10 AM wire plane.

One permanent callback is registered per UCXX worker via
``ucxx.register_am_receiver_callback``; sequential B10 agents attach their
dispatcher behind it because UCXX does not support unregistering callbacks.
Every B10 message (control / READY / RESULT / DATA) arrives through that
auto-re-arming callback; this class parses the 32-byte in-band header
(`protocol._AmHeader`) and routes on the active agent event loop:

- ``control``      -> the control handler installed by the receive pipeline
                      (spawns one incoming-write task per transfer)
- ``DATA``         -> the per-(source endpoint, transfer_id,
                      endpoint_generation) chunk sink registered by the
                      active receive transfer
- ``READY/RESULT`` -> the per-(transfer_id, endpoint_generation, kind) reply
                      future registered by the active send transfer

A message with no registered target is stale by definition (its transfer
failed, timed out, or never existed) and is dropped with a log — the AM
analog of the tag-quarantine discipline. Endpoint retirement already kills
the channel itself; this drop handles the narrower "late message from a
failed transfer on a still-live endpoint" case.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Callable, Optional

from tensorrt_llm._torch.disaggregation.b10.protocol import (
    _AM_HEADER_SIZE,
    _AM_KIND_CONTROL,
    _AM_KIND_DATA,
    _AM_KIND_READY,
    _AM_KIND_RESULT,
    _AM_RECEIVER_ID,
    _AM_RECEIVER_OWNER,
    _AmHeader,
    _unpack_am_header,
    _unpack_message,
)

logger = logging.getLogger(__name__)


def _describe_am_buffer(view: Optional[memoryview]) -> str:
    """Describe a message we could not route: its size and its leading bytes.

    A header that fails to parse is nearly always a buffer something else
    wrote into, so the raw bytes are the evidence. They say which: a previous
    message's header means the buffer was recycled while still in use, and
    payload-looking bytes mean it was being written by another transfer.
    """
    if view is None:
        return "buffer=<unavailable>"
    prefix = bytes(view[:_AM_HEADER_SIZE])
    return f"buffer_len={len(view)} leading_bytes={prefix.hex()}"


class _AmCallbackSlot:
    """Permanent UCXX callback that forwards to the active B10 agent."""

    def __init__(self):
        self.dispatcher: Optional[B10AmDispatcher] = None

    def __call__(self, request: Any, ep_handle: int) -> None:
        dispatcher = self.dispatcher
        if dispatcher is not None:
            dispatcher._on_am(request, ep_handle)


# UCXX callbacks live for the worker's lifetime and cannot be unregistered.
# Keep one permanent trampoline per process-global worker, while sequential
# B10 agents attach and detach their dispatcher behind it.
_AM_CALLBACK_SLOTS: dict[int, _AmCallbackSlot] = {}
_AM_CALLBACK_SLOTS_LOCK = threading.Lock()

# sink(chunk_index, payload_view) — payload_view is a view over the
# ucxx-allocated receive buffer and transitively keeps that buffer alive
# (ucxx hands the numpy array ownership of the malloc'd data), so the sink
# may retain the view past return without a copy: B10's recv stores it in a
# per-chunk future and copies into pinned staging later. The buffer is freed
# once the last reference to the view drops.
DataSink = Callable[[int, Any], None]
# handler(ep_handle, header, payload_dict) — must not block; spawns its own task.
ControlHandler = Callable[[int, _AmHeader, dict[str, Any]], None]


class B10AmDispatcher:
    """Worker-scoped router for the B10 AM wire plane.

    Receives messages from the UCXX worker's permanent callback and fans them
    out on the agent event loop by kind: CONTROL to the recv pipeline's
    handler, DATA to the active transfer's chunk sink, and READY/RESULT to the
    send pipeline's reply futures. Messages with no live target are dropped
    as stale. See the module docstring for the full routing and staleness
    contract.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop
        self._callback_slot: Optional[_AmCallbackSlot] = None
        self._control_handler: Optional[ControlHandler] = None
        self._data_sinks: dict[tuple[int, int, int], DataSink] = {}
        self._reply_futures: dict[tuple[int, int, int], asyncio.Future] = {}

    def attach(self, ucxx: Any) -> None:
        """Attach to the worker's permanent AM callback trampoline."""
        worker_key = int(ucxx.get_ucxx_worker())
        with _AM_CALLBACK_SLOTS_LOCK:
            slot = _AM_CALLBACK_SLOTS.get(worker_key)
            if slot is None:
                slot = _AmCallbackSlot()
                ucxx.register_am_receiver_callback(
                    _AM_RECEIVER_OWNER,
                    _AM_RECEIVER_ID,
                    slot,
                )
                _AM_CALLBACK_SLOTS[worker_key] = slot
            active = slot.dispatcher
            if active is not None and active is not self and not active._loop.is_closed():
                raise RuntimeError("B10 AM callback is already attached to a live agent")
            slot.dispatcher = self
            self._callback_slot = slot

    def detach(self) -> None:
        """Stop forwarding AMs to this agent; the UCXX callback remains."""
        slot = self._callback_slot
        if slot is None:
            return
        with _AM_CALLBACK_SLOTS_LOCK:
            if slot.dispatcher is self:
                slot.dispatcher = None
            self._callback_slot = None

    def set_control_handler(self, handler: ControlHandler) -> None:
        self._control_handler = handler

    # ---- send-side reply plumbing (loop thread only) ----

    def register_reply_future(
        self,
        transfer_id: int,
        endpoint_generation: int,
        kind: int,
    ) -> asyncio.Future:
        """Create the future a READY/RESULT resolves. Must be registered
        before the control message is sent so the reply cannot race it."""
        key = (transfer_id, endpoint_generation, kind)
        if key in self._reply_futures:
            raise RuntimeError(f"B10 AM reply future already registered: {key}")
        future: asyncio.Future = self._loop.create_future()
        self._reply_futures[key] = future
        return future

    def discard_reply_future(
        self,
        transfer_id: int,
        endpoint_generation: int,
        kind: int,
    ) -> None:
        self._reply_futures.pop((transfer_id, endpoint_generation, kind), None)

    # ---- recv-side data plumbing (loop thread only) ----

    def register_data_sink(
        self,
        ep_handle: int,
        transfer_id: int,
        endpoint_generation: int,
        sink: DataSink,
    ) -> None:
        key = (ep_handle, transfer_id, endpoint_generation)
        if key in self._data_sinks:
            raise RuntimeError(f"B10 AM data sink already registered: {key}")
        self._data_sinks[key] = sink

    def unregister_data_sink(
        self, ep_handle: int, transfer_id: int, endpoint_generation: int
    ) -> None:
        self._data_sinks.pop((ep_handle, transfer_id, endpoint_generation), None)

    # ---- delivery ----

    def _on_am(self, request: Any, ep_handle: int) -> None:
        """UCXX receiver callback: runs on the progress thread; only hops to
        the agent loop (blocking here stalls all UCX progress).

        A message can arrive between the loop being stopped/closed and the
        UCXX worker being torn down at shutdown; `call_soon_threadsafe` then
        raises RuntimeError on the progress thread. Drop such late messages
        quietly rather than surface a noisy shutdown failure — the transfer
        they belong to is already being abandoned.
        """
        if self._loop.is_closed():
            return
        try:
            self._loop.call_soon_threadsafe(self._dispatch, request, ep_handle)
        except RuntimeError:
            pass

    def _dispatch(self, request: Any, ep_handle: int) -> None:
        view = None
        # UCXX invokes this callback from the request's completion callback
        # whatever the outcome - its trampoline takes the ucs_status_t and
        # discards it - so a receive that failed or was cancelled arrives here
        # looking like any other. Its buffer holds whatever was written before
        # the failure, which for an AM landing in a recycled staging buffer is
        # likely the previous message's bytes: plausible enough to parse and
        # be routed somewhere. Check the outcome first, while the header is
        # still just bytes. The request has completed by definition of being
        # here, so this can only reject a receive that genuinely failed.
        try:
            request.check_error()
        except Exception as exc:
            logger.warning(
                f"B10 AM receive failed; dropped without parsing: "
                f"{type(exc).__name__}: {exc} (ep_handle={ep_handle})"
            )
            return
        try:
            buf = request.recv_buffer
            if buf is None:
                logger.error(
                    f"B10 AM message with no receive buffer; dropped (ep_handle={ep_handle})"
                )
                return
            view = memoryview(buf).cast("B")
            if len(view) < _AM_HEADER_SIZE:
                logger.error(
                    f"B10 AM runt message; dropped (ep_handle={ep_handle} "
                    f"{_describe_am_buffer(view)})"
                )
                return
            header = _unpack_am_header(view)
            payload = view[_AM_HEADER_SIZE : _AM_HEADER_SIZE + header.payload_len]
            if len(payload) != header.payload_len:
                logger.error(
                    f"B10 AM {header.kind_name} truncated: header says "
                    f"{header.payload_len} B, got {len(payload)} B "
                    f"(transfer_id={header.transfer_id} ep_handle={ep_handle}); dropped"
                )
                return
        except Exception as exc:
            logger.error(
                f"B10 AM header parse failed; dropped: {exc} "
                f"(ep_handle={ep_handle} {_describe_am_buffer(view)})"
            )
            return

        if header.kind == _AM_KIND_DATA:
            sink = self._data_sinks.get((ep_handle, header.transfer_id, header.endpoint_generation))
            if sink is None:
                self._log_stale(header, ep_handle)
                return
            sink(header.chunk_index, payload)
        elif header.kind in (_AM_KIND_READY, _AM_KIND_RESULT):
            future = self._reply_futures.pop(
                (header.transfer_id, header.endpoint_generation, header.kind),
                None,
            )
            if future is None or future.done():
                self._log_stale(header, ep_handle)
                return
            future.set_result(_unpack_message(payload))
        elif header.kind == _AM_KIND_CONTROL:
            if self._control_handler is None:
                logger.error("B10 AM control arrived before handler installed; dropped")
                return
            self._control_handler(ep_handle, header, _unpack_message(payload))
        else:
            logger.error(f"B10 AM unknown kind {header.kind}; dropped")

    def _log_stale(self, header: _AmHeader, ep_handle: int) -> None:
        # Stale delivery is expected after a transfer fails or times out on
        # this side while the peer still had the message in flight. What is
        # not expected is a live target for this transfer id under some other
        # key: that says the message was mis-keyed rather than late - the
        # header may not even belong to the message it arrived in - so name
        # the near miss instead of leaving it to be inferred.
        near_misses = [key for key in self._data_sinks if key[1] == header.transfer_id]
        near_misses += [key for key in self._reply_futures if key[0] == header.transfer_id]
        near_miss_detail = f" live_targets_for_this_transfer={near_misses}" if near_misses else ""
        logger.warning(
            f"B10 AM stale {header.kind_name} dropped: "
            f"transfer_id={header.transfer_id} "
            f"chunk_index={header.chunk_index} "
            f"endpoint_generation={header.endpoint_generation} "
            f"payload_len={header.payload_len} "
            f"ep_handle={ep_handle}"
            f"{near_miss_detail}"
        )
