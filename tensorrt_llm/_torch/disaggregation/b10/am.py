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

One instance per agent, registered once with the UCXX worker at startup via
``ucxx.register_am_receiver_callback``. Every B10 message (control / READY /
RESULT / DATA) arrives through the single auto-re-arming receiver callback;
this class parses the 32-byte in-band header (`protocol._AmHeader`) and
routes on the agent event loop:

- ``control``      -> the control handler installed by the receive pipeline
                      (spawns one incoming-write task per transfer)
- ``DATA``         -> the per-(transfer_id, endpoint_generation) chunk sink
                      registered by the active receive transfer
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

# sink(chunk_index, payload_view) — payload_view is a view over the
# ucxx-allocated receive buffer and transitively keeps that buffer alive
# (ucxx hands the numpy array ownership of the malloc'd data), so the sink
# may retain the view past return without a copy: B10's recv stores it in a
# per-chunk future and copies into pinned staging later. The buffer is freed
# once the last reference to the view drops.
DataSink = Callable[[int, Any], None]
# handler(header, payload_dict) — must not block; spawns its own task.
ControlHandler = Callable[[_AmHeader, dict[str, Any]], None]


class B10AmDispatcher:
    """Worker-scoped router for the B10 AM wire plane.

    Owns the single AM receiver callback registered with the UCXX worker and
    fans every inbound message out on the agent event loop by kind: CONTROL
    to the recv pipeline's handler, DATA to the active transfer's chunk sink,
    READY/RESULT to the send pipeline's reply futures. Messages with no live
    target are dropped as stale. See the module docstring for the full
    routing and staleness contract.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop
        self._control_handler: Optional[ControlHandler] = None
        self._data_sinks: dict[tuple[int, int], DataSink] = {}
        self._reply_futures: dict[tuple[int, int, int], asyncio.Future] = {}

    def attach(self, ucxx: Any) -> None:
        """Register with the UCXX worker; call once at agent startup."""
        ucxx.register_am_receiver_callback(_AM_RECEIVER_OWNER, _AM_RECEIVER_ID, self._on_am)

    def set_control_handler(self, handler: ControlHandler) -> None:
        self._control_handler = handler

    # ---- send-side reply plumbing (loop thread only) ----

    def register_reply_future(
        self, transfer_id: int, endpoint_generation: int, kind: int
    ) -> asyncio.Future:
        """Create the future a READY/RESULT resolves. Must be registered
        before the control message is sent so the reply cannot race it."""
        key = (transfer_id, endpoint_generation, kind)
        if key in self._reply_futures:
            raise RuntimeError(f"B10 AM reply future already registered: {key}")
        future: asyncio.Future = self._loop.create_future()
        self._reply_futures[key] = future
        return future

    def discard_reply_future(self, transfer_id: int, endpoint_generation: int, kind: int) -> None:
        self._reply_futures.pop((transfer_id, endpoint_generation, kind), None)

    # ---- recv-side data plumbing (loop thread only) ----

    def register_data_sink(
        self, transfer_id: int, endpoint_generation: int, sink: DataSink
    ) -> None:
        key = (transfer_id, endpoint_generation)
        if key in self._data_sinks:
            raise RuntimeError(f"B10 AM data sink already registered: {key}")
        self._data_sinks[key] = sink

    def unregister_data_sink(self, transfer_id: int, endpoint_generation: int) -> None:
        self._data_sinks.pop((transfer_id, endpoint_generation), None)

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
        try:
            buf = request.recv_buffer
            if buf is None:
                logger.error("B10 AM message with no receive buffer; dropped")
                return
            view = memoryview(buf).cast("B")
            if len(view) < _AM_HEADER_SIZE:
                logger.error(f"B10 AM runt message ({len(view)} B); dropped")
                return
            header = _unpack_am_header(view)
            payload = view[_AM_HEADER_SIZE : _AM_HEADER_SIZE + header.payload_len]
            if len(payload) != header.payload_len:
                logger.error(
                    f"B10 AM {header.kind_name} truncated: header says "
                    f"{header.payload_len} B, got {len(payload)} B "
                    f"(transfer_id={header.transfer_id}); dropped"
                )
                return
        except Exception as exc:
            logger.error(f"B10 AM header parse failed; dropped: {exc}")
            return

        if header.kind == _AM_KIND_DATA:
            sink = self._data_sinks.get((header.transfer_id, header.endpoint_generation))
            if sink is None:
                self._log_stale(header)
                return
            sink(header.chunk_index, payload)
        elif header.kind in (_AM_KIND_READY, _AM_KIND_RESULT):
            future = self._reply_futures.pop(
                (header.transfer_id, header.endpoint_generation, header.kind), None
            )
            if future is None or future.done():
                self._log_stale(header)
                return
            future.set_result(_unpack_message(payload))
        elif header.kind == _AM_KIND_CONTROL:
            if self._control_handler is None:
                logger.error("B10 AM control arrived before handler installed; dropped")
                return
            self._control_handler(header, _unpack_message(payload))
        else:
            logger.error(f"B10 AM unknown kind {header.kind}; dropped")

    @staticmethod
    def _log_stale(header: _AmHeader) -> None:
        # Stale delivery is expected after a transfer fails or times out on
        # this side while the peer still had the message in flight.
        logger.warning(
            f"B10 AM stale {header.kind_name} dropped: "
            f"transfer_id={header.transfer_id} "
            f"chunk_index={header.chunk_index} "
            f"endpoint_generation={header.endpoint_generation} "
            f"payload_len={header.payload_len}"
        )
