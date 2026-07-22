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
"""Unit tests for the B10 AM wire plane.

Covers the in-band header codec and the worker-scoped dispatcher
(`B10AmDispatcher`). These are pure-Python — no UCX/ucxx required; AM
deliveries are simulated with fake request objects.
"""

import asyncio
import types

import pytest

import tensorrt_llm._torch.disaggregation.b10.protocol as b10_protocol
from tensorrt_llm._torch.disaggregation.b10.am import B10AmDispatcher
from tensorrt_llm._torch.disaggregation.b10.protocol import (
    _AM_HEADER_SIZE,
    _AM_KIND_CONTROL,
    _AM_KIND_DATA,
    _AM_KIND_READY,
    _AM_KIND_RESULT,
    _pack_am_header,
    _pack_am_message,
    _unpack_am_header,
)


def test_am_header_roundtrip():
    hdr_bytes = _pack_am_header(_AM_KIND_DATA, 123456, 42, 7, 4096)
    assert len(hdr_bytes) == _AM_HEADER_SIZE == 32
    hdr = _unpack_am_header(hdr_bytes)
    assert hdr.kind == _AM_KIND_DATA
    assert hdr.transfer_id == 123456
    assert hdr.chunk_index == 42
    assert hdr.endpoint_generation == 7
    assert hdr.payload_len == 4096
    assert hdr.kind_name == "DATA"


def test_am_header_rejects_bad_magic_and_version():
    good = bytearray(_pack_am_header(_AM_KIND_READY, 1, 0, 1, 0))
    bad_magic = bytes([0xFF]) + bytes(good[1:])
    with pytest.raises(ValueError, match="magic"):
        _unpack_am_header(bad_magic)
    bad_version = bytes(good[:4]) + bytes([99]) + bytes(good[5:])
    with pytest.raises(ValueError, match="version"):
        _unpack_am_header(bad_version)


def test_am_message_packs_header_plus_msgpack():
    msg = _pack_am_message(_AM_KIND_READY, 9, 3, {"ok": True, "transfer_id": 9})
    hdr = _unpack_am_header(msg)
    assert hdr.kind == _AM_KIND_READY
    assert hdr.transfer_id == 9
    assert hdr.endpoint_generation == 3
    assert hdr.payload_len == len(msg) - _AM_HEADER_SIZE
    payload = b10_protocol._unpack_message(msg[_AM_HEADER_SIZE:])
    assert payload == {"ok": True, "transfer_id": 9}


def test_agent_descriptor_worker_address_roundtrip():
    desc = b10_protocol.B10AgentDescriptor(
        name="ctx",
        host="10.0.0.1",
        port=1,
        features=(b10_protocol._FEATURE_PACKED_DESCS,),
        worker_address=b"\x01\x02\x03worker",
    )
    decoded = b10_protocol.B10AgentDescriptor.from_bytes(desc.to_bytes())
    assert decoded == desc
    assert decoded.worker_address == b"\x01\x02\x03worker"


def _fake_request(message: bytes) -> types.SimpleNamespace:
    import numpy as np

    return types.SimpleNamespace(recv_buffer=np.frombuffer(bytearray(message), dtype=np.uint8))


@pytest.mark.asyncio
async def test_dispatcher_routes_control_data_and_replies():
    loop = asyncio.get_running_loop()
    dispatcher = B10AmDispatcher(loop)

    controls: list[tuple] = []
    dispatcher.set_control_handler(
        lambda ep_handle, hdr, payload: controls.append((ep_handle, hdr, payload))
    )
    data: list[tuple] = []
    dispatcher.register_data_sink(10, 5, 2, lambda idx, payload: data.append((idx, bytes(payload))))
    ready_future = dispatcher.register_reply_future(5, 2, _AM_KIND_READY)

    dispatcher._dispatch(
        _fake_request(_pack_am_message(_AM_KIND_CONTROL, 5, 2, {"transfer_id": 5})), 10
    )
    dispatcher._dispatch(
        _fake_request(_pack_am_header(_AM_KIND_DATA, 5, 1, 2, 4) + b"\xaa\xbb\xcc\xdd"), 10
    )
    dispatcher._dispatch(
        _fake_request(_pack_am_message(_AM_KIND_READY, 5, 2, {"transfer_id": 5, "ok": True})), 0
    )

    assert len(controls) == 1 and controls[0][0] == 10
    assert controls[0][2] == {"transfer_id": 5}
    assert data == [(1, b"\xaa\xbb\xcc\xdd")]
    assert ready_future.done() and ready_future.result() == {"transfer_id": 5, "ok": True}


@pytest.mark.asyncio
async def test_dispatcher_scopes_data_routes_to_source_endpoint():
    dispatcher = B10AmDispatcher(asyncio.get_running_loop())
    peer_a: list[bytes] = []
    peer_b: list[bytes] = []
    dispatcher.register_data_sink(10, 1, 1, lambda idx, payload: peer_a.append(bytes(payload)))
    dispatcher.register_data_sink(20, 1, 1, lambda idx, payload: peer_b.append(bytes(payload)))

    message_a = _pack_am_header(_AM_KIND_DATA, 1, 0, 1, 1) + b"a"
    message_b = _pack_am_header(_AM_KIND_DATA, 1, 0, 1, 1) + b"b"
    dispatcher._dispatch(_fake_request(message_b), 20)
    dispatcher._dispatch(_fake_request(message_a), 10)

    assert peer_a == [b"a"]
    assert peer_b == [b"b"]


@pytest.mark.asyncio
async def test_dispatcher_drops_stale_messages():
    loop = asyncio.get_running_loop()
    dispatcher = B10AmDispatcher(loop)
    dispatcher.set_control_handler(lambda ep_handle, hdr, payload: None)

    # DATA with no registered sink, reply with no waiter: dropped, no raise.
    dispatcher._dispatch(_fake_request(_pack_am_header(_AM_KIND_DATA, 99, 0, 1, 2) + b"xy"), 0)
    dispatcher._dispatch(_fake_request(_pack_am_message(_AM_KIND_RESULT, 99, 1, {"ok": True})), 0)
    # Unregistered sink after registration behaves the same.
    dispatcher.register_data_sink(0, 7, 1, lambda idx, payload: pytest.fail("sink must be gone"))
    dispatcher.unregister_data_sink(0, 7, 1)
    dispatcher._dispatch(_fake_request(_pack_am_header(_AM_KIND_DATA, 7, 0, 1, 2) + b"xy"), 0)


@pytest.mark.asyncio
async def test_dispatcher_reply_future_single_use_and_duplicate_registration():
    loop = asyncio.get_running_loop()
    dispatcher = B10AmDispatcher(loop)

    future = dispatcher.register_reply_future(1, 1, _AM_KIND_RESULT)
    with pytest.raises(RuntimeError, match="already registered"):
        dispatcher.register_reply_future(1, 1, _AM_KIND_RESULT)

    dispatcher._dispatch(_fake_request(_pack_am_message(_AM_KIND_RESULT, 1, 1, {"ok": True})), 0)
    assert future.result() == {"ok": True}
    # The future was popped on delivery; a duplicate RESULT is stale-dropped.
    dispatcher._dispatch(_fake_request(_pack_am_message(_AM_KIND_RESULT, 1, 1, {"ok": True})), 0)
    # Discard after consumption is a no-op.
    dispatcher.discard_reply_future(1, 1, _AM_KIND_RESULT)


@pytest.mark.asyncio
async def test_dispatcher_drops_truncated_and_runt_messages():
    loop = asyncio.get_running_loop()
    dispatcher = B10AmDispatcher(loop)
    sink_calls: list = []
    dispatcher.register_data_sink(0, 3, 1, lambda idx, payload: sink_calls.append(idx))

    # Runt: shorter than the header.
    dispatcher._dispatch(_fake_request(b"short"), 0)
    # Truncated: header claims more payload than delivered.
    dispatcher._dispatch(
        _fake_request(_pack_am_header(_AM_KIND_DATA, 3, 0, 1, 100) + b"only-a-few"), 0
    )
    assert sink_calls == []
