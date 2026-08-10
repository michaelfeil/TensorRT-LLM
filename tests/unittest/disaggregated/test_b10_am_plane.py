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


@pytest.mark.asyncio
async def test_dispatchers_reuse_worker_callback_after_detach():
    class FakeUcxx:
        def __init__(self):
            self.worker_key = id(self)
            self.callback = None
            self.registrations = 0

        def get_ucxx_worker(self):
            return self.worker_key

        def register_am_receiver_callback(self, owner, identifier, callback):
            assert (owner, identifier) == ("b10", 0)
            self.registrations += 1
            self.callback = callback

    ucxx = FakeUcxx()
    loop = asyncio.get_running_loop()
    first = B10AmDispatcher(loop)
    second = B10AmDispatcher(loop)
    first.attach(ucxx)

    with pytest.raises(RuntimeError, match="already attached to a live agent"):
        second.attach(ucxx)

    first.detach()
    second.attach(ucxx)
    delivered = []
    second._on_am = lambda request, ep_handle: delivered.append((request, ep_handle))
    ucxx.callback("request", 7)

    assert ucxx.registrations == 1
    assert delivered == [("request", 7)]
    second.detach()


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


def _staging_pool(num_buffers: int = 2, buffer_size: int = 4096):
    from tensorrt_llm._torch.disaggregation.b10.pools import _PinnedStagingBufferPool

    return _PinnedStagingBufferPool(num_buffers=num_buffers, buffer_size=buffer_size)


def test_am_staging_allocator_serves_and_claim_transfers_ownership():
    from tensorrt_llm._torch.disaggregation.b10.pools import _AmStagingAllocator

    pool = _staging_pool()
    allocator = _AmStagingAllocator(pool, min_bytes=64)

    array = allocator.allocate(1000)
    assert array is not None and array.nbytes == 1000
    assert pool.snapshot()["checked_out"] == 1

    array[:4] = list(b"b10a")
    view = allocator.claim(array.ctypes.data)
    assert view is not None
    assert bytes(view.buffer[:4].numpy()) == b"b10a"
    # Claimed: the buffer's death must not return the view to the pool...
    del array
    assert pool.snapshot()["checked_out"] == 1
    # ...only the claimer's release does.
    pool.release([view])
    assert pool.snapshot()["checked_out"] == 0

    # A claim only succeeds once, and unknown addresses are not ours.
    assert allocator.claim(view.buffer.data_ptr()) is None
    assert allocator.claim(0) is None


def test_am_staging_allocator_declines_unservable_sizes():
    from tensorrt_llm._torch.disaggregation.b10.pools import _AmStagingAllocator

    pool = _staging_pool(num_buffers=1, buffer_size=4096)
    allocator = _AmStagingAllocator(pool, min_bytes=64)

    assert allocator.allocate(63) is None  # below the DATA-size gate
    assert allocator.allocate(4097) is None  # larger than a pool buffer
    held = allocator.allocate(64)
    assert held is not None
    assert allocator.allocate(64) is None  # pool momentarily empty


def test_am_staging_allocator_returns_unclaimed_buffer_on_release():
    from tensorrt_llm._torch.disaggregation.b10.pools import _AmStagingAllocator

    pool = _staging_pool(num_buffers=1)
    allocator = _AmStagingAllocator(pool, min_bytes=64)

    # A message that is never claimed (e.g. stale DATA for a failed
    # transfer): dropping the last reference must return the pool buffer.
    array = allocator.allocate(128)
    assert array is not None
    del array
    assert pool.snapshot()["checked_out"] == 0
    assert allocator.allocate(128) is not None


def test_release_staging_slots_skips_am_direct_views():
    import threading

    from tensorrt_llm._torch.disaggregation.b10.core import _AgentCore
    from tensorrt_llm._torch.disaggregation.b10.memory import _BufferView
    from tensorrt_llm._torch.disaggregation.b10.pools import _STAGING_VIEW_AM_DIRECT

    pool = _staging_pool()
    slots = threading.BoundedSemaphore(1)
    core = types.SimpleNamespace(
        staging_buffer_pool=pool,
        staging_buffer_slots=slots,
        _event_ready_for_slot_release=_AgentCore._event_ready_for_slot_release,
    )

    normal = pool.acquire(100)
    slots.acquire()
    direct = _BufferView(
        buffer=normal.buffer,
        owner=normal.owner,
        pool=pool,
        metadata={_STAGING_VIEW_AM_DIRECT: True},
    )
    # The am-direct view took no slot permit, so none is returned for it;
    # the normal view's permit is. A second release for the am-direct view
    # would raise ValueError on the bounded semaphore.
    _AgentCore._release_staging_slots_for_views(core, [direct, normal])
    with pytest.raises(ValueError):
        slots.release()


def _attempt_classifier(expectation_check=None, loop=None):
    """A RecvPipeline carrying only what attempt classification reads.

    The full pipeline needs a UCX worker, CUDA pools and a dispatcher;
    classification is pure policy over the attempt registry, so it is built
    directly rather than standing all of that up.
    """
    from tensorrt_llm._torch.disaggregation.b10.recv import RecvPipeline, _RecvRequestRegistry

    pipeline = RecvPipeline.__new__(RecvPipeline)
    pipeline._requests = _RecvRequestRegistry(loop or asyncio.new_event_loop(), 120.0)
    pipeline._expectation_check = expectation_check
    return pipeline


def _control_ctx(request_id, endpoint_generation, sender=b"sender-a", transfer_id=999):
    """A stand-in for the fields _classify_incoming_attempt reads.

    attempt_key mirrors _RecvTransfer.attempt_key, which is what scopes a
    generation comparison to the sender that produced it.
    """
    return types.SimpleNamespace(
        request_id=request_id,
        endpoint_generation=endpoint_generation,
        transfer_id=transfer_id,
        src_worker_address=sender,
        attempt_key=(sender, transfer_id),
        deadline=None,
        endpoint=None,
    )


def test_classify_attempt_decides_by_endpoint_generation():
    pipeline = _attempt_classifier()
    ctx = _control_ctx(7, 3)

    # Nothing in flight yet: the first attempt is accepted, and no expectation
    # check is installed, so an uncorrelated attempt stays acceptable.
    assert pipeline._classify_incoming_attempt(ctx) == "accept"
    pipeline._requests.begin_attempt(ctx.attempt_key, 3)

    # Same generation is the same attempt arriving twice.
    assert pipeline._classify_incoming_attempt(_control_ctx(7, 3)) == "drop_duplicate"
    # A later generation means the sender rebuilt its endpoint, so the attempt
    # being held can never complete.
    assert pipeline._classify_incoming_attempt(_control_ctx(7, 4)) == "supersede"
    # An earlier generation is from an endpoint already written off.
    assert pipeline._classify_incoming_attempt(_control_ctx(7, 2)) == "reject_stale"
    # A transfer with no request id is still tracked per attempt.
    assert pipeline._classify_incoming_attempt(_control_ctx(None, 3)) == "drop_duplicate"

    # Ending the live attempt reopens it, absent an expectation check.
    pipeline._requests.end_attempt(ctx.attempt_key, 3)
    assert pipeline._classify_incoming_attempt(_control_ctx(7, 3)) == "accept"


def test_classify_attempt_scopes_generations_to_one_sender():
    """Concurrent senders for one request must not classify each other.

    When ctx and gen parallel layouts differ, several sender ranks write the
    same request at once. Each carries its own transfer id and its own
    endpoint generation, drawn from its own endpoint slot, so comparing them
    across senders would drop or cancel legitimate transfers.
    """
    pipeline = _attempt_classifier()

    a = _control_ctx(7, 3, sender=b"rank-0", transfer_id=100)
    assert pipeline._classify_incoming_attempt(a) == "accept"
    pipeline._requests.begin_attempt(a.attempt_key, a.endpoint_generation)

    # Same request, different sender. Every generation relationship that would
    # mean something within one sender must mean nothing across two.
    for generation in (3, 4, 2):
        b = _control_ctx(7, generation, sender=b"rank-1", transfer_id=250)
        assert pipeline._classify_incoming_attempt(b) == "accept"

    # Same sender and request but a different transfer is also independent.
    other = _control_ctx(7, 3, sender=b"rank-0", transfer_id=101)
    assert pipeline._classify_incoming_attempt(other) == "accept"

    # ...while the original attempt still classifies against itself.
    assert pipeline._classify_incoming_attempt(a) == "drop_duplicate"


def test_endpoint_generation_uses_the_whole_header_field():
    """A narrow counter makes attempt ordering ambiguous once it wraps.

    The header field is a uint32, so the counter must use all of it rather than
    the 12 bits it was limited to when identity shared a 64-bit UCX tag.
    """
    from tensorrt_llm._torch.disaggregation.b10.protocol import (
        _MAX_ENDPOINT_GENERATION,
        _pack_am_header,
        _unpack_am_header,
    )

    assert _MAX_ENDPOINT_GENERATION == (1 << 32) - 1
    # The widest generation still round-trips through the wire header.
    hdr = _unpack_am_header(_pack_am_header(_AM_KIND_CONTROL, 1, 0, _MAX_ENDPOINT_GENERATION, 0))
    assert hdr.endpoint_generation == _MAX_ENDPOINT_GENERATION


def test_classify_attempt_survives_generation_wraparound():
    """The generation counter wraps, so ordering cannot be a plain compare."""
    from tensorrt_llm._torch.disaggregation.b10.protocol import (
        _MAX_ENDPOINT_GENERATION,
        _next_endpoint_generation,
    )

    pipeline = _attempt_classifier()
    last = _MAX_ENDPOINT_GENERATION
    wrapped = _next_endpoint_generation(last)
    assert wrapped < last, "expected the counter to wrap for this test to mean anything"

    ctx = _control_ctx(7, last)
    pipeline._requests.begin_attempt(ctx.attempt_key, last)
    # A retry across the wrap is newer, not stale.
    assert pipeline._classify_incoming_attempt(_control_ctx(7, wrapped)) == "supersede"

    # And the reverse still reads as stale: an attempt from just before the
    # wrap arriving after it must not supersede.
    pipeline._requests.begin_attempt(ctx.attempt_key, wrapped)
    assert pipeline._classify_incoming_attempt(_control_ctx(7, last)) == "reject_stale"


def test_classify_attempt_rejects_retry_after_request_finished():
    expecting = {7}
    pipeline = _attempt_classifier(expectation_check=lambda rid: rid in expecting)
    ctx = _control_ctx(7, 3)

    assert pipeline._classify_incoming_attempt(ctx) == "accept"
    pipeline._requests.begin_attempt(ctx.attempt_key, 3)
    pipeline._requests.end_attempt(ctx.attempt_key, 3)

    # The request is still awaiting data, so a fresh attempt is a real retry.
    assert pipeline._classify_incoming_attempt(_control_ctx(7, 4)) == "accept"

    # Once the session layer stops expecting data, the destination is no longer
    # this transfer's to write: a retry from a written-off NIC must be rejected
    # rather than treated as a first attempt.
    expecting.clear()
    assert pipeline._classify_incoming_attempt(_control_ctx(7, 4)) == "reject_unexpected"
    # Another sender's first attempt for the same finished request is rejected
    # too - the request, not the sender, is what stopped expecting data.
    other = _control_ctx(7, 1, sender=b"rank-1", transfer_id=250)
    assert pipeline._classify_incoming_attempt(other) == "reject_unexpected"

    # An attempt already in flight still classifies by generation - the
    # expectation check only resolves the "nothing in flight" case.
    pipeline._requests.begin_attempt(ctx.attempt_key, 4)
    assert pipeline._classify_incoming_attempt(_control_ctx(7, 4)) == "drop_duplicate"
    assert pipeline._classify_incoming_attempt(_control_ctx(7, 5)) == "supersede"


def test_classify_attempt_fails_open_when_expectation_check_raises():
    def broken(_rid):
        raise RuntimeError("session table unavailable")

    pipeline = _attempt_classifier(expectation_check=broken)
    # A broken predicate must never reject a live transfer.
    assert pipeline._classify_incoming_attempt(_control_ctx(7, 3)) == "accept"


def _write_handler_pipeline(reply_endpoint, loop, expectation_check=None):
    """A RecvPipeline that can run _handle_incoming_write's prologue.

    Classifying, taking the attempt slot and connecting back to the sender all
    happen before any transport or CUDA resource is touched, so everything past
    that point is recorded rather than performed.
    """
    pipeline = _attempt_classifier(expectation_check=expectation_check, loop=loop)
    pipeline._duplicate_attempts_dropped = 0
    pipeline._unexpected_attempts_rejected = 0
    pipeline._stale_attempts_rejected = 0
    pipeline._attempts_superseded = 0
    pipeline._get_reply_endpoint = reply_endpoint
    pipeline.performed = []

    async def _record(step):
        pipeline.performed.append(step)

    pipeline._reserve_transfer_resources = lambda _ctx: _record("reserve")
    pipeline._send_ready = lambda _ctx: _record("ready")
    pipeline._receive_chunks = lambda _ctx: _record("chunks")
    pipeline._finalize_success = lambda _ctx: _record("success")
    pipeline._finish_transfer = lambda _ctx: pipeline.performed.append("finish")
    return pipeline


def test_handler_releases_the_attempt_slot_when_connecting_back_fails():
    """A slot taken before the reply endpoint exists must not outlive a failure.

    Connecting back to the sender is deadline-bounded, and it fails precisely
    when the sender lost a NIC - the case this classification exists for. The
    slot is taken before that connect and released by _finish_transfer, which
    only runs once the handler is cancellable, so a failed connect would leave
    the slot behind permanently and make every later attempt of the transfer
    read as a duplicate of an attempt nobody is running.
    """

    async def scenario():
        async def refuse_connect(_address, _deadline):
            raise TimeoutError("reply endpoint timed out")

        pipeline = _write_handler_pipeline(refuse_connect, asyncio.get_running_loop())
        ctx = _control_ctx(7, 3)
        pipeline._validate_and_prepare = lambda _handle, _control: ctx

        with pytest.raises(TimeoutError):
            await pipeline._handle_incoming_write(1, {})

        assert pipeline._requests.current_attempt_generation(ctx.attempt_key) is None
        assert pipeline.performed == []
        # So the sender's retry on that same endpoint is a first attempt again.
        assert pipeline._classify_incoming_attempt(_control_ctx(7, 3)) == "accept"

    asyncio.run(scenario())


def test_handler_yields_to_an_attempt_that_took_over_while_connecting():
    """Superseding cannot cancel a handler that is still connecting back.

    track_task() is what makes a handler cancellable, and it is reached only
    after the reply endpoint exists. An attempt superseded before that point
    therefore keeps running, and must notice it no longer owns the slot before
    reserving anything - otherwise both attempts write the same destination,
    which is the outcome superseding exists to prevent.
    """

    async def scenario():
        connected = asyncio.Event()

        async def slow_connect(_address, _deadline):
            await connected.wait()
            return object()

        pipeline = _write_handler_pipeline(slow_connect, asyncio.get_running_loop())
        first = _control_ctx(7, 3)
        pipeline._validate_and_prepare = lambda _handle, _control: first

        handler = asyncio.ensure_future(pipeline._handle_incoming_write(1, {}))
        await asyncio.sleep(0)  # let it park on the connect
        assert pipeline._requests.current_attempt_generation(first.attempt_key) == 3

        # The retry arrives while the first attempt is still connecting, so
        # there is no handler task registered for it to cancel: superseding can
        # only take the slot.
        retry = _control_ctx(7, 4)
        assert pipeline._classify_incoming_attempt(retry) == "supersede"
        pipeline._requests.supersede_attempt(retry.attempt_key)
        pipeline._requests.begin_attempt(retry.attempt_key, retry.endpoint_generation)

        connected.set()
        await handler

        assert pipeline.performed == []
        # And it left the retry's slot alone on the way out.
        assert pipeline._requests.current_attempt_generation(first.attempt_key) == 4

    asyncio.run(scenario())
