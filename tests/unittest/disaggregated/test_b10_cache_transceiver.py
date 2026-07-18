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
import asyncio
import contextlib
import json
import os
import sys
import threading
import time
import types
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch

import tensorrt_llm._torch.disaggregation.b10.agent as b10_agent
import tensorrt_llm._torch.disaggregation.b10.async_utils as b10_async_utils
import tensorrt_llm._torch.disaggregation.b10.config as b10_config
import tensorrt_llm._torch.disaggregation.b10.copy_engine as b10_copy_engine
import tensorrt_llm._torch.disaggregation.b10.kernels as b10_kernels
import tensorrt_llm._torch.disaggregation.b10.memory as b10_memory
import tensorrt_llm._torch.disaggregation.b10.net as b10_net
import tensorrt_llm._torch.disaggregation.b10.planning as b10_planning
import tensorrt_llm._torch.disaggregation.b10.pools as b10_pools
import tensorrt_llm._torch.disaggregation.b10.protocol as b10_protocol
import tensorrt_llm._torch.disaggregation.b10.recv as b10_recv
import tensorrt_llm._torch.disaggregation.b10.send as b10_send
import tensorrt_llm._torch.disaggregation.b10.timings as b10_timings
import tensorrt_llm._torch.disaggregation.b10.transceiver as b10_transceiver
import tensorrt_llm._torch.disaggregation.base.agent as base_agent
import tensorrt_llm._torch.disaggregation.native.transfer as native_transfer
from tensorrt_llm._torch.disaggregation.b10.agent import B10CacheTransferAgent
from tensorrt_llm._torch.disaggregation.b10.protocol import (
    B10TransferIdAllocator,
    _data_tag,
    _ready_tag,
    _result_tag,
)
from tensorrt_llm._torch.disaggregation.b10.state import B10TransferStatus, _TransferAbortHandle
from tensorrt_llm._torch.disaggregation.b10.transceiver import B10CacheTransceiver
from tensorrt_llm._torch.disaggregation.base.transfer import (
    SessionStatus,
    WaitResult,
    get_unique_rid,
)
from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2
from tensorrt_llm._torch.pyexecutor.cache_transceiver_runtime import (
    is_python_cache_transceiver_runtime,
)
from tensorrt_llm._torch.pyexecutor.kv_cache_transceiver import (
    create_kv_cache_transceiver,
    should_defer_kv_cache_secondary_pool_allocation,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm.llmapi.llm_args import CacheTransceiverConfig


def _make_uninitialized_b10_agent(
    *, loop: asyncio.AbstractEventLoop | None = None
) -> B10CacheTransferAgent:
    class FakePool:
        def __init__(self, *, num_buffers=4, buffer_size=4):
            self.num_buffers = num_buffers
            self.buffer_size = buffer_size

        @staticmethod
        def metadata_fits(spans):
            return True

        @staticmethod
        def preallocate(device):
            return None

    agent = B10CacheTransferAgent.__new__(B10CacheTransferAgent)
    core = b10_agent._AgentCore.__new__(b10_agent._AgentCore)
    core.loop = loop or Mock()
    core.transfer_timeout_s = None
    core.max_in_flight_ops = 64
    core.tag_registry = b10_protocol.B10TagRegistry(
        quarantine_ttl_s=b10_protocol._DEFAULT_TAG_QUARANTINE_TTL_S
    )
    core.sync_cuda_before_transfer = False
    core.staging_buffer_pool = FakePool()
    core.staging_buffer_slots = asyncio.BoundedSemaphore(4)
    core._staging_quarantine_ttl_s = b10_protocol._DEFAULT_TAG_QUARANTINE_TTL_S
    core._quarantined_staging_views = []
    core.recv_scratch_buffer_pool = FakePool()
    core.cuda_copy_streams = b10_pools._CudaCopyStreamPool()
    agent._core = core
    agent._endpoints = b10_agent.EndpointPool(
        core,
        ucxx=None,
        tag_domain=1,
        endpoint_pool_size=1,
    )
    agent._tracer = b10_agent.TransferTracer(
        core,
        trace_transfer_level=b10_config._TRACE_LEVEL_NONE,
    )
    agent._copies = b10_copy_engine._CopyEngine(core)
    agent._recv = b10_agent.RecvPipeline(
        core,
        agent._copies,
        agent._tracer,
    )
    agent._send = b10_agent.SendPipeline(
        core,
        agent._copies,
        agent._endpoints,
        agent._tracer,
        validate_send_source=False,
        send_admission_limit=0,
        send_admission_bypass_bytes=0,
    )
    agent._tag_domain = 1
    # Deterministic regardless of local CUDA availability: tests that
    # exercise the scratch paths set the device explicitly.
    agent._recv._recv_scratch_device = None
    return agent


@pytest.mark.asyncio
async def test_b10_lock_with_timeout_scopes_ownership():
    lock = asyncio.Lock()

    with pytest.raises(RuntimeError, match="body failed"):
        async with b10_async_utils._lock_with_timeout(lock, 1.0):
            assert lock.locked()
            raise RuntimeError("body failed")
    assert not lock.locked()

    await lock.acquire()
    try:
        with pytest.raises(TimeoutError):
            async with b10_async_utils._lock_with_timeout(lock, 0.001):
                pytest.fail("timed-out waiter must not enter the block")
        assert lock.locked()
    finally:
        lock.release()


@pytest.mark.asyncio
async def test_b10_interrupted_acquire_releases_late_ownership():
    # Model acquisition completing after its caller has stopped waiting.
    class LateLock(asyncio.Lock):
        def __init__(self):
            super().__init__()
            self.started = asyncio.Event()
            self.finish = asyncio.Event()
            self.released = asyncio.Event()

        async def acquire(self) -> bool:
            self.started.set()
            try:
                await self.finish.wait()
            except asyncio.CancelledError:
                await self.finish.wait()
            return await super().acquire()

        def release(self) -> None:
            super().release()
            self.released.set()

    lock = LateLock()
    waiter = asyncio.create_task(b10_async_utils._acquire_with_timeout(lock, 60.0))
    await lock.started.wait()
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    lock.finish.set()
    await asyncio.wait_for(lock.released.wait(), 1.0)
    assert not lock.locked()


@pytest.mark.asyncio
async def test_b10_detached_wait_does_not_cancel_native_work():
    started = asyncio.Event()
    finish = asyncio.Event()
    completed = asyncio.Event()
    cancelled = asyncio.Event()

    async def native_work():
        started.set()
        try:
            await finish.wait()
            completed.set()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    waiter = asyncio.create_task(b10_async_utils._await_detached_with_timeout(native_work(), None))
    await started.wait()
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    assert not cancelled.is_set()
    finish.set()
    await asyncio.wait_for(completed.wait(), 1.0)
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_b10_send_cancel_wakes_backpressure_wait():
    abort_handle = _TransferAbortHandle()
    lock = asyncio.Lock()
    await lock.acquire()
    waiting = asyncio.Event()

    async def wait_for_lock():
        abort_handle.bind_current_task()
        waiting.set()
        async with b10_async_utils._lock_with_timeout(lock, 60.0):
            pytest.fail("cancelled send must not acquire the lock")

    task = asyncio.create_task(wait_for_lock())
    await waiting.wait()
    abort_handle.abort()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert lock.locked()
    lock.release()


@pytest.mark.asyncio
async def test_b10_send_cancel_before_task_start():
    abort_handle = _TransferAbortHandle()
    abort_handle.abort()

    with pytest.raises(asyncio.CancelledError):
        abort_handle.bind_current_task()


@pytest.mark.asyncio
async def test_b10_repeated_send_cancel_does_not_interrupt_cleanup():
    abort_handle = _TransferAbortHandle()
    transfer_started = asyncio.Event()
    cleanup_started = asyncio.Event()
    cleanup_release = asyncio.Event()

    async def transfer():
        try:
            abort_handle.bind_current_task()
            transfer_started.set()
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cleanup_started.set()
            await cleanup_release.wait()
            raise

    task = asyncio.create_task(transfer())
    await transfer_started.wait()
    abort_handle.abort()
    await cleanup_started.wait()

    abort_handle.abort()
    await asyncio.sleep(0)
    assert not task.done()

    cleanup_release.set()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.parametrize("abort_before_arm", [False, True])
def test_b10_send_cancel_consumes_endpoint_callbacks(abort_before_arm):
    abort_handle = _TransferAbortHandle()
    retired = []
    aborted = []
    if abort_before_arm:
        abort_handle.abort()

    abort_handle.bind_endpoint(object(), lambda: retired.append(True), lambda: aborted.append(True))
    abort_handle.abort()
    abort_handle.abort()

    assert retired == [True]
    assert aborted == [True]


@pytest.mark.asyncio
async def test_b10_recv_cancel_wakes_backpressure_wait():
    agent = _make_uninitialized_b10_agent(loop=asyncio.get_running_loop())
    lock = asyncio.Lock()
    await lock.acquire()
    waiting = asyncio.Event()

    async def wait_for_lock():
        with agent._recv._requests.track_task(123):
            waiting.set()
            async with b10_async_utils._lock_with_timeout(lock, 60.0):
                pytest.fail("cancelled receive must not acquire the lock")

    task = asyncio.create_task(wait_for_lock())
    await waiting.wait()
    agent.cancel_recv_request(123)

    with pytest.raises(asyncio.CancelledError):
        await task
    assert lock.locked()
    activity = agent._recv._requests.new_activity(123)
    with pytest.raises(asyncio.CancelledError):
        activity.start_copy()
    lock.release()


@pytest.mark.asyncio
async def test_b10_repeated_recv_cancel_does_not_interrupt_cleanup():
    agent = _make_uninitialized_b10_agent(loop=asyncio.get_running_loop())
    transfer_started = asyncio.Event()
    cleanup_started = asyncio.Event()
    cleanup_release = asyncio.Event()

    async def transfer():
        with agent._recv._requests.track_task(123):
            try:
                transfer_started.set()
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cleanup_started.set()
                await cleanup_release.wait()
                raise

    task = asyncio.create_task(transfer())
    await transfer_started.wait()
    agent.cancel_recv_request(123)
    await cleanup_started.wait()

    agent.cancel_recv_request(123)
    await asyncio.sleep(0)
    assert not task.done()

    cleanup_release.set()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_b10_recv_request_activity_owns_both_counts():
    agent = _make_uninitialized_b10_agent(loop=asyncio.get_running_loop())
    activity = agent._recv._requests.new_activity(123)

    activity.start_transfer()
    activity.start_copy()
    assert agent.has_active_recv_request(123)
    assert agent.has_active_recv_copy_request(123)

    activity.finish()
    assert not agent.has_active_recv_request(123)
    assert not agent.has_active_recv_copy_request(123)


@pytest.mark.asyncio
async def test_b10_run_limited_cancels_and_drains_siblings_on_failure():
    sibling_started = asyncio.Event()
    sibling_finished = asyncio.Event()

    async def sibling():
        sibling_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            sibling_finished.set()

    async def fail():
        await sibling_started.wait()
        raise RuntimeError("worker failed")

    with pytest.raises(RuntimeError, match="worker failed"):
        await b10_async_utils._run_limited([sibling, fail], max_in_flight=2)
    assert sibling_finished.is_set()


def test_b10_transfer_trace_records_completed_and_failed_attempts(monkeypatch):
    clock = iter((0.0, 1.0, 3.0, 4.0, 7.0, 8.0, 13.0, 15.0))
    monkeypatch.setattr(b10_timings.time, "perf_counter", lambda: next(clock))
    trace = b10_timings.TransferTracer(
        Mock(), trace_transfer_level=b10_config._TRACE_LEVEL_INFO
    ).start("send", 1)

    with trace.measure("completed"):
        pass
    with pytest.raises(RuntimeError):
        with trace.measure("failed"):
            raise RuntimeError
    with pytest.raises(RuntimeError):
        with trace.measure_attempt("failed_attempt"):
            raise RuntimeError
    timer = trace.timer("first", "second")
    timer.stop()
    timer.stop()

    assert trace._value("completed") == 2_000.0
    assert trace._value("failed") == 0.0
    assert trace._value("failed_attempt") == 1_000.0
    assert trace._value("first") == trace._value("second") == 2_000.0


def test_b10_disabled_transfer_trace_does_not_read_clock(monkeypatch):
    perf_counter = Mock(side_effect=AssertionError("disabled tracing read the clock"))
    monkeypatch.setattr(b10_timings.time, "perf_counter", perf_counter)
    trace = b10_timings.TransferTracer(
        Mock(), trace_transfer_level=b10_config._TRACE_LEVEL_NONE
    ).start("send", 1)

    with trace.measure("ignored"):
        pass
    trace.timer("ignored").stop()

    perf_counter.assert_not_called()


def test_b10_runtime_is_python_style():
    config = CacheTransceiverConfig(backend="UCX", transceiver_runtime="B10")

    assert is_python_cache_transceiver_runtime(config)
    assert not should_defer_kv_cache_secondary_pool_allocation(config)


def test_python_v2_and_b10_transceivers_implement_runtime_interface():
    assert KvCacheTransceiverV2.__abstractmethods__ == frozenset()
    assert B10CacheTransceiver.__abstractmethods__ == frozenset()


def test_b10_factory_routes_to_b10_transceiver(monkeypatch):
    class DummyB10CacheTransceiver:
        def __init__(self, mapping, dist, kv_cache_manager, cache_transceiver_config):
            self.args = (mapping, dist, kv_cache_manager, cache_transceiver_config)

    module_name = "tensorrt_llm._torch.disaggregation.b10.transceiver"
    monkeypatch.setitem(
        sys.modules,
        module_name,
        types.SimpleNamespace(B10CacheTransceiver=DummyB10CacheTransceiver),
    )

    config = CacheTransceiverConfig(backend="UCX", transceiver_runtime="B10")

    transceiver = create_kv_cache_transceiver(
        mapping=object(),
        dist=object(),
        kv_cache_manager=object(),
        attention_type=None,
        cache_transceiver_config=config,
    )

    assert isinstance(transceiver, DummyB10CacheTransceiver)


@pytest.mark.parametrize("backend", [None, "DEFAULT", "NIXL"])
def test_b10_factory_requires_ucx_backend(backend):
    config = CacheTransceiverConfig(backend=backend, transceiver_runtime="B10")

    with pytest.raises(ValueError, match="requires.*backend='UCX'"):
        create_kv_cache_transceiver(
            mapping=object(),
            dist=object(),
            kv_cache_manager=object(),
            attention_type=None,
            cache_transceiver_config=config,
        )


class _FakeB10RxSession:
    def __init__(self, status=SessionStatus.INIT, aux_pending=False):
        self.status = status
        self.aux_pending = aux_pending
        self.fail_cancelled_calls = 0
        self.close_calls = 0

    def is_completed(self):
        return self.status == SessionStatus.FULLY_TRANSFERRED

    def has_failed(self):
        return self.status in (SessionStatus.CANCELLED, SessionStatus.ERROR)

    def wait_complete(self, blocking=False):
        del blocking
        return WaitResult.COMPLETED

    def has_transferring_tasks(self):
        return False

    def cancel(self):
        self.status = SessionStatus.CANCELLED

    def fail_cancelled_transfers(self, _exc):
        self.fail_cancelled_calls += 1
        self.aux_pending = False

    def close(self):
        self.close_calls += 1


class _FakeB10TxSession:
    def __init__(self, transferring=True):
        self.status = SessionStatus.INIT
        self.transferring = transferring
        self.cancel_calls = 0
        self.close_calls = 0

    def is_completed(self):
        return False

    def has_failed(self):
        return self.status == SessionStatus.CANCELLED

    def wait_complete(self, blocking=False):
        del blocking
        return WaitResult.FAILED

    def cancel(self):
        self.cancel_calls += 1
        self.status = SessionStatus.CANCELLED

    def has_transferring_tasks(self):
        return self.transferring

    def close(self):
        self.close_calls += 1


class _FakeB10TransceiverAgent:
    def __init__(self):
        self.active_recv = True
        self.active_copy = False
        self.cancelled_request_ids = []

    def cancel_recv_request(self, request_id):
        self.cancelled_request_ids.append(request_id)

    def has_active_recv_request(self, _request_id):
        return self.active_recv

    def has_active_recv_copy_request(self, _request_id):
        return self.active_copy

    def discard_source_ready_event(self, _request_id):
        pass


def _make_b10_timeout_request(py_request_id=17, disagg_request_id=7007):
    return types.SimpleNamespace(
        py_request_id=py_request_id,
        py_disaggregated_params=types.SimpleNamespace(
            disagg_request_id=disagg_request_id, schedule_style=None
        ),
        state=LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS,
    )


def _make_b10_timeout_transceiver(request, session, *, context=False):
    transceiver = B10CacheTransceiver.__new__(B10CacheTransceiver)
    rid = get_unique_rid(request)
    transceiver._mapping = types.SimpleNamespace(
        rank=3, enable_attention_dp=False, world_size=1, pp_size=1
    )
    transceiver._dist = types.SimpleNamespace(
        tp_allgather=lambda payload: [payload], pp_allgather=lambda payload: [payload]
    )
    transceiver._ctx_need_tp_sync = False
    transceiver._ctx_need_pp_sync = False
    transceiver._gen_need_sync = False
    transceiver._gen_allgather = lambda payload: [payload]
    transceiver._wait_reqs = {}
    transceiver._send_sessions = {rid: session} if context else {}
    transceiver._send_reqs = {rid: request} if context else {}
    transceiver._recv_sessions = {} if context else {rid: session}
    transceiver._recv_reqs = {} if context else {rid: request}
    transceiver._transfer_worker = types.SimpleNamespace(sweep_stale_req_infos=lambda: None)
    transceiver._context_kv_transfer_error_events = []
    transceiver._generation_kv_transfer_error_events = []
    return transceiver


def test_b10_timeout_ids_wait_for_recv_and_copy_drain():
    request = _make_b10_timeout_request()
    session = _FakeB10RxSession(aux_pending=True)
    transceiver = _make_b10_timeout_transceiver(request, session)
    agent = _FakeB10TransceiverAgent()
    transceiver._b10_transfer_agent = agent
    transceiver.record_generation_kv_transfer_failure_event(request)

    status = transceiver.check_gen_transfer_status(
        0,
        collect_kv_transfer_events=True,
        timed_out_generation_request_ids=[request.py_request_id],
    )
    assert status == ([], [], [], [])
    assert set(agent.cancelled_request_ids) == {7007}
    assert session.close_calls == 0
    assert request.state == LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS

    agent.active_recv = False
    agent.active_copy = True
    assert transceiver.check_gen_transfer_status(
        0, timed_out_generation_request_ids=[request.py_request_id]
    ) == ([], [], [], [])
    assert session.close_calls == 0

    agent.active_copy = False
    assert transceiver.check_gen_transfer_status(
        0,
        collect_kv_transfer_events=True,
        timed_out_generation_request_ids=[request.py_request_id],
    ) == ([], [17], [], [(3, 17)])
    assert session.fail_cancelled_calls == 1
    assert not session.aux_pending
    assert session.close_calls == 1
    assert request.state == LlmRequestState.DISAGG_TRANS_ERROR
    assert 7007 not in transceiver._recv_sessions


def test_b10_cancel_retries_until_recv_copy_drain():
    request = _make_b10_timeout_request()
    session = _FakeB10RxSession(aux_pending=True)
    transceiver = _make_b10_timeout_transceiver(request, session)
    agent = _FakeB10TransceiverAgent()
    agent.active_recv = False
    agent.active_copy = True
    transceiver._b10_transfer_agent = agent

    assert not transceiver.cancel_request(request)
    assert session.close_calls == 0
    assert 7007 in transceiver._recv_sessions

    agent.active_copy = False
    assert transceiver.cancel_request(request)
    assert not session.aux_pending
    assert session.close_calls == 1
    assert 7007 not in transceiver._recv_sessions


def test_b10_generation_close_deferral_includes_active_peer():
    request = _make_b10_timeout_request()
    session = _FakeB10RxSession()
    transceiver = _make_b10_timeout_transceiver(request, session)
    agent = _FakeB10TransceiverAgent()
    agent.active_recv = False
    transceiver._b10_transfer_agent = agent
    transceiver._gen_need_sync = True
    transceiver._gen_allgather = lambda local_ids: [local_ids, [7007]]

    assert transceiver._prepare_gen_session_errors([7007]) == {7007}
    assert session.fail_cancelled_calls == 0
    assert session.close_calls == 0


def test_b10_status_without_timeout_ids_keeps_success_path():
    request = _make_b10_timeout_request()
    session = _FakeB10RxSession(SessionStatus.FULLY_TRANSFERRED)
    transceiver = _make_b10_timeout_transceiver(request, session)
    transceiver._b10_transfer_agent = _FakeB10TransceiverAgent()

    assert transceiver.check_gen_transfer_status(0) == ([17], [], [], [])
    assert request.state == LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE
    assert session.close_calls == 1


def test_b10_prepare_context_requests_skips_consensus_without_waiters():
    transceiver = B10CacheTransceiver.__new__(B10CacheTransceiver)
    transceiver._send_sessions = {}
    transceiver._wait_reqs = {}
    transceiver._ctx_consensus = Mock()

    transceiver.prepare_context_requests([])

    transceiver._ctx_consensus.assert_not_called()


def test_b10_context_timeout_ids_wait_for_agent_writes_to_drain():
    request = _make_b10_timeout_request()
    request.state = LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS
    session = _FakeB10TxSession()
    transceiver = _make_b10_timeout_transceiver(request, session, context=True)
    transceiver._b10_transfer_agent = _FakeB10TransceiverAgent()
    transceiver.record_context_kv_transfer_failure_event(request)

    status = transceiver.check_context_transfer_status(
        0,
        collect_kv_transfer_events=True,
        timed_out_context_request_ids=[request.py_request_id],
    )

    assert status == ([], [], [], [])
    assert session.cancel_calls > 0
    assert session.close_calls == 0
    assert request.state == LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS

    session.transferring = False
    status = transceiver.check_context_transfer_status(
        0,
        collect_kv_transfer_events=True,
        timed_out_context_request_ids=[request.py_request_id],
    )

    assert status == ([], [request.py_request_id], [], [(3, request.py_request_id)])
    assert session.close_calls == 1
    assert request.state == LlmRequestState.DISAGG_TRANS_ERROR
    assert 7007 not in transceiver._send_sessions


@pytest.mark.parametrize("active_tp_peer,active_pp_peer", [(True, False), (False, True)])
def test_b10_context_close_defers_for_active_tp_or_pp_peer(active_tp_peer, active_pp_peer):
    request = _make_b10_timeout_request()
    session = _FakeB10TxSession(transferring=False)
    transceiver = _make_b10_timeout_transceiver(request, session, context=True)
    transceiver._ctx_need_tp_sync = True
    transceiver._ctx_need_pp_sync = True
    transceiver._dist = types.SimpleNamespace(
        tp_allgather=lambda local_ids: [local_ids, [7007] if active_tp_peer else []],
        pp_allgather=lambda local_ids: [local_ids, [7007] if active_pp_peer else []],
    )

    assert transceiver._prepare_context_session_errors([7007]) == {7007}
    assert session.close_calls == 0


def test_tx_session_aux_agent_write_counts_as_transferring():
    session = native_transfer.TxSession.__new__(native_transfer.TxSession)
    session.kv_tasks = []
    session.aux_task = types.SimpleNamespace(status=native_transfer.TaskStatus.TRANSFERRING)

    assert session.has_transferring_tasks()


def test_b10_create_transfer_agent_stores_direct_result(monkeypatch):
    agent = object()
    calls = []

    def create_agent(name, **kwargs):
        calls.append((name, kwargs))
        return agent

    monkeypatch.setattr(b10_transceiver, "create_b10_transfer_agent", create_agent)
    transceiver = B10CacheTransceiver.__new__(B10CacheTransceiver)
    transceiver.kv_transfer_timeout_ms = 250
    transceiver._b10_staging_pool_num_buffers = 8
    transceiver._b10_transfer_agent = None

    assert transceiver._create_transfer_agent("ctx") is agent
    assert transceiver._b10_transfer_agent is agent
    assert calls == [
        (
            "ctx",
            {"transfer_timeout_s": 0.25, "staging_pool_num_buffers": 8},
        )
    ]


class _FakeNotifierWorker:
    def __init__(self, enable_python_future=True):
        self.enable_python_future = enable_python_future
        self.cleared_futures_pool = 0

    def clear_python_futures_pool(self):
        self.cleared_futures_pool += 1


class _FakeNotifierContext:
    def __init__(self, worker):
        self.worker = worker
        self.notifier_stops = 0
        self.notifier_starts = 0

    def stop_notifier_thread(self):
        self.notifier_stops += 1

    def start_notifier_thread(self):
        self.notifier_starts += 1


def _fake_ucxx_module_with_context(ctx, *, preexisting):
    core = types.SimpleNamespace(_ctx=ctx if preexisting else None, _get_ctx=lambda: ctx)
    return types.SimpleNamespace(core=core)


def _run_bind_notifier_on_loop(ucxx_module):
    asyncio.run(_bind_notifier_coro(ucxx_module))


async def _bind_notifier_coro(ucxx_module):
    b10_net._bind_ucxx_python_future_notifier(ucxx_module)


def test_b10_bind_notifier_rebinds_preexisting_context():
    ctx = _FakeNotifierContext(_FakeNotifierWorker())
    ucxx_module = _fake_ucxx_module_with_context(ctx, preexisting=True)

    _run_bind_notifier_on_loop(ucxx_module)

    assert ctx.notifier_stops == 1
    assert ctx.worker.cleared_futures_pool == 1
    assert ctx.notifier_starts == 1


def test_b10_bind_notifier_keeps_fresh_context_binding():
    ctx = _FakeNotifierContext(_FakeNotifierWorker())
    ucxx_module = _fake_ucxx_module_with_context(ctx, preexisting=False)

    _run_bind_notifier_on_loop(ucxx_module)

    assert ctx.notifier_stops == 0
    assert ctx.worker.cleared_futures_pool == 0
    assert ctx.notifier_starts == 0


def test_b10_bind_notifier_noop_when_python_futures_disabled():
    ctx = _FakeNotifierContext(_FakeNotifierWorker(enable_python_future=False))
    ucxx_module = _fake_ucxx_module_with_context(ctx, preexisting=True)

    _run_bind_notifier_on_loop(ucxx_module)

    assert ctx.notifier_stops == 0
    assert ctx.worker.cleared_futures_pool == 0
    assert ctx.notifier_starts == 0


def test_b10_bind_notifier_tolerates_module_without_core():
    _run_bind_notifier_on_loop(types.SimpleNamespace())


def test_b10_defaults_enable_ucxx_python_futures(monkeypatch):
    monkeypatch.delenv("UCXPY_ENABLE_PYTHON_FUTURE", raising=False)
    monkeypatch.delenv("UCXPY_PROGRESS_MODE", raising=False)

    b10_net._apply_default_ucxx_progress_mode()

    assert os.environ["UCXPY_PROGRESS_MODE"] == "thread-polling"
    assert os.environ["UCXPY_ENABLE_PYTHON_FUTURE"] == "1"


def test_b10_usage_manifest_exposes_public_runtime():
    manifest = json.loads(Path("tensorrt_llm/usage/llm_args_golden_manifest.json").read_text())
    runtime_entries = [
        entry
        for entries in manifest.values()
        for entry in entries
        if entry["path"] == "cache_transceiver_config.transceiver_runtime"
    ]

    assert runtime_entries
    assert all("B10" in entry["allowed_values"] for entry in runtime_entries)


def test_b10_transfer_id_allocator_skips_quarantined_ids():
    allocator = B10TransferIdAllocator(start=0, tag_space_size=4, quarantine_ttl_s=60.0)

    quarantined = allocator.allocate()
    allocator.quarantine(quarantined)
    allocated = [allocator.allocate() for _ in range(3)]

    assert quarantined not in allocated
    with pytest.raises(RuntimeError, match="No B10 transfer tags available"):
        allocator.allocate()


def test_b10_transfer_id_allocator_rejects_unencodable_tag_space():
    with pytest.raises(ValueError, match="tag_space_size must be in"):
        B10TransferIdAllocator(tag_space_size=b10_protocol._MAX_TRANSFER_ID + 2)


def test_b10_message_tags_are_scoped_by_transfer_chunk_generation_and_domain():
    tags = {
        _data_tag(transfer_id=1, chunk_index=0, endpoint_generation=1),
        _data_tag(transfer_id=1, chunk_index=1, endpoint_generation=1),
        _data_tag(transfer_id=2, chunk_index=0, endpoint_generation=1),
        _data_tag(transfer_id=1, chunk_index=0, endpoint_generation=2),
        _ready_tag(transfer_id=1, endpoint_generation=1),
        _result_tag(transfer_id=1, endpoint_generation=1),
        _ready_tag(transfer_id=2, endpoint_generation=1),
        _ready_tag(transfer_id=1, endpoint_generation=2),
        _ready_tag(transfer_id=1, endpoint_generation=1, tag_domain=1),
        _ready_tag(transfer_id=1, endpoint_generation=1, tag_domain=2),
        _result_tag(transfer_id=1, endpoint_generation=1, tag_domain=1),
        _data_tag(transfer_id=1, chunk_index=0, endpoint_generation=1, tag_domain=1),
    }

    assert len(tags) == 12


def test_b10_tag_registry_rejects_active_and_quarantined_tags():
    registry = b10_protocol.B10TagRegistry(quarantine_ttl_s=60.0)

    registry.reserve("first", [11, 12])
    with pytest.raises(b10_protocol.B10TagCollisionError, match="collides"):
        registry.reserve("second", [12])

    registry.release("first")
    registry.reserve("second", [12])
    registry.quarantine("second")
    with pytest.raises(b10_protocol.B10TagCollisionError, match="collides"):
        registry.reserve("third", [12])


def test_b10_tag_registry_rejects_duplicate_tags_in_one_reservation():
    registry = b10_protocol.B10TagRegistry(quarantine_ttl_s=60.0)

    with pytest.raises(b10_protocol.B10TagCollisionError, match="duplicate"):
        registry.reserve("first", [11, 11])


def test_b10_tag_registry_skips_future_quarantine_expiry_scan():
    class ExplodingItemsDict(dict):
        def items(self):
            raise AssertionError("future quarantine should not be scanned")

    registry = b10_protocol.B10TagRegistry(quarantine_ttl_s=60.0)
    registry.reserve("first", [11])
    registry.quarantine("first")
    registry._quarantined_until = ExplodingItemsDict(registry._quarantined_until)
    registry._next_quarantine_expiry_s = time.monotonic() + 60.0

    registry.reserve("second", [12])


def test_b10_status_releases_reserved_tags_on_success():
    allocator = B10TransferIdAllocator(start=0, tag_space_size=1, quarantine_ttl_s=60.0)
    tag_registry = b10_protocol.B10TagRegistry(quarantine_ttl_s=60.0)
    tag_owner = ("send", 0)
    tag_registry.reserve(tag_owner, [99])
    transfer_id = allocator.allocate()
    future = Future()
    status = B10TransferStatus(
        future,
        transfer_id,
        allocator,
        _TransferAbortHandle(),
        default_timeout_ms=1000,
        tag_registry=tag_registry,
        tag_owner=tag_owner,
    )

    future.set_result(True)

    assert status.wait()
    tag_registry.reserve("next", [99])


def test_b10_status_quarantines_reserved_tags_on_cancel():
    allocator = B10TransferIdAllocator(start=0, tag_space_size=1, quarantine_ttl_s=60.0)
    tag_registry = b10_protocol.B10TagRegistry(quarantine_ttl_s=60.0)
    tag_owner = ("send", 0)
    tag_registry.reserve(tag_owner, [99])
    transfer_id = allocator.allocate()
    status = B10TransferStatus(
        Future(),
        transfer_id,
        allocator,
        _TransferAbortHandle(),
        default_timeout_ms=1000,
        tag_registry=tag_registry,
        tag_owner=tag_owner,
    )

    status.cancel()

    with pytest.raises(b10_protocol.B10TagCollisionError, match="collides"):
        tag_registry.reserve("next", [99])


def test_b10_status_wait_uses_default_timeout_and_quarantines_id():
    allocator = B10TransferIdAllocator(start=0, tag_space_size=1, quarantine_ttl_s=60.0)
    transfer_id = allocator.allocate()
    status = B10TransferStatus(
        Future(),
        transfer_id,
        allocator,
        _TransferAbortHandle(),
        default_timeout_ms=1,
    )

    assert not status.wait()
    with pytest.raises(RuntimeError, match="No B10 transfer tags available"):
        allocator.allocate()


def test_b10_status_timeout_retires_endpoint_without_cancelling_future():
    allocator = B10TransferIdAllocator(start=0, tag_space_size=1, quarantine_ttl_s=60.0)
    transfer_id = allocator.allocate()
    retire_calls = []
    future = Future()
    abort_handle = _TransferAbortHandle()
    abort_handle.bind_endpoint(object(), lambda: retire_calls.append(True))
    status = B10TransferStatus(
        future,
        transfer_id,
        allocator,
        abort_handle,
        default_timeout_ms=1,
    )

    assert not status.wait()
    assert not future.cancelled()
    assert retire_calls == [True]


def test_b10_status_cancel_quarantines_tags_without_cancelling_future():
    allocator = B10TransferIdAllocator(start=0, tag_space_size=1, quarantine_ttl_s=60.0)
    tag_registry = b10_protocol.B10TagRegistry(quarantine_ttl_s=60.0)
    tag_owner = ("send", 0)
    tag_registry.reserve(tag_owner, [99])
    future = Future()
    abort_handle = _TransferAbortHandle()
    retire_calls = []
    abort_handle.bind_endpoint(object(), lambda: retire_calls.append(True))
    status = B10TransferStatus(
        future,
        allocator.allocate(),
        allocator,
        abort_handle,
        default_timeout_ms=1000,
        tag_registry=tag_registry,
        tag_owner=tag_owner,
    )

    status.cancel()

    assert not future.cancelled()
    assert retire_calls == [True]
    with pytest.raises(b10_protocol.B10TagCollisionError, match="collides"):
        tag_registry.reserve("next", [99])


def test_b10_status_timeout_waits_for_cleanup_event():
    allocator = B10TransferIdAllocator(start=0, tag_space_size=1, quarantine_ttl_s=60.0)
    cleanup_event = threading.Event()
    threading.Timer(0.01, cleanup_event.set).start()
    status = B10TransferStatus(
        Future(),
        allocator.allocate(),
        allocator,
        _TransferAbortHandle(),
        default_timeout_ms=1,
        cleanup_event=cleanup_event,
    )

    assert not status.wait()
    assert cleanup_event.is_set()


def test_b10_submit_status_can_quarantine_reserved_tags_on_cancel(monkeypatch):
    submitted_future = Future()

    def fake_run_coroutine_threadsafe(coro, loop):
        coro.close()
        return submitted_future

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", fake_run_coroutine_threadsafe)
    agent = _make_uninitialized_b10_agent()
    agent._core.loop = object()
    agent._core.transfer_ids = B10TransferIdAllocator(
        start=7, tag_space_size=16, quarantine_ttl_s=60.0
    )
    src_descs = base_agent.MemoryDescs(base_agent.MemoryType.DRAM, [(1, 4, 0)])
    dst_descs = base_agent.MemoryDescs(base_agent.MemoryType.DRAM, [(2, 4, 0)])
    request = base_agent.TransferRequest(
        base_agent.TransferOp.WRITE, src_descs, dst_descs, "remote"
    )

    status = agent.submit_transfer_requests(request)
    tag_owner = ("send", 7)
    agent._core.tag_registry.reserve(tag_owner, [99])

    status.cancel()

    with pytest.raises(b10_protocol.B10TagCollisionError, match="collides"):
        agent._core.tag_registry.reserve("next", [99])


def test_b10_sender_waits_on_source_ready_event_without_current_stream():
    agent = _make_uninitialized_b10_agent()
    fake_device = torch.device("cuda", 0)
    fake_source_ready_event = object()

    class FakeCopyStream:
        def __init__(self):
            self.waited_events = []
            self.waited_streams = []

        def wait_event(self, event):
            self.waited_events.append(event)

        def wait_stream(self, stream):
            self.waited_streams.append(stream)

    fake_copy_stream = FakeCopyStream()
    agent._core.cuda_copy_streams = types.SimpleNamespace(
        stream_for=lambda device: fake_copy_stream
    )

    copy_stream = agent._copies._copy_stream_for_device(
        fake_device,
        {},
        wait_current_stream=True,
        source_ready_events=[(fake_device, fake_source_ready_event)],
        source_ready_waited_keys=set(),
    )

    assert copy_stream is fake_copy_stream
    assert fake_copy_stream.waited_events == [fake_source_ready_event]
    assert fake_copy_stream.waited_streams == []


def test_b10_records_source_ready_event_on_supplied_stream(monkeypatch):
    recorded_streams = []
    fake_device = torch.device("cuda", 0)

    class FakeEvent:
        def record(self, stream):
            recorded_streams.append(stream)

    fake_stream = types.SimpleNamespace(device=fake_device)
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda device: pytest.fail("supplied stream should be used")
    )
    agent = _make_uninitialized_b10_agent()

    agent.record_source_ready_event(123, stream=fake_stream)

    events = agent._send._get_source_ready_events(123)
    assert events is not None
    assert events[0][0] == fake_device
    assert isinstance(events[0][1], FakeEvent)
    assert recorded_streams == [fake_stream]


def test_b10_sender_rejects_vram_request_without_sync_metadata():
    agent = _make_uninitialized_b10_agent()
    src_descs = base_agent.MemoryDescs(base_agent.MemoryType.VRAM, [(1, 1, 0)])
    dst_descs = base_agent.MemoryDescs(base_agent.MemoryType.VRAM, [(2, 1, 0)])
    request = base_agent.TransferRequest(
        base_agent.TransferOp.WRITE, src_descs, dst_descs, "remote"
    )

    status = agent.submit_transfer_requests(request)

    assert status.is_completed()
    assert not status.wait()


def test_b10_sender_rejects_vram_request_without_source_ready_event():
    agent = _make_uninitialized_b10_agent()
    src_descs = base_agent.MemoryDescs(base_agent.MemoryType.VRAM, [(1, 1, 0)])
    dst_descs = base_agent.MemoryDescs(base_agent.MemoryType.VRAM, [(2, 1, 0)])
    request = base_agent.TransferRequest(
        base_agent.TransferOp.WRITE, src_descs, dst_descs, "remote", sync_message="123"
    )

    status = agent.submit_transfer_requests(request)

    assert status.is_completed()
    assert not status.wait()


@pytest.mark.asyncio
async def test_b10_recv_scratch_slot_waits_when_event_query_fails():
    agent = _make_uninitialized_b10_agent()
    agent._recv._recv_scratch_buffer_slots = asyncio.BoundedSemaphore(1)
    await agent._recv._recv_scratch_buffer_slots.acquire()

    class QueryFailingEvent:
        def __init__(self):
            self.synchronized = False

        def query(self):
            raise RuntimeError("query failed")

        def synchronize(self):
            self.synchronized = True

    event = QueryFailingEvent()
    view = b10_memory._BufferView(
        buffer=None,
        owner=object(),
        pool=agent._core.recv_scratch_buffer_pool,
        ready_event=event,
    )

    agent._recv._release_recv_scratch_slots_for_views([view])

    await asyncio.wait_for(agent._recv._recv_scratch_buffer_slots.acquire(), timeout=0.5)
    assert event.synchronized
    agent._recv._recv_scratch_buffer_slots.release()


def test_b10_request_scatter_selection_requires_fragmentation_and_capacity():
    agent = _make_uninitialized_b10_agent()

    class FakeScratchPool:
        buffer_size = 64

        def __init__(self, *, num_buffers=4):
            self.num_buffers = num_buffers

        @staticmethod
        def metadata_fits(spans):
            return True

    agent._recv._recv_scratch_device = torch.device("cuda", 0)
    chunks = [
        b10_memory._TransferChunk(start=0, count=8, size=64),
        b10_memory._TransferChunk(start=8, count=8, size=64),
    ]
    contiguous_descs = _desc_array_view_from(
        [
            b10_memory._NormalizedMemoryDesc(ptr=1000 + idx * 8, size=8, device_id=0)
            for idx in range(16)
        ]
    )
    fragmented_descs = _desc_array_view_from(
        [
            b10_memory._NormalizedMemoryDesc(ptr=1000 + idx * 16, size=8, device_id=0)
            for idx in range(16)
        ]
    )

    agent._core.recv_scratch_buffer_pool = FakeScratchPool()

    def spans_per_chunk(descs):
        return [b10_planning._contiguous_desc_spans(descs, chunk) for chunk in chunks]

    assert (
        agent._recv._request_level_recv_scatter_chunk_indices(
            contiguous_descs,
            "VRAM",
            chunks,
            spans_per_chunk(contiguous_descs),
            descs_have_overlap=False,
        )
        == set()
    )
    assert agent._recv._request_level_recv_scatter_chunk_indices(
        fragmented_descs,
        "VRAM",
        chunks,
        spans_per_chunk(fragmented_descs),
        descs_have_overlap=False,
    ) == {0, 1}

    agent._core.recv_scratch_buffer_pool = FakeScratchPool(num_buffers=1)
    assert (
        agent._recv._request_level_recv_scatter_chunk_indices(
            fragmented_descs,
            "VRAM",
            chunks,
            spans_per_chunk(fragmented_descs),
            descs_have_overlap=False,
        )
        == set()
    )

    agent._core.recv_scratch_buffer_pool = FakeScratchPool(num_buffers=3)
    agent._recv._retired_recv_scratch_views = [object(), object()]
    assert (
        agent._recv._request_level_recv_scatter_chunk_indices(
            fragmented_descs,
            "VRAM",
            chunks,
            spans_per_chunk(fragmented_descs),
            descs_have_overlap=False,
        )
        == set()
    )
    agent._recv._retired_recv_scratch_views = []


def test_b10_aligned_scatter_paths_use_u64_kernels(monkeypatch):
    copy_stream = object()
    launches = []
    fakes = _install_scatter_kernel_fakes(monkeypatch, launches)

    span_size = b10_kernels._SCATTER_KERNEL_BLOCK_SIZE + 8
    descs = _desc_array_view_from(
        [
            b10_memory._NormalizedMemoryDesc(ptr=2000, size=span_size, device_id=0),
        ]
    )
    spans = _span_arrays_from(
        [
            b10_memory._ContiguousSpan(start=0, size=span_size, chunk_offset=0),
        ]
    )

    b10_kernels._scatter_cuda_buffer_to_vram_spans(
        _FakeScatterCudaTensor(), descs, spans, copy_stream
    )

    assert [(launch["kernel"], launch["grid"]) for launch in launches] == [("span_u64", (2,))]
    assert launches[0]["kwargs"] == {
        "block_words": b10_kernels._SCATTER_ALIGNED_KERNEL_BLOCK_WORDS,
        "num_warps": 8,
    }
    launches.clear()
    request_size = b10_kernels._REQUEST_SCATTER_ALIGNED_KERNEL_BLOCK_BYTES + 8
    aligned_plan = b10_memory._DestinationScatterPlan(
        src_ptrs=np.array([1000, 2000], dtype=np.int64),
        dst_ptrs=np.array([2000, 3000], dtype=np.int64),
        sizes=np.array([8, request_size], dtype=np.int64),
        device_ids=np.array([0, 0], dtype=np.int64),
        total_bytes=request_size + 8,
    )

    used_aligned_kernel = b10_kernels._scatter_cuda_buffers_to_vram_destination_order(
        aligned_plan, copy_stream
    )

    assert used_aligned_kernel
    assert [(launch["kernel"], launch["grid"]) for launch in launches] == [("request_u64", (3,))]
    assert launches[0]["kwargs"] == {
        "block_words": b10_kernels._REQUEST_SCATTER_ALIGNED_KERNEL_BLOCK_WORDS,
        "num_warps": 8,
    }
    launches.clear()
    ragged_size = b10_kernels._SCATTER_KERNEL_BLOCK_SIZE + 1
    ragged_plan = b10_memory._DestinationScatterPlan(
        src_ptrs=np.array([1001], dtype=np.int64),
        dst_ptrs=np.array([2001], dtype=np.int64),
        sizes=np.array([ragged_size], dtype=np.int64),
        device_ids=np.array([0], dtype=np.int64),
        total_bytes=ragged_size,
    )

    used_aligned_kernel = b10_kernels._scatter_cuda_buffers_to_vram_destination_order(
        ragged_plan, copy_stream
    )

    assert not used_aligned_kernel
    assert [(launch["kernel"], launch["grid"]) for launch in launches] == [("request_byte", (2,))]
    assert launches[0]["kwargs"] == {
        "block_size": b10_kernels._SCATTER_KERNEL_BLOCK_SIZE,
        "num_warps": 8,
    }
    # Each of the three scatters uploads exactly one flat device metadata
    # tensor, recorded on the copy stream.
    assert len(fakes.device_tensors) == 3
    assert all(tensor.recorded_streams == [copy_stream] for tensor in fakes.device_tensors)


def test_b10_coalesces_one_mib_descriptors_to_full_staging_chunks():
    one_mib = 1024 * 1024
    descs = _desc_array_view_from(
        [b10_memory._NormalizedMemoryDesc(ptr=idx, size=one_mib, device_id=0) for idx in range(513)]
    )

    chunks = b10_planning._coalesce_memory_descs(descs, 512 * one_mib)

    assert [(chunk.start, chunk.count, chunk.size) for chunk in chunks] == [
        (0, 512, 512 * one_mib),
        (512, 1, one_mib),
    ]


def test_b10_reorders_descriptor_pairs_to_reduce_source_spans_on_tie():
    src_descs = _desc_array_view_from(
        [
            b10_memory._NormalizedMemoryDesc(ptr=1002, size=2, device_id=0),
            b10_memory._NormalizedMemoryDesc(ptr=1000, size=2, device_id=0),
            b10_memory._NormalizedMemoryDesc(ptr=1004, size=2, device_id=0),
        ]
    )
    dst_descs = _desc_array_view_from(
        [
            b10_memory._NormalizedMemoryDesc(ptr=2004, size=2, device_id=0),
            b10_memory._NormalizedMemoryDesc(ptr=2000, size=2, device_id=0),
            b10_memory._NormalizedMemoryDesc(ptr=2002, size=2, device_id=0),
        ]
    )

    order = b10_planning._reorder_desc_pairs_for_contiguity(src_descs, dst_descs, 6)

    assert order.strategy == "src_ptr"
    assert order.src_span_count == 1
    assert order.dst_span_count == 3
    assert [desc.ptr for desc in order.src_descs] == [1000, 1002, 1004]
    assert [desc.ptr for desc in order.dst_descs] == [2000, 2004, 2002]


def test_b10_keeps_descriptor_order_when_ranges_overlap():
    src_descs = [
        b10_memory._NormalizedMemoryDesc(ptr=300, size=2, device_id=0),
        b10_memory._NormalizedMemoryDesc(ptr=100, size=2, device_id=0),
        b10_memory._NormalizedMemoryDesc(ptr=200, size=2, device_id=0),
    ]
    dst_descs = [
        b10_memory._NormalizedMemoryDesc(ptr=1002, size=2, device_id=0),
        b10_memory._NormalizedMemoryDesc(ptr=1000, size=2, device_id=0),
        b10_memory._NormalizedMemoryDesc(ptr=1001, size=2, device_id=0),
    ]

    order = b10_planning._reorder_desc_pairs_for_contiguity(
        _desc_array_view_from(src_descs), _desc_array_view_from(dst_descs), 6
    )

    assert order.strategy == "original"
    assert list(order.src_descs) == src_descs
    assert list(order.dst_descs) == dst_descs


def test_b10_overlap_detects_contained_range_via_running_max():
    # [100, 300) contains [150, 160); the desc between them ends before 150,
    # so adjacent-pair comparison alone would miss it.
    descs = _desc_array_view_from(
        [
            b10_memory._NormalizedMemoryDesc(ptr=100, size=200, device_id=0),
            b10_memory._NormalizedMemoryDesc(ptr=120, size=2, device_id=0),
            b10_memory._NormalizedMemoryDesc(ptr=150, size=10, device_id=0),
        ]
    )
    assert b10_planning._has_overlapping_descs(descs)


def test_b10_overlap_ignores_ranges_on_different_devices():
    # Identical address ranges on different devices do not overlap, and one
    # device's running max must not leak into the next device's ranges.
    descs = _desc_array_view_from(
        [
            b10_memory._NormalizedMemoryDesc(ptr=100, size=1000, device_id=0),
            b10_memory._NormalizedMemoryDesc(ptr=100, size=1000, device_id=1),
            b10_memory._NormalizedMemoryDesc(ptr=2000, size=10, device_id=1),
        ]
    )
    assert not b10_planning._has_overlapping_descs(descs)
    zero_sized = _desc_array_view_from(
        [
            b10_memory._NormalizedMemoryDesc(ptr=100, size=0, device_id=0),
            b10_memory._NormalizedMemoryDesc(ptr=100, size=0, device_id=0),
        ]
    )
    assert not b10_planning._has_overlapping_descs(zero_sized)


def test_b10_transfer_chunks_round_trip_through_control_metadata():
    chunks = [
        b10_memory._TransferChunk(start=0, count=64, size=64 * 1024 * 1024),
        b10_memory._TransferChunk(start=64, count=1, size=1024),
    ]
    descs = _desc_array_view_from(
        [
            b10_memory._NormalizedMemoryDesc(ptr=idx, size=1024 * 1024, device_id=0)
            for idx in range(64)
        ]
        + [b10_memory._NormalizedMemoryDesc(ptr=64, size=1024, device_id=0)]
    )

    control = {"transfer_chunks": b10_planning._transfer_chunks_to_control(chunks)}

    assert b10_planning._transfer_chunks_from_control(control, descs) == chunks


def test_b10_staging_pool_defaults_match_pipelined_configuration():
    assert b10_pools._DEFAULT_STAGING_POOL_NUM_BUFFERS == 32
    assert b10_pools._DEFAULT_STAGING_POOL_BUFFER_SIZE == 512 * 1024 * 1024


def test_b10_send_admission_defaults():
    assert b10_config._DEFAULT_SEND_ADMISSION_LIMIT == 3
    assert b10_config._DEFAULT_SEND_ADMISSION_BYPASS_BYTES == 512 * 1024 * 1024


def test_b10_incoming_write_listener_mapping():
    calls = []
    agent = types.SimpleNamespace(_incoming_write_listener=lambda rid, ok: calls.append((rid, ok)))
    notify = b10_recv.RecvPipeline._notify_incoming_write_listener
    notify(agent, 7, "success")
    notify(agent, 7, "failed")
    notify(agent, 7, "cancelled")  # receiver-initiated; no signal
    notify(agent, None, "success")  # no request id; no signal
    assert calls == [(7, True), (7, False)]

    # A raising listener must not propagate into the recv handler's finally.
    boom = types.SimpleNamespace(_incoming_write_listener=lambda rid, ok: 1 / 0)
    notify(boom, 7, "success")

    # No listener registered is a no-op.
    notify(types.SimpleNamespace(_incoming_write_listener=None), 7, "success")


def test_b10_derived_staging_pool_default_from_token_budget():
    class FakeKVCacheManager:
        num_local_layers = 2

        def _calculate_cache_bytes_per_token_for_layers(self, layers):
            assert layers == {0, 1}
            return 1024

    assert b10_transceiver._derive_staging_pool_num_buffers(
        FakeKVCacheManager(), max_tokens_in_buffer=8193, staging_pool_buffer_size=4096
    ) == (2049, 4, 1024)


def test_b10_derived_staging_pool_includes_v1_indexer_cache_bytes():
    class FakeIndexerPool:
        shape = (10, 2, 1, 1536)

        @staticmethod
        def element_size():
            return 1

    class FakeImpl:
        enable_indexer_k_cache = True

        @staticmethod
        def get_indexer_k_cache_pool():
            return FakeIndexerPool()

    class FakeKVCacheManager:
        num_local_layers = 2
        tokens_per_block = 128
        impl = FakeImpl()

        def _calculate_cache_bytes_per_token_for_layers(self, layers):
            assert layers == {0, 1}
            return 1024

    assert not hasattr(FakeKVCacheManager, "enable_indexer_k_cache")
    assert b10_transceiver._derive_staging_pool_num_buffers(
        FakeKVCacheManager(), max_tokens_in_buffer=8193, staging_pool_buffer_size=4096
    ) == (2731, 3, 1048)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="pinned staging pool requires CUDA runtime"
)
def test_b10_pinned_staging_pool_basic_invariants():
    with pytest.raises(ValueError, match="num_buffers must be positive"):
        b10_pools._PinnedStagingBufferPool(num_buffers=0, buffer_size=4)

    pool = b10_pools._PinnedStagingBufferPool(num_buffers=1, buffer_size=4)
    first = pool.acquire(4)

    assert isinstance(first.buffer, torch.Tensor)
    assert first.buffer.device.type == "cpu"
    assert first.buffer.dtype is torch.uint8
    assert first.buffer.shape == (4,)
    assert first.buffer.is_pinned()
    with pytest.raises(RuntimeError, match="No B10 pinned staging buffers available"):
        pool.acquire(4)
    with pytest.raises(RuntimeError, match="exceeds B10 staging buffer size"):
        pool.acquire(5)

    pool.release([first])
    second = pool.acquire(4)

    assert second.owner is first.owner


def test_b10_agent_descriptor_features_roundtrip_and_old_peer_compat():
    import msgpack

    desc = b10_protocol.B10AgentDescriptor(
        name="ctx",
        host="10.0.0.1",
        port=13337,
        tag_domain=7,
        features=(b10_protocol._FEATURE_PACKED_DESCS,),
    )

    decoded = b10_protocol.B10AgentDescriptor.from_bytes(desc.to_bytes())

    assert decoded == desc
    assert decoded.features == ("packed_descs",)

    # A payload from an old peer has no "features" key and decodes to ().
    old_payload = msgpack.packb(
        {
            "protocol": "b10-ucxx",
            "version": 1,
            "name": "gen",
            "host": "10.0.0.2",
            "port": 1,
            "tag_domain": 3,
        },
        use_bin_type=True,
    )

    assert b10_protocol.B10AgentDescriptor.from_bytes(old_payload).features == ()

    # Verbatim copy of the pre-features from_bytes parser: an old peer must
    # decode a new descriptor by silently ignoring the additive key.
    def old_from_bytes(data):
        payload = msgpack.unpackb(data, raw=False)
        if payload.get("protocol") != "b10-ucxx":
            raise ValueError(f"Unexpected B10 descriptor protocol: {payload.get('protocol')}")
        if payload.get("version") != 1:
            raise ValueError(f"Unexpected B10 descriptor version: {payload.get('version')}")
        return dict(
            name=payload["name"],
            host=payload["host"],
            port=int(payload["port"]),
            tag_domain=int(payload.get("tag_domain", 0)),
        )

    assert old_from_bytes(desc.to_bytes()) == {
        "name": "ctx",
        "host": "10.0.0.1",
        "port": 13337,
        "tag_domain": 7,
    }


def _make_send_control_fixture():
    dst_descs = [
        b10_memory._NormalizedMemoryDesc(
            ptr=4096 + idx * 512, size=256 if idx % 2 else 128, device_id=idx % 3
        )
        for idx in range(9)
    ]
    chunk_size = sum(desc.size for desc in dst_descs)
    plan = types.SimpleNamespace(
        transfer_id=11,
        src_type="VRAM",
        dst_type="VRAM",
        dst_descs=_desc_array_view_from(dst_descs),
        transfer_chunks=[b10_memory._TransferChunk(start=0, count=9, size=chunk_size)],
        sync_message=None,
        remote=b10_protocol.B10AgentDescriptor(
            name="peer",
            host="10.0.0.3",
            port=1,
            tag_domain=0,
            features=(b10_protocol._FEATURE_PACKED_DESCS,),
        ),
    )
    lease = types.SimpleNamespace(endpoint_generation=2, tag_domain=5)
    return plan, lease, dst_descs


def test_b10_send_control_packs_descs():
    plan, lease, dst_descs = _make_send_control_fixture()

    control = b10_send.SendPipeline._make_send_control(plan, lease)

    assert "dst_descs" not in control
    assert isinstance(control["dst_descs_packed"], bytes)

    # Wire roundtrip, then the receiver decode branch.
    received = b10_protocol._unpack_message(b10_protocol._pack_message(control))
    decoded = b10_recv._dst_descs_from_control(received)

    assert isinstance(decoded, b10_memory._DescArrayView)
    assert decoded.ptrs.tolist() == [desc.ptr for desc in dst_descs]
    assert decoded.sizes.tolist() == [desc.size for desc in dst_descs]
    assert decoded.device_ids.tolist() == [desc.device_id for desc in dst_descs]
    assert list(decoded) == dst_descs
    assert decoded[4] == dst_descs[4]


def test_b10_dst_descs_from_control_rejects_truncated_packed_payload():
    plan, lease, _ = _make_send_control_fixture()
    control = b10_send.SendPipeline._make_send_control(plan, lease)
    received = b10_protocol._unpack_message(b10_protocol._pack_message(control))

    corrupt = dict(received)
    corrupt["dst_descs_packed"] = received["dst_descs_packed"][:-8]

    with pytest.raises(ValueError, match="multiple of 3"):
        b10_recv._dst_descs_from_control(corrupt)


def test_b10_dst_descs_from_control_rejects_legacy_encoding():
    # A control from a pre-packed_descs sender carries list-of-dicts
    # "dst_descs" and no "dst_descs_packed"; this build rejects it loudly.
    plan, lease, dst_descs = _make_send_control_fixture()
    control = b10_send.SendPipeline._make_send_control(plan, lease)
    legacy = {key: value for key, value in control.items() if key != "dst_descs_packed"}
    legacy["dst_descs"] = [
        {
            "ptr": desc.ptr,
            "size": desc.size,
            "device_id": desc.device_id,
        }
        for desc in dst_descs
    ]
    received = b10_protocol._unpack_message(b10_protocol._pack_message(legacy))

    with pytest.raises(
        ValueError,
        match=r"legacy descriptor encoding; this build requires "
        r"packed_descs — upgrade the sender",
    ):
        b10_recv._dst_descs_from_control(received)


def test_b10_build_send_transfer_plan_rejects_peer_without_packed_descs():
    stub = types.SimpleNamespace(
        _core=types.SimpleNamespace(staging_buffer_pool=types.SimpleNamespace(buffer_size=1 << 20)),
        _endpoints=types.SimpleNamespace(
            _remote_agents={
                "peer": b10_protocol.B10AgentDescriptor(
                    name="peer", host="10.0.0.3", port=1, tag_domain=0, features=()
                )
            }
        ),
    )
    descs = _desc_array_view_from(
        [b10_memory._NormalizedMemoryDesc(ptr=4096, size=128, device_id=0)]
    )

    with pytest.raises(
        RuntimeError,
        match=r"peer 'peer' does not advertise packed_descs; B10 no "
        r"longer supports the legacy descriptor encoding — upgrade the "
        r"peer build",
    ):
        b10_send.SendPipeline._build_send_transfer_plan(
            stub, "peer", 11, "VRAM", descs, "VRAM", descs
        )


def _desc_array_view_from(descs):
    columns = (
        np.array([(desc.ptr, desc.size, desc.device_id) for desc in descs], dtype="<i8")
        .reshape(-1, 3)
        .T
    )
    packed = np.ascontiguousarray(columns).tobytes()
    # Mirror the wire decode exactly, including read-only frombuffer arrays.
    flat = np.frombuffer(packed, dtype="<i8").reshape(3, -1)
    return b10_memory._DescArrayView(flat[0], flat[1], flat[2])


def _span_arrays_from(spans):
    count = len(spans)
    return b10_memory._SpanArrays(
        np.fromiter((span.start for span in spans), dtype=np.int64, count=count),
        np.fromiter((span.size for span in spans), dtype=np.int64, count=count),
        np.fromiter((span.chunk_offset for span in spans), dtype=np.int64, count=count),
    )


def test_b10_transfer_chunks_from_control_validates_and_falls_back():
    descs = [
        b10_memory._NormalizedMemoryDesc(ptr=idx * 1000, size=(idx % 4) * 64, device_id=0)
        for idx in range(12)
    ]
    view = _desc_array_view_from(descs)

    def chunk_dicts(*specs):
        return {
            "transfer_chunks": [
                {
                    "start": start,
                    "count": count,
                    "size": size,
                }
                for start, count, size in specs
            ]
        }

    def sizes_sum(start, count):
        return sum(desc.size for desc in descs[start : start + count])

    valid = chunk_dicts((0, 5, sizes_sum(0, 5)), (5, 7, sizes_sum(5, 7)))
    chunks = b10_planning._transfer_chunks_from_control(valid, view)
    assert [(chunk.start, chunk.count, chunk.size) for chunk in chunks] == [
        (0, 5, sizes_sum(0, 5)),
        (5, 7, sizes_sum(5, 7)),
    ]

    # Missing transfer_chunks falls back to one chunk per descriptor.
    assert b10_planning._transfer_chunks_from_control({}, view) == [
        b10_memory._TransferChunk(start=idx, count=1, size=desc.size)
        for idx, desc in enumerate(descs)
    ]

    failing_controls = [
        (chunk_dicts((1, 11, sizes_sum(1, 11))), "starts at 1, expected 0"),
        (chunk_dicts((0, 5, sizes_sum(0, 5)), (6, 6, sizes_sum(6, 6))), "starts at 6, expected 5"),
        (chunk_dicts((0, 0, 0)), "count must be positive"),
        (chunk_dicts((0, 13, sizes_sum(0, 12))), "exceeds descriptor list"),
        (
            chunk_dicts((0, 12, sizes_sum(0, 12) + 1)),
            f"size mismatch: expected {sizes_sum(0, 12)}, got {sizes_sum(0, 12) + 1}",
        ),
        (chunk_dicts((0, 5, sizes_sum(0, 5))), "cover 5 descriptors, expected 12"),
    ]
    for control, message in failing_controls:
        with pytest.raises(ValueError, match=message):
            b10_planning._transfer_chunks_from_control(control, view)


def test_b10_contiguous_desc_spans_array_view_matches_reference():
    import random

    def reference_contiguous_desc_spans(descs, chunk):
        # Verbatim copy of the pre-vectorization implementation.
        spans = []
        span_start = chunk.start
        span_count = 0
        span_size = 0
        span_offset = 0
        span_device_id = None
        next_ptr = None
        chunk_offset = 0
        for idx in range(chunk.start, chunk.start + chunk.count):
            desc = descs[idx]
            contiguous = (
                span_count > 0 and desc.device_id == span_device_id and desc.ptr == next_ptr
            )
            if not contiguous and span_count > 0:
                spans.append(
                    b10_memory._ContiguousSpan(
                        start=span_start, size=span_size, chunk_offset=span_offset
                    )
                )
                span_start = idx
                span_count = 0
                span_size = 0
                span_offset = chunk_offset

            if span_count == 0:
                span_start = idx
                span_offset = chunk_offset
                span_device_id = desc.device_id
            span_count += 1
            span_size += desc.size
            next_ptr = desc.ptr + desc.size
            chunk_offset += desc.size

        if span_count > 0:
            spans.append(
                b10_memory._ContiguousSpan(
                    start=span_start, size=span_size, chunk_offset=span_offset
                )
            )
        return spans

    rng = random.Random(20260711)
    for trial in range(200):
        desc_count = rng.choice([0, 1, 2, 3, 7, 33, 128])
        descs = []
        ptr = rng.randrange(1 << 40)
        for _ in range(desc_count):
            size = rng.choice([0, 0, 1, 64, 4096])
            if rng.random() < 0.6:
                base = ptr  # extend the current contiguous run
            else:
                base = ptr + rng.randrange(1, 1 << 20)
            descs.append(
                b10_memory._NormalizedMemoryDesc(
                    ptr=base, size=size, device_id=rng.choice([0, 0, 0, 1])
                )
            )
            ptr = base + size
        view = _desc_array_view_from(descs)

        start = 0
        while start < desc_count:
            count = rng.randrange(1, desc_count - start + 1)
            chunk = b10_memory._TransferChunk(
                start=start,
                count=count,
                size=sum(desc.size for desc in descs[start : start + count]),
            )
            expected = reference_contiguous_desc_spans(descs, chunk)
            spans = b10_planning._contiguous_desc_spans(view, chunk)
            assert isinstance(spans, b10_memory._SpanArrays)
            assert list(spans) == expected
            start += count

        empty_chunk = b10_memory._TransferChunk(start=0, count=0, size=0)
        assert list(b10_planning._contiguous_desc_spans(view, empty_chunk)) == []


def test_b10_memory_desc_stats_and_overlap_on_desc_array_view():
    view = _desc_array_view_from(
        [
            b10_memory._NormalizedMemoryDesc(ptr=100, size=50, device_id=0),
            b10_memory._NormalizedMemoryDesc(ptr=150, size=0, device_id=0),
            b10_memory._NormalizedMemoryDesc(ptr=150, size=75, device_id=1),
        ]
    )

    assert b10_planning._memory_desc_stats(view) == (3, 125, 75)
    empty_view = _desc_array_view_from([])
    assert b10_planning._memory_desc_stats(empty_view) == (0, 0, 0)

    assert not b10_planning._has_overlapping_descs(view)
    overlapping = [
        b10_memory._NormalizedMemoryDesc(ptr=100, size=50, device_id=0),
        b10_memory._NormalizedMemoryDesc(ptr=120, size=10, device_id=0),
    ]
    assert b10_planning._has_overlapping_descs(_desc_array_view_from(overlapping))


def test_b10_span_arrays_random_access_and_sequence_protocol():
    spans_list = [
        b10_memory._ContiguousSpan(start=0, size=8, chunk_offset=0),
        b10_memory._ContiguousSpan(start=3, size=0, chunk_offset=8),
        b10_memory._ContiguousSpan(start=4, size=24, chunk_offset=8),
    ]
    spans = _span_arrays_from(spans_list)

    assert len(spans) == 3
    assert spans[1] == spans_list[1]
    assert spans[-1] == spans_list[-1]
    # No __iter__ by design (mirrors _DescArrayView); iteration falls back
    # to the sequence protocol over __getitem__ on cold paths.
    assert list(spans) == spans_list
    with pytest.raises(IndexError):
        spans[3]
    empty = _span_arrays_from([])
    assert len(empty) == 0 and not empty and list(empty) == []


def test_b10_coalesce_memory_descs_matches_greedy_reference():
    import random

    def reference_coalesce(sizes, max_chunk_size):
        # Verbatim copy of the pre-vectorization greedy loop (desc objects
        # reduced to their sizes, the only field it read).
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be positive")
        chunks = []
        start = 0
        count = 0
        size = 0
        for idx, desc_size in enumerate(sizes):
            if desc_size < 0:
                raise ValueError("memory descriptor size must be non-negative")
            if count > 0 and size > 0 and size + desc_size > max_chunk_size:
                chunks.append((start, count, size))
                start = idx
                count = 0
                size = 0
            if count == 0:
                start = idx
            count += 1
            size += desc_size
            if size == max_chunk_size:
                chunks.append((start, count, size))
                count = 0
                size = 0
        if count > 0:
            chunks.append((start, count, size))
        return chunks

    def coalesce(sizes, max_chunk_size):
        descs = _desc_array_view_from(
            [
                b10_memory._NormalizedMemoryDesc(ptr=idx << 20, size=size, device_id=0)
                for idx, size in enumerate(sizes)
            ]
        )
        return [
            (chunk.start, chunk.count, chunk.size)
            for chunk in b10_planning._coalesce_memory_descs(descs, max_chunk_size)
        ]

    directed = [
        [],
        [0],
        [0, 0],
        [10],  # exact fit alone
        [10, 0],  # exact-fit flush must not absorb the trailing zero
        [10, 0, 3],
        [5, 0, 5, 2],  # exact fit reached across a zero-size desc
        [20],  # single oversize desc
        [0, 20, 5],  # oversize preceded by zero-size descs
        [0, 20],
        [3, 0, 9],
        [10, 10, 10],  # back-to-back exact fits
        [9, 1, 9, 1],
    ]
    for sizes in directed:
        for max_chunk_size in (1, 3, 10, 100):
            assert coalesce(sizes, max_chunk_size) == reference_coalesce(sizes, max_chunk_size), (
                sizes,
                max_chunk_size,
            )

    rng = random.Random(20260712)
    for _ in range(300):
        sizes = [
            rng.choice([0, 0, 1, 2, 5, 10, 11, 25])
            for _ in range(rng.choice([0, 1, 2, 3, 7, 20, 65]))
        ]
        max_chunk_size = rng.choice([1, 2, 5, 10, 25, 1 << 30])
        assert coalesce(sizes, max_chunk_size) == reference_coalesce(sizes, max_chunk_size), (
            sizes,
            max_chunk_size,
        )

    with pytest.raises(ValueError, match="must be positive"):
        coalesce([1], 0)
    with pytest.raises(ValueError, match="non-negative"):
        coalesce([-1], 4)


def test_b10_normalize_memory_descs_bulk_ingestion():
    # Tuple descs (pure-Python MemoryDescs): one np.asarray pass.
    view = b10_planning._normalize_memory_descs(types.SimpleNamespace(descs=[(1, 4, 0), (5, 8, 1)]))
    assert isinstance(view, b10_memory._DescArrayView)
    assert view.ptrs.tolist() == [1, 5]
    assert view.sizes.tolist() == [4, 8]
    assert view.device_ids.tolist() == [0, 1]
    assert view[1] == b10_memory._NormalizedMemoryDesc(ptr=5, size=8, device_id=1)

    # Attribute descs mirroring the C++ MemoryDesc binding (addr/len).
    cpp_like = [types.SimpleNamespace(addr=9 + idx, len=16, device_id=2) for idx in range(3)]
    view = b10_planning._normalize_memory_descs(types.SimpleNamespace(descs=cpp_like))
    assert view.ptrs.tolist() == [9, 10, 11]
    assert view.sizes.tolist() == [16, 16, 16]
    assert view.device_ids.tolist() == [2, 2, 2]

    # ptr/size attribute spelling is also accepted.
    view = b10_planning._normalize_memory_descs(
        types.SimpleNamespace(descs=[types.SimpleNamespace(ptr=7, size=2, device_id=0)])
    )
    assert view.ptrs.tolist() == [7]

    empty = b10_planning._normalize_memory_descs(types.SimpleNamespace(descs=[]))
    assert len(empty) == 0 and not empty

    with pytest.raises(TypeError, match="Unsupported memory descriptor"):
        b10_planning._normalize_memory_descs(types.SimpleNamespace(descs=[object()]))


class _FakeScatterCudaTensor:
    """Fake CUDA tensor for the Triton scatter-path tests.

    Coerces values to ints, tracks record_stream calls, and reports every
    ``.to()`` upload into the list installed by
    _install_scatter_kernel_fakes.
    """

    # Rebound per test by _install_scatter_kernel_fakes (monkeypatch-scoped).
    device_tensors = None

    def __init__(self, values=None):
        self.values = [int(v) for v in (values or [])]
        self.is_cuda = True
        self.recorded_streams = []

    def record_stream(self, stream):
        self.recorded_streams.append(stream)

    def split(self, split_sizes, dim=0):
        parts = []
        start = 0
        for size in split_sizes:
            parts.append(_FakeScatterCudaTensor(self.values[start : start + size]))
            start += size
        return tuple(parts)

    def to(self, device, non_blocking=False):
        tensor = _FakeScatterCudaTensor(self.values)
        if _FakeScatterCudaTensor.device_tensors is not None:
            _FakeScatterCudaTensor.device_tensors.append(tensor)
        return tensor

    def copy_(self, src, non_blocking=False):
        assert non_blocking
        self.values = [int(v) for v in src.values]
        return self


class _FakeScatterPreallocatedTensor:
    """Fake pool metadata tensor; hands out tracked capacity-checked views."""

    def __init__(self, capacity, device):
        self.capacity = capacity
        self.device = device
        self.views = []

    def numel(self):
        return self.capacity

    def __getitem__(self, item):
        assert item == slice(None, item.stop)
        assert item.stop <= self.capacity
        view = _FakeScatterCudaTensor([0] * item.stop)
        self.views.append(view)
        return view


class _FakeScatterKernel:
    """Records every launch (kernel name, grid, arg values, kwargs)."""

    def __init__(self, name, launches):
        self.name = name
        self.launches = launches

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.launches.append(
                {
                    "kernel": self.name,
                    "grid": grid,
                    "args": [list(arg.values) for arg in args],
                    "kwargs": dict(kwargs),
                }
            )

        return launch


def _install_scatter_kernel_fakes(monkeypatch, launches, pinned_arrays=None):
    """Install the fake torch/kernel surface for the scatter paths.

    Kernel launches are recorded into ``launches``; the flat numpy arrays
    handed to _pin_scatter_metadata are appended to ``pinned_arrays`` when
    given. Returns a namespace with ``device_tensors`` (every device tensor
    produced by a fake ``.to()`` upload) and ``pinned_tensors`` (every fake
    pinned host staging tensor handed out).
    """
    fakes = types.SimpleNamespace(device_tensors=[], pinned_tensors=[])
    monkeypatch.setattr(_FakeScatterCudaTensor, "device_tensors", fakes.device_tensors)

    def fake_pin_scatter_metadata(values):
        assert values.dtype == np.int64
        if pinned_arrays is not None:
            pinned_arrays.append(values)
        tensor = _FakeScatterCudaTensor(values.tolist())
        fakes.pinned_tensors.append(tensor)
        return tensor

    monkeypatch.setattr(torch, "Tensor", _FakeScatterCudaTensor)
    monkeypatch.setattr(b10_kernels, "_pin_scatter_metadata", fake_pin_scatter_metadata)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: contextlib.nullcontext())
    monkeypatch.setattr(
        b10_kernels,
        "_get_scatter_cuda_buffer_to_vram_spans_kernel",
        lambda: (
            None,
            _FakeScatterKernel("span_byte", launches),
            _FakeScatterKernel("span_u64", launches),
            _FakeScatterKernel("request_byte", launches),
            _FakeScatterKernel("request_u64", launches),
        ),
    )
    return fakes


def _aligned_scatter_span_case():
    """Three aligned spans (middle one zero-size) plus expected sections."""
    block_size = b10_kernels._SCATTER_KERNEL_BLOCK_SIZE
    word = b10_kernels._SCATTER_ALIGNED_WORD_BYTES
    span_sizes = [block_size + word, 0, word]
    spans = []
    descs = []
    chunk_offset = 0
    for idx, size in enumerate(span_sizes):
        spans.append(b10_memory._ContiguousSpan(start=idx, size=size, chunk_offset=chunk_offset))
        descs.append(b10_memory._NormalizedMemoryDesc(ptr=1 << (20 + idx), size=size, device_id=0))
        chunk_offset += size
    expected_sections = [
        [desc.ptr for desc in descs],
        [span.chunk_offset for span in spans],
        [size // word for size in span_sizes],
        [0, 0, 2],
        [0, block_size // word, 0],
    ]
    return _desc_array_view_from(descs), _span_arrays_from(spans), expected_sections


def test_b10_scatter_program_expansion_matches_span_reference():
    import random

    def reference_scatter_programs_for_spans(spans, block_size):
        # Verbatim copy of the pre-vectorization _scatter_programs_for_spans.
        program_span_indices = []
        program_offsets = []
        for span_idx, span in enumerate(spans):
            for block_offset in range(0, span.size, block_size):
                program_span_indices.append(span_idx)
                program_offsets.append(block_offset)
        return program_span_indices, program_offsets

    rng = random.Random(20260711)
    block_size = b10_kernels._SCATTER_KERNEL_BLOCK_SIZE
    for trial in range(200):
        span_count = rng.choice([0, 1, 2, 3, 8, 64, 512])
        spans = []
        chunk_offset = 0
        for _ in range(span_count):
            size = rng.choice(
                [
                    0,
                    1,
                    8,
                    block_size - 1,
                    block_size,
                    block_size + 1,
                    4 * block_size,
                    4 * block_size + 8,
                ]
            )
            spans.append(
                b10_memory._ContiguousSpan(start=len(spans), size=size, chunk_offset=chunk_offset)
            )
            chunk_offset += size
        sizes_arr = np.fromiter((span.size for span in spans), dtype=np.int64, count=len(spans))

        indices, offsets = b10_planning._scatter_programs_for_fragment_sizes(sizes_arr, block_size)

        expected = reference_scatter_programs_for_spans(spans, block_size)
        assert indices.tolist() == expected[0]
        assert offsets.tolist() == expected[1]
        # metadata_fits sizes preallocated buffers with this count.
        assert indices.shape[0] == b10_planning._scatter_program_count_for_spans(
            _span_arrays_from(spans), block_size
        )


def test_b10_scatter_span_metadata_modes_upload_identical_sections(monkeypatch):
    copy_stream = object()
    launches = []
    pinned_arrays = []
    fakes = _install_scatter_kernel_fakes(monkeypatch, launches, pinned_arrays=pinned_arrays)
    descs, spans, expected_sections = _aligned_scatter_span_case()

    lifetime_refs = []
    b10_kernels._scatter_cuda_buffer_to_vram_spans(
        _FakeScatterCudaTensor(), descs, spans, copy_stream, lifetime_refs=lifetime_refs
    )

    # Ad-hoc mode: one flat pinned int64 staging array, one device upload
    # whose split views feed the kernel, and both tensors kept alive.
    assert pinned_arrays[0].dtype == np.int64
    assert pinned_arrays[0].tolist() == sum(expected_sections, [])
    assert launches == [
        {
            "kernel": "span_u64",
            "grid": (3,),
            "args": [[]] + expected_sections,
            "kwargs": {
                "block_words": b10_kernels._SCATTER_ALIGNED_KERNEL_BLOCK_WORDS,
                "num_warps": 8,
            },
        }
    ]
    assert [ref.buffer for ref in lifetime_refs] == [
        fakes.device_tensors[0],
        fakes.pinned_tensors[0],
    ]
    # Ad-hoc mode records the flat device upload on the copy stream.
    assert len(fakes.device_tensors) == 1
    assert fakes.device_tensors[0].recorded_streams == [copy_stream]

    launches.clear()
    device = torch.device("cuda", 0)
    metadata = {
        name: _FakeScatterPreallocatedTensor(8, device)
        for name in b10_kernels._SCATTER_METADATA_SECTION_NAMES
    }

    pool_lifetime_refs = []
    b10_kernels._scatter_cuda_buffer_to_vram_spans(
        _FakeScatterCudaTensor(),
        descs,
        spans,
        copy_stream,
        lifetime_refs=pool_lifetime_refs,
        metadata=metadata,
    )

    # Preallocated mode: identical sections land in the pool tensor views.
    assert [launch["args"][1:] for launch in launches] == [expected_sections]
    # The only lifetime ref is the pinned host staging tensor; the pool
    # metadata tensors are process-lifetime and deliberately not retained.
    assert [ref.buffer for ref in pool_lifetime_refs] == [fakes.pinned_tensors[-1]]
    # Pool mode records nothing on the copy stream: no ad-hoc device upload
    # happens (count unchanged from the ad-hoc run above) and the
    # process-lifetime pool views need no record_stream.
    assert len(fakes.device_tensors) == 1
    assert all(view.recorded_streams == [] for tensor in metadata.values() for view in tensor.views)


def test_b10_scatter_span_metadata_pool_rejects_short_or_missing_tensors(monkeypatch):
    launches = []
    _install_scatter_kernel_fakes(monkeypatch, launches)
    descs, spans, _ = _aligned_scatter_span_case()
    device = torch.device("cuda", 0)
    metadata = {
        name: _FakeScatterPreallocatedTensor(8, device)
        for name in b10_kernels._SCATTER_METADATA_SECTION_NAMES
    }

    for name in b10_kernels._SCATTER_METADATA_SECTION_NAMES:
        short_metadata = dict(metadata)
        short_metadata[name] = _FakeScatterPreallocatedTensor(2, device)
        with pytest.raises(
            RuntimeError, match=f"capacity exceeded for {name}: needed=3 available=2"
        ):
            b10_kernels._scatter_cuda_buffer_to_vram_spans(
                _FakeScatterCudaTensor(), descs, spans, object(), metadata=short_metadata
            )
        missing_metadata = dict(metadata)
        del missing_metadata[name]
        with pytest.raises(RuntimeError, match=f"tensor {name} was not preallocated"):
            b10_kernels._scatter_cuda_buffer_to_vram_spans(
                _FakeScatterCudaTensor(), descs, spans, object(), metadata=missing_metadata
            )
    assert launches == []


def test_b10_send_gather_plan_matches_span_loop_copy_plan():
    import random

    rng = random.Random(20260712)
    for trial in range(100):
        desc_count = rng.choice([1, 2, 3, 8, 33, 128])
        descs = []
        ptr = rng.randrange(1 << 40)
        for _ in range(desc_count):
            size = rng.choice([0, 0, 1, 8, 64, 4096, 16 * 1024 + 8])
            if rng.random() < 0.6:
                base = ptr  # extend the current contiguous run
            else:
                base = ptr + rng.randrange(1, 1 << 20)
            descs.append(
                b10_memory._NormalizedMemoryDesc(
                    ptr=base, size=size, device_id=rng.choice([0, 0, 0, 1])
                )
            )
            ptr = base + size
        chunk = b10_memory._TransferChunk(
            start=0, count=desc_count, size=sum(d.size for d in descs)
        )
        staging_base = rng.randrange(1 << 40)

        view = _desc_array_view_from(descs)
        spans = b10_planning._contiguous_desc_spans(view, chunk)
        plan = b10_planning._gather_plan_for_vram_spans(view, spans, staging_base)
        # Frozen reference: the per-span copy loop moved
        # descs[span.start].ptr -> staging + span.chunk_offset for
        # span.size bytes; zero-size spans moved nothing.
        expected = [
            (
                descs[span.start].ptr,
                staging_base + span.chunk_offset,
                span.size,
                descs[span.start].device_id,
            )
            for span in spans
            if span.size > 0
        ]
        assert (
            list(
                zip(
                    plan.src_ptrs.tolist(),
                    plan.dst_ptrs.tolist(),
                    plan.sizes.tolist(),
                    plan.device_ids.tolist(),
                )
            )
            == expected
        )
        assert plan.total_bytes == chunk.size


class _FakePinnedStagingTensor(_FakeScatterCudaTensor):
    """Fake pinned host staging tensor for the send-gather tests."""

    def __init__(self, base_ptr=1 << 30, pinned=True, cuda=False):
        super().__init__()
        self.is_cuda = cuda
        self.device = types.SimpleNamespace(type="cuda" if cuda else "cpu")
        self._pinned = pinned
        self._base_ptr = base_ptr

    def is_pinned(self):
        return self._pinned

    def data_ptr(self):
        return self._base_ptr


def _gather_span_case(span_sizes, base_ptr=1 << 20, device_ids=None):
    descs = []
    spans = []
    chunk_offset = 0
    ptr = base_ptr
    for idx, size in enumerate(span_sizes):
        descs.append(
            b10_memory._NormalizedMemoryDesc(
                ptr=ptr, size=size, device_id=device_ids[idx] if device_ids else 0
            )
        )
        spans.append(b10_memory._ContiguousSpan(start=idx, size=size, chunk_offset=chunk_offset))
        chunk_offset += size
        ptr += size + 4096  # keep spans non-contiguous in source
    return _desc_array_view_from(descs), _span_arrays_from(spans)


def test_b10_send_gather_launches_single_absolute_kernel(monkeypatch):
    launches = []
    pinned_arrays = []
    fakes = _install_scatter_kernel_fakes(monkeypatch, launches, pinned_arrays=pinned_arrays)
    set_devices = []
    monkeypatch.setattr(torch.cuda, "set_device", set_devices.append)
    copy_stream = object()
    word = b10_kernels._SCATTER_ALIGNED_WORD_BYTES
    block = b10_kernels._REQUEST_SCATTER_ALIGNED_KERNEL_BLOCK_BYTES

    # Aligned spans (middle one zero-size, dropped from the plan) take the
    # absolute u64 kernel in one launch that stores straight into staging.
    descs, spans = _gather_span_case([block + word, 0, word])
    staging = _FakePinnedStagingTensor()
    lifetime_refs = []
    launched = b10_kernels._gather_vram_spans_to_pinned_staging(
        descs, spans, staging, copy_stream, lifetime_refs=lifetime_refs
    )

    assert launched
    assert set_devices == [0]
    expected_sections = [
        [descs[0].ptr, descs[2].ptr],
        [staging.data_ptr() + spans[0].chunk_offset, staging.data_ptr() + spans[2].chunk_offset],
        [(block + word) // word, 1],
        [0, 0, 1],
        [0, block // word, 0],
    ]
    assert pinned_arrays[0].tolist() == sum(expected_sections, [])
    assert launches == [
        {
            "kernel": "request_u64",
            "grid": (3,),
            "args": expected_sections,
            "kwargs": {
                "block_words": b10_kernels._REQUEST_SCATTER_ALIGNED_KERNEL_BLOCK_WORDS,
                "num_warps": 8,
            },
        }
    ]
    # Ad-hoc metadata lifetime: flat device upload + pinned host staging.
    assert [ref.buffer for ref in lifetime_refs] == [
        fakes.device_tensors[0],
        fakes.pinned_tensors[0],
    ]
    assert fakes.device_tensors[0].recorded_streams == [copy_stream]

    # An unaligned span size falls back to the absolute byte kernel.
    launches.clear()
    descs, spans = _gather_span_case([word, 1])
    assert b10_kernels._gather_vram_spans_to_pinned_staging(
        descs, spans, _FakePinnedStagingTensor(), copy_stream
    )
    assert [launch["kernel"] for launch in launches] == ["request_byte"]
    assert launches[0]["kwargs"] == {
        "block_size": b10_kernels._SCATTER_KERNEL_BLOCK_SIZE,
        "num_warps": 8,
    }


def test_b10_send_gather_validates_staging_and_empty_spans(monkeypatch):
    launches = []
    _install_scatter_kernel_fakes(monkeypatch, launches)
    set_devices = []
    monkeypatch.setattr(torch.cuda, "set_device", set_devices.append)
    descs, spans = _gather_span_case([8, 8])

    for staging in (
        object(),
        _FakePinnedStagingTensor(pinned=False),
        _FakePinnedStagingTensor(cuda=True),
    ):
        with pytest.raises(TypeError, match="pinned host torch tensor"):
            b10_kernels._gather_vram_spans_to_pinned_staging(descs, spans, staging, None)

    # All-zero spans move no bytes: no launch, no device pinning.
    zero_descs, zero_spans = _gather_span_case([0, 0])
    assert not b10_kernels._gather_vram_spans_to_pinned_staging(
        zero_descs, zero_spans, _FakePinnedStagingTensor(), None
    )
    assert launches == []
    assert set_devices == []


def test_b10_send_gather_gating_routes_by_eligibility(monkeypatch):
    agent = _make_uninitialized_b10_agent()
    monkeypatch.setattr(torch, "Tensor", _FakeScatterCudaTensor)
    monkeypatch.setattr(b10_copy_engine, "_scatter_kernels_available", lambda: True)
    min_spans = b10_pools._DEFAULT_SEND_GATHER_MIN_SPANS
    descs, spans = _gather_span_case([8] * min_spans)
    staging = _FakePinnedStagingTensor()

    assert agent._copies.send_gather_device(descs, "VRAM", staging, spans) == torch.device(
        "cuda", 0
    )
    # Triton missing/broken routes otherwise-eligible chunks to the loop.
    monkeypatch.setattr(b10_copy_engine, "_scatter_kernels_available", lambda: False)
    assert agent._copies.send_gather_device(descs, "VRAM", staging, spans) is None
    monkeypatch.setattr(b10_copy_engine, "_scatter_kernels_available", lambda: True)
    assert agent._copies.send_gather_device(descs, "DRAM", staging, spans) is None
    short_descs, short_spans = _gather_span_case([8] * (min_spans - 1))
    assert agent._copies.send_gather_device(short_descs, "VRAM", staging, short_spans) is None
    assert agent._copies.send_gather_device(descs, "VRAM", object(), spans) is None
    assert (
        agent._copies.send_gather_device(
            descs, "VRAM", _FakePinnedStagingTensor(pinned=False), spans
        )
        is None
    )
    assert (
        agent._copies.send_gather_device(descs, "VRAM", _FakePinnedStagingTensor(cuda=True), spans)
        is None
    )
    mixed_descs, mixed_spans = _gather_span_case(
        [8] * min_spans, device_ids=[idx % 2 for idx in range(min_spans)]
    )
    assert agent._copies.send_gather_device(mixed_descs, "VRAM", staging, mixed_spans) is None


def test_b10_copy_chunk_send_gather_and_trimmed_loop(monkeypatch):
    device = torch.device("cuda", 0)
    source_ready_event = object()

    class FakeCopyStream:
        def __init__(self):
            self.waited_events = []

        def wait_event(self, event):
            self.waited_events.append(event)

    fake_copy_stream = FakeCopyStream()
    agent = _make_uninitialized_b10_agent()
    agent._core.cuda_copy_streams = types.SimpleNamespace(stream_for=lambda dev: fake_copy_stream)
    monkeypatch.setattr(torch, "Tensor", _FakeScatterCudaTensor)
    monkeypatch.setattr(b10_copy_engine, "_scatter_kernels_available", lambda: True)

    class _FakeCudaEvent:
        def __init__(self):
            self.recorded_streams = []

        def record(self, stream):
            self.recorded_streams.append(stream)

    # raising=False: the fake-torch harness environments lack cuda.Event.
    monkeypatch.setattr(torch.cuda, "Event", _FakeCudaEvent, raising=False)

    stream_contexts = []

    class _FakeStreamContext:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            stream_contexts.append(("enter", self.stream))
            return self

        def __exit__(self, *exc):
            stream_contexts.append(("exit", self.stream))
            return False

    monkeypatch.setattr(torch.cuda, "stream", _FakeStreamContext)

    class _FakeSpanBuffer:
        def __init__(self):
            self.device = device
            self.copies = []

        def copy_(self, src, non_blocking=False):
            self.copies.append((src, non_blocking))
            return self

    span_buffers = {}

    def fake_span_view(descs, span, memory_type):
        assert memory_type == "VRAM"
        buffer = _FakeSpanBuffer()
        span_buffers[span.chunk_offset] = buffer
        return b10_memory._BufferView(buffer, buffer)

    monkeypatch.setattr(b10_copy_engine, "_span_view", fake_span_view)

    class _FakeStagingSlice:
        def __init__(self, start, stop):
            self.start = start
            self.stop = stop
            self.copies = []

        def copy_(self, src, non_blocking=False):
            self.copies.append((src, non_blocking))
            return self

    class _FakeLoopStagingTensor(_FakePinnedStagingTensor):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.slices = []

        def __getitem__(self, item):
            fake_slice = _FakeStagingSlice(item.start, item.stop)
            self.slices.append(fake_slice)
            return fake_slice

    gather_calls = []
    monkeypatch.setattr(
        b10_copy_engine,
        "_gather_vram_spans_to_pinned_staging",
        lambda *args, **kwargs: gather_calls.append((args, kwargs)) or True,
    )

    min_spans = b10_pools._DEFAULT_SEND_GATHER_MIN_SPANS

    # Eligible send chunk: the gather kernel path is taken, the loop is not.
    descs, spans = _gather_span_case([8] * min_spans)
    chunk = b10_memory._TransferChunk(start=0, count=len(descs), size=sum(d.size for d in descs))
    staging = _FakeLoopStagingTensor()
    staging_view = b10_memory._BufferView(staging, staging)
    lifetime_refs = []
    span_count, copy_devices = agent._copies.copy_chunk(
        descs,
        chunk,
        "VRAM",
        staging,
        copy_from_staging=False,
        lifetime_refs=lifetime_refs,
        staging_view=staging_view,
        source_ready_events=[(device, source_ready_event)],
    )
    assert span_count == len(spans)
    assert copy_devices == [device]
    assert len(gather_calls) == 1
    gather_args = gather_calls[0][0]
    assert gather_args[0] is descs
    assert list(gather_args[1]) == list(spans)
    assert gather_args[2:] == (staging, fake_copy_stream, lifetime_refs)
    assert fake_copy_stream.waited_events == [source_ready_event]
    assert span_buffers == {} and staging.slices == []
    # The gather kernel's staging writes are not allocator-tracked, so the
    # copy-stream event must be attached for quarantine pruning.
    assert isinstance(staging_view.ready_event, _FakeCudaEvent)
    assert staging_view.ready_event.recorded_streams == [fake_copy_stream]

    # Below the span threshold the trimmed loop runs: one stream context per
    # chunk, direct copy_ calls, zero-size spans skipped, D2H non_blocking.
    gather_calls.clear()
    fake_copy_stream.waited_events.clear()
    descs, spans = _gather_span_case([16, 0, 24])
    chunk = b10_memory._TransferChunk(start=0, count=len(descs), size=sum(d.size for d in descs))
    staging = _FakeLoopStagingTensor()
    lifetime_refs = []
    span_count, copy_devices = agent._copies.copy_chunk(
        descs,
        chunk,
        "VRAM",
        staging,
        copy_from_staging=False,
        lifetime_refs=lifetime_refs,
        source_ready_events=[(device, source_ready_event)],
    )
    assert gather_calls == []
    assert span_count == 3
    assert copy_devices == [device, device]
    assert stream_contexts == [("enter", fake_copy_stream), ("exit", fake_copy_stream)]
    assert [(s.start, s.stop) for s in staging.slices] == [(0, 16), (16, 40)]
    assert [s.copies for s in staging.slices] == [
        [(span_buffers[0], True)],
        [(span_buffers[16], True)],
    ]
    assert [ref.buffer for ref in lifetime_refs] == [span_buffers[0], span_buffers[16]]

    # copy_from_staging chunks never gather, even when otherwise eligible:
    # the loop copies staging slices into the span views.
    stream_contexts.clear()
    span_buffers.clear()
    descs, spans = _gather_span_case([8] * min_spans)
    chunk = b10_memory._TransferChunk(start=0, count=len(descs), size=sum(d.size for d in descs))
    staging = _FakeLoopStagingTensor()
    span_count, copy_devices = agent._copies.copy_chunk(
        descs, chunk, "VRAM", staging, copy_from_staging=True
    )
    assert gather_calls == []
    assert span_count == min_spans
    assert copy_devices == [device] * min_spans
    assert stream_contexts == [("enter", fake_copy_stream), ("exit", fake_copy_stream)]
    assert all(
        buffer.copies == [(staging.slices[idx], True)]
        for idx, (offset, buffer) in enumerate(sorted(span_buffers.items()))
    )


def test_b10_prune_quarantined_views_waits_for_pending_ready_event():
    class _FakeReadyEvent:
        def __init__(self):
            self.ready = False

        def query(self):
            return self.ready

    agent = _make_uninitialized_b10_agent()
    agent._core._staging_quarantine_ttl_s = 0.0  # every entry is past the TTL
    event = _FakeReadyEvent()
    quarantined_at = time.monotonic() - 1.0
    pending = b10_pools._QuarantinedBufferViews(
        views=[b10_memory._BufferView(object(), object(), ready_event=event)],
        quarantined_at=quarantined_at,
    )
    eventless = b10_pools._QuarantinedBufferViews(
        views=[b10_memory._BufferView(object(), object())], quarantined_at=quarantined_at
    )
    agent._core._quarantined_staging_views = [pending, eventless]

    # The un-fired gather-kernel event keeps its entry alive past the TTL
    # (the kernel may still be pending on a wedged stream); entries without
    # events keep the plain TTL behavior.
    agent._core._prune_quarantined_staging_buffers()
    assert agent._core._quarantined_staging_views == [pending]

    event.ready = True
    agent._core._prune_quarantined_staging_buffers()
    assert agent._core._quarantined_staging_views == []


def test_b10_send_gather_byte_kernel_tripwire_warns_once(monkeypatch):
    launches = []
    _install_scatter_kernel_fakes(monkeypatch, launches)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(b10_kernels, "_send_gather_byte_kernel_warned", False)
    warnings = []
    monkeypatch.setattr(b10_kernels.logger, "warning", warnings.append)

    word = b10_kernels._SCATTER_ALIGNED_WORD_BYTES
    descs, spans = _gather_span_case([word, 1])
    assert b10_kernels._gather_vram_spans_to_pinned_staging(
        descs, spans, _FakePinnedStagingTensor(), object()
    )
    assert [launch["kernel"] for launch in launches] == ["request_byte"]
    assert len(warnings) == 1
    assert "byte kernel" in warnings[0]
    assert "misaligned_fragments=1/2" in warnings[0]

    # Once per process: a second byte-kernel gather stays silent.
    assert b10_kernels._gather_vram_spans_to_pinned_staging(
        descs, spans, _FakePinnedStagingTensor(), object()
    )
    assert len(warnings) == 1

    # Aligned gathers never trip the warning.
    monkeypatch.setattr(b10_kernels, "_send_gather_byte_kernel_warned", False)
    launches.clear()
    descs, spans = _gather_span_case([word, word])
    assert b10_kernels._gather_vram_spans_to_pinned_staging(
        descs, spans, _FakePinnedStagingTensor(), object()
    )
    assert [launch["kernel"] for launch in launches] == ["request_u64"]
    assert len(warnings) == 1


def test_b10_desc_view_from_arrays_matches_normalize_fallback():
    ptrs = np.array([4096, 8192, 12288, 40960], dtype=np.int64)
    sizes = np.array([128, 256, 128, 512], dtype=np.int64)
    descs = base_agent.MemoryDescs(
        base_agent.MemoryType.VRAM, [(int(p), int(s), 3) for p, s in zip(ptrs, sizes)]
    )

    fallback = b10_planning._normalize_memory_descs(descs)
    side = b10_planning._desc_view_from_arrays(ptrs, sizes, 3)

    assert isinstance(side, b10_memory._DescArrayView)
    assert np.array_equal(side.ptrs, fallback.ptrs)
    assert np.array_equal(side.sizes, fallback.sizes)
    assert np.array_equal(side.device_ids, fallback.device_ids)
    assert side.device_ids.dtype == np.int64
    # int64 inputs are aliased, not copied (the producer never mutates them).
    assert side.ptrs is ptrs
    assert side.sizes is sizes

    per_desc = b10_planning._desc_view_from_arrays(ptrs, sizes, np.full(4, 3, dtype=np.int64))
    assert np.array_equal(per_desc.device_ids, fallback.device_ids)

    empty = np.array([], dtype=np.int64)
    empty_view = b10_planning._desc_view_from_arrays(empty, empty, 0)
    assert len(empty_view) == 0


def test_b10_submit_side_channel_and_fallback_deliver_identical_views():
    agent = _make_uninitialized_b10_agent()
    captured = []
    agent._send._submit_transfer_request = (
        lambda request, src_descs, dst_descs: captured.append((src_descs, dst_descs)) or "status"
    )

    src_ptrs = np.array([4096, 8192, 12288], dtype=np.int64)
    dst_ptrs = np.array([1 << 20, (1 << 20) + 128, 1 << 21], dtype=np.int64)
    sizes = np.array([128, 128, 64], dtype=np.int64)
    src_descs = base_agent.MemoryDescs(
        base_agent.MemoryType.VRAM, [(int(p), int(s), 0) for p, s in zip(src_ptrs, sizes)]
    )
    dst_descs = base_agent.MemoryDescs(
        base_agent.MemoryType.VRAM, [(int(p), int(s), 1) for p, s in zip(dst_ptrs, sizes)]
    )
    request = base_agent.TransferRequest(
        base_agent.TransferOp.WRITE, src_descs, dst_descs, "remote"
    )

    assert agent.submit_transfer_requests(request) == "status"
    assert (
        agent.submit_transfer_requests_with_desc_arrays(
            request, (src_ptrs, sizes, 0), (dst_ptrs, sizes, 1)
        )
        == "status"
    )

    (fb_src, fb_dst), (sc_src, sc_dst) = captured
    for fallback, side in ((fb_src, sc_src), (fb_dst, sc_dst)):
        assert np.array_equal(side.ptrs, fallback.ptrs)
        assert np.array_equal(side.sizes, fallback.sizes)
        assert np.array_equal(side.device_ids, fallback.device_ids)


def test_b10_submit_side_channel_produces_identical_send_plan():
    agent = _make_uninitialized_b10_agent()
    agent._core.staging_buffer_pool.buffer_size = 1 << 20
    agent._endpoints._remote_agents = {
        "peer": b10_protocol.B10AgentDescriptor(
            name="peer",
            host="10.0.0.3",
            port=1,
            tag_domain=0,
            features=(b10_protocol._FEATURE_PACKED_DESCS,),
        )
    }

    # Fragmented production shape: contiguous runs of 4 blocks with gaps.
    n = 64
    block = 512
    src_ptrs = (
        np.arange(n, dtype=np.int64) * block
        + (np.arange(n, dtype=np.int64) // 4) * (8 * block)
        + 4096
    )
    dst_ptrs = src_ptrs[::-1].copy() + (1 << 30)
    sizes = np.full(n, block, dtype=np.int64)
    src_mds = base_agent.MemoryDescs(
        base_agent.MemoryType.VRAM, [(int(p), int(s), 0) for p, s in zip(src_ptrs, sizes)]
    )
    dst_mds = base_agent.MemoryDescs(
        base_agent.MemoryType.VRAM, [(int(p), int(s), 1) for p, s in zip(dst_ptrs, sizes)]
    )

    plans = []
    for src_view, dst_view in (
        (
            b10_planning._normalize_memory_descs(src_mds),
            b10_planning._normalize_memory_descs(dst_mds),
        ),
        (
            b10_planning._desc_view_from_arrays(src_ptrs, sizes, 0),
            b10_planning._desc_view_from_arrays(dst_ptrs, sizes, 1),
        ),
    ):
        plans.append(
            agent._send._build_send_transfer_plan("peer", 7, "VRAM", src_view, "VRAM", dst_view)
        )
    fallback_plan, side_plan = plans

    for name in (
        "remote_name",
        "transfer_id",
        "src_type",
        "dst_type",
        "transfer_chunks",
        "desc_count",
        "total_bytes",
        "max_desc_size",
        "wire_chunk_count",
        "max_wire_chunk_size",
        "desc_order_strategy",
        "src_span_count",
        "dst_span_count",
        "sync_message",
    ):
        assert getattr(side_plan, name) == getattr(fallback_plan, name), name
    for side_view, fallback_view in (
        (side_plan.src_descs, fallback_plan.src_descs),
        (side_plan.dst_descs, fallback_plan.dst_descs),
    ):
        assert np.array_equal(side_view.ptrs, fallback_view.ptrs)
        assert np.array_equal(side_view.sizes, fallback_view.sizes)
        assert np.array_equal(side_view.device_ids, fallback_view.device_ids)


def test_b10_side_channel_submit_keeps_vram_gating_and_empty_fastpath():
    agent = _make_uninitialized_b10_agent()
    empty = np.array([], dtype=np.int64)

    src_mds = base_agent.MemoryDescs(base_agent.MemoryType.VRAM, [(1, 1, 0)])
    dst_mds = base_agent.MemoryDescs(base_agent.MemoryType.VRAM, [(2, 1, 0)])
    request = base_agent.TransferRequest(base_agent.TransferOp.WRITE, src_mds, dst_mds, "remote")
    one = np.array([1], dtype=np.int64)
    # VRAM send without sync metadata fails exactly like the fallback path.
    status = agent.submit_transfer_requests_with_desc_arrays(
        request, (one, one, 0), (np.array([2], dtype=np.int64), one, 0)
    )
    assert status.is_completed()
    assert not status.wait()

    empty_request = base_agent.TransferRequest(
        base_agent.TransferOp.WRITE,
        base_agent.MemoryDescs(base_agent.MemoryType.VRAM, []),
        base_agent.MemoryDescs(base_agent.MemoryType.VRAM, []),
        "remote",
    )
    status = agent.submit_transfer_requests_with_desc_arrays(
        empty_request, (empty, empty, 0), (empty, empty, 0)
    )
    assert status.is_completed()
    assert status.wait()


def test_b10_desc_view_from_arrays_rejects_malformed_side_channel():
    ptrs = np.array([1, 2, 3], dtype=np.int64)
    sizes = np.ones(3, dtype=np.int64)

    with pytest.raises(ValueError, match="malformed"):
        b10_planning._desc_view_from_arrays(ptrs, sizes[:2], 0)
    with pytest.raises(ValueError, match="malformed"):
        b10_planning._desc_view_from_arrays(ptrs, sizes, np.array([0, 1], dtype=np.int64))
    with pytest.raises(ValueError, match="malformed"):
        b10_planning._desc_view_from_arrays(ptrs.reshape(3, 1), np.ones((3, 1), dtype=np.int64), 0)

    # src/dst descriptor-count mismatch still raises through the
    # side-channel submit (the cross-check that needs no nanobind walk).
    agent = _make_uninitialized_b10_agent()
    src_mds = base_agent.MemoryDescs(base_agent.MemoryType.VRAM, [(1, 1, 0), (9, 1, 0)])
    dst_mds = base_agent.MemoryDescs(base_agent.MemoryType.VRAM, [(2, 1, 0)])
    request = base_agent.TransferRequest(base_agent.TransferOp.WRITE, src_mds, dst_mds, "remote")
    with pytest.raises(ValueError, match="count mismatch"):
        agent.submit_transfer_requests_with_desc_arrays(
            request,
            (np.array([1, 9], dtype=np.int64), np.ones(2, dtype=np.int64), 0),
            (np.array([2], dtype=np.int64), np.ones(1, dtype=np.int64), 0),
        )


def _make_b10_native_sender():
    sender = native_transfer.Sender.__new__(native_transfer.Sender)
    sender._agent = B10CacheTransferAgent
    sender._sessions = {}
    sender._sessions_lock = threading.Lock()
    sender._send_failed_result_to_receiver = Mock()
    sender._save_peer_req_info = Mock()
    request = native_transfer.RecvReqInfo(
        sender_req_id=1,
        instance_name="decode",
        instance_rank=0,
        block_ids_per_layer_groups=[],
        unique_rid=42,
    )
    return sender, request


def test_b10_request_without_prefill_session_fails_immediately():
    sender, request = _make_b10_native_sender()

    sender._respond_with_kv(b"", [native_transfer.MessageType.REQUEST_DATA, request.to_bytes()])

    sender._send_failed_result_to_receiver.assert_called_once_with(request)
    sender._save_peer_req_info.assert_not_called()


def test_b10_request_after_session_creation_is_dispatched_by_send():
    sender, request = _make_b10_native_sender()
    pending_requests = {}
    sender._save_peer_req_info.side_effect = lambda info: pending_requests.update(
        {info.instance_rank: info}
    )
    sender._get_req_info = lambda _rid: pending_requests
    sender.dispatch_task = Mock()

    session = native_transfer.TxSession.__new__(native_transfer.TxSession)
    session._sender = sender
    session._base_args = types.SimpleNamespace(
        params=types.SimpleNamespace(disagg_request_id=request.unique_rid),
        prompt_len=None,
        beam_width=1,
    )
    session.request_id = request.unique_rid
    session.receiver_ready = False
    session.kv_tasks = []
    session.aux_task = None
    session.lock = threading.Lock()
    session._exception = None
    session._terminal_status = None
    sender._sessions[request.unique_rid] = lambda: session

    sender._respond_with_kv(b"", [native_transfer.MessageType.REQUEST_DATA, request.to_bytes()])
    session.send(Mock())

    sender._send_failed_result_to_receiver.assert_not_called()
    sender.dispatch_task.assert_called_once()
    assert sender.dispatch_task.call_args.args[0] is session.kv_tasks[0]
    assert sender.dispatch_task.call_args.args[1] == {request.instance_rank: request}


def test_b10_request_for_cancelled_prefill_session_fails_immediately():
    sender, request = _make_b10_native_sender()
    session = types.SimpleNamespace(
        lock=threading.Lock(),
        kv_tasks=[Mock()],
        status=native_transfer.SessionStatus.CANCELLED,
    )
    sender._sessions[request.unique_rid] = lambda: session
    sender._build_kv_write_meta = Mock()

    sender._respond_with_kv(b"", [native_transfer.MessageType.REQUEST_DATA, request.to_bytes()])

    sender._send_failed_result_to_receiver.assert_called_once_with(request)
    sender._save_peer_req_info.assert_called_once_with(request)
    sender._build_kv_write_meta.assert_not_called()


def test_b10_queued_request_fails_if_prefill_session_closes():
    sender = native_transfer.Sender.__new__(native_transfer.Sender)
    sender._agent = B10CacheTransferAgent
    sender._instance_rank = 3
    sender._sessions = {}
    sender._sessions_lock = threading.Lock()
    dealer = Mock()
    sender._get_or_connect_thread_dealer = Mock(return_value=dealer)
    task = Mock()
    write_meta = native_transfer.WriteMeta(
        task=task,
        expected_transfers=1,
        peer_name="decode0",
        peer_rank=0,
        peer_endpoint="tcp://decode",
        unique_rid=42,
        src_ptrs=np.array([], dtype=np.int64),
        dst_ptrs=np.array([], dtype=np.int64),
        sizes=np.array([], dtype=np.int64),
        slice_id=0,
    )

    sender._deliver_kv_to_agent(write_meta)

    task.fail.assert_called_once()
    dealer.send.assert_called_once_with(
        [
            native_transfer.MessageType.KV_AGENT_RESULT,
            b"3",
            b"42",
            b"0",
            b"True",
            native_transfer.AgentResult.FAILED.value.encode("ascii"),
        ]
    )


def test_native_sender_side_channel_seam_passes_writemeta_arrays(monkeypatch):
    """Lock the native Sender's descriptor side-channel seam.

    The native Sender hands the WriteMeta descriptor columns to a
    side-channel-capable agent unmodified (aliased, not copied), paired
    with the devices _make_agent_request derived; agents without the side
    channel get the identical TransferRequest via the plain fallback.
    """

    class _SeamTransferStatus:
        def __init__(self):
            self.wait_calls = 0

        def is_completed(self):
            return True

        def wait(self, timeout_ms=None):
            self.wait_calls += 1
            return True

    class _SeamTask:
        def __init__(self):
            self.registered = []
            self.cleared = []

        def set_agent_status(self, status):
            self.registered.append(status)

        def clear_agent_status(self, status):
            self.cleared.append(status)

    # Record the TransferRequest _make_agent_request builds so both submit
    # paths can be checked for identity against it.
    made_requests = []
    real_make_agent_request = native_transfer.Sender._make_agent_request

    def _recording_make_agent_request(write_meta, device_id, sync_message=None):
        result = real_make_agent_request(write_meta, device_id=device_id, sync_message=sync_message)
        made_requests.append(result[0])
        return result

    monkeypatch.setattr(
        native_transfer.Sender, "_make_agent_request", staticmethod(_recording_make_agent_request)
    )

    def _make_sender(agent, device_id):
        # Sender.__init__ spins up ZMQ sockets and worker threads; build a
        # bare instance and set only what _submit_and_wait_agent_write
        # reads, probing the side channel exactly like __init__ does.
        sender = native_transfer.Sender.__new__(native_transfer.Sender)
        sender._agent = agent
        sender._device_id = device_id
        sender._submit_with_desc_arrays = getattr(
            agent, "submit_transfer_requests_with_desc_arrays", None
        )
        return sender

    def _make_write_meta(meta_type, task, dst_device_id):
        return native_transfer.WriteMeta(
            task=task,
            expected_transfers=1,
            peer_name="peer0",
            peer_rank=0,
            peer_endpoint="tcp://127.0.0.1:5555",
            unique_rid=42,
            src_ptrs=np.array([4096, 8192, 12288], dtype=np.int64),
            dst_ptrs=np.array([1 << 20, (1 << 20) + 256, 1 << 21], dtype=np.int64),
            sizes=np.array([256, 256, 128], dtype=np.int64),
            dst_device_id=dst_device_id,
            meta_type=meta_type,
        )

    class _SideChannelAgent:
        supports_request_sync_message_metadata = True

        def __init__(self):
            self.calls = []

        def submit_transfer_requests_with_desc_arrays(self, request, src_arrays, dst_arrays):
            status = _SeamTransferStatus()
            self.calls.append((request, src_arrays, dst_arrays, status))
            return status

        def submit_transfer_requests(self, request):
            raise AssertionError("side-channel agent must not take the fallback path")

    # KV-shaped meta: VRAM transfer between distinct devices.
    agent = _SideChannelAgent()
    sender = _make_sender(agent, device_id=3)
    task = _SeamTask()
    kv_meta = _make_write_meta(native_transfer.WriteMetaType.KV, task, dst_device_id=5)
    assert sender._submit_and_wait_agent_write(task, kv_meta)
    ((request, src_arrays, dst_arrays, status),) = agent.calls
    assert request is made_requests[-1]
    assert request.sync_message == "42"
    assert request.src_descs.type == base_agent.MemoryType.VRAM
    assert request.dst_descs.type == base_agent.MemoryType.VRAM
    assert src_arrays[0] is kv_meta.src_ptrs
    assert src_arrays[1] is kv_meta.sizes
    assert src_arrays[2] == 3
    assert dst_arrays[0] is kv_meta.dst_ptrs
    assert dst_arrays[1] is kv_meta.sizes
    assert dst_arrays[2] == 5
    assert status.wait_calls == 1
    assert task.registered == [status]
    assert task.cleared == [status]

    # AUX-shaped meta: DRAM transfer, both devices pinned to 0 regardless
    # of the sender's device.
    agent = _SideChannelAgent()
    sender = _make_sender(agent, device_id=3)
    task = _SeamTask()
    aux_meta = _make_write_meta(native_transfer.WriteMetaType.AUX, task, dst_device_id=None)
    assert sender._submit_and_wait_agent_write(task, aux_meta)
    ((request, src_arrays, dst_arrays, status),) = agent.calls
    assert request is made_requests[-1]
    assert request.src_descs.type == base_agent.MemoryType.DRAM
    assert request.dst_descs.type == base_agent.MemoryType.DRAM
    assert src_arrays[0] is aux_meta.src_ptrs
    assert src_arrays[1] is aux_meta.sizes
    assert src_arrays[2] == 0
    assert dst_arrays[0] is aux_meta.dst_ptrs
    assert dst_arrays[1] is aux_meta.sizes
    assert dst_arrays[2] == 0
    assert task.registered == [status]
    assert task.cleared == [status]

    # An agent without the side channel gets the identical request through
    # the plain submit_transfer_requests fallback.
    class _PlainAgent:
        def __init__(self):
            self.requests = []
            self.statuses = []

        def submit_transfer_requests(self, request):
            self.requests.append(request)
            status = _SeamTransferStatus()
            self.statuses.append(status)
            return status

    plain_agent = _PlainAgent()
    sender = _make_sender(plain_agent, device_id=3)
    assert sender._submit_with_desc_arrays is None
    task = _SeamTask()
    kv_meta = _make_write_meta(native_transfer.WriteMetaType.KV, task, dst_device_id=5)
    assert sender._submit_and_wait_agent_write(task, kv_meta)
    assert len(plain_agent.requests) == 1
    assert plain_agent.requests[0] is made_requests[-1]
    # _PlainAgent does not advertise sync-message support, so the request
    # is built without one.
    assert plain_agent.requests[0].sync_message is None
    assert plain_agent.statuses[0].wait_calls == 1
    assert task.registered == plain_agent.statuses
    assert task.cleared == plain_agent.statuses
