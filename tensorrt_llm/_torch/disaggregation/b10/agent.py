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

import asyncio
import os
import threading
from typing import Any, Callable, Optional

import numpy as np
import torch

from tensorrt_llm import logger
from tensorrt_llm._torch.disaggregation.b10 import net as b10_net
from tensorrt_llm._torch.disaggregation.b10.am import B10AmDispatcher
from tensorrt_llm._torch.disaggregation.b10.config import B10AgentConfig
from tensorrt_llm._torch.disaggregation.b10.copy_engine import _CopyEngine
from tensorrt_llm._torch.disaggregation.b10.core import _AgentCore
from tensorrt_llm._torch.disaggregation.b10.endpoints import EndpointPool
from tensorrt_llm._torch.disaggregation.b10.kernels import (
    _warm_absolute_kernel_parity_specializations,
    _warm_scatter_kernels,
)
from tensorrt_llm._torch.disaggregation.b10.pools import (
    _AmStagingAllocator,
    _CudaCopyStreamPool,
    _CudaScratchBufferPool,
    _format_cuda_scratch_pool_state,
    _format_staging_pool_state,
    _PinnedStagingBufferPool,
)
from tensorrt_llm._torch.disaggregation.b10.protocol import (
    _FEATURE_PACKED_DESCS,
    B10AgentDescriptor,
    B10TransferIdAllocator,
)
from tensorrt_llm._torch.disaggregation.b10.recv import RecvPipeline
from tensorrt_llm._torch.disaggregation.b10.send import SendPipeline
from tensorrt_llm._torch.disaggregation.b10.timings import TransferTracer
from tensorrt_llm._torch.disaggregation.base.agent import (
    BaseTransferAgent,
    MemoryDescs,
    RegMemoryDescs,
    TransferRequest,
    TransferStatus,
)


class B10CacheTransferAgent(BaseTransferAgent):
    """UCXX-backed transfer agent used by B10CacheTransceiver.

    The native Python transceiver stack still owns session and KV slicing
    semantics. This class owns only the data-plane WRITE operation and its
    UCXX tag lifecycle.
    """

    supports_request_sync_message_metadata = True
    fail_request_without_active_send_session = True

    def __init__(
        self,
        name: str,
        *,
        ucxx_module: Optional[Any] = None,
        port: Optional[int] = None,
        endpoint_pool_size: Optional[int] = None,
        max_in_flight_ops: Optional[int] = None,
        tag_space_size: Optional[int] = None,
        tag_quarantine_ttl_s: Optional[float] = None,
        transfer_timeout_s: Optional[float] = None,
        staging_pool_num_buffers: Optional[int] = None,
        staging_pool_buffer_size: Optional[int] = None,
        recv_scratch_pool_num_buffers: Optional[int] = None,
        send_admission_limit: Optional[int] = None,
        send_admission_bypass_bytes: Optional[int] = None,
    ):
        self.name = name
        self._advertised_ifname = b10_net._advertised_ifname_from_ucx_net_devices()
        b10_net._augment_ucx_net_devices_for_sockaddr()
        self._ucxx = ucxx_module or b10_net._load_ucxx_module()
        cfg = B10AgentConfig.from_env(
            port=port,
            endpoint_pool_size=endpoint_pool_size,
            max_in_flight_ops=max_in_flight_ops,
            tag_space_size=tag_space_size,
            tag_quarantine_ttl_s=tag_quarantine_ttl_s,
            transfer_timeout_s=transfer_timeout_s,
            staging_pool_num_buffers=staging_pool_num_buffers,
            staging_pool_buffer_size=staging_pool_buffer_size,
            recv_scratch_pool_num_buffers=recv_scratch_pool_num_buffers,
            send_admission_limit=send_admission_limit,
            send_admission_bypass_bytes=send_admission_bypass_bytes,
        )
        self._port = cfg.port

        tag_space = cfg.tag_space_size
        quarantine_ttl_s = cfg.tag_quarantine_ttl_s
        transfer_ids = B10TransferIdAllocator(
            tag_space_size=tag_space,
            quarantine_ttl_s=quarantine_ttl_s,
        )
        pool_num_buffers = cfg.staging_pool_num_buffers
        pool_buffer_size = cfg.staging_pool_buffer_size
        staging_buffer_pool = _PinnedStagingBufferPool(
            num_buffers=pool_num_buffers,
            buffer_size=pool_buffer_size,
        )
        staging_buffer_slots = asyncio.BoundedSemaphore(pool_num_buffers)
        # Landing pad for eager AM receives (registered with the ucxx worker
        # in _start_am_plane); a negative min-bytes knob disables it.
        self._am_staging_allocator = (
            _AmStagingAllocator(staging_buffer_pool, cfg.am_direct_staging_min_bytes)
            if cfg.am_direct_staging_min_bytes >= 0
            else None
        )
        scratch_pool_num_buffers = cfg.recv_scratch_pool_num_buffers
        scratch_metadata_max_spans = cfg.recv_scratch_metadata_max_spans
        recv_scratch_buffer_pool = _CudaScratchBufferPool(
            num_buffers=scratch_pool_num_buffers,
            buffer_size=staging_buffer_pool.buffer_size,
            metadata_max_spans=scratch_metadata_max_spans,
        )
        self._core = _AgentCore(
            cfg,
            loop=asyncio.new_event_loop(),
            transfer_ids=transfer_ids,
            staging_buffer_pool=staging_buffer_pool,
            staging_buffer_slots=staging_buffer_slots,
            recv_scratch_buffer_pool=recv_scratch_buffer_pool,
            cuda_copy_streams=_CudaCopyStreamPool(),
        )
        self._endpoints = EndpointPool(
            self._core,
            ucxx=self._ucxx,
            endpoint_pool_size=cfg.endpoint_pool_size,
        )
        self._tracer = TransferTracer(
            self._core,
            trace_transfer_level=cfg.trace_transfer_level,
        )
        self._copies = _CopyEngine(self._core)
        # One AM dispatcher per agent routes control/READY/RESULT/DATA on the
        # agent loop. _start_am_plane attaches it to the worker's permanent
        # receiver callback.
        self._dispatcher = B10AmDispatcher(self._core.loop)
        self._recv = RecvPipeline(
            self._core,
            self._copies,
            self._tracer,
            self._dispatcher,
            self._ucxx,
            am_staging_allocator=self._am_staging_allocator,
        )
        self._send = SendPipeline(
            self._core,
            self._copies,
            self._endpoints,
            self._tracer,
            self._dispatcher,
            validate_send_source=cfg.validate_send_source,
            send_admission_limit=cfg.send_admission_limit,
            send_admission_bypass_bytes=cfg.send_admission_bypass_bytes,
        )
        self._warm_scatter_kernels_at_startup()

        self._loop_started = threading.Event()
        self._loop_thread = threading.Thread(
            target=self._run_loop, name=f"b10-ucxx-{name}", daemon=True
        )
        self._loop_thread.start()
        self._loop_started.wait()

        descriptor_future = asyncio.run_coroutine_threadsafe(
            self._start_am_plane(), self._core.loop
        )
        try:
            self._descriptor = descriptor_future.result(timeout=cfg.startup_timeout_s)
        except Exception:
            self.shutdown()
            raise
        logger.info(
            f"B10 UCXX agent ready: name={self.name} "
            f"host={self._descriptor.host} port={self._descriptor.port} "
            f"advertised_ifname={self._advertised_ifname} "
            f"transfer_timeout_s={self._core.transfer_timeout_s} "
            f"endpoint_pool_size={self._endpoints._endpoint_pool_size} "
            f"max_in_flight_ops={self._core.max_in_flight_ops} "
            f"send_admission_limit={cfg.send_admission_limit} "
            f"send_admission_bypass_bytes={cfg.send_admission_bypass_bytes} "
            f"tag_space_size={tag_space} "
            f"tag_quarantine_ttl_s={quarantine_ttl_s} "
            f"trace_transfers={self._tracer.level} "
            f"validate_send_source={cfg.validate_send_source} "
            f"ucx_net_devices={os.getenv('UCX_NET_DEVICES', '')} "
            f"{_format_staging_pool_state(self._core.staging_buffer_pool)} "
            f"{_format_cuda_scratch_pool_state(self._core.recv_scratch_buffer_pool)}"
        )

    # BaseTransferAgent submit surface and the source-ready-event hooks the
    # transceiver drives: permanent thin delegators to the send pipeline.

    def submit_transfer_requests(self, request: TransferRequest) -> TransferStatus:
        return self._send.submit_transfer_requests(request)

    def submit_transfer_requests_with_desc_arrays(
        self,
        request: TransferRequest,
        src_arrays: tuple[np.ndarray, np.ndarray, int | np.ndarray],
        dst_arrays: tuple[np.ndarray, np.ndarray, int | np.ndarray],
    ) -> TransferStatus:
        # Discovered by callers via ``getattr(agent, ..., None)`` (see the
        # SendPipeline method for the side-channel contract), so this
        # delegator must exist exactly when the send pipeline supports it.
        return self._send.submit_transfer_requests_with_desc_arrays(request, src_arrays, dst_arrays)

    def record_source_ready_event(self, request_id: int, *, stream: Optional[Any] = None) -> None:
        self._send.record_source_ready_event(request_id, stream=stream)

    def discard_source_ready_event(self, request_id: int) -> None:
        self._send.discard_source_ready_event(request_id)

    # Recv surface the transceiver's timeout recovery and the native
    # Receiver drive: permanent thin delegators to the recv pipeline.

    def cancel_recv_request(self, request_id: int) -> None:
        self._recv.cancel_recv_request(request_id)

    def has_active_recv_request(self, request_id: int) -> bool:
        return self._recv.has_active_recv_request(request_id)

    def has_active_recv_copy_request(self, request_id: int) -> bool:
        return self._recv.has_active_recv_copy_request(request_id)

    def set_incoming_write_listener(self, listener: Callable[[int, bool], None]) -> None:
        # Discovered by the native Receiver via ``getattr(agent, ..., None)``
        # (see the RecvPipeline method for the listener contract), so this
        # plain delegator must exist exactly when the recv pipeline
        # supports it.
        self._recv.set_incoming_write_listener(listener)

    def register_memory(self, descs: RegMemoryDescs) -> None:
        pass

    def deregister_memory(self, descs: RegMemoryDescs) -> None:
        pass

    def load_remote_agent(self, name: str, agent_desc: bytes) -> None:
        descriptor = B10AgentDescriptor.from_bytes(agent_desc)
        previous = self._endpoints._remote_agents.get(name)
        self._endpoints._remote_agents[name] = descriptor
        if previous != descriptor and name in self._endpoints._remote_slots:
            future = asyncio.run_coroutine_threadsafe(
                self._endpoints._clear_remote_slot_endpoints(name), self._core.loop
            )
            future.result(timeout=5)

    def get_local_agent_desc(self) -> bytes:
        return self._descriptor.to_bytes()

    def invalidate_remote_agent(self, name: str) -> None:
        self._endpoints._remote_agents.pop(name, None)
        future = asyncio.run_coroutine_threadsafe(
            self._endpoints._drop_remote_slots(name), self._core.loop
        )
        future.result(timeout=5)

    def notify_sync_message(self, name: str, sync_message: str) -> None:
        raise NotImplementedError("B10 sync messages are not implemented")

    def check_remote_descs(self, name: str, memory_descs: MemoryDescs) -> bool:
        return name in self._endpoints._remote_agents

    def _warm_scatter_kernels_at_startup(self) -> None:
        # Request-level receive scatter runs regardless of the per-chunk
        # receive-scratch configuration, so always warm its absolute kernels.
        if not torch.cuda.is_available():
            return
        # Same device choice as _preallocate_recv_scratch_buffers.
        warm_device = (
            self._recv._recv_scratch_device
            if self._recv._recv_scratch_device is not None
            else torch.device("cuda", torch.cuda.current_device())
        )
        try:
            copy_stream = self._core.cuda_copy_streams.stream_for(warm_device)
            if self._recv._recv_scratch_device is not None:
                _warm_scatter_kernels(warm_device, copy_stream)
            else:
                _warm_absolute_kernel_parity_specializations(warm_device, copy_stream)
                copy_stream.synchronize()
        except Exception as exc:
            # Warmup is an optimization: keep the current fail-at-first-use
            # semantics (e.g. Triton unavailable) instead of failing startup.
            logger.warning(
                "B10 scatter kernel warmup failed; kernels will JIT on the "
                f"first fragmented transfer: error={type(exc).__name__}: {exc}"
            )

    def shutdown(self) -> None:
        if self._core.shutdown:
            return
        self._core.shutdown = True
        if self._core.loop.is_running():
            future = asyncio.run_coroutine_threadsafe(self._shutdown_async(), self._core.loop)
            try:
                future.result(timeout=10)
            except Exception as exc:
                logger.warning(f"B10 shutdown failed: {exc}")
            self._core.loop.call_soon_threadsafe(self._core.loop.stop)
        self._loop_thread.join(timeout=10)

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._core.loop)
        self._loop_started.set()
        try:
            self._core.loop.run_forever()
        finally:
            pending = asyncio.all_tasks(self._core.loop)
            for task in pending:
                task.cancel()
            if pending:
                self._core.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            self._core.loop.close()

    async def _start_am_plane(self) -> B10AgentDescriptor:
        # First ucxx context touch in the process must happen here, on the
        # agent loop, so the Python-future notifier binds to it; rebinds a
        # foreign-bound pre-existing context. See net.py for the failure
        # mode this prevents.
        b10_net._bind_ucxx_python_future_notifier(self._ucxx)
        # No UCX listener: endpoints on both sides are created from worker
        # addresses (the only endpoint kind UCX failover supports), and all
        # inbound messages arrive via the worker-scoped AM receiver callback.
        self._dispatcher.attach(self._ucxx)
        register_am_host_allocator = getattr(self._ucxx, "register_am_host_allocator", None)
        if self._am_staging_allocator is not None and register_am_host_allocator is not None:
            # Eager AM receives land directly in pinned staging buffers,
            # skipping the ucxx-internal host buffer and the copy out of it.
            register_am_host_allocator(self._am_staging_allocator.allocate)
        elif self._am_staging_allocator is not None:
            logger.info(
                "B10 ucxx module has no register_am_host_allocator; AM "
                "receives use ucxx-internal buffers plus a staging copy"
            )
        return B10AgentDescriptor(
            name=self.name,
            host=self._ucxx.get_address(ifname=self._advertised_ifname),
            port=int(self._port),
            features=(_FEATURE_PACKED_DESCS,),
            worker_address=bytes(self._ucxx.get_worker_address()),
        )

    async def _shutdown_async(self) -> None:
        for address, endpoint in list(self._recv._reply_endpoints.items()):
            self._recv._drop_reply_endpoint(address, endpoint)
        for name in list(self._endpoints._remote_slots):
            await self._endpoints._drop_remote_slots(name)
        self._dispatcher.detach()


def create_b10_transfer_agent(
    name: str,
    transfer_timeout_s: Optional[float] = None,
    staging_pool_num_buffers: Optional[int] = None,
) -> B10CacheTransferAgent:
    return B10CacheTransferAgent(
        name,
        transfer_timeout_s=transfer_timeout_s,
        staging_pool_num_buffers=staging_pool_num_buffers,
    )
