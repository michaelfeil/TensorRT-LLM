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
import math
import os
from typing import Optional

import torch

from tensorrt_llm import logger
from tensorrt_llm._torch.disaggregation.b10.agent import create_b10_transfer_agent
from tensorrt_llm._torch.disaggregation.b10.mla_root_fanout import MLARootFanout
from tensorrt_llm._torch.disaggregation.b10.pools import _DEFAULT_STAGING_POOL_BUFFER_SIZE
from tensorrt_llm._torch.disaggregation.base.agent import BaseTransferAgent
from tensorrt_llm._torch.disaggregation.base.transfer import KVSlice, TokenRange, get_unique_rid
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import indexer_k_cache_enabled
from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2
from tensorrt_llm._torch.distributed.communicator import Distributed
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings import LlmRequestState
from tensorrt_llm.llmapi.llm_args import CacheTransceiverConfig
from tensorrt_llm.mapping import Mapping

_STAGING_POOL_NUM_BUFFERS_ENV = "TRTLLM_B10_UCXX_STAGING_POOL_NUM_BUFFERS"
_STAGING_POOL_BUFFER_SIZE_ENV = "TRTLLM_B10_UCXX_STAGING_POOL_BUFFER_SIZE_BYTES"


def _indexer_cache_bytes_per_token(kv_cache_manager: KVCacheManager) -> int:
    impl = getattr(kv_cache_manager, "impl", None)
    if impl is None or not indexer_k_cache_enabled(kv_cache_manager):
        return 0

    tokens_per_block = getattr(kv_cache_manager, "tokens_per_block", 0)
    if tokens_per_block <= 0:
        return 0

    get_indexer_pool = getattr(impl, "get_indexer_k_cache_pool", None)
    if get_indexer_pool is None:
        return 0

    indexer_pool = get_indexer_pool()
    slot_elements = math.prod(int(dim) for dim in indexer_pool.shape[1:])
    slot_bytes = slot_elements * int(indexer_pool.element_size())
    return (slot_bytes + tokens_per_block - 1) // tokens_per_block


def _derive_staging_pool_num_buffers(
    kv_cache_manager: KVCacheManager,
    max_tokens_in_buffer: Optional[int],
    staging_pool_buffer_size: int,
) -> Optional[tuple[int, int, int]]:
    if max_tokens_in_buffer is None or max_tokens_in_buffer <= 0:
        return None
    if staging_pool_buffer_size <= 0:
        return None

    layers = set(range(kv_cache_manager.num_local_layers))
    if not layers:
        return None

    bytes_per_token = kv_cache_manager._calculate_cache_bytes_per_token_for_layers(layers)
    bytes_per_token += _indexer_cache_bytes_per_token(kv_cache_manager)
    if bytes_per_token <= 0:
        return None

    tokens_per_buffer = max(1, staging_pool_buffer_size // bytes_per_token)
    num_buffers = max(1, math.ceil(max_tokens_in_buffer / tokens_per_buffer))
    return num_buffers, tokens_per_buffer, bytes_per_token


class B10CacheTransceiver(KvCacheTransceiverV2):
    def __init__(
        self,
        mapping: Mapping,
        dist: Distributed,
        kv_cache_manager: KVCacheManager,
        cache_transceiver_config: CacheTransceiverConfig,
    ):
        self._b10_transfer_agent = None
        self._b10_staging_pool_num_buffers = None
        if _STAGING_POOL_NUM_BUFFERS_ENV not in os.environ:
            staging_pool_buffer_size = int(
                os.getenv(_STAGING_POOL_BUFFER_SIZE_ENV, str(_DEFAULT_STAGING_POOL_BUFFER_SIZE))
            )
            try:
                derived = _derive_staging_pool_num_buffers(
                    kv_cache_manager,
                    cache_transceiver_config.max_tokens_in_buffer,
                    staging_pool_buffer_size,
                )
            except (AttributeError, IndexError, TypeError, ValueError) as exc:
                logger.warning(
                    "B10 could not derive staging pool buffers from "
                    f"max_tokens_in_buffer="
                    f"{cache_transceiver_config.max_tokens_in_buffer}: "
                    f"{exc}. Falling back to the agent default."
                )
                derived = None
            if derived is not None:
                (self._b10_staging_pool_num_buffers, tokens_per_buffer, bytes_per_token) = derived
                logger.info(
                    "B10 derived staging pool buffers: "
                    f"num_buffers={self._b10_staging_pool_num_buffers} "
                    f"max_tokens_in_buffer="
                    f"{cache_transceiver_config.max_tokens_in_buffer} "
                    f"tokens_per_buffer={tokens_per_buffer} "
                    f"bytes_per_token={bytes_per_token} "
                    f"buffer_size={staging_pool_buffer_size}"
                )

        super().__init__(mapping, dist, kv_cache_manager, cache_transceiver_config)
        self._mla_root_fanout = MLARootFanout.create_if_enabled(
            mapping, dist, kv_cache_manager, self._page_table, self._device_id
        )

    def _create_kv_slice(
        self,
        req,
        token_range: Optional[TokenRange] = None,
        is_last_slice: bool = True,
    ) -> KVSlice:
        kv_slice = super()._create_kv_slice(req, token_range, is_last_slice)
        if self._mla_root_fanout is not None and req.is_generation_only_request():
            return self._mla_root_fanout.register_receive_slice(req, kv_slice)
        return kv_slice

    def request_and_receive_sync(self, req) -> None:
        request_id = get_unique_rid(req)
        already_receiving = request_id in self._recv_sessions
        use_mla_fanout = self._mla_root_fanout is not None and req.is_generation_only_request()
        if use_mla_fanout:
            duplicate_sessions = self._dist.tp_allgather(already_receiving)
            if any(duplicate_sessions) and not all(duplicate_sessions):
                req.state = LlmRequestState.DISAGG_TRANS_ERROR
                raise RuntimeError(
                    "B10 MLA root fanout found inconsistent sync receive sessions: "
                    f"rid={request_id} already_receiving={duplicate_sessions}"
                )

        local_error = None
        try:
            super().request_and_receive_sync(req)
        except Exception as error:
            local_error = error

        if use_mla_fanout and not already_receiving:
            assert self._mla_root_fanout is not None
            self._mla_root_fanout.finish_sync_receive(req, local_error)
        if local_error is not None:
            raise local_error

    def _create_transfer_agent(self, name: str) -> BaseTransferAgent:
        timeout_s = (
            None if self.kv_transfer_timeout_ms is None else self.kv_transfer_timeout_ms / 1000.0
        )
        agent = create_b10_transfer_agent(
            name,
            transfer_timeout_s=timeout_s,
            staging_pool_num_buffers=self._b10_staging_pool_num_buffers,
        )
        self._b10_transfer_agent = agent
        return agent

    def _ensure_source_kv_on_device(self, req) -> None:
        resume_request = getattr(self._kv_cache_manager, "resume_request", None)
        if resume_request is None:
            return
        if not resume_request(req):
            raise RuntimeError(
                f"B10 failed to resume KV cache before transfer for request {req.py_request_id}"
            )

        kv_stream = getattr(self._kv_cache_manager, "_stream", None)
        if kv_stream is not None:
            kv_device = getattr(kv_stream, "device", None)
            current_stream = (
                torch.cuda.current_stream(kv_device)
                if kv_device is not None
                else torch.cuda.current_stream()
            )
            if current_stream is not kv_stream:
                current_stream.wait_stream(kv_stream)

    def _record_source_ready_event(self, request_id: int) -> None:
        if self._b10_transfer_agent is None:
            raise RuntimeError("B10 transfer agent must exist before recording source-ready events")
        source_stream = getattr(self._kv_cache_manager, "_stream", None)
        self._b10_transfer_agent.record_source_ready_event(request_id, stream=source_stream)

    def respond_and_send_async(self, req) -> None:
        rid = get_unique_rid(req)
        self._ensure_source_kv_on_device(req)
        if rid is None:
            raise RuntimeError("B10 send requires request sync metadata")
        self._record_source_ready_event(rid)
        try:
            super().respond_and_send_async(req)
        except Exception:
            if self._b10_transfer_agent is not None:
                self._b10_transfer_agent.discard_source_ready_event(rid)
            raise

    def check_context_transfer_status(
        self,
        at_least_request_num: Optional[int],
        mark_complete: bool = False,
        collect_kv_transfer_events: bool = False,
        timed_out_context_request_ids: Optional[list[int]] = None,
    ):
        py_request_ids = {rid: req.py_request_id for rid, req in self._send_reqs.items()}
        timed_out_ids = set(timed_out_context_request_ids or ())
        for rid, req in list(self._send_reqs.items()):
            if req.py_request_id in timed_out_ids:
                self._send_sessions[rid].cancel()

        result = super().check_context_transfer_status(
            at_least_request_num,
            mark_complete=mark_complete,
            collect_kv_transfer_events=collect_kv_transfer_events,
        )
        completed, failed, success_events, error_events = result
        if self._b10_transfer_agent is not None:
            for rid in completed + failed:
                self._b10_transfer_agent.discard_source_ready_event(rid)
        return (
            [py_request_ids.get(rid, rid) for rid in completed],
            [py_request_ids.get(rid, rid) for rid in failed],
            [(rank, py_request_ids.get(rid, rid)) for rank, rid in success_events],
            [(rank, py_request_ids.get(rid, rid)) for rank, rid in error_events],
        )

    def _has_active_recv_work(self, request_id: int) -> bool:
        agent = self._b10_transfer_agent
        return agent is not None and (
            agent.has_active_recv_request(request_id)
            or agent.has_active_recv_copy_request(request_id)
        )

    def check_gen_transfer_status(
        self,
        at_least_request_num: Optional[int],
        collect_kv_transfer_events: bool = False,
        timed_out_generation_request_ids: Optional[list[int]] = None,
    ):
        py_request_ids = {rid: req.py_request_id for rid, req in self._recv_reqs.items()}
        timed_out_ids = set(timed_out_generation_request_ids or ())
        for rid, req in list(self._recv_reqs.items()):
            if req.py_request_id in timed_out_ids:
                if self._b10_transfer_agent is not None:
                    self._b10_transfer_agent.cancel_recv_request(rid)
                self._recv_sessions[rid].cancel()
        result = super().check_gen_transfer_status(
            at_least_request_num, collect_kv_transfer_events=collect_kv_transfer_events
        )
        completed, failed, success_events, error_events = result
        if self._mla_root_fanout is not None:
            self._mla_root_fanout.finish_async_receives(completed, failed)
        return (
            [py_request_ids.get(rid, rid) for rid in completed],
            [py_request_ids.get(rid, rid) for rid in failed],
            [(rank, py_request_ids.get(rid, rid)) for rank, rid in success_events],
            [(rank, py_request_ids.get(rid, rid)) for rank, rid in error_events],
        )

    def _prepare_gen_session_errors(self, request_ids: list[int]) -> set[int]:
        if not request_ids:
            return set()
        agent = self._b10_transfer_agent
        if agent is None:
            return set()

        # Tombstone before sampling so no new destination copy can start.
        for rid in request_ids:
            agent.cancel_recv_request(rid)
            self._recv_sessions[rid].cancel()

        active_ids = [rid for rid in request_ids if self._has_active_recv_work(rid)]
        deferred = self._union(
            self._allgather_or_passthrough(active_ids, self._gen_allgather, self._gen_need_sync)
        )
        for rid in request_ids:
            if rid not in deferred:
                self._recv_sessions[rid].fail_cancelled_transfers(
                    RuntimeError(f"B10 receive request {rid} cancelled")
                )
        return set(request_ids) & deferred

    def _prepare_context_session_errors(self, request_ids: list[int]) -> set[int]:
        for rid in request_ids:
            self._send_sessions[rid].cancel()

        active_ids = [
            rid for rid in request_ids if self._send_sessions[rid].has_transferring_tasks()
        ]
        deferred = self._union(
            self._allgather_or_passthrough(
                active_ids, self._dist.tp_allgather, self._ctx_need_tp_sync
            )
        )
        if self._ctx_need_pp_sync:
            pp_allgather = getattr(self._dist, "pp_allgather")
            deferred = self._union(
                self._allgather_or_passthrough(
                    sorted(deferred), pp_allgather, self._ctx_need_pp_sync
                )
            )
        return set(request_ids) & deferred

    def _close_sessions_with_error(
        self, sessions: dict, reqs: dict, request_ids: list, cancel_first: bool = False
    ) -> list:
        if sessions is self._send_sessions:
            deferred = self._prepare_context_session_errors(request_ids)
        elif sessions is self._recv_sessions:
            deferred = self._prepare_gen_session_errors(request_ids)
        else:
            return super()._close_sessions_with_error(sessions, reqs, request_ids, cancel_first)

        ready = [rid for rid in request_ids if rid not in deferred]
        return super()._close_sessions_with_error(sessions, reqs, ready, cancel_first)

    def has_pending_gen_transfer(self, req) -> bool:
        rid = get_unique_rid(req)
        if rid is not None and self._has_active_recv_work(rid):
            return True
        return super().has_pending_gen_transfer(req)

    def cancel_request(self, req) -> bool:
        rid = get_unique_rid(req)
        if rid is not None and rid in self._recv_sessions:
            if self._b10_transfer_agent is not None:
                self._b10_transfer_agent.cancel_recv_request(rid)
            session = self._recv_sessions[rid]
            session.cancel()
            if self._has_active_recv_work(rid):
                return False
            session.fail_cancelled_transfers(RuntimeError(f"B10 receive request {rid} cancelled"))

        cancelled = super().cancel_request(req)
        if cancelled:
            if rid is not None and self._b10_transfer_agent is not None:
                self._b10_transfer_agent.discard_source_ready_event(rid)
        return cancelled
