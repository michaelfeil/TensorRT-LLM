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
"""Transfer tracing and timing logs for the B10 UCXX transfer agent.

`TransferTrace` is the tracing collaborator constructed by
`B10CacheTransferAgent.__init__`.

Constructor-injected:

- ``core`` (`_AgentCore`; the timing-log formatters append pool-state
  snapshots read from ``core.staging_buffer_pool`` /
  ``core.recv_scratch_buffer_pool``)
- ``trace_transfer_level`` (trace level string from B10AgentConfig)
- ``request_id_from_sync_message`` (agent-shell helper; injected reference)
"""

from __future__ import annotations

import time
from typing import Callable, Optional

from tensorrt_llm import logger
from tensorrt_llm._torch.disaggregation.b10.config import _TRACE_LEVEL_DEBUG, _TRACE_LEVEL_NONE
from tensorrt_llm._torch.disaggregation.b10.core import _AgentCore
from tensorrt_llm._torch.disaggregation.b10.pools import (
    _format_cuda_scratch_pool_state,
    _format_staging_pool_state,
)
from tensorrt_llm._torch.disaggregation.b10.state import _SendTransferPlan


class _TraceSpan:
    """Times a `with` block into a timings dict via TransferTrace._add_elapsed_ms."""

    __slots__ = ("_trace", "_timings", "_key", "_start_s")

    def __init__(self, trace: TransferTrace, timings: dict[str, float], key: str):
        self._trace = trace
        self._timings = timings
        self._key = key

    def __enter__(self):
        self._start_s = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        # Match the manual pattern: no timing is recorded when the
        # operation raises.
        if exc_type is None:
            self._trace._add_elapsed_ms(self._timings, self._key, self._start_s)
        return False


class TransferTrace:
    def __init__(
        self,
        core: _AgentCore,
        *,
        trace_transfer_level: str,
        request_id_from_sync_message: Callable[[Optional[str]], Optional[int]],
    ):
        self._core = core
        self._trace_transfer_level = trace_transfer_level
        self._request_id_from_sync_message = request_id_from_sync_message

    def _transfer_trace_level(self) -> str:
        return self._trace_transfer_level

    def _trace_timings_enabled(self) -> bool:
        return self._transfer_trace_level() != _TRACE_LEVEL_NONE

    def _trace_debug_enabled(self) -> bool:
        return self._transfer_trace_level() == _TRACE_LEVEL_DEBUG

    def _trace_transfer(self, message: Callable[[], str]) -> None:
        if self._trace_debug_enabled():
            logger.info(message())

    @staticmethod
    def _elapsed_ms(start_s: float) -> float:
        return (time.perf_counter() - start_s) * 1000.0

    def _add_elapsed_ms(self, timings: dict[str, float], key: str, start_s: float) -> None:
        if not self._trace_timings_enabled():
            return
        timings[key] = timings.get(key, 0.0) + self._elapsed_ms(start_s)

    def span(self, timings: dict[str, float], key: str) -> _TraceSpan:
        """Context manager recording the block's wall time under `key`."""
        return _TraceSpan(self, timings, key)

    def _log_recv_transfer_timings(
        self,
        transfer_id: int,
        status: str,
        timings: dict[str, float],
        total_start_s: float,
        *,
        request_id: Optional[int] = None,
        desc_count: int,
        total_bytes: int,
        wire_chunk_count: int,
        max_wire_chunk_size: int,
        copy_events: int,
        span_count: int,
        recv_in_flight: int,
    ) -> None:
        if not self._trace_timings_enabled():
            return
        logger.info(
            f"B10 recv transfer {transfer_id} timings: status={status} "
            f"request_id={request_id} "
            f"total_ms={self._elapsed_ms(total_start_s):.3f} "
            f"ready_send_ms={timings.get('ready_send_ms', 0.0):.3f} "
            f"data_phase_wall_ms={timings.get('data_phase_wall_ms', 0.0):.3f} "
            f"staging_acquire_ms={timings.get('staging_acquire_ms', 0.0):.3f} "
            f"recv_scratch_acquire_ms={timings.get('recv_scratch_acquire_ms', 0.0):.3f} "
            f"ucxx_recv_ms={timings.get('ucxx_recv_ms', 0.0):.3f} "
            f"dst_copy_ms={timings.get('dst_copy_ms', 0.0):.3f} "
            f"h2scratch_ms={timings.get('h2scratch_ms', 0.0):.3f} "
            f"request_scatter_ms={timings.get('request_scatter_ms', 0.0):.3f} "
            f"cuda_event_record_ms={timings.get('cuda_event_record_ms', 0.0):.3f} "
            f"staging_release_ms={timings.get('staging_release_ms', 0.0):.3f} "
            f"scratch_release_ms={timings.get('scratch_release_ms', 0.0):.3f} "
            f"copy_event_wait_ms={timings.get('copy_event_wait_ms', 0.0):.3f} "
            f"request_scatter_recovery_sync_ms="
            f"{timings.get('request_scatter_recovery_sync_ms', 0.0):.3f} "
            f"result_send_ms={timings.get('result_send_ms', 0.0):.3f} "
            f"descs={desc_count} data_chunks={wire_chunk_count} "
            f"spans={span_count} total_bytes={total_bytes} "
            f"max_data_chunk_size={max_wire_chunk_size} copy_events={copy_events} "
            f"request_scatter_chunks={int(timings.get('request_scatter_chunks', 0.0))} "
            f"request_scatter_fragments={int(timings.get('request_scatter_fragments', 0.0))} "
            f"request_scatter_kernels={int(timings.get('request_scatter_kernels', 0.0))} "
            f"request_scatter_aligned_kernels="
            f"{int(timings.get('request_scatter_aligned_kernels', 0.0))} "
            f"recv_in_flight={recv_in_flight} "
            f"{_format_staging_pool_state(self._core.staging_buffer_pool)} "
            f"{_format_cuda_scratch_pool_state(self._core.recv_scratch_buffer_pool)}"
        )

    def _log_send_transfer_timings(
        self,
        plan: _SendTransferPlan,
        status: str,
        timings: dict[str, float],
        total_start_s: float,
        *,
        span_count: int,
        send_in_flight: int,
    ) -> None:
        if not self._trace_timings_enabled():
            return
        logger.info(
            f"B10 send transfer {plan.transfer_id} timings: status={status} "
            f"request_id={self._request_id_from_sync_message(plan.sync_message)} "
            f"total_ms={self._elapsed_ms(total_start_s):.3f} "
            f"plan_build_ms={timings.get('plan_build_ms', 0.0):.3f} "
            f"admission_wait_ms={timings.get('admission_wait_ms', 0.0):.3f} "
            f"slot_lock_wait_ms={timings.get('slot_lock_wait_ms', 0.0):.3f} "
            f"lease_ms={timings.get('lease_ms', 0.0):.3f} "
            f"control_ready_ms={timings.get('control_ready_ms', 0.0):.3f} "
            f"data_phase_wall_ms={timings.get('data_phase_wall_ms', 0.0):.3f} "
            f"staging_acquire_ms={timings.get('staging_acquire_ms', 0.0):.3f} "
            f"src_copy_ms={timings.get('src_copy_ms', 0.0):.3f} "
            f"cuda_event_record_ms={timings.get('cuda_event_record_ms', 0.0):.3f} "
            f"copy_event_wait_ms={timings.get('copy_event_wait_ms', 0.0):.3f} "
            f"ucxx_send_ms={timings.get('ucxx_send_ms', 0.0):.3f} "
            f"staging_release_ms={timings.get('staging_release_ms', 0.0):.3f} "
            f"result_recv_ms={timings.get('result_recv_ms', 0.0):.3f} "
            f"descs={plan.desc_count} data_chunks={plan.wire_chunk_count} "
            f"spans={span_count} src_spans={plan.src_span_count} "
            f"dst_spans={plan.dst_span_count} "
            f"desc_order={plan.desc_order_strategy} "
            f"total_bytes={plan.total_bytes} "
            f"max_data_chunk_size={plan.max_wire_chunk_size} "
            f"send_in_flight={send_in_flight} "
            f"{_format_staging_pool_state(self._core.staging_buffer_pool)}"
        )
