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

`TransferTracer` is the tracing collaborator constructed by
`B10CacheTransferAgent.__init__`. Each active transfer owns a short-lived
`TransferTrace`, which keeps timing state out of the send and receive
pipelines.

Constructor-injected:

- ``core`` (`_AgentCore`; the timing-log formatters append pool-state
  snapshots read from ``core.staging_buffer_pool`` /
  ``core.recv_scratch_buffer_pool``)
- ``trace_transfer_level`` (trace level string from B10AgentConfig)
"""

from __future__ import annotations

import time
from types import TracebackType
from typing import Callable, Optional

from tensorrt_llm import logger
from tensorrt_llm._torch.disaggregation.b10.config import _TRACE_LEVEL_DEBUG, _TRACE_LEVEL_NONE
from tensorrt_llm._torch.disaggregation.b10.core import _AgentCore
from tensorrt_llm._torch.disaggregation.b10.pools import (
    _format_cuda_scratch_pool_state,
    _format_staging_pool_state,
)
from tensorrt_llm._torch.disaggregation.b10.protocol import _request_id_from_sync_message
from tensorrt_llm._torch.disaggregation.b10.state import _SendTransferPlan


class _TraceTimer:
    """Measure either a `with` block or an explicitly stopped interval."""

    __slots__ = ("_trace", "_names", "_record_on_error", "_start_s")

    def __init__(
        self,
        trace: TransferTrace,
        names: tuple[str, ...],
        *,
        record_on_error: bool = False,
    ) -> None:
        self._trace = trace
        self._names = names
        self._record_on_error = record_on_error
        self._start_s = time.perf_counter() if trace.enabled else None

    def __enter__(self) -> _TraceTimer:
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        _exc: Optional[BaseException],
        _traceback: Optional[TracebackType],
    ) -> bool:
        if exc_type is None or self._record_on_error:
            self.stop()
        else:
            self._start_s = None
        return False

    def stop(self) -> None:
        if self._start_s is None:
            return
        self._trace._add_elapsed(self._names, self._start_s)
        self._start_s = None


class TransferTrace:
    """Timing state owned by one send or receive transfer."""

    __slots__ = ("_tracer", "_side", "transfer_id", "_started_s", "_values")

    def __init__(self, tracer: TransferTracer, side: str, transfer_id: int) -> None:
        self._tracer = tracer
        self._side = side
        self.transfer_id = transfer_id
        self._started_s = time.perf_counter() if tracer.timings_enabled else None
        self._values: dict[str, float] = {}

    @property
    def enabled(self) -> bool:
        return self._started_s is not None

    def measure(self, *names: str) -> _TraceTimer:
        """Measure a successfully completed block."""
        return _TraceTimer(self, names)

    def measure_attempt(self, *names: str) -> _TraceTimer:
        """Measure a block whether it completes or raises."""
        return _TraceTimer(self, names, record_on_error=True)

    def timer(self, *names: str) -> _TraceTimer:
        return _TraceTimer(self, names)

    def increment(self, name: str, value: int = 1) -> None:
        if self.enabled:
            self._values[name] = self._values.get(name, 0.0) + value

    def set(self, name: str, value: int) -> None:
        if self.enabled:
            self._values[name] = float(value)

    def debug(self, message: Callable[[], str]) -> None:
        if self._tracer.debug_enabled:
            logger.info(f"B10 {self._side} transfer {self.transfer_id} {message()}")

    def _add_elapsed(self, names: tuple[str, ...], start_s: float) -> None:
        elapsed_ms = (time.perf_counter() - start_s) * 1000.0
        for name in names:
            self._values[name] = self._values.get(name, 0.0) + elapsed_ms

    def _value(self, name: str) -> float:
        return self._values.get(name, 0.0)

    def _total_ms(self) -> float:
        assert self._started_s is not None
        return (time.perf_counter() - self._started_s) * 1000.0


class TransferTracer:
    def __init__(
        self,
        core: _AgentCore,
        *,
        trace_transfer_level: str,
    ) -> None:
        self._core = core
        self._trace_transfer_level = trace_transfer_level

    @property
    def level(self) -> str:
        return self._trace_transfer_level

    @property
    def timings_enabled(self) -> bool:
        return self.level != _TRACE_LEVEL_NONE

    @property
    def debug_enabled(self) -> bool:
        return self.level == _TRACE_LEVEL_DEBUG

    def debug(self, message: Callable[[], str]) -> None:
        if self.debug_enabled:
            logger.info(message())

    def start(self, side: str, transfer_id: int) -> TransferTrace:
        return TransferTrace(self, side, transfer_id)

    def log_recv(
        self,
        trace: TransferTrace,
        status: str,
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
        if not trace.enabled:
            return
        logger.info(
            f"B10 recv transfer {trace.transfer_id} timings: status={status} "
            f"request_id={request_id} "
            f"total_ms={trace._total_ms():.3f} "
            f"ready_send_ms={trace._value('ready_send'):.3f} "
            f"data_phase_wall_ms={trace._value('data_phase_wall'):.3f} "
            f"staging_acquire_ms={trace._value('staging_acquire'):.3f} "
            f"recv_scratch_acquire_ms={trace._value('recv_scratch_acquire'):.3f} "
            f"ucxx_recv_ms={trace._value('ucxx_recv'):.3f} "
            f"dst_copy_ms={trace._value('dst_copy'):.3f} "
            f"h2scratch_ms={trace._value('h2scratch'):.3f} "
            f"request_scatter_ms={trace._value('request_scatter'):.3f} "
            f"cuda_event_record_ms={trace._value('cuda_event_record'):.3f} "
            f"staging_release_ms={trace._value('staging_release'):.3f} "
            f"scratch_release_ms={trace._value('scratch_release'):.3f} "
            f"copy_event_wait_ms={trace._value('copy_event_wait'):.3f} "
            f"request_scatter_recovery_sync_ms="
            f"{trace._value('request_scatter_recovery_sync'):.3f} "
            f"result_send_ms={trace._value('result_send'):.3f} "
            f"descs={desc_count} data_chunks={wire_chunk_count} "
            f"spans={span_count} total_bytes={total_bytes} "
            f"max_data_chunk_size={max_wire_chunk_size} copy_events={copy_events} "
            f"request_scatter_chunks={int(trace._value('request_scatter_chunks'))} "
            f"request_scatter_fragments={int(trace._value('request_scatter_fragments'))} "
            f"request_scatter_kernels={int(trace._value('request_scatter_kernels'))} "
            f"request_scatter_aligned_kernels="
            f"{int(trace._value('request_scatter_aligned_kernels'))} "
            f"recv_in_flight={recv_in_flight} "
            f"{_format_staging_pool_state(self._core.staging_buffer_pool)} "
            f"{_format_cuda_scratch_pool_state(self._core.recv_scratch_buffer_pool)}"
        )

    def log_send(
        self,
        trace: TransferTrace,
        plan: _SendTransferPlan,
        status: str,
        *,
        span_count: int,
        send_in_flight: int,
    ) -> None:
        if not trace.enabled:
            return
        logger.info(
            f"B10 send transfer {plan.transfer_id} timings: status={status} "
            f"request_id={_request_id_from_sync_message(plan.sync_message)} "
            f"total_ms={trace._total_ms():.3f} "
            f"plan_build_ms={trace._value('plan_build'):.3f} "
            f"admission_wait_ms={trace._value('admission_wait'):.3f} "
            f"slot_lock_wait_ms={trace._value('slot_lock_wait'):.3f} "
            f"lease_ms={trace._value('lease'):.3f} "
            f"control_ready_ms={trace._value('control_ready'):.3f} "
            f"data_phase_wall_ms={trace._value('data_phase_wall'):.3f} "
            f"staging_acquire_ms={trace._value('staging_acquire'):.3f} "
            f"src_copy_ms={trace._value('src_copy'):.3f} "
            f"cuda_event_record_ms={trace._value('cuda_event_record'):.3f} "
            f"copy_event_wait_ms={trace._value('copy_event_wait'):.3f} "
            f"ucxx_send_ms={trace._value('ucxx_send'):.3f} "
            f"staging_release_ms={trace._value('staging_release'):.3f} "
            f"result_recv_ms={trace._value('result_recv'):.3f} "
            f"descs={plan.desc_count} data_chunks={plan.wire_chunk_count} "
            f"spans={span_count} src_spans={plan.src_span_count} "
            f"dst_spans={plan.dst_span_count} "
            f"desc_order={plan.desc_order_strategy} "
            f"total_bytes={plan.total_bytes} "
            f"max_data_chunk_size={plan.max_wire_chunk_size} "
            f"send_in_flight={send_in_flight} "
            f"{_format_staging_pool_state(self._core.staging_buffer_pool)}"
        )
