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
"""Shared staging-to-descriptor copy operations for B10 send and receive."""

from __future__ import annotations

from typing import Any, Optional

import torch

try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart

from tensorrt_llm._torch.disaggregation.b10 import memory as b10_memory
from tensorrt_llm._torch.disaggregation.b10.core import _AgentCore
from tensorrt_llm._torch.disaggregation.b10.kernels import (
    _copy_buffer,
    _cuda_copy_device,
    _scatter_cuda_buffer_to_vram_spans,
    _span_view,
)
from tensorrt_llm._torch.disaggregation.b10.memory import (
    _BufferView,
    _DescArrayView,
    _TransferChunk,
)
from tensorrt_llm._torch.disaggregation.b10.planning import _contiguous_desc_spans
from tensorrt_llm._torch.disaggregation.b10.pools import _device_key
from tensorrt_llm._torch.disaggregation.b10.state import _SourceReadyEvents


def _copy_vram_spans_to_pinned_staging_batch(
    descs: _DescArrayView,
    spans: b10_memory._SpanArrays,
    staging_buffer: torch.Tensor,
    copy_stream: Any,
) -> bool:
    """Submit multiple disjoint D2H spans as one CUDA batch."""
    active = spans.sizes > 0
    count = int(active.sum())
    batch_copy = getattr(cudart, "cudaMemcpyBatchAsync", None)
    if count < 2 or batch_copy is None:
        return False

    src_ptrs = descs.ptrs[spans.starts[active]].tolist()
    dst_ptrs = (staging_buffer.data_ptr() + spans.chunk_offsets[active]).tolist()
    sizes = spans.sizes[active].tolist()
    attr = cudart.cudaMemcpyAttributes()
    attr.srcAccessOrder = cudart.cudaMemcpySrcAccessOrder.cudaMemcpySrcAccessOrderStream
    attr.flags = cudart.cudaMemcpyFlags.cudaMemcpyFlagPreferOverlapWithCompute
    # Raw runtime calls use the calling thread's current CUDA device.
    torch.cuda.set_device(copy_stream.device)
    (error,) = batch_copy(
        dst_ptrs,
        src_ptrs,
        sizes,
        count,
        [attr],
        [0],
        1,
        copy_stream.cuda_stream,
    )
    if error != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"B10 cudaMemcpyBatchAsync failed for {count} D2H spans: {error!r}")
    return True


class _CopyEngine:
    def __init__(self, core: _AgentCore):
        self._core = core

    @staticmethod
    def single_cuda_device(
        descs: _DescArrayView,
        spans: b10_memory._SpanArrays,
    ) -> Optional[torch.device]:
        if not spans:
            return None
        devices = descs.device_ids[spans.starts]
        if not bool((devices == devices[0]).all()):
            return None
        return torch.device("cuda", int(devices[0]))

    def _copy_stream_for_device(
        self,
        copy_device: torch.device,
        prepared_copy_streams: dict[tuple[str, int], Any],
        *,
        wait_current_stream: bool,
        source_ready_events: Optional[_SourceReadyEvents] = None,
        source_ready_waited_keys: Optional[set[tuple[str, int]]] = None,
    ) -> Any:
        key = _device_key(copy_device)
        copy_stream = prepared_copy_streams.get(key)
        if copy_stream is not None:
            return copy_stream

        copy_stream = self._core.cuda_copy_streams.stream_for(copy_device)
        waited_on_source = source_ready_waited_keys is not None and key in source_ready_waited_keys
        if source_ready_events and not waited_on_source:
            for event_device, event in source_ready_events:
                if _device_key(event_device) == key:
                    copy_stream.wait_event(event)
                    waited_on_source = True
            if waited_on_source and source_ready_waited_keys is not None:
                source_ready_waited_keys.add(key)
        if source_ready_events is not None and not waited_on_source:
            raise RuntimeError(f"B10 missing source-ready event for copy device {copy_device}")
        if not waited_on_source and wait_current_stream:
            current_stream = torch.cuda.current_stream(copy_device)
            if current_stream is not copy_stream:
                copy_stream.wait_stream(current_stream)
        prepared_copy_streams[key] = copy_stream
        return copy_stream

    def _scatter_staging_to_vram_spans(
        self,
        staging_buffer: Any,
        descs: _DescArrayView,
        chunk: _TransferChunk,
        spans: b10_memory._SpanArrays,
        scratch_view: _BufferView,
        copy_stream: Any,
        lifetime_refs: Optional[list[_BufferView]],
    ) -> list[torch.device]:
        scratch_buffer = scratch_view.buffer[: chunk.size]
        _copy_buffer(staging_buffer[: chunk.size], scratch_buffer, copy_stream=copy_stream)
        _scatter_cuda_buffer_to_vram_spans(
            scratch_buffer, descs, spans, copy_stream, lifetime_refs, scratch_view.metadata
        )
        return [scratch_buffer.device]

    def copy_chunk(
        self,
        descs: _DescArrayView,
        chunk: _TransferChunk,
        memory_type: str,
        staging_buffer: Any,
        *,
        copy_from_staging: bool,
        lifetime_refs: Optional[list[_BufferView]] = None,
        spans: Optional[b10_memory._SpanArrays] = None,
        scratch_view: Optional[_BufferView] = None,
        source_ready_events: Optional[_SourceReadyEvents] = None,
        source_ready_waited_keys: Optional[set[tuple[str, int]]] = None,
    ) -> tuple[int, list[torch.device]]:
        copy_devices: list[torch.device] = []
        prepared_copy_streams: dict[tuple[str, int], Any] = {}
        spans = _contiguous_desc_spans(descs, chunk) if spans is None else spans
        if scratch_view is not None:
            if not copy_from_staging:
                raise RuntimeError("B10 recv scratch copy only supports staging-to-VRAM copies")
            if memory_type != "VRAM":
                raise RuntimeError("B10 recv scratch copy only supports VRAM destinations")
            copy_device = _cuda_copy_device(staging_buffer, scratch_view.buffer)
            if copy_device is None:
                raise RuntimeError("B10 recv scratch copy requires a CUDA scratch buffer")
            copy_stream = self._copy_stream_for_device(
                copy_device, prepared_copy_streams, wait_current_stream=False
            )
            copy_devices.extend(
                self._scatter_staging_to_vram_spans(
                    staging_buffer, descs, chunk, spans, scratch_view, copy_stream, lifetime_refs
                )
            )
            return len(spans), copy_devices

        staging_tensor = staging_buffer if isinstance(staging_buffer, torch.Tensor) else None
        if (
            memory_type == "VRAM"
            and staging_tensor is not None
            and staging_tensor.device.type == "cpu"
        ):
            return self._copy_host_staging_spans(
                descs,
                spans,
                staging_tensor,
                memory_type,
                copy_from_staging=copy_from_staging,
                lifetime_refs=lifetime_refs,
                source_ready_events=source_ready_events,
                source_ready_waited_keys=source_ready_waited_keys,
            )

        for span in spans:
            desc_view = _span_view(descs, span, memory_type)
            if lifetime_refs is not None:
                lifetime_refs.append(desc_view)
            staging_slice = staging_buffer[span.chunk_offset : span.chunk_offset + span.size]
            if copy_from_staging:
                src_buffer, dst_buffer = staging_slice, desc_view.buffer
            else:
                src_buffer, dst_buffer = desc_view.buffer, staging_slice
            copy_device = _cuda_copy_device(src_buffer, dst_buffer)
            copy_stream = None
            if copy_device is not None:
                copy_devices.append(copy_device)
                copy_stream = self._copy_stream_for_device(
                    copy_device,
                    prepared_copy_streams,
                    wait_current_stream=not copy_from_staging,
                    source_ready_events=source_ready_events,
                    source_ready_waited_keys=source_ready_waited_keys,
                )
            if copy_stream is None:
                _copy_buffer(src_buffer, dst_buffer)
            else:
                _copy_buffer(src_buffer, dst_buffer, copy_stream=copy_stream)
        return len(spans), copy_devices

    def _copy_host_staging_spans(
        self,
        descs: _DescArrayView,
        spans: b10_memory._SpanArrays,
        staging_buffer: torch.Tensor,
        memory_type: str,
        *,
        copy_from_staging: bool,
        lifetime_refs: Optional[list[_BufferView]],
        source_ready_events: Optional[_SourceReadyEvents],
        source_ready_waited_keys: Optional[set[tuple[str, int]]],
    ) -> tuple[int, list[torch.device]]:
        copy_devices: list[torch.device] = []
        prepared_copy_streams: dict[tuple[str, int], Any] = {}
        non_blocking = staging_buffer.is_pinned()
        if not copy_from_staging and non_blocking:
            copy_device = self.single_cuda_device(descs, spans)
            if copy_device is not None:
                copy_stream = self._copy_stream_for_device(
                    copy_device,
                    prepared_copy_streams,
                    wait_current_stream=True,
                    source_ready_events=source_ready_events,
                    source_ready_waited_keys=source_ready_waited_keys,
                )
                if _copy_vram_spans_to_pinned_staging_batch(
                    descs, spans, staging_buffer, copy_stream
                ):
                    return len(spans), [copy_device]

        active_stream = None
        stream_context = None
        try:
            for span in spans:
                if span.size == 0:
                    continue
                desc_view = _span_view(descs, span, memory_type)
                if lifetime_refs is not None:
                    lifetime_refs.append(desc_view)
                desc_buffer = desc_view.buffer
                staging_slice = staging_buffer[span.chunk_offset : span.chunk_offset + span.size]
                copy_device = desc_buffer.device
                copy_devices.append(copy_device)
                copy_stream = self._copy_stream_for_device(
                    copy_device,
                    prepared_copy_streams,
                    wait_current_stream=not copy_from_staging,
                    source_ready_events=source_ready_events,
                    source_ready_waited_keys=source_ready_waited_keys,
                )
                if copy_stream is not active_stream:
                    if stream_context is not None:
                        stream_context.__exit__(None, None, None)
                    stream_context = torch.cuda.stream(copy_stream)
                    stream_context.__enter__()
                    active_stream = copy_stream
                if copy_from_staging:
                    desc_buffer.copy_(staging_slice, non_blocking=non_blocking)
                else:
                    staging_slice.copy_(desc_buffer, non_blocking=non_blocking)
        finally:
            if stream_context is not None:
                stream_context.__exit__(None, None, None)
        return len(spans), copy_devices
