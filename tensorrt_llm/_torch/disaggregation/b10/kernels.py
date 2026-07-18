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

import contextlib
import ctypes
from typing import Any, Callable, Optional

import numpy as np
import torch

from tensorrt_llm import logger
from tensorrt_llm._torch.disaggregation.b10.memory import (
    _BufferView,
    _ContiguousSpan,
    _DescArrayView,
    _DestinationScatterPlan,
    _NormalizedMemoryDesc,
    _SpanArrays,
)
from tensorrt_llm._torch.disaggregation.b10.planning import (
    _gather_plan_for_vram_spans,
    _scatter_programs_for_fragment_sizes,
)
from tensorrt_llm._utils import (
    TensorWrapper,
    convert_to_torch_tensor,
    maybe_pin_memory,
    prefer_pinned,
)

_SCATTER_KERNEL_BLOCK_SIZE = 16 * 1024
_SCATTER_ALIGNED_WORD_BYTES = 8
_SCATTER_ALIGNED_KERNEL_BLOCK_WORDS = _SCATTER_KERNEL_BLOCK_SIZE // _SCATTER_ALIGNED_WORD_BYTES
_REQUEST_SCATTER_ALIGNED_KERNEL_BLOCK_BYTES = 64 * 1024
_REQUEST_SCATTER_ALIGNED_KERNEL_BLOCK_WORDS = (
    _REQUEST_SCATTER_ALIGNED_KERNEL_BLOCK_BYTES // _SCATTER_ALIGNED_WORD_BYTES
)


def _span_view(descs: _DescArrayView, span: _ContiguousSpan, memory_type: str) -> _BufferView:
    # O(1) per span for _DescArrayView inputs (single __getitem__).
    first_desc = descs[span.start]
    return _make_buffer_view(
        _NormalizedMemoryDesc(ptr=first_desc.ptr, size=span.size, device_id=first_desc.device_id),
        memory_type,
    )


def _make_buffer_view(desc: _NormalizedMemoryDesc, memory_type: str) -> _BufferView:
    if desc.size == 0:
        return _BufferView(np.empty(0, dtype=np.uint8), None)
    if memory_type == "VRAM":
        torch.cuda.set_device(desc.device_id)
        owner = TensorWrapper(desc.ptr, torch.uint8, [desc.size])
        tensor = convert_to_torch_tensor(owner)
        return _BufferView(tensor, owner)
    if memory_type == "DRAM":
        array_type = ctypes.c_uint8 * desc.size
        owner = array_type.from_address(desc.ptr)
        return _BufferView(np.ctypeslib.as_array(owner), owner)
    raise NotImplementedError(f"B10 does not support memory type {memory_type}")


def _cuda_copy_device(src: Any, dst: Any) -> Optional[torch.device]:
    if isinstance(dst, torch.Tensor) and dst.is_cuda:
        return dst.device
    if isinstance(src, torch.Tensor) and src.is_cuda:
        return src.device
    return None


def _copy_buffer(src: Any, dst: Any, copy_stream: Optional[Any] = None) -> None:
    if isinstance(src, torch.Tensor) or isinstance(dst, torch.Tensor):
        src_tensor = src if isinstance(src, torch.Tensor) else torch.as_tensor(src)
        dst_tensor = dst if isinstance(dst, torch.Tensor) else torch.as_tensor(dst)
        h2d = dst_tensor.is_cuda and src_tensor.device.type == "cpu" and src_tensor.is_pinned()
        d2h = src_tensor.is_cuda and dst_tensor.device.type == "cpu" and dst_tensor.is_pinned()
        non_blocking = h2d or d2h
        device = _cuda_copy_device(src_tensor, dst_tensor)
        if copy_stream is not None and device is not None:
            with torch.cuda.stream(copy_stream):
                dst_tensor.copy_(src_tensor, non_blocking=non_blocking)
        else:
            dst_tensor.copy_(src_tensor, non_blocking=non_blocking)
        return

    np.copyto(np.asarray(dst), np.asarray(src), casting="no")


G_SCATTER_CUDA_BUFFER_TO_VRAM_SPANS_KERNEL: Optional[Any] = None


def _get_scatter_cuda_buffer_to_vram_spans_kernel() -> tuple[Any, Any, Any, Any, Any]:
    global G_SCATTER_CUDA_BUFFER_TO_VRAM_SPANS_KERNEL
    if G_SCATTER_CUDA_BUFFER_TO_VRAM_SPANS_KERNEL is None:
        import triton
        import triton.language as tl

        @triton.jit
        def scatter_kernel(
            src_buffer,
            dst_ptrs,
            src_offsets,
            sizes,
            program_span_indices,
            program_offsets,
            block_size: tl.constexpr,
        ):
            program_idx = tl.program_id(0)
            span_idx = tl.load(program_span_indices + program_idx)
            block_offset = tl.load(program_offsets + program_idx)
            offsets = block_offset + tl.arange(0, block_size)
            span_size = tl.load(sizes + span_idx)
            mask = offsets < span_size
            src_offset = tl.load(src_offsets + span_idx)
            dst_ptr = tl.load(dst_ptrs + span_idx).to(tl.pointer_type(tl.uint8))
            values = tl.load(src_buffer + src_offset + offsets, mask=mask)
            tl.store(dst_ptr + offsets, values, mask=mask)

        @triton.jit
        def scatter_aligned_u64_kernel(
            src_buffer,
            dst_ptrs,
            src_offsets,
            sizes,
            program_span_indices,
            program_offsets,
            block_words: tl.constexpr,
        ):
            program_idx = tl.program_id(0)
            span_idx = tl.load(program_span_indices + program_idx)
            block_word_offset = tl.load(program_offsets + program_idx)
            word_offsets = block_word_offset + tl.arange(0, block_words)
            span_words = tl.load(sizes + span_idx)
            mask = word_offsets < span_words
            src_byte_offset = tl.load(src_offsets + span_idx)
            dst_ptr = tl.load(dst_ptrs + span_idx).to(tl.pointer_type(tl.uint64))
            src_ptr = (src_buffer + src_byte_offset).to(tl.pointer_type(tl.uint64))
            values = tl.load(src_ptr + word_offsets, mask=mask)
            tl.store(dst_ptr + word_offsets, values, mask=mask)

        @triton.jit
        def scatter_absolute_aligned_u64_kernel(
            src_ptrs,
            dst_ptrs,
            sizes,
            program_fragment_indices,
            program_offsets,
            block_words: tl.constexpr,
        ):
            program_idx = tl.program_id(0)
            fragment_idx = tl.load(program_fragment_indices + program_idx)
            block_word_offset = tl.load(program_offsets + program_idx)
            word_offsets = block_word_offset + tl.arange(0, block_words)
            fragment_words = tl.load(sizes + fragment_idx)
            mask = word_offsets < fragment_words
            src_ptr = tl.load(src_ptrs + fragment_idx).to(tl.pointer_type(tl.uint64))
            dst_ptr = tl.load(dst_ptrs + fragment_idx).to(tl.pointer_type(tl.uint64))
            values = tl.load(src_ptr + word_offsets, mask=mask)
            tl.store(dst_ptr + word_offsets, values, mask=mask)

        @triton.jit
        def scatter_absolute_kernel(
            src_ptrs,
            dst_ptrs,
            sizes,
            program_fragment_indices,
            program_offsets,
            block_size: tl.constexpr,
        ):
            program_idx = tl.program_id(0)
            fragment_idx = tl.load(program_fragment_indices + program_idx)
            block_offset = tl.load(program_offsets + program_idx)
            offsets = block_offset + tl.arange(0, block_size)
            fragment_size = tl.load(sizes + fragment_idx)
            mask = offsets < fragment_size
            src_ptr = tl.load(src_ptrs + fragment_idx).to(tl.pointer_type(tl.uint8))
            dst_ptr = tl.load(dst_ptrs + fragment_idx).to(tl.pointer_type(tl.uint8))
            values = tl.load(src_ptr + offsets, mask=mask)
            tl.store(dst_ptr + offsets, values, mask=mask)

        G_SCATTER_CUDA_BUFFER_TO_VRAM_SPANS_KERNEL = (
            triton,
            scatter_kernel,
            scatter_aligned_u64_kernel,
            scatter_absolute_kernel,
            scatter_absolute_aligned_u64_kernel,
        )
    return G_SCATTER_CUDA_BUFFER_TO_VRAM_SPANS_KERNEL


_scatter_kernels_availability: Optional[bool] = None


def _scatter_kernels_available() -> bool:
    """Whether the Triton copy kernels can be built (memoized, warns once).

    Probed at gate time so a missing/broken Triton is a routing decision:
    eligible send-gather chunks fall back to the per-span copy loop instead
    of failing inside the transfer. Only the import/getter failure is
    absorbed here; launch errors still propagate.
    """
    global _scatter_kernels_availability
    if _scatter_kernels_availability is None:
        try:
            _get_scatter_cuda_buffer_to_vram_spans_kernel()
        except Exception as exc:
            _scatter_kernels_availability = False
            logger.warning(
                "B10 Triton copy kernels unavailable; fragmented sends fall "
                "back to the per-span copy loop: "
                f"error={type(exc).__name__}: {exc}"
            )
        else:
            _scatter_kernels_availability = True
    return _scatter_kernels_availability


def _use_aligned_scatter_kernel(descs: _DescArrayView, spans: _SpanArrays) -> bool:
    if not spans:
        return True
    word = _SCATTER_ALIGNED_WORD_BYTES
    return not bool(
        (descs.ptrs[spans.starts] % word).any()
        or (spans.chunk_offsets % word).any()
        or (spans.sizes % word).any()
    )


def _use_aligned_destination_scatter_plan(plan: _DestinationScatterPlan) -> bool:
    return not bool(
        (plan.src_ptrs % _SCATTER_ALIGNED_WORD_BYTES).any()
        or (plan.dst_ptrs % _SCATTER_ALIGNED_WORD_BYTES).any()
        or (plan.sizes % _SCATTER_ALIGNED_WORD_BYTES).any()
    )


_pin_scatter_metadata_warned = False


def _pin_scatter_metadata(values: np.ndarray) -> torch.Tensor:
    """Stage flat int64 scatter metadata in pinned host memory.

    Pinned staging makes the following H2D copy asynchronous; the caller
    must keep the returned tensor alive until that copy completes on its
    stream. Callers that keep it alive via lifetime_refs are covered by the
    refs' release path; callers passing lifetime_refs=None must synchronize
    the copy stream before dropping the pinned tensor (today only
    _warm_scatter_kernels does). torch's CachingHostAllocator records an
    event on non_blocking copies from pin_memory tensors, which backstops an
    early drop for pinned staging, but NOT for the pageable fallback below —
    that fallback is host-synchronous anyway. If pinning fails (e.g.
    pinned-memory pressure), fall back to the pageable host tensor at the
    cost of a synchronous H2D copy.
    """
    host = torch.from_numpy(values)
    try:
        return maybe_pin_memory(host)
    except RuntimeError as e:
        global _pin_scatter_metadata_warned
        if not _pin_scatter_metadata_warned:
            _pin_scatter_metadata_warned = True
            logger.warning(
                "B10 scatter metadata pin_memory failed; falling back to "
                f"pageable host staging (slower sync H2D copy): {e}"
            )
        return host


# Names match the _new_scatter_metadata keys. ORDER matters: it defines the
# flat staging layout and the positional kernel-argument unpack in
# _scatter_cuda_buffer_to_vram_spans. The request-order path
# (_scatter_cuda_buffers_to_vram_destination_order) has a different first
# section (src_ptrs, not dst_ptrs) and does not use this constant.
_SCATTER_METADATA_SECTION_NAMES = (
    "dst_ptrs",
    "src_offsets",
    "sizes",
    "program_span_indices",
    "program_offsets",
)


def _new_scatter_metadata(
    max_spans: int, max_programs: int, device: torch.device
) -> dict[str, torch.Tensor]:
    lengths = (max_spans,) * 3 + (max_programs,) * 2
    return {
        name: torch.empty((length,), dtype=torch.int64, device=device)
        for name, length in zip(_SCATTER_METADATA_SECTION_NAMES, lengths)
    }


def _record_tensors_on_stream(tensors: list[torch.Tensor], stream: Optional[Any]) -> None:
    if stream is None:
        return
    for tensor in tensors:
        tensor.record_stream(stream)


def _preallocated_scatter_metadata_view(
    metadata: dict[str, Any],
    name: str,
    length: int,
) -> torch.Tensor:
    # No device check: pool metadata tensors are allocated on the owning
    # scratch buffer's device at preallocation
    # (_CudaScratchBufferPool._preallocate_sync) and the pool is keyed by
    # device, while both callers gate span devices against the scratch
    # device (RecvPipeline._should_use_recv_scratch and
    # RecvPipeline._request_level_recv_scatter_chunk_indices), so a
    # mismatch is
    # structurally impossible here.
    tensor = metadata.get(name)
    if tensor is None:
        raise RuntimeError(f"B10 recv scratch metadata tensor {name} was not preallocated")
    if tensor.numel() < length:
        raise RuntimeError(
            f"B10 recv scratch metadata capacity exceeded for {name}: "
            f"needed={length} available={tensor.numel()}"
        )
    return tensor[:length]


def _upload_scatter_metadata_adhoc(
    host_metadata: np.ndarray,
    section_lengths: list[int],
    device: torch.device,
    copy_stream: Optional[Any],
    lifetime_refs: Optional[list[_BufferView]],
) -> tuple[torch.Tensor, ...]:
    """Pin and upload flat ad-hoc scatter metadata; return per-section views.

    Lifetime contract: the flat device tensor must outlive the kernel launch
    that consumes the returned views, and the pinned host tensor must
    outlive the in-flight H2D copy on copy_stream. The device tensor is
    recorded on copy_stream, and both tensors are appended to lifetime_refs
    when it is provided; a caller passing lifetime_refs=None must
    synchronize copy_stream before dropping the result (see
    _pin_scatter_metadata).
    """
    # One pinned host staging tensor for all metadata sections; the H2D
    # copy below is asynchronous from pinned memory instead of per-section
    # synchronous pageable copies built element-wise from Python ints.
    host_tensor = _pin_scatter_metadata(host_metadata)
    device_metadata = host_tensor.to(device, non_blocking=True)
    metadata_tensors = torch.split(device_metadata, section_lengths)
    _record_tensors_on_stream([device_metadata], copy_stream)
    if lifetime_refs is not None:
        lifetime_refs.append(_BufferView(device_metadata, device_metadata))
        lifetime_refs.append(_BufferView(host_tensor, host_tensor))
    return metadata_tensors


def _scatter_cuda_buffer_to_vram_spans(
    src_buffer: torch.Tensor,
    descs: _DescArrayView,
    spans: _SpanArrays,
    copy_stream: Optional[Any],
    lifetime_refs: Optional[list[_BufferView]] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    if not isinstance(src_buffer, torch.Tensor) or not src_buffer.is_cuda:
        raise TypeError("B10 recv scratch scatter source must be a CUDA tensor")
    if not spans:
        return
    span_devices = descs.device_ids[spans.starts]
    device_id = int(span_devices[0])
    if bool((span_devices != device_id).any()):
        raise ValueError(
            "B10 recv scratch scatter requires all destination spans on one CUDA device"
        )
    dst_ptr_values = descs.ptrs[spans.starts]
    use_aligned_kernel = _use_aligned_scatter_kernel(descs, spans)
    device = torch.device("cuda", device_id)
    span_count = len(spans)
    span_sizes = spans.sizes
    span_offsets = spans.chunk_offsets
    # Zero-size spans keep their metadata slot but emit no programs
    # (ceil(0 / block) == 0); program span indices index the unfiltered
    # span arrays.
    program_span_indices, program_offsets = _scatter_programs_for_fragment_sizes(
        span_sizes, _SCATTER_KERNEL_BLOCK_SIZE
    )
    program_count = int(program_span_indices.shape[0])
    if program_count == 0:
        return
    if use_aligned_kernel:
        program_offsets = program_offsets // _SCATTER_ALIGNED_WORD_BYTES
        size_values = span_sizes // _SCATTER_ALIGNED_WORD_BYTES
    else:
        size_values = span_sizes
    section_lengths = [
        span_count,
        span_count,
        span_count,
        program_count,
        program_count,
    ]
    flat_metadata = np.concatenate(
        [
            dst_ptr_values,
            span_offsets,
            size_values,
            program_span_indices,
            program_offsets,
        ]
    )
    stream_context = (
        torch.cuda.stream(copy_stream) if copy_stream is not None else contextlib.nullcontext()
    )
    with stream_context:
        if metadata is None:
            metadata_tensors = _upload_scatter_metadata_adhoc(
                flat_metadata, section_lengths, device, copy_stream, lifetime_refs
            )
        else:
            # One pinned host staging tensor for all five sections; each
            # section is copied asynchronously into its preallocated pool
            # tensor.
            host_metadata = _pin_scatter_metadata(flat_metadata)
            metadata_tensors = []
            for name, host_section, length in zip(
                _SCATTER_METADATA_SECTION_NAMES,
                torch.split(host_metadata, section_lengths),
                section_lengths,
            ):
                tensor_view = _preallocated_scatter_metadata_view(metadata, name, length)
                tensor_view.copy_(host_section, non_blocking=True)
                metadata_tensors.append(tensor_view)
            # No record_stream on pool metadata tensors: they live for the
            # process lifetime in _CudaScratchBufferPool._metadata and are
            # never freed, and record_stream only protects tensors that can
            # be deallocated while a stream still uses them.
            if lifetime_refs is not None:
                # Pinned staging must outlive the in-flight H2D copies on
                # copy_stream (see _pin_scatter_metadata).
                lifetime_refs.append(_BufferView(host_metadata, host_metadata))
        (dst_ptrs, src_offsets, sizes, program_span_indices_tensor, program_offsets_tensor) = (
            metadata_tensors
        )
        (
            _triton,
            scatter_kernel,
            scatter_aligned_u64_kernel,
            _request_kernel,
            _request_aligned_u64_kernel,
        ) = _get_scatter_cuda_buffer_to_vram_spans_kernel()
        grid = (program_count,)
        if use_aligned_kernel:
            scatter_aligned_u64_kernel[grid](
                src_buffer,
                dst_ptrs,
                src_offsets,
                sizes,
                program_span_indices_tensor,
                program_offsets_tensor,
                block_words=_SCATTER_ALIGNED_KERNEL_BLOCK_WORDS,
                num_warps=8,
            )
        else:
            scatter_kernel[grid](
                src_buffer,
                dst_ptrs,
                src_offsets,
                sizes,
                program_span_indices_tensor,
                program_offsets_tensor,
                block_size=_SCATTER_KERNEL_BLOCK_SIZE,
                num_warps=8,
            )


def _scatter_cuda_buffers_to_vram_destination_order(
    plan: _DestinationScatterPlan,
    copy_stream: Optional[Any],
    lifetime_refs: Optional[list[_BufferView]] = None,
    mark_launch_started: Optional[Callable[[], None]] = None,
) -> bool:
    """Launch one absolute-pointer copy kernel over the plan's fragments.

    Destinations are absolute pointers — VRAM for the recv scatter,
    UVA-mapped pinned host for the send gather.
    """
    fragment_count = plan.fragment_count
    if fragment_count == 0:
        return False
    # Every caller enforces single-device fragments before a plan is built:
    # the request-level recv gate in recv.py, CopyEngine's send-gather gate,
    # and the warmup constructions.
    device_id = int(plan.device_ids[0])
    device = torch.device("cuda", device_id)
    use_aligned_kernel = _use_aligned_destination_scatter_plan(plan)
    program_fragment_indices, program_offsets = _scatter_programs_for_fragment_sizes(
        plan.sizes,
        (
            _REQUEST_SCATTER_ALIGNED_KERNEL_BLOCK_BYTES
            if use_aligned_kernel
            else _SCATTER_KERNEL_BLOCK_SIZE
        ),
    )
    program_count = int(program_fragment_indices.shape[0])
    if use_aligned_kernel:
        program_offsets = program_offsets // _SCATTER_ALIGNED_WORD_BYTES
        size_values = plan.sizes // _SCATTER_ALIGNED_WORD_BYTES
    else:
        size_values = plan.sizes
    section_lengths = [
        fragment_count,
        fragment_count,
        fragment_count,
        program_count,
        program_count,
    ]
    flat_metadata = np.concatenate(
        [
            plan.src_ptrs,
            plan.dst_ptrs,
            size_values,
            program_fragment_indices,
            program_offsets,
        ]
    )
    stream_context = (
        torch.cuda.stream(copy_stream) if copy_stream is not None else contextlib.nullcontext()
    )
    with stream_context:
        (src_ptrs, dst_ptrs, sizes, program_fragment_indices_tensor, program_offsets_tensor) = (
            _upload_scatter_metadata_adhoc(
                flat_metadata, section_lengths, device, copy_stream, lifetime_refs
            )
        )
        (
            _triton,
            _scatter_kernel,
            _scatter_aligned_u64_kernel,
            request_kernel,
            request_aligned_u64_kernel,
        ) = _get_scatter_cuda_buffer_to_vram_spans_kernel()
        grid = (program_count,)
        if use_aligned_kernel:
            launcher = request_aligned_u64_kernel[grid]
            if mark_launch_started is not None:
                mark_launch_started()
            launcher(
                src_ptrs,
                dst_ptrs,
                sizes,
                program_fragment_indices_tensor,
                program_offsets_tensor,
                block_words=_REQUEST_SCATTER_ALIGNED_KERNEL_BLOCK_WORDS,
                num_warps=8,
            )
        else:
            launcher = request_kernel[grid]
            if mark_launch_started is not None:
                mark_launch_started()
            launcher(
                src_ptrs,
                dst_ptrs,
                sizes,
                program_fragment_indices_tensor,
                program_offsets_tensor,
                block_size=_SCATTER_KERNEL_BLOCK_SIZE,
                num_warps=8,
            )
    return use_aligned_kernel


_send_gather_byte_kernel_warned = False


def _warn_send_gather_byte_kernel_once(plan: _DestinationScatterPlan) -> None:
    """Perf tripwire: the byte kernel's UVA stores (~21GB/s) are slower than
    the per-span copy loop, and production KV layouts are 8-byte aligned
    today, so a byte-kernel send gather means the layout changed. Routing is
    unchanged; this only warns once per process.
    """
    global _send_gather_byte_kernel_warned
    if _send_gather_byte_kernel_warned:
        return
    _send_gather_byte_kernel_warned = True
    word = _SCATTER_ALIGNED_WORD_BYTES
    misaligned = (
        (plan.src_ptrs % word != 0) | (plan.dst_ptrs % word != 0) | (plan.sizes % word != 0)
    )
    first = int(np.flatnonzero(misaligned)[0])
    logger.warning(
        "B10 send gather selected the unaligned byte kernel; sends will be "
        f"slower until the KV layout is {word}-byte aligned: "
        f"misaligned_fragments={int(misaligned.sum())}/{plan.fragment_count} "
        f"first_src_ptr={int(plan.src_ptrs[first])} "
        f"first_dst_ptr={int(plan.dst_ptrs[first])} "
        f"first_size={int(plan.sizes[first])}"
    )


def _gather_vram_spans_to_pinned_staging(
    descs: _DescArrayView,
    spans: _SpanArrays,
    staging_buffer: torch.Tensor,
    copy_stream: Optional[Any],
    lifetime_refs: Optional[list[_BufferView]] = None,
    *,
    warn_byte_kernel: bool = True,
) -> bool:
    """Gather a chunk's contiguous VRAM spans into pinned host staging with
    one Triton launch (the send-side hot path).

    Pinned host memory is UVA-mapped, so the absolute-pointer scatter
    kernels (which are direction-agnostic src_ptr -> dst_ptr copy kernels)
    store directly into the staging buffer through its host address: one
    launch replaces the per-span D2H copy_ dispatches AND the D2H copies
    themselves — the same bytes cross the bus from inside the kernel.
    Sources are read-only, so no overlap check is needed. Metadata lifetime
    follows the _upload_scatter_metadata_adhoc contract: pass lifetime_refs
    and keep them until the copy_stream event recorded after this call
    completes.

    Returns True when a kernel was launched, False when the spans move no
    bytes.
    """
    if (
        not isinstance(staging_buffer, torch.Tensor)
        or staging_buffer.device.type != "cpu"
        or not staging_buffer.is_pinned()
    ):
        raise TypeError("B10 send gather staging destination must be a pinned host torch tensor")
    plan = _gather_plan_for_vram_spans(descs, spans, staging_base_ptr=staging_buffer.data_ptr())
    if plan.fragment_count == 0:
        return False
    # CopyEngine's send-gather gate and the warmup construction enforce
    # single-device sources, matching the request-level scatter contract in
    # _scatter_cuda_buffers_to_vram_destination_order.
    device_id = int(plan.device_ids[0])
    # The per-span loop pinned the thread's CUDA device via _make_buffer_view
    # (torch.cuda.set_device per span view); the kernel path must do the same
    # once so the metadata upload and launch land on the spans' device.
    torch.cuda.set_device(device_id)
    use_aligned_kernel = _scatter_cuda_buffers_to_vram_destination_order(
        plan, copy_stream, lifetime_refs
    )
    if warn_byte_kernel and not use_aligned_kernel:
        _warn_send_gather_byte_kernel_once(plan)
    return True


def _warm_scatter_kernels(device: torch.device, copy_stream: Any) -> None:
    """Compile every Triton scatter kernel variant at startup.

    Triton JIT-compiles a kernel on its first launch, which costs hundreds
    of milliseconds per process and otherwise lands in the data phase of the
    first fragmented recv this rank serves. Launch each variant once through
    the production entry points so the compiled specializations match.
    """
    src = torch.zeros(64, dtype=torch.uint8, device=device)
    dst = torch.zeros(64, dtype=torch.uint8, device=device)
    device_id = device.index
    # Pool metadata tensors have a different Triton alignment specialization
    # than split views of one flat upload; warm both argument layouts.
    warm_metadata = _new_scatter_metadata(max_spans=4, max_programs=4, device=device)
    # Aligned sizes take the u64 kernels; a 1-byte span takes the byte kernels.
    for size in (_SCATTER_ALIGNED_WORD_BYTES, 1):
        descs = _DescArrayView(
            np.array([dst.data_ptr()], dtype=np.int64),
            np.array([size], dtype=np.int64),
            np.array([device_id], dtype=np.int64),
        )
        spans = _SpanArrays(
            np.zeros(1, dtype=np.int64),
            np.array([size], dtype=np.int64),
            np.zeros(1, dtype=np.int64),
        )
        _scatter_cuda_buffer_to_vram_spans(src, descs, spans, copy_stream)
        _scatter_cuda_buffer_to_vram_spans(src, descs, spans, copy_stream, metadata=warm_metadata)
    # No direct absolute-kernel launches here: the parity warmup's
    # single-span unit cases compile byte-identical specializations of both
    # absolute kernels.
    _warm_absolute_kernel_parity_specializations(device, copy_stream)
    if copy_stream is not None:
        copy_stream.synchronize()
    else:
        torch.cuda.synchronize(device)


def _warm_absolute_kernel_parity_specializations(device: torch.device, copy_stream: Any) -> None:
    """Compile every absolute-kernel parity specialization shared by the
    send gather and the request-level recv scatter (before this warmup the
    request-level recv path could JIT mid-transfer).

    The metadata sections are torch.split views of one flat int64 upload,
    and Triton's specialization key includes each pointer argument's
    16-byte divisibility. Section byte offsets are multiples of
    8 * span_count (and of 8 * program_count for the program sections), so
    the alignment pattern — hence the compiled specialization — depends on
    the PARITY of the span count and of the program count. Warm all four
    parity combinations for both the aligned-u64 and byte kernels through
    the production entry point so arbitrary production span/program counts
    never JIT in the data phase of a transfer.
    """
    word = _SCATTER_ALIGNED_WORD_BYTES
    device_id = device.index
    span_size_cases: list[list[int]] = []
    for block, unit in (
        (_REQUEST_SCATTER_ALIGNED_KERNEL_BLOCK_BYTES, word),
        (_SCATTER_KERNEL_BLOCK_SIZE, 1),
    ):
        span_size_cases.extend(
            [
                [unit],  # spans odd, programs odd
                [unit, unit],  # spans even, programs even
                [block + unit],  # spans odd, programs even
                [block + unit, unit],  # spans even, programs odd
            ]
        )
    max_chunk_size = max(sum(sizes) for sizes in span_size_cases)
    gather_src = torch.zeros(max_chunk_size, dtype=torch.uint8, device=device)
    gather_staging = torch.empty(
        (max_chunk_size,), dtype=torch.uint8, device="cpu", pin_memory=prefer_pinned()
    )
    for span_sizes in span_size_cases:
        sizes = np.asarray(span_sizes, dtype=np.int64)
        chunk_offsets = np.concatenate(([0], np.cumsum(sizes[:-1])))
        descs = _DescArrayView(
            gather_src.data_ptr() + chunk_offsets,
            sizes,
            np.full(sizes.shape[0], device_id, dtype=np.int64),
        )
        spans = _SpanArrays(np.arange(sizes.shape[0], dtype=np.int64), sizes, chunk_offsets)
        # warn_byte_kernel=False: the byte-kernel cases here are deliberate
        # warmups, not production layout regressions.
        _gather_vram_spans_to_pinned_staging(
            descs, spans, gather_staging, copy_stream, warn_byte_kernel=False
        )
