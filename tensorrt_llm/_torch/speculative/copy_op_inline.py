# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from functools import lru_cache

import torch
from torch.utils.cpp_extension import load_inline


_EXTENSION_NAME = "trtllm_inplace_slice_copy_inline"
_EXTRA_CFLAGS = ["-O3", "-DNDEBUG"]

_CPP_SOURCE = r"""
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>

// Copy src[:, :] into dest[:src.size(0), dim1_start:dim1_end] using
// cudaMemcpy2DAsync. This intentionally avoids TensorRT-LLM headers so it can
// be built as a small inline PyTorch extension in the dev container.
void inplace_slice_copy(at::Tensor dest,
                        at::Tensor const& src,
                        int64_t dim1_start,
                        int64_t dim1_end) {
    TORCH_CHECK(dest.is_cuda(), "dest must be a CUDA tensor");
    TORCH_CHECK(src.is_cuda(), "src must be a CUDA tensor");
    TORCH_CHECK(dest.get_device() == src.get_device(),
                "dest and src must be on the same CUDA device");
    TORCH_CHECK(dest.is_contiguous(), "dest must be contiguous");
    TORCH_CHECK(src.is_contiguous(), "src must be contiguous");
    TORCH_CHECK(dest.dim() == 2, "dest must be 2-D");
    TORCH_CHECK(src.dim() == 2, "src must be 2-D");
    TORCH_CHECK(dest.scalar_type() == src.scalar_type(),
                "dest and src must have the same dtype");

    int64_t const num_tokens = src.size(0);
    int64_t const slice_width = dim1_end - dim1_start;
    TORCH_CHECK(dim1_start >= 0, "dim1_start must be non-negative");
    TORCH_CHECK(slice_width > 0, "dim1_end must be greater than dim1_start");
    TORCH_CHECK(num_tokens <= dest.size(0), "num_tokens exceeds dest row count");
    TORCH_CHECK(dim1_end <= dest.size(1), "dim1_end exceeds dest column count");
    TORCH_CHECK(src.size(1) == slice_width,
                "src column count must equal dim1_end - dim1_start");

    if (num_tokens == 0) {
        return;
    }

    c10::cuda::CUDAGuard device_guard(dest.device());

    int64_t const elem_size = dest.element_size();
    int64_t const dest_pitch = dest.size(1) * elem_size;
    int64_t const src_pitch = src.size(1) * elem_size;
    int64_t const width = slice_width * elem_size;

    char* dest_ptr = static_cast<char*>(dest.data_ptr()) + dim1_start * elem_size;
    char const* src_ptr = static_cast<char const*>(src.data_ptr());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(dest.get_device());

    C10_CUDA_CHECK(cudaMemcpy2DAsync(dest_ptr,
                                     dest_pitch,
                                     src_ptr,
                                     src_pitch,
                                     width,
                                     static_cast<size_t>(num_tokens),
                                     cudaMemcpyDeviceToDevice,
                                     stream));
}
"""


@lru_cache(maxsize=1)
def _get_inplace_slice_copy_op():
    verbose = os.environ.get("TRTLLM_INLINE_COPY_VERBOSE", "0") == "1"
    extension = load_inline(
        name=_EXTENSION_NAME,
        cpp_sources=[_CPP_SOURCE],
        functions=["inplace_slice_copy"],
        extra_cflags=_EXTRA_CFLAGS,
        with_cuda=True,
        verbose=verbose,
    )
    return extension.inplace_slice_copy


def preload_inplace_slice_copy() -> None:
    _get_inplace_slice_copy_op()


def inplace_slice_copy(dest: torch.Tensor, src: torch.Tensor,
                       dim1_start: int, dim1_end: int) -> None:
    _get_inplace_slice_copy_op()(dest, src, int(dim1_start), int(dim1_end))
