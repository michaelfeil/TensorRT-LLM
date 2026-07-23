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

import os
from dataclasses import dataclass
from typing import Optional

from tensorrt_llm._torch.disaggregation.b10 import net as b10_net
from tensorrt_llm._torch.disaggregation.b10.pools import (
    _DEFAULT_STAGING_POOL_BUFFER_SIZE,
    _DEFAULT_STAGING_POOL_NUM_BUFFERS,
)
from tensorrt_llm._torch.disaggregation.b10.protocol import (
    _DEFAULT_TAG_QUARANTINE_TTL_S,
    _DEFAULT_TAG_SPACE_SIZE,
)

_DEFAULT_ENDPOINT_POOL_SIZE = 1
_DEFAULT_MAX_IN_FLIGHT_OPS = 64
# Cap on concurrent large sends per agent. Unbounded concurrency makes all
# in-flight transfers share the D2H copy stream, NIC, and staging pool, so
# every transfer's latency balloons under load (v1 C++ uses a send
# concurrency of 1 for the same reason). Small transfers bypass the gate.
_DEFAULT_SEND_ADMISSION_LIMIT = 3
_DEFAULT_SEND_ADMISSION_BYPASS_BYTES = 512 * 1024 * 1024
# Smallest AM message the receive-side staging allocator serves from the
# pinned staging pool; smaller messages (control/READY/RESULT) use ucxx's
# internal host allocation so they never tie up a whole staging buffer.
_DEFAULT_AM_DIRECT_STAGING_MIN_BYTES = 1024 * 1024
_TRACE_TRANSFERS_ENV = "TRTLLM_B10_UCXX_TRACE_TRANSFERS"
_VALIDATE_SEND_SOURCE_ENV = "TRTLLM_B10_UCXX_VALIDATE_SEND_SOURCE"
_TRACE_LEVEL_NONE = "none"
_TRACE_LEVEL_INFO = "info"
_TRACE_LEVEL_DEBUG = "debug"

# UCXX application-context creation has been observed to take >90s under
# node-wide init contention (8 ranks opening ~16 RC devices while weights
# load); the startup budget must comfortably exceed that.
_DEFAULT_AGENT_STARTUP_TIMEOUT_S = 120.0


def _trace_transfer_level_from_env() -> str:
    value = os.getenv(_TRACE_TRANSFERS_ENV, _TRACE_LEVEL_NONE).lower()
    if value == _TRACE_LEVEL_INFO:
        return _TRACE_LEVEL_INFO
    if value == _TRACE_LEVEL_DEBUG:
        return _TRACE_LEVEL_DEBUG
    return _TRACE_LEVEL_NONE


@dataclass(frozen=True)
class B10AgentConfig:
    """Resolved B10 agent knobs (env-derived, constructor-overridable).

    Fields are listed in the order `from_env` resolves them, which mirrors
    the parse order of the inline `B10CacheTransferAgent.__init__` reads it
    replaced; `from_env` reproduces that parsing exactly (same env names,
    same defaults, same constructor-argument precedence).
    """

    port: int
    endpoint_pool_size: int
    max_in_flight_ops: int
    sync_cuda_before_transfer: bool
    validate_send_source: bool
    transfer_timeout_s: Optional[float]
    tag_space_size: int
    tag_quarantine_ttl_s: float
    staging_pool_num_buffers: int
    staging_pool_buffer_size: int
    am_direct_staging_min_bytes: int
    send_admission_limit: int
    send_admission_bypass_bytes: int
    recv_scratch_pool_num_buffers: int
    recv_scratch_metadata_max_spans: Optional[int]
    trace_transfer_level: str
    startup_timeout_s: float

    @classmethod
    def from_env(
        cls,
        *,
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
    ) -> "B10AgentConfig":
        port = int(os.getenv("TRTLLM_B10_UCXX_PORT", "0")) if port is None else port
        endpoint_pool_size = endpoint_pool_size or int(
            os.getenv("TRTLLM_B10_UCXX_ENDPOINT_POOL_SIZE", str(_DEFAULT_ENDPOINT_POOL_SIZE))
        )
        max_in_flight_ops = max_in_flight_ops or int(
            os.getenv("TRTLLM_B10_UCXX_MAX_IN_FLIGHT_OPS", str(_DEFAULT_MAX_IN_FLIGHT_OPS))
        )
        sync_cuda_before_transfer = (
            os.getenv("TRTLLM_B10_UCXX_SYNC_CUDA_BEFORE_TRANSFER", "0") == "1"
        )
        validate_send_source = os.getenv(_VALIDATE_SEND_SOURCE_ENV, "0") == "1"
        transfer_timeout_s = (
            b10_net._timeout_from_env() if transfer_timeout_s is None else transfer_timeout_s
        )
        if endpoint_pool_size <= 0:
            raise ValueError("endpoint_pool_size must be positive")
        if max_in_flight_ops <= 0:
            raise ValueError("max_in_flight_ops must be positive")

        tag_space = tag_space_size or int(
            os.getenv("TRTLLM_B10_UCXX_TAG_SPACE_SIZE", str(_DEFAULT_TAG_SPACE_SIZE))
        )
        quarantine_ttl_s = tag_quarantine_ttl_s or float(
            os.getenv("TRTLLM_B10_UCXX_TAG_QUARANTINE_TTL_S", str(_DEFAULT_TAG_QUARANTINE_TTL_S))
        )
        pool_num_buffers = (
            int(
                os.getenv(
                    "TRTLLM_B10_UCXX_STAGING_POOL_NUM_BUFFERS",
                    str(_DEFAULT_STAGING_POOL_NUM_BUFFERS),
                )
            )
            if staging_pool_num_buffers is None
            else staging_pool_num_buffers
        )
        pool_buffer_size = (
            int(
                os.getenv(
                    "TRTLLM_B10_UCXX_STAGING_POOL_BUFFER_SIZE_BYTES",
                    str(_DEFAULT_STAGING_POOL_BUFFER_SIZE),
                )
            )
            if staging_pool_buffer_size is None
            else staging_pool_buffer_size
        )
        am_direct_staging_min_bytes = int(
            os.getenv(
                "TRTLLM_B10_UCXX_AM_DIRECT_STAGING_MIN_BYTES",
                str(_DEFAULT_AM_DIRECT_STAGING_MIN_BYTES),
            )
        )
        admission_limit = (
            int(
                os.getenv(
                    "TRTLLM_B10_UCXX_SEND_ADMISSION_LIMIT", str(_DEFAULT_SEND_ADMISSION_LIMIT)
                )
            )
            if send_admission_limit is None
            else send_admission_limit
        )
        admission_bypass_bytes = (
            int(
                os.getenv(
                    "TRTLLM_B10_UCXX_SEND_ADMISSION_BYPASS_BYTES",
                    str(_DEFAULT_SEND_ADMISSION_BYPASS_BYTES),
                )
            )
            if send_admission_bypass_bytes is None
            else send_admission_bypass_bytes
        )
        scratch_pool_num_buffers = (
            int(os.getenv("TRTLLM_B10_UCXX_RECV_SCRATCH_POOL_NUM_BUFFERS", str(pool_num_buffers)))
            if recv_scratch_pool_num_buffers is None
            else recv_scratch_pool_num_buffers
        )
        scratch_metadata_max_spans_env = os.getenv(
            "TRTLLM_B10_UCXX_RECV_SCRATCH_METADATA_MAX_SPANS"
        )
        scratch_metadata_max_spans = (
            None if scratch_metadata_max_spans_env is None else int(scratch_metadata_max_spans_env)
        )
        startup_timeout_s = float(
            os.getenv(
                "TRTLLM_B10_UCXX_AGENT_STARTUP_TIMEOUT_S", str(_DEFAULT_AGENT_STARTUP_TIMEOUT_S)
            )
        )
        if startup_timeout_s <= 0:
            raise ValueError("startup_timeout_s must be positive")
        return cls(
            port=port,
            endpoint_pool_size=endpoint_pool_size,
            max_in_flight_ops=max_in_flight_ops,
            sync_cuda_before_transfer=sync_cuda_before_transfer,
            validate_send_source=validate_send_source,
            transfer_timeout_s=transfer_timeout_s,
            tag_space_size=tag_space,
            tag_quarantine_ttl_s=quarantine_ttl_s,
            staging_pool_num_buffers=pool_num_buffers,
            staging_pool_buffer_size=pool_buffer_size,
            am_direct_staging_min_bytes=am_direct_staging_min_bytes,
            send_admission_limit=admission_limit,
            send_admission_bypass_bytes=admission_bypass_bytes,
            recv_scratch_pool_num_buffers=scratch_pool_num_buffers,
            recv_scratch_metadata_max_spans=scratch_metadata_max_spans,
            trace_transfer_level=_trace_transfer_level_from_env(),
            startup_timeout_s=startup_timeout_s,
        )
