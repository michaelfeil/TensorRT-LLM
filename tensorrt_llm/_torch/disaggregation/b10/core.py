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
"""Shared live state for the B10 UCXX transfer agent.

`_AgentCore` is a plain object constructed by
`B10CacheTransferAgent.__init__` and handed to every collaborator. It owns
the state both pipelines (and the agent shell) share:

- ``config``: the resolved `B10AgentConfig`
- ``loop``: the agent asyncio event loop
- ``transfer_ids``: `B10TransferIdAllocator`
- ``tag_registry``: `B10TagRegistry`
- ``staging_buffer_pool`` / ``staging_buffer_slots``: pinned staging pool and
  its concurrency permits (the staging-slot helper methods live here)
- ``recv_scratch_buffer_pool``: CUDA scratch pool (recv fill; timing logs)
- ``cuda_copy_streams``: per-device copy streams
- ``max_in_flight_ops`` / ``sync_cuda_before_transfer`` /
  ``transfer_timeout_s``: config-derived knobs both pipelines read
- ``shutdown``: agent shutdown flag, set by the shell's ``shutdown()`` and
  read by the recv pipeline's endpoint listener loop
"""

from __future__ import annotations

import asyncio
from typing import Any

from tensorrt_llm import logger
from tensorrt_llm._torch.disaggregation.b10.async_utils import (
    _await_with_timeout,
    _TransferDeadline,
)
from tensorrt_llm._torch.disaggregation.b10.config import B10AgentConfig
from tensorrt_llm._torch.disaggregation.b10.memory import _BufferView
from tensorrt_llm._torch.disaggregation.b10.pools import (
    _CudaCopyStreamPool,
    _CudaScratchBufferPool,
    _NoStagingBufferAvailableError,
    _PinnedStagingBufferPool,
)
from tensorrt_llm._torch.disaggregation.b10.protocol import B10TagRegistry, B10TransferIdAllocator


class _AgentCore:
    def __init__(
        self,
        config: B10AgentConfig,
        *,
        loop: asyncio.AbstractEventLoop,
        transfer_ids: B10TransferIdAllocator,
        tag_registry: B10TagRegistry,
        staging_buffer_pool: _PinnedStagingBufferPool,
        staging_buffer_slots: asyncio.BoundedSemaphore,
        recv_scratch_buffer_pool: _CudaScratchBufferPool,
        cuda_copy_streams: _CudaCopyStreamPool,
    ):
        self.config = config
        self.loop = loop
        self.transfer_ids = transfer_ids
        self.tag_registry = tag_registry
        self.staging_buffer_pool = staging_buffer_pool
        self.staging_buffer_slots = staging_buffer_slots
        self.recv_scratch_buffer_pool = recv_scratch_buffer_pool
        self.cuda_copy_streams = cuda_copy_streams
        self.max_in_flight_ops = config.max_in_flight_ops
        self.sync_cuda_before_transfer = config.sync_cuda_before_transfer
        self.transfer_timeout_s = config.transfer_timeout_s
        self.shutdown = False

    async def _acquire_staging_buffer(self, size: int, deadline: _TransferDeadline) -> _BufferView:
        while True:
            await _await_with_timeout(self.staging_buffer_slots.acquire(), deadline.remaining_s())
            try:
                return self.staging_buffer_pool.acquire(size)
            except _NoStagingBufferAvailableError:
                self.staging_buffer_slots.release()
                # A permit was free but no buffer was ready (transient while
                # failed transfers hold buffers in quarantine). Back off and
                # retry; remaining_s() raises TimeoutError once the transfer
                # deadline expires, so the loop is time-bounded.
                deadline.remaining_s()
                await asyncio.sleep(0.001)
            except Exception:
                self.staging_buffer_slots.release()
                raise

    async def _release_staging_slot_after_event(self, event: Any) -> None:
        # The slot is a concurrency permit, not the buffer: the pool keeps the
        # buffer quarantined while its event is pending, so the permit must be
        # returned even when the event wait fails or capacity leaks for good.
        try:
            await asyncio.to_thread(event.synchronize)
        except Exception as exc:
            logger.warning(
                f"B10 staging slot event wait failed before slot release: "
                f"error={type(exc).__name__}: {exc}"
            )
        finally:
            self.staging_buffer_slots.release()

    def _release_staging_slots_for_views(self, views: list[_BufferView]) -> None:
        for view in views:
            if view.pool is not self.staging_buffer_pool:
                continue
            event = view.ready_event
            if event is not None and not self._event_ready_for_slot_release(event, "staging"):
                asyncio.create_task(self._release_staging_slot_after_event(event))
            else:
                self.staging_buffer_slots.release()

    def _release_staging_buffers(self, views: list[_BufferView]) -> None:
        self.staging_buffer_pool.release(views)
        self._release_staging_slots_for_views(views)

    @staticmethod
    def _event_ready_for_slot_release(event: Any, label: str) -> bool:
        try:
            return event.query()
        except Exception as exc:
            logger.warning(
                f"B10 {label} copy event query failed before slot release; "
                f"waiting for event before slot release: "
                f"error={type(exc).__name__}: {exc}"
            )
            return False

    @staticmethod
    def _wait_copy_events(events: list[Any]) -> None:
        for event in events:
            event.synchronize()

    async def _wait_copy_events_async(self, events: list[Any], deadline: _TransferDeadline) -> None:
        if not events:
            return
        await _await_with_timeout(
            asyncio.to_thread(self._wait_copy_events, events), deadline.remaining_s()
        )
