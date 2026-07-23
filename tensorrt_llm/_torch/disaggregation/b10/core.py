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
"""State and buffer-ownership operations shared by both B10 pipelines.

``_AgentCore`` owns the event loop, identity registries, staging and scratch
pools, CUDA copy streams, and resolved runtime limits. Buffer permit helpers
live here so send and receive share one ownership contract.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable

from tensorrt_llm import logger
from tensorrt_llm._torch.disaggregation.b10.async_utils import (
    _acquire_with_timeout,
    _await_with_timeout,
    _TransferDeadline,
)
from tensorrt_llm._torch.disaggregation.b10.config import B10AgentConfig
from tensorrt_llm._torch.disaggregation.b10.memory import _BufferView
from tensorrt_llm._torch.disaggregation.b10.pools import (
    _STAGING_VIEW_AM_DIRECT,
    _CudaCopyStreamPool,
    _CudaScratchBufferPool,
    _format_staging_pool_state,
    _NoStagingBufferAvailableError,
    _PinnedStagingBufferPool,
    _QuarantinedBufferViews,
)
from tensorrt_llm._torch.disaggregation.b10.protocol import B10TransferIdAllocator


class _AgentCore:
    def __init__(
        self,
        config: B10AgentConfig,
        *,
        loop: asyncio.AbstractEventLoop,
        transfer_ids: B10TransferIdAllocator,
        staging_buffer_pool: _PinnedStagingBufferPool,
        staging_buffer_slots: asyncio.BoundedSemaphore,
        recv_scratch_buffer_pool: _CudaScratchBufferPool,
        cuda_copy_streams: _CudaCopyStreamPool,
    ):
        self.config = config
        self.loop = loop
        self.transfer_ids = transfer_ids
        self.staging_buffer_pool = staging_buffer_pool
        self.staging_buffer_slots = staging_buffer_slots
        self.recv_scratch_buffer_pool = recv_scratch_buffer_pool
        self.cuda_copy_streams = cuda_copy_streams
        self._staging_quarantine_ttl_s = config.tag_quarantine_ttl_s
        self._quarantined_staging_views: list[_QuarantinedBufferViews] = []
        self.max_in_flight_ops = config.max_in_flight_ops
        self.sync_cuda_before_transfer = config.sync_cuda_before_transfer
        self.transfer_timeout_s = config.transfer_timeout_s
        self.shutdown = False

    async def _acquire_staging_buffer(self, size: int, deadline: _TransferDeadline) -> _BufferView:
        return await self._acquire_pooled_buffer(
            self.staging_buffer_slots,
            lambda: self.staging_buffer_pool.acquire(size),
            deadline,
        )

    @staticmethod
    async def _acquire_pooled_buffer(
        slots: asyncio.BoundedSemaphore,
        acquire: Callable[[], _BufferView],
        deadline: _TransferDeadline,
    ) -> _BufferView:
        """Checkout a pooled buffer and transfer its capacity permit."""

        while True:
            await _acquire_with_timeout(slots, deadline.remaining_s())
            try:
                return acquire()
            except _NoStagingBufferAvailableError:
                slots.release()
                # A permit was free but no buffer was ready (transient while
                # failed transfers hold buffers in quarantine). Back off and
                # retry; remaining_s() raises TimeoutError once the transfer
                # deadline expires, so the loop is time-bounded.
                deadline.remaining_s()
                await asyncio.sleep(0.001)
            except Exception:
                slots.release()
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
            if view.metadata and view.metadata.get(_STAGING_VIEW_AM_DIRECT):
                # Delivered straight into staging by the AM allocator: no
                # slot permit was taken for it, so none goes back. The pool
                # checkout itself is still released/quarantined normally.
                continue
            event = view.ready_event
            if event is not None and not self._event_ready_for_slot_release(event, "staging"):
                asyncio.create_task(self._release_staging_slot_after_event(event))
            else:
                self.staging_buffer_slots.release()

    def _release_staging_buffers(self, views: list[_BufferView]) -> None:
        self.staging_buffer_pool.release(views)
        self._release_staging_slots_for_views(views)

    def _prune_quarantined_staging_buffers(self) -> None:
        now = time.monotonic()
        self._quarantined_staging_views = [
            entry
            for entry in self._quarantined_staging_views
            if now - entry.quarantined_at < self._staging_quarantine_ttl_s
            or any(
                view.ready_event is not None and not view.ready_event.query()
                for view in entry.views
            )
        ]

    def _quarantine_staging_buffers(
        self, transfer_id: int, views: list[_BufferView], direction: str
    ) -> None:
        self._prune_quarantined_staging_buffers()
        if not views:
            return
        self.staging_buffer_pool.quarantine(views)
        self._release_staging_slots_for_views(views)
        self._quarantined_staging_views.append(
            _QuarantinedBufferViews(views=views, quarantined_at=time.monotonic())
        )
        logger.warning(
            f"B10 {direction} transfer {transfer_id} quarantined staging buffers: "
            f"count={len(views)} "
            f"active_quarantines={len(self._quarantined_staging_views)} "
            f"{_format_staging_pool_state(self.staging_buffer_pool)}"
        )

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
