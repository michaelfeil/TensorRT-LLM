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
import threading
from concurrent.futures import CancelledError, Future, TimeoutError
from dataclasses import dataclass
from typing import Any, Callable, Hashable, Optional

import torch

from tensorrt_llm import logger
from tensorrt_llm._torch.disaggregation.b10 import memory as b10_memory
from tensorrt_llm._torch.disaggregation.b10 import protocol as b10_protocol
from tensorrt_llm._torch.disaggregation.b10.memory import _BufferView, _TransferChunk
from tensorrt_llm._torch.disaggregation.base.agent import TransferStatus

_DEFAULT_CANCEL_DRAIN_TIMEOUT_S = 1.0


class _EndpointSlot:
    """Persistent endpoint state for one serialized send lane.

    The send pipeline holds ``transfer_lock`` for the complete
    control/READY/DATA/RESULT exchange.
    """

    def __init__(self):
        self.endpoint: Optional[Any] = None
        self.generation: int = 0
        self.transfer_lock = asyncio.Lock()


@dataclass(frozen=True)
class _SendEndpointLease:
    remote_name: str
    slot_index: int
    slot: _EndpointSlot
    endpoint: Any
    endpoint_generation: int
    tag_domain: int


@dataclass(frozen=True)
class _SendTransferPlan:
    remote_name: str
    remote: b10_protocol.B10AgentDescriptor
    transfer_id: int
    src_type: str
    src_descs: b10_memory._DescArrayView
    dst_type: str
    dst_descs: b10_memory._DescArrayView
    transfer_chunks: list[b10_memory._TransferChunk]
    desc_count: int
    total_bytes: int
    max_desc_size: int
    wire_chunk_count: int
    max_wire_chunk_size: int
    desc_order_strategy: str
    src_span_count: int
    dst_span_count: int
    sync_message: Optional[str] = None


class _BufferCheckoutTracker:
    """Tracks checked-out views until normal or terminal cleanup owns them."""

    def __init__(self):
        self._views: list[b10_memory._BufferView] = []

    def track(self, staging_view: b10_memory._BufferView) -> None:
        self._views.append(staging_view)

    def untrack(self, staging_view: b10_memory._BufferView) -> None:
        for index, view in enumerate(self._views):
            if view is staging_view:
                del self._views[index]
                return

    def take_all(self) -> list[b10_memory._BufferView]:
        views = list(self._views)
        self._views.clear()
        return views


@dataclass(frozen=True)
class _EndpointAbortBinding:
    endpoint: Any
    retire: Callable[[], None]
    abort: Optional[Callable[[], None]]


class _TransferAbortHandle:
    def __init__(self):
        self._lock = threading.Lock()
        self._endpoint_binding: Optional[_EndpointAbortBinding] = None
        self._task_binding: Optional[tuple[asyncio.AbstractEventLoop, asyncio.Task]] = None
        self._cancel_requested = False

    def bind_current_task(self) -> None:
        """Bind cancellation to the current transfer task until it finishes.

        ``abort()`` consumes the binding before cancelling the task, so a
        repeated timeout or cancel cannot interrupt the task's cleanup.
        """

        task = asyncio.current_task()
        assert task is not None
        binding = (asyncio.get_running_loop(), task)
        with self._lock:
            cancel_requested = self._cancel_requested
            if not cancel_requested:
                assert self._task_binding is None
                self._task_binding = binding
        if cancel_requested:
            raise asyncio.CancelledError
        task.add_done_callback(self._clear_task_binding)

    def _clear_task_binding(self, task: asyncio.Task) -> None:
        with self._lock:
            if self._task_binding is not None and self._task_binding[1] is task:
                self._task_binding = None

    def bind_endpoint(
        self,
        endpoint: Any,
        retire_endpoint: Callable[[], None],
        abort_endpoint: Optional[Callable[[], None]] = None,
    ) -> None:
        binding = _EndpointAbortBinding(endpoint, retire_endpoint, abort_endpoint)
        with self._lock:
            cancel_requested = self._cancel_requested
            if not cancel_requested:
                self._endpoint_binding = binding
        if cancel_requested:
            retire_endpoint()
            if abort_endpoint is not None:
                abort_endpoint()

    def unbind_endpoint(self, endpoint: Any) -> None:
        with self._lock:
            binding = self._endpoint_binding
            if binding is not None and binding.endpoint is endpoint:
                self._endpoint_binding = None

    def abort(self) -> None:
        with self._lock:
            self._cancel_requested = True
            task_binding = self._task_binding
            self._task_binding = None
            endpoint_binding = self._endpoint_binding
            self._endpoint_binding = None
        if endpoint_binding is not None:
            endpoint_binding.retire()
            if endpoint_binding.abort is not None:
                endpoint_binding.abort()
        if task_binding is not None:
            loop, task = task_binding
            loop.call_soon_threadsafe(task.cancel)


class B10TransferStatus(TransferStatus):
    def __init__(
        self,
        future: Future,
        transfer_id: int,
        allocator: b10_protocol.B10TransferIdAllocator,
        abort_handle: _TransferAbortHandle,
        default_timeout_ms: Optional[int],
        cleanup_event: Optional[threading.Event] = None,
        tag_registry: Optional[b10_protocol.B10TagRegistry] = None,
        tag_owner: Optional[Hashable] = None,
    ):
        self._future = future
        self._transfer_id = transfer_id
        self._allocator = allocator
        self._abort_handle = abort_handle
        self._default_timeout_ms = default_timeout_ms
        self._cleanup_event = cleanup_event
        self._tag_registry = tag_registry
        self._tag_owner = tag_owner
        self._timed_out = False
        self._cancelled = False
        self._lock = threading.Lock()
        self._future.add_done_callback(self._on_done)

    def is_completed(self) -> bool:
        return self._future.done()

    def wait(self, timeout_ms: Optional[int] = None) -> bool:
        with self._lock:
            cancelled = self._cancelled
        if cancelled:
            self._drain_cleanup_after_cancel()
            return False

        effective_timeout_ms = self._default_timeout_ms if timeout_ms is None else timeout_ms
        timeout_s = None if effective_timeout_ms is None else effective_timeout_ms / 1000.0
        try:
            return bool(self._future.result(timeout=timeout_s))
        except TimeoutError:
            with self._lock:
                self._timed_out = True
            self._quarantine_resources()
            self._abort_handle.abort()
            self._drain_cleanup_after_cancel()
            logger.warning(
                f"B10 transfer {self._transfer_id} timed out after {effective_timeout_ms} ms"
            )
            return False
        except CancelledError:
            self._drain_cleanup_after_cancel()
            return False
        except Exception as exc:
            logger.error(f"B10 transfer {self._transfer_id} failed: {exc}")
            return False

    def cancel(self) -> None:
        with self._lock:
            if self._future.done():
                return
            self._cancelled = True
        self._quarantine_resources()
        self._abort_handle.abort()

    def _drain_cleanup_after_cancel(self) -> None:
        if self._cleanup_event is None:
            return
        if self._cleanup_event.wait(_DEFAULT_CANCEL_DRAIN_TIMEOUT_S):
            return
        logger.warning(
            f"B10 transfer {self._transfer_id} cleanup did not drain within "
            f"{_DEFAULT_CANCEL_DRAIN_TIMEOUT_S}s after cancellation"
        )

    def _on_done(self, future: Future) -> None:
        with self._lock:
            timed_out = self._timed_out
            cancelled = self._cancelled
        if timed_out or cancelled:
            return
        try:
            success = bool(future.result())
        except CancelledError:
            # BaseException since Python 3.8, so `except Exception` misses it.
            # Reached when the future is cancelled without going through
            # cancel()/wait()-timeout (e.g. event-loop shutdown); the wire
            # state is unknown, so quarantine rather than release.
            self._quarantine_resources()
            return
        except Exception:
            self._quarantine_resources()
            return
        if success:
            self._release_resources()
        else:
            self._quarantine_resources()

    def _release_resources(self) -> None:
        if self._tag_registry is not None and self._tag_owner is not None:
            self._tag_registry.release(self._tag_owner)
        self._allocator.release(self._transfer_id)

    def _quarantine_resources(self) -> None:
        if self._tag_registry is not None and self._tag_owner is not None:
            self._tag_registry.quarantine(self._tag_owner)
        self._allocator.quarantine(self._transfer_id)


class _CompletedTransferStatus(TransferStatus):
    def __init__(self):
        # Deliberately does not call super().__init__(): under the C++
        # binding, TransferStatus is a nanobind abstract base with no bound
        # constructor, so invoking the inherited __init__ raises "no
        # constructor defined". Shadowing it — exactly as B10TransferStatus
        # and _FailedTransferStatus already do — is what makes this class
        # instantiable with the real binding (the pure-Python ABC fallback
        # never hits this because object.__init__ is fine).
        pass

    def is_completed(self) -> bool:
        return True

    def wait(self, timeout_ms: Optional[int] = None) -> bool:
        return True


class _FailedTransferStatus(TransferStatus):
    def __init__(self, reason: str):
        self._reason = reason

    def is_completed(self) -> bool:
        return True

    def wait(self, timeout_ms: Optional[int] = None) -> bool:
        logger.error(self._reason)
        return False


_SourceReadyEvents = list[tuple[torch.device, Any]]


@dataclass(frozen=True)
class _ReceivedTransferChunk:
    chunk: _TransferChunk
    index: int
    staging_view: _BufferView
    spans: b10_memory._SpanArrays
    use_recv_scratch: bool


@dataclass(frozen=True)
class _RequestScatterChunk:
    chunk: _TransferChunk
    scratch_view: _BufferView
    spans: b10_memory._SpanArrays
