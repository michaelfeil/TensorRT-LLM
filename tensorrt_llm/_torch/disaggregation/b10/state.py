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
    def __init__(self):
        self.endpoint: Optional[Any] = None
        self.generation: int = 0
        self.lock = asyncio.Lock()


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


class _StagingCheckoutTracker:
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


class _TransferAbortHandle:
    def __init__(self):
        self._lock = threading.Lock()
        self._endpoint: Optional[Any] = None
        self._retire: Optional[Callable[[], None]] = None
        self._abort: Optional[Callable[[], None]] = None
        self._cancel_requested = False
        self._abort_called = False

    def set_endpoint(
        self, endpoint: Any, retire: Callable[[], None], abort: Optional[Callable[[], None]] = None
    ) -> None:
        with self._lock:
            cancel_requested = self._cancel_requested
            self._endpoint = endpoint
            self._retire = retire
            self._abort = abort
            should_abort = cancel_requested and abort is not None and not self._abort_called
            if should_abort:
                self._abort_called = True
        if cancel_requested:
            retire()
            if should_abort:
                abort()

    def clear_endpoint(self, endpoint: Any) -> None:
        with self._lock:
            if self._endpoint is endpoint:
                self._endpoint = None
                self._retire = None
                self._abort = None

    def abort(self) -> None:
        with self._lock:
            self._cancel_requested = True
            endpoint = self._endpoint
            retire = self._retire
            abort = self._abort
            should_abort = endpoint is not None and abort is not None and not self._abort_called
            if should_abort:
                self._abort_called = True
        if retire is not None:
            retire()
        if should_abort:
            abort()

    def is_cancel_requested(self) -> bool:
        with self._lock:
            return self._cancel_requested


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
