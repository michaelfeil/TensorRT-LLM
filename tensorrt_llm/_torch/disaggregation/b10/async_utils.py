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
import time
from concurrent.futures import TimeoutError
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

from tensorrt_llm import logger


def _format_endpoint_handles(endpoint: Any) -> str:
    """The UCX pointers by which UCX's own logs name an endpoint.

    UCX identifies an endpoint in its diagnostics only as `worker %p ... ep %p`
    (e.g. `ucp_worker.c: keepalive failed on ep 0x.. lane[n]`). Emitting the
    same two pointers next to a peer's name is what makes those lines
    attributable. Never raises: this only ever decorates a log line, and a
    closed endpoint must not turn that into a failure.
    """
    try:
        return f"ucp_ep=0x{endpoint.ucp_endpoint:x} ucp_worker=0x{endpoint.ucp_worker:x}"
    except Exception as exc:
        return f"ucp_ep=unavailable ({type(exc).__name__})"


def _abort_endpoint_background(endpoint: Any) -> None:
    def abort() -> None:
        try:
            endpoint.abort()
        except Exception as exc:
            logger.debug(f"B10 endpoint abort failed: {exc}")

    threading.Thread(target=abort, name="b10-endpoint-abort", daemon=True).start()


def _consume_background_task_result(
    task: asyncio.Task, on_result: Optional[Callable[[Any], None]] = None
) -> None:
    try:
        result = task.result()
    except asyncio.CancelledError:
        return
    except Exception as exc:
        logger.debug(f"B10 background operation completed with error: {exc}")
        return
    if on_result is not None:
        try:
            on_result(result)
        except Exception as exc:
            logger.debug(f"B10 background result callback failed: {exc}")


class _TransferDeadline:
    """Absolute deadline fencing every await inside one transfer.

    `remaining_s()` raises `TimeoutError` the moment the deadline passes —
    it never returns zero or a negative — so waits and retry loops built on
    it are time-bounded by construction. A `None` timeout disables the
    fence entirely. See DESIGN.md "Timeout and failure handling".
    """

    def __init__(self, timeout_s: Optional[float]):
        self._deadline_s = None if timeout_s is None else time.monotonic() + timeout_s

    def remaining_s(self) -> Optional[float]:
        if self._deadline_s is None:
            return None
        remaining_s = self._deadline_s - time.monotonic()
        if remaining_s <= 0:
            raise TimeoutError
        return remaining_s


async def _wait_with_timeout(
    awaitable: Any,
    timeout_s: Optional[float],
    on_late_result: Optional[Callable[[Any], None]] = None,
    *,
    cancel_pending: bool,
) -> Any:
    task = asyncio.ensure_future(awaitable)

    def finish_in_background() -> None:
        if cancel_pending:
            task.cancel()
        task.add_done_callback(lambda task: _consume_background_task_result(task, on_late_result))

    try:
        done, _ = await asyncio.wait({task}, timeout=timeout_s)
    except asyncio.CancelledError:
        finish_in_background()
        raise
    if task in done:
        return await task
    finish_in_background()
    raise TimeoutError


async def _await_with_timeout(
    awaitable: Any,
    timeout_s: Optional[float],
    on_late_result: Optional[Callable[[Any], None]] = None,
) -> Any:
    """Await until completion or timeout, cancelling unfinished work."""

    return await _wait_with_timeout(
        awaitable,
        timeout_s,
        on_late_result,
        cancel_pending=True,
    )


async def _await_detached_with_timeout(
    awaitable: Any,
    timeout_s: Optional[float],
    on_late_result: Optional[Callable[[Any], None]] = None,
) -> Any:
    """Stop waiting on timeout or cancellation without cancelling the work.

    UCXX operations use this because cancelling their Python awaitable does
    not prove that the native operation has stopped touching its buffers.
    The detached task consumes its eventual result in the background.
    """

    return await _wait_with_timeout(
        awaitable,
        timeout_s,
        on_late_result,
        cancel_pending=False,
    )


async def _acquire_with_timeout(
    synchronizer: asyncio.Lock | asyncio.Semaphore,
    timeout_s: Optional[float],
) -> None:
    """Acquire a lock or permit without leaking a late acquisition.

    On return, ownership belongs to the caller, which must release it. If
    timeout or cancellation wins a race with acquisition, any concurrent
    acquisition is released automatically.
    """

    def release_if_acquired(acquired: bool) -> None:
        if acquired:
            synchronizer.release()

    await _await_with_timeout(
        synchronizer.acquire(),
        timeout_s,
        on_late_result=release_if_acquired,
    )


@asynccontextmanager
async def _lock_with_timeout(
    lock: asyncio.Lock,
    timeout_s: Optional[float],
) -> AsyncIterator[None]:
    """Acquire ``lock`` within ``timeout_s`` and release it after the block.

    The timeout applies only to acquisition; the block uses its own deadlines.
    """

    await _acquire_with_timeout(lock, timeout_s)
    try:
        yield
    finally:
        lock.release()


async def _gather_cancel_on_failure(tasks: list[asyncio.Task]) -> None:
    """Await all tasks together; if any fails or the caller is cancelled,
    cancel the stragglers and drain them before re-raising."""

    try:
        await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def _run_limited(
    coroutine_factories: list[Callable[[], Awaitable[Any]]],
    *,
    max_in_flight: int,
    transfer_id: Optional[int] = None,
    phase: str = "transfer",
) -> None:
    """Run a bounded worker pool, cancelling and draining it on failure."""

    total = len(coroutine_factories)
    next_index = 0
    completed = 0

    async def worker() -> None:
        nonlocal next_index, completed
        while next_index < total:
            factory = coroutine_factories[next_index]
            next_index += 1
            await factory()
            completed += 1

    workers = [asyncio.create_task(worker()) for _ in range(min(total, max(1, max_in_flight)))]
    try:
        await _gather_cancel_on_failure(workers)
    except BaseException as exc:
        cancelled = isinstance(exc, asyncio.CancelledError)
        error_detail = "" if cancelled else f" error={type(exc).__name__}: {exc}"
        logger.warning(
            f"B10 {phase} transfer {transfer_id} batch "
            f"{'cancelled' if cancelled else 'failed'}: "
            f"started={next_index} completed={completed} total={total}{error_detail}"
        )
        raise


def _is_retryable_endpoint_error(exc: Exception) -> bool:
    if isinstance(exc, ConnectionError):
        return True
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    if "timeout" in name:
        return False
    return (
        "ucxx" in name
        and ("connection" in name or "endpoint" in name or "reset" in name or "closed" in name)
    ) or (
        "connection reset" in message
        or "connection closed" in message
        or "endpoint closed" in message
    )
