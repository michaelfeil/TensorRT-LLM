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
from typing import Any, Callable, Optional

from tensorrt_llm import logger


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


async def _await_with_timeout(
    awaitable: Any,
    timeout_s: Optional[float],
    on_late_result: Optional[Callable[[Any], None]] = None,
    cancel_on_timeout: bool = True,
) -> Any:
    if timeout_s is None:
        return await awaitable
    task = asyncio.ensure_future(awaitable)
    try:
        done, _ = await asyncio.wait({task}, timeout=timeout_s)
    except asyncio.CancelledError:
        if cancel_on_timeout:
            task.cancel()
        task.add_done_callback(lambda task: _consume_background_task_result(task, on_late_result))
        raise
    if task in done:
        return await task
    if cancel_on_timeout:
        task.cancel()
    task.add_done_callback(lambda task: _consume_background_task_result(task, on_late_result))
    raise TimeoutError


async def _gather_cancel_on_failure(tasks: list[asyncio.Task]) -> None:
    """Await all tasks together; if any fails or the caller is cancelled,
    cancel the stragglers and drain them before re-raising."""

    async def _cancel_and_drain() -> None:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        await _cancel_and_drain()
        raise
    except Exception:
        await _cancel_and_drain()
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
