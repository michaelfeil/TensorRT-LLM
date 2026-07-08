# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Asynchronous grammar compilation for guided decoding."""

import enum
import os
import threading
import time
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Dict, Optional, Tuple

from ...bindings.executor import GuidedDecodingParams
from .grammar_matcher import GrammarMatcher, GrammarMatcherFactory


def compile_timeout_message(budget_s: float) -> str:
    return f"Guided decoding error: grammar compilation did not finish within {budget_s:.0f}s."


def format_guided_error(e: BaseException) -> str:
    """Format an exception as a guided decoding error message (idempotently)."""
    msg = str(e)
    if msg.startswith("Guided decoding error"):
        return msg
    if not msg:
        msg = type(e).__name__
    return f"Guided decoding error: {msg}"


class CompileState(enum.Enum):
    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"


class AsyncGrammarCompiler:
    """Compiles grammar matchers on a thread pool, keyed by request id.

    Grammar compilation can take arbitrarily long (e.g. superlinear in JSON
    schema size), so it is submitted when a request is activated and overlaps
    prefill / KV-cache transfer instead of blocking the forward path.

    Threading contract: the executor thread submits, polls and discards;
    CUDA-callback threads call take(). All state is guarded by a lock and any
    wait in take() is bounded, so it is safe inside a CUDA host callback.
    The compiled matcher is carried through the future itself, so it cannot
    be evicted from the factory's LRU cache between submit and take.
    """

    def __init__(self, factory: GrammarMatcherFactory, *, allow_sync_fallback: bool):
        self.factory = factory
        # Whether take() may compile synchronously when a request was never
        # submitted. Must be False when take() runs inside CUDA host callbacks.
        self.allow_sync_fallback = allow_sync_fallback
        self._pool: Optional[ThreadPoolExecutor] = None
        self._lock = threading.Lock()
        # request_id -> (future, submit timestamp)
        self._pending: Dict[int, Tuple[Future, float]] = {}
        self._ready: Dict[int, GrammarMatcher] = {}
        self._failed: Dict[int, str] = {}
        self._compile_timeout_s = float(os.getenv("TRTLLM_GUIDED_COMPILE_TIMEOUT_SEC", "60"))
        self._take_wait_s = float(os.getenv("TRTLLM_GUIDED_ATTACH_WAIT_SEC", "10"))

    def submit(self, request_id: int, params: Optional[GuidedDecodingParams]) -> None:
        """Kick off grammar compilation for a request; idempotent."""
        if params is None:
            return
        with self._lock:
            if (
                request_id in self._pending
                or request_id in self._ready
                or request_id in self._failed
            ):
                return
            if self._pool is None:
                # Lazy so the compiler survives shutdown(): the KV-estimation
                # probe executor shares this decoder with the final executor.
                self._pool = ThreadPoolExecutor(
                    max_workers=int(os.getenv("TRTLLM_GUIDED_COMPILE_WORKERS", "4")),
                    thread_name_prefix="guided-compile",
                )
            future = self._pool.submit(self.factory.create, params)
            self._pending[request_id] = (future, time.monotonic())

    def poll(self, request_id: int) -> CompileState:
        """Return the compile state for a request, harvesting finished futures.

        A compile pending past the compile timeout (from submit) is failed —
        the backstop against a wedged compile. The tighter scheduling-defer
        budget is enforced by the coordinator.
        """
        with self._lock:
            if request_id in self._ready:
                return CompileState.READY
            if request_id in self._failed:
                return CompileState.FAILED
            entry = self._pending.get(request_id)
            if entry is None:
                # Never submitted (e.g. entered through a path that skips
                # activation); let take() resolve it.
                return CompileState.READY
            future, submitted_at = entry
            if future.done():
                del self._pending[request_id]
                self._harvest_finished_future(request_id, future)
                return CompileState.READY if request_id in self._ready else CompileState.FAILED
            if time.monotonic() - submitted_at > self._compile_timeout_s:
                del self._pending[request_id]
                future.cancel()
                self._failed[request_id] = compile_timeout_message(self._compile_timeout_s)
                return CompileState.FAILED
            return CompileState.PENDING

    def take(
        self, request_id: int, params: Optional[GuidedDecodingParams]
    ) -> Tuple[Optional[GrammarMatcher], str]:
        """Pop the compiled matcher for a request, or explain why not.

        Safe to call from CUDA-callback threads: a wait on an in-flight
        compile is bounded by the attach-wait budget (schedule gating makes
        such waits rare) and Future.result releases the GIL while waiting.
        """
        with self._lock:
            matcher = self._ready.pop(request_id, None)
            if matcher is not None:
                return matcher, ""
            error_msg = self._failed.pop(request_id, None)
            if error_msg is not None:
                return None, error_msg
            entry = self._pending.pop(request_id, None)
        if entry is not None:
            future, _ = entry
            try:
                return future.result(timeout=self._take_wait_s), ""
            except FuturesTimeoutError:
                error_msg = (
                    "Guided decoding error: grammar compilation still pending "
                    f"after waiting {self._take_wait_s:.0f}s at matcher attach."
                )
                # Cache the verdict so later queries stay consistent; cancel()
                # frees the pool slot if the compile hasn't started.
                future.cancel()
                self._record_failed(request_id, error_msg)
                return None, error_msg
            # CancelledError is a BaseException; it must not escape into a
            # CUDA host callback.
            except (CancelledError, Exception) as e:
                error_msg = format_guided_error(e)
                self._record_failed(request_id, error_msg)
                return None, error_msg
        if self.allow_sync_fallback:
            try:
                return self.factory.create(params), ""
            except Exception as e:
                return None, format_guided_error(e)
        return None, (f"Guided decoding error: no precompiled grammar for request {request_id}.")

    def mark_failed(self, request_id: int, error_msg: str) -> None:
        """Force a request's compile state to failed.

        Used to apply a broadcast rank-0 verdict on the other TP ranks so
        matcher attach fails immediately and identically everywhere: a ready
        matcher is dropped (its request is being terminated), while an
        already-recorded failure keeps its more specific local message.
        """
        with self._lock:
            if request_id in self._failed:
                return
            entry = self._pending.pop(request_id, None)
            if entry is not None:
                entry[0].cancel()
            self._ready.pop(request_id, None)
            self._failed[request_id] = error_msg

    def discard(self, request_id: int) -> None:
        """Drop any compile state for a finished/canceled request."""
        with self._lock:
            entry = self._pending.pop(request_id, None)
            if entry is not None:
                entry[0].cancel()
            self._ready.pop(request_id, None)
            self._failed.pop(request_id, None)

    def shutdown(self) -> None:
        """Drop queued compiles without waiting (a wedged compile thread must
        not block teardown); the pool re-creates on the next submit()."""
        with self._lock:
            for entry in self._pending.values():
                entry[0].cancel()
            self._pending.clear()
            pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)

    def _record_failed(self, request_id: int, error_msg: str) -> None:
        # setdefault: a concurrent mark_failed / earlier verdict wins.
        with self._lock:
            self._failed.setdefault(request_id, error_msg)

    def _harvest_finished_future(self, request_id: int, future: Future) -> None:
        # Caller must hold _lock.
        try:
            self._ready[request_id] = future.result(timeout=0)
        except (CancelledError, Exception) as e:
            self._failed[request_id] = format_guided_error(e)
