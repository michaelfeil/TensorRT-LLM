# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""Per-request tracing for LLM requests.

What: assigns each request a ``trace_id`` at admission and logs its lifecycle
transitions as ``trace_id``-prefixed lines, keeping them in memory for a
post-mortem dump on failure. A process-global registry maps ``request_id`` to
its tracer (and back).

Why: the OpenTelemetry span exported by ``do_tracing`` in
:mod:`tensorrt_llm.executor.result` only lands at completion. This gives a
``trace_id`` from the start so pod logs are greppable mid-flight, lets any code
path recover a request's ``trace_id`` from its id, and lets
``scripts/log_trace_annotator.py`` join the C++ ``[request_id=N]`` log lines to
the same ``trace_id``. Works without the OpenTelemetry SDK (log-only).

How::

    tracer = create_request_tracer(request_id)  # at admission
    tracer.transition("first_token", ...)  # at each lifecycle step
    tracer.record_error(err)  # on failure (dumps the trace)
    release_request_tracer(request_id)  # when done
"""

__all__ = [
    "RequestTracer",
    "RequestTracerRegistry",
    "get_request_tracer",
    "create_request_tracer",
    "release_request_tracer",
    "registry",
]

import dataclasses
import os
import secrets
import threading
import time
from typing import Any, Mapping, Optional

from tensorrt_llm.llmapi import tracing
from tensorrt_llm.logger import logger

_TRACE_ID_HEX_LEN = 32  # 128-bit, matches OTel trace_id format
_SPAN_ID_HEX_LEN = 16  # 64-bit, matches OTel span_id format

# Setting TRTLLM_REQUEST_TRACE_LOG_LEVEL=info makes the transition log lines
# show up at INFO; otherwise they are emitted at DEBUG so existing
# deployments see no extra log volume.
_TRANSITION_LOG_LEVEL = os.environ.get("TRTLLM_REQUEST_TRACE_LOG_LEVEL", "debug").lower()


def _new_trace_id() -> str:
    """Generate a 32-char hex trace id (compatible with OTel TraceFlags)."""
    return secrets.token_hex(_TRACE_ID_HEX_LEN // 2)


def _new_span_id() -> str:
    """Generate a 16-char hex span id."""
    return secrets.token_hex(_SPAN_ID_HEX_LEN // 2)


@dataclasses.dataclass
class _TransitionRecord:
    """In-memory record of a state transition; used for late dump on error."""

    timestamp: float  # time.time(), seconds since epoch
    event: str
    attributes: Mapping[str, Any]


class RequestTracer:
    """Tracks the lifecycle of one LLM request.

    Construct via :func:`create_request_tracer` so the instance lands in the
    process-global registry; the registry handles request_id ↔ trace_id
    lookups, which is the main reason to use this class over directly calling
    the OTel SDK.

    Logging: every :meth:`transition` call emits a log line prefixed with
    ``trace_id=<hex>``. Set ``TRTLLM_REQUEST_TRACE_LOG_LEVEL=info``
    to surface them at INFO (default is DEBUG so we don't disturb existing
    log volume).
    """

    def __init__(
        self,
        request_id: int,
        *,
        trace_id: Optional[str] = None,
        attributes: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.request_id = int(request_id)
        self.trace_id = trace_id or _new_trace_id()
        self.created_at = time.time()
        self._attributes = dict(attributes) if attributes else {}
        self._transitions: list[_TransitionRecord] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def transition(self, event: str, **attributes: Any) -> None:
        """Record one state transition and log it with a trace_id prefix.

        The log line is emitted regardless of whether OpenTelemetry is
        configured; OTel span events are added on top if a span is active.
        """
        record = _TransitionRecord(
            timestamp=time.time(),
            event=event,
            attributes=attributes,
        )
        with self._lock:
            self._transitions.append(record)
        self._emit_log(event, attributes)
        if tracing.is_tracing_enabled():
            tracing.add_event(event, attributes=attributes)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def record_error(self, error: Any, **attributes: Any) -> None:
        """Record a terminal error and dump the full trace at ERROR level.

        Records a ``request_error`` transition (so it joins the in-memory log
        and any active OTel span) and then emits the whole accumulated trace as
        a single ERROR line, so a failed request leaves a post-mortem keyed by
        ``trace_id`` even when transition logging is otherwise at DEBUG.
        """
        self.transition("request_error", error=str(error), **attributes)
        logger.error(self.dump())

    def snapshot(self) -> list[_TransitionRecord]:
        """Return a copy of the transition log so far."""
        with self._lock:
            return list(self._transitions)

    def dump(self) -> str:
        """Render the transitions as a multi-line string.

        Intended for post-mortem dumps from error paths.
        """
        with self._lock:
            transitions = list(self._transitions)
        lines = [
            f"Request trace request_id={self.request_id} "
            f"trace_id={self.trace_id} transitions={len(transitions)}:"
        ]
        for rec in transitions:
            delta_ms = (rec.timestamp - self.created_at) * 1000.0
            attrs = (
                " " + " ".join(f"{k}={v}" for k, v in rec.attributes.items())
                if rec.attributes
                else ""
            )
            lines.append(f"  +{delta_ms:.2f}ms {rec.event}{attrs}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    # Events that always log at INFO regardless of TRTLLM_REQUEST_TRACE_LOG_LEVEL.
    # `request_created` carries the canonical trace_id ↔ request_id mapping line:
    # if it gets DEBUG-filtered, downstream tools (e.g. log_trace_annotator.py)
    # cannot join existing [request_id=N] log lines to a trace_id. The single
    # info-level line per request is acceptable log volume.
    _ALWAYS_INFO_EVENTS = frozenset({"request_created"})

    def _emit_log(self, event: str, attributes: Mapping[str, Any]) -> None:
        base = (
            " " + " ".join(f"{k}={v}" for k, v in self._attributes.items())
            if self._attributes
            else ""
        )
        suffix = " " + " ".join(f"{k}={v}" for k, v in attributes.items()) if attributes else ""
        message = (
            f"trace_id={self.trace_id} request_id={self.request_id}{base} event={event}{suffix}"
        )
        if event in self._ALWAYS_INFO_EVENTS or _TRANSITION_LOG_LEVEL == "info":
            logger.info(message)
        else:
            logger.debug(message)


class RequestTracerRegistry:
    """Process-global registry mapping ``request_id`` ↔ ``RequestTracer``.

    Used so any code path that has the request id can recover the trace id
    (and the transition log so far) without threading a context object
    everywhere.
    """

    # Bound the registry so a code path that fails to release a tracer (an
    # aborted or stuck request that never reaches a terminal transition) cannot
    # grow it without limit; the oldest live tracer is evicted past this cap.
    _DEFAULT_MAX_TRACERS = 8192

    def __init__(self, max_size: int = _DEFAULT_MAX_TRACERS) -> None:
        self._lock = threading.Lock()
        self._max_size = max_size
        self._by_request_id: dict[int, RequestTracer] = {}
        self._by_trace_id: dict[str, RequestTracer] = {}

    def create(
        self,
        request_id: int,
        *,
        trace_id: Optional[str] = None,
        attributes: Optional[Mapping[str, Any]] = None,
    ) -> RequestTracer:
        tracer = RequestTracer(request_id, trace_id=trace_id, attributes=attributes)
        with self._lock:
            # Replace any prior tracer for the same id (e.g. if the previous
            # request was never properly released).
            existing = self._by_request_id.pop(int(request_id), None)
            if existing is not None:
                self._by_trace_id.pop(existing.trace_id, None)
            # Evict the oldest (insertion-ordered) entries if a leak has pushed
            # us to the cap, so the registry stays bounded.
            while len(self._by_request_id) >= self._max_size:
                oldest_id, oldest = next(iter(self._by_request_id.items()))
                self._by_request_id.pop(oldest_id, None)
                self._by_trace_id.pop(oldest.trace_id, None)
            self._by_request_id[tracer.request_id] = tracer
            self._by_trace_id[tracer.trace_id] = tracer
        tracer.transition("request_created")
        return tracer

    def get(self, request_id: int) -> Optional[RequestTracer]:
        with self._lock:
            return self._by_request_id.get(int(request_id))

    def get_by_trace_id(self, trace_id: str) -> Optional[RequestTracer]:
        with self._lock:
            return self._by_trace_id.get(trace_id)

    def release(self, request_id: int) -> Optional[RequestTracer]:
        with self._lock:
            tracer = self._by_request_id.pop(int(request_id), None)
            if tracer is not None:
                self._by_trace_id.pop(tracer.trace_id, None)
        if tracer is not None:
            tracer.transition("request_released")
        return tracer

    def clear(self) -> None:
        with self._lock:
            self._by_request_id.clear()
            self._by_trace_id.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._by_request_id)


registry = RequestTracerRegistry()


def create_request_tracer(
    request_id: int,
    *,
    trace_id: Optional[str] = None,
    attributes: Optional[Mapping[str, Any]] = None,
) -> RequestTracer:
    """Create + register a tracer for ``request_id``.

    Convenience wrapper around :py:obj:`registry.create`.
    """
    return registry.create(request_id, trace_id=trace_id, attributes=attributes)


def get_request_tracer(request_id: int) -> Optional[RequestTracer]:
    """Look up the tracer for ``request_id``, or ``None`` if not registered."""
    return registry.get(request_id)


def release_request_tracer(request_id: int) -> Optional[RequestTracer]:
    """Remove the tracer for ``request_id`` from the registry.

    Returns the tracer (so callers can still :meth:`dump` it if needed).
    """
    return registry.release(request_id)
