# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""Unit tests for tensorrt_llm.llmapi.request_tracer."""

import logging
import threading

import pytest

from tensorrt_llm.llmapi.request_tracer import (
    RequestTracer,
    RequestTracerRegistry,
    create_request_tracer,
    get_request_tracer,
    registry,
    release_request_tracer,
)


@pytest.fixture(autouse=True)
def clean_registry():
    """Each test starts with an empty process-global registry."""
    registry.clear()
    yield
    registry.clear()


# --------------------------------------------------------------------------
# Trace id generation
# --------------------------------------------------------------------------


def test_trace_id_is_unique_32_char_hex():
    t1 = RequestTracer(1)
    t2 = RequestTracer(2)
    assert t1.trace_id != t2.trace_id
    for tid in (t1.trace_id, t2.trace_id):
        assert len(tid) == 32
        int(tid, 16)  # raises if not hex


def test_explicit_trace_id_is_honoured():
    t = RequestTracer(7, trace_id="deadbeef" * 4)
    assert t.trace_id == "deadbeef" * 4


# --------------------------------------------------------------------------
# Transition logging
# --------------------------------------------------------------------------


def test_transition_is_recorded(caplog):
    tracer = RequestTracer(42)
    tracer.transition("prefill_started", tokens=128)
    tracer.transition("first_token", ttft_ms=12.5)
    snap = tracer.snapshot()
    assert [r.event for r in snap] == ["prefill_started", "first_token"]
    assert snap[0].attributes == {"tokens": 128}
    assert snap[1].attributes == {"ttft_ms": 12.5}


def test_log_line_contains_trace_id_and_request_id(monkeypatch, caplog):
    # Force the log level used by the tracer to INFO so caplog captures it
    # under any default logger configuration.
    monkeypatch.setenv("TRTLLM_REQUEST_TRACE_LOG_LEVEL", "info")
    # Re-import to pick up the new env. We re-create the tracer; the level is
    # read at module import, so re-evaluating it requires a reload.
    import importlib

    import tensorrt_llm.llmapi.request_tracer as rt_module

    importlib.reload(rt_module)
    try:
        with caplog.at_level(logging.INFO, logger="tensorrt_llm"):
            tracer = rt_module.RequestTracer(99)
            tracer.transition("queued", priority=1)
        joined = "\n".join(rec.getMessage() for rec in caplog.records)
        assert f"trace_id={tracer.trace_id}" in joined
        assert "request_id=99" in joined
        assert "event=queued" in joined
        assert "priority=1" in joined
    finally:
        importlib.reload(rt_module)


def test_request_created_always_logs_at_info(caplog):
    # The `request_created` event carries the canonical trace_id ↔ request_id
    # mapping line that downstream tooling (scripts/log_trace_annotator.py) uses
    # to join existing [request_id=N] log lines to a trace_id. If this line gets
    # DEBUG-filtered in production (default log level is INFO), the entire
    # binding chain breaks. Guard against a future maintainer accidentally
    # moving the emit back to DEBUG by checking the level is INFO regardless
    # of the TRTLLM_REQUEST_TRACE_LOG_LEVEL setting.
    with caplog.at_level(logging.DEBUG, logger="tensorrt_llm"):
        tracer = create_request_tracer(31415)
    created_records = [
        r
        for r in caplog.records
        if f"trace_id={tracer.trace_id}" in r.getMessage()
        and "event=request_created" in r.getMessage()
    ]
    assert len(created_records) == 1, [r.getMessage() for r in caplog.records]
    assert created_records[0].levelno == logging.INFO


def test_request_created_line_bridges_external_id(caplog):
    # The whole point of stamping a client-facing id (e.g. an OpenAI
    # `chatcmpl-*`) as a tracer attribute: the always-INFO `request_created`
    # line then carries external id + trace_id + request_id together, so a
    # single grep on the external id recovers the engine-internal trace_id.
    with caplog.at_level(logging.INFO, logger="tensorrt_llm"):
        tracer = create_request_tracer(120937, attributes={"external_request_id": "chatcmpl-9db23"})
    created = [r.getMessage() for r in caplog.records if "event=request_created" in r.getMessage()]
    assert len(created) == 1, [r.getMessage() for r in caplog.records]
    line = created[0]
    assert "external_request_id=chatcmpl-9db23" in line
    assert f"trace_id={tracer.trace_id}" in line
    assert "request_id=120937" in line


def test_non_created_events_default_to_debug(caplog):
    # Counterpart to the test above: subsequent transitions must remain at
    # DEBUG when TRTLLM_REQUEST_TRACE_LOG_LEVEL is unset, otherwise we flood
    # INFO logs with per-token transitions.
    with caplog.at_level(logging.DEBUG, logger="tensorrt_llm"):
        tracer = create_request_tracer(2718)
        tracer.transition("token_emitted", index=0)
    token_records = [r for r in caplog.records if "event=token_emitted" in r.getMessage()]
    assert len(token_records) == 1
    assert token_records[0].levelno == logging.DEBUG


def test_dump_includes_all_transitions():
    tracer = RequestTracer(11)
    tracer.transition("a")
    tracer.transition("b", x=1)
    dumped = tracer.dump()
    assert "request_id=11" in dumped
    assert "trace_id=" + tracer.trace_id in dumped
    assert "a" in dumped
    assert "b" in dumped
    assert "x=1" in dumped


def test_record_error_records_transition_and_dumps_at_error(caplog):
    tracer = RequestTracer(12)
    tracer.transition("prefill_started", tokens=64)
    with caplog.at_level(logging.ERROR, logger="tensorrt_llm"):
        tracer.record_error("kv cache transfer failed")

    # The error becomes a transition in the in-memory log...
    events = [r.event for r in tracer.snapshot()]
    assert events == ["prefill_started", "request_error"]
    assert tracer.snapshot()[-1].attributes == {"error": "kv cache transfer failed"}

    # ...and the full trace is dumped at ERROR level, keyed by trace_id, so a
    # failed request leaves a post-mortem even if transitions log at DEBUG.
    error_text = "\n".join(
        rec.getMessage() for rec in caplog.records if rec.levelno >= logging.ERROR
    )
    assert f"trace_id={tracer.trace_id}" in error_text
    assert "request_id=12" in error_text
    assert "prefill_started" in error_text
    assert "kv cache transfer failed" in error_text


# --------------------------------------------------------------------------
# Registry behaviour
# --------------------------------------------------------------------------


def test_create_registers_and_get_by_id():
    tracer = create_request_tracer(123)
    assert get_request_tracer(123) is tracer
    assert registry.get_by_trace_id(tracer.trace_id) is tracer


def test_release_removes_tracer():
    create_request_tracer(7)
    assert get_request_tracer(7) is not None
    released = release_request_tracer(7)
    assert released is not None
    assert get_request_tracer(7) is None
    # Releasing an unknown id is a no-op.
    assert release_request_tracer(7) is None


def test_create_with_same_request_id_replaces_previous_entry():
    first = create_request_tracer(3)
    second = create_request_tracer(3)
    assert first is not second
    assert get_request_tracer(3) is second
    # The first tracer's trace_id should no longer be reachable.
    assert registry.get_by_trace_id(first.trace_id) is None


def test_len_tracks_unique_request_ids():
    create_request_tracer(1)
    create_request_tracer(2)
    create_request_tracer(1)  # replacement, still one slot
    assert len(registry) == 2
    release_request_tracer(1)
    assert len(registry) == 1


def test_registry_evicts_oldest_past_cap():
    # Bounded registry: a leak (tracers never released) must not grow without
    # limit; the oldest live entries are evicted past the cap.
    reg = RequestTracerRegistry(max_size=3)
    tracers = [reg.create(rid) for rid in range(5)]
    assert len(reg) == 3
    # The two oldest were evicted, by both id and trace_id (no map leak).
    assert reg.get(0) is None
    assert reg.get(1) is None
    assert reg.get_by_trace_id(tracers[0].trace_id) is None
    # The three newest are retained and reachable both ways.
    for rid in (2, 3, 4):
        assert reg.get(rid) is not None
        assert reg.get_by_trace_id(tracers[rid].trace_id) is reg.get(rid)


# --------------------------------------------------------------------------
# Thread safety smoke test
# --------------------------------------------------------------------------


def test_concurrent_create_and_release_does_not_corrupt_registry():
    threads = []

    def worker(start_id: int):
        for i in range(50):
            rid = start_id + i
            tracer = create_request_tracer(rid)
            tracer.transition("x")
            release_request_tracer(rid)

    for t in range(4):
        threads.append(threading.Thread(target=worker, args=(t * 1000,)))
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert len(registry) == 0
