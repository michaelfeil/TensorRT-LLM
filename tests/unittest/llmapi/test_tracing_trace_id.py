# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""Unit tests for trace_id unification helpers in tensorrt_llm.llmapi.tracing.

These back the guarantee that the per-request tracer's trace_id equals the
exported ``llm_request`` span's trace_id, so a trace_id grepped from pod logs
matches the trace_id the OTLP backend indexes by.
"""

import pytest

from tensorrt_llm.llmapi import tracing

pytestmark = pytest.mark.skipif(
    not tracing.is_otel_available(), reason="OpenTelemetry SDK not installed"
)


def test_trace_id_from_empty_context_is_none():
    # No inbound traceparent -> no valid span -> None (caller then seeds its own).
    ctx = tracing.extract_trace_context(None)
    assert tracing.trace_id_from_context(ctx) is None


def test_parent_context_roundtrips_trace_id():
    trace_id = "0123456789abcdef0123456789abcdef"
    ctx = tracing.parent_context_for_trace_id(trace_id)
    assert tracing.trace_id_from_context(ctx) == trace_id


def test_seeded_span_adopts_trace_id():
    """A span started under parent_context_for_trace_id inherits that id."""
    from opentelemetry.sdk.trace import TracerProvider

    trace_id = "abcdef0123456789abcdef0123456789"
    tracer = TracerProvider().get_tracer("test")
    ctx = tracing.parent_context_for_trace_id(trace_id)
    with tracer.start_as_current_span("llm_request", context=ctx) as span:
        assert format(span.get_span_context().trace_id, "032x") == trace_id


def test_propagated_traceparent_is_extracted():
    # W3C traceparent: version-traceid-spanid-flags
    trace_id = "11111111111111111111111111111111"
    headers = {"traceparent": f"00-{trace_id}-2222222222222222-01"}
    ctx = tracing.extract_trace_context(headers)
    assert tracing.trace_id_from_context(ctx) == trace_id
