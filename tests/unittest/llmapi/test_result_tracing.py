# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""Unit tests for request-trace helpers in tensorrt_llm.executor.result."""

from tensorrt_llm.executor.result import _MAX_EXTERNAL_REQUEST_ID_LEN, _sanitize_trace_attr_value


def test_sanitize_passes_through_a_clean_id():
    # A normal chatcmpl-* id has no whitespace and is left untouched.
    assert _sanitize_trace_attr_value("chatcmpl-9db23") == "chatcmpl-9db23"


def test_sanitize_collapses_whitespace_to_block_log_injection():
    # A value with spaces/newlines must not be able to forge extra key=value
    # tokens (e.g. a fake event=) into a space-joined trace log line.
    injected = "chatcmpl-x event=spoofed\nrequest_id=999"
    out = _sanitize_trace_attr_value(injected)
    assert " " not in out and "\n" not in out
    assert out == "chatcmpl-x_event=spoofed_request_id=999"


def test_sanitize_truncates_to_cap():
    out = _sanitize_trace_attr_value("a" * (_MAX_EXTERNAL_REQUEST_ID_LEN + 50))
    assert len(out) == _MAX_EXTERNAL_REQUEST_ID_LEN


def test_sanitize_coerces_non_str():
    assert _sanitize_trace_attr_value(12345) == "12345"
