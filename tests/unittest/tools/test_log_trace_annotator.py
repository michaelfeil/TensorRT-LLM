# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import io
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent.parent.resolve()

# Dynamically load the script as a module without polluting sys.path so the
# test does not depend on adding scripts/ to PYTHONPATH.
_spec = importlib.util.spec_from_file_location(
    "log_trace_annotator", REPO_ROOT / "scripts" / "log_trace_annotator.py"
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
annotate_line = _module.annotate_line
harvest_mapping = _module.harvest_mapping
stream_annotate = _module.stream_annotate
two_pass_annotate = _module.two_pass_annotate

# 32-char hex string, the trace_id length emitted by RequestTracer.
SAMPLE_TRACE_ID = "0123456789abcdef0123456789abcdef"
SAMPLE_TRACE_ID_2 = "fedcba9876543210fedcba9876543210"

# A line in the canonical mapping form that request_tracer._emit_log produces
# at INFO for the `request_created` event.
MAPPING_LINE = (
    f"2026-06-03 00:00:00 INFO trtllm.llmapi.request_tracer "
    f"trace_id={SAMPLE_TRACE_ID} request_id=42 event=request_created\n"
)


class HarvestMappingTest(unittest.TestCase):
    """Mapping extraction must only fire on the exact ``trace_id ... request_id`` shape."""

    def test_extracts_from_canonical_line(self):
        mapping = {}
        harvest_mapping(MAPPING_LINE, mapping)
        self.assertEqual(mapping, {42: SAMPLE_TRACE_ID})

    def test_ignores_lines_without_both_tokens(self):
        # request_id alone — no trace_id, no mapping.
        mapping = {}
        harvest_mapping("[TensorRT-LLM][WARNING][request_id=42] cancelled\n", mapping)
        self.assertEqual(mapping, {})

    def test_later_observation_overwrites(self):
        # Useful for worker restarts: when the same request_id appears again
        # with a new trace_id, follow the latest.
        mapping = {}
        harvest_mapping(MAPPING_LINE, mapping)
        new_line = MAPPING_LINE.replace(SAMPLE_TRACE_ID, SAMPLE_TRACE_ID_2)
        harvest_mapping(new_line, mapping)
        self.assertEqual(mapping, {42: SAMPLE_TRACE_ID_2})


class AnnotateLineTest(unittest.TestCase):
    """``[request_id=N]`` should be rewritten exactly when the mapping is known."""

    def test_rewrites_known_request_id(self):
        line = "[TensorRT-LLM][WARNING][request_id=42] cancel timeout\n"
        out = annotate_line(line, {42: SAMPLE_TRACE_ID})
        self.assertIn(f"[request_id=42 trace_id={SAMPLE_TRACE_ID}]", out)
        # Surrounding content preserved.
        self.assertIn("cancel timeout", out)

    def test_unknown_request_id_passes_through(self):
        # An unmapped request_id must remain untouched so the same line can be
        # picked up by a later --two-pass run that has the full mapping.
        line = "[TensorRT-LLM][WARNING][request_id=99] unknown\n"
        out = annotate_line(line, {})
        self.assertEqual(out, line)

    def test_multiple_occurrences_on_one_line(self):
        # Belt-and-suspenders: a line with two distinct request_ids (e.g. a
        # cancel propagation log mentioning both upstream and downstream ids)
        # gets each one annotated independently.
        line = "edge [request_id=1] -> [request_id=2] propagation\n"
        out = annotate_line(line, {1: "aaaaaaaaaaaaaaaa", 2: "bbbbbbbbbbbbbbbb"})
        self.assertIn("[request_id=1 trace_id=aaaaaaaaaaaaaaaa]", out)
        self.assertIn("[request_id=2 trace_id=bbbbbbbbbbbbbbbb]", out)

    def test_does_not_touch_inline_request_id_outside_brackets(self):
        # Existing legacy log lines using "request id: 42" without the bracket
        # form must stay verbatim; the macro contract is the bracket form only.
        line = "Exception in sendAndRemoveResponse: oom request id: 42\n"
        out = annotate_line(line, {42: SAMPLE_TRACE_ID})
        self.assertEqual(out, line)


class StreamAnnotateTest(unittest.TestCase):
    """End-to-end one-pass annotation."""

    def test_mapping_line_then_later_log(self):
        # The mapping line comes first, so by the time the cancel log is
        # processed the mapping is already populated — the one-pass code path
        # must produce a fully annotated cancel log.
        input_lines = [
            MAPPING_LINE,
            "[TensorRT-LLM][WARNING][request_id=42] cancel timeout\n",
        ]
        out = io.StringIO()
        stream_annotate(iter(input_lines), out)
        out_lines = out.getvalue().splitlines(keepends=True)
        self.assertEqual(len(out_lines), 2)
        # The mapping line itself is preserved (it already has trace_id in it
        # outside the bracket form, so no rewrite is needed there).
        self.assertEqual(out_lines[0], MAPPING_LINE)
        self.assertIn(f"[request_id=42 trace_id={SAMPLE_TRACE_ID}]", out_lines[1])

    def test_log_before_mapping_is_left_alone(self):
        # If a [request_id=N] line precedes the mapping line in stream order,
        # one-pass cannot annotate it. Verify the line is forwarded verbatim
        # (not dropped, not partially rewritten) so users can rerun in
        # --two-pass for a complete annotation.
        input_lines = [
            "[TensorRT-LLM][WARNING][request_id=42] cancel timeout\n",
            MAPPING_LINE,
        ]
        out = io.StringIO()
        stream_annotate(iter(input_lines), out)
        out_lines = out.getvalue().splitlines(keepends=True)
        self.assertEqual(out_lines[0], input_lines[0])
        self.assertEqual(out_lines[1], MAPPING_LINE)


class TwoPassAnnotateTest(unittest.TestCase):
    """Two-pass mode must annotate even when mappings come after their request id."""

    def test_log_before_mapping_is_annotated_in_two_pass(self):
        # This is the value-add of --two-pass: pass 1 collects the mapping
        # from line 2, pass 2 rewrites the [request_id=42] on line 1.
        input_lines = [
            "[TensorRT-LLM][WARNING][request_id=42] cancel timeout\n",
            MAPPING_LINE,
        ]
        out = io.StringIO()
        two_pass_annotate(iter(input_lines), out)
        out_lines = out.getvalue().splitlines(keepends=True)
        self.assertIn(f"[request_id=42 trace_id={SAMPLE_TRACE_ID}]", out_lines[0])

    def test_empty_input(self):
        # Pure safety: a degenerate empty input should not throw.
        out = io.StringIO()
        two_pass_annotate(iter([]), out)
        self.assertEqual(out.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
