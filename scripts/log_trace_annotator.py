#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Back-fill ``trace_id`` into trtllm pod logs for request-level debugging.

Why: trtllm's C++ log lines (``TLLM_LOG_REQ_*``) are tagged with
``[request_id=N]`` but carry no trace_id, while the Python request tracer logs
a one-time mapping line ``... trace_id=<hex> request_id=<int>
event=request_created ...``. This tool joins the two so you can pick a single
trace_id and grep one request's whole timeline across both C++ and Python lines.

What it is: a stand-alone, offline text filter — not part of the serving path,
and it persists nothing. Point it at logs you already have (a ``kubectl logs``
tail or a saved file); it reads stdin (or ``--in FILE``) and writes the
annotated stream to stdout for you to read or redirect. "Annotate" just means
rewriting each ``[request_id=N]`` in place to ``[request_id=N trace_id=<hex>]``;
lines whose request id has no known mapping pass through unchanged.

Modes:

* default (streaming): one pass, suitable for ``kubectl logs -f`` tails. A
  ``[request_id=N]`` line seen before its trace_id mapping (rare, possible
  across rank reorderings) is left as-is.

* ``--two-pass``: read a static file twice (pass 1 builds the full mapping,
  pass 2 rewrites) so every request id is annotated even when its mapping
  appears later in the file.

Examples::

    kubectl logs <pod> -f | python scripts/log_trace_annotator.py
    python scripts/log_trace_annotator.py --in pod.log --two-pass > pod.annotated.log
"""

import argparse
import re
import sys
from typing import Dict, Iterable, TextIO

# Matches the request-scoped log prefix added by TLLM_LOG_REQ_* macros in
# cpp/include/tensorrt_llm/common/logger.h. Capture group 1 = request id.
REQUEST_ID_RE = re.compile(r"\[request_id=(\d+)\]")

# Matches the trace_id token emitted by request_tracer._emit_log. Capture
# group 1 = trace_id (16-32 hex chars to cover both span-id and trace-id
# lengths), group 2 = paired request id.
JOINT_RE = re.compile(r"trace_id=([0-9a-fA-F]{16,32})\s+request_id=(\d+)")


def harvest_mapping(line: str, mapping: Dict[int, str]) -> None:
    """If ``line`` contains a ``trace_id=... request_id=...`` pair, record it.

    The mapping is updated in-place. Later observations overwrite earlier ones
    so that if a request id is reused (e.g. after a worker restart) we follow
    the most recent trace.
    """
    match = JOINT_RE.search(line)
    if match:
        trace_id, request_id = match.group(1), int(match.group(2))
        mapping[request_id] = trace_id


def annotate_line(line: str, mapping: Dict[int, str]) -> str:
    """Return ``line`` with every ``[request_id=N]`` annotated when possible.

    No-op when the request id has no mapping yet — preserves the original line
    verbatim so a subsequent two-pass run can still annotate it.
    """

    def _replace(match: "re.Match[str]") -> str:
        rid = int(match.group(1))
        tid = mapping.get(rid)
        if tid is None:
            return match.group(0)
        return f"[request_id={rid} trace_id={tid}]"

    return REQUEST_ID_RE.sub(_replace, line)


def stream_annotate(input_stream: TextIO, output_stream: TextIO) -> None:
    """Single-pass: forward each line after annotating with mapping so-far."""
    mapping: Dict[int, str] = {}
    for line in input_stream:
        harvest_mapping(line, mapping)
        output_stream.write(annotate_line(line, mapping))


def two_pass_annotate(lines: Iterable[str], output_stream: TextIO) -> None:
    """Two-pass: build the full mapping first, then rewrite.

    Materializes the input into memory; use only for static files small enough
    to fit. For unbounded streams use :func:`stream_annotate` instead.
    """
    buffered = list(lines)
    mapping: Dict[int, str] = {}
    for line in buffered:
        harvest_mapping(line, mapping)
    for line in buffered:
        output_stream.write(annotate_line(line, mapping))


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--in",
        dest="input_path",
        default=None,
        help="Read from this file instead of stdin.",
    )
    parser.add_argument(
        "--out",
        dest="output_path",
        default=None,
        help="Write to this file instead of stdout.",
    )
    parser.add_argument(
        "--two-pass",
        action="store_true",
        help=(
            "Make two passes over the input so that request_ids whose trace_id "
            "mapping appears later are still annotated."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    input_stream: TextIO = open(args.input_path, "r") if args.input_path else sys.stdin
    output_stream: TextIO = open(args.output_path, "w") if args.output_path else sys.stdout
    try:
        if args.two_pass:
            two_pass_annotate(input_stream, output_stream)
        else:
            stream_annotate(input_stream, output_stream)
    finally:
        if args.input_path:
            input_stream.close()
        if args.output_path:
            output_stream.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
