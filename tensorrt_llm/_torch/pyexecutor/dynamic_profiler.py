import json
import os
import re
import time
import traceback
import uuid

import torch

from tensorrt_llm.logger import logger

DYNAMIC_PROFILE_DIR_ENV = "TLLM_DYNAMIC_PROFILE_DIR"
MAX_SECONDS_ENV = "TLLM_DYNAMIC_PROFILE_MAX_SECONDS"
MAX_ITERS_ENV = "TLLM_DYNAMIC_PROFILE_MAX_ITERS"
EST_BYTES_PER_ITER_ENV = "TLLM_DYNAMIC_PROFILE_EST_BYTES_PER_ITER"
DEFAULT_MAX_SECONDS = 120
DEFAULT_MAX_ITERS = 1000
DEFAULT_EST_BYTES_PER_ITER = 20 * 1024 * 1024  # ~uncompressed trace bytes per iter per rank
DISK_HEADROOM_FACTOR = 1.2
POLL_INTERVAL_ITERS = 16
ACTIVE_LOG_INTERVAL_ITERS = 100
# Session ids become file names; reject anything path-unsafe.
_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")


class DynamicProfilerSession:
    """File-triggered torch profiler for the PyExecutor loop.

    Watches <dir>/request.json; on a new request id, starts a torch
    profiler and stops at whichever comes first: the requested window
    (seconds or iterations), MAX_ITERS, or MAX_SECONDS. Writes
    <dir>/trace-<id>-rank<N>.json.gz plus a done marker recording why
    it stopped. All entry points swallow exceptions — profiling must
    never crash the executor loop; after the first unexpected failure
    the session disables itself for the process lifetime.
    """

    def __init__(self, profile_dir: str, global_rank: int):
        self._dir = profile_dir
        self._rank = global_rank
        self._max_seconds = int(os.environ.get(MAX_SECONDS_ENV, DEFAULT_MAX_SECONDS))
        self._max_iters = int(os.environ.get(MAX_ITERS_ENV, DEFAULT_MAX_ITERS))
        self._est_bytes_per_iter = int(
            os.environ.get(EST_BYTES_PER_ITER_ENV, DEFAULT_EST_BYTES_PER_ITER)
        )
        self._last_id = None
        self._profiler = None
        self._request = None
        self._start_iter = None
        self._start_time = None
        self._req_seconds = None  # requested wall-clock window, or None
        self._req_iters = None  # requested iteration window, or None
        self._failed = False
        os.makedirs(profile_dir, exist_ok=True)

    @property
    def active(self) -> bool:
        return self._profiler is not None

    def step(self, iter_counter: int) -> None:
        """Called once per executor iteration from profile_step()."""
        if self._failed:
            return
        try:
            if self.active:
                iters_done = iter_counter - self._start_iter
                if iters_done and iters_done % ACTIVE_LOG_INTERVAL_ITERS == 0:
                    logger.info(
                        f"Dynamic profiling active: id={self._request['id']} "
                        f"rank={self._rank} {iters_done} iters "
                        f"{time.monotonic() - self._start_time:.1f}s"
                    )
                reason = self._stop_reason(iter_counter)
                if reason is not None:
                    self._stop_and_export(iter_counter, reason)
            elif iter_counter % POLL_INTERVAL_ITERS == 0:
                self._maybe_start(iter_counter)
        except Exception:
            logger.error(
                "Dynamic profiler failed; disabling for this process\n" + traceback.format_exc()
            )
            self._failed = True
            self._profiler = None

    def shutdown(self, iter_counter: int) -> None:
        """Flush an in-flight session on loop exit."""
        if self.active and not self._failed:
            try:
                self._stop_and_export(iter_counter, "executor_shutdown")
            except Exception:
                logger.error("Dynamic profiler failed during shutdown\n" + traceback.format_exc())

    def _stop_reason(self, iter_counter: int):
        iters_done = iter_counter - self._start_iter
        elapsed = time.monotonic() - self._start_time
        if self._req_iters is not None and iters_done >= self._req_iters:
            return "requested"
        if self._req_seconds is not None and elapsed >= self._req_seconds:
            return "requested"
        if iters_done >= self._max_iters:
            return "max_iters"
        if elapsed >= self._max_seconds:
            return "max_seconds"
        return None

    def _maybe_start(self, iter_counter: int) -> None:
        request_path = os.path.join(self._dir, "request.json")
        try:
            with open(request_path) as f:
                request = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return
        req_id = request.get("id")
        if not req_id or req_id == self._last_id:
            return
        self._last_id = req_id  # set first: never retry a request, even a bad one
        if not _ID_RE.fullmatch(req_id):
            # No error marker: the id is unusable in a file name.
            logger.warning(f"Dynamic profiling request ignored: unsafe id {req_id!r}")
            return

        ranks = request.get("ranks")
        if ranks is not None and self._rank not in ranks:
            return  # not our rank; no marker — the collector knows the rank set

        # REST layer enforces seconds XOR iterations; be lenient here.
        req_iters = request.get("iterations")
        req_seconds = request.get("seconds")
        if req_iters is None and req_seconds is None:
            self._write_error_marker(req_id, "request must specify seconds or iterations")
            return
        if req_iters is not None:
            req_iters = min(int(req_iters), self._max_iters)
            req_seconds = None  # iterations wins if both slipped through
        else:
            req_seconds = min(int(req_seconds), self._max_seconds)

        # Disk precheck: export writes uncompressed JSON before gzipping.
        iter_bound = req_iters if req_iters is not None else self._max_iters
        needed = int(iter_bound * self._est_bytes_per_iter * DISK_HEADROOM_FACTOR)
        stat = os.statvfs(self._dir)
        free = stat.f_bavail * stat.f_frsize
        if free < needed:
            self._write_error_marker(
                req_id,
                f"insufficient disk space in {self._dir}: "
                f"{free} bytes free, ~{needed} needed; reduce iterations, "
                f"limit ranks, or free disk",
            )
            return

        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        self._profiler = torch.profiler.profile(
            activities=activities,
            record_shapes=bool(request.get("record_shapes", False)),
            with_stack=bool(request.get("with_stack", False)),
            with_modules=True,
        )
        self._profiler.start()
        self._request = request
        self._req_iters = req_iters
        self._req_seconds = req_seconds
        self._start_iter = iter_counter
        self._start_time = time.monotonic()
        logger.info(
            f"Dynamic profiling started: id={req_id} rank={self._rank} "
            f"iter={iter_counter} window="
            f"{f'{req_iters} iters' if req_iters else f'{req_seconds}s'} "
            f"caps=({self._max_iters} iters, {self._max_seconds}s)"
        )

    def _stop_and_export(self, iter_counter: int, reason: str) -> None:
        profiler, request = self._profiler, self._request
        self._profiler = None
        profiler.stop()
        req_id = request["id"]
        trace_name = f"trace-{req_id}-rank{self._rank}.json.gz"
        trace_path = os.path.join(self._dir, trace_name)
        tmp_path = f"{trace_path}.tmp-{uuid.uuid4().hex}.gz"
        profiler.export_chrome_trace(tmp_path)
        os.rename(tmp_path, trace_path)
        del profiler  # release kineto/CUPTI state

        self._write_marker(
            req_id,
            {
                "id": req_id,
                "rank": self._rank,
                "stopped_by": reason,
                "iter_start": self._start_iter,
                "iter_stop": iter_counter,
                "iterations": iter_counter - self._start_iter,
                "wall_seconds": round(time.monotonic() - self._start_time, 3),
                "trace_file": trace_name,
                "size_bytes": os.path.getsize(trace_path),
            },
        )

    def _write_error_marker(self, req_id: str, message: str) -> None:
        logger.warning(f"Dynamic profiling request {req_id} rejected: {message}")
        self._write_marker(
            req_id,
            {
                "id": req_id,
                "rank": self._rank,
                "error": message,
            },
        )

    def _write_marker(self, req_id: str, payload: dict) -> None:
        marker_path = os.path.join(self._dir, f"done-{req_id}-rank{self._rank}.json")
        tmp = f"{marker_path}.tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.rename(tmp, marker_path)
        logger.info(f"Dynamic profiling marker written: {payload}")
