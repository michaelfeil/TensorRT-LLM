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
"""Watches PyExecutor's requests and loop health, and logs anomalies.

- ``request_state_dwell`` (INFO): a request left a state after longer than
  that state's budget.
- ``request_state_stuck`` (INFO, once per request): a request has sat in
  the same state past ``STUCK_THRESHOLD_S``.
- ``pyexecutor_heartbeat`` (INFO): iteration timing, per-phase maxima,
  state counts, and waiting-queue depth.
- Slow-phase WARN (rate-limited): a loop phase exceeded its budget.

The request scan runs at most once per ``SCAN_INTERVAL_S``; per-iteration
cost is O(1).
"""

import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from tensorrt_llm.logger import logger

# A request that leaves one of these states after more than its budget logs
# request_state_dwell. Budgets sit half a second under nominal so scan
# jitter does not push detection a full interval past the intended bound.
DWELL_BUDGETS_S = {
    "DISAGG_GENERATION_INIT": 4.5,
    "DISAGG_GENERATION_TRANS_IN_PROGRESS": 14.5,
    "DISAGG_GENERATION_TRANS_COMPLETE": 4.5,
    "DISAGG_CONTEXT_TRANS_IN_PROGRESS": 14.5,
    "DISAGG_TRANS_ERROR": 4.5,
}

STUCK_THRESHOLD_S = 29.5
STUCK_EXEMPT_STATES = frozenset({"GENERATION_IN_PROGRESS"})

# A loop phase measured above its budget logs a rate-limited WARN. Phases
# are timed exactly at their call sites, so no jitter adjustment is needed.
PHASE_BUDGETS_S = {
    "schedule": 1.0,
    "sample_update": 5.0,
    "respond": 1.0,
    "disagg_status": 2.0,
}
SLOW_UPDATE_WARN_S = PHASE_BUDGETS_S["sample_update"]

OBSERVER_ENV_VAR = "TRTLLM_B10_PYEXECUTOR_OBSERVER"

# Minimum wall-clock gap between request scans.
SCAN_INTERVAL_S = 1.0
# Cadence of the pyexecutor_heartbeat line.
HEARTBEAT_INTERVAL_S = 120.0
_PHASE_WARN_INTERVAL_S = 60.0


def maybe_create_b10_pyexecutor_observer(
    rank: int, should_log_request_lines: bool
) -> Optional["B10PyExecutorObserver"]:
    """Returns None when the observer is disabled via OBSERVER_ENV_VAR."""
    if os.environ.get(OBSERVER_ENV_VAR, "1").strip().lower() in ("0", "false", "off"):
        return None
    return B10PyExecutorObserver(rank=rank, should_log_request_lines=should_log_request_lines)


def _state_name(request) -> str:
    state = request.state
    return getattr(state, "name", None) or str(state)


def _ext_id(request) -> str:
    return getattr(request, "py_external_request_id", None) or "-"


@dataclass
class _RequestStateEntryData:
    state: str
    entered_at: float
    ext_id: str
    is_stuck_logged: bool = False


class B10PyExecutorObserver:
    def __init__(self, rank: int, should_log_request_lines: bool):
        self._rank: int = rank
        # Without attention-dp every rank sees identical requests, so only
        # one rank logs the per-request lines.
        self._should_log_request_lines: bool = should_log_request_lines
        self._request_state_data: Dict[int, _RequestStateEntryData] = {}
        self._last_heartbeat_at: Optional[float] = None
        self._last_iteration_at: Optional[float] = None
        self._last_scan_at: Optional[float] = None
        self._last_phase_warn_at: Dict[str, float] = {}
        # Refreshed by the scan, consumed by the heartbeat.
        self._state_counts: Dict[str, int] = {}
        self._waiting_len: int = 0
        # Window counters, reset each heartbeat.
        self._iterations: int = 0
        self._iter_s_sum: float = 0.0
        self._iter_s_max: float = 0.0
        self._phase_s_max: Dict[str, float] = {}
        self._departed: int = 0

    def on_iteration_start(self, active_requests, waiting_len: int) -> None:
        """Called once at the top of each executor iteration."""
        now = time.monotonic()
        if self._last_iteration_at is not None:
            iter_s = now - self._last_iteration_at
            self._iterations += 1
            self._iter_s_sum += iter_s
            self._iter_s_max = max(self._iter_s_max, iter_s)
        self._last_iteration_at = now

        if self._last_scan_at is None or now - self._last_scan_at >= SCAN_INTERVAL_S:
            self._last_scan_at = now
            self._scan_requests_and_log(active_requests, waiting_len, now)

        self._maybe_heartbeat(now)

    def record_phase(
        self, phase: str, duration_s: float, detail: Optional[Callable[[], str]] = None
    ) -> None:
        """Record one loop phase's wall time; warn (rate-limited) if slow."""
        if duration_s > self._phase_s_max.get(phase, 0.0):
            self._phase_s_max[phase] = duration_s
        budget_s = PHASE_BUDGETS_S.get(phase)
        if budget_s is None or duration_s <= budget_s:
            return
        now = time.monotonic()
        last_warn_at = self._last_phase_warn_at.get(phase)
        if last_warn_at is not None and now - last_warn_at < _PHASE_WARN_INTERVAL_S:
            return
        self._last_phase_warn_at[phase] = now
        suffix = f" {detail()}" if detail is not None else ""
        logger.warning(
            f"slow pyexecutor phase: phase={phase} "
            f"duration_s={duration_s:.1f} budget_s={budget_s:.1f} "
            f"rank={self._rank}{suffix}"
        )

    def record_update_duration(
        self, update_s: float, n_requests: int = -1, n_guided: int = -1
    ) -> None:
        if n_requests < 0:
            self.record_phase("sample_update", update_s)
            return

        def detail() -> str:
            return f"batch_requests={n_requests} batch_guided={n_guided}"

        self.record_phase("sample_update", update_s, detail)

    def _scan_requests_and_log(self, active_requests, waiting_len: int, now: float) -> None:
        """Log requests that left a state past its dwell budget or sat in
        one state past the stuck threshold; refresh heartbeat state counts."""
        seen = set()
        state_counts: Dict[str, int] = {}
        for request in active_requests:
            request_id = request.py_request_id
            seen.add(request_id)
            state = _state_name(request)
            state_counts[state] = state_counts.get(state, 0) + 1
            track = self._request_state_data.get(request_id)
            if track is None:
                self._request_state_data[request_id] = _RequestStateEntryData(
                    state=state, entered_at=now, ext_id=_ext_id(request)
                )
            elif track.state != state:
                dwell_s = now - track.entered_at
                budget_s = DWELL_BUDGETS_S.get(track.state)
                if self._should_log_request_lines and budget_s is not None and dwell_s > budget_s:
                    logger.info(
                        f"request_state_dwell request_id={request_id} "
                        f"ext_request_id={track.ext_id} rank={self._rank} "
                        f"state={track.state} dwell_s={dwell_s:.1f} "
                        f"next={state}"
                    )
                track.state = state
                track.entered_at = now
            elif (
                self._should_log_request_lines
                and not track.is_stuck_logged
                and state not in STUCK_EXEMPT_STATES
                and now - track.entered_at > STUCK_THRESHOLD_S
            ):
                track.is_stuck_logged = True
                logger.info(
                    f"request_state_stuck request_id={request_id} "
                    f"ext_request_id={track.ext_id} rank={self._rank} "
                    f"state={state} stuck_s={now - track.entered_at:.1f}"
                )

        for request_id in [r for r in self._request_state_data if r not in seen]:
            del self._request_state_data[request_id]
            self._departed += 1

        self._state_counts = state_counts
        self._waiting_len = waiting_len

    def _maybe_heartbeat(self, now: float) -> None:
        if self._last_heartbeat_at is None:
            self._last_heartbeat_at = now
            return
        if now - self._last_heartbeat_at < HEARTBEAT_INTERVAL_S:
            return
        mean_ms = (self._iter_s_sum / self._iterations * 1000.0) if self._iterations else 0.0
        states = ",".join(f"{name}:{count}" for name, count in sorted(self._state_counts.items()))
        phases = ",".join(
            f"{name}:{seconds * 1000.0:.0f}" for name, seconds in sorted(self._phase_s_max.items())
        )
        logger.info(
            f"pyexecutor_heartbeat rank={self._rank} "
            f"iters={self._iterations} iter_ms_mean={mean_ms:.1f} "
            f"iter_ms_max={self._iter_s_max * 1000.0:.1f} "
            f'phase_ms_max="{phases}" '
            f"active={sum(self._state_counts.values())} "
            f"waiting={self._waiting_len} "
            f"departed={self._departed} "
            f'states="{states}"'
        )
        self._last_heartbeat_at = now
        self._iterations = 0
        self._iter_s_sum = 0.0
        self._iter_s_max = 0.0
        self._phase_s_max = {}
        self._departed = 0
