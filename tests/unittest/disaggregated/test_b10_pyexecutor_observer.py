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
"""B10PyExecutorObserver: silent for healthy traffic, loud exactly when stuck."""

from types import SimpleNamespace

import pytest

import tensorrt_llm._torch.pyexecutor.b10_pyexecutor_observer as observer_module
from tensorrt_llm._torch.pyexecutor.b10_pyexecutor_observer import (
    STUCK_THRESHOLD_S,
    B10PyExecutorObserver,
)


@pytest.fixture
def fake_clock(monkeypatch):
    clock = SimpleNamespace(value=1_000.0)
    monkeypatch.setattr(observer_module, "time", SimpleNamespace(monotonic=lambda: clock.value))
    return clock


@pytest.fixture
def logged(monkeypatch):
    lines = SimpleNamespace(info=[], warning=[])
    monkeypatch.setattr(
        observer_module,
        "logger",
        SimpleNamespace(info=lines.info.append, warning=lines.warning.append),
    )
    return lines


def make_request(request_id: int, state_name: str, ext_id: str = "b10-abc") -> SimpleNamespace:
    return SimpleNamespace(
        py_request_id=request_id,
        state=SimpleNamespace(name=state_name),
        py_external_request_id=ext_id,
    )


def make_observer(**kwargs) -> B10PyExecutorObserver:
    kwargs.setdefault("rank", 0)
    kwargs.setdefault("should_log_request_lines", True)
    return B10PyExecutorObserver(**kwargs)


def test_healthy_request_logs_nothing(fake_clock, logged):
    observer = make_observer()
    request = make_request(1, "DISAGG_GENERATION_INIT")

    observer.on_iteration_start([request], 0)
    fake_clock.value += 1.5
    request.state = SimpleNamespace(name="GENERATION_IN_PROGRESS")
    observer.on_iteration_start([request], 0)
    fake_clock.value += 2.0
    observer.on_iteration_start([], 0)

    assert not logged.info
    assert not logged.warning


def test_slow_state_exit_logs_dwell_line(fake_clock, logged):
    observer = make_observer()
    request = make_request(2, "DISAGG_GENERATION_INIT")

    observer.on_iteration_start([request], 0)
    fake_clock.value += 12.0
    request.state = SimpleNamespace(name="DISAGG_GENERATION_TRANS_IN_PROGRESS")
    observer.on_iteration_start([request], 0)

    dwell_lines = [line for line in logged.info if "request_state_dwell" in line]
    assert len(dwell_lines) == 1
    assert "state=DISAGG_GENERATION_INIT" in dwell_lines[0]
    assert "next=DISAGG_GENERATION_TRANS_IN_PROGRESS" in dwell_lines[0]
    assert "ext_request_id=b10-abc" in dwell_lines[0]


def test_stuck_request_logs_once(fake_clock, logged):
    observer = make_observer()
    request = make_request(3, "DISAGG_GENERATION_INIT")

    observer.on_iteration_start([request], 0)
    for _ in range(4):
        fake_clock.value += STUCK_THRESHOLD_S / 2
        observer.on_iteration_start([request], 0)

    stuck_lines = [line for line in logged.info if "request_state_stuck" in line]
    assert len(stuck_lines) == 1
    assert "state=DISAGG_GENERATION_INIT" in stuck_lines[0]


def test_stuck_logs_at_most_once_per_request(fake_clock, logged):
    observer = make_observer()
    request = make_request(4, "DISAGG_GENERATION_INIT")

    observer.on_iteration_start([request], 0)
    fake_clock.value += STUCK_THRESHOLD_S + 1
    observer.on_iteration_start([request], 0)
    request.state = SimpleNamespace(name="DISAGG_GENERATION_TRANS_IN_PROGRESS")
    fake_clock.value += 1.5
    observer.on_iteration_start([request], 0)
    fake_clock.value += STUCK_THRESHOLD_S + 1
    observer.on_iteration_start([request], 0)

    stuck_lines = [line for line in logged.info if "request_state_stuck" in line]
    assert len(stuck_lines) == 1
    assert "state=DISAGG_GENERATION_INIT" in stuck_lines[0]


def test_token_generation_is_never_stuck_or_dwell(fake_clock, logged):
    observer = make_observer()
    request = make_request(5, "GENERATION_IN_PROGRESS")

    observer.on_iteration_start([request], 0)
    for _ in range(10):
        fake_clock.value += 3_600.0
        observer.on_iteration_start([request], 0)

    heartbeat_free = [line for line in logged.info if "pyexecutor_heartbeat" not in line]
    assert not heartbeat_free


def test_request_lines_suppressed_when_disabled(fake_clock, logged):
    observer = make_observer(rank=3, should_log_request_lines=False)
    request = make_request(6, "DISAGG_GENERATION_INIT")

    observer.on_iteration_start([request], 0)
    fake_clock.value += STUCK_THRESHOLD_S + 1
    observer.on_iteration_start([request], 0)

    assert not logged.info


def test_heartbeat_fires_once_per_interval(fake_clock, logged):
    observer = make_observer()
    request = make_request(8, "GENERATION_IN_PROGRESS")

    for _ in range(10):
        observer.on_iteration_start([request], 2)
        fake_clock.value += 30.0

    heartbeats = [line for line in logged.info if "pyexecutor_heartbeat" in line]
    # 300s of iterations after the first observation -> 2 heartbeats.
    assert len(heartbeats) == 2
    assert "waiting=2" in heartbeats[0]
    assert "GENERATION_IN_PROGRESS:1" in heartbeats[0]
    assert "iter_ms_mean=30000.0" in heartbeats[0]


def test_slow_update_warns_with_rate_limit(fake_clock, logged):
    observer = make_observer()

    observer.record_update_duration(0.02)
    assert not logged.warning

    observer.record_update_duration(9.0, n_requests=17, n_guided=3)
    observer.record_update_duration(9.0, n_requests=17, n_guided=3)
    assert len(logged.warning) == 1
    assert "batch_guided=3" in logged.warning[0]

    fake_clock.value += 61.0
    observer.record_update_duration(9.0, n_requests=17, n_guided=3)
    assert len(logged.warning) == 2


def test_slow_phase_warns_with_rate_limit(fake_clock, logged):
    observer = make_observer()

    observer.record_phase("schedule", 0.2)
    assert not logged.warning

    observer.record_phase("schedule", 3.0)
    observer.record_phase("schedule", 3.0)
    assert len(logged.warning) == 1
    assert "phase=schedule" in logged.warning[0]

    fake_clock.value += 61.0
    observer.record_phase("schedule", 3.0)
    assert len(logged.warning) == 2


def test_env_var_disables_observer(monkeypatch):
    monkeypatch.setenv(observer_module.OBSERVER_ENV_VAR, "0")
    assert (
        observer_module.maybe_create_b10_pyexecutor_observer(rank=0, should_log_request_lines=True)
        is None
    )


def test_observer_enabled_by_default(monkeypatch):
    monkeypatch.delenv(observer_module.OBSERVER_ENV_VAR, raising=False)
    observer = observer_module.maybe_create_b10_pyexecutor_observer(
        rank=0, should_log_request_lines=True
    )
    assert isinstance(observer, B10PyExecutorObserver)
