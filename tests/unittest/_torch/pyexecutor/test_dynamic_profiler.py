"""Unit tests for the file-triggered dynamic torch profiler session."""

import gzip
import json
import os
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor import dynamic_profiler as dp
from tensorrt_llm._torch.pyexecutor.dynamic_profiler import (
    POLL_INTERVAL_ITERS,
    DynamicProfilerSession,
)


@pytest.fixture
def fake_clock(monkeypatch):
    """Replace time.monotonic as seen by dynamic_profiler with a settable clock."""
    clock = SimpleNamespace(value=1000.0)
    monkeypatch.setattr(dp, "time", SimpleNamespace(monotonic=lambda: clock.value))
    return clock


@pytest.fixture
def make_session(tmp_path, monkeypatch):
    """Build a session with a tiny per-iter size estimate so the disk
    precheck passes on any machine; extra env vars per test via kwargs.

    Teardown flushes any still-active session: the kineto profiler is
    process-global, so a leaked running profiler breaks every later test
    with "Profiler is already enabled on this thread".
    """
    sessions = []

    def _make(rank=0, **env):
        monkeypatch.setenv(dp.EST_BYTES_PER_ITER_ENV, "1")
        for key, value in env.items():
            monkeypatch.setenv(key, str(value))
        session = DynamicProfilerSession(str(tmp_path), rank)
        sessions.append(session)
        return session

    yield _make
    for session in sessions:
        session.shutdown(10**9)


def write_request(tmp_path, **fields):
    (tmp_path / "request.json").write_text(json.dumps(fields))


def read_marker(tmp_path, req_id, rank=0):
    return json.loads((tmp_path / f"done-{req_id}-rank{rank}.json").read_text())


def start_session(session, req_iter=POLL_INTERVAL_ITERS):
    """Step on a poll boundary and assert the profiler started."""
    session.step(req_iter)
    assert session.active
    return req_iter


def test_noop_without_request(make_session, tmp_path):
    session = make_session()
    for i in range(3 * POLL_INTERVAL_ITERS):
        session.step(i)
    assert not session.active
    assert list(tmp_path.iterdir()) == []


def test_starts_only_on_poll_boundary(make_session, tmp_path):
    session = make_session()
    write_request(tmp_path, id="t0", iterations=8)
    session.step(POLL_INTERVAL_ITERS - 1)
    assert not session.active
    session.step(POLL_INTERVAL_ITERS)
    assert session.active


def test_seconds_mode_lifecycle(make_session, tmp_path, fake_clock):
    session = make_session()
    write_request(tmp_path, id="t1", seconds=5)
    start = start_session(session)

    fake_clock.value += 4.9
    session.step(start + 1)
    assert session.active

    fake_clock.value += 0.2
    session.step(start + 2)
    assert not session.active

    marker = read_marker(tmp_path, "t1")
    assert marker["stopped_by"] == "requested"
    assert marker["rank"] == 0
    assert marker["wall_seconds"] == pytest.approx(5.1)
    trace_path = tmp_path / marker["trace_file"]
    assert trace_path.name == "trace-t1-rank0.json.gz"
    assert marker["size_bytes"] == os.path.getsize(trace_path)
    with gzip.open(trace_path, "rt") as f:
        assert isinstance(json.load(f), dict)  # valid gzipped chrome trace


def test_iterations_mode_lifecycle(make_session, tmp_path, fake_clock):
    session = make_session()
    write_request(tmp_path, id="t2", iterations=32)
    start = start_session(session)

    for i in range(start + 1, start + 32):
        session.step(i)
        assert session.active
    session.step(start + 32)
    assert not session.active

    marker = read_marker(tmp_path, "t2")
    assert marker["stopped_by"] == "requested"
    assert marker["iterations"] == 32
    assert marker["iter_start"] == start
    assert marker["iter_stop"] == start + 32
    assert (tmp_path / "trace-t2-rank0.json.gz").exists()


def test_max_iters_cap_in_seconds_mode(make_session, tmp_path, fake_clock):
    session = make_session(**{dp.MAX_ITERS_ENV: 8})
    write_request(tmp_path, id="t3", seconds=120)
    start = start_session(session)

    for i in range(start + 1, start + 8):
        session.step(i)
        assert session.active
    session.step(start + 8)
    assert not session.active

    marker = read_marker(tmp_path, "t3")
    assert marker["stopped_by"] == "max_iters"
    assert marker["iterations"] == 8


def test_max_seconds_rescue_in_iterations_mode(make_session, tmp_path, fake_clock):
    session = make_session(**{dp.MAX_SECONDS_ENV: 7})
    write_request(tmp_path, id="t4", iterations=999999)
    start = start_session(session)

    fake_clock.value += 8
    session.step(start + 1)
    assert not session.active

    marker = read_marker(tmp_path, "t4")
    assert marker["stopped_by"] == "max_seconds"
    assert marker["iterations"] == 1


def test_iterations_clamped_to_cap(make_session, tmp_path, fake_clock):
    session = make_session(**{dp.MAX_ITERS_ENV: 8})
    write_request(tmp_path, id="t5", iterations=10**6)
    start = start_session(session)

    for i in range(start + 1, start + 9):
        session.step(i)
    assert not session.active
    assert read_marker(tmp_path, "t5")["iterations"] == 8


def test_seconds_clamped_to_cap(make_session, tmp_path, fake_clock):
    session = make_session(**{dp.MAX_SECONDS_ENV: 5})
    write_request(tmp_path, id="t6", seconds=10000)
    start = start_session(session)

    fake_clock.value += 5.1
    session.step(start + 1)
    assert not session.active
    assert read_marker(tmp_path, "t6")["wall_seconds"] == pytest.approx(5.1)


def test_rank_filter_excluded(make_session, tmp_path):
    session = make_session(rank=0)
    write_request(tmp_path, id="t7", iterations=8, ranks=[3])
    for i in range(4 * POLL_INTERVAL_ITERS):
        session.step(i)
    assert not session.active
    assert not list(tmp_path.glob("done-*"))
    assert not list(tmp_path.glob("trace-*"))


def test_rank_filter_included(make_session, tmp_path):
    session = make_session(rank=3)
    write_request(tmp_path, id="t8", iterations=8, ranks=[3])
    session.step(POLL_INTERVAL_ITERS)
    assert session.active


def test_missing_window_writes_error_marker(make_session, tmp_path):
    session = make_session()
    write_request(tmp_path, id="t9")
    session.step(POLL_INTERVAL_ITERS)
    assert not session.active
    marker = read_marker(tmp_path, "t9")
    assert "seconds or iterations" in marker["error"]


def test_insufficient_disk_writes_error_marker(make_session, tmp_path, monkeypatch):
    session = make_session()
    monkeypatch.setattr(dp.os, "statvfs", lambda path: SimpleNamespace(f_bavail=0, f_frsize=1))
    write_request(tmp_path, id="t10", iterations=8)
    session.step(POLL_INTERVAL_ITERS)
    assert not session.active
    marker = read_marker(tmp_path, "t10")
    assert "disk" in marker["error"]


def test_same_id_not_retriggered(make_session, tmp_path, fake_clock):
    session = make_session()
    write_request(tmp_path, id="t11", iterations=4)
    start = start_session(session)
    for i in range(start + 1, start + 5):
        session.step(i)
    assert not session.active

    # request.json is still on disk with the same id
    for i in range(start + 5, start + 5 + 4 * POLL_INTERVAL_ITERS):
        session.step(i)
    assert not session.active
    assert len(list(tmp_path.glob("trace-*"))) == 1


def test_new_id_supersedes(make_session, tmp_path, fake_clock):
    session = make_session()
    write_request(tmp_path, id="t12a", iterations=4)
    start = start_session(session)
    for i in range(start + 1, start + 5):
        session.step(i)
    assert not session.active

    write_request(tmp_path, id="t12b", iterations=4)
    next_poll = ((start + 5) // POLL_INTERVAL_ITERS + 1) * POLL_INTERVAL_ITERS
    session.step(next_poll)
    assert session.active
    for i in range(next_poll + 1, next_poll + 5):
        session.step(i)
    assert (tmp_path / "trace-t12b-rank0.json.gz").exists()


def test_profiler_failure_disables_session(make_session, tmp_path, monkeypatch):
    session = make_session()

    def boom(*args, **kwargs):
        raise RuntimeError("kineto exploded")

    monkeypatch.setattr(dp.torch.profiler, "profile", boom)
    write_request(tmp_path, id="t13", iterations=8)
    session.step(POLL_INTERVAL_ITERS)  # must not raise
    assert not session.active

    # Session is disabled for good, even for a fresh request id.
    monkeypatch.undo()
    write_request(tmp_path, id="t13b", iterations=8)
    session.step(2 * POLL_INTERVAL_ITERS)
    assert not session.active
    assert not list(tmp_path.glob("trace-*"))


def test_shutdown_flushes_in_flight_session(make_session, tmp_path, fake_clock):
    session = make_session()
    write_request(tmp_path, id="t14", iterations=100)
    start = start_session(session)
    for i in range(start + 1, start + 5):
        session.step(i)
    assert session.active

    session.shutdown(start + 4)
    assert not session.active
    marker = read_marker(tmp_path, "t14")
    assert marker["stopped_by"] == "executor_shutdown"
    assert marker["iterations"] == 4
    assert (tmp_path / "trace-t14-rank0.json.gz").exists()


def test_unsafe_id_ignored(make_session, tmp_path):
    session = make_session()
    write_request(tmp_path, id="../evil", iterations=8)
    for i in range(3 * POLL_INTERVAL_ITERS):
        session.step(i)
    assert not session.active
    assert not list(tmp_path.glob("done-*")) and not list(tmp_path.glob("trace-*"))
    # a later well-formed request still works
    write_request(tmp_path, id="safe1", iterations=8)
    session.step(4 * POLL_INTERVAL_ITERS)
    assert session.active
