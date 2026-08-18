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
from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Optional

from tensorrt_llm import logger

_DEFAULT_TRANSFER_TIMEOUT_S = 60.0
_STATUS_WAIT_CLEANUP_GRACE_S = 1.0
_UCXX_PROGRESS_MODE_ENV = "UCXPY_PROGRESS_MODE"
_UCXX_PYTHON_FUTURE_ENV = "UCXPY_ENABLE_PYTHON_FUTURE"
_DEFAULT_UCXX_PROGRESS_MODE = "thread-polling"
_UCXX_ERROR_HANDLING_MODE_ENV = "UCXX_ERROR_HANDLING_MODE"
_DEFAULT_UCXX_ERROR_HANDLING_MODE = "failover"
_UCX_MAX_EAGER_RAILS_ENV = "UCX_MAX_EAGER_RAILS"
_DEFAULT_UCX_MAX_EAGER_RAILS = "2"
_UCX_RECOVERY_RETRIES_ENV = "UCX_RECOVERY_RETRIES"
_DEFAULT_UCX_RECOVERY_RETRIES = "5"


def _apply_default_ucx_env_vars() -> None:
    """Default the UCX/UCXX environment B10 depends on, before ucxx imports.

    Every value here is a floor rather than a policy: each is applied with
    setdefault, so a deployment that sets one keeps it. They cover the
    progress thread, Python futures, and the two fault-tolerance settings
    whose stock values do not suit a wire protocol built entirely on active
    messages.

    The v1 C++ UCX transceiver runs startProgressThread(pollingMode=true);
    ucxx's Python default is the interrupt-driven "thread" mode, which adds
    an epoll wake and a scheduler round trip to every rendezvous state
    transition on both sides of a transfer. Match v1's polling behavior;
    deployments can still override via UCXPY_PROGRESS_MODE.

    Python futures MUST also be enabled: ucxx defaults them off, and without
    them every `await request.wait()` falls back to wait_yield() — a
    sleep(0) hot spin on the agent's event loop. Because an idle endpoint
    always has a bootstrap control recv posted, that spin runs continuously,
    burning a core and starving the executor thread of the GIL on every
    rank (observed as ucxx recv dominating py-spy on decode). With futures
    on, the progress thread's notifier completes awaits and the loop sleeps
    when idle.
    """
    os.environ.setdefault(_UCXX_PROGRESS_MODE_ENV, _DEFAULT_UCXX_PROGRESS_MODE)
    # NOTE: enabling Python futures is only safe because the agent binds
    # (or rebinds) ucxx's future notifier to its private event loop before
    # creating the listener — see _bind_ucxx_python_future_notifier below.
    # ucxx binds the notifier and its pre-created future pool to the event
    # loop captured when its process-wide context is first created, and a
    # context first touched off the agent loop leaves every request future
    # attached to a foreign loop (observed fleet-wide as listener-handler
    # "Future attached to a different loop" failures).
    os.environ.setdefault(_UCXX_PYTHON_FUTURE_ENV, "1")
    # NIC fault-tolerance on by default: ucxx creates worker-address
    # endpoints with UCP_ERR_HANDLING_MODE_FAILOVER, so a NIC/lane failure
    # mid-transfer is transparently rerouted onto surviving rails instead of
    # failing the endpoint. Measured cost vs peer mode is ~2% at the current
    # stack ceiling. Both peers must agree only in the sense that each side's
    # own endpoints are failover-capable; deployments can force the previous
    # behavior with UCXX_ERROR_HANDLING_MODE=peer. Fragment tuning
    # (UCX_RC_MLX5_SEG_SIZE) is hardware-specific and intentionally left to
    # deployment config.
    os.environ.setdefault(_UCXX_ERROR_HANDLING_MODE_ENV, _DEFAULT_UCXX_ERROR_HANDLING_MODE)
    # Two eager rails, because B10 puts its entire wire protocol - control,
    # READY, DATA, RESULT - on active messages, and UCX stripes eager traffic
    # over MAX_EAGER_RAILS lanes with a default of 1. With a single AM lane,
    # losing that lane's NIC leaves reconfiguration nothing to move to and it
    # bails out ("AM lane not found after reconfiguration"), so the endpoint
    # fails rather than failing over - the exact case the failover mode above
    # exists to survive. Two is the minimum that makes an AM-lane death
    # recoverable. This reads like the striping knob it shares a name with,
    # but for the AM plane it is a fault-tolerance requirement, so it belongs
    # with the code that depends on it rather than in deployment config.
    os.environ.setdefault(_UCX_MAX_EAGER_RAILS_ENV, _DEFAULT_UCX_MAX_EAGER_RAILS)
    # Recovery rounds have to be finite. UCX runs one round per keepalive
    # interval and, once they are exhausted, either declares an endpoint with
    # no live lanes dead or gives up on the failed lanes and lets normal
    # keepalive resume - and a recovering endpoint has its keepalive
    # suppressed until then. The upstream default of "inf" reaches neither
    # outcome, and cannot: rebuilding a failed lane is still an unimplemented
    # stub upstream (ucp_ep_recovery_prepare_lanes returns 0 unconditionally),
    # so every round is futile by construction. Left at "inf" a single lane
    # failure parks the endpoint in a permanent recovery loop with no liveness
    # detection.
    #
    # Five rounds is ~100s at the 20s default keepalive interval. The bias is
    # deliberately toward patience: giving up costs the whole endpoint, which
    # b10 then has to rebuild while every transfer riding it dies, whereas
    # waiting only defers that. Individual transfers do not depend on this
    # window - they carry their own deadline and stop on their own - so a
    # longer one buys a NIC flap or a switch reconvergence the chance to pass
    # without taking the endpoint with it. The cost is that peer-death
    # detection stays suppressed for that window, which is the reason this is
    # a bounded number at all rather than "inf".
    os.environ.setdefault(_UCX_RECOVERY_RETRIES_ENV, _DEFAULT_UCX_RECOVERY_RETRIES)


def _bind_ucxx_python_future_notifier(ucxx_module: Any) -> None:
    """Bind ucxx's Python-future notifier to the running event loop.

    Must be called from a coroutine running on the agent's private event
    loop, before the agent creates its first listener or endpoint.

    ucxx binds its request-completion notifier — and the pool of asyncio
    futures it pre-creates for requests — to the event loop captured when
    its process-wide ApplicationContext is first created. ucxx's
    get_event_loop() silently creates a brand-new, never-running loop when
    the creating thread has no running loop, so a context first touched
    off the agent loop leaves every subsequent request future attached to
    a foreign loop; every await then fails with "RuntimeError: Task ...
    got Future ... attached to a different loop" (the fleet-wide
    listener-handler failure seen when futures were first enabled).

    Touching the context here pins the notifier to the agent loop. If some
    earlier code already created the context, restart the notifier from
    this loop and drop the wrongly bound pool futures. This is safe at
    agent startup: B10 is the only ucxx user in the process and no
    transfer can be in flight before the listener exists.
    """
    core = getattr(ucxx_module, "core", None)
    get_ctx = getattr(core, "_get_ctx", None)
    if get_ctx is None:
        logger.warning(
            "B10 cannot access ucxx.core._get_ctx; skipping UCXX "
            "Python-future notifier binding check"
        )
        return
    preexisting = getattr(core, "_ctx", None) is not None
    ctx = get_ctx()
    if not getattr(ctx.worker, "enable_python_future", False):
        logger.info(
            "B10 UCXX Python futures disabled; request awaits will busy-spin via wait_yield()"
        )
        return
    loop_id = id(asyncio.get_running_loop())
    if not preexisting:
        logger.info(
            f"B10 created the UCXX application context on the agent event "
            f"loop (id={loop_id}); Python-future notifier bound correctly"
        )
        return
    logger.warning(
        f"B10 found a pre-existing UCXX application context; its "
        f"Python-future notifier may be bound to a foreign event loop. "
        f"Rebinding the notifier to the agent event loop (id={loop_id}) "
        f"and dropping pre-created request futures."
    )
    ctx.stop_notifier_thread()
    ctx.worker.clear_python_futures_pool()
    ctx.start_notifier_thread()


def local_process_identity(agent_name: str) -> str:
    """Human-joinable identity for this process, advertised to peers.

    UCX names endpoints only by pointer: a keepalive failure reports
    `worker 0x.. ep 0x.. lane[n]` and nothing about who is on the other end.
    The pointer is joinable to a peer only if something logs the mapping, and
    the peer is nameable only if it says who it is. This is the "says who it
    is" half - it rides every control message so a receiver can name a sender
    it has never registered.

    In Kubernetes HOSTNAME is the pod name, which is exactly the identifier
    log queries are already filtered by. NODE_NAME has no default and only
    appears if the deployment maps it through the downward API; it is omitted
    rather than guessed at when absent.
    """
    parts = [f"agent={agent_name}", f"pid={os.getpid()}"]
    pod = os.getenv("HOSTNAME")
    if pod:
        parts.append(f"pod={pod}")
    node = os.getenv("NODE_NAME")
    if node:
        parts.append(f"node={node}")
    return " ".join(parts)


def _timeout_ms(timeout_s: Optional[float]) -> Optional[int]:
    if timeout_s is None:
        return None
    return max(1, int(timeout_s * 1000.0))


def _status_wait_timeout_ms(timeout_s: Optional[float]) -> Optional[int]:
    if timeout_s is None:
        return None
    return _timeout_ms(timeout_s + _STATUS_WAIT_CLEANUP_GRACE_S)


def _timeout_from_env() -> Optional[float]:
    value = os.getenv("TRTLLM_B10_UCXX_TRANSFER_TIMEOUT_S")
    if value is None:
        return _DEFAULT_TRANSFER_TIMEOUT_S
    try:
        parsed = float(value)
    except ValueError:
        logger.warning(
            f"Invalid TRTLLM_B10_UCXX_TRANSFER_TIMEOUT_S={value}; using default "
            f"{_DEFAULT_TRANSFER_TIMEOUT_S}s"
        )
        return _DEFAULT_TRANSFER_TIMEOUT_S
    return None if parsed <= 0 else parsed


def _ucx_net_devices_from_env() -> list[str]:
    value = os.getenv("UCX_NET_DEVICES")
    if value is None:
        return []
    return [device.strip() for device in value.split(",") if device.strip()]


def _netdev_from_ucx_net_device(device: str) -> Optional[str]:
    if device == "all" or device.startswith("^"):
        return None

    device_name, _, port = device.partition(":")
    if not device_name:
        return None
    if os.path.isdir(f"/sys/class/net/{device_name}"):
        return device_name

    port = port or "1"
    ndev_path = f"/sys/class/infiniband/{device_name}/ports/{port}/gid_attrs/ndevs/0"
    try:
        with open(ndev_path, encoding="utf-8") as ndev_file:
            netdev = ndev_file.readline().strip()
    except OSError:
        return None
    if not netdev or not os.path.isdir(f"/sys/class/net/{netdev}"):
        return None
    return netdev


def _advertised_ifname_from_ucx_net_devices() -> Optional[str]:
    for device in _ucx_net_devices_from_env():
        netdev = _netdev_from_ucx_net_device(device)
        if netdev is not None:
            return netdev
    return None


# A NIC failure and a dead peer look identical from a transfer error: both end
# in a timeout or an endpoint error. They need different responses, and only the
# first one is what NIC failover covers - failover moves traffic to a surviving
# local NIC, and can do nothing about a peer that is gone. Classify by looking
# at the local RDMA devices: if one of ours is down, this is a NIC failure; if
# they are all healthy, the peer stopped answering.
_RDMA_HEALTH_CACHE_TTL_S = 2.0
_rdma_health_cache: tuple[float, str] = (0.0, "")


def _ib_port_dir(device: str, port: str) -> str:
    return f"/sys/class/infiniband/{device}/ports/{port}"


def _read_sysfs(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as sysfs_file:
            return sysfs_file.readline().strip()
    except OSError:
        return ""


def _ib_port_has_global_gid(port_dir: str) -> bool:
    """Whether the port still has a routable (non link-local) GID.

    Required in addition to the port state: on RoCE, removing the netdev IP
    deletes the GID while the port keeps reporting ACTIVE, so the port state
    alone would call a rail healthy when it can no longer address anyone. The
    GID *index* is not stable, so scan rather than probe a fixed slot.
    """
    try:
        gid_names = os.listdir(f"{port_dir}/gids")
    except OSError:
        return True  # cannot tell; do not claim the device is broken
    for gid_name in gid_names:
        gid = _read_sysfs(f"{port_dir}/gids/{gid_name}")
        if gid and not gid.startswith("fe80") and set(gid) != {"0", ":"}:
            return True
    return False


def _local_rdma_device_faults() -> list[str]:
    """Return one description per configured local device that is not usable."""
    faults = []
    for device in _ucx_net_devices_from_env():
        if device == "all" or device.startswith("^"):
            continue
        device_name, _, port = device.partition(":")
        port_dir = _ib_port_dir(device_name, port or "1")
        state = _read_sysfs(f"{port_dir}/state")
        if not state:
            continue  # not an RDMA device (e.g. a plain netdev), nothing to check
        if "ACTIVE" not in state:
            faults.append(f"{device_name}: state={state or 'unknown'}")
        elif not _ib_port_has_global_gid(port_dir):
            faults.append(f"{device_name}: no routable GID (address removed?)")
    return faults


def classify_transfer_failure_cause() -> str:
    """One-line cause classification for a transfer failure log.

    Cached briefly so a burst of failures does not re-read sysfs per transfer.
    """
    global _rdma_health_cache
    now = time.monotonic()
    cached_at, cached = _rdma_health_cache
    if cached and (now - cached_at) < _RDMA_HEALTH_CACHE_TTL_S:
        return cached

    faults = _local_rdma_device_faults()
    if faults:
        cause = "cause=LOCAL_NIC_DOWN [" + "; ".join(faults) + "] (NIC failover applies)"
    else:
        cause = (
            "cause=PEER_UNREACHABLE (all local RDMA devices healthy; peer "
            "process/pod or its NIC is gone - NIC failover does not cover this)"
        )
    _rdma_health_cache = (now, cause)
    return cause


def _augment_ucx_net_devices_for_sockaddr() -> None:
    devices = _ucx_net_devices_from_env()
    if not devices:
        return

    expanded_devices = []
    seen_devices = set()
    for device in devices:
        if device not in seen_devices:
            expanded_devices.append(device)
            seen_devices.add(device)
        netdev = _netdev_from_ucx_net_device(device)
        if netdev is not None and netdev not in seen_devices:
            expanded_devices.append(netdev)
            seen_devices.add(netdev)

    os.environ["UCX_NET_DEVICES"] = ",".join(expanded_devices)


def _load_ucxx_module() -> Any:
    # ucxx reads its env configuration at import time, so the defaults must
    # land before its first import; keep this as the single import site.
    _apply_default_ucx_env_vars()
    import ucxx

    return ucxx
