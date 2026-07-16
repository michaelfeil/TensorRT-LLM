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
from typing import Any, Optional

from tensorrt_llm import logger

_DEFAULT_TRANSFER_TIMEOUT_S = 60.0
_STATUS_WAIT_CLEANUP_GRACE_S = 1.0
_UCXX_PROGRESS_MODE_ENV = "UCXPY_PROGRESS_MODE"
_UCXX_PYTHON_FUTURE_ENV = "UCXPY_ENABLE_PYTHON_FUTURE"
_DEFAULT_UCXX_PROGRESS_MODE = "thread-polling"


def _apply_default_ucxx_progress_mode() -> None:
    """Default the UCXX progress thread to busy-polling with Python futures.

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
    _apply_default_ucxx_progress_mode()
    import ucxx

    return ucxx
