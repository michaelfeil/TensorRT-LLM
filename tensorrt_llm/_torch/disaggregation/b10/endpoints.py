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
"""Persistent send-endpoint slots and their generation lifecycle.

Peer descriptors and slots live together so a descriptor change can retire
the peer's endpoints atomically. One slot serializes a complete WRITE and each
new endpoint advances its generation, isolating stale messages.
"""

from __future__ import annotations

from functools import partial
from typing import Any

from tensorrt_llm import logger
from tensorrt_llm._torch.disaggregation.b10.async_utils import (
    _abort_endpoint_background,
    _await_detached_with_timeout,
    _format_endpoint_handles,
    _TransferDeadline,
)
from tensorrt_llm._torch.disaggregation.b10.core import _AgentCore
from tensorrt_llm._torch.disaggregation.b10.protocol import (
    B10AgentDescriptor,
    _next_endpoint_generation,
)
from tensorrt_llm._torch.disaggregation.b10.state import (
    _EndpointSlot,
    _SendEndpointLease,
    _SendTransferPlan,
    _TransferAbortHandle,
)


class EndpointPool:
    def __init__(
        self,
        core: _AgentCore,
        *,
        ucxx: Any,
        endpoint_pool_size: int,
    ):
        self._core = core
        self._ucxx = ucxx
        self._endpoint_pool_size = endpoint_pool_size
        self._remote_agents: dict[str, B10AgentDescriptor] = {}
        self._remote_slots: dict[str, list[_EndpointSlot]] = {}

    async def _lease_send_endpoint(
        self,
        plan: _SendTransferPlan,
        slot_index: int,
        slot: _EndpointSlot,
        deadline: _TransferDeadline,
        abort_handle: _TransferAbortHandle,
    ) -> _SendEndpointLease:
        endpoint = await self._get_or_create_endpoint(slot, plan.remote, deadline)
        lease = _SendEndpointLease(
            remote_name=plan.remote_name,
            slot_index=slot_index,
            slot=slot,
            endpoint=endpoint,
            endpoint_generation=slot.generation,
        )
        self._bind_send_abort_handle(abort_handle, lease)
        return lease

    def _bind_send_abort_handle(
        self,
        abort_handle: _TransferAbortHandle,
        lease: _SendEndpointLease,
        *,
        abort_endpoint: bool = True,
    ) -> None:
        def retire_endpoint() -> None:
            self._core.loop.call_soon_threadsafe(
                self._retire_endpoint_slot,
                lease.remote_name,
                lease.slot_index,
                lease.slot,
                lease.endpoint,
            )

        abort_endpoint_callback = (
            partial(_abort_endpoint_background, lease.endpoint) if abort_endpoint else None
        )
        abort_handle.bind_endpoint(
            lease.endpoint,
            retire_endpoint,
            abort_endpoint_callback,
        )

    def _retire_send_endpoint(
        self,
        lease: _SendEndpointLease,
        abort_handle: _TransferAbortHandle,
        abort_endpoint: bool = True,
    ) -> None:
        self._retire_endpoint_slot(lease.remote_name, lease.slot_index, lease.slot, lease.endpoint)
        if abort_endpoint:
            _abort_endpoint_background(lease.endpoint)
        abort_handle.unbind_endpoint(lease.endpoint)

    async def _refresh_stale_send_endpoint(
        self,
        plan: _SendTransferPlan,
        lease: _SendEndpointLease,
        deadline: _TransferDeadline,
        abort_handle: _TransferAbortHandle,
    ) -> _SendEndpointLease:
        self._retire_send_endpoint(lease, abort_handle)
        return await self._lease_send_endpoint(
            plan, lease.slot_index, lease.slot, deadline, abort_handle
        )

    def _get_endpoint_slot(self, remote_name: str, transfer_id: int) -> tuple[int, _EndpointSlot]:
        slots = self._remote_slots.get(remote_name)
        if slots is None:
            slots = [_EndpointSlot() for _ in range(self._endpoint_pool_size)]
            self._remote_slots[remote_name] = slots
        slot_index = transfer_id % len(slots)
        return slot_index, slots[slot_index]

    async def _get_or_create_endpoint(
        self, slot: _EndpointSlot, remote: B10AgentDescriptor, deadline: _TransferDeadline
    ) -> Any:
        # Worker-address endpoints, not sockaddr: UCX failover (NIC-loss lane
        # reconfiguration) is rejected for any endpoint with a CM lane, which
        # every host:port/listener endpoint carries. The peer's worker address
        # blob rides its descriptor.
        if slot.endpoint is None:
            if not remote.worker_address:
                raise RuntimeError(
                    f"B10 remote agent {remote.name} descriptor has no UCX "
                    "worker address (peer running a pre-AM build?)"
                )
            slot.generation = _next_endpoint_generation(slot.generation)
            address = self._ucxx.get_ucx_address_from_buffer(remote.worker_address)
            slot.endpoint = await _await_detached_with_timeout(
                self._ucxx.create_endpoint_from_worker_address(address),
                deadline.remaining_s(),
                on_late_result=_abort_endpoint_background,
            )
            # The only place the ucp_ep_h is knowable together with the peer it
            # points at. UCX's own endpoint-level diagnostics (keepalive
            # failures, failover reconfiguration) identify an endpoint by that
            # pointer alone, so without this line they cannot be attributed to
            # a peer.
            logger.info(
                f"B10 send endpoint created: remote={remote.name} "
                f"remote_host={remote.host}:{remote.port} "
                f"slot_generation={slot.generation} "
                f"{_format_endpoint_handles(slot.endpoint)}"
            )
        return slot.endpoint

    def endpoint_inventory(self) -> list[str]:
        """One `peer -> ucp_ep` line per live send endpoint."""
        lines = []
        for remote_name, slots in self._remote_slots.items():
            remote = self._remote_agents.get(remote_name)
            host = f"{remote.host}:{remote.port}" if remote is not None else "unregistered"
            for slot_index, slot in enumerate(slots):
                if slot.endpoint is not None:
                    lines.append(
                        f"remote={remote_name} remote_host={host} slot={slot_index} "
                        f"generation={slot.generation} "
                        f"{_format_endpoint_handles(slot.endpoint)}"
                    )
        return lines

    def _retire_endpoint_slot(
        self, remote_name: str, slot_index: int, slot: _EndpointSlot, endpoint: Any
    ) -> None:
        slots = self._remote_slots.get(remote_name)
        if (
            slots is not None
            and slot_index < len(slots)
            and slots[slot_index] is slot
            and slot.endpoint is endpoint
        ):
            slot.endpoint = None

    async def _clear_remote_slot_endpoints(self, name: str) -> None:
        slots = self._remote_slots.get(name, [])
        for slot in slots:
            if slot.endpoint is not None:
                _abort_endpoint_background(slot.endpoint)
                slot.endpoint = None

    async def _drop_remote_slots(self, name: str) -> None:
        await self._clear_remote_slot_endpoints(name)
        self._remote_slots.pop(name, None)
