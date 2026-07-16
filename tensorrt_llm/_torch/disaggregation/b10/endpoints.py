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
"""Endpoint slot and generation lifecycle for the B10 UCXX transfer agent.

`EndpointPool` is the endpoint collaborator constructed by
`B10CacheTransferAgent.__init__`.

Constructor-injected:

- ``core`` (`_AgentCore`; uses ``core.loop`` for thread-safe slot retirement)
- ``ucxx`` (UCXX module used to create endpoints)
- ``tag_domain`` (local half of the pair tag domain)
- ``endpoint_pool_size`` (slots per remote)
- ``raise_if_cancel_requested`` (module-level send.py helper; injected
  reference)

Own state:

- ``_remote_agents`` (remote name -> B10AgentDescriptor). Lives here rather
  than on the core because peer descriptors and their endpoint slots must
  stay coherent: a descriptor change invalidates the remote's slots.
- ``_remote_slots`` (remote name -> list of _EndpointSlot; slot
  ``endpoint``/``generation`` fields are mutated through the entries)
"""

from __future__ import annotations

from typing import Any, Callable

from tensorrt_llm._torch.disaggregation.b10.async_utils import (
    _abort_endpoint_background,
    _await_with_timeout,
    _TransferDeadline,
)
from tensorrt_llm._torch.disaggregation.b10.core import _AgentCore
from tensorrt_llm._torch.disaggregation.b10.protocol import (
    B10AgentDescriptor,
    _next_endpoint_generation,
    _pair_tag_domain,
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
        tag_domain: int,
        endpoint_pool_size: int,
        raise_if_cancel_requested: Callable[[_TransferAbortHandle], None],
    ):
        self._core = core
        self._ucxx = ucxx
        self._tag_domain = tag_domain
        self._endpoint_pool_size = endpoint_pool_size
        self._raise_if_cancel_requested = raise_if_cancel_requested
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
            tag_domain=_pair_tag_domain(self._tag_domain, plan.remote.tag_domain, slot_index),
        )
        self._arm_send_abort_handle(abort_handle, lease)
        return lease

    def _arm_send_abort_handle(
        self,
        abort_handle: _TransferAbortHandle,
        lease: _SendEndpointLease,
        *,
        abort_endpoint: bool = True,
    ) -> None:
        abort_handle.set_endpoint(
            lease.endpoint,
            lambda: self._core.loop.call_soon_threadsafe(
                self._retire_endpoint_slot,
                lease.remote_name,
                lease.slot_index,
                lease.slot,
                lease.endpoint,
            ),
            lambda: _abort_endpoint_background(lease.endpoint) if abort_endpoint else None,
        )
        self._raise_if_cancel_requested(abort_handle)

    def _retire_send_endpoint(
        self,
        lease: _SendEndpointLease,
        abort_handle: _TransferAbortHandle,
        abort_endpoint: bool = True,
    ) -> None:
        self._retire_endpoint_slot(lease.remote_name, lease.slot_index, lease.slot, lease.endpoint)
        if abort_endpoint:
            _abort_endpoint_background(lease.endpoint)
        abort_handle.clear_endpoint(lease.endpoint)

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
        if slot.endpoint is None:
            slot.generation = _next_endpoint_generation(slot.generation)
            timeout_s = deadline.remaining_s()
            try:
                slot.endpoint = await self._ucxx.create_endpoint(
                    remote.host, remote.port, connect_timeout=timeout_s
                )
            except TypeError as exc:
                if "connect_timeout" not in str(exc):
                    raise
                slot.endpoint = await _await_with_timeout(
                    self._ucxx.create_endpoint(remote.host, remote.port),
                    timeout_s,
                    on_late_result=_abort_endpoint_background,
                    cancel_on_timeout=False,
                )
        return slot.endpoint

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
