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

import struct
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import msgpack
import numpy as np

from tensorrt_llm._torch.disaggregation.b10 import async_utils as b10_async_utils

_B10_PROTOCOL = "b10-ucxx"
_B10_PROTOCOL_VERSION = 1

# Feature flags advertised in B10AgentDescriptor.features. Compatibility
# rides on additive descriptor keys (old peers ignore unknown keys); the
# protocol version stays fixed because from_bytes hard-rejects mismatches.
# packed_descs is required, not negotiated: the packed int64 descriptor
# encoding is the only wire format, senders refuse peers that do not
# advertise it, and receivers reject controls without `dst_descs_packed`.
_FEATURE_PACKED_DESCS = "packed_descs"

_CHUNK_INDEX_BITS = 16
_TRANSFER_ID_BITS = 32
# Full width of the header's endpoint_generation field. It was 12 bits back when
# transfer identity had to share a single 64-bit UCX tag with the chunk index and
# transfer id; identity now rides the 32-byte AM header, where the field is a
# whole uint32, so the counter uses all of it. Receivers compare generations to
# order attempts, and a narrow counter makes that ordering ambiguous once it
# wraps - at 12 bits, only 4095 endpoint rebuilds on one slot.
_ENDPOINT_GENERATION_BITS = 32
_MAX_CHUNK_INDEX = (1 << _CHUNK_INDEX_BITS) - 1
_MAX_TRANSFER_ID = (1 << _TRANSFER_ID_BITS) - 1
_MAX_ENDPOINT_GENERATION = (1 << _ENDPOINT_GENERATION_BITS) - 1
# Modulus for comparing two endpoint generations. Wrapping is now unreachable in
# practice, but ordering by forward distance costs nothing and keeps the
# comparison correct by construction rather than by argument.
_ENDPOINT_GENERATION_RING = 1 << _ENDPOINT_GENERATION_BITS

# Named for the TRTLLM_B10_UCXX_TAG_* env knobs they back (kept stable for
# deployments); they now size the transfer-id ring and its quarantine TTL.
_DEFAULT_TAG_SPACE_SIZE = 1 << _TRANSFER_ID_BITS
_DEFAULT_TAG_QUARANTINE_TTL_S = 120.0

# ---- AM wire plane -------------------------------------------------------
#
# B10's wire protocol rides UCX active messages instead of tag send/recv so
# transfers use UCX's failover-capable AM protocol and transparently survive
# a NIC failure mid-transfer (tag has no failover-eligible protocol). AM
# carries no tag, so per-message identity moves into a fixed 32-byte in-band
# header prepended to every message. Endpoint scoping replaces tag quarantine for staleness defense:
# one endpoint carries at most one transfer at a time (slot lock), a retired
# endpoint's message stream dies with it, and a header mismatch on a live
# endpoint is dropped loudly.
#
# The plane also requires exchanging UCX worker addresses: endpoints are
# created from the peer's worker address (carried in B10AgentDescriptor),
# because UCX failover reconfiguration is only supported on worker-address
# endpoints, not sockaddr/CM ones. A peer that does not advertise a worker
# address (a pre-AM build) is therefore incompatible and is rejected rather
# than silently downgraded.
#
# All B10 active messages are sent with receiver callback info
# (_AM_RECEIVER_OWNER, _AM_RECEIVER_ID) so they route to the worker-scoped
# callback registered at agent startup (auto-re-arming; never matched by
# `am_recv()`).

_AM_RECEIVER_OWNER = "b10"
_AM_RECEIVER_ID = 0
_AM_RECEIVER_CALLBACK_INFO = (_AM_RECEIVER_OWNER, _AM_RECEIVER_ID)

_AM_KIND_CONTROL = 0
_AM_KIND_READY = 1
_AM_KIND_RESULT = 2
_AM_KIND_DATA = 3
_AM_KIND_NAMES = {
    _AM_KIND_CONTROL: "control",
    _AM_KIND_READY: "READY",
    _AM_KIND_RESULT: "RESULT",
    _AM_KIND_DATA: "DATA",
}

_AM_HEADER_MAGIC = b"B10A"
_AM_HEADER_VERSION = 1
# magic(4) | version(1) | kind(1) | pad(2) | transfer_id(8) | chunk_index(4)
# | endpoint_generation(4) | payload_len(8) = 32 bytes
_AM_HEADER_STRUCT = struct.Struct("<4sBBHQIIQ")
_AM_HEADER_SIZE = _AM_HEADER_STRUCT.size
assert _AM_HEADER_SIZE == 32


@dataclass(frozen=True)
class _AmHeader:
    kind: int
    transfer_id: int
    chunk_index: int
    endpoint_generation: int
    payload_len: int

    @property
    def kind_name(self) -> str:
        return _AM_KIND_NAMES.get(self.kind, f"kind{self.kind}")


def _pack_am_header(
    kind: int,
    transfer_id: int,
    chunk_index: int,
    endpoint_generation: int,
    payload_len: int,
) -> bytes:
    return _AM_HEADER_STRUCT.pack(
        _AM_HEADER_MAGIC,
        _AM_HEADER_VERSION,
        kind,
        0,
        transfer_id,
        chunk_index,
        endpoint_generation,
        payload_len,
    )


def _unpack_am_header(buf: Any) -> _AmHeader:
    """Parse the 32-byte header at the start of `buf` (any buffer protocol
    object of at least `_AM_HEADER_SIZE` bytes)."""
    magic, version, kind, _pad, transfer_id, chunk_index, endpoint_generation, payload_len = (
        _AM_HEADER_STRUCT.unpack_from(buf, 0)
    )
    if magic != _AM_HEADER_MAGIC:
        raise ValueError(f"B10 AM header bad magic: {magic!r}")
    if version != _AM_HEADER_VERSION:
        raise ValueError(f"B10 AM header unexpected version: {version}")
    return _AmHeader(
        kind=kind,
        transfer_id=transfer_id,
        chunk_index=chunk_index,
        endpoint_generation=endpoint_generation,
        payload_len=payload_len,
    )


def _pack_am_message(
    kind: int, transfer_id: int, endpoint_generation: int, payload: dict[str, Any]
) -> bytes:
    """One control-plane AM message: 32-byte header + msgpack payload."""
    body = _pack_message(payload)
    return _pack_am_header(kind, transfer_id, 0, endpoint_generation, len(body)) + body


async def _am_send_message(
    endpoint: Any,
    kind: int,
    transfer_id: int,
    endpoint_generation: int,
    payload: dict[str, Any],
    timeout_s: Optional[float],
) -> None:
    """Send one header+msgpack control-plane message via AM."""
    buf = np.frombuffer(
        _pack_am_message(kind, transfer_id, endpoint_generation, payload), dtype=np.uint8
    )
    await b10_async_utils._await_detached_with_timeout(
        endpoint.am_send(buf, receiver_callback_info=_AM_RECEIVER_CALLBACK_INFO),
        timeout_s,
    )


def _request_id_from_sync_message(sync_message: Optional[str]) -> Optional[int]:
    if not sync_message:
        return None
    try:
        return int(sync_message)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class B10AgentDescriptor:
    name: str
    host: str
    port: int
    features: tuple[str, ...] = ()
    # UCX worker address blob for the AM plane. Endpoints are created from
    # this (worker-address endpoints are the only kind UCX failover
    # supports); host/port remain for registration-plane identity only.
    worker_address: bytes = b""

    def to_bytes(self) -> bytes:
        return msgpack.packb(
            {
                "protocol": _B10_PROTOCOL,
                "version": _B10_PROTOCOL_VERSION,
                "name": self.name,
                "host": self.host,
                "port": self.port,
                "features": list(self.features),
                "worker_address": self.worker_address,
            },
            use_bin_type=True,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "B10AgentDescriptor":
        payload = msgpack.unpackb(data, raw=False)
        if payload.get("protocol") != _B10_PROTOCOL:
            raise ValueError(f"Unexpected B10 descriptor protocol: {payload.get('protocol')}")
        if payload.get("version") != _B10_PROTOCOL_VERSION:
            raise ValueError(f"Unexpected B10 descriptor version: {payload.get('version')}")
        return cls(
            name=payload["name"],
            host=payload["host"],
            port=int(payload["port"]),
            features=tuple(payload.get("features", ())),
            worker_address=payload.get("worker_address", b""),
        )


class B10TransferIdAllocator:
    """Allocates transfer ids from a fixed ring, with failure quarantine.

    Transfer ids identify every AM message via the in-band header, so an id
    must not be reused while a peer could still emit messages carrying it.
    `release` returns an id to the ring on clean completion; `quarantine`
    instead parks it for `quarantine_ttl_s` after a failure or timeout,
    because a wedged or slow peer may still send DATA/replies stamped with
    that id — early reuse could route a stale message into a fresh
    transfer (belt to the endpoint-generation scoping's suspenders).
    Quarantined ids re-enter circulation lazily on `allocate` once their
    TTL expires.
    """

    def __init__(
        self,
        start: int = 1,
        tag_space_size: int = _DEFAULT_TAG_SPACE_SIZE,
        quarantine_ttl_s: float = _DEFAULT_TAG_QUARANTINE_TTL_S,
    ):
        if tag_space_size <= 0:
            raise ValueError("tag_space_size must be positive")
        if tag_space_size > _MAX_TRANSFER_ID + 1:
            raise ValueError(f"tag_space_size must be in [1, {_MAX_TRANSFER_ID + 1}]")
        self._next = start % tag_space_size
        self._tag_space_size = tag_space_size
        self._quarantine_ttl_s = quarantine_ttl_s
        self._active: set[int] = set()
        self._quarantined_until: dict[int, float] = {}
        self._lock = threading.Lock()

    def allocate(self) -> int:
        with self._lock:
            self._expire_quarantine_locked()
            for _ in range(self._tag_space_size):
                transfer_id = self._next
                self._next = (self._next + 1) % self._tag_space_size
                if transfer_id not in self._active and transfer_id not in self._quarantined_until:
                    self._active.add(transfer_id)
                    return transfer_id
            raise RuntimeError("No B10 transfer tags available")

    def release(self, transfer_id: int) -> None:
        with self._lock:
            self._active.discard(transfer_id)

    def quarantine(self, transfer_id: int) -> None:
        with self._lock:
            self._active.discard(transfer_id)
            self._quarantined_until[transfer_id] = time.monotonic() + self._quarantine_ttl_s

    def _expire_quarantine_locked(self) -> None:
        now = time.monotonic()
        expired = [
            transfer_id for transfer_id, expiry in self._quarantined_until.items() if expiry <= now
        ]
        for transfer_id in expired:
            self._quarantined_until.pop(transfer_id, None)


def _next_endpoint_generation(generation: int) -> int:
    """Advance a slot's generation, keeping 0 reserved for "never leased"."""
    generation = (generation + 1) & _MAX_ENDPOINT_GENERATION
    return 1 if generation == 0 else generation


def _pack_message(payload: dict[str, Any]) -> bytes:
    return msgpack.packb(payload, use_bin_type=True)


def _unpack_message(payload: bytes) -> dict[str, Any]:
    return msgpack.unpackb(payload, raw=False)


# The tag-plane wire helpers (_send_obj/_recv_obj/_send_reply/_recv_reply)
# were removed with the move to the AM plane; control-plane messages now
# flow through _am_send_message and the B10AmDispatcher.
