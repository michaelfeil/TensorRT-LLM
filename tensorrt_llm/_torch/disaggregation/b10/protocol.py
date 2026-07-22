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
from typing import Any, Hashable, Iterable, Optional

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

_BOOTSTRAP_CONTROL_TAG = 0
_CHUNK_INDEX_BITS = 16
_TRANSFER_ID_BITS = 32
_ENDPOINT_GENERATION_BITS = 12
_TAG_MASK = (1 << 64) - 1
_MAX_CHUNK_INDEX = (1 << _CHUNK_INDEX_BITS) - 1
_MAX_TRANSFER_ID = (1 << _TRANSFER_ID_BITS) - 1
_MAX_ENDPOINT_GENERATION = (1 << _ENDPOINT_GENERATION_BITS) - 1
_MAX_TAG_DOMAIN = _TAG_MASK

_TAG_KIND_READY = 1
_TAG_KIND_RESULT = 2
_TAG_KIND_DATA = 3

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
    tag_domain: int = 0
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
                "tag_domain": self.tag_domain,
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
            tag_domain=int(payload.get("tag_domain", 0)),
            features=tuple(payload.get("features", ())),
            worker_address=payload.get("worker_address", b""),
        )


class B10TransferIdAllocator:
    """Allocates transfer ids from a fixed ring, with failure quarantine.

    Transfer ids seed UCX message-tag derivation (`_message_tag`), so an id
    must not be reused while a peer could still match messages tagged with
    it. `release` returns an id to the ring on clean completion;
    `quarantine` instead parks it for `quarantine_ttl_s` after a failure or
    timeout, because a wedged or slow peer may still post sends/receives
    with tags derived from that id — early reuse could route a stale
    message into a fresh transfer. Quarantined ids re-enter circulation
    lazily on `allocate` once their TTL expires.
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


class B10TagCollisionError(RuntimeError):
    pass


class B10TagRegistry:
    def __init__(
        self,
        quarantine_ttl_s: float = _DEFAULT_TAG_QUARANTINE_TTL_S,
    ):
        self._quarantine_ttl_s = quarantine_ttl_s
        self._active_tags: set[int] = set()
        self._owner_tags: dict[Hashable, tuple[int, ...]] = {}
        self._quarantined_until: dict[int, float] = {}
        self._next_quarantine_expiry_s: Optional[float] = None
        self._lock = threading.Lock()

    def reserve(self, owner: Hashable, tags: Iterable[int]) -> None:
        reserved_tags: list[int] = []
        seen_tags: set[int] = set()
        with self._lock:
            self._expire_quarantine_locked()
            if owner in self._owner_tags:
                raise B10TagCollisionError(f"B10 tag owner already has active tags: {owner}")
            for tag in tags:
                if tag in seen_tags:
                    raise B10TagCollisionError("B10 tag reservation contains duplicate tags")
                if tag in self._active_tags or tag in self._quarantined_until:
                    raise B10TagCollisionError(
                        f"B10 tag reservation collides with active/quarantined tag {tag}"
                    )
                seen_tags.add(tag)
                reserved_tags.append(tag)
            self._active_tags.update(reserved_tags)
            self._owner_tags[owner] = tuple(reserved_tags)

    def release(self, owner: Hashable) -> None:
        with self._lock:
            tags = self._owner_tags.pop(owner, set())
            self._active_tags.difference_update(tags)

    def quarantine(self, owner: Hashable) -> None:
        with self._lock:
            tags = self._owner_tags.pop(owner, set())
            if not tags:
                return
            self._active_tags.difference_update(tags)
            expiry = time.monotonic() + self._quarantine_ttl_s
            for tag in tags:
                self._quarantined_until[tag] = expiry
            next_expiry = self._next_quarantine_expiry_s
            if next_expiry is None or expiry < next_expiry:
                self._next_quarantine_expiry_s = expiry

    def _expire_quarantine_locked(self) -> None:
        if not self._quarantined_until:
            self._next_quarantine_expiry_s = None
            return
        now = time.monotonic()
        next_expiry = self._next_quarantine_expiry_s
        if next_expiry is not None and next_expiry > now:
            return
        expired: list[int] = []
        next_expiry = None
        for tag, expiry in self._quarantined_until.items():
            if expiry <= now:
                expired.append(tag)
            elif next_expiry is None or expiry < next_expiry:
                next_expiry = expiry
        for tag in expired:
            self._quarantined_until.pop(tag, None)
        self._next_quarantine_expiry_s = next_expiry


def _next_endpoint_generation(generation: int) -> int:
    generation = (generation + 1) & _MAX_ENDPOINT_GENERATION
    return 1 if generation == 0 else generation


def _mix64(value: int) -> int:
    value &= _TAG_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _TAG_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _TAG_MASK
    return (value ^ (value >> 31)) & _TAG_MASK


def _fold_tag_value(seed: int, value: int) -> int:
    return _mix64(
        seed ^ (_mix64(value) + 0x9E3779B97F4A7C15 + ((seed << 6) & _TAG_MASK) + (seed >> 2))
    )


def _pair_tag_domain(local_tag_domain: int, remote_tag_domain: int, slot_index: int) -> int:
    seed = _fold_tag_value(0, local_tag_domain)
    seed = _fold_tag_value(seed, remote_tag_domain)
    return _fold_tag_value(seed, slot_index)


def _message_tag(
    endpoint_generation: int,
    transfer_id: int,
    tag_kind: int,
    chunk_index: int = 0,
    tag_domain: int = 0,
) -> int:
    """Derive the 64-bit UCX tag for one message of a transfer.

    Tags are computed, not negotiated: sender and receiver independently
    fold (tag_domain, endpoint_generation, transfer_id, chunk_index,
    tag_kind) through a deterministic bit mixer (`_mix64` is the splitmix64
    finalizer; `_fold_tag_value` combines values boost::hash_combine-style)
    and arrive at the same tag with no extra round-trip. The mixer is a
    hash, not randomness — identical inputs always produce the identical
    tag, and distinct tuples collide only with birthday probability in the
    64-bit space (~n^2 / 2^65 for n live tags; negligible at realistic
    in-flight counts).

    Correctness does not rest on that probability alone: both pipelines
    reserve every tag of a transfer in `B10TagRegistry` before posting
    sends/receives, so a collision with an active or quarantined tag
    raises `B10TagCollisionError` and fails the transfer up front rather
    than risking a mis-matched message. Tag `_BOOTSTRAP_CONTROL_TAG` (0)
    is reserved for endpoint bootstrap, so a derived tag landing on it is
    remapped to 1. See DESIGN.md "Tag discipline".
    """
    if endpoint_generation <= 0 or endpoint_generation > _MAX_ENDPOINT_GENERATION:
        raise ValueError(f"endpoint_generation must be in [1, {_MAX_ENDPOINT_GENERATION}]")
    if transfer_id < 0 or transfer_id > _MAX_TRANSFER_ID:
        raise ValueError(f"transfer_id must be in [0, {_MAX_TRANSFER_ID}]")
    if chunk_index < 0 or chunk_index > _MAX_CHUNK_INDEX:
        raise ValueError(f"chunk_index must be in [0, {_MAX_CHUNK_INDEX}]")
    if tag_domain < 0 or tag_domain > _MAX_TAG_DOMAIN:
        raise ValueError(f"tag_domain must be in [0, {_MAX_TAG_DOMAIN}]")
    seed = _fold_tag_value(tag_domain, endpoint_generation)
    seed = _fold_tag_value(seed, transfer_id)
    seed = _fold_tag_value(seed, chunk_index)
    tag = _fold_tag_value(seed, tag_kind)
    return 1 if tag == _BOOTSTRAP_CONTROL_TAG else tag


def _ready_tag(transfer_id: int, endpoint_generation: int = 1, tag_domain: int = 0) -> int:
    return _message_tag(endpoint_generation, transfer_id, _TAG_KIND_READY, tag_domain=tag_domain)


def _result_tag(transfer_id: int, endpoint_generation: int = 1, tag_domain: int = 0) -> int:
    return _message_tag(endpoint_generation, transfer_id, _TAG_KIND_RESULT, tag_domain=tag_domain)


def _data_tag(
    transfer_id: int, chunk_index: int, endpoint_generation: int = 1, tag_domain: int = 0
) -> int:
    return _message_tag(
        endpoint_generation, transfer_id, _TAG_KIND_DATA, chunk_index, tag_domain=tag_domain
    )


def _transfer_message_tags(
    transfer_id: int, endpoint_generation: int, tag_domain: int, wire_chunk_count: int
) -> tuple[int, ...]:
    if wire_chunk_count < 0:
        raise ValueError("wire_chunk_count must be non-negative")
    return (
        _ready_tag(transfer_id, endpoint_generation, tag_domain),
        _result_tag(transfer_id, endpoint_generation, tag_domain),
        *(
            _data_tag(transfer_id, chunk_index, endpoint_generation, tag_domain)
            for chunk_index in range(wire_chunk_count)
        ),
    )


def _pack_message(payload: dict[str, Any]) -> bytes:
    return msgpack.packb(payload, use_bin_type=True)


def _unpack_message(payload: bytes) -> dict[str, Any]:
    return msgpack.unpackb(payload, raw=False)


# The tag-plane wire helpers (_send_obj/_recv_obj/_send_reply/_recv_reply)
# were removed with the move to the AM plane; control-plane messages now
# flow through _am_send_message and the B10AmDispatcher.
