# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
This file contains the primary interface for the KV Cache Connector.

The KV Cache Connector is a component that allows for remote KV cache access.
It is responsible for:
- Orchestrating the loading and saving of KV cache blocks.
- Managing asynchronous block tx/rx.

It can be used to provide functionalities such as:
1. Disagg
2. KV offload/onboard
3. KV cache sharing
4. P2P KV cache transfer
etc.

The Connector API is split into two parts:
1. The scheduler, which is responsible for orchestration, and building metadata for the workers.
2. The worker, which performs and monitors transfers indicated by the scheduler's metadata.

To implement a custom KV connector, you need to implement both the scheduler and worker-side interfaces.
"""

import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple

import torch

from tensorrt_llm._utils import mpi_allgather, mpi_broadcast, mpi_rank
from tensorrt_llm.bindings import LlmRequestState
from tensorrt_llm.bindings.internal.batch_manager import (
    KvCacheConnectorManager as KvCacheConnectorManagerCpp,
)
from tensorrt_llm.bindings.internal.batch_manager import KvCachePersistenceLease, LlmRequest
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
from tensorrt_llm.logger import logger

from ..llm_request import get_draft_token_length
from ..scheduler import ScheduledRequests

if TYPE_CHECKING:
    from ..resource_manager import KVCacheManager


# Used to store data for a single inflight request.
@dataclass
class RequestData:
    # The request ID.
    request_id: int
    # The new tokens that were generated in the prior forward pass.
    new_tokens: List[int]
    # The new block IDs allocated in the prior forward pass.
    new_block_ids: List[int]
    # The position of the latest token with computed (valid) kv cache values.
    computed_position: int
    # The number of scheduled tokens for the upcoming forward pass.
    num_scheduled_tokens: int
    # Block hashes for full blocks of beam 0 when that chain changed in this
    # scheduler step. Incremental connectors receive the suffix beginning at
    # ``block_hash_start``; other connectors retain the cumulative chain.
    block_hashes: Optional[List[int]] = None
    # The retention priorities for each new block (same length as new_block_ids).
    # Used for priority-based offload filtering. None means use default priority.
    priorities: Optional[List[int]] = None
    # Per-request cache salt that the KV cache manager uses to isolate reuse
    # between requests carrying different salts. Connectors that key cached
    # content on token sequences (e.g. by hashing tokens to a file path or
    # remote object id) MUST mix cache_salt into their identifiers,
    # otherwise blocks from a different salt could be incorrectly reused.
    cache_salt: Optional[str] = None
    # Stable TRT block-object IDs aligned one-to-one with ``block_hashes``.
    # Device transfers use new_block_ids, which are current primary memory-pool
    # offsets and can differ after native host onboarding.
    block_object_ids: Optional[List[int]] = None
    # Completed block index corresponding to the first incremental block hash.
    block_hash_start: int = 0


# A class to store some basic data regarding all inflight requests.
# This is used when calling `build_connector_meta` on the scheduler.
@dataclass
class SchedulerOutput:
    # Requests being scheduled for the first time. Requests will show up in `new_request` exactly once.
    new_requests: List[RequestData] = field(default_factory=list)

    # Requests being scheduled, that have already shown up in `new_requests`.
    cached_requests: List[RequestData] = field(default_factory=list)


@dataclass(frozen=True)
class ConnectorStateOnly:
    """The scheduler advanced without producing worker-visible work."""


@dataclass(frozen=True)
class ConnectorWorkerMetadata:
    """Metadata that workers must bind before the next forward pass."""

    metadata: object


ConnectorUpdate = ConnectorStateOnly | ConnectorWorkerMetadata


class KvCacheConnectorWorker(ABC):
    requires_load_finalization: bool = False

    def __init__(self, llm_args: TorchLlmArgs):
        self._llm_args = llm_args
        self._metadata = None
        super().__init__()

    def bind_connector_meta(self, metadata: object):
        self._metadata = metadata

    def get_connector_meta(self) -> object:
        return self._metadata

    def shutdown(self) -> None:
        """Release resources owned by the connector worker."""

    def _clear_connector_meta(self):
        self._metadata = None

    def can_skip_scheduler_match(self, request_num_tokens: int, num_computed_tokens: int) -> bool:
        """Return whether scheduler-side matching cannot add reusable tokens.

        Implementations that opt in must return the same result on every
        distributed rank. The request will still be included in connector
        metadata, but ``get_num_new_matched_tokens`` will not run.
        """
        return False

    def supports_rank_local_metadata_skip(self) -> bool:
        """Return whether every rank can make the same metadata-skip decision.

        Opting in requires rank-identical scheduling. Unless
        ``supports_sparse_metadata_updates`` is also enabled, the connector
        scheduler must implement ``advance_without_worker_metadata`` so
        leader-owned request state still advances on skipped exchanges.
        """
        return False

    def supports_sparse_metadata_updates(self) -> bool:
        """Return whether unchanged decode steps may defer connector state.

        The paired scheduler must consume the complete token and block delta
        at the next worker-visible boundary. Worker batch, forward, and save
        hooks must only be needed after a real metadata bind.
        """
        return False

    def supports_state_only_connector_updates(self) -> bool:
        """Return whether the scheduler can atomically suppress worker metadata.

        The paired scheduler must return ``ConnectorStateOnly`` only after it
        has applied all leader-owned state for the update and proved that no
        new worker slot or transfer operation is pending.
        """
        return False

    def supports_incremental_persistence_identities(self) -> bool:
        """Return whether completed block identities may be sent as tail deltas."""
        return False

    def supports_schedulable_reuse_preview(self) -> bool:
        """Return whether local KV reuse may be previewed before scheduling.

        The preview only changes the scheduler's compute estimate; allocation
        still performs the authoritative local and connector lookups. A
        connector must opt in only when it accepts a nonzero locally-computed
        prefix in ``get_num_new_matched_tokens`` on every distributed rank.
        """
        return False

    def requires_layerwise_transfer_hooks(self) -> bool:
        """Return whether the executor must invoke per-layer load/save hooks.

        Connectors that perform transfers only at batch and whole-forward
        boundaries may opt out to avoid two Python hooks per decoder layer.
        """
        return True

    def uses_secondary_kv_pool_as_persistence_staging(self) -> bool:
        """Return whether native host memory is borrowed as connector staging.

        Connectors that opt in must persist every lease before reporting its ID
        as terminally complete. Until then TensorRT-LLM keeps the secondary slot
        pinned and unavailable for another eviction.
        """
        return False

    def bind_persistence_lease_manager(self, manager: Any) -> None:
        """Bind the manager drained by pre-forward persistence submission."""
        raise NotImplementedError(
            "Secondary-pool persistence staging requires a bound lease manager"
        )

    def poll_globally_completed_persistence_leases(self) -> List[int]:
        """Return lease IDs coordinated for release at this rank-lockstep epoch."""
        raise NotImplementedError(
            "Secondary-pool persistence staging requires coordinated completion polling"
        )

    def submit_pending_persistence_leases(self, stream: torch.cuda.Stream) -> None:
        """Drain and submit staged leases after native D2H is ordered on stream.

        Workers using secondary-pool persistence staging must record readiness
        on ``stream`` and return without synchronizing the CPU.
        """
        raise NotImplementedError(
            "Secondary-pool persistence staging requires pre-forward lease submission"
        )

    def request_finished_without_save(self, request_id: int) -> None:
        """Retire rank-local request state without dispatching persistence.

        Workers using secondary-pool persistence staging must implement this
        hook because those requests never enter the asynchronous save lifecycle
        consumed by ``get_finished``.
        """
        raise NotImplementedError(
            "Secondary-pool persistence staging requires explicit worker no-save cleanup"
        )

    def register_forward_pass_callable(self) -> Callable:
        """
        This callable will be called at the end of the forward pass.

        Any CUDA calls which happen in the callable will execute on the
        same stream as the forward pass.

        This method is typically used by the connector to insert a
        cuda event into the forward pass cuda stream to obtain a
        signal of when it's appropriate to start offloading cache blocks.
        """

    @abstractmethod
    def register_kv_caches(
        self,
        kv_cache_tensor: torch.Tensor,
        secondary_kv_cache_tensor: Optional[torch.Tensor] = None,
    ):
        """
        Register the KV cache tensors to the worker.
        This can be used for something like NIXL registration.

        Args:
            kv_cache_tensor: The contiguous KV cache tensor.
            secondary_kv_cache_tensor: The optional contiguous secondary pool
                used by connectors that opt into persistence staging. This is
                ``None`` on replicated TP ranks that own no secondary blocks.
        """

    @abstractmethod
    def start_load_kv(self, stream: torch.cuda.Stream):
        """
        Begin loading the KV cache in preparation for the next forward pass.
        Specific blocks to transfer are indicated by the scheduler's metadata.
        """

    @abstractmethod
    def wait_for_layer_load(self, layer_idx: int, stream: torch.cuda.Stream):
        """
        Wait for a layer to finish being loaded before proceeding with the forward pass on the layer.
        Note: This function is called immediately before the layer's work is enqueued into the stream.

        Args:
            layer_idx: The index of the layer to wait for.
            stream: The stream the forward pass is being executed on.
        """

    @abstractmethod
    def save_kv_layer(self, layer_idx: int, stream: torch.cuda.Stream):
        """
        Begin saving the KV cache for a layer.
        Note: This function is called immediately after the layer's work is enqueued into the stream.

        Args:
            layer_idx: The index of the layer to save.
            stream: The stream the forward pass is being executed on.
        """

    @abstractmethod
    def wait_for_save(self, stream: torch.cuda.Stream):
        """
        Block until all synchronous saving operations are complete. Called at the end of the forward pass.
        """

    @abstractmethod
    def get_finished(
        self, finished_gen_req_ids: List[int], started_loading_req_ids: List[int]
    ) -> Tuple[List[int], List[int]]:
        """
        Get the requests that have finished loading and saving.

        Args:
            finished_gen_req_ids: The IDs of the requests that have
                finished generating tokens, and are now asynchronously saving.
            started_loading_req_ids: The IDs of the requests that have
                started asynchronously loading.

        Returns:
            The IDs of the requests that have finished saving.
            The IDs of the requests that have finished loading.

        Note: IDs may only be returned from this call after they've been
        provided in the ``finished_gen_req_ids`` and
        ``started_loading_req_ids`` arguments.  Additionally, the runtime
        will only take action based on these returned IDs once they've
        been returned by ALL workers. This allows some workers to take
        longer than others to complete the operations.
        """

    def get_load_finalization_status(self) -> Optional[Tuple[int, str]]:
        """Return local readiness for a load requiring TP-wide finalization.

        Most connectors complete their payload transfer before returning an ID
        from :meth:`get_finished` and therefore return ``None``. A connector
        with an asynchronous preparation phase may return ``(request_id,
        status)`` for its deterministic queue head, where status is one of
        ``pending``, ``ready``, or ``failed``. The manager combines this value
        across TP ranks before invoking :meth:`finalize_finished_loads`.
        """
        return None

    def finalize_finished_loads(self, request_ids: List[int], stream: torch.cuda.Stream) -> None:
        """Synchronously finalize globally-ready loads before rescheduling.

        This hook runs with the same ordered request IDs on every TP rank after
        host-side consensus and before the requests return to ``CONTEXT_INIT``.
        Connectors may fence ``stream`` and launch required payload collectives
        here. The default is a no-op for already-complete loads.
        """


class KvCacheConnectorScheduler(ABC):
    def __init__(self, llm_args: TorchLlmArgs):
        self._llm_args = llm_args
        super().__init__()

    def shutdown(self) -> None:
        """Release resources owned by the connector scheduler."""

    @abstractmethod
    def build_connector_meta(self, scheduler_output: SchedulerOutput):
        """
        Build the metadata for the worker.
        This is called by the KV Cache Manager when adding a sequence.
        Args:
            scheduler_output: The data for all inflight requests.

        Returns:
            The metadata for the workers.
        """

    def build_connector_update(self, scheduler_output: SchedulerOutput) -> ConnectorUpdate:
        """Advance scheduler state and describe any worker-visible work.

        Existing connectors produce worker metadata on every visible scheduler
        update. Connectors may override this method to return
        ``ConnectorStateOnly`` after atomically applying leader-only state.
        """
        return ConnectorWorkerMetadata(self.build_connector_meta(scheduler_output))

    @abstractmethod
    def get_num_new_matched_tokens(
        self, request: LlmRequest, num_computed_tokens: int
    ) -> Tuple[int, bool]:
        """
        Get the number of tokens that can be loaded from remote KV cache.
        This does not include the tokens already matched on device (indicated by `num_computed_tokens`).

        Args:
            request: The request to get the number of tokens for.
            num_computed_tokens: The number of tokens already matched on device.

        Returns:
            The number of tokens that can be loaded from remote KV cache.
            Whether the tokens will be loaded asynchronously.
        """

    def prepare_scheduler_match_skip(self, request: LlmRequest, num_computed_tokens: int) -> None:
        """Prepare leader state before a worker-approved scheduler-match skip.

        Workers that opt into ``can_skip_scheduler_match`` must pair with a
        scheduler that implements this hook. It runs only on the leader and
        must not perform remote matching or distributed collectives.
        """
        raise RuntimeError(
            "Connector worker skipped scheduler matching without a scheduler "
            "implementation of prepare_scheduler_match_skip"
        )

    def advance_without_worker_metadata(self, scheduler_output: SchedulerOutput) -> None:
        """Advance leader state without producing worker-visible metadata.

        This runs only on the leader. Implementations must retain asynchronous
        worker operations until a later full metadata exchange and fail if the
        provided output itself requires a worker operation.
        """
        raise RuntimeError(
            "Connector worker skipped metadata exchange without a scheduler "
            "implementation of advance_without_worker_metadata"
        )

    @abstractmethod
    def request_finished(self, request: LlmRequest, cache_block_ids: List[int]) -> bool:
        """
        Called when a request is finished generating tokens.

        Args:
            request: The request that finished generating tokens.

        Returns:
            Whether the request is performing asynchronous saving operations.
            If true, this indicates that the kv cache manager should wait
            to deallocate the blocks until the saving has completed
            (determined by ``get_finished`` on the workers).
        """

    def request_finished_without_save(self, request: LlmRequest) -> None:
        """Retire request state without dispatching connector persistence.

        Connectors using secondary-pool persistence staging must implement
        explicit no-save cleanup; falling back to ``request_finished`` would
        reintroduce request-scoped D2H transfers.
        """
        raise NotImplementedError(
            "Secondary-pool persistence staging requires explicit no-save request cleanup"
        )

    def resolve_persistence_keys(
        self, source_block_ids: List[int], framework_block_hashes: List[int]
    ) -> List[int]:
        """Resolve evicted framework blocks into the connector's keyspace.

        Connectors whose persistence key is the framework block hash need no
        translation. Connectors with an independent canonical keyspace may
        override this method, using ``source_block_ids`` to keep the mapping
        bounded by TRT block-object lifetime.
        """
        if len(source_block_ids) != len(framework_block_hashes):
            raise RuntimeError(
                "Persistence source block IDs and framework hashes must have equal length"
            )
        return framework_block_hashes

    def retire_persistence_identities(self, identities: List[Tuple[int, int]]) -> None:
        """Retire exact block-object residencies that can no longer produce a lease."""

    def register_persistence_identities(
        self,
        request: LlmRequest,
        block_object_ids: List[int],
        external_sequence_hashes: List[int],
    ) -> None:
        """Register stable persistence identities before blocks can be evicted.

        Secondary-pool persistence leases are published by ``refresh_blocks``.
        Connectors with an independent canonical keyspace must override this
        hook so a newly allocated residency is resolvable at that boundary.
        """

    @abstractmethod
    def update_state_after_alloc(self, request: LlmRequest, block_ids: List[int]):
        """
        Called after get_num_new_matched_tokens is called to provide the block ids to the scheduler.

        Args:
            request: The request that was allocated resources.
            block_ids: The KV cacheblock IDs that were allocated.
        """

    def on_rewind(self, request: LlmRequest, live_block_ids: List[int]):
        """Notify the scheduler that a request's KV cache was rewound.

        Called after ``rewind_kv_cache`` frees blocks due to speculative-decoding
        rejection.  The scheduler should trim any per-request block bookkeeping
        to match ``live_block_ids`` (the post-rewind cache indices).

        Default implementation is a no-op; connectors that track per-request
        block state (e.g. KVBM) override this to trim stale freed block ids.

        Args:
            request: The request whose KV cache was rewound.
            live_block_ids: The post-rewind live cache block IDs for the request.
        """

    def wait_for_initialization(self):
        """
        Some connectors need to wait for some resources to be initialized.
        For example, FlexKV needs to wait for the FlexKV manager to be initialized.
        """
        return


# An internal dataclass to handle async saving/loading requests.
@dataclass
class AsyncRequests:
    saving: Dict[int, LlmRequest]
    loading: Dict[int, LlmRequest]

    def add_from(self, other: "AsyncRequests"):
        """
        Remove requests from the other `AsyncRequests` object, and add them to this one.
        """
        self.saving.update(other.saving)
        self.loading.update(other.loading)

        other.saving = dict()
        other.loading = dict()

    def extract_by_id(self, saving_ids: List[int], loading_ids: List[int]) -> "AsyncRequests":
        """
        Extract the requests with the given IDs from this `AsyncRequests` object.

        Args:
            saving_ids: The IDs of the requests to extract.
            loading_ids: The IDs of the requests to extract.
        """
        new_async_requests = AsyncRequests(dict(), dict())

        for req_id in saving_ids:
            new_async_requests.saving[req_id] = self.saving[req_id]
            del self.saving[req_id]
        for req_id in loading_ids:
            new_async_requests.loading[req_id] = self.loading[req_id]
            del self.loading[req_id]

        return new_async_requests

    @property
    def saving_ids(self) -> Set[int]:
        """
        Get the IDs of the requests that are being saved asynchronously.
        """
        return set(self.saving.keys())

    @property
    def loading_ids(self) -> Set[int]:
        """
        Get the IDs of the requests that are being loaded asynchronously.
        """
        return set(self.loading.keys())

    @property
    def is_empty(self) -> bool:
        return not self.saving and not self.loading


class KvCacheConnectorSchedulerOutputRequest:
    def __init__(self):
        self.block_ids = []
        self.block_object_ids = []
        self.tokens = []
        self.hash_probe_state = None
        self.block_hash_count = 0

    def update_and_build_data(
        self,
        req: LlmRequest,
        kv_cache_manager: "KVCacheManager",
        incremental_persistence_identities: bool = False,
    ):
        num_tokens = req.get_num_tokens(0)
        if req.state in (
            LlmRequestState.CONTEXT_INIT,
            LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS,
        ):
            is_generation = False
            computed_position = req.context_current_position
            num_scheduled_tokens = min(req.context_remaining_length, req.context_chunk_size)
        else:
            is_generation = True
            computed_position = num_tokens - 1
            num_scheduled_tokens = 1 + get_draft_token_length(
                req
            )  # Account for the next token plus any spec-dec draft tokens.
            # https://basetenlabs.slack.com/archives/C0BGBSQDQUS/p1784038072194869?thread_ts=1783883497.594279&cid=C0BGBSQDQUS

        # Context scheduling is infrequent and may add a large token chunk, so
        # retain the bulk read there. During generation, avoid converting the
        # entire request (often tens of thousands of tokens) from C++ on every
        # decode step just to retrieve the 1-4 newly accepted tokens.
        if not self.tokens:
            tokens = req.get_tokens(0)
            num_tokens = len(tokens)
            new_tokens = tokens
        elif is_generation:
            if num_tokens < len(self.tokens):
                raise RuntimeError(
                    "Connector token state exceeds the live request; "
                    "on_rewind must run before the next scheduler update"
                )
            new_tokens = [
                req.get_token(0, position) for position in range(len(self.tokens), num_tokens)
            ]
        elif num_tokens == len(self.tokens):
            new_tokens = []
        else:
            tokens = req.get_tokens(0)
            if len(tokens) < len(self.tokens):
                raise RuntimeError("Connector token state exceeds the live context request")
            num_tokens = len(tokens)
            new_tokens = tokens[len(self.tokens) :]

        tokens_per_block = kv_cache_manager.tokens_per_block

        # Active generation keeps stable block IDs until its allocation grows.
        # Avoid rematerializing the full block list on every decode step.
        next_position = computed_position + num_scheduled_tokens
        required_blocks = (next_position + tokens_per_block - 1) // tokens_per_block
        if not is_generation or not self.block_ids or required_blocks > len(self.block_ids):
            block_object_ids = kv_cache_manager.get_cache_indices(req)
            block_ids = kv_cache_manager.get_connector_cache_indices(req)
            if len(block_ids) != len(block_object_ids):
                raise RuntimeError("Connector device block indices do not match TRT block objects")
            old_block_count = len(self.block_ids)
            allocation_prefix_unchanged = (
                block_ids[:old_block_count] == self.block_ids
                and block_object_ids[:old_block_count] == self.block_object_ids
            )
            if allocation_prefix_unchanged:
                new_block_ids = block_ids[old_block_count:]
                new_block_object_ids = block_object_ids[old_block_count:]
            else:
                # The allocation callback has already replaced the connector's
                # device state. Send the full replacement so retention
                # priorities are refreshed; KVBM's overlap guard prevents
                # these IDs from being appended as additional blocks.
                new_block_ids = block_ids
                new_block_object_ids = block_object_ids
                self.hash_probe_state = None
                self.block_hash_count = 0
            self.block_ids = block_ids
            self.block_object_ids = block_object_ids
        else:
            new_block_ids = []
            new_block_object_ids = []

        # Cumulative block hashes are immutable between full-block boundaries.
        # Probe and forward the chain only when another block can have become
        # full. Rewinds invalidate this marker in ``on_rewind``.
        if is_generation and req.state != LlmRequestState.GENERATION_TO_COMPLETE:
            hashed_position = computed_position
        else:
            hashed_position = num_tokens
        num_hashed_tokens = (hashed_position // tokens_per_block) * tokens_per_block
        hash_probe_state = (
            num_hashed_tokens if is_generation else (num_hashed_tokens, len(self.block_ids))
        )
        block_hash_start = 0
        if hash_probe_state != self.hash_probe_state:
            if incremental_persistence_identities:
                block_hash_start = self.block_hash_count
                block_hashes = kv_cache_manager.commit_and_get_block_hashes_range(
                    req, block_hash_start
                )
            else:
                block_hashes = kv_cache_manager.commit_and_get_block_hashes(req)
            self.block_hash_count = block_hash_start + len(block_hashes)
            self.hash_probe_state = hash_probe_state
        else:
            block_hashes = None

        self.tokens.extend(new_tokens)
        # Get retention priority for each new block only if retention config is provided
        # (for priority-based offload filtering)
        priorities = None
        if req.kv_cache_retention_config is not None:
            priorities = [
                kv_cache_manager.get_priority_by_block_id(block_id)
                for block_id in new_block_object_ids
            ]

        return RequestData(
            request_id=req.request_id,
            new_tokens=new_tokens,
            new_block_ids=new_block_ids,
            computed_position=computed_position,
            num_scheduled_tokens=num_scheduled_tokens,
            block_object_ids=(
                list(self.block_object_ids[block_hash_start : self.block_hash_count])
                if block_hashes is not None and incremental_persistence_identities
                else list(self.block_object_ids)
                if block_hashes is not None
                else None
            ),
            block_hashes=block_hashes,
            block_hash_start=block_hash_start,
            priorities=priorities,
            cache_salt=req.cache_salt,
        )


class KvCacheConnectorSchedulerOutputManager:
    def __init__(self):
        self.requests = defaultdict(KvCacheConnectorSchedulerOutputRequest)
        self.external_loads = dict()

    def build_scheduler_output(
        self,
        scheduled_batch: ScheduledRequests,
        new_async_requests: AsyncRequests,
        kv_cache_manager: "KVCacheManager",
        request_ids: Optional[Set[int]] = None,
        incremental_persistence_identities: bool = False,
    ):
        scheduler_output = SchedulerOutput()

        for req in scheduled_batch.context_requests:
            if request_ids is not None and req.request_id not in request_ids:
                continue
            if req.request_id in new_async_requests.loading_ids:
                continue

            is_new = req.request_id not in self.requests

            request_data = self.requests[req.request_id].update_and_build_data(
                req, kv_cache_manager, incremental_persistence_identities
            )

            # Don't include the connector matched tokens in the initial scheduler output.
            if req.request_id in self.external_loads:
                request_data.computed_position -= self.external_loads[req.request_id]

            if is_new:
                scheduler_output.new_requests.append(request_data)
            else:
                scheduler_output.cached_requests.append(request_data)

        for req in scheduled_batch.generation_requests:
            if request_ids is not None and req.request_id not in request_ids:
                continue
            request_data = self.requests[req.request_id].update_and_build_data(
                req, kv_cache_manager, incremental_persistence_identities
            )

            scheduler_output.cached_requests.append(request_data)

        if request_ids is None:
            self.external_loads = dict()
        else:
            for request_id in request_ids:
                self.external_loads.pop(request_id, None)

        return scheduler_output

    def record_new_matched_tokens(self, request: LlmRequest, num_new_matched_tokens: int):
        self.external_loads[request.request_id] = num_new_matched_tokens

    def rewind_may_affect_connector_state(
        self, request_id: int, num_tokens: int, tokens_per_block: int
    ) -> bool:
        """Return whether a rewind can trim state already sent to the connector."""
        request_state = self.requests.get(request_id)
        if request_state is None:
            return False
        computed_tokens = max(num_tokens - 1, 0)
        minimum_live_blocks = (computed_tokens + tokens_per_block - 1) // tokens_per_block
        return num_tokens < len(request_state.tokens) or minimum_live_blocks < len(
            request_state.block_ids
        )

    def on_rewind(
        self,
        req: LlmRequest,
        live_block_ids: List[int],
        live_block_object_ids: List[int],
        num_tokens: int,
    ) -> Optional[List[int]]:
        """Re-sync connector bookkeeping after speculative-decoding rewind.

        When draft tokens are rejected and ``rewindKVCache`` frees blocks that
        crossed a block boundary, the per-request ``block_ids`` list must be
        trimmed to drop the now-freed block ids.  Without this, the connector
        retains stale references to freed blocks, which can cause incorrect
        saves/loads and hangs in ``get_finished`` (cross-rank intersection
        never completes).

        ``tokens`` is only shrunk (never extended) here: ``on_rewind`` runs
        after sampling has already appended accepted tokens to the request,
        so ``req.get_tokens(0)`` may be longer than the stored list.  Extending
        ``tokens`` would suppress those accepted tokens from the next
        ``build_scheduler_output``'s ``new_tokens`` delta.  We only trim if the
        rewind actually shortened the token list (e.g. rejected draft tokens
        were rolled back). Returns the connector-visible block IDs whenever
        they differ from the physical allocation passed by the caller.
        """
        req_state = self.requests.get(req.request_id)
        if req_state is None:
            return None
        if len(live_block_ids) != len(live_block_object_ids):
            raise RuntimeError("Speculative rewind device indices do not match TRT block objects")

        recorded_block_ids = req_state.block_ids
        scheduler_live_block_ids: Optional[List[int]] = None
        if len(live_block_ids) > len(recorded_block_ids):
            if (
                live_block_ids[: len(recorded_block_ids)] != recorded_block_ids
                or live_block_object_ids[: len(recorded_block_ids)] != req_state.block_object_ids
            ):
                raise RuntimeError(
                    "Speculative allocation growth replaced connector-visible block IDs"
                )
            # The speculative suffix has not been sent to the connector. Keep
            # the reported length so the next metadata update emits it, and
            # prevent the rewind callback from exposing it early.
            scheduler_live_block_ids = list(recorded_block_ids)
        elif (
            live_block_ids != recorded_block_ids
            or live_block_object_ids != req_state.block_object_ids
        ):
            req_state.block_ids = list(live_block_ids)
            req_state.block_object_ids = list(live_block_object_ids)
            scheduler_live_block_ids = req_state.block_ids

        if num_tokens < len(req_state.tokens):
            req_state.tokens = list(req.get_tokens(0))
            # If accepted state also moved backward, synchronize the connector
            # against only the prefix it has already seen.
            if scheduler_live_block_ids is None:
                scheduler_live_block_ids = list(req_state.block_ids)
        if scheduler_live_block_ids is not None:
            req_state.hash_probe_state = None
            req_state.block_hash_count = 0
        return scheduler_live_block_ids


class KvCacheConnectorManager(KvCacheConnectorManagerCpp):
    """
    The KvCacheConnectorManager is used to manager connector-related state.

    It has the following responsibilities:
    1. Managing the state of async requests (both offload and onboard)
    2. Handling MPI communication. We only run the leader on one rank,
       but need the results of the leader API on all ranks.

    Note: This class is solely an implementation detail, and is not part of the connector interface itself.
    When implementing a connector API, you do not need to implement this class.
    """

    def __init__(
        self, worker: KvCacheConnectorWorker, scheduler: Optional[KvCacheConnectorScheduler]
    ):
        assert (scheduler is not None) == (mpi_rank() == 0), (
            "The scheduler may only exist on rank 0!"
        )

        super().__init__()

        self.worker = worker
        self.scheduler = scheduler
        self._is_shutdown = False
        self._uses_secondary_persistence_staging = (
            self.worker.uses_secondary_kv_pool_as_persistence_staging() is True
        )
        self._rank_local_metadata_skip_enabled = (
            self.worker.supports_rank_local_metadata_skip() is True
        )
        self._sparse_metadata_updates_enabled = (
            self.worker.supports_sparse_metadata_updates() is True
        )
        self._state_only_updates_enabled = (
            self.worker.supports_state_only_connector_updates() is True
        )
        self._incremental_persistence_identities_enabled = (
            self.worker.supports_incremental_persistence_identities() is True
        )
        if self._sparse_metadata_updates_enabled and not self._rank_local_metadata_skip_enabled:
            raise ValueError("Sparse connector metadata updates require rank-local metadata skip")
        self._drop_pending_persistence_reason: Optional[str] = None

        # Requests that haven't yet been passed into get_finished.
        self.new_async_requests = AsyncRequests(dict(), dict())

        # Requests that have been passed into get_finished, but haven't yet been returned.
        self.pending_async_requests = AsyncRequests(dict(), dict())

        # Requests that have been returned from get_finished locally, but haven't yet been returned by all workers.
        self.local_finished_async_requests = AsyncRequests(dict(), dict())

        # Requests that have finished loading asynchronously.
        self.finished_async_loading_requests = dict()

        # TP consensus arms these loads at the post-forward completion poll.
        # Payload collectives are deferred to the next executor-iteration
        # boundary, where the previous TRT execution stream can be quiesced.
        self._armed_load_finalizations: Dict[int, bool] = {}
        # Transient TP skew on finalization consensus, keyed by (kind, id).
        # Cleared whenever consensus converges; bounded by the timeout below.
        self._finalization_skew_since: Dict[Tuple[str, int], float] = {}

        self._scheduler_output = None
        self._metadata_pending = False
        self._skip_metadata_exchange = False
        self._worker_batch_hooks_pending = False
        self._worker_batch_hooks_active = False
        self._force_metadata_exchange = False
        self._metadata_exchange_progress = {}
        self._worker_visible_progress = {}
        self._metadata_request_ids: Optional[Set[int]] = None
        self._rank_local_state_only_pending = False
        self._generation_computed_tokens: Dict[int, int] = {}
        self._metadata_exchange_dirty_request_ids: Set[int] = set()
        self._metadata_exchange_included_dirty_request_ids: Set[int] = set()
        self._generation_progress_events_enabled = False
        self._generation_progress_fast_path_armed = False
        self.scheduler_output_manager = KvCacheConnectorSchedulerOutputManager()
        self._pending_persistence_leases: List[KvCachePersistenceLease] = []
        self._pending_persistence_tail_to_discard: List[KvCachePersistenceLease] = []
        self._persistence_tail_discard_pending = False
        self._resolved_persistence_keys: Optional[List[int]] = None
        self._persistence_resolution_error: Optional[Tuple[str, str]] = None
        self._outstanding_persistence_lease_ids: Set[int] = set()
        self._outstanding_persistence_lease_logical_descriptors: Dict[int, Tuple[int, int]] = {}
        self._rank_converged_outstanding_persistence_lease_count = 0
        self._pending_terminal_persistence_lease_ids: List[int] = []
        self._kv_cache_manager: Optional["KVCacheManager"] = None
        if self._uses_secondary_persistence_staging:
            self.worker.bind_persistence_lease_manager(self)

    def uses_secondary_kv_pool_as_persistence_staging(self) -> bool:
        """Expose the worker capability to the C++ cache-manager constructor."""
        return self._uses_secondary_persistence_staging

    def add_persistence_leases(self, leases: List[KvCachePersistenceLease]) -> None:
        """Receive one scheduler iteration's staged blocks from C++."""
        if not leases:
            return
        if self._pending_persistence_leases or self._persistence_tail_discard_pending:
            raise RuntimeError("Cannot add persistence leases while a batch is awaiting submission")
        lease_ids = [lease.lease_id for lease in leases]
        unique_lease_ids = set(lease_ids)
        if len(unique_lease_ids) != len(lease_ids):
            raise RuntimeError(f"Duplicate persistence lease IDs in batch: {lease_ids}")
        duplicate_lease_ids = unique_lease_ids & self._outstanding_persistence_lease_ids
        if duplicate_lease_ids:
            raise RuntimeError(
                f"Duplicate outstanding persistence lease IDs: {sorted(duplicate_lease_ids)}"
            )

        # refresh_blocks publishes leases before build_scheduler_output can
        # register a newly allocated residency for the same stable block
        # object. Snapshot the leader's canonical keys at that boundary; the
        # later metadata exchange only broadcasts the already-resolved batch.
        if self.scheduler is not None:
            descriptors = self._persistence_lease_descriptors(leases)
            try:
                persistence_keys = list(
                    self.scheduler.resolve_persistence_keys(
                        [descriptor[1] for descriptor in descriptors],
                        [descriptor[2] for descriptor in descriptors],
                    )
                )
                if len(persistence_keys) != len(leases):
                    raise RuntimeError(
                        "Connector resolved "
                        f"{len(persistence_keys)} persistence keys for "
                        f"{len(leases)} leases"
                    )
                self._resolved_persistence_keys = [int(key) for key in persistence_keys]
            # pyo3 PanicException inherits BaseException. Retain the leases and
            # broadcast the leader error in handle_metadata so peer ranks fail
            # together instead of diverging around a rank-local callback.
            except BaseException as error:
                self._persistence_resolution_error = (type(error).__name__, str(error))

        self._outstanding_persistence_lease_ids.update(unique_lease_ids)
        self._outstanding_persistence_lease_logical_descriptors.update(
            {
                int(lease.lease_id): (
                    int(lease.block_hash),
                    int(lease.secondary_block_index),
                )
                for lease in leases
            }
        )
        self._pending_persistence_leases.extend(leases)
        # refresh_blocks runs before build_scheduler_output. Force that update
        # through the existing metadata broadcast so resolution adds no
        # collective to ordinary decode iterations.
        self._force_metadata_exchange = True

    def retire_persistence_identities(self, identities: List[Tuple[int, int]]) -> None:
        """Discard translations after TRT proves their residencies cannot lease."""
        if self.scheduler is not None:
            self.scheduler.retire_persistence_identities(identities)

    @staticmethod
    def _persistence_lease_descriptors(
        leases: List[KvCachePersistenceLease],
    ) -> List[Tuple[int, int, int, int, int]]:
        return [
            (
                int(lease.lease_id),
                int(lease.source_block_id),
                int(lease.block_hash),
                int(lease.secondary_block_index),
                int(lease.priority),
            )
            for lease in leases
        ]

    def get_resolved_pending_persistence_leases(
        self,
    ) -> Tuple[List[KvCachePersistenceLease], List[int]]:
        """Peek at the resolved batch; submission commits the destructive drain."""
        if not self._pending_persistence_leases:
            return [], []
        if self._resolved_persistence_keys is None:
            raise RuntimeError("Persistence leases reached submission without metadata resolution")
        return self._pending_persistence_leases, self._resolved_persistence_keys

    def mark_persistence_leases_submitted(self, lease_ids: List[int]) -> None:
        pending_lease_ids = [int(lease.lease_id) for lease in self._pending_persistence_leases]
        if lease_ids != pending_lease_ids:
            raise RuntimeError(
                "Submitted persistence lease IDs do not match the pending batch: "
                f"submitted={lease_ids}, pending={pending_lease_ids}"
            )
        self._rank_converged_outstanding_persistence_lease_count += len(lease_ids)
        self._pending_persistence_leases = []
        self._resolved_persistence_keys = None
        self._persistence_resolution_error = None
        self._drop_pending_persistence_reason = None

    def discard_pending_persistence_leases(self, lease_ids: List[int]) -> None:
        """Release an unpublishable batch after its native D2H copy is safe."""
        if self._drop_pending_persistence_reason is None:
            raise RuntimeError("Cannot discard a publishable persistence lease batch")
        pending_lease_ids = [int(lease.lease_id) for lease in self._pending_persistence_leases]
        if lease_ids != pending_lease_ids:
            raise RuntimeError(
                "Discarded persistence lease IDs do not match the pending batch: "
                f"discarded={lease_ids}, pending={pending_lease_ids}"
            )
        unknown_lease_ids = set(lease_ids) - self._outstanding_persistence_lease_ids
        if unknown_lease_ids:
            raise RuntimeError(
                f"Cannot discard unknown persistence lease IDs: {sorted(unknown_lease_ids)}"
            )
        if lease_ids:
            self.complete_persistence_leases(lease_ids)
            self._outstanding_persistence_lease_ids.difference_update(lease_ids)
            for lease_id in lease_ids:
                del self._outstanding_persistence_lease_logical_descriptors[lease_id]
        self._canonicalize_free_secondary_staging_block_order()
        self._pending_persistence_leases = []
        self._resolved_persistence_keys = None
        self._persistence_resolution_error = None
        self._drop_pending_persistence_reason = None

    def has_pending_persistence_leases(self) -> bool:
        """Return whether native D2H produced leases awaiting publication."""
        return bool(self._pending_persistence_leases)

    def _synchronize_persistence_discard(self, stream: torch.cuda.Stream) -> None:
        """Prove every rank's native D2H is safe before releasing staged blocks."""
        try:
            d2h_complete = torch.cuda.Event()
            d2h_complete.record(stream)
            d2h_complete.synchronize()
            local_d2h_ready = (True, None)
        except BaseException as error:
            local_d2h_ready = (False, f"{type(error).__name__}: {error}")
        rank_d2h_readiness = mpi_allgather(local_d2h_ready)
        if not all(ready for ready, _ in rank_d2h_readiness):
            raise RuntimeError(
                "Cannot discard persistence leases before every rank's D2H copy is safe: "
                f"{rank_d2h_readiness}"
            )

    def _discard_persistence_lease_tail(self) -> None:
        if not self._persistence_tail_discard_pending:
            raise RuntimeError("No persistence lease tail is pending discard")
        tail_ids = [int(lease.lease_id) for lease in self._pending_persistence_tail_to_discard]
        unknown_tail_ids = set(tail_ids) - self._outstanding_persistence_lease_ids
        if unknown_tail_ids:
            raise RuntimeError(
                f"Cannot discard unknown persistence lease IDs: {sorted(unknown_tail_ids)}"
            )
        if tail_ids:
            self.complete_persistence_leases(tail_ids)
            self._outstanding_persistence_lease_ids.difference_update(tail_ids)
            for lease_id in tail_ids:
                del self._outstanding_persistence_lease_logical_descriptors[lease_id]
        self._canonicalize_free_secondary_staging_block_order()
        self._pending_persistence_tail_to_discard = []
        self._persistence_tail_discard_pending = False

    def submit_pending_persistence_leases(self, stream: torch.cuda.Stream) -> None:
        """Submit the full D2H-ready batch before forward without CPU waiting."""
        if not self._uses_secondary_persistence_staging:
            return
        # These flags are collective state. Check them before the rank-local
        # pending batch so an empty rank cannot skip a barrier entered by peers.
        if self._drop_pending_persistence_reason is not None:
            self._synchronize_persistence_discard(stream)
            lease_ids = [int(lease.lease_id) for lease in self._pending_persistence_leases]
            logger.error(
                "Discarding connector persistence leases with unresolved exact identities: "
                "count=%s cause=%s",
                len(lease_ids),
                self._drop_pending_persistence_reason,
            )
            self.discard_pending_persistence_leases(lease_ids)
            return
        if self._persistence_tail_discard_pending:
            self._synchronize_persistence_discard(stream)
            self._discard_persistence_lease_tail()
        if not self.has_pending_persistence_leases():
            return
        self.worker.submit_pending_persistence_leases(stream)
        if self.has_pending_persistence_leases():
            raise RuntimeError("Persistence staging worker did not drain all pending leases")

    def bind_kv_cache_manager(self, kv_cache_manager: "KVCacheManager") -> None:
        """Bind terminal lease completion to the cache manager that owns the slots."""
        self._kv_cache_manager = kv_cache_manager

    def complete_persistence_leases(self, lease_ids: List[int]) -> None:
        """Release only leases whose persistence reached terminal completion."""
        if self._kv_cache_manager is None:
            raise RuntimeError("Persistence lease completion requires a bound KV cache manager")
        self._kv_cache_manager.complete_persistence_leases(lease_ids)

    def _canonicalize_free_secondary_staging_block_order(self) -> None:
        """Realign replicated TP staging-slot allocation after a discard."""
        if self._kv_cache_manager is None:
            raise RuntimeError(
                "Persistence staging canonicalization requires a bound KV cache manager"
            )
        self._kv_cache_manager.canonicalize_free_secondary_staging_block_order()

    def _converge_persistence_lease_prefix(
        self,
        rank_descriptors: List[List[Tuple[int, int, int, int, int]]],
        leader_descriptors: List[Tuple[int, int, int, int, int]],
        persistence_keys: List[int],
    ) -> Tuple[List[Tuple[int, int, int, int, int]], List[int]]:
        """Keep the logical eviction prefix staged by every replicated TP rank.

        Secondary-pool completions become visible to each executor rank at a
        slightly different wall-clock instant. A rank can therefore have a
        shorter staging tail for one allocation even though all ranks evicted
        the same logical block order. Publishing only the common prefix is
        safe; a rank-local tail has not been submitted to the connector yet and
        can be released like an unsuccessful native offload. C++ canonicalizes
        the free secondary-slot order after release so future TP allocations
        remain aligned.

        Lease and source block IDs are rank-local handles. Priority is also
        owner-local metadata. The framework hash proves logical contents and
        the global secondary index proves every rank refers to the same slot.
        """
        if not rank_descriptors:
            raise RuntimeError("Persistence lease convergence requires at least one rank")

        common_count = min(len(descriptors) for descriptors in rank_descriptors)
        leader_prefix = leader_descriptors[:common_count]
        leader_logical_prefix = [descriptor[2:4] for descriptor in leader_prefix]
        for rank, descriptors in enumerate(rank_descriptors):
            logical_prefix = [descriptor[2:4] for descriptor in descriptors[:common_count]]
            if logical_prefix == leader_logical_prefix:
                continue
            first_mismatch = next(
                index
                for index, (leader_descriptor, rank_descriptor) in enumerate(
                    zip(leader_logical_prefix, logical_prefix, strict=True)
                )
                if leader_descriptor != rank_descriptor
            )
            raise RuntimeError(
                "Logical persistence lease prefixes diverged across ranks: "
                f"rank={rank}, lengths={[len(value) for value in rank_descriptors]}, "
                f"first_mismatch={first_mismatch}, "
                f"leader={leader_logical_prefix[first_mismatch]}, "
                f"rank_value={logical_prefix[first_mismatch]}"
            )

        local_tail = self._pending_persistence_leases[common_count:]
        if self.scheduler is not None and local_tail:
            self.scheduler.retire_persistence_identities(
                [(int(lease.source_block_id), int(lease.block_hash)) for lease in local_tail]
            )
        self._pending_persistence_tail_to_discard = list(local_tail)
        self._persistence_tail_discard_pending = any(
            len(descriptors) != common_count for descriptors in rank_descriptors
        )
        self._pending_persistence_leases = self._pending_persistence_leases[:common_count]

        self._resolved_persistence_keys = [int(key) for key in persistence_keys[:common_count]]
        return leader_prefix, self._resolved_persistence_keys

    def reap_completed_persistence_leases(self) -> None:
        """Stage terminal leases for release before the next KV allocation.

        Persistence publication finishes on background threads, so its result
        becomes visible at slightly different wall-clock times on each TP rank.
        Releasing here would let one rank recycle a global secondary slot before
        its peers and break the single-owner replicated host topology.  The next
        allocation boundary converges the logical terminal prefix and releases
        every rank's local lease handles together.
        """
        if not self._outstanding_persistence_lease_ids:
            return

        terminal_ids = [
            int(lease_id) for lease_id in self.worker.poll_globally_completed_persistence_leases()
        ]
        terminal_set = set(terminal_ids)
        if len(terminal_set) != len(terminal_ids):
            raise RuntimeError(
                f"Connector globally completed duplicate persistence lease IDs: {terminal_ids}"
            )
        unknown_terminal = terminal_set - self._outstanding_persistence_lease_ids
        if unknown_terminal:
            raise RuntimeError(
                "Connector globally completed unknown persistence lease IDs: "
                f"{sorted(unknown_terminal)}"
            )
        duplicate_terminal = terminal_set & set(self._pending_terminal_persistence_lease_ids)
        if duplicate_terminal:
            raise RuntimeError(
                "Connector repeated terminal persistence lease IDs before release: "
                f"{sorted(duplicate_terminal)}"
            )
        self._pending_terminal_persistence_lease_ids.extend(terminal_ids)

    def synchronize_terminal_persistence_leases_before_allocation(self) -> None:
        """Recycle terminal staging slots at the next allocation boundary."""
        outstanding_count = self._rank_converged_outstanding_persistence_lease_count
        if outstanding_count == 0:
            return
        if self._pending_persistence_leases or self._persistence_tail_discard_pending:
            raise RuntimeError(
                "Terminal persistence convergence reached allocation with an "
                "unsubmitted persistence batch"
            )
        local_descriptors = self._pending_terminal_persistence_logical_descriptors()
        if len(local_descriptors) > outstanding_count:
            raise RuntimeError(
                "Terminal persistence descriptors exceed the rank-converged "
                f"outstanding lease count: terminal={len(local_descriptors)}, "
                f"outstanding={outstanding_count}"
            )
        rank_logical_descriptors = mpi_allgather(local_descriptors)
        if any(len(descriptors) > outstanding_count for descriptors in rank_logical_descriptors):
            raise RuntimeError(
                "A rank reported more terminal persistence descriptors than "
                f"the rank-converged outstanding lease count: ranks="
                f"{[len(descriptors) for descriptors in rank_logical_descriptors]}, "
                f"outstanding={outstanding_count}"
            )
        self._release_converged_terminal_persistence_prefix(rank_logical_descriptors)

    def _pending_terminal_persistence_logical_descriptors(
        self,
    ) -> List[Tuple[int, int]]:
        return [
            self._outstanding_persistence_lease_logical_descriptors[lease_id]
            for lease_id in self._pending_terminal_persistence_lease_ids
        ]

    def _release_converged_terminal_persistence_prefix(
        self, rank_logical_descriptors: List[List[Tuple[int, int]]]
    ) -> None:
        """Release only the terminal logical prefix visible on every TP rank."""
        if not rank_logical_descriptors:
            raise RuntimeError("Terminal persistence convergence requires at least one rank")
        common_count = min(len(value) for value in rank_logical_descriptors)
        if common_count == 0:
            return
        leader_prefix = rank_logical_descriptors[0][:common_count]
        for rank, descriptors in enumerate(rank_logical_descriptors[1:], start=1):
            logical_prefix = descriptors[:common_count]
            if logical_prefix != leader_prefix:
                raise RuntimeError(
                    "Terminal persistence prefixes diverged across ranks: "
                    f"rank={rank}, lengths="
                    f"{[len(value) for value in rank_logical_descriptors]}, "
                    f"leader={leader_prefix}, rank_value={logical_prefix}"
                )

        local_terminal_ids = self._pending_terminal_persistence_lease_ids[:common_count]
        self.complete_persistence_leases(local_terminal_ids)
        self._outstanding_persistence_lease_ids.difference_update(local_terminal_ids)
        for lease_id in local_terminal_ids:
            del self._outstanding_persistence_lease_logical_descriptors[lease_id]
        del self._pending_terminal_persistence_lease_ids[:common_count]
        self._rank_converged_outstanding_persistence_lease_count -= common_count

    def shutdown(self) -> None:
        if self._is_shutdown:
            return
        self._is_shutdown = True

        try:
            self.worker.shutdown()
        finally:
            if self.scheduler is not None:
                self.scheduler.shutdown()

    def _run_on_leader(self, f: Callable[[], Any]) -> Any:
        """
        Run a function on the leader rank, and broadcast the result to all other ranks.
        """
        if self.scheduler is not None:
            assert mpi_rank() == 0, "The scheduler may only exist on rank 0!"
            res = f()
        else:
            res = None
        return mpi_broadcast(res, root=0)

    def get_num_new_matched_tokens(self, request: LlmRequest, num_computed_tokens: int) -> int:
        if request.is_generation_only_request:
            raise RuntimeError("Connector API is not supported for generation-only requests!")

        if not request.multimodal_positions:
            request_num_tokens = request.get_num_tokens(0)
            if self.worker.can_skip_scheduler_match(request_num_tokens, num_computed_tokens):
                if self.scheduler is not None:
                    self.scheduler.prepare_scheduler_match_skip(request, num_computed_tokens)
                return 0

        num_tokens, load_kv_async = self._run_on_leader(
            lambda: self.scheduler.get_num_new_matched_tokens(request, num_computed_tokens)
        )

        if num_tokens == 0 and load_kv_async:
            raise RuntimeError("load_kv_async must be False when num_tokens is 0!")

        # TODO(jthomson04): This part is a bit ugly.
        # When the connector indicates that a request will be loaded
        # asynchronously, we need to suspend its execution. This is
        # problematic, since at the point when this function is called,
        # the request has already been scheduled! Because of this, we
        # need to remove it from our list of scheduled requests
        # (see `take_scheduled_requests_pending_load`).
        if load_kv_async:
            self.new_async_requests.loading[request.request_id] = request

        if self.scheduler is not None or self._uses_secondary_persistence_staging:
            self.scheduler_output_manager.record_new_matched_tokens(request, num_tokens)

        request.py_num_connector_matched_tokens = num_tokens

        return num_tokens

    def supports_schedulable_reuse_preview(self) -> bool:
        return self.worker.supports_schedulable_reuse_preview()

    def should_add_sequence(self, request: LlmRequest) -> bool:
        req_id = request.request_id
        return req_id not in self.finished_async_loading_requests

    def is_loading(self, request_id: int) -> bool:
        """Return whether a request is awaiting async-load completion."""
        return any(
            request_id in requests.loading
            for requests in (
                self.new_async_requests,
                self.pending_async_requests,
                self.local_finished_async_requests,
            )
        )

    def build_scheduler_output(
        self,
        scheduled_batch: ScheduledRequests,
        kv_cache_manager: "KVCacheManager",
        generation_progress_events_enabled: bool = False,
    ):
        self._metadata_exchange_included_dirty_request_ids.clear()
        self._skip_metadata_exchange = self._can_skip_metadata_exchange(
            scheduled_batch,
            kv_cache_manager,
            generation_progress_events_enabled,
        )
        if (
            self._skip_metadata_exchange
            and self._sparse_metadata_updates_enabled
            and not self._rank_local_state_only_pending
        ):
            # Leave SchedulerOutputManager's token/block cursor untouched so
            # the next visible boundary carries the complete accumulated delta.
            self._scheduler_output = None
            self._metadata_pending = False
            return

        self._metadata_pending = True
        if self.scheduler is not None or self._uses_secondary_persistence_staging:
            scheduler_output = self.scheduler_output_manager.build_scheduler_output(
                scheduled_batch,
                self.new_async_requests,
                kv_cache_manager,
                self._metadata_request_ids,
                self._incremental_persistence_identities_enabled
                and kv_cache_manager.supports_incremental_persistence_identities() is True,
            )
            if self.scheduler is not None:
                self._scheduler_output = scheduler_output
            self._metadata_exchange_included_dirty_request_ids.update(
                req.request_id
                for req in scheduled_batch.generation_requests
                if req.request_id in self._metadata_exchange_dirty_request_ids
                and (
                    self._metadata_request_ids is None
                    or req.request_id in self._metadata_request_ids
                )
            )

    def _can_skip_metadata_exchange(
        self,
        scheduled_batch: ScheduledRequests,
        kv_cache_manager: "KVCacheManager",
        generation_progress_events_enabled: bool = False,
    ) -> bool:
        """Return a conservative decision that is identical on replicated ranks."""
        self._metadata_request_ids = None
        self._rank_local_state_only_pending = False
        self._generation_progress_events_enabled = False
        if not self._rank_local_metadata_skip_enabled:
            return False

        self._generation_progress_events_enabled = bool(
            generation_progress_events_enabled
            and self._sparse_metadata_updates_enabled
            and self._uses_secondary_persistence_staging
        )
        has_non_generation_work = bool(
            scheduled_batch.encoder_requests
            or scheduled_batch.context_requests
            or scheduled_batch.paused_requests
        )
        has_generation_work = bool(scheduled_batch.generation_requests)
        can_use_generation_progress = (
            self._generation_progress_events_enabled
            and has_generation_work
            and not has_non_generation_work
            and self._generation_progress_fast_path_armed
        )
        if can_use_generation_progress:
            has_async_work = not (
                self.new_async_requests.is_empty
                and self.pending_async_requests.is_empty
                and self.local_finished_async_requests.is_empty
            )
            if not self._force_metadata_exchange and not has_async_work:
                if not self._metadata_exchange_dirty_request_ids:
                    return True
                self._metadata_request_ids = {
                    req.request_id
                    for req in scheduled_batch.generation_requests
                    if req.request_id in self._metadata_exchange_dirty_request_ids
                }
                if not self._metadata_request_ids:
                    # Boundary progress for a temporarily unscheduled request
                    # remains dirty until that request is scheduled again.
                    return True
                self._rank_local_state_only_pending = self._state_only_updates_enabled
                if self._rank_local_state_only_pending:
                    return True

        block_size = kv_cache_manager.tokens_per_block
        progress_changed = False
        changed_request_ids = set()
        active_request_ids = set()
        generation_request_ids = {req.request_id for req in scheduled_batch.generation_requests}

        for req in scheduled_batch.all_requests():
            request_id = req.request_id
            active_request_ids.add(request_id)
            num_tokens = req.get_num_tokens(0)

            if request_id in generation_request_ids:
                # The newest sampled token is input to the next forward and
                # therefore has no KV yet, except when the request is marked
                # to complete on this forward. That final forward must expose
                # a newly completed block because no later iteration will.
                completed_position = (
                    num_tokens
                    if req.state == LlmRequestState.GENERATION_TO_COMPLETE
                    else max(num_tokens - 1, 0)
                )
                completed_blocks = completed_position // block_size
                # Draft lookahead reserves device capacity but cannot produce a
                # worker transfer before those tokens are accepted into the
                # request. Accepted full blocks are therefore the only
                # worker-visible generation boundary.
                transfer_boundary = completed_blocks
                self._generation_computed_tokens[request_id] = completed_position
            else:
                completed_blocks = num_tokens // block_size
                next_position = req.context_current_position + min(
                    req.context_remaining_length, req.context_chunk_size
                )
                transfer_boundary = next_position // block_size
            # New device block IDs are consumed by the leader-only state
            # advance or retained for the next sparse delta. Workers need
            # metadata only once the block can produce a transfer or its
            # completed hash chain changes.
            progress = (completed_blocks, transfer_boundary)
            if self._metadata_exchange_progress.get(request_id) != progress:
                progress_changed = True
                changed_request_ids.add(request_id)
            self._metadata_exchange_progress[request_id] = progress

        for request_id in list(self._metadata_exchange_progress):
            if (
                request_id not in active_request_ids
                and request_id not in self._metadata_exchange_dirty_request_ids
            ):
                del self._metadata_exchange_progress[request_id]

        has_async_work = not (
            self.new_async_requests.is_empty
            and self.pending_async_requests.is_empty
            and self.local_finished_async_requests.is_empty
        )
        has_unseen_worker_request = any(
            request_id not in self._worker_visible_progress for request_id in active_request_ids
        )
        pure_generation_update = (
            not self._force_metadata_exchange
            and not has_non_generation_work
            and not has_async_work
            and not has_unseen_worker_request
        )
        if self._uses_secondary_persistence_staging and pure_generation_update and progress_changed:
            # Eviction staging creates no request-scoped decode Stores. Every
            # rank can therefore consume the same sparse cursor while only the
            # leader advances hashes and block identities. New/context work,
            # persistence leases, and async work all make
            # ``pure_generation_update`` false and retain the full exchange;
            # any leader-queued repair remains pending until that event.
            self._metadata_request_ids = changed_request_ids
            self._rank_local_state_only_pending = self._state_only_updates_enabled
        can_skip = pure_generation_update and (
            not progress_changed or self._rank_local_state_only_pending
        )
        if not can_skip:
            self._force_metadata_exchange = False
            for req in scheduled_batch.all_requests():
                self._worker_visible_progress[req.request_id] = self._metadata_exchange_progress[
                    req.request_id
                ]
        self._generation_progress_fast_path_armed = (
            self._generation_progress_events_enabled
            and has_generation_work
            and not has_non_generation_work
        )
        return can_skip

    def record_generation_progress(
        self,
        req: LlmRequest,
        kv_cache_manager: "KVCacheManager",
        rewind_required: bool,
        computed_tokens: int,
    ) -> None:
        """Record a generation boundary found in an already-visited request loop."""
        if not self._generation_progress_events_enabled:
            if rewind_required:
                self._on_rewind(req, kv_cache_manager, computed_tokens + 1)
            return

        request_id = req.request_id
        if request_id not in self._generation_computed_tokens:
            self._generation_progress_fast_path_armed = False
            self._force_metadata_exchange = True
            if rewind_required:
                self._on_rewind(req, kv_cache_manager, computed_tokens + 1)
            return

        self._generation_computed_tokens[request_id] = computed_tokens
        completed_blocks = computed_tokens // kv_cache_manager.tokens_per_block
        progress = (completed_blocks, completed_blocks)
        if self._metadata_exchange_progress.get(request_id) != progress:
            self._metadata_exchange_dirty_request_ids.add(request_id)
        self._metadata_exchange_progress[request_id] = progress

        if rewind_required:
            self._on_rewind(req, kv_cache_manager, computed_tokens + 1)

    def on_generation_will_complete(self) -> None:
        """Force one exact scan so the final forward can publish its KV block."""
        self._generation_progress_fast_path_armed = False

    def on_rewind(self, req: LlmRequest, kv_cache_manager: "KVCacheManager"):
        """Notify the connector that a request's KV cache was rewound.

        Trims the Python-side ``scheduler_output_manager`` bookkeeping and
        notifies the external scheduler (e.g. KVBM leader) so it can trim
        its own per-request slot state.
        """
        num_tokens = req.get_num_tokens(0)
        self._on_rewind(req, kv_cache_manager, num_tokens)

    def _on_rewind(
        self,
        req: LlmRequest,
        kv_cache_manager: "KVCacheManager",
        num_tokens: int,
    ) -> None:
        # Native rewind already treats an overlap batch for a released TRT
        # sequence as a no-op. Mirror that lifecycle boundary before querying
        # the sequence for connector indices.
        if not kv_cache_manager.has_active_sequence(req.py_request_id):
            return

        block_size = kv_cache_manager.tokens_per_block
        # The final live token is the next-forward input and has no KV yet.
        completed_blocks = max(num_tokens - 1, 0) // block_size
        progress = (completed_blocks, completed_blocks)
        self._metadata_exchange_progress[req.request_id] = progress
        self._generation_computed_tokens[req.request_id] = max(num_tokens - 1, 0)
        # Progress events already dirtied any newly visible full-block boundary.
        # A rejected suffix that cannot trim an acknowledged prefix needs no
        # block-ID materialization or connector callback.
        if (
            self._generation_progress_events_enabled
            and self._sparse_metadata_updates_enabled
            and not self.scheduler_output_manager.rewind_may_affect_connector_state(
                req.request_id, num_tokens, block_size
            )
        ):
            # Speculative suffixes are invisible until an accepted full-block
            # boundary. Rewinding only that suffix cannot change connector state.
            return
        worker_visible_progress = self._worker_visible_progress.get(req.request_id)
        rewind_crossed_worker_boundary = worker_visible_progress != progress
        self._force_metadata_exchange |= rewind_crossed_worker_boundary
        live_block_ids = kv_cache_manager.get_connector_cache_indices(req)
        live_block_object_ids = kv_cache_manager.get_cache_indices(req)
        scheduler_live_block_ids = None
        if self.scheduler is not None or self._uses_secondary_persistence_staging:
            scheduler_live_block_ids = self.scheduler_output_manager.on_rewind(
                req, live_block_ids, live_block_object_ids, num_tokens
            )
        if self.scheduler is not None:
            self.scheduler.on_rewind(
                req,
                live_block_ids if scheduler_live_block_ids is None else scheduler_live_block_ids,
            )

    def take_scheduled_requests_pending_load(self, scheduled_requests: ScheduledRequests):
        """
        Remove context requests from our list of scheduled requests that are being loaded asynchronously.
        This is done to prevent the runtime from attempting to load the KV cache for these requests.

        Args:
            scheduled_requests: The scheduled requests.

        Returns:
            The scheduled requests with the context requests that are being loaded asynchronously removed.
        """

        for key in ["context_requests_chunking", "context_requests_last_chunk"]:
            allowed_context_requests = []
            for req in getattr(scheduled_requests, key):
                # If this request is being loaded asynchronously, in
                # addition to removing it from the list of scheduled
                # requests, we also need to update its state.
                if req.request_id in self.new_async_requests.loading.keys():
                    req.state = LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS

                    # Replace the request with the canonical request.
                    self.new_async_requests.loading[req.request_id] = req
                else:
                    allowed_context_requests.append(req)
            setattr(scheduled_requests, key, allowed_context_requests)

    def handle_metadata(self) -> object:
        if not self._metadata_pending:
            return

        metadata_exchange_skipped = self._skip_metadata_exchange
        worker_metadata_available = False
        if metadata_exchange_skipped:
            if self._pending_persistence_leases:
                raise RuntimeError(
                    "Metadata exchange cannot be skipped with pending persistence leases"
                )
            if self.scheduler is not None:
                self.scheduler.advance_without_worker_metadata(self._scheduler_output)
            metadata = None
        else:

            def build_exchange():
                def build_update() -> ConnectorUpdate:
                    if self._state_only_updates_enabled:
                        return self.scheduler.build_connector_update(self._scheduler_output)
                    return ConnectorWorkerMetadata(
                        self.scheduler.build_connector_meta(self._scheduler_output)
                    )

                try:
                    leases = self._pending_persistence_leases
                    if not leases:
                        update = build_update()
                        return True, update, None, None, None
                    descriptors = self._persistence_lease_descriptors(leases)
                    if self._drop_pending_persistence_reason is not None:
                        update = build_update()
                        return (
                            True,
                            update,
                            descriptors,
                            None,
                            self._drop_pending_persistence_reason,
                        )
                    if self._persistence_resolution_error is not None:
                        error_type, error_message = self._persistence_resolution_error
                        discardable_identity_errors = (
                            "persistence hash mismatch",
                            "no canonical G2PB persistence key is attached to TRT block",
                        )
                        if any(marker in error_message for marker in discardable_identity_errors):
                            update = build_update()
                            return True, update, descriptors, None, error_message
                        return False, error_type, error_message, None, None
                    persistence_keys = self._resolved_persistence_keys
                    if persistence_keys is None:
                        raise RuntimeError(
                            "Leader did not snapshot persistence keys when leases were created"
                        )
                    update = build_update()
                    return True, update, descriptors, persistence_keys, None
                # pyo3 PanicException inherits BaseException. Convert every
                # leader failure into data so peer ranks reach the broadcast.
                except BaseException as error:
                    return False, type(error).__name__, str(error), None, None

            exchange = self._run_on_leader(build_exchange)
            (
                exchange_ok,
                update,
                leader_descriptors,
                persistence_keys,
                drop_persistence_reason,
            ) = exchange
            if not exchange_ok:
                raise RuntimeError(
                    "Connector leader metadata or persistence resolution failed: "
                    f"{update}: {leader_descriptors}"
                )

            if self._uses_secondary_persistence_staging:
                local_descriptors = self._persistence_lease_descriptors(
                    self._pending_persistence_leases
                )
                rank_descriptors = mpi_allgather(local_descriptors)
                if leader_descriptors is None:
                    if any(rank_descriptors):
                        leader_descriptors, persistence_keys = (
                            self._converge_persistence_lease_prefix(
                                rank_descriptors,
                                [],
                                [],
                            )
                        )
                    self._resolved_persistence_keys = None
                elif persistence_keys is None:
                    # Identity resolution failed on the leader, so nothing in this
                    # batch is publishable. Every rank drops its complete local
                    # batch after the shared D2H safety barrier. C++ canonicalizes
                    # free secondary-slot order after the rank-local releases.
                    if drop_persistence_reason is None:
                        raise RuntimeError("Unpublishable persistence batch has no failure reason")
                    self._resolved_persistence_keys = None
                    self._drop_pending_persistence_reason = drop_persistence_reason
                else:
                    leader_descriptors, persistence_keys = self._converge_persistence_lease_prefix(
                        rank_descriptors,
                        leader_descriptors,
                        persistence_keys,
                    )
            if isinstance(update, ConnectorWorkerMetadata):
                metadata = update.metadata
                worker_metadata_available = True
            elif isinstance(update, ConnectorStateOnly):
                metadata = None
                worker_metadata_available = False
            else:
                raise RuntimeError(
                    "Connector scheduler returned an unsupported update type: "
                    f"{type(update).__name__}"
                )

        processed_request_ids = set(self._metadata_exchange_included_dirty_request_ids)
        self._scheduler_output = None
        self._metadata_pending = False
        self._skip_metadata_exchange = False
        self._metadata_request_ids = None
        self._metadata_exchange_included_dirty_request_ids.clear()
        self._rank_local_state_only_pending = False

        self._metadata_exchange_dirty_request_ids.difference_update(processed_request_ids)

        if worker_metadata_available:
            self.worker.bind_connector_meta(metadata)
            self._worker_batch_hooks_pending = True

    def start_worker_batch(self, scheduled_requests: ScheduledRequests) -> bool:
        """Run metadata-driven worker hooks for the next forward pass."""
        if not self._worker_batch_hooks_pending:
            self._worker_batch_hooks_active = False
            return False

        self.take_scheduled_requests_pending_load(scheduled_requests)
        self.worker.start_load_kv(torch.cuda.current_stream())
        self._worker_batch_hooks_pending = False
        self._worker_batch_hooks_active = True
        return True

    def request_finished(self, req: LlmRequest, cache_block_ids: List[int]) -> bool:
        """
        Called when a request is finished generating tokens.

        Args:
            req: The request that finished generating tokens.

        Returns:
            Whether the request is performing asynchronous saving
            operations. If true, we do not immediately call
            free_resources on the request.
        """

        if self.uses_secondary_kv_pool_as_persistence_staging():
            # An empty block list prevents request completion from creating a
            # new eager save, while normal finalization still preserves Store
            # operations dispatched during prefill. If none were dispatched,
            # retire the worker slot immediately through the no-save path.
            saving_async = self._run_on_leader(lambda: self.scheduler.request_finished(req, []))
            if not saving_async:
                self.worker.request_finished_without_save(req.request_id)
                self.scheduler_output_manager.requests.pop(req.request_id, None)
                self.scheduler_output_manager.external_loads.pop(req.request_id, None)
        else:
            saving_async = self._run_on_leader(
                lambda: self.scheduler.request_finished(req, cache_block_ids)
            )

        if req.request_id in self.finished_async_loading_requests:
            del self.finished_async_loading_requests[req.request_id]
        self._metadata_exchange_progress.pop(req.request_id, None)
        self._worker_visible_progress.pop(req.request_id, None)
        self._generation_computed_tokens.pop(req.request_id, None)
        self._metadata_exchange_dirty_request_ids.discard(req.request_id)

        # This is similar to take_scheduled_requests_pending_load.
        # We need to update the request's state to indicate that it's still being used, but isn't schedulable.
        if saving_async:
            # Synchronous finish was already broadcast above and has no
            # follow-up worker metadata. Async save keeps worker lifecycle
            # state active and therefore retains the conservative full path.
            self._force_metadata_exchange = True
            self.new_async_requests.saving[req.request_id] = req
            req.state = LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS

        return saving_async

    def get_finished(self) -> List[LlmRequest]:
        """
        Process requests that have finished loading and saving.

        Returns:
            The requests that have newly finished saving.
        """
        # Admission/finalization decisions are broadcast, and a rank that
        # finishes early retains the request locally until every rank agrees.
        # Empty state is therefore rank-consistent and needs no collective.
        if (
            self.new_async_requests.is_empty
            and self.pending_async_requests.is_empty
            and self.local_finished_async_requests.is_empty
        ):
            return []

        started_loading_req_ids = list(self.new_async_requests.loading_ids)
        finished_gen_req_ids = list(self.new_async_requests.saving_ids)

        # Add the requests to our list of outstanding (still in progress)
        # requests.
        self.pending_async_requests.add_from(self.new_async_requests)

        # Pass these newly finished requests into get_finished, and get
        # the list of requests that have finished saving and loading.
        (finished_saving, finished_loading) = self.worker.get_finished(
            finished_gen_req_ids, started_loading_req_ids
        )

        # Remove the requests from our pending list that have finished locally.
        # A connector report for an ID no longer pending is a contract
        # violation; surface it loudly but do not crash the event loop.
        unknown_saving = [i for i in finished_saving if i not in self.pending_async_requests.saving]
        unknown_loading = [
            i for i in finished_loading if i not in self.pending_async_requests.loading
        ]
        if unknown_saving or unknown_loading:
            logger.warning(
                f"[rank {mpi_rank()}] KV connector reported IDs that are not pending: "
                f"saving={unknown_saving} loading={unknown_loading} "
                f"pending_saving={sorted(self.pending_async_requests.saving)} "
                f"pending_loading={sorted(self.pending_async_requests.loading)} "
                f"local_loading={sorted(self.local_finished_async_requests.loading)}"
            )
            finished_saving = [i for i in finished_saving if i not in unknown_saving]
            finished_loading = [i for i in finished_loading if i not in unknown_loading]
        new_local_finished_async_requests = self.pending_async_requests.extract_by_id(
            finished_saving, finished_loading
        )

        # Add these requests to our list of locally finished requests.
        self.local_finished_async_requests.add_from(new_local_finished_async_requests)

        # Broadcast this whole list to all other workers.
        finished_saving = list(self.local_finished_async_requests.saving_ids)
        finished_loading = list(self.local_finished_async_requests.loading_ids)

        local_finalization_status = (
            self.worker.get_load_finalization_status()
            if self.worker.requires_load_finalization is True
            else None
        )
        all_results = mpi_allgather((finished_saving, finished_loading, local_finalization_status))

        # Find only the requests that have been reported complete by all workers.
        intersect_finished_saving = set.intersection(*[set(res[0]) for res in all_results])
        intersect_finished_loading = set.intersection(*[set(res[1]) for res in all_results])

        finalization_statuses = [result[2] for result in all_results]
        finalization_skew_key = None
        if any(status is not None for status in finalization_statuses):
            if any(status is None for status in finalization_statuses):
                # Normal scheduling skew: a rank can observe the wire-ordered
                # queue head before every rank has locally announced the load.
                # Wait for convergence; genuine divergence trips the bounded
                # timeout below.
                known_ids = sorted(s[0] for s in finalization_statuses if s is not None)
                finalization_skew_key = ("presence", known_ids[0])
            else:
                request_ids = {status[0] for status in finalization_statuses}
                if len(request_ids) != 1:
                    finalization_skew_key = ("order", min(request_ids))
                else:
                    request_id = next(iter(request_ids))
                    preparation_states = [status[1] for status in finalization_statuses]
                    unknown_states = set(preparation_states) - {"pending", "ready", "failed"}
                    if unknown_states:
                        raise RuntimeError(
                            "KV connector returned invalid load-finalization states: "
                            f"{sorted(unknown_states)}"
                        )
                    if "failed" in preparation_states:
                        self._armed_load_finalizations[request_id] = True
                    elif all(state == "ready" for state in preparation_states):
                        if request_id in intersect_finished_loading:
                            if self._finalization_skew_since:
                                skew_key, skew_since = next(
                                    iter(self._finalization_skew_since.items())
                                )
                                logger.warning(
                                    f"[rank {mpi_rank()}] KV connector finalization skew converged "
                                    f"for {skew_key} after {time.monotonic() - skew_since:.3f}s; "
                                    f"arming request {request_id}"
                                )
                            self._armed_load_finalizations[request_id] = False
                        else:
                            # Every rank prepared the payload but at least one
                            # manager has not yet accepted the load locally;
                            # the connector re-reports until it converges.
                            finalization_skew_key = ("intersect", request_id)
        if finalization_skew_key is None:
            self._finalization_skew_since.clear()
        else:
            self._note_finalization_skew(finalization_skew_key, finalization_statuses)

        # Keep an armed load out of CONTEXT_INIT until its payload collective
        # has completed at the next executor-iteration boundary.
        intersect_finished_loading -= set(self._armed_load_finalizations)

        # Remove these requests from our list of locally finished requests.
        all_finished = self.local_finished_async_requests.extract_by_id(
            intersect_finished_saving, intersect_finished_loading
        )

        # Scheduler-output cursors retain the complete prompt token list. They
        # are no longer needed once every rank has completed an asynchronous
        # save, so release them with the request instead of retaining every
        # finished prompt for the lifetime of the engine.
        for request_id in all_finished.saving:
            self.scheduler_output_manager.requests.pop(request_id, None)
            self.scheduler_output_manager.external_loads.pop(request_id, None)

        # For requests that have finished loading, move them back to the context state.
        for id, req in all_finished.loading.items():
            req.state = LlmRequestState.CONTEXT_INIT
            self.finished_async_loading_requests[id] = req

        # Return the requests that have finished saving.
        # The execution loop will call _terminate_request on these requests.
        return list(all_finished.saving.values())

    # A load that stays skewed this long is treated as true divergence and the
    # worker fail-stops. All ranks act on the same allgathered inputs, so any
    # rank that raises brings the replica down before NCCL can wedge.
    _FINALIZATION_SKEW_TIMEOUT_S = 120.0

    def _note_finalization_skew(self, key, statuses) -> None:
        now = time.monotonic()
        since = self._finalization_skew_since.get(key)
        if since is None:
            # Inherit the earliest outstanding timestamp so skew that
            # oscillates between kinds (presence/order/intersect) cannot
            # restart the clock and evade the global deadline.
            since = min(self._finalization_skew_since.values(), default=now)
            self._finalization_skew_since.clear()
            self._finalization_skew_since[key] = since
            logger.warning(
                f"[rank {mpi_rank()}] KV connector finalization skew began for {key}: "
                f"statuses={statuses} "
                f"local_loading={sorted(self.local_finished_async_requests.loading)} "
                f"pending_loading={sorted(self.pending_async_requests.loading)} "
                f"new_loading={sorted(self.new_async_requests.loading)} "
                f"armed={dict(self._armed_load_finalizations)}"
            )
        if now - since > self._FINALIZATION_SKEW_TIMEOUT_S:
            raise RuntimeError(
                "KV connector load-finalization consensus did not converge for "
                f"{key} within {self._FINALIZATION_SKEW_TIMEOUT_S:.0f}s: {statuses}"
            )

    def finalize_armed_loads(self, stream: torch.cuda.Stream) -> None:
        """Finalize TP-consensed loads before scheduling the next forward."""
        if not self._armed_load_finalizations:
            return

        request_ids = sorted(self._armed_load_finalizations)
        failed_request_ids = [
            request_id for request_id in request_ids if self._armed_load_finalizations[request_id]
        ]
        stream_sync_error = None
        try:
            stream.synchronize()
        except RuntimeError as error:
            stream_sync_error = repr(error)
        stream_sync_errors = mpi_allgather(stream_sync_error)
        if any(error is not None for error in stream_sync_errors):
            raise RuntimeError(
                "KV connector could not quiesce the prior TRT iteration on every TP rank: "
                f"{stream_sync_errors}"
            )

        self.worker.finalize_finished_loads(request_ids, stream)
        if failed_request_ids:
            raise RuntimeError(
                f"KV connector load preparation failed for requests {failed_request_ids}"
            )

        finalized = self.local_finished_async_requests.extract_by_id(set(), set(request_ids))
        if set(finalized.loading) != set(request_ids):
            raise RuntimeError(
                "KV connector finalized loads missing from the local completion set: "
                f"expected={request_ids}, actual={sorted(finalized.loading)}"
            )
        for request_id, req in finalized.loading.items():
            req.state = LlmRequestState.CONTEXT_INIT
            self.finished_async_loading_requests[request_id] = req
        self._armed_load_finalizations.clear()

    def update_state_after_alloc(self, req: LlmRequest, block_ids: List[int]):
        self._generation_progress_fast_path_armed = False
        if self.scheduler is not None:
            self.scheduler.update_state_after_alloc(req, block_ids)

    def register_persistence_identities_after_alloc(
        self, req: LlmRequest, kv_cache_manager: "KVCacheManager"
    ) -> None:
        """Register a new context allocation before ``refresh_blocks``.

        Every TP rank records the same physical-block identity locally so
        future persistence leases remain rank-consistent. Only the leader
        forwards the identity mapping into connector scheduler state.
        """
        if not self._uses_secondary_persistence_staging:
            return
        block_object_ids = kv_cache_manager.get_cache_indices(req)
        block_hashes = kv_cache_manager.commit_and_get_block_hashes(req)
        if self.scheduler is not None:
            self.scheduler.register_persistence_identities(
                req,
                block_object_ids,
                block_hashes,
            )

    def set_scheduler_output(self, scheduler_output: SchedulerOutput):
        self._scheduler_output = scheduler_output
        self._metadata_pending = True
        self._skip_metadata_exchange = False

    def layer_pre_hook(self, module, *args):
        if not self._worker_batch_hooks_active:
            return
        self.worker.wait_for_layer_load(module.layer_idx, torch.cuda.current_stream())

    def layer_post_hook(self, module, *args):
        if not self._worker_batch_hooks_active:
            return
        self.worker.save_kv_layer(module.layer_idx, torch.cuda.current_stream())

    def wait_for_initialization(self):
        if self.scheduler is not None:
            self.scheduler.wait_for_initialization()
