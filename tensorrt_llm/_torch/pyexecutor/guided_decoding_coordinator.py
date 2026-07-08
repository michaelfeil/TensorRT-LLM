# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Executor-side orchestration of asynchronous guided-decoding compilation."""

import os
import time
from typing import Dict, List, Optional, Set, Tuple

from ...logger import logger
from ..distributed import Distributed
from .grammar_compiler import CompileState, compile_timeout_message
from .guided_decoder import GuidedDecoder
from .llm_request import LlmRequest
from .scheduler import ScheduledRequests


class GuidedDecodingCoordinator:
    """Coordinates guided-decoding grammar compilation with the executor loop.

    Grammar compilation is started at request activation (overlapping prefill
    and KV-cache transfer) and requests are deferred from scheduling until it
    finishes, so that matcher attach — which with one-model speculative
    decoding runs inside CUDA host callbacks — never has to compile.

    Two invariants drive the design:
    1. CUDA-callback safety: nothing on the forward path may run an unbounded
       grammar compile or raise; failures are recorded by the decoder and
       drained here into per-request error responses.
    2. Rank consistency: on ranks that share batch composition (non-attention-
       DP TP), defer verdicts and failure terminations must be identical on
       every rank at the same iteration, or batch composition diverges and
       ranks deadlock in collectives.

    Known residual risk to invariant 2: a non-deciding rank whose compile
    lags past the matcher-attach wait fails rank-locally, diverging batches.

    All methods no-op when guided decoding is disabled (decoder is None).
    """

    def __init__(
        self, decoder: Optional[GuidedDecoder], dist: Distributed, enable_attention_dp: bool
    ):
        self.decoder = decoder
        self.dist = dist
        self.enable_attention_dp = enable_attention_dp
        # Max deferral, anchored at first deferral so compile time overlapping
        # prefill / KV transfer is not charged. Deferred disagg-gen requests
        # hold KV and transfer accounting, so this must stay well below the
        # KV-transfer timeouts.
        self._defer_timeout_s = float(os.getenv("TRTLLM_GUIDED_DEFER_TIMEOUT_SEC", "5"))
        # request_id -> monotonic timestamp of its first deferral.
        self._deferred_since: Dict[int, float] = {}
        # Failures drained from CUDA-callback records, awaiting termination
        # at the post-sampling point (see drain_failures/take_failures).
        self._pending_failures: Dict[int, str] = {}

    def start_compiles(self, requests: List[LlmRequest]) -> None:
        """Kick off grammar compilation for newly activated guided requests."""
        if self.decoder is None:
            return
        for request in requests:
            if request.guided_decoding_params is not None:
                self.decoder.grammar_compiler.submit(
                    request.py_request_id, request.guided_decoding_params
                )
                # Eager flag: a rank whose only request is fresh pads an ADP
                # dummy instead of stalling the group; may over-pad one dummy
                # for one iteration on multi-chunk context requests.
                request.py_guided_compile_pending = True

    def filter_schedulable(
        self, context_requests: List[LlmRequest], generation_requests: List[LlmRequest]
    ) -> Tuple[List[LlmRequest], List[LlmRequest]]:
        """Defer scheduling of guided requests whose grammar is still compiling.

        Deferred requests stay in their current state and are naturally
        re-scheduled on a later iteration; failed compiles are scheduled and
        turn into per-request error responses at matcher attach.
        """
        if self.decoder is None:
            return context_requests, generation_requests
        # The defer decision is time-dependent, so ranks sharing batch
        # composition must agree: ADP requests are rank-local, while non-ADP
        # TP ranks schedule the same replicated requests, so rank 0 decides
        # and broadcasts.
        rank_local = self.enable_attention_dp or self.dist.tp_size == 1
        if not rank_local and self.dist.pp_size > 1:
            # Under PP only the scheduling rank gets here, so the broadcast
            # below would be unpaired; fall back to no gating (the bounded
            # matcher-attach wait applies instead).
            deferred_ids: Set = frozenset()
        elif rank_local:
            deferred_ids, _ = self._classify(context_requests, generation_requests)
        elif not any(self._gates_scheduling(r, True) for r in context_requests) and not any(
            self._gates_scheduling(r, False) for r in generation_requests
        ):
            # Request state is replicated, so every rank sees the same
            # (absent) gated set and skips the broadcast together.
            deferred_ids = frozenset()
        else:
            # A request rank 0 failed must fail attach on every rank in the
            # same iteration, so non-root ranks force their local compile
            # state to failed rather than trusting their own.
            payload = (
                self._classify(context_requests, generation_requests)
                if self.dist.tp_rank == 0
                else None
            )
            deferred_ids, failed_ids = self.dist.tp_broadcast(payload, root=0)
            if self.dist.tp_rank != 0:
                for req_id in failed_ids:
                    self.decoder.grammar_compiler.mark_failed(
                        req_id, compile_timeout_message(self._defer_timeout_s)
                    )

        return (
            self._keep_schedulable(context_requests, deferred_ids),
            self._keep_schedulable(generation_requests, deferred_ids),
        )

    def drain_failures(
        self, synced_batch: Optional[ScheduledRequests], active_requests: List[LlmRequest]
    ) -> None:
        """Collect guided-decoding failures recorded on CUDA-callback threads.

        Must be called with a just-synchronized batch: the sync guarantees
        this rank has executed that batch's callbacks, so restricting the
        drain to it yields an identical failure set on every rank. A record
        with request_id None fails every guided request of the batch.

        Failures are parked, not terminated: with the overlap scheduler the
        request is still in flight and freeing its resources mid-iteration
        wedges the executor. Termination happens via take_failures.
        """
        if self.decoder is None or synced_batch is None:
            return
        if not self.decoder.has_async_failures():
            # Common case; skip building the batch/active id sets.
            return
        batch_requests = list(synced_batch.all_requests())
        failures = self.decoder.drain_async_failures(
            allowed_req_ids={r.py_request_id for r in batch_requests},
            active_req_ids={r.py_request_id for r in active_requests},
        )
        if not failures:
            return
        guided_ids = {
            r.py_request_id for r in batch_requests if r.guided_decoding_params is not None
        }
        for req_id, error_msg in failures:
            if req_id is None:
                for rid in guided_ids:
                    self._pending_failures.setdefault(rid, error_msg)
            else:
                self._pending_failures.setdefault(req_id, error_msg)

    def take_failures(
        self, scheduled_batch: ScheduledRequests, active_requests: List[LlmRequest]
    ) -> Dict[int, str]:
        """Pop parked failures for requests in the given batch.

        Entries for requests that already finished through another path are
        dropped; entries for still-active requests outside this batch are
        kept for a later call.
        """
        if not self._pending_failures:
            return {}
        batch_ids = {r.py_request_id for r in scheduled_batch.all_requests()}
        active_ids = {r.py_request_id for r in active_requests}
        failures = {}
        for req_id in list(self._pending_failures):
            if req_id in batch_ids:
                failures[req_id] = self._pending_failures.pop(req_id)
                logger.error(
                    f"Terminating request {req_id} due to guided "
                    f"decoding failure: {failures[req_id]}"
                )
            elif req_id not in active_ids:
                del self._pending_failures[req_id]
        return failures

    def release(self, request: LlmRequest) -> None:
        """Drop compile state for a finished/canceled request."""
        if self.decoder is not None:
            self.decoder.grammar_compiler.discard(request.py_request_id)
            self._deferred_since.pop(request.py_request_id, None)

    def shutdown(self) -> None:
        """Stop the compile pool; safe to call multiple times."""
        if self.decoder is not None:
            self.decoder.grammar_compiler.shutdown()

    def _gates_scheduling(self, request: LlmRequest, is_context: bool) -> bool:
        # Only matcher-attach steps need the compile: the last context chunk
        # and the first disagg-gen step. The step check comes first because
        # reading guided_decoding_params copies the schema out of the C++
        # request.
        if is_context:
            if not (request.is_context_init_state and request.is_last_context_chunk):
                return False
        elif not request.is_disagg_generation_transmission_complete:
            return False
        return request.guided_decoding_params is not None

    def _classify(
        self, context_requests: List[LlmRequest], generation_requests: List[LlmRequest]
    ) -> Tuple[Set, Set]:
        """Split gated requests into deferred (still compiling) and failed.

        Runs only on the deciding rank. Requests deferred past the budget are
        force-failed so matcher attach errors out instead of waiting.
        """
        deferred, failed = set(), set()
        for is_context, requests in ((True, context_requests), (False, generation_requests)):
            for request in requests:
                if not self._gates_scheduling(request, is_context):
                    continue
                req_id = request.py_request_id
                state = self.decoder.grammar_compiler.poll(req_id)
                if state == CompileState.PENDING:
                    deferred_since = self._deferred_since.setdefault(req_id, time.monotonic())
                    if time.monotonic() - deferred_since > self._defer_timeout_s:
                        self.decoder.grammar_compiler.mark_failed(
                            req_id, compile_timeout_message(self._defer_timeout_s)
                        )
                        failed.add(req_id)
                    else:
                        deferred.add(req_id)
                        continue
                elif state == CompileState.FAILED:
                    failed.add(req_id)
                self._deferred_since.pop(req_id, None)
        return deferred, failed

    def _keep_schedulable(self, requests: List[LlmRequest], deferred_ids: Set) -> List[LlmRequest]:
        if not deferred_ids:
            # Common case: nothing deferred; clear stale flags in place.
            for request in requests:
                request.py_guided_compile_pending = False
            return requests
        kept = []
        for request in requests:
            deferred = request.py_request_id in deferred_ids
            request.py_guided_compile_pending = deferred
            if not deferred:
                kept.append(request)
        return kept
