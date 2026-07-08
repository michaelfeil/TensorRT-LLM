import math
import traceback
from collections import deque
from dataclasses import dataclass
from queue import Queue
from typing import Iterable, List, Optional, Tuple

import torch

from tensorrt_llm.llmapi.llm_args import GuidedDecodingConfig

from ..._utils import nvtx_range, prefer_pinned
from ...bindings.executor import GuidedDecodingParams
from ...bindings.internal.batch_manager import LlmRequestType
from ...logger import logger
from ..hostfunc import hostfunc
from .grammar_compiler import AsyncGrammarCompiler, format_guided_error
from .grammar_matcher import (GrammarMatcher, LLGuidanceMatcherFactory,
                              XGrammarMatcherFactory)
from .llm_request import LlmRequest
from .scheduler import ScheduledRequests


@dataclass(slots=True)
class GuidedRequest:
    """A snapshot of an LlmRequest that contains relevant fields for guided decoding.

    The instances of this class should be produced on the host and consumed on the device (by hostfunc).
    """
    guided_decoding_params: Optional[GuidedDecodingParams] = None

    request_id: Optional[int] = None
    seq_slot: Optional[int] = None
    prev_seq_slot: Optional[int] = None
    is_context_init_state: bool = False
    is_last_context_chunk: bool = False
    is_generation_in_progress_state: bool = False
    is_generation_only_first_iteration: bool = False

    new_token: Optional[int] = None
    is_draft: bool = False
    draft_tokens: Optional[List[int]] = None
    num_accepted_draft_tokens: Optional[int] = None

    def require_matcher_init(self) -> bool:
        if self.guided_decoding_params is None:
            return False
        if self.is_draft:
            return False
        # The request is in the last chunk of a context forward step.
        return self.is_context_init_state and self.is_last_context_chunk

    def require_matcher_advance(self) -> bool:
        if self.guided_decoding_params is None:
            return False
        if self.is_draft:
            if self.is_context_init_state and self.is_last_context_chunk:
                return True
            if self.is_generation_in_progress_state:
                return True
            return False
        # The request is in a generation forward step.
        return self.is_generation_in_progress_state

    @classmethod
    def from_llm_request(cls, request: LlmRequest):
        return cls(
            guided_decoding_params=request.guided_decoding_params,
            request_id=request.py_request_id,
            seq_slot=(request.py_target_seq_slot
                      if request.py_is_draft else request.py_seq_slot),
            prev_seq_slot=request.py_batch_idx,
            is_context_init_state=request.is_context_init_state,
            is_last_context_chunk=(request.is_context_init_state
                                   and request.is_last_context_chunk),
            is_generation_in_progress_state=request.
            is_generation_in_progress_state,
            is_generation_only_first_iteration=(
                request.llm_request_type
                == LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY
                and request.py_decoding_iter == 1
                and request.py_batch_idx is None),
            new_token=request.get_last_tokens(0),
            is_draft=request.py_is_draft,
            draft_tokens=request.py_draft_tokens,
            num_accepted_draft_tokens=request.py_num_accepted_draft_tokens)

    def cast_to_draft(self) -> None:
        self.is_draft = True
        self.draft_tokens = []


@dataclass(slots=True)
class GuidedRequests:
    requests: List[GuidedRequest]
    num_contexts: int
    num_generations: int
    max_num_draft_tokens: int

    @classmethod
    def from_scheduled_requests(cls,
                                scheduled_requests: ScheduledRequests,
                                max_num_draft_tokens: int = 0):
        requests = [
            GuidedRequest.from_llm_request(req)
            for req in scheduled_requests.all_requests()
        ]
        return cls(requests,
                   num_contexts=scheduled_requests.num_context_requests,
                   num_generations=scheduled_requests.num_generation_requests,
                   max_num_draft_tokens=max_num_draft_tokens)

    @property
    def num_bitmask_tokens(self) -> int:
        if self.requests[0].is_draft:
            return len(self.requests)
        else:
            return self.num_contexts + self.num_generations * (
                self.max_num_draft_tokens + 1)

    def valid_requests_with_offsets(
            self) -> Iterable[Tuple[GuidedRequest, int]]:
        offset: int = 0
        for req in self.requests:
            if req.guided_decoding_params is not None and req.seq_slot is not None:
                yield req, offset
            offset += 1
            if not req.is_draft and req.is_generation_in_progress_state:
                offset += self.max_num_draft_tokens

    def valid_requests(self) -> Iterable[GuidedRequest]:
        for req in self.requests:
            if req.guided_decoding_params is not None and req.seq_slot is not None:
                yield req

    def __iter__(self) -> Iterable[GuidedRequest]:
        return iter(self.requests)

    def __len__(self) -> int:
        return len(self.requests)


class GuidedDecoder:
    bitmask_dtype = torch.int32
    token_mask_dtype = torch.int32
    # Whether matcher attach may compile synchronously when no precompiled
    # grammar exists; must be False when attach runs in CUDA host callbacks.
    _allow_sync_compile = True

    def __init__(self,
                 guided_decoding_config: GuidedDecodingConfig,
                 max_num_sequences: int,
                 vocab_size_padded: int,
                 max_num_draft_tokens: int = 0,
                 rank: int = 0,
                 tokenizer=None):
        self.guided_decoding_backend = guided_decoding_config.backend
        self.max_num_sequences = max_num_sequences
        self.vocab_size_padded = vocab_size_padded
        self.max_num_draft_tokens = max_num_draft_tokens
        self.rank = rank

        if self.guided_decoding_backend == GuidedDecodingConfig.GuidedDecodingBackend.XGRAMMAR:
            self.grammar_matcher_factory = XGrammarMatcherFactory(
                guided_decoding_config,
                vocab_size_padded,
                max_num_draft_tokens=max_num_draft_tokens,
                tokenizer=tokenizer)
        elif self.guided_decoding_backend == GuidedDecodingConfig.GuidedDecodingBackend.LLGUIDANCE:
            self.grammar_matcher_factory = LLGuidanceMatcherFactory(
                guided_decoding_config, vocab_size_padded)
        else:
            raise ValueError(
                f"Invalid guided decoding backend: {self.guided_decoding_backend}"
            )
        logger.info(
            f"Guided decoder initialized with backend: {self.guided_decoding_backend}"
        )
        self.grammar_matchers: List[
            Optional[GrammarMatcher]] = [None] * self.max_num_sequences

        self.bitmask = torch.empty(self.max_num_sequences *
                                   (self.max_num_draft_tokens + 1),
                                   self.bitmask_size,
                                   dtype=self.bitmask_dtype,
                                   device='cuda')
        self.bitmask_host = torch.empty_like(self.bitmask,
                                             device='cpu',
                                             pin_memory=prefer_pinned())
        self.token_mask = torch.empty(self.max_num_sequences *
                                      (self.max_num_draft_tokens + 1),
                                      dtype=self.token_mask_dtype,
                                      device='cuda')
        self.token_mask_host = torch.empty_like(self.token_mask,
                                                device='cpu',
                                                pin_memory=prefer_pinned())

        # The number of tokens accepted by the grammar matcher in a build step.
        self.num_advanced_tokens: List[int] = [0] * self.max_num_sequences
        # The number of tokens with filled bitmask in a build step.
        self.num_guided_tokens: List[int] = [0] * self.max_num_sequences
        # The accumulated number of tokens accepted by the grammar matcher in a drafting loop.
        self.num_advanced_draft_tokens: List[int] = [0] * self.max_num_sequences
        # Whether is guided drafting is terminated because of unacceptable drafted tokens.
        self.is_draft_terminated: List[bool] = [False] * self.max_num_sequences

        self.requests: Optional[GuidedRequests] = None

        # Compilation is kicked off at request activation so it overlaps
        # prefill / KV transfer instead of blocking the forward path.
        self.grammar_compiler = AsyncGrammarCompiler(
            self.grammar_matcher_factory,
            allow_sync_fallback=self._allow_sync_compile)
        # request_id that installed the matcher at each slot; makes matcher
        # attach idempotent under CUDA-graph warmup/replay re-execution.
        self._matcher_owner: List[Optional[int]] = [None
                                                    ] * self.max_num_sequences
        # (request_id | None, error_msg) records produced on CUDA-callback
        # threads and drained by the executor thread; request_id None means
        # the failure could not be attributed to a single request.
        self._async_failures: deque = deque()

        self.stream = torch.cuda.Stream()
        self.token_event = torch.cuda.Event()
        self.bitmask_event = torch.cuda.Event()

    def drain_async_failures(
        self,
        allowed_req_ids: Optional[set] = None,
        active_req_ids: Optional[set] = None
    ) -> List[Tuple[Optional[int], str]]:
        """Pop failure records produced on CUDA-callback threads.

        Callback completion timing is rank-dependent, so callers restrict
        draining to `allowed_req_ids` (the just-synchronized batch) to keep
        the drained set rank-consistent: a record outside that batch is
        requeued if its request is still in `active_req_ids` (its batch has
        not synchronized yet) and dropped otherwise (request already gone).
        None-keyed records are batch-wide and always returned.
        """
        failures = []
        requeue = []
        # The executor thread is the only consumer; callback threads only
        # append, so this drains at least everything enqueued so far.
        while self._async_failures:
            record = self._async_failures.popleft()
            req_id = record[0]
            if allowed_req_ids is None or req_id is None or req_id in allowed_req_ids:
                failures.append(record)
            elif active_req_ids is not None and req_id in active_req_ids:
                requeue.append(record)
            # else: request already terminated; drop the record.
        self._async_failures.extend(requeue)
        return failures

    def has_async_failures(self) -> bool:
        return bool(self._async_failures)

    def _record_async_failure(self, request_id: Optional[int],
                              error_msg: str) -> None:
        # Thread-safe and non-blocking; may run on a CUDA-callback thread.
        self._async_failures.append((request_id, error_msg))

    @property
    def bitmask_size(self) -> int:
        return math.ceil(self.vocab_size_padded / 32)

    def _build(self, requests: GuidedRequests) -> List[Tuple[int, str]]:
        """Build the bitmask for requests with guided decoding enabled.

        Specifically, this method:
        - build and advance the grammar matcher for context and generation requests, respectively;
        - call the grammar matcher to fill the bitmask on CPU;
        - asynchronously copy the bitmask to GPU.
        """
        failed_requests = []
        self.token_mask_host[:requests.num_bitmask_tokens].fill_(0)

        for req, offset in requests.valid_requests_with_offsets():
            slot = req.seq_slot
            try:
                self.num_advanced_tokens[slot] = 0
                self.num_guided_tokens[slot] = 0

                matcher_init: bool = req.require_matcher_init()
                matcher_advance: bool = req.require_matcher_advance()
                if not (matcher_init or matcher_advance):
                    continue

                if matcher_init:
                    matcher, error_msg = self.grammar_compiler.take(
                        req.request_id, req.guided_decoding_params)
                    # Claim the slot even on failure so a later matcher_advance
                    # sees None, not a previous occupant's matcher.
                    self.grammar_matchers[slot] = matcher
                    self._matcher_owner[slot] = req.request_id
                    if matcher is None:
                        failed_requests.append((req.request_id, error_msg))
                        logger.error(
                            f"Request {req.request_id} at slot {slot} failed during guided decoding: {error_msg}"
                        )
                        continue

                if matcher_advance:
                    matcher = self.grammar_matchers[slot]
                    # The last new token must be acceptable unless the matcher is terminated or None:
                    # 1. For the main model loop, when overlap scheduler is enabled, the matcher may have accepted the EOS token in the draft tokens at the previous iteration.
                    # 2. For the draft model loop, the matcher may have accepted the EOS token at the previous drafting iteration.
                    # 3. The matcher can be None if there was an error during its creation.
                    if matcher is None or matcher.is_terminated(
                    ) or self.is_draft_terminated[slot]:
                        continue
                    accepted = matcher.accept_token(req.new_token)
                    if not accepted:
                        if req.is_draft:
                            self.is_draft_terminated[slot] = True
                            logger.debug(
                                f"Draft request {req.request_id} at slot {slot} failed to accept last new token: {req.new_token}."
                            )
                            continue
                        raise ValueError(
                            f"Request {req.request_id} at slot {slot} failed to accept last new token: {req.new_token}."
                        )

                self.num_advanced_tokens[slot] += 1
                if not matcher.is_terminated():
                    matcher.fill_next_token_bitmask(self.bitmask_host, offset)
                    self.token_mask_host[offset] = 1
                    self.num_guided_tokens[slot] += 1
                    # Process draft tokens. Bound by the layout's draft length:
                    # the new_tokens buffer always holds the static max, but only
                    # `max_num_draft_tokens` slots are reserved this iteration.
                    for i, tid in enumerate(
                            req.draft_tokens[:requests.max_num_draft_tokens],
                            1):
                        accepted = matcher.accept_token(tid)
                        if not accepted:
                            break
                        self.num_advanced_tokens[slot] += 1
                        if matcher.is_terminated():
                            break
                        matcher.fill_next_token_bitmask(self.bitmask_host,
                                                        offset + i)
                        self.token_mask_host[offset + i] = 1
                        self.num_guided_tokens[slot] += 1

                if req.is_draft:
                    assert len(req.draft_tokens) == 0
                    self.num_advanced_draft_tokens[
                        slot] += self.num_advanced_tokens[slot]

            except Exception as e:
                error_msg = format_guided_error(e)
                failed_requests.append((req.request_id, error_msg))
                logger.error(
                    f"Request {req.request_id} at slot {slot} failed during guided decoding: {error_msg}"
                )

        return failed_requests

    def _copy_bitmask(self,
                      requests: GuidedRequests,
                      num_bitmask_tokens: Optional[int] = None) -> None:
        if num_bitmask_tokens is None:
            num_bitmask_tokens = requests.num_bitmask_tokens
        self.bitmask[:num_bitmask_tokens].copy_(
            self.bitmask_host[:num_bitmask_tokens], non_blocking=True)
        self.token_mask[:num_bitmask_tokens].copy_(
            self.token_mask_host[:num_bitmask_tokens], non_blocking=True)

    @torch.inference_mode()
    def _apply_bitmask(self,
                       requests: GuidedRequests,
                       logits: torch.Tensor,
                       d2t: Optional[torch.Tensor] = None,
                       num_bitmask_tokens: Optional[int] = None) -> None:
        """Apply the bitmask to the corresponding logits for requests with guided decoding enabled.

        This method inplace modifies the logits tensor so that any tokens that violate the grammar constraints are masked out.
        """
        if num_bitmask_tokens is None:
            num_bitmask_tokens = requests.num_bitmask_tokens

        # In general, the logits passed to GuidedDecoder are complete in the vocabulary dimension.
        # In some special cases (e.g., MTP), the logits are sharded in the vocabulary dimension.
        vocab_size_padded = self.vocab_size_padded if d2t is None else d2t.size(
            0)
        assert vocab_size_padded % logits.size(1) == 0
        tp_size = vocab_size_padded // logits.size(1)
        assert self.bitmask_size % tp_size == 0
        tp_rank = self.rank % tp_size
        bitmask_start = tp_rank * self.bitmask_size // tp_size
        bitmask_end = bitmask_start + self.bitmask_size // tp_size

        if d2t is not None:
            d2t_start = tp_rank * vocab_size_padded // tp_size
            d2t_end = d2t_start + vocab_size_padded // tp_size
            d2t = d2t[d2t_start:d2t_end]

        torch.ops.trtllm.logits_bitmask(
            logits[:num_bitmask_tokens],
            self.bitmask[:num_bitmask_tokens, bitmask_start:bitmask_end],
            token_mask=self.token_mask[:num_bitmask_tokens],
            d2t=d2t)

    @nvtx_range("GuidedDecoder.add_batch")
    def add_batch(self,
                  scheduled_requests: ScheduledRequests,
                  runtime_draft_len: Optional[int] = None) -> None:
        num_draft_tokens = (self.max_num_draft_tokens
                            if runtime_draft_len is None else runtime_draft_len)
        self.requests = GuidedRequests.from_scheduled_requests(
            scheduled_requests, num_draft_tokens)

    @nvtx_range("GuideDecoder.build")
    def build(self) -> List[Tuple[int, str]]:
        return self._build(self.requests)

    @nvtx_range("GuideDecoder.copy_bitmask")
    def copy_bitmask(self, num_bitmask_tokens: Optional[int] = None) -> None:
        self._copy_bitmask(self.requests, num_bitmask_tokens=num_bitmask_tokens)

    @nvtx_range("GuidedDecoder.apply_bitmask")
    def apply_bitmask(self,
                      logits: torch.Tensor,
                      d2t: Optional[torch.Tensor] = None,
                      num_bitmask_tokens: Optional[int] = None) -> None:
        self._apply_bitmask(self.requests,
                            logits,
                            d2t=d2t,
                            num_bitmask_tokens=num_bitmask_tokens)

    def execute(self,
                logits: torch.Tensor,
                d2t: Optional[torch.Tensor] = None) -> List[Tuple[int, str]]:
        failed_requests = self.build()

        with torch.cuda.stream(self.stream):
            torch.cuda.current_stream().wait_event(self.token_event)
            self.copy_bitmask()
            self.bitmask_event.record()

        torch.cuda.current_stream().wait_event(self.bitmask_event)
        self.apply_bitmask(logits, d2t=d2t)
        self.token_event.record()

        return failed_requests

    def _rollback_rejected_tokens(self, requests: GuidedRequests) -> None:
        """Rollback the grammar matcher for rejected tokens.

        This method should be called:
        - after the verification (so that the accepted tokens are ready) and
        - before the first guided decoding build of the next drafting loop.
        """
        if self.max_num_draft_tokens <= 0:
            return

        for req in requests.valid_requests():
            slot = req.seq_slot
            if self.num_advanced_tokens[slot] <= 0:
                continue
            try:
                matcher = self.grammar_matchers[slot]
                if matcher is None:
                    # Matcher creation failed; request is being terminated.
                    continue
                num_accepted_tokens = 1 + req.num_accepted_draft_tokens
                # Rollback the grammar matcher to the last accepted token.
                num_rollback_tokens = self.num_advanced_tokens[
                    slot] - num_accepted_tokens
                if num_rollback_tokens < 0:
                    raise ValueError(
                        f"Failed to rollback: num_advanced_tokens={self.num_advanced_tokens[slot]}, num_accepted_tokens={num_accepted_tokens}, num_rollback_tokens={num_rollback_tokens}"
                    )
                matcher.rollback(num_rollback_tokens)
            except Exception as e:
                # May run on a CUDA-callback thread: record, never raise.
                self._record_async_failure(req.request_id,
                                           format_guided_error(e))

    def _rollback_draft_tokens(self, requests: GuidedRequests) -> None:
        """Rollback the grammar matcher for draft tokens.

        This method should be called:
        - after the the drafting loop and
        - before the guided decoding build of the target model.
        """
        if self.max_num_draft_tokens <= 0:
            return

        for req in requests.valid_requests():
            slot = req.seq_slot
            try:
                matcher = self.grammar_matchers[slot]
                if self.num_advanced_draft_tokens[
                        slot] > 0 and matcher is not None:
                    matcher.rollback(self.num_advanced_draft_tokens[slot])
            except Exception as e:
                # May run on a CUDA-callback thread: record, never raise.
                self._record_async_failure(req.request_id,
                                           format_guided_error(e))
            finally:
                # Reset the drafting states.
                self.num_advanced_draft_tokens[slot] = 0
                self.is_draft_terminated[slot] = False

    @nvtx_range("GuidedDecoder.rollback_rejected_tokens")
    def rollback_rejected_tokens(self) -> None:
        self._rollback_rejected_tokens(self.requests)

    @nvtx_range("GuidedDecoder.rollback_draft_tokens")
    def rollback_draft_tokens(self) -> None:
        self._rollback_draft_tokens(self.requests)

    def _init_disagg_gen_requests(self, requests: GuidedRequests) -> None:
        """Attach precompiled grammar matchers for disagg gen requests.

        Attach-only by design: this may run inside a CUDA host callback, so
        it must never run an unbounded grammar compile nor raise.
        """
        for req in requests.valid_requests():
            if not req.is_generation_only_first_iteration:
                continue
            slot = req.seq_slot
            if self._matcher_owner[slot] == req.request_id:
                # CUDA-graph warmup/replay re-executes this callback for the
                # same batch; the attach (or its failure) already happened.
                continue
            matcher, error_msg = self.grammar_compiler.take(
                req.request_id, req.guided_decoding_params)
            self.grammar_matchers[slot] = matcher
            self._matcher_owner[slot] = req.request_id
            if matcher is None:
                self._record_async_failure(req.request_id, error_msg)

    @nvtx_range("GuidedDecoder.init_disagg_gen_requests")
    def init_disagg_gen_requests(self) -> None:
        self._init_disagg_gen_requests(self.requests)


def _on_hostfunc_error(e: BaseException, fn, args, kwargs) -> None:
    """hostfunc on_error hook: record the failure on the decoder instance.

    An unobserved exception in a CUDA host callback leaves the captured graph
    node in an undefined state, which manifests as a silent engine hang (peer
    ranks block in collectives)."""
    args[0]._record_hostfunc_error(fn.__name__, e)


class CapturableGuidedDecoder(GuidedDecoder):
    # Matcher attach runs inside CUDA host callbacks here, where a synchronous
    # compile could stall the rank; requests are gated on compile completion
    # before being scheduled instead (see GuidedDecodingCoordinator).
    _allow_sync_compile = False

    def _record_hostfunc_error(self, fn_name: str, e: BaseException) -> None:
        # request_id None fails every guided request of the batch. The message
        # reaches client responses, so the traceback stays log-only.
        logger.error(f"Guided decoding internal error in {fn_name}: {str(e)}\n"
                     f"{''.join(traceback.format_exception(e))}")
        self._record_async_failure(
            None,
            f"Guided decoding error: internal error in {fn_name}: {str(e)}")

    def __init__(self,
                 guided_decoding_config: GuidedDecodingConfig,
                 max_num_sequences: int,
                 vocab_size_padded: int,
                 max_num_draft_tokens: int = 0,
                 rank: int = 0,
                 tokenizer=None):
        super().__init__(guided_decoding_config=guided_decoding_config,
                         max_num_sequences=max_num_sequences,
                         vocab_size_padded=vocab_size_padded,
                         max_num_draft_tokens=max_num_draft_tokens,
                         rank=rank,
                         tokenizer=tokenizer)
        # self.requests should be accessed by normal host code;
        # self.requests_hostfunc should be accessed by hostfunc (CUDA callback).
        self.requests_hostfunc: Optional[GuidedRequests] = None
        self.queue = Queue()

        self.new_tokens = torch.empty(self.max_num_draft_tokens + 1,
                                      self.max_num_sequences,
                                      dtype=torch.int32,
                                      pin_memory=prefer_pinned())
        self.num_accepted_tokens = torch.empty(self.max_num_sequences,
                                               dtype=torch.int32,
                                               pin_memory=prefer_pinned())

        # torch.compile kernels are called with GIL being held;
        # this could cause deadlock with CUDA callback to Python code.
        # See: https://github.com/pytorch/pytorch/issues/163061
        torch.compiler.set_stance("force_eager")

    @nvtx_range("GuidedDecoder.add_batch")
    def add_batch(self,
                  scheduled_requests: ScheduledRequests,
                  new_tokens: Optional[torch.Tensor] = None,
                  runtime_draft_len: Optional[int] = None) -> None:
        # See GuidedDecoder.add_batch: the layout must follow the runtime draft
        # length so the captured graph's bitmask matches the target logits.
        num_draft_tokens = (self.max_num_draft_tokens
                            if runtime_draft_len is None else runtime_draft_len)
        self.requests = GuidedRequests.from_scheduled_requests(
            scheduled_requests, num_draft_tokens)
        if new_tokens is not None:
            self.new_tokens.copy_(new_tokens.squeeze(-1), non_blocking=True)
        self.queue.put((self.requests, new_tokens is not None))
        # self.token_event.record() should be called inside CUDA graph capturing;
        # currently, it is in PyTorchModelEngine._preprocess_inputs.

    @hostfunc(on_error=_on_hostfunc_error)
    def fetch_batch(self) -> None:
        # CUDA graph warmup calls model forward for multiple times for one prepared inputs
        if self.queue.empty():
            return
        self.requests_hostfunc, has_new_tokens = self.queue.get()
        if not has_new_tokens:
            return

        for req in self.requests_hostfunc.valid_requests():
            if req.prev_seq_slot is None:
                continue
            req.new_token, *req.draft_tokens = self.new_tokens[:, req.
                                                               seq_slot].tolist(
                                                               )

    @hostfunc(on_error=_on_hostfunc_error)
    def build(self) -> None:
        # A hostfunc cannot return results (its return value is a graph-node
        # handle), so per-request failures surface via the async failure
        # records instead.
        for failure in self._build(self.requests_hostfunc):
            self._record_async_failure(*failure)

    def execute(self,
                logits: torch.Tensor,
                d2t: Optional[torch.Tensor] = None) -> None:
        with torch.cuda.stream(self.stream):
            torch.cuda.current_stream().wait_event(self.token_event)
            self.fetch_batch()
            self.init_disagg_gen_requests()
            self.build()
            self.copy_bitmask()
            self.bitmask_event.record()

        torch.cuda.current_stream().wait_event(self.bitmask_event)
        self.apply_bitmask(logits, d2t=d2t)

    @hostfunc(on_error=_on_hostfunc_error)
    def rollback_rejected_tokens(self) -> None:
        self._rollback_rejected_tokens(self.requests_hostfunc)

    @hostfunc(on_error=_on_hostfunc_error)
    def rollback_draft_tokens(self) -> None:
        self._rollback_draft_tokens(self.requests_hostfunc)

    @hostfunc(on_error=_on_hostfunc_error)
    def init_disagg_gen_requests(self) -> None:
        self._init_disagg_gen_requests(self.requests_hostfunc)

    @nvtx_range("GuidedDecoder.add_draft_batch")
    def add_draft_batch(self,
                        new_tokens: torch.Tensor,
                        num_accepted_tokens: torch.Tensor,
                        draft_step: int = 0) -> None:
        batch_size = len(self.requests)
        assert new_tokens.size(0) == batch_size
        self.new_tokens[0, :batch_size].copy_(new_tokens, non_blocking=True)
        if draft_step == 0:
            assert num_accepted_tokens.size(0) == batch_size
            self.num_accepted_tokens[:batch_size].copy_(num_accepted_tokens,
                                                        non_blocking=True)
        self.token_event.record()

    @hostfunc(on_error=_on_hostfunc_error)
    def fetch_draft_batch(self, draft_step: int = 0) -> None:
        batch_size = len(self.requests_hostfunc)
        new_tokens_list = self.new_tokens[0, :batch_size].tolist()
        if draft_step == 0:
            num_accepted_tokens_list = self.num_accepted_tokens[:
                                                                batch_size].tolist(
                                                                )
        for i, req in enumerate(self.requests_hostfunc.requests):
            if req.guided_decoding_params is None or (slot :=
                                                      req.seq_slot) is None:
                continue
            req.new_token = new_tokens_list[i]
            if draft_step == 0:
                # When overlap scheduler is enabled, it is possible that
                # - The EOS token is in the draft tokens, and
                # - Some draft tokens after the EOS token are accepted by the target model.
                # These requests should be terminated at this executor iteration.
                req.num_accepted_draft_tokens = min(
                    num_accepted_tokens_list[i],
                    self.num_advanced_tokens[slot]) - 1
                assert not req.is_draft
                req.cast_to_draft()
            else:
                assert req.is_draft

    def execute_draft_batch(self,
                            logits: torch.Tensor,
                            d2t: Optional[torch.Tensor] = None,
                            draft_step: int = 0) -> None:
        with torch.cuda.stream(self.stream):
            torch.cuda.current_stream().wait_event(self.token_event)
            self.fetch_draft_batch(draft_step=draft_step)
            if draft_step == 0:
                self.rollback_rejected_tokens()
            self.build()
            if draft_step == self.max_num_draft_tokens - 1:
                self.rollback_draft_tokens()
            # Overwrite num_bitmask_tokens since the request might not be updated on CUDA stream yet.
            self.copy_bitmask(num_bitmask_tokens=len(self.requests))
            self.bitmask_event.record()

        torch.cuda.current_stream().wait_event(self.bitmask_event)
        # Overwrite num_bitmask_tokens since the request might not be updated on CUDA stream yet.
        self.apply_bitmask(logits,
                           d2t=d2t,
                           num_bitmask_tokens=len(self.requests))
