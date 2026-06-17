import dataclasses
import threading
from typing import Any, Callable, Iterable, List, Mapping, Optional, Tuple

import orjson

from tensorrt_llm.bindings import LlmRequestState
from tensorrt_llm.bindings.executor import IterationStats

KV_TRANSFER_DIRECTION_CONTEXT = "context"
KV_TRANSFER_DIRECTION_GENERATION = "generation"
KV_TRANSFER_STATUS_FAILURE = "failure"
KV_TRANSFER_STATUS_SUCCESS = "success"

_KV_TRANSFER_SYNTHETIC_EVENT_ID_BIT = 1 << 63
_KV_TRANSFER_SYNTHETIC_EVENT_TAG_MASK = 0xFFFF
_KV_TRANSFER_UCX_PASSIVE_HANDSHAKE_FAILURE = "ucxPassiveHandshakeFailure"
_KV_TRANSFER_UCX_PASSIVE_HANDSHAKE_TAGS = (0xF1, 0xF2)
_KV_TRANSFER_UCX_PASSIVE_HANDSHAKE_EVENT_IDS = frozenset(
    _KV_TRANSFER_SYNTHETIC_EVENT_ID_BIT | tag
    for tag in _KV_TRANSFER_UCX_PASSIVE_HANDSHAKE_TAGS)
_KV_CACHE_TRANSFER_EVENTS_FIELD = "kvCacheTransferEvents"
ObserverStatsPayload = dict[str, Any]


def _kv_transfer_event(rank: int, direction: str, status: str,
                       request_id: int) -> dict[str, int | str]:
    request_id = int(request_id)
    synthetic_request_id = request_id
    if request_id in _KV_TRANSFER_UCX_PASSIVE_HANDSHAKE_EVENT_IDS:
        request_id = -(request_id & _KV_TRANSFER_SYNTHETIC_EVENT_TAG_MASK)
    event = {
        "rank": int(rank),
        "direction": direction,
        "status": status,
        "requestId": request_id,
    }
    if synthetic_request_id in _KV_TRANSFER_UCX_PASSIVE_HANDSHAKE_EVENT_IDS:
        event["syntheticEvent"] = _KV_TRANSFER_UCX_PASSIVE_HANDSHAKE_FAILURE
        event["syntheticRequestId"] = str(synthetic_request_id)
        event["ucxHandshakeTag"] = (
            synthetic_request_id & _KV_TRANSFER_SYNTHETIC_EVENT_TAG_MASK)
    return event


@dataclasses.dataclass(frozen=True)
class KvTransferObservation:
    direction: str
    status: str
    requests: Iterable[Any] = ()
    ranked_request_ids: Iterable[Tuple[int, int]] = ()
    dedupe_failure: bool = False
    requests_by_id: Optional[Mapping[int, Any]] = None

    @classmethod
    def context_success(
            cls,
            requests: Iterable[Any] = (),
            *,
            ranked_request_ids: Iterable[Tuple[int, int]] = ()):
        return cls(KV_TRANSFER_DIRECTION_CONTEXT, KV_TRANSFER_STATUS_SUCCESS,
                   requests, ranked_request_ids)

    @classmethod
    def context_failure(
            cls,
            requests: Iterable[Any] = (),
            *,
            ranked_request_ids: Iterable[Tuple[int, int]] = ()):
        return cls(KV_TRANSFER_DIRECTION_CONTEXT, KV_TRANSFER_STATUS_FAILURE,
                   requests, ranked_request_ids)

    @classmethod
    def generation_success(
            cls,
            requests: Iterable[Any] = (),
            *,
            ranked_request_ids: Iterable[Tuple[int, int]] = ()):
        return cls(KV_TRANSFER_DIRECTION_GENERATION,
                   KV_TRANSFER_STATUS_SUCCESS, requests, ranked_request_ids)

    @classmethod
    def generation_failure_once(
            cls,
            requests: Iterable[Any] = (),
            *,
            ranked_request_ids: Iterable[Tuple[int, int]] = (),
            requests_by_id: Optional[Mapping[int, Any]] = None):
        return cls(KV_TRANSFER_DIRECTION_GENERATION,
                   KV_TRANSFER_STATUS_FAILURE, requests, ranked_request_ids,
                   dedupe_failure=True,
                   requests_by_id=requests_by_id)


@dataclasses.dataclass
class IterationStatsWithObserverPayload:
    iteration_stats: IterationStats
    observer_payload: ObserverStatsPayload

    def to_json_str(self) -> str:
        stats_dict = orjson.loads(self.iteration_stats.to_json_str())
        stats_dict.update(self.observer_payload)
        return orjson.dumps(stats_dict).decode("utf-8")


class PyExecutorObserver:

    def observe(self, event: Any) -> None:
        pass

    def has_stats(self) -> bool:
        return False

    def drain_stats(self) -> ObserverStatsPayload:
        return {}

    def should_record_synthetic_kv_transfer_events_in_transceiver(
            self, *, kv_cache_transceiver, enable_attention_dp: bool,
            dist) -> bool:
        return False

    def record_context_failure_events(
            self, requests, *, kv_cache_transceiver=None,
            use_transceiver: bool = False) -> None:
        pass

    def record_generation_failure_events(
            self, requests, *, kv_cache_transceiver=None,
            use_transceiver: bool = False) -> None:
        pass

    def record_generation_success_events(self, requests) -> None:
        pass

    def record_context_status_events(self, success_events,
                                     error_events) -> None:
        pass

    def record_generation_status_events(
            self, transfer_status, generation_transfer_request_ids: set[int],
            active_requests) -> None:
        pass


class KvTransferStatsObserver(PyExecutorObserver):
    """Buffers KV transfer events for publication with iteration stats."""

    def __init__(self, *, global_rank: int,
                 enable_iter_perf_stats: bool,
                 canceled_req_ids: Callable[[], Iterable[int]]):
        self._global_rank = global_rank
        self._enable_iter_perf_stats = enable_iter_perf_stats
        self._canceled_req_ids = canceled_req_ids
        self._events: List[dict[str, int | str]] = []
        self._failure_event_keys: set[tuple[int, int, str]] = set()
        self._lock = threading.Lock()

    def observe(self, event: Any) -> None:
        if not isinstance(event, KvTransferObservation):
            return
        if event.dedupe_failure:
            if event.ranked_request_ids:
                self._record_ranked_failure_once(
                    event.ranked_request_ids, event.requests_by_id or {},
                    event.direction)
            else:
                self._record_failure_once(event.requests, event.direction)
        elif event.ranked_request_ids:
            self._record_ranked(event.ranked_request_ids, event.direction,
                                event.status)
        else:
            self._record(event.requests, event.direction, event.status)

    def has_stats(self) -> bool:
        with self._lock:
            return bool(self._events)

    def drain_stats(self) -> ObserverStatsPayload:
        if not self._enable_iter_perf_stats:
            return {}
        with self._lock:
            events = self._events
            self._events = []
        if not events:
            return {}
        return {_KV_CACHE_TRANSFER_EVENTS_FIELD: events}

    def should_record_synthetic_kv_transfer_events_in_transceiver(
            self, *, kv_cache_transceiver, enable_attention_dp: bool,
            dist) -> bool:
        # Synthetic KV failures are timeout-observed failures: Python has
        # decided the request failed before the native transfer future returned.
        # In TP, buffer them in the transceiver so C++ can publish them with the
        # same rank attribution and gather semantics as native transfer errors.
        return (self._enable_iter_perf_stats and kv_cache_transceiver is not None
                and not enable_attention_dp and dist is not None
                and getattr(dist, "world_size", 1) != 1)

    def record_context_failure_events(
            self, requests, *, kv_cache_transceiver=None,
            use_transceiver: bool = False) -> None:
        if not self._enable_iter_perf_stats or not requests:
            return
        if use_transceiver:
            if kv_cache_transceiver is None:
                return
            for request in requests:
                kv_cache_transceiver.record_context_kv_transfer_failure_event(
                    request)
            return

        if kv_cache_transceiver is None:
            return
        reportable_requests = [
            request for request in requests
            if kv_cache_transceiver.take_context_kv_transfer_event_report(
                request)
        ]
        self.observe(KvTransferObservation.context_failure(reportable_requests))

    def record_generation_failure_events(
            self, requests, *, kv_cache_transceiver=None,
            use_transceiver: bool = False) -> None:
        if not self._enable_iter_perf_stats or not requests:
            return
        if use_transceiver:
            if kv_cache_transceiver is None:
                return
            for request in requests:
                kv_cache_transceiver.record_generation_kv_transfer_failure_event(
                    request)
            return

        self.observe(KvTransferObservation.generation_failure_once(requests))

    def record_generation_success_events(self, requests) -> None:
        self.observe(KvTransferObservation.generation_success(requests))

    def record_context_status_events(self, success_events,
                                     error_events) -> None:
        self.observe(
            KvTransferObservation.context_success(
                ranked_request_ids=success_events))
        self.observe(
            KvTransferObservation.context_failure(
                ranked_request_ids=error_events))

    def record_generation_status_events(
            self, transfer_status, generation_transfer_request_ids: set[int],
            active_requests) -> None:
        if transfer_status is None:
            kv_transfer_success_events = []
            kv_transfer_error_events = []
        else:
            (
                _,
                _,
                kv_transfer_success_events,
                kv_transfer_error_events,
            ) = transfer_status

        active_requests = list(active_requests)
        successful_requests = []
        failed_requests = []
        timed_out_request_ids = set()
        for request in active_requests:
            if (request.py_kv_transfer_timed_out and request.state
                    == LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE):
                request.state = LlmRequestState.DISAGG_TRANS_ERROR
                failed_requests.append(request)
                timed_out_request_ids.add(request.py_request_id)
            elif (request.state
                  == LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE
                  and request.py_request_id in generation_transfer_request_ids):
                successful_requests.append(request)
            elif (request.state == LlmRequestState.DISAGG_TRANS_ERROR
                  and request.py_request_id in generation_transfer_request_ids):
                failed_requests.append(request)

        if transfer_status is None:
            self.observe(
                KvTransferObservation.generation_success(successful_requests))
        else:
            if timed_out_request_ids:
                kv_transfer_success_events = [
                    event for event in kv_transfer_success_events
                    if event[1] not in timed_out_request_ids
                ]
            requests_by_id = {
                request.py_request_id: request
                for request in active_requests
            }
            self.observe(
                KvTransferObservation.generation_success(
                    ranked_request_ids=kv_transfer_success_events))
            self.observe(
                KvTransferObservation.generation_failure_once(
                    ranked_request_ids=kv_transfer_error_events,
                    requests_by_id=requests_by_id))
        self.observe(KvTransferObservation.generation_failure_once(
            failed_requests))

    def _record(self, requests, direction: str, status: str) -> None:
        if not self._enable_iter_perf_stats or not requests:
            return

        events = [
            _kv_transfer_event(self._global_rank, direction, status,
                               request.py_request_id) for request in requests
        ]
        with self._lock:
            self._events.extend(events)

    def _record_ranked(self, event_records: Iterable[Tuple[int, int]],
                       direction: str, status: str) -> None:
        if not self._enable_iter_perf_stats or not event_records:
            return

        events = [
            _kv_transfer_event(rank, direction, status, request_id)
            for rank, request_id in event_records
        ]
        with self._lock:
            self._events.extend(events)

    def _record_failure_once(self, requests, direction: str) -> None:
        if not self._enable_iter_perf_stats or not requests:
            return

        events = []
        canceled_req_ids = set(self._canceled_req_ids() or [])
        with self._lock:
            for request in requests:
                if not self._should_record_failure(request, direction,
                                                   canceled_req_ids):
                    continue
                event_key = (self._global_rank, request.py_request_id,
                             direction)
                if event_key in self._failure_event_keys:
                    continue
                self._failure_event_keys.add(event_key)
                events.append(
                    _kv_transfer_event(self._global_rank, direction,
                                       KV_TRANSFER_STATUS_FAILURE,
                                       request.py_request_id))
            self._events.extend(events)

    def _record_ranked_failure_once(
            self, event_records: Iterable[Tuple[int, int]], requests_by_id,
            direction: str) -> None:
        if not self._enable_iter_perf_stats or not event_records:
            return

        events = []
        canceled_req_ids = set(self._canceled_req_ids() or [])
        with self._lock:
            for rank, request_id in event_records:
                rank = int(rank)
                request_id = int(request_id)
                request = requests_by_id.get(request_id)
                if not self._should_record_failure(request, direction,
                                                   canceled_req_ids):
                    continue
                event_key = (rank, request_id, direction)
                if event_key in self._failure_event_keys:
                    continue
                self._failure_event_keys.add(event_key)
                events.append(
                    _kv_transfer_event(rank, direction,
                                       KV_TRANSFER_STATUS_FAILURE,
                                       request_id))
            self._events.extend(events)

    def _should_record_failure(self, request, direction: str,
                               canceled_req_ids: set[int]) -> bool:
        if request is None:
            return True
        if (direction == KV_TRANSFER_DIRECTION_GENERATION
                and self._is_canceled_generation_request(
                    request, canceled_req_ids)):
            return False
        return (not request.py_kv_transfer_timed_out
                or request.py_should_report_kv_transfer_failure_event)

    def _is_canceled_generation_request(self, request,
                                        canceled_req_ids: set[int]) -> bool:
        if not request.is_generation_only_request():
            return False
        request_id = (request.py_request_id
                      if not request.is_child else request.parent_request_id)
        return (request.is_finished_due_to_cancellation
                or request_id in canceled_req_ids)


def merge_observer_stats_payloads(
        payloads: Iterable[ObserverStatsPayload]) -> ObserverStatsPayload:
    merged: ObserverStatsPayload = {}
    for payload in payloads:
        if not payload:
            continue
        for key, value in payload.items():
            if not value:
                continue
            existing_value = merged.get(key)
            if isinstance(existing_value, list) and isinstance(value, list):
                existing_value.extend(value)
            else:
                merged[key] = list(value) if isinstance(value, list) else value
    return merged


def should_publish_local_observer_stats(*, enable_attention_dp: bool,
                                        world_size: int, rank: int,
                                        gather_all_responses: bool) -> bool:
    should_gather_across_adp = enable_attention_dp and world_size != 1
    return not should_gather_across_adp or rank == 0 or gather_all_responses


def append_observer_stats_entry(stats: list, max_stats_len: int,
                                observer_stats: ObserverStatsPayload) -> None:
    if len(stats) > max_stats_len:
        stats.pop(0)
    stats.append((
        IterationStatsWithObserverPayload(
            IterationStats(),
            observer_stats,
        ),
        None,
    ))


def gather_with_observer_stats(
        items: Iterable[Any], *, dist, gather_all_responses: bool,
        observer: PyExecutorObserver,
        enable_iter_perf_stats: bool) -> Tuple[List[Any], ObserverStatsPayload]:
    items = list(items)
    if not enable_iter_perf_stats:
        gathered_items = (dist.allgather(items) if gather_all_responses else
                          dist.tp_gather(items))
        if dist.rank == 0 or gather_all_responses:
            return flatten_gathered_items(gathered_items), {}
        return items, {}

    local_observer_stats = {}
    if observer.has_stats():
        local_observer_stats = observer.drain_stats()
    gather_payload = (items, local_observer_stats)
    if not gather_all_responses:
        gathered_payloads = dist.tp_gather(gather_payload)
    else:
        gathered_payloads = dist.allgather(gather_payload)

    if dist.rank == 0 or gather_all_responses:
        return split_gathered_response_payloads(gathered_payloads)
    return items, {}


def flatten_gathered_items(gathered_items) -> List[Any]:
    items = []
    if gathered_items is None:
        return items

    for rank_items in gathered_items:
        if rank_items is not None:
            items.extend(rank_items)
    return items


def split_gathered_response_payloads(
        gathered_payloads) -> Tuple[List, ObserverStatsPayload]:
    gathered_responses = []
    gathered_observer_stats = []
    if gathered_payloads is None:
        return gathered_responses, {}

    for rank_payload in gathered_payloads:
        if rank_payload is None:
            continue
        rank_responses, rank_observer_stats = rank_payload
        gathered_responses.extend(rank_responses)
        if rank_observer_stats:
            gathered_observer_stats.append(rank_observer_stats)
    return gathered_responses, merge_observer_stats_payloads(
        gathered_observer_stats)
