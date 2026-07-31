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

import pickle
import sys
from unittest.mock import MagicMock, patch

import cloudpickle
import mpi4py
import pytest

from tensorrt_llm import mpi_rank
from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector import (
    AsyncRequests, KvCacheConnectorManager, KvCacheConnectorWorker,
    KvCacheConnectorSchedulerOutputManager)
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.resource_manager import (CacheTypeCpp,
                                                             KVCacheManager)
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests

cloudpickle.register_pickle_by_value(sys.modules[__name__])
mpi4py.MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)


def run_across_mpi(executor, fun, num_ranks):
    return list(executor.starmap(fun, [() for i in range(num_ranks)]))


def test_send_kv_async_skips_stale_finished_connector_request():
    request = MagicMock(py_request_id=42, is_finished=True)
    executor = object.__new__(PyExecutor)
    executor.kv_cache_transceiver = None
    executor.kv_cache_manager = MagicMock()
    executor.kv_cache_manager.get_cache_indices.return_value = [1, 2]
    executor.kv_connector_manager = MagicMock()
    executor.kv_connector_manager.request_finished.return_value = False
    executor.async_transfer_manager = MagicMock()
    executor.disable_overlap_scheduler = False
    executor.previous_batch = MagicMock()
    executor.previous_batch.scheduled_requests.all_requests.return_value = [
        request
    ]
    executor.active_requests = [request]

    executor._send_kv_async([])

    executor.kv_cache_manager.get_cache_indices.assert_called_once_with(request)
    executor.kv_connector_manager.request_finished.assert_called_once_with(
        request, [1, 2])

    executor.active_requests = []
    executor._send_kv_async([])

    executor.kv_cache_manager.get_cache_indices.assert_called_once_with(request)
    executor.kv_connector_manager.request_finished.assert_called_once_with(
        request, [1, 2])


@pytest.mark.parametrize("disable_overlap_scheduler", [True, False])
def test_send_kv_async_avoids_unnecessary_active_request_scan(
        disable_overlap_scheduler: bool) -> None:
    request = MagicMock(py_request_id=42, is_finished=True)
    executor = object.__new__(PyExecutor)
    executor.kv_cache_transceiver = None
    executor.kv_cache_manager = MagicMock()
    executor.kv_cache_manager.get_cache_indices.return_value = [1, 2]
    executor.kv_connector_manager = MagicMock()
    executor.kv_connector_manager.request_finished.return_value = False
    executor.async_transfer_manager = MagicMock()
    executor.disable_overlap_scheduler = disable_overlap_scheduler
    executor.previous_batch = None
    executor.active_requests = MagicMock()
    executor.active_requests.__iter__.side_effect = AssertionError(
        "active requests should not be scanned")

    scheduled_requests = [request] if disable_overlap_scheduler else []
    executor._send_kv_async(scheduled_requests)

    if disable_overlap_scheduler:
        executor.kv_cache_manager.get_cache_indices.assert_called_once_with(
            request)
    else:
        executor.kv_cache_manager.get_cache_indices.assert_not_called()


@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
# TODO(jthomson04): I don't have the slightest idea why this test is leaking threads.
@pytest.mark.threadleak(enabled=False)
def test_connector_manager_get_finished_allgather(mpi_pool_executor):

    def test():
        worker = MagicMock()

        if mpi_rank() == 0:
            scheduler = MagicMock()

            scheduler.request_finished.return_value = True
        else:
            scheduler = None

        manager = KvCacheConnectorManager(worker, scheduler=scheduler)

        req = MagicMock()

        req.request_id = 42

        manager.request_finished(req, [])

        # To start, make both workers return nothing.
        worker.get_finished.return_value = ([], [])

        assert manager.get_finished() == []

        assert worker.get_finished.call_count == 1
        assert worker.get_finished.call_args[0] == ([42], [])

        worker.get_finished.reset_mock()

        # Now, only return the request id on one worker.
        if mpi_rank() == 0:
            worker.get_finished.return_value = ([42], [])
        else:
            worker.get_finished.return_value = ([], [])

        # It should still return nothing, since rank 1 is still saving.
        assert manager.get_finished() == []

        assert worker.get_finished.call_count == 1
        assert worker.get_finished.call_args[0] == ([], [])

        # Now, also return it on worker 1.
        if mpi_rank() == 0:
            worker.get_finished.return_value = ([], [])
        else:
            worker.get_finished.return_value = ([42], [])

        assert manager.get_finished() == [req]

    run_across_mpi(mpi_pool_executor, test, 2)


@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_connector_manager_num_matched_tokens(mpi_pool_executor):

    def test():
        worker = MagicMock()
        worker.can_skip_scheduler_match.return_value = False

        if mpi_rank() == 0:
            scheduler = MagicMock()
            scheduler.get_num_new_matched_tokens.return_value = (16, True)
        else:
            scheduler = None

        manager = KvCacheConnectorManager(worker, scheduler=scheduler)

        req = MagicMock()

        req.request_id = 42
        req.is_generation_only_request = False
        req.multimodal_positions = []

        assert manager.get_num_new_matched_tokens(req, 32) == 16

        if mpi_rank() == 0:
            assert scheduler.get_num_new_matched_tokens.call_count == 1
            assert scheduler.get_num_new_matched_tokens.call_args[0] == (req,
                                                                         32)

    run_across_mpi(mpi_pool_executor, test, 2)


@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_connector_manager_skips_collective_for_device_complete_match(
        mpi_pool_executor):

    def test():
        worker = MagicMock()
        worker.can_skip_scheduler_match.return_value = True

        scheduler = MagicMock() if mpi_rank() == 0 else None
        manager = KvCacheConnectorManager(worker, scheduler=scheduler)

        req = MagicMock()
        req.request_id = 42
        req.is_generation_only_request = False
        req.multimodal_positions = []
        req.get_tokens.return_value = list(range(65))

        with patch("tensorrt_llm._torch.pyexecutor.connectors."
                   "kv_cache_connector.mpi_broadcast") as broadcast:
            assert manager.get_num_new_matched_tokens(req, 64) == 0

        worker.can_skip_scheduler_match.assert_called_once_with(65, 64)
        broadcast.assert_not_called()
        if scheduler is not None:
            scheduler.prepare_scheduler_match_skip.assert_called_once_with(
                req, 64)
            scheduler.get_num_new_matched_tokens.assert_not_called()

    run_across_mpi(mpi_pool_executor, test, 2)


@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_connector_manager_does_not_skip_multimodal_scheduler_match(
        mpi_pool_executor):

    def test():
        worker = MagicMock()
        worker.can_skip_scheduler_match.return_value = True

        scheduler = MagicMock() if mpi_rank() == 0 else None
        if scheduler is not None:
            scheduler.get_num_new_matched_tokens.return_value = (0, False)
        manager = KvCacheConnectorManager(worker, scheduler=scheduler)

        req = MagicMock()
        req.request_id = 42
        req.is_generation_only_request = False
        req.multimodal_positions = [MagicMock()]
        req.get_tokens.return_value = list(range(65))

        assert manager.get_num_new_matched_tokens(req, 64) == 0

        worker.can_skip_scheduler_match.assert_not_called()
        if scheduler is not None:
            scheduler.get_num_new_matched_tokens.assert_called_once_with(
                req, 64)

    run_across_mpi(mpi_pool_executor, test, 2)


def test_connector_schedulable_reuse_preview_is_opt_in():
    worker = MagicMock(spec=KvCacheConnectorWorker)
    worker.supports_schedulable_reuse_preview.return_value = False
    manager = KvCacheConnectorManager(worker, scheduler=MagicMock())

    assert not manager.supports_schedulable_reuse_preview()

    worker.supports_schedulable_reuse_preview.return_value = True
    assert manager.supports_schedulable_reuse_preview()


@pytest.mark.parametrize(
    ("connector_manager", "expected"),
    [
        (None, True),
        (MagicMock(supports_schedulable_reuse_preview=lambda: False), False),
        (MagicMock(supports_schedulable_reuse_preview=lambda: True), True),
    ],
)
def test_schedulable_reuse_preview_respects_connector_opt_in(
        connector_manager, expected):
    executor = object.__new__(PyExecutor)
    executor.enable_kv_cache_reuse = True
    executor.kv_cache_manager = MagicMock(
        enable_partial_reuse=False,
        is_vswa=False,
        has_linear_attention_layers=False,
    )
    executor.kv_cache_manager.estimate_reusable_prompt_len = MagicMock()
    executor.kv_connector_manager = connector_manager

    assert executor._should_apply_schedulable_reuse_preview() is expected


@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_connector_manager_take_scheduled_requests(mpi_pool_executor):

    def test():
        worker = MagicMock()
        worker.can_skip_scheduler_match.return_value = False

        if mpi_rank() == 0:
            scheduler = MagicMock()
        else:
            scheduler = None

        manager = KvCacheConnectorManager(worker, scheduler=scheduler)

        scheduled_requests = ScheduledRequests()

        req0 = MagicMock()
        req0.request_id = 0
        req0.is_generation_only_request = False
        req0.multimodal_positions = []

        req1 = MagicMock()
        req1.request_id = 1
        req1.is_generation_only_request = False
        req1.multimodal_positions = []

        if mpi_rank() == 0:
            scheduler.get_num_new_matched_tokens.return_value = (16, True)

        assert manager.get_num_new_matched_tokens(req0, 0) == 16
        if mpi_rank() == 0:
            assert scheduler.get_num_new_matched_tokens.call_count == 1
            assert scheduler.get_num_new_matched_tokens.call_args[0] == (req0,
                                                                         0)

            scheduler.get_num_new_matched_tokens.reset_mock()
            scheduler.get_num_new_matched_tokens.return_value = (32, False)

        assert manager.get_num_new_matched_tokens(req1, 0) == 32
        if mpi_rank() == 0:
            assert scheduler.get_num_new_matched_tokens.call_count == 1
            assert scheduler.get_num_new_matched_tokens.call_args[0] == (req1,
                                                                         0)

        scheduled_requests.context_requests_last_chunk = [req0, req1]

        manager.take_scheduled_requests_pending_load(scheduled_requests)

        assert scheduled_requests.context_requests_last_chunk == [req1]

    run_across_mpi(mpi_pool_executor, test, 2)


def test_scheduler_output_num_scheduled_tokens_with_mtp():
    """Test that num_scheduled_tokens is correctly set for MTP (multi-token prediction)."""
    NUM_DRAFT_TOKENS = 3

    kv_cache_manager = MagicMock()
    kv_cache_manager.get_cache_indices.return_value = [0, 1, 2]
    kv_cache_manager.commit_and_get_block_hashes.return_value = []

    # Create a mock request in generation state with draft tokens
    req = MagicMock()
    req.request_id = 42
    req.state = LlmRequestState.GENERATION_IN_PROGRESS
    req.get_tokens.return_value = [1, 2, 3, 4, 5]  # 5 tokens already generated
    req.py_draft_tokens = [100, 101, 102]  # 3 MTP draft tokens

    scheduled_batch = ScheduledRequests()
    scheduled_batch.generation_requests = [req]

    manager = KvCacheConnectorSchedulerOutputManager()
    scheduler_output = manager.build_scheduler_output(scheduled_batch,
                                                      AsyncRequests({}, {}),
                                                      kv_cache_manager)

    assert len(scheduler_output.cached_requests) == 1
    request_data = scheduler_output.cached_requests[0]

    # For generation requests: num_scheduled_tokens = 1 + draft_token_length
    expected_num_scheduled_tokens = 1 + NUM_DRAFT_TOKENS
    assert request_data.num_scheduled_tokens == expected_num_scheduled_tokens, \
        f"Expected {expected_num_scheduled_tokens}, got {request_data.num_scheduled_tokens}"


def test_scheduler_output_block_hashes_read_through():
    """``RequestData.block_hashes`` reflects the chain returned by the KV cache manager.

    The connector path does not recompute hashes Python-side; each scheduler step
    is a pure pass-through of whatever ``commit_and_get_block_hashes`` returns.
    A subsequent step that observes a longer chain simply forwards the longer
    chain. The block-completion semantics (when the next hash actually appears)
    are owned by the C++ KV cache manager and exercised by the C++ unit tests
    for ``commitAndGetBlockHashesForRequest``.
    """
    kv_cache_manager = MagicMock()
    kv_cache_manager.get_cache_indices.return_value = [0]
    # Two consecutive scheduler steps: first sees no full block yet, second sees
    # one full block whose hash has just been committed by the manager.
    kv_cache_manager.commit_and_get_block_hashes.side_effect = [[], [12345]]

    req = MagicMock()
    req.request_id = 42
    req.state = LlmRequestState.GENERATION_IN_PROGRESS
    req.py_draft_tokens = []
    req.get_tokens.return_value = [1, 2, 3]

    scheduled_batch = ScheduledRequests()
    scheduled_batch.generation_requests = [req]

    manager = KvCacheConnectorSchedulerOutputManager()

    output = manager.build_scheduler_output(scheduled_batch,
                                            AsyncRequests({}, {}),
                                            kv_cache_manager)
    assert output.cached_requests[0].block_hashes == []

    req.get_tokens.return_value = [1, 2, 3, 4]
    output = manager.build_scheduler_output(scheduled_batch,
                                            AsyncRequests({}, {}),
                                            kv_cache_manager)
    assert output.cached_requests[0].block_hashes == [12345]

    # Each scheduler step asks the manager exactly once per request; no Python
    # caching layer reshapes the request between calls.
    assert kv_cache_manager.commit_and_get_block_hashes.call_count == 2
    for call in kv_cache_manager.commit_and_get_block_hashes.call_args_list:
        assert call.args == (req, )


def test_scheduler_output_on_rewind_trims_stale_block_ids():
    """``on_rewind`` re-syncs block_ids after specdec rewind frees blocks.

    Covers the full cycle:
      1. Build with blocks [0, 1, 2] (accepted + draft tokens).
      2. Rewind frees block 2 → on_rewind trims block_ids to [0, 1].
      3. Next step allocates a new block 3 → build_scheduler_output must emit
         new_block_ids == [3] and still emit accepted new_tokens.

    Also verifies that ``on_rewind`` does NOT extend ``tokens`` when the
    request's token list grew (accepted tokens added by sampling before
    rewind).  Extending would suppress those tokens from the next
    ``new_tokens`` delta.
    """
    kv_cache_manager = MagicMock()

    req = MagicMock()
    req.request_id = 42
    req.state = LlmRequestState.GENERATION_IN_PROGRESS
    req.py_draft_tokens = []

    scheduled_batch = ScheduledRequests()
    scheduled_batch.generation_requests = [req]

    manager = KvCacheConnectorSchedulerOutputManager()

    # Step 1: normal build — blocks [0, 1, 2], tokens [1..5]
    req.get_tokens.return_value = [1, 2, 3, 4, 5]
    kv_cache_manager.get_cache_indices.return_value = [0, 1, 2]
    kv_cache_manager.commit_and_get_block_hashes.return_value = []
    manager.build_scheduler_output(scheduled_batch, AsyncRequests({}, {}),
                                   kv_cache_manager)
    req_state = manager.requests[42]
    assert req_state.block_ids == [0, 1, 2]
    assert req_state.tokens == [1, 2, 3, 4, 5]

    # Step 2: sampling accepted 1 draft token (token 6) then rewind freed
    # block 2.  req.get_tokens now includes the accepted token [1..6],
    # but live cache indices shrank to [0, 1].
    req.get_tokens.return_value = [1, 2, 3, 4, 5, 6]
    kv_cache_manager.get_cache_indices.return_value = [0, 1]
    manager.on_rewind(req, kv_cache_manager)

    # block_ids trimmed to live indices
    assert req_state.block_ids == [0, 1], \
        f"Expected [0, 1] after rewind, got {req_state.block_ids}"
    # tokens NOT extended — accepted token 6 must still be emitted as new
    assert req_state.tokens == [1, 2, 3, 4, 5], \
        f"on_rewind must not extend tokens; got {req_state.tokens}"

    # Step 3: next step allocates new block 3 for the accepted token.
    # build_scheduler_output must emit new_block_ids == [3] and
    # new_tokens == [6] (the accepted token not yet reported).
    kv_cache_manager.get_cache_indices.return_value = [0, 1, 3]
    output3 = manager.build_scheduler_output(scheduled_batch,
                                             AsyncRequests({}, {}),
                                             kv_cache_manager)
    cached = output3.cached_requests[0]
    assert cached.new_block_ids == [3], \
        f"Expected new_block_ids == [3], got {cached.new_block_ids}"
    assert cached.new_tokens == [6], \
        f"Expected accepted token 6 in new_tokens, got {cached.new_tokens}"


def _make_kv_cache_manager_for_update_resources(is_draft: bool):
    manager = object.__new__(KVCacheManager)
    manager.kv_cache_type = CacheTypeCpp.SELF
    manager.is_draft = is_draft
    manager.kv_connector_manager = MagicMock()
    manager._kv_reserve_draft_tokens = 0
    manager.rewind_kv_cache = MagicMock()
    return manager


def _make_generation_request_for_rewind(rewind_len: int):
    req = MagicMock()
    req.state = LlmRequestState.GENERATION_IN_PROGRESS
    req.py_rewind_len = rewind_len
    req.py_num_accepted_draft_tokens = 0
    return req


def test_update_resources_notifies_connector_only_from_target_kv_manager():
    req = _make_generation_request_for_rewind(rewind_len=1)
    scheduled_batch = ScheduledRequests()
    scheduled_batch.generation_requests = [req]

    manager = _make_kv_cache_manager_for_update_resources(is_draft=False)

    with patch(
            "tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2._update_kv_cache_draft_token_location"
    ):
        manager.update_resources(scheduled_batch)

    manager.rewind_kv_cache.assert_called_once_with(req, 1)
    manager.kv_connector_manager.on_rewind.assert_called_once_with(req, manager)


def test_update_resources_draft_kv_manager_does_not_notify_connector():
    req = _make_generation_request_for_rewind(rewind_len=1)
    scheduled_batch = ScheduledRequests()
    scheduled_batch.generation_requests = [req]

    manager = _make_kv_cache_manager_for_update_resources(is_draft=True)
    manager._kv_reserve_draft_tokens = 2

    manager.update_resources(scheduled_batch)

    assert manager.rewind_kv_cache.call_count == 2
    manager.rewind_kv_cache.assert_any_call(req, 1)
    manager.kv_connector_manager.on_rewind.assert_not_called()


def test_connector_manager_on_rewind_forwards_to_scheduler():
    """KvCacheConnectorManager.on_rewind must forward live_block_ids to the
    external scheduler on rank 0."""
    worker = MagicMock()
    scheduler = MagicMock()

    manager = KvCacheConnectorManager(worker, scheduler=scheduler)

    req = MagicMock()
    req.request_id = 42
    req.get_tokens.return_value = [1, 2, 3]

    kv_cache_manager = MagicMock()
    kv_cache_manager.get_cache_indices.return_value = [0, 1]

    req_state = manager.scheduler_output_manager.requests[42]
    req_state.block_ids = [0, 1, 2]
    req_state.tokens = [1, 2, 3, 4]

    manager.on_rewind(req, kv_cache_manager)

    assert req_state.block_ids == [0, 1]
    assert req_state.tokens == [1, 2, 3]

    # scheduler.on_rewind must be called with the post-rewind live block ids.
    scheduler.on_rewind.assert_called_once()
    forwarded_req, forwarded_ids = scheduler.on_rewind.call_args.args
    assert forwarded_req is req
    assert forwarded_ids == [0, 1]


def test_connector_manager_shutdown_is_ordered_and_idempotent():
    shutdown_order = []
    worker = MagicMock()
    scheduler = MagicMock()
    worker.shutdown.side_effect = lambda: shutdown_order.append("worker")
    scheduler.shutdown.side_effect = lambda: shutdown_order.append("scheduler")
    manager = KvCacheConnectorManager(worker, scheduler=scheduler)

    manager.shutdown()
    manager.shutdown()

    assert shutdown_order == ["worker", "scheduler"]
