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
from unittest.mock import MagicMock, call, patch

import cloudpickle
import mpi4py
import pytest

from tensorrt_llm import mpi_rank
from tensorrt_llm._torch.pyexecutor.connectors import kv_cache_connector
from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector import (
    AsyncRequests, KvCacheConnectorManager,
    KvCacheConnectorSchedulerOutputManager, KvCacheConnectorWorker)
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

        worker.get_finished.reset_mock()
        assert manager.get_finished() == []
        worker.get_finished.assert_not_called()

    run_across_mpi(mpi_pool_executor, test, 2)


def test_connector_manager_get_finished_skips_empty_poll():
    worker = MagicMock()
    manager = KvCacheConnectorManager(worker, scheduler=MagicMock())

    with patch("tensorrt_llm._torch.pyexecutor.connectors."
               "kv_cache_connector.mpi_allgather") as allgather:
        assert manager.get_finished() == []

    worker.get_finished.assert_not_called()
    allgather.assert_not_called()


@pytest.mark.parametrize(
    "async_requests_attribute",
    [
        "new_async_requests",
        "pending_async_requests",
        "local_finished_async_requests",
    ],
)
def test_connector_manager_reports_async_load_in_progress(
        async_requests_attribute):
    manager = KvCacheConnectorManager(MagicMock(), scheduler=MagicMock())
    getattr(manager, async_requests_attribute).loading[7] = MagicMock()

    assert manager.is_loading(7)
    assert not manager.is_loading(8)


def test_executor_defers_cancel_while_connector_load_is_in_progress():
    executor = object.__new__(PyExecutor)
    executor.kv_connector_manager = MagicMock()
    executor.kv_connector_manager.is_loading.return_value = True
    executor.kv_cache_transceiver = MagicMock()
    request = MagicMock()
    request.py_request_id = 7

    assert not executor._try_cancel_request(request)
    executor.kv_connector_manager.is_loading.assert_called_once_with(7)
    executor.kv_cache_transceiver.cancel_request.assert_not_called()


@pytest.mark.parametrize("has_connector", [False, True])
def test_executor_preserves_cancel_without_connector_load(has_connector):
    executor = object.__new__(PyExecutor)
    executor.kv_connector_manager = MagicMock() if has_connector else None
    if executor.kv_connector_manager is not None:
        executor.kv_connector_manager.is_loading.return_value = False
    executor.kv_cache_transceiver = None
    request = MagicMock()
    request.py_request_id = 7

    assert executor._try_cancel_request(request)


def _make_persistence_staging_manager():
    worker = MagicMock()
    worker.uses_secondary_kv_pool_as_persistence_staging.return_value = True
    worker.poll_globally_completed_persistence_leases.return_value = []
    scheduler = MagicMock()
    manager = KvCacheConnectorManager(worker, scheduler=scheduler)
    kv_cache_manager = MagicMock()
    manager.bind_kv_cache_manager(kv_cache_manager)
    return manager, worker, scheduler, kv_cache_manager


def test_persistence_staging_binds_worker_and_skips_request_finish_save():
    manager, worker, scheduler, _ = _make_persistence_staging_manager()
    worker.bind_persistence_lease_manager.assert_called_once_with(manager)

    request = MagicMock()
    request.request_id = 42
    manager.scheduler_output_manager.requests[42]

    assert manager.request_finished(request, [1, 2]) is False
    scheduler.request_finished.assert_not_called()
    scheduler.request_finished_without_save.assert_called_once_with(request)
    worker.request_finished_without_save.assert_called_once_with(42)
    assert 42 not in manager.scheduler_output_manager.requests


def test_persistence_staging_finish_does_not_require_request_cache_indices():
    executor = object.__new__(PyExecutor)
    executor.kv_cache_transceiver = None
    executor.kv_connector_manager = MagicMock()
    executor.kv_connector_manager.uses_secondary_kv_pool_as_persistence_staging.return_value = (
        True)
    executor.kv_connector_manager.request_finished.return_value = False
    executor.kv_cache_manager = MagicMock()
    executor.async_transfer_manager = MagicMock()
    executor.disable_overlap_scheduler = True

    request = MagicMock()
    request.request_id = 42
    request.py_request_id = 42
    request.is_finished = True
    executor.active_requests = [request]

    executor._send_kv_async([request])

    executor.kv_cache_manager.get_cache_indices.assert_not_called()
    executor.kv_connector_manager.request_finished.assert_called_once_with(
        request, [])


def test_persistence_staging_reaps_coordinated_terminal_lease_without_mpi():
    manager, worker, _, kv_cache_manager = _make_persistence_staging_manager()
    lease = MagicMock()
    lease.lease_id = 7
    manager.add_persistence_leases([lease])
    assert manager.take_pending_persistence_leases() == [lease]
    assert manager.take_pending_persistence_leases() == []

    worker.poll_globally_completed_persistence_leases.return_value = [7]
    with patch(
            "tensorrt_llm._torch.pyexecutor.connectors."
            "kv_cache_connector.mpi_allgather") as allgather:
        assert manager.get_finished() == []

    allgather.assert_not_called()
    kv_cache_manager.complete_persistence_leases.assert_called_once_with([7])

    worker.poll_globally_completed_persistence_leases.reset_mock()
    assert manager.get_finished() == []
    worker.poll_globally_completed_persistence_leases.assert_not_called()


def test_persistence_staging_rejects_unknown_coordinated_terminal_lease():
    manager, worker, _, kv_cache_manager = _make_persistence_staging_manager()
    lease = MagicMock()
    lease.lease_id = 9
    manager.add_persistence_leases([lease])
    worker.poll_globally_completed_persistence_leases.return_value = [10]

    with pytest.raises(RuntimeError, match="globally completed unknown"):
        manager.get_finished()

    kv_cache_manager.complete_persistence_leases.assert_not_called()


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
def test_connector_manager_builds_scheduler_output_only_on_leader(
        mpi_pool_executor):

    def test():
        worker = MagicMock()
        scheduler = MagicMock() if mpi_rank() == 0 else None
        if scheduler is not None:
            scheduler.build_connector_meta.return_value = {"request_id": 42}

        manager = KvCacheConnectorManager(worker, scheduler=scheduler)
        manager.scheduler_output_manager.build_scheduler_output = MagicMock(
            return_value="leader-output")
        build_scheduler_output = (
            manager.scheduler_output_manager.build_scheduler_output)

        scheduled_batch = MagicMock()
        kv_cache_manager = MagicMock()
        manager.build_scheduler_output(scheduled_batch, kv_cache_manager)

        if scheduler is not None:
            build_scheduler_output.assert_called_once_with(
                scheduled_batch, manager.new_async_requests, kv_cache_manager)
        else:
            build_scheduler_output.assert_not_called()
            assert manager._scheduler_output is None

        manager.handle_metadata()

        if scheduler is not None:
            scheduler.build_connector_meta.assert_called_once_with(
                "leader-output")
        worker.bind_connector_meta.assert_called_once_with({"request_id": 42})

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
        req.get_num_tokens.return_value = 65

        with patch("tensorrt_llm._torch.pyexecutor.connectors."
                   "kv_cache_connector.mpi_broadcast") as broadcast:
            assert manager.get_num_new_matched_tokens(req, 64) == 0

        worker.can_skip_scheduler_match.assert_called_once_with(65, 64)
        req.get_tokens.assert_not_called()
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


def _make_generation_batch(num_tokens: int):
    req = MagicMock()
    req.request_id = 42
    req.state = LlmRequestState.GENERATION_IN_PROGRESS
    req.py_draft_tokens = []
    req.get_num_tokens.return_value = num_tokens
    req.get_tokens.return_value = list(range(num_tokens))
    req.get_token.side_effect = lambda _beam, position: position
    req.kv_cache_retention_config = None
    req.cache_salt = None

    scheduled_batch = ScheduledRequests()
    scheduled_batch.generation_requests = [req]
    return req, scheduled_batch


def test_sparse_metadata_updates_defer_state_and_worker_hooks_until_boundary():
    worker = MagicMock(spec=KvCacheConnectorWorker)
    worker.supports_rank_local_metadata_skip.return_value = True
    worker.supports_sparse_metadata_updates.return_value = True
    scheduler = MagicMock()
    scheduler.build_connector_meta.return_value = b"metadata"
    manager = KvCacheConnectorManager(worker, scheduler=scheduler)
    req, scheduled_batch = _make_generation_batch(30)
    kv_cache_manager = MagicMock()
    kv_cache_manager.tokens_per_block = 32
    kv_cache_manager.get_cache_indices.return_value = [10]
    kv_cache_manager.commit_and_get_block_hashes.return_value = []

    manager.build_scheduler_output(scheduled_batch, kv_cache_manager)
    manager.handle_metadata()
    with patch.object(kv_cache_connector.torch.cuda,
                      "current_stream") as current_stream:
        assert manager.start_worker_batch(scheduled_batch)
    current_stream.assert_called_once()

    layer = MagicMock()
    layer.layer_idx = 3
    with patch.object(kv_cache_connector.torch.cuda,
                      "current_stream") as current_stream:
        manager.layer_pre_hook(layer)
        manager.layer_post_hook(layer)
    assert current_stream.call_count == 2
    worker.wait_for_layer_load.assert_called_once_with(
        3, current_stream.return_value)
    worker.save_kv_layer.assert_called_once_with(3, current_stream.return_value)

    scheduler.build_connector_meta.reset_mock()
    worker.bind_connector_meta.reset_mock()
    worker.start_load_kv.reset_mock()
    worker.wait_for_layer_load.reset_mock()
    worker.save_kv_layer.reset_mock()

    # Same-block decode has no connector scheduler, PyO3, MPI, or worker work.
    req.get_num_tokens.return_value = 31
    manager.build_scheduler_output(scheduled_batch, kv_cache_manager)
    manager.handle_metadata()
    with patch.object(kv_cache_connector.torch.cuda,
                      "current_stream") as current_stream:
        assert not manager.start_worker_batch(scheduled_batch)
        manager.layer_pre_hook(layer)
        manager.layer_post_hook(layer)
    current_stream.assert_not_called()
    scheduler.build_connector_meta.assert_not_called()
    scheduler.advance_without_worker_metadata.assert_not_called()
    worker.bind_connector_meta.assert_not_called()
    worker.start_load_kv.assert_not_called()
    worker.wait_for_layer_load.assert_not_called()
    worker.save_kv_layer.assert_not_called()

    # Token 32 is sampled but not computed, so the suffix remains deferred.
    req.get_num_tokens.return_value = 32
    manager.build_scheduler_output(scheduled_batch, kv_cache_manager)
    manager.handle_metadata()
    scheduler.build_connector_meta.assert_not_called()

    # Token 33 proves that 32 KV positions exist and carries the full delta.
    req.get_num_tokens.return_value = 33
    kv_cache_manager.get_cache_indices.return_value = [10, 11]
    manager.build_scheduler_output(scheduled_batch, kv_cache_manager)
    manager.handle_metadata()
    output = scheduler.build_connector_meta.call_args.args[0]
    assert output.cached_requests[0].new_tokens == [30, 31, 32]
    assert output.cached_requests[0].new_block_ids == [11]
    worker.bind_connector_meta.assert_called_once_with(b"metadata")


def test_sparse_metadata_updates_force_final_block_completion():
    worker = MagicMock(spec=KvCacheConnectorWorker)
    worker.supports_rank_local_metadata_skip.return_value = True
    worker.supports_sparse_metadata_updates.return_value = True
    scheduler = MagicMock()
    scheduler.build_connector_meta.return_value = b"metadata"
    manager = KvCacheConnectorManager(worker, scheduler=scheduler)
    req, scheduled_batch = _make_generation_batch(31)
    kv_cache_manager = MagicMock()
    kv_cache_manager.tokens_per_block = 32
    kv_cache_manager.get_cache_indices.return_value = [10]
    kv_cache_manager.commit_and_get_block_hashes.side_effect = [[], [12345]]

    manager.build_scheduler_output(scheduled_batch, kv_cache_manager)
    manager.handle_metadata()
    scheduler.build_connector_meta.reset_mock()
    worker.bind_connector_meta.reset_mock()

    # The last forward computes token position 31 and completes block 0. No
    # token-33 iteration exists to make that boundary visible afterward.
    req.state = LlmRequestState.GENERATION_TO_COMPLETE
    req.get_num_tokens.return_value = 32
    req.get_tokens.return_value = list(range(32))
    manager.build_scheduler_output(scheduled_batch, kv_cache_manager)
    manager.handle_metadata()

    scheduler.build_connector_meta.assert_called_once()
    output = scheduler.build_connector_meta.call_args.args[0]
    assert output.cached_requests[0].block_hashes == [12345]
    worker.bind_connector_meta.assert_called_once_with(b"metadata")


def test_sparse_metadata_updates_require_rank_local_skip():
    worker = MagicMock(spec=KvCacheConnectorWorker)
    worker.supports_rank_local_metadata_skip.return_value = False
    worker.supports_sparse_metadata_updates.return_value = True

    with pytest.raises(ValueError, match="require rank-local metadata skip"):
        KvCacheConnectorManager(worker, scheduler=MagicMock())


def test_pending_persistence_lease_submits_pre_forward_without_worker_hooks():
    worker = MagicMock(spec=KvCacheConnectorWorker)
    worker.supports_rank_local_metadata_skip.return_value = True
    worker.supports_sparse_metadata_updates.return_value = True
    worker.uses_secondary_kv_pool_as_persistence_staging.return_value = True
    manager = KvCacheConnectorManager(worker, scheduler=MagicMock())
    lease = MagicMock()
    lease.lease_id = 7
    manager.add_persistence_leases([lease])
    worker.submit_pending_persistence_leases.side_effect = (
        lambda stream: manager.take_pending_persistence_leases())

    executor = object.__new__(PyExecutor)
    executor.kv_connector_manager = manager
    executor.model_engine = MagicMock()
    executor.execution_stream = MagicMock()
    executor._kv_connector_worker_hooks_active = True
    scheduled_batch = ScheduledRequests()

    executor._kv_connector_start_batch(scheduled_batch)
    executor._kv_connector_wait_for_save()

    assert executor._kv_connector_worker_hooks_active is False
    worker.submit_pending_persistence_leases.assert_called_once_with(
        executor.execution_stream)
    assert manager.has_pending_persistence_leases() is False
    worker.start_load_kv.assert_not_called()
    worker.wait_for_save.assert_not_called()
    executor.model_engine.set_forward_pass_callable_enabled.assert_called_once_with(
        False)


def test_executor_skips_idle_metadata_driven_worker_hooks():
    executor = object.__new__(PyExecutor)
    executor.kv_connector_manager = MagicMock()
    executor.model_engine = MagicMock()
    executor._kv_connector_worker_hooks_active = True
    executor.execution_stream = MagicMock()
    scheduled_batch = ScheduledRequests()
    executor.kv_connector_manager.start_worker_batch.return_value = False

    executor._kv_connector_start_batch(scheduled_batch)
    executor._kv_connector_wait_for_save()

    executor.kv_connector_manager.start_worker_batch.assert_called_once_with(
        scheduled_batch)
    executor.model_engine.set_forward_pass_callable_enabled.assert_called_once_with(
        False)
    executor.kv_connector_manager.worker.wait_for_save.assert_not_called()


@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_connector_manager_skips_same_block_metadata_collective(
        mpi_pool_executor):

    def test():
        worker = MagicMock(spec=KvCacheConnectorWorker)
        worker.supports_rank_local_metadata_skip.return_value = True
        scheduler = MagicMock() if mpi_rank() == 0 else None
        if scheduler is not None:
            scheduler.build_connector_meta.return_value = b"metadata"

        manager = KvCacheConnectorManager(worker, scheduler=scheduler)
        req, scheduled_batch = _make_generation_batch(30)
        kv_cache_manager = MagicMock()
        kv_cache_manager.tokens_per_block = 32
        kv_cache_manager.get_cache_indices.return_value = [10]
        kv_cache_manager.commit_and_get_block_hashes.return_value = []

        # The first observation establishes worker-visible state.
        manager.build_scheduler_output(scheduled_batch, kv_cache_manager)
        manager.handle_metadata()
        worker.bind_connector_meta.reset_mock()
        if scheduler is not None:
            scheduler.build_connector_meta.reset_mock()

        # Token 31 remains in the same block and needs only leader state.
        req.get_num_tokens.return_value = 31
        with patch.object(kv_cache_connector, "mpi_broadcast") as broadcast:
            manager.build_scheduler_output(scheduled_batch, kv_cache_manager)
            manager.handle_metadata()
        broadcast.assert_not_called()
        worker.bind_connector_meta.assert_not_called()
        if scheduler is not None:
            scheduler.advance_without_worker_metadata.assert_called_once()
            scheduler.advance_without_worker_metadata.reset_mock()

        # Token 32 is sampled but has not produced KV, so it stays state-only.
        req.get_num_tokens.return_value = 32
        with patch.object(kv_cache_connector, "mpi_broadcast") as broadcast:
            manager.build_scheduler_output(scheduled_batch, kv_cache_manager)
            manager.handle_metadata()
        broadcast.assert_not_called()
        worker.bind_connector_meta.assert_not_called()
        if scheduler is not None:
            scheduler.advance_without_worker_metadata.assert_called_once()
            scheduler.advance_without_worker_metadata.reset_mock()

        # Token 33 proves that 32 positions were computed. The transfer
        # boundary and next-block allocation travel in one real exchange.
        req.get_num_tokens.return_value = 33
        kv_cache_manager.get_cache_indices.return_value = [10, 11]
        real_broadcast = kv_cache_connector.mpi_broadcast
        with patch.object(kv_cache_connector,
                          "mpi_broadcast",
                          wraps=real_broadcast) as broadcast:
            manager.build_scheduler_output(scheduled_batch, kv_cache_manager)
            manager.handle_metadata()
        broadcast.assert_called_once()
        worker.bind_connector_meta.assert_called_once_with(b"metadata")
        if scheduler is not None:
            scheduler.advance_without_worker_metadata.assert_not_called()
            scheduler.build_connector_meta.assert_called_once()
            output = scheduler.build_connector_meta.call_args.args[0]
            assert output.cached_requests[0].new_block_ids == [11]

    run_across_mpi(mpi_pool_executor, test, 2)


@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_connector_manager_mtp_lookahead_waits_for_accepted_block(
        mpi_pool_executor):

    def test():
        worker = MagicMock(spec=KvCacheConnectorWorker)
        worker.supports_rank_local_metadata_skip.return_value = True
        scheduler = MagicMock() if mpi_rank() == 0 else None
        if scheduler is not None:
            scheduler.build_connector_meta.return_value = b"metadata"

        manager = KvCacheConnectorManager(worker, scheduler=scheduler)
        req, scheduled_batch = _make_generation_batch(28)
        req.py_draft_tokens = [0, 0, 0]
        kv_cache_manager = MagicMock()
        kv_cache_manager.tokens_per_block = 32
        kv_cache_manager.get_cache_indices.return_value = [10]
        kv_cache_manager.commit_and_get_block_hashes.return_value = []

        # Establish worker-visible progress below the lookahead boundary.
        manager.build_scheduler_output(scheduled_batch, kv_cache_manager)
        manager.handle_metadata()
        worker.bind_connector_meta.reset_mock()
        if scheduler is not None:
            scheduler.build_connector_meta.reset_mock()

        # MTP3 now schedules through position 32, but token 32 has not been
        # accepted. KVBM cannot transfer that block, so only leader state moves.
        req.get_num_tokens.return_value = 29
        with patch.object(kv_cache_connector, "mpi_broadcast") as broadcast:
            manager.build_scheduler_output(scheduled_batch, kv_cache_manager)
            manager.handle_metadata()
        broadcast.assert_not_called()
        worker.bind_connector_meta.assert_not_called()
        if scheduler is not None:
            scheduler.advance_without_worker_metadata.assert_called_once()
            scheduler.advance_without_worker_metadata.reset_mock()

        # Lookahead allocation is also leader-local; it does not make the
        # still-incomplete block transferable.
        req.get_num_tokens.return_value = 30
        kv_cache_manager.get_cache_indices.return_value = [10, 11]
        with patch.object(kv_cache_connector, "mpi_broadcast") as broadcast:
            manager.build_scheduler_output(scheduled_batch, kv_cache_manager)
            manager.handle_metadata()
        broadcast.assert_not_called()
        worker.bind_connector_meta.assert_not_called()
        if scheduler is not None:
            scheduler.advance_without_worker_metadata.assert_called_once()
            output = scheduler.advance_without_worker_metadata.call_args.args[0]
            assert output.cached_requests[0].new_block_ids == [11]
            scheduler.advance_without_worker_metadata.reset_mock()

        # Token 32 is sampled but has no KV yet.
        req.get_num_tokens.return_value = 32
        with patch.object(kv_cache_connector, "mpi_broadcast") as broadcast:
            manager.build_scheduler_output(scheduled_batch, kv_cache_manager)
            manager.handle_metadata()
        broadcast.assert_not_called()
        worker.bind_connector_meta.assert_not_called()
        if scheduler is not None:
            scheduler.advance_without_worker_metadata.assert_called_once()
            scheduler.advance_without_worker_metadata.reset_mock()

        # Token 33 proves the full accepted block exists on device.
        req.get_num_tokens.return_value = 33
        real_broadcast = kv_cache_connector.mpi_broadcast
        with patch.object(kv_cache_connector,
                          "mpi_broadcast",
                          wraps=real_broadcast) as broadcast:
            manager.build_scheduler_output(scheduled_batch, kv_cache_manager)
            manager.handle_metadata()
        broadcast.assert_called_once()
        worker.bind_connector_meta.assert_called_once_with(b"metadata")
        if scheduler is not None:
            scheduler.advance_without_worker_metadata.assert_not_called()
            scheduler.build_connector_meta.assert_called_once()

    run_across_mpi(mpi_pool_executor, test, 2)


def test_connector_manager_mtp_allocation_rewind_stays_state_only():
    worker = MagicMock(spec=KvCacheConnectorWorker)
    worker.supports_rank_local_metadata_skip.return_value = True
    scheduler = MagicMock()
    scheduler.build_connector_meta.return_value = b"metadata"
    manager = KvCacheConnectorManager(worker, scheduler=scheduler)
    req, scheduled_batch = _make_generation_batch(28)
    req.py_draft_tokens = [0, 0, 0]
    kv_cache_manager = MagicMock()
    kv_cache_manager.tokens_per_block = 32
    kv_cache_manager.get_cache_indices.return_value = [10]
    kv_cache_manager.commit_and_get_block_hashes.return_value = []
    manager._run_on_leader = MagicMock(return_value=b"metadata")

    manager.build_scheduler_output(scheduled_batch, kv_cache_manager)
    manager.handle_metadata()
    manager._run_on_leader.reset_mock()

    # Lookahead allocates block 11 without crossing an accepted-token boundary.
    req.get_num_tokens.return_value = 30
    kv_cache_manager.get_cache_indices.return_value = [10, 11]
    manager.build_scheduler_output(scheduled_batch, kv_cache_manager)
    manager.handle_metadata()
    output = scheduler.advance_without_worker_metadata.call_args.args[0]
    assert output.cached_requests[0].new_block_ids == [11]

    # Rejecting the lookahead frees block 11. The rewind trims both TRT and
    # connector leader state without forcing worker metadata.
    scheduler.advance_without_worker_metadata.reset_mock()
    req.get_num_tokens.return_value = 29
    req.get_tokens.return_value = list(range(29))
    kv_cache_manager.get_cache_indices.return_value = [10]
    manager.on_rewind(req, kv_cache_manager)
    manager.build_scheduler_output(scheduled_batch, kv_cache_manager)
    manager.handle_metadata()

    manager._run_on_leader.assert_not_called()
    scheduler.on_rewind.assert_called_once_with(req, [10])
    scheduler.advance_without_worker_metadata.assert_called_once()

    # A replacement lookahead allocation is emitted as a new leader-local ID.
    scheduler.advance_without_worker_metadata.reset_mock()
    req.get_num_tokens.return_value = 30
    kv_cache_manager.get_cache_indices.return_value = [10, 12]
    manager.build_scheduler_output(scheduled_batch, kv_cache_manager)
    manager.handle_metadata()
    output = scheduler.advance_without_worker_metadata.call_args.args[0]
    assert output.cached_requests[0].new_block_ids == [12]
    manager._run_on_leader.assert_not_called()


def test_connector_manager_rewind_hides_unreported_speculative_block():
    worker = MagicMock(spec=KvCacheConnectorWorker)
    scheduler = MagicMock()
    manager = KvCacheConnectorManager(worker, scheduler=scheduler)
    req, scheduled_batch = _make_generation_batch(28)
    req.py_draft_tokens = [0, 0, 0]
    kv_cache_manager = MagicMock()
    kv_cache_manager.tokens_per_block = 32
    kv_cache_manager.get_cache_indices.return_value = [10]
    kv_cache_manager.commit_and_get_block_hashes.return_value = []
    manager._run_on_leader = MagicMock(return_value=b"metadata")

    manager.build_scheduler_output(scheduled_batch, kv_cache_manager)
    manager.handle_metadata()

    # Sparse metadata can leave a speculative allocation unreported. A
    # partial rewind must not reveal that physical suffix to the connector.
    req.get_num_tokens.return_value = 29
    req.get_tokens.return_value = list(range(29))
    kv_cache_manager.get_cache_indices.reset_mock()
    kv_cache_manager.get_cache_indices.return_value = [10, 11]
    manager.on_rewind(req, kv_cache_manager)

    kv_cache_manager.get_cache_indices.assert_called_once_with(req)
    scheduler.on_rewind.assert_called_once_with(req, [10])


def test_connector_manager_same_block_rewinds_stay_state_only():
    worker = MagicMock(spec=KvCacheConnectorWorker)
    worker.supports_rank_local_metadata_skip.return_value = True
    scheduler = MagicMock()
    scheduler.build_connector_meta.return_value = b"metadata"
    manager = KvCacheConnectorManager(worker, scheduler=scheduler)
    req, scheduled_batch = _make_generation_batch(10)
    kv_cache_manager = MagicMock()
    kv_cache_manager.tokens_per_block = 32
    kv_cache_manager.get_cache_indices.return_value = [10]
    kv_cache_manager.commit_and_get_block_hashes.return_value = []
    manager._run_on_leader = MagicMock(return_value=b"metadata")

    manager.build_scheduler_output(scheduled_batch, kv_cache_manager)
    manager.handle_metadata()
    manager._run_on_leader.reset_mock()

    for _ in range(3):
        manager.on_rewind(req, kv_cache_manager)
        manager.build_scheduler_output(scheduled_batch, kv_cache_manager)
        manager.handle_metadata()

    manager._run_on_leader.assert_not_called()
    assert scheduler.advance_without_worker_metadata.call_count == 3


def test_connector_manager_boundary_rewind_forces_one_exchange():
    worker = MagicMock(spec=KvCacheConnectorWorker)
    worker.supports_rank_local_metadata_skip.return_value = True
    scheduler = MagicMock()
    manager = KvCacheConnectorManager(worker, scheduler=scheduler)
    req, scheduled_batch = _make_generation_batch(33)
    kv_cache_manager = MagicMock()
    kv_cache_manager.tokens_per_block = 32
    kv_cache_manager.get_cache_indices.return_value = [10]
    kv_cache_manager.commit_and_get_block_hashes.return_value = []
    manager._run_on_leader = MagicMock(return_value=b"metadata")

    manager.build_scheduler_output(scheduled_batch, kv_cache_manager)
    manager.handle_metadata()
    manager._run_on_leader.reset_mock()

    req.get_num_tokens.return_value = 32
    req.get_tokens.return_value = list(range(32))
    manager.on_rewind(req, kv_cache_manager)
    manager.build_scheduler_output(scheduled_batch, kv_cache_manager)
    manager.handle_metadata()
    manager._run_on_leader.assert_called_once()

    manager._run_on_leader.reset_mock()
    manager.build_scheduler_output(scheduled_batch, kv_cache_manager)
    manager.handle_metadata()
    manager._run_on_leader.assert_not_called()


@pytest.mark.parametrize("force_reason", ["context", "paused", "async"])
def test_connector_manager_non_decode_work_forces_metadata_exchange(
        force_reason):
    worker = MagicMock(spec=KvCacheConnectorWorker)
    worker.supports_rank_local_metadata_skip.return_value = True
    scheduler = MagicMock()
    scheduler.request_finished.return_value = False
    manager = KvCacheConnectorManager(worker, scheduler=scheduler)
    req, scheduled_batch = _make_generation_batch(10)
    kv_cache_manager = MagicMock()
    kv_cache_manager.tokens_per_block = 32
    kv_cache_manager.get_cache_indices.return_value = [10]

    # A new request exchanges once, then unchanged cached decode is eligible.
    assert not manager._can_skip_metadata_exchange(scheduled_batch,
                                                   kv_cache_manager)
    assert manager._can_skip_metadata_exchange(scheduled_batch,
                                               kv_cache_manager)

    if force_reason == "context":
        scheduled_batch.generation_requests = []
        scheduled_batch.context_requests_last_chunk = [req]
        req.context_current_position = 0
        req.context_remaining_length = 10
        req.context_chunk_size = 10
    elif force_reason == "paused":
        scheduled_batch.paused_requests = [req]
    elif force_reason == "async":
        manager.pending_async_requests.loading[99] = MagicMock()
    else:
        raise AssertionError(f"Unhandled force reason: {force_reason}")

    assert not manager._can_skip_metadata_exchange(scheduled_batch,
                                                   kv_cache_manager)


@pytest.mark.parametrize("saving_async", [False, True])
def test_connector_manager_finish_only_forces_when_saving_async(saving_async):
    worker = MagicMock(spec=KvCacheConnectorWorker)
    worker.supports_rank_local_metadata_skip.return_value = True
    manager = KvCacheConnectorManager(worker, scheduler=MagicMock())
    active_req, scheduled_batch = _make_generation_batch(10)
    finished_req = MagicMock(request_id=99)
    kv_cache_manager = MagicMock()
    kv_cache_manager.tokens_per_block = 32
    kv_cache_manager.get_cache_indices.return_value = [10]

    assert not manager._can_skip_metadata_exchange(scheduled_batch,
                                                   kv_cache_manager)
    assert manager._can_skip_metadata_exchange(scheduled_batch,
                                               kv_cache_manager)
    manager._run_on_leader = MagicMock(return_value=saving_async)

    manager.request_finished(finished_req, [99])

    assert manager._force_metadata_exchange is saving_async
    assert manager._can_skip_metadata_exchange(
        scheduled_batch, kv_cache_manager) is not saving_async
    if saving_async:
        assert manager.new_async_requests.saving == {99: finished_req}
        assert finished_req.state == LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS
    else:
        assert manager.new_async_requests.is_empty
        assert active_req.request_id in manager._metadata_exchange_progress


def test_connector_manager_synchronous_finish_preserves_existing_force():
    manager = KvCacheConnectorManager(MagicMock(), scheduler=MagicMock())
    manager._force_metadata_exchange = True
    manager._run_on_leader = MagicMock(return_value=False)

    manager.request_finished(MagicMock(request_id=99), [99])

    assert manager._force_metadata_exchange


def test_connector_manager_unsupported_skip_avoids_progress_scan():
    worker = MagicMock(spec=KvCacheConnectorWorker)
    worker.supports_rank_local_metadata_skip.return_value = False
    manager = KvCacheConnectorManager(worker, scheduler=MagicMock())
    scheduled_batch = MagicMock(spec=ScheduledRequests)
    scheduled_batch.all_requests.side_effect = AssertionError(
        "unsupported metadata skip must not scan the scheduled batch")

    assert not manager._can_skip_metadata_exchange(scheduled_batch, MagicMock())
    scheduled_batch.all_requests.assert_not_called()


def test_connector_layerwise_transfer_hooks_are_enabled_by_default():
    worker = MagicMock(spec=KvCacheConnectorWorker)

    assert KvCacheConnectorWorker.requires_layerwise_transfer_hooks(worker)


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
    kv_cache_manager.tokens_per_block = 32

    # Create a mock request in generation state with draft tokens
    req = MagicMock()
    req.request_id = 42
    req.state = LlmRequestState.GENERATION_IN_PROGRESS
    req.get_tokens.return_value = [1, 2, 3, 4, 5]  # 5 tokens already generated
    req.get_num_tokens.return_value = 5
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


def test_scheduler_output_only_reads_hashes_at_block_boundaries():
    """Unchanged cumulative block hashes are not rematerialized each step."""
    kv_cache_manager = MagicMock()
    kv_cache_manager.get_cache_indices.return_value = [0, 1]
    kv_cache_manager.tokens_per_block = 4
    # The sampled token is not computed until the following scheduler step.
    kv_cache_manager.commit_and_get_block_hashes.side_effect = [[], [12345]]

    req = MagicMock()
    req.request_id = 42
    req.state = LlmRequestState.GENERATION_IN_PROGRESS
    req.py_draft_tokens = []
    req.get_tokens.return_value = [1, 2, 3]
    req.get_num_tokens.return_value = 3

    scheduled_batch = ScheduledRequests()
    scheduled_batch.generation_requests = [req]

    manager = KvCacheConnectorSchedulerOutputManager()

    output = manager.build_scheduler_output(scheduled_batch,
                                            AsyncRequests({}, {}),
                                            kv_cache_manager)
    assert output.cached_requests[0].block_hashes == []

    req.get_tokens.return_value = [1, 2, 3, 4]
    req.get_num_tokens.return_value = 4
    req.get_token.return_value = 4
    output = manager.build_scheduler_output(scheduled_batch,
                                            AsyncRequests({}, {}),
                                            kv_cache_manager)
    assert output.cached_requests[0].block_hashes is None

    # The next token proves that four KV positions were computed, so the
    # cumulative hash chain advances once.
    req.get_num_tokens.return_value = 5
    req.get_token.return_value = 5
    output = manager.build_scheduler_output(scheduled_batch,
                                            AsyncRequests({}, {}),
                                            kv_cache_manager)
    assert output.cached_requests[0].block_hashes == [12345]

    assert kv_cache_manager.commit_and_get_block_hashes.call_count == 2
    for recorded_call in kv_cache_manager.commit_and_get_block_hashes.call_args_list:
        assert recorded_call.args == (req, )
    assert kv_cache_manager.get_cache_indices.call_count == 1
    assert req.get_tokens.call_count == 1
    assert req.get_token.call_args_list == [call(0, 3), call(0, 4)]


def test_scheduler_output_mtp_allocation_does_not_refresh_generation_hashes():
    kv_cache_manager = MagicMock()
    kv_cache_manager.tokens_per_block = 32
    kv_cache_manager.get_cache_indices.return_value = [10]
    kv_cache_manager.commit_and_get_block_hashes.side_effect = [[], [12345]]

    req, scheduled_batch = _make_generation_batch(28)
    req.py_draft_tokens = [0, 0, 0]
    manager = KvCacheConnectorSchedulerOutputManager()

    initial = manager.build_scheduler_output(scheduled_batch,
                                             AsyncRequests({}, {}),
                                             kv_cache_manager)
    assert initial.cached_requests[0].block_hashes == []

    # MTP3 allocation grows before another accepted block exists. Physical
    # capacity does not change the logical cumulative hash chain.
    req.get_num_tokens.return_value = 30
    kv_cache_manager.get_cache_indices.return_value = [10, 11]
    allocation = manager.build_scheduler_output(scheduled_batch,
                                                AsyncRequests({}, {}),
                                                kv_cache_manager)
    assert allocation.cached_requests[0].new_block_ids == [11]
    assert allocation.cached_requests[0].block_hashes is None
    assert kv_cache_manager.commit_and_get_block_hashes.call_count == 1

    # Token 32 is sampled but still has no KV.
    req.get_num_tokens.return_value = 32
    completion = manager.build_scheduler_output(scheduled_batch,
                                                AsyncRequests({}, {}),
                                                kv_cache_manager)
    assert completion.cached_requests[0].block_hashes is None
    assert kv_cache_manager.commit_and_get_block_hashes.call_count == 1

    # Token 33 proves that 32 KV positions were computed.
    req.get_num_tokens.return_value = 33
    completion = manager.build_scheduler_output(scheduled_batch,
                                                AsyncRequests({}, {}),
                                                kv_cache_manager)
    assert completion.cached_requests[0].block_hashes == [12345]
    assert kv_cache_manager.commit_and_get_block_hashes.call_count == 2


def test_scheduler_output_refreshes_hashes_when_context_allocation_grows():
    """Chunked prefill refreshes hashes when more prompt blocks are allocated."""
    kv_cache_manager = MagicMock()
    kv_cache_manager.tokens_per_block = 4
    kv_cache_manager.get_cache_indices.side_effect = [[0], [0, 1]]
    kv_cache_manager.commit_and_get_block_hashes.side_effect = [[11], [11, 22]]

    req = MagicMock()
    req.request_id = 42
    req.state = LlmRequestState.CONTEXT_INIT
    req.get_num_tokens.return_value = 8
    req.get_tokens.return_value = list(range(8))
    req.context_current_position = 0
    req.context_remaining_length = 8
    req.context_chunk_size = 4

    scheduled_batch = ScheduledRequests()
    scheduled_batch.context_requests_last_chunk = [req]
    manager = KvCacheConnectorSchedulerOutputManager()

    first = manager.build_scheduler_output(scheduled_batch, AsyncRequests({},
                                                                          {}),
                                           kv_cache_manager)
    second = manager.build_scheduler_output(scheduled_batch,
                                            AsyncRequests({}, {}),
                                            kv_cache_manager)

    assert first.new_requests[0].block_hashes == [11]
    assert second.cached_requests[0].new_block_ids == [1]
    assert second.cached_requests[0].block_hashes == [11, 22]
    assert req.get_tokens.call_count == 1
    assert kv_cache_manager.commit_and_get_block_hashes.call_count == 2


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
    kv_cache_manager.tokens_per_block = 2

    # Step 1: normal build — blocks [0, 1, 2], tokens [1..5]
    req.get_tokens.return_value = [1, 2, 3, 4, 5]
    req.get_num_tokens.return_value = 5
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
    req.get_num_tokens.return_value = 6
    req.get_token.return_value = 6
    kv_cache_manager.get_cache_indices.return_value = [0, 1]
    manager.on_rewind(req, [0, 1], 6)

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
    assert kv_cache_manager.commit_and_get_block_hashes.call_count == 2


def test_scheduler_output_on_rewind_preserves_unreported_speculative_growth():
    kv_cache_manager = MagicMock()
    kv_cache_manager.tokens_per_block = 32
    kv_cache_manager.get_cache_indices.return_value = [0, 1]
    kv_cache_manager.commit_and_get_block_hashes.return_value = []

    req, scheduled_batch = _make_generation_batch(60)
    req.py_draft_tokens = [0, 0, 0]
    req.get_tokens.return_value = list(range(60))
    manager = KvCacheConnectorSchedulerOutputManager()

    manager.build_scheduler_output(scheduled_batch, AsyncRequests({}, {}),
                                   kv_cache_manager)
    req.get_tokens.return_value = list(range(62))
    req.get_num_tokens.return_value = 62

    scheduler_live_block_ids = manager.on_rewind(req, [0, 1, 2], 62)

    req_state = manager.requests[req.request_id]
    assert scheduler_live_block_ids is None
    assert req_state.block_ids == [0, 1]

    kv_cache_manager.get_cache_indices.return_value = [0, 1, 2]
    output = manager.build_scheduler_output(scheduled_batch,
                                            AsyncRequests({}, {}),
                                            kv_cache_manager)
    assert output.cached_requests[0].new_block_ids == [2]


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
    """The external scheduler receives only connector-visible live blocks."""
    worker = MagicMock()
    scheduler = MagicMock()

    manager = KvCacheConnectorManager(worker, scheduler=scheduler)

    req = MagicMock()
    req.request_id = 42
    req.get_num_tokens.return_value = 3
    req.get_tokens.return_value = [1, 2, 3]

    kv_cache_manager = MagicMock()
    kv_cache_manager.tokens_per_block = 32
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

    # A speculative suffix that has not crossed a metadata boundary is not
    # connector-visible. When accepted token state also rewinds, forward only
    # the acknowledged prefix rather than the growing physical list.
    scheduler.on_rewind.reset_mock()
    req_state.block_ids = [0, 1]
    req_state.tokens = list(range(64))
    req.get_num_tokens.return_value = 60
    req.get_tokens.return_value = list(range(60))
    kv_cache_manager.get_cache_indices.return_value = [0, 1, 2]

    manager.on_rewind(req, kv_cache_manager)

    scheduler.on_rewind.assert_called_once_with(req, [0, 1])


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
