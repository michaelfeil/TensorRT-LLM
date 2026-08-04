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
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

import tensorrt_llm._torch.disaggregation.b10.mla_root_fanout as fanout_module
from tensorrt_llm._torch.disaggregation.b10.mla_root_fanout import MLARootFanout
from tensorrt_llm._torch.disaggregation.b10.transceiver import B10CacheTransceiver
from tensorrt_llm._torch.disaggregation.base.transfer import KVSlice
from tensorrt_llm._torch.disaggregation.resource.page import KVCachePageTable
from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2
from tensorrt_llm.bindings import LlmRequestState


def _make_fanout(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    tp_rank: int,
    dist: object | None = None,
) -> MLARootFanout:
    monkeypatch.setattr(fanout_module, "mpi_disabled", lambda: False)
    mapping = SimpleNamespace(
        tp_size=tp_size,
        tp_rank=tp_rank,
        tp_group=list(range(tp_size)),
    )
    page_table = KVCachePageTable(tokens_per_block=1, layer_groups=[], pool_groups=[])
    return MLARootFanout(mapping, dist or Mock(), page_table, device_id=0)


def _make_request(request_id: int, state: object) -> SimpleNamespace:
    return SimpleNamespace(
        py_disaggregated_params=None,
        request_id=request_id,
        state=state,
    )


@pytest.mark.parametrize(
    "tp_size,request_id,expected_roots",
    [
        (2, 0, [0]),
        (2, 1, [1]),
        (4, 3, [3]),
        (6, 0, [0, 4]),
        (6, 1, [1, 5]),
        (8, 0, [0, 4]),
        (8, 3, [3, 7]),
        (8, 4, [0, 4]),
    ],
)
def test_receive_leaders_rotate_within_groups(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    request_id: int,
    expected_roots: list[int],
) -> None:
    kv_slice = KVSlice(block_ids_per_layer_groups=[np.asarray([10, 11], dtype=np.int64)])
    roots = []

    for tp_rank in range(tp_size):
        fanout = _make_fanout(monkeypatch, tp_size, tp_rank)
        request = _make_request(request_id, LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS)
        receive_slice = fanout.register_receive_slice(request, kv_slice)
        if receive_slice is kv_slice:
            roots.append(tp_rank)
        else:
            assert receive_slice.block_ids_per_layer_groups[0].size == 0
        assert fanout._pending_slices[request_id] is kv_slice

    assert roots == expected_roots


def test_tp8_creates_fixed_four_rank_process_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fanout_module, "mpi_disabled", lambda: True)
    split = Mock(return_value="subgroup")
    monkeypatch.setattr(fanout_module, "split", split)
    tp_group_pg = Mock()
    tp_group_pg.boxed.return_value = "tp-group"
    mapping = SimpleNamespace(
        tp_size=8,
        tp_rank=5,
        tp_group=list(range(8)),
        tp_group_pg=tp_group_pg,
    )
    page_table = KVCachePageTable(tokens_per_block=1, layer_groups=[], pool_groups=[])

    fanout = MLARootFanout(mapping, Mock(), page_table, device_id=0)

    assert fanout._fanout_group == [4, 5, 6, 7]
    assert fanout._fanout_group_rank == 1
    assert fanout._fanout_group_boxed == "subgroup"
    split.assert_called_once_with(color=1, key=1, pg_boxed="tp-group")


def test_sync_receive_failure_reaches_consensus_before_fanout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_id = 7
    local_outcome = (request_id, True, True, None)
    failed_outcome = (request_id, False, True, "RuntimeError: receive failed")
    dist = Mock()
    dist.tp_allgather.return_value = [local_outcome, failed_outcome]
    fanout = _make_fanout(monkeypatch, tp_size=2, tp_rank=0, dist=dist)
    kv_slice = KVSlice()
    fanout._pending_slices[request_id] = kv_slice
    fanout._fanout = Mock()
    request = _make_request(request_id, LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE)

    with pytest.raises(RuntimeError, match="sync receive failed before fanout"):
        fanout.finish_sync_receive(request)

    assert request.state == LlmRequestState.DISAGG_TRANS_ERROR
    fanout._fanout.assert_not_called()


def test_sync_receive_success_fans_out_after_consensus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_id = 8
    outcome = (request_id, True, True, None)
    dist = Mock()
    dist.tp_allgather.return_value = [outcome, outcome]
    fanout = _make_fanout(monkeypatch, tp_size=2, tp_rank=0, dist=dist)
    kv_slice = KVSlice()
    fanout._pending_slices[request_id] = kv_slice
    fanout._fanout = Mock()
    request = _make_request(request_id, LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE)

    fanout.finish_sync_receive(request)

    fanout._fanout.assert_called_once_with(request_id, kv_slice)


def test_sync_receive_runs_consensus_after_local_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_error = ValueError("receive failed")

    def fail_receive(self: KvCacheTransceiverV2, request: SimpleNamespace) -> None:
        request.state = LlmRequestState.DISAGG_TRANS_ERROR
        raise local_error

    monkeypatch.setattr(
        KvCacheTransceiverV2,
        "request_and_receive_sync",
        fail_receive,
    )
    transceiver = B10CacheTransceiver.__new__(B10CacheTransceiver)
    transceiver._recv_sessions = {}
    transceiver._dist = Mock()
    transceiver._dist.tp_allgather.return_value = [False, False]
    transceiver._mla_root_fanout = Mock()
    transceiver._mla_root_fanout.finish_sync_receive.side_effect = RuntimeError(
        "global receive failure"
    )
    request = _make_request(
        request_id=9,
        state=LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS,
    )
    request.is_generation_only_request = lambda: True

    with pytest.raises(RuntimeError, match="global receive failure"):
        transceiver.request_and_receive_sync(request)

    transceiver._mla_root_fanout.finish_sync_receive.assert_called_once_with(request, local_error)


def test_broadcast_sends_payload_only_from_root(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = torch.arange(8, dtype=torch.uint8)
    gathered = payload + 1
    calls = []

    def fake_allgather(
        input: torch.Tensor,
        group: list[int],
        rank: int,
        group_boxed: object,
        dim: int,
        sizes: list[int],
    ) -> torch.Tensor:
        calls.append((input.clone(), group, rank, group_boxed, dim, sizes))
        return gathered

    monkeypatch.setattr(fanout_module, "_allgather", fake_allgather)
    monkeypatch.setattr(fanout_module, "mpi_disabled", lambda: False)

    result = fanout_module._broadcast_tensor(
        payload,
        group=[4, 5, 6, 7],
        rank=2,
        root=1,
        group_boxed="boxed-group",
    )

    assert torch.equal(result, gathered)
    local_input, group, rank, group_boxed, dim, sizes = calls[0]
    assert local_input.numel() == 0
    assert group == [4, 5, 6, 7]
    assert rank == 2
    assert group_boxed == "boxed-group"
    assert dim == 0
    assert sizes == [0, payload.numel(), 0, 0]
