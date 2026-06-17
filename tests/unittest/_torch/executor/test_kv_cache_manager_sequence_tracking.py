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

import pytest

from tensorrt_llm._torch.attention_backend.sparse.dsa import DSACacheManager
from tensorrt_llm._torch.attention_backend.sparse.rocket import \
    RocketKVCacheManager
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests


class _TrackingKVCacheImpl:

    def __init__(self):
        self.add_sequence_calls = 0
        self.add_token_calls = []
        self.cross_kv = False
        self.live_request_ids = set()

    def sync_transfer_manager_with_buffer_manager(self):
        pass

    def add_sequence(self, request_id, input_length, beam_width, request):
        if request_id in self.live_request_ids:
            raise RuntimeError(f"duplicate add_sequence for {request_id}")
        self.live_request_ids.add(request_id)
        self.add_sequence_calls += 1

    def add_sequence_batch(self, request_infos, requests):
        for request_info, request in zip(request_infos, requests):
            request_id, input_length, beam_width = request_info
            self.add_sequence(request_id, input_length, beam_width, request)

    def add_token(self, request_id):
        self.add_token_calls.append(request_id)

    def get_kv_cache_stats(self):
        return SimpleNamespace(free_num_blocks=100)

    def refresh_blocks(self):
        pass

    def remove_sequence(self, request_id, request, pin_on_release):
        self.live_request_ids.discard(request_id)
        return None


class _TrackingKTCacheManager:

    def __init__(self):
        self.add_token_calls = []

    def add_tokens(self, request_id, token_num):
        self.add_token_calls.append((request_id, token_num))


def _make_kv_cache_manager(num_extra_kv_tokens=0, manager_cls=KVCacheManager):
    manager = manager_cls.__new__(manager_cls)
    manager.impl = _TrackingKVCacheImpl()
    manager.kv_cache_type = None
    manager.is_draft = False
    manager.is_vswa = False
    manager.mapping = SimpleNamespace(cp_config={},
                                      has_cp_helix=lambda: False)
    manager.num_extra_kv_tokens = num_extra_kv_tokens
    manager.kv_connector_manager = None
    manager._active_sequence_owners = {}
    return manager


def _make_rocket_kv_cache_manager(num_extra_kv_tokens=0):
    manager = _make_kv_cache_manager(num_extra_kv_tokens,
                                     RocketKVCacheManager)
    manager.page_size = 2
    manager.kt_cache_manager = _TrackingKTCacheManager()
    return manager


def _make_context_batch(request_id=0, draft_tokens=None):
    req = SimpleNamespace(
        py_request_id=request_id,
        prompt_len=3,
        py_draft_tokens=draft_tokens or [],
        py_beam_width=1,
        sampling_config=SimpleNamespace(beam_width=1),
        is_first_context_chunk=True,
        is_last_context_chunk=True,
    )
    scheduled_batch = ScheduledRequests()
    scheduled_batch.append_context_request(req)
    return scheduled_batch, req


def test_prepare_resources_does_not_add_live_sequence_twice():
    kv_cache_manager = _make_kv_cache_manager(num_extra_kv_tokens=1)
    scheduled_batch, _ = _make_context_batch(draft_tokens=[1, 2])

    kv_cache_manager.prepare_resources(scheduled_batch)
    kv_cache_manager.prepare_resources(scheduled_batch)

    assert kv_cache_manager.impl.add_sequence_calls == 1
    assert kv_cache_manager.impl.add_token_calls == [0, 0, 0]


def test_prepare_resources_rejects_different_live_request_until_free():
    kv_cache_manager = _make_kv_cache_manager()
    scheduled_batch, request = _make_context_batch()
    duplicate_batch, _ = _make_context_batch()

    kv_cache_manager.prepare_resources(scheduled_batch)
    with pytest.raises(RuntimeError, match="already has an active KV sequence"):
        kv_cache_manager.prepare_resources(duplicate_batch)

    kv_cache_manager.free_resources(request)
    kv_cache_manager.prepare_resources(duplicate_batch)

    assert kv_cache_manager.impl.add_sequence_calls == 2


def test_duplicate_dummy_generation_request_does_not_add_draft_tokens_again():
    kv_cache_manager = _make_kv_cache_manager(num_extra_kv_tokens=1)

    kv_cache_manager.add_dummy_requests([0],
                                        is_gen=True,
                                        max_num_draft_tokens=2)
    kv_cache_manager.add_dummy_requests([0],
                                        is_gen=True,
                                        max_num_draft_tokens=2)

    assert kv_cache_manager.impl.add_sequence_calls == 1
    assert kv_cache_manager.impl.add_token_calls == [0, 0, 0]


@pytest.mark.parametrize("draft_manager_cls", [KVCacheManager, DSACacheManager])
def test_duplicate_dummy_generation_request_does_not_add_draft_sequence_again(
        draft_manager_cls):
    kv_cache_manager = _make_kv_cache_manager(num_extra_kv_tokens=1)
    draft_kv_cache_manager = _make_kv_cache_manager(
        manager_cls=draft_manager_cls)

    kv_cache_manager.add_dummy_requests(
        [0],
        is_gen=True,
        max_num_draft_tokens=2,
        draft_kv_cache_manager=draft_kv_cache_manager)
    kv_cache_manager.add_dummy_requests(
        [0],
        is_gen=True,
        max_num_draft_tokens=2,
        draft_kv_cache_manager=draft_kv_cache_manager)

    assert kv_cache_manager.impl.add_sequence_calls == 1
    assert draft_kv_cache_manager.impl.add_sequence_calls == 1
    assert kv_cache_manager.impl.add_token_calls == [0, 0, 0]
    assert draft_kv_cache_manager.impl.add_token_calls == [0, 0, 0]


def test_new_draft_dummy_sequence_gets_draft_tokens_when_primary_exists():
    kv_cache_manager = _make_kv_cache_manager(num_extra_kv_tokens=1)
    draft_kv_cache_manager = _make_kv_cache_manager()

    kv_cache_manager.add_dummy_requests([0],
                                        is_gen=True,
                                        max_num_draft_tokens=2)
    kv_cache_manager.add_dummy_requests(
        [0],
        is_gen=True,
        max_num_draft_tokens=2,
        draft_kv_cache_manager=draft_kv_cache_manager)

    assert kv_cache_manager.impl.add_sequence_calls == 1
    assert draft_kv_cache_manager.impl.add_sequence_calls == 1
    assert kv_cache_manager.impl.add_token_calls == [0, 0, 0]
    assert draft_kv_cache_manager.impl.add_token_calls == [0, 0, 0]


def test_duplicate_rocket_dummy_request_does_not_add_kt_tokens_again():
    kv_cache_manager = _make_rocket_kv_cache_manager()

    kv_cache_manager.add_dummy_requests([0],
                                        is_gen=True,
                                        max_num_draft_tokens=2)
    kv_cache_manager.add_dummy_requests([0],
                                        is_gen=True,
                                        max_num_draft_tokens=2)

    assert kv_cache_manager.impl.add_sequence_calls == 1
    assert len(kv_cache_manager.kt_cache_manager.add_token_calls) == 1


def test_add_dummy_requests_rejects_non_v1_draft_kv_cache_manager():
    kv_cache_manager = _make_kv_cache_manager()

    with pytest.raises(TypeError,
                       match="KVCacheManager or subclass"):
        kv_cache_manager.add_dummy_requests(
            [0], draft_kv_cache_manager=SimpleNamespace())

    assert kv_cache_manager.impl.add_sequence_calls == 0
