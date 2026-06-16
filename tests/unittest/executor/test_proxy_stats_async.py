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

import asyncio
import json
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from tensorrt_llm.executor.proxy import GenerationExecutorProxy


class _RemoteResult:

    def __init__(self, value):
        self._value = value

    def remote(self):
        return self._value


class _StatsRpcClient:

    def __init__(self, stats):
        self._stats = deque(stats)

    def fetch_stats_wait_async(self, timeout):
        del timeout
        return _RemoteResult(self._stats.popleft() if self._stats else [])


def _make_proxy(stats):
    proxy = GenerationExecutorProxy.__new__(GenerationExecutorProxy)
    proxy._is_llm_executor = True
    proxy._iter_stats_result = None
    proxy._iter_kv_events_result = None
    proxy.rpc_client = _StatsRpcClient(stats)
    proxy.workers_started = False
    return proxy


async def _collect_async_stats(stats_result):
    return [stat async for stat in stats_result]


def test_proxy_aget_stats_result_from_callback_thread_drains_async():
    payload = {
        "iter": 1,
        "numActiveRequests": 3,
        "maxNumActiveRequests": 8,
        "kvCacheStats": {
            "usedNumBlocks": 4,
            "maxNumBlocks": 16,
        },
    }

    async def run():
        proxy = _make_proxy([[json.dumps(payload)]])
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            stats_result = await loop.run_in_executor(
                executor, lambda: proxy.aget_stats(timeout=0.01))
        stats_result.set_timeout(0.01)
        return await _collect_async_stats(stats_result)

    assert asyncio.run(run()) == [payload]
