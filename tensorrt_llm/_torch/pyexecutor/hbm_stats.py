# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .resource_manager import ResourceManagerType


@dataclasses.dataclass
class KvPoolHbmStats:
    allocated_bytes: int
    used_bytes: int
    free_bytes: int

    def to_dict(self) -> dict:
        return {
            "allocatedBytes": self.allocated_bytes,
            "usedBytes": self.used_bytes,
            "freeBytes": self.free_bytes,
        }


@dataclasses.dataclass
class HbmStats:
    rank: int
    device_id: int
    cuda_total_bytes: int
    cuda_free_bytes: int
    total_hbm_used_bytes: int
    torch_allocated_bytes: int
    torch_reserved_bytes: int
    model_weights_bytes: int
    activations_bytes: int
    other_non_torch_bytes: int
    kv_cache_transfer_buffer_bytes: int
    kv_pools: dict["ResourceManagerType", KvPoolHbmStats]

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "deviceId": self.device_id,
            "cudaTotalBytes": self.cuda_total_bytes,
            "cudaFreeBytes": self.cuda_free_bytes,
            "totalHbmUsedBytes": self.total_hbm_used_bytes,
            "torchAllocatedBytes": self.torch_allocated_bytes,
            "torchReservedBytes": self.torch_reserved_bytes,
            "modelWeightsBytes": self.model_weights_bytes,
            "activationsBytes": self.activations_bytes,
            "otherNonTorchBytes": self.other_non_torch_bytes,
            "kvCacheTransferBufferBytes": self.kv_cache_transfer_buffer_bytes,
            "kvPools": {
                pool.value: pool_stats.to_dict() for pool, pool_stats in self.kv_pools.items()
            },
        }
