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
from typing import Optional, Protocol


class _CacheTransceiverConfigLike(Protocol):
    transceiver_runtime: Optional[str]


PYTHON_CACHE_TRANSCEIVER_RUNTIMES = ("PYTHON", "B10")


def is_python_cache_transceiver_runtime(
    cache_transceiver_config: Optional[_CacheTransceiverConfigLike],
) -> bool:
    return (
        cache_transceiver_config is not None
        and cache_transceiver_config.transceiver_runtime in PYTHON_CACHE_TRANSCEIVER_RUNTIMES
    )
