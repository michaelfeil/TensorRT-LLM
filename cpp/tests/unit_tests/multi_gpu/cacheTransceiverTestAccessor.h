/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "tensorrt_llm/batch_manager/cacheTransceiver.h"

#include <cstddef>
#include <future>
#include <memory>
#include <utility>

namespace tensorrt_llm::batch_manager
{

class CacheTransceiverTestAccessor
{
public:
    static size_t getKvTransferBufferManagerCount(CacheTransceiver const& transceiver)
    {
        return transceiver.mCacheTransBufferManagers.size();
    }

    static runtime::MemoryType getKvTransferBufferManagerMemoryType(CacheTransceiver const& transceiver, size_t index)
    {
        return transceiver.mCacheTransBufferManagers.at(index)->getBufferMemoryType();
    }

    static void addRequesterFuture(
        CacheTransceiver& transceiver, std::shared_ptr<LlmRequest> request, std::future<void>&& future)
    {
        auto const requestId = request->mRequestId;
        transceiver.mRequesterFutures.emplace_back(requestId, std::move(request), std::move(future));
    }

    static bool hasTimedOutRequesterId(CacheTransceiver const& transceiver, LlmRequest::RequestIdType requestId)
    {
        return transceiver.mTimedOutRequesterIds.find(requestId) != transceiver.mTimedOutRequesterIds.end();
    }
};

} // namespace tensorrt_llm::batch_manager
