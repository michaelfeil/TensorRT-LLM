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

#include "tensorrt_llm/batch_manager/llmRequest.h"

#include <cstddef>
#include <mutex>
#include <string>
#include <unordered_set>

namespace tensorrt_llm::batch_manager::utils
{

/// One-line diagnostic for "session not found in transceiver map" errors.
/// Sampled up to `maxIdsToList` ids from `sessionMap` for context.
/// `senderMutex` is briefly acquired to peek at `cancelledRequests`; caller
/// must not already hold it.
template <typename SessionMap, typename CancelFlagMap, typename CancelledRequestsSet>
std::string formatSessionNotFoundDiagnostic(LlmRequest::RequestIdType requestId, SessionMap const& sessionMap,
    CancelFlagMap const& cancelFlagMap, std::mutex& senderMutex, CancelledRequestsSet const& cancelledRequests,
    std::size_t maxIdsToList = 8)
{
    std::string activeIds;
    std::size_t listed = 0;
    for (auto const& kv : sessionMap)
    {
        if (listed >= maxIdsToList)
        {
            activeIds += ", ...";
            break;
        }
        if (!activeIds.empty())
        {
            activeIds += ", ";
        }
        activeIds += std::to_string(kv.first);
        ++listed;
    }
    bool const hasCancelFlag = cancelFlagMap.find(requestId) != cancelFlagMap.end();

    std::string inCancelledRequestsStr;
    {
        std::unique_lock lk(senderMutex, std::try_to_lock);
        if (lk.owns_lock())
        {
            inCancelledRequestsStr = cancelledRequests.find(requestId) != cancelledRequests.end() ? "true" : "false";
        }
        else
        {
            // try_to_lock failed: another thread holds senderMutex (likely an
            // active send/cancel). Reporting "unknown" keeps the throw site
            // non-blocking; blocking here could stall the diagnostic behind
            // unrelated sender activity.
            inCancelledRequestsStr = "unknown";
        }
    }

    return "requestId=" + std::to_string(requestId) + " activeSessions=" + std::to_string(sessionMap.size()) + " ["
        + activeIds + "]" + " inCancelledRequests=" + inCancelledRequestsStr
        + " hasCancelFlag=" + (hasCancelFlag ? "true" : "false");
}

} // namespace tensorrt_llm::batch_manager::utils
