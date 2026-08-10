/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/runtime/common.h"

#include <cstdint>
#include <utility>
#include <vector>

using SizeType32 = tensorrt_llm::runtime::SizeType32;
using RequestIdType = tensorrt_llm::batch_manager::LlmRequest::RequestIdType;

/// See tensorrt_llm/_torch/pyexecutor/connector.py for details on the Connector API.

namespace tensorrt_llm::batch_manager::kv_connector
{

/// @brief A secondary-pool block kept alive while a connector persists its contents.
struct KvCachePersistenceLease
{
    std::uint64_t leaseId;
    executor::IdType blockHash;
    SizeType32 secondaryBlockIndex;
    SizeType32 priority;
};

/// @brief The KV connector manager. This is passed into the C++ KV Cache Manager when adding sequences.
class KvCacheConnectorManager
{
public:
    KvCacheConnectorManager() = default;
    virtual ~KvCacheConnectorManager() = default;

    /// @brief Handle the getNumNewMatchedTokens call inside the C++ KV Cache Manager.
    /// @return The number of tokens that can be loaded from remote KV cache.
    virtual SizeType32 getNumNewMatchedTokens(LlmRequest const& request, SizeType32 numComputedTokens) = 0;

    /// @brief Whether the connector consumes the secondary pool as persistence staging.
    /// @details Disabled by default so existing connectors retain native host-offload behavior.
    [[nodiscard]] virtual bool usesSecondaryKvPoolAsPersistenceStaging() const
    {
        return false;
    }

    /// @brief Publish leases after their native D2H copies have been ordered before the model stream.
    /// @details Implementations that opt into secondary-pool staging must override this method.
    virtual void addPersistenceLeases(std::vector<KvCachePersistenceLease> const& /*leases*/)
    {
        TLLM_THROW("Connector enabled secondary-pool persistence staging without accepting persistence leases.");
    }
};

} // namespace tensorrt_llm::batch_manager::kv_connector
