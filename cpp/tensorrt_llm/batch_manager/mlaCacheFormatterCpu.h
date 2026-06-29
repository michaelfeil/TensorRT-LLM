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

#include "cacheFormatter.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager::mla_cache_formatter_cpu
{

bool shouldUseTransferBuffer(std::vector<executor::kv_cache::Connection const*> const& connections,
    std::vector<size_t> const& pickUpConnections, uint8_t bufferKind,
    CacheTransBufferManager const& transferBufferManager);

void format(TransferSession& session, std::vector<runtime::ITensor::SharedPtr> const& inputKvCacheBlocks,
    std::vector<size_t> const& pickUpConnections, executor::kv_cache::CacheState const& destConfig,
    executor::kv_cache::CacheState const& selfConfig, int selfIdx, bool transferIndexerKCache,
    CacheTransBufferManager& transferBufferManager, std::vector<size_t> const& bufferEleSizes, size_t pPDomainSize,
    size_t cPDomainSize, int deviceId);

void unformat(TransferSession& session, std::vector<runtime::ITensor::SharedPtr> const& outputBuffers,
    std::vector<size_t> const& pickUpConnections, executor::kv_cache::CacheState const& destConfig,
    executor::kv_cache::CacheState const& selfConfig, int selfIdx, bool transferIndexerKCache,
    CacheTransBufferManager& transferBufferManager, std::vector<size_t> const& bufferEleSizes, int deviceId);

} // namespace tensorrt_llm::batch_manager::kv_cache_manager::mla_cache_formatter_cpu
