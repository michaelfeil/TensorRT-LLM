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

#include "mlaCacheFormatterCpu.h"

#include "tensorrt_llm/batch_manager/dataTransceiver.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/nvtxUtils.h"
#include "tensorrt_llm/executor/cache_transmission/agent_utils/connection.h"
#include "tensorrt_llm/executor/cache_transmission/cacheSplitConcat.h"
#include "tensorrt_llm/runtime/iTensor.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <future>
#include <tuple>
#include <utility>

namespace tensorrt_llm::batch_manager::kv_cache_manager::mla_cache_formatter_cpu
{

namespace
{

using CacheState = executor::kv_cache::CacheState;

struct LayerDomainInfo
{
    int mLayerIdInDomainPP;
    int mRankInDomainPP;
    int mLayerNumInSpecPP;
};

struct BlockDomainInfo
{
    int mBlockIdInDomainCP;
    int mRankInDomainCP;
};

struct MlaCpuTransferLayout
{
    executor::kv_cache::TargetRanksInfo mTargetInfo;
    int mNumLayers;
    int mHeadNum;
    int mKvFactor;
    size_t mLayerStride;
    size_t mHeadStride;
    size_t mKvOffset;
    size_t mHeadEleSize;
    std::vector<LayerDomainInfo> mLayerInfos;
    std::vector<BlockDomainInfo> mBlockInfos;
};

bool hasSelectedAgentConnection(
    std::vector<executor::kv_cache::Connection const*> const& connections, std::vector<size_t> const& pickUpConnections)
{
    return std::any_of(pickUpConnections.begin(), pickUpConnections.end(),
        [&](auto connectionIdx)
        { return dynamic_cast<executor::kv_cache::AgentConnection const*>(connections.at(connectionIdx)) != nullptr; });
}

bool hasSelectedPreAssignedBuffer(std::vector<executor::kv_cache::Connection const*> const& connections,
    std::vector<size_t> const& pickUpConnections, uint8_t bufferKind)
{
    return std::any_of(pickUpConnections.begin(), pickUpConnections.end(),
        [&](auto connectionIdx)
        { return connections.at(connectionIdx)->getPreAssignedBufferId(bufferKind).has_value(); });
}

LayerDomainInfo getLayerDomainInfo(int layerId, executor::kv_cache::TargetRanksInfo const& targetInfo)
{
    int prefixLayerNum = 0;
    for (int ppRank = 0; ppRank < targetInfo.mDomainPPSize; ppRank++)
    {
        auto const layerNumInSpecPP = targetInfo.mPeerLayerNumInDomainPP.at(ppRank);
        auto const nextPrefixLayerNum = prefixLayerNum + layerNumInSpecPP;
        if (layerId >= prefixLayerNum && layerId < nextPrefixLayerNum)
        {
            return LayerDomainInfo{layerId - prefixLayerNum, ppRank, layerNumInSpecPP};
        }
        prefixLayerNum = nextPrefixLayerNum;
    }
    TLLM_THROW("MLA CPU cache formatter could not map layer %d into PP domain", layerId);
}

BlockDomainInfo getBlockDomainInfo(int blockId, int domainCPSize, int inputBlockNum)
{
    if (domainCPSize == 1)
    {
        return BlockDomainInfo{blockId, 0};
    }

    if (common::getEnvUseRoundRobinBlockDistForCP())
    {
        return BlockDomainInfo{blockId / domainCPSize, blockId % domainCPSize};
    }

    int prefixBlockNum = 0;
    for (int cpRank = 0; cpRank < domainCPSize; cpRank++)
    {
        auto const blockNumInDomainCP
            = executor::kv_cache::getBlockNumAccountingForCP(cpRank, domainCPSize, inputBlockNum);
        auto const nextPrefixBlockNum = prefixBlockNum + blockNumInDomainCP;
        if (blockId >= prefixBlockNum && blockId < nextPrefixBlockNum)
        {
            return BlockDomainInfo{blockId - prefixBlockNum, cpRank};
        }
        prefixBlockNum = nextPrefixBlockNum;
    }
    TLLM_THROW("MLA CPU cache formatter could not map block %d into CP domain", blockId);
}

MlaCpuTransferLayout makeMlaCpuTransferLayout(
    CacheState const& destCacheState, CacheState const& selfCacheState, int selfIdx, int blockNum, bool isIndexerKCache)
{
    auto targetInfo = executor::kv_cache::targetIRanks(destCacheState, selfCacheState, selfIdx);
    auto const& selfParallelConfig = selfCacheState.getParallelConfig();
    auto const& selfModelConfig = selfCacheState.getModelConfig();
    auto const selfPPRank = selfIdx / (selfParallelConfig.mTensorParallelism * selfParallelConfig.mContextParallelism);
    auto const numLayers = selfParallelConfig.mAttentionLayerNumPerPP.at(selfPPRank);
    auto const headNum = selfModelConfig.mNbKvHeadsPerLayer.at(0);
    auto const tokensPerBlock = selfModelConfig.mTokensPerBlock;
    auto const dimsPerHead = executor::kv_cache::computeDimsPerHead(selfCacheState, isIndexerKCache);
    auto const kvFactor = selfCacheState.getAttentionConfig().mKvFactor;
    auto const layerStride = static_cast<size_t>(kvFactor) * headNum * tokensPerBlock * dimsPerHead;
    auto const headStride = static_cast<size_t>(tokensPerBlock) * dimsPerHead;
    auto const kvOffset = static_cast<size_t>(headNum) * tokensPerBlock * dimsPerHead;
    auto const headEleSize = static_cast<size_t>(tokensPerBlock) * dimsPerHead;

    std::vector<LayerDomainInfo> layerInfos;
    layerInfos.reserve(numLayers);
    for (int layerId = 0; layerId < numLayers; layerId++)
    {
        layerInfos.push_back(getLayerDomainInfo(layerId, targetInfo));
    }

    std::vector<BlockDomainInfo> blockInfos;
    blockInfos.reserve(blockNum);
    for (int blockId = 0; blockId < blockNum; blockId++)
    {
        blockInfos.push_back(getBlockDomainInfo(blockId, targetInfo.mDomainCPSize, blockNum));
    }

    return MlaCpuTransferLayout{std::move(targetInfo), numLayers, headNum, kvFactor, layerStride, headStride, kvOffset,
        headEleSize, std::move(layerInfos), std::move(blockInfos)};
}

runtime::ITensor::SharedPtr flattenTensor(runtime::ITensor::SharedPtr const& tensor)
{
    TLLM_CHECK(tensor);
    auto flatTensor = runtime::ITensor::view(
        tensor, runtime::ITensor::makeShape({static_cast<runtime::ITensor::DimType64>(tensor->getSize())}));
    return runtime::ITensor::SharedPtr{std::move(flatTensor)};
}

std::vector<runtime::ITensor::SharedPtr> flattenTensors(std::vector<runtime::ITensor::SharedPtr> const& tensors)
{
    std::vector<runtime::ITensor::SharedPtr> flatTensors;
    flatTensors.reserve(tensors.size());
    for (auto const& tensor : tensors)
    {
        flatTensors.push_back(flattenTensor(tensor));
    }
    return flatTensors;
}

void splitKVCacheToTransferBuffer(std::vector<runtime::ITensor::SharedPtr> const& inputKvCacheBlocks,
    std::vector<runtime::ITensor::SharedPtr> const& outputSplitCaches, CacheState const& destCacheState,
    CacheState const& selfCacheState, int selfIdx, runtime::BufferManager const& bufferManager, bool isIndexerKCache)
{
    auto const inputBlockNum = static_cast<int>(inputKvCacheBlocks.size());
    TLLM_CHECK(inputBlockNum > 0);
    auto const layout
        = makeMlaCpuTransferLayout(destCacheState, selfCacheState, selfIdx, inputBlockNum, isIndexerKCache);
    TLLM_CHECK(outputSplitCaches.size()
        == static_cast<size_t>(layout.mTargetInfo.mDomainPPSize * layout.mTargetInfo.mDomainCPSize));
    auto const flatOutputSplitCaches = flattenTensors(outputSplitCaches);

    for (int blockId = 0; blockId < inputBlockNum; blockId++)
    {
        auto const flatInputKvCacheBlock = flattenTensor(inputKvCacheBlocks.at(blockId));
        auto const& blockInfo = layout.mBlockInfos.at(blockId);
        for (int layerId = 0; layerId < layout.mNumLayers; layerId++)
        {
            auto const& layerInfo = layout.mLayerInfos.at(layerId);
            auto outputCacheIdx = static_cast<size_t>(
                blockInfo.mRankInDomainCP * layout.mTargetInfo.mDomainPPSize + layerInfo.mRankInDomainPP);
            auto const outputBlockStride = static_cast<size_t>(layerInfo.mLayerNumInSpecPP) * layout.mLayerStride;
            for (int headId = 0; headId < layout.mHeadNum; headId++)
            {
                auto const inputOffset
                    = static_cast<size_t>(layerId) * layout.mLayerStride + headId * layout.mHeadStride;
                auto const outputOffset = static_cast<size_t>(blockInfo.mBlockIdInDomainCP) * outputBlockStride
                    + static_cast<size_t>(layerInfo.mLayerIdInDomainPP) * layout.mLayerStride
                    + headId * layout.mHeadStride;
                for (int kvId = 0; kvId < layout.mKvFactor; kvId++)
                {
                    auto inputSlice = runtime::ITensor::slice(
                        flatInputKvCacheBlock, inputOffset + kvId * layout.mKvOffset, layout.mHeadEleSize);
                    auto outputSlice = runtime::ITensor::slice(flatOutputSplitCaches.at(outputCacheIdx),
                        outputOffset + kvId * layout.mKvOffset, layout.mHeadEleSize);
                    bufferManager.copy(*inputSlice, *outputSlice);
                }
            }
        }
    }
    bufferManager.getStream().synchronize();
}

void concatTransferBufferToKVCache(std::vector<runtime::ITensor::SharedPtr> const& inputSplitCaches,
    std::vector<runtime::ITensor::SharedPtr> const& outputKvCacheBlocks, CacheState const& destCacheState,
    CacheState const& selfCacheState, int selfIdx, runtime::BufferManager const& bufferManager, bool isIndexerKCache)
{
    auto const outputBlockNum = static_cast<int>(outputKvCacheBlocks.size());
    TLLM_CHECK(outputBlockNum > 0);
    auto const layout
        = makeMlaCpuTransferLayout(destCacheState, selfCacheState, selfIdx, outputBlockNum, isIndexerKCache);
    TLLM_CHECK(inputSplitCaches.size() == static_cast<size_t>(layout.mTargetInfo.mDomainPPSize));
    auto const flatInputSplitCaches = flattenTensors(inputSplitCaches);

    for (int blockId = 0; blockId < outputBlockNum; blockId++)
    {
        auto const flatOutputKvCacheBlock = flattenTensor(outputKvCacheBlocks.at(blockId));
        for (int layerId = 0; layerId < layout.mNumLayers; layerId++)
        {
            auto const& layerInfo = layout.mLayerInfos.at(layerId);
            auto const inputBlockStride = static_cast<size_t>(layerInfo.mLayerNumInSpecPP) * layout.mLayerStride;
            for (int headId = 0; headId < layout.mHeadNum; headId++)
            {
                auto const inputOffset = static_cast<size_t>(blockId) * inputBlockStride
                    + static_cast<size_t>(layerInfo.mLayerIdInDomainPP) * layout.mLayerStride
                    + headId * layout.mHeadStride;
                auto const outputOffset
                    = static_cast<size_t>(layerId) * layout.mLayerStride + headId * layout.mHeadStride;
                for (int kvId = 0; kvId < layout.mKvFactor; kvId++)
                {
                    auto inputSlice = runtime::ITensor::slice(flatInputSplitCaches.at(layerInfo.mRankInDomainPP),
                        inputOffset + kvId * layout.mKvOffset, layout.mHeadEleSize);
                    auto outputSlice = runtime::ITensor::slice(
                        flatOutputKvCacheBlock, outputOffset + kvId * layout.mKvOffset, layout.mHeadEleSize);
                    bufferManager.copy(*inputSlice, *outputSlice);
                }
            }
        }
    }
    bufferManager.getStream().synchronize();
}

template <typename TransferFn>
void runIndexedTransfers(
    size_t transferNum, size_t concurrencyNum, TransferFn const& transferFn, bool rebalanceFinalBatch = false)
{
    TLLM_CHECK(transferNum > 0);
    if (transferNum == 1)
    {
        transferFn(0);
        return;
    }

    if (!common::getEnvEnableReceiveKVCacheParallel())
    {
        TLLM_LOG_DEBUG("Disable parallel receiving of the KV cache.");
        for (size_t i = 0; i < transferNum; i++)
        {
            transferFn(i);
        }
        return;
    }

    concurrencyNum = std::min(std::max(static_cast<size_t>(1), concurrencyNum), transferNum);
    auto remainTransferNum = transferNum;
    while (remainTransferNum > 0)
    {
        auto currentConcurrencyNum = std::min(remainTransferNum, concurrencyNum);
        if (rebalanceFinalBatch && remainTransferNum > concurrencyNum && remainTransferNum < 2 * concurrencyNum)
        {
            currentConcurrencyNum = remainTransferNum - concurrencyNum;
        }

        std::vector<std::future<void>> futures;
        futures.reserve(currentConcurrencyNum);
        for (size_t i = 0; i < currentConcurrencyNum; i++)
        {
            auto const idx = i + (transferNum - remainTransferNum);
            TLLM_CHECK(idx < transferNum);
            futures.push_back(std::async(std::launch::async, transferFn, idx));
        }
        for (auto& future : futures)
        {
            future.get();
        }
        remainTransferNum -= currentConcurrencyNum;
    }
}

template <typename GetBufferIdx>
void sendTransferBuffers(TransferSession& session, int deviceId,
    std::vector<runtime::ITensor::SharedPtr> const& outputSplitCaches, std::vector<size_t> const& pickUpConnections,
    GetBufferIdx const& getBufferIdx)
{
    auto sendBuffer = [&](size_t localIdx)
    {
        NVTX3_SCOPED_RANGE(sendBuffer);
        TLLM_CUDA_CHECK(cudaSetDevice(deviceId));
        auto const connIdx = pickUpConnections.at(localIdx);
        auto const bufferIdx = getBufferIdx(localIdx);
        TLLM_CHECK(connIdx < session.getConnections().size());
        auto& buffer = outputSplitCaches.at(bufferIdx);
        auto const size = buffer->getSizeInBytes();
        auto startTime = LlmRequest::getSteadyClockNow();
        session.send(connIdx, buffer->data(), size);
        auto endTime = LlmRequest::getSteadyClockNow();
        session.appendMeasure(startTime, endTime, size);
    };

    runIndexedTransfers(pickUpConnections.size(), pickUpConnections.size(), sendBuffer);
}

} // namespace

bool shouldUseTransferBuffer(std::vector<executor::kv_cache::Connection const*> const& connections,
    std::vector<size_t> const& pickUpConnections, uint8_t bufferKind,
    CacheTransBufferManager const& transferBufferManager)
{
    if (transferBufferManager.getBufferMemoryType() != runtime::MemoryType::kPINNEDPOOL)
    {
        return false;
    }
    TLLM_CHECK_WITH_INFO(!hasSelectedAgentConnection(connections, pickUpConnections),
        "MLA CPU KV cache transfer buffers are not supported with agent connections.");
    TLLM_CHECK_WITH_INFO(!hasSelectedPreAssignedBuffer(connections, pickUpConnections, bufferKind),
        "MLA CPU KV cache transfer buffers are not supported with pre-assigned connection buffers.");
    return true;
}

void format(TransferSession& session, std::vector<runtime::ITensor::SharedPtr> const& inputKvCacheBlocks,
    std::vector<size_t> const& pickUpConnections, CacheState const& destConfig, CacheState const& selfConfig,
    int selfIdx, bool transferIndexerKCache, CacheTransBufferManager& transferBufferManager,
    std::vector<size_t> const& bufferEleSizes, size_t pPDomainSize, size_t cPDomainSize, int deviceId)
{
    auto& bufferManager = session.getBufferManager();
    auto sendBufferLease = transferBufferManager.assignBufferIndexForSendLease();
    auto cacheBufferId = sendBufferLease.get();
    auto result = transferBufferManager.getOrAllocateSendBuffers(
        cacheBufferId, static_cast<int>(pPDomainSize * cPDomainSize), bufferEleSizes, bufferManager);
    auto& outputSplitCaches = std::get<0>(result);
    auto const bufferCoverTargetNum = std::get<1>(result);
    auto const onlyUseDynamicBuffer = std::get<2>(result);
    TLLM_CHECK_WITH_INFO(bufferCoverTargetNum == pPDomainSize * cPDomainSize && !onlyUseDynamicBuffer,
        "MLA CPU KV cache transfer mode requires full persistent pinned CPU transfer buffers.");
    splitKVCacheToTransferBuffer(
        inputKvCacheBlocks, outputSplitCaches, destConfig, selfConfig, selfIdx, bufferManager, transferIndexerKCache);
    session.setTime(TransferSession::kTimePreprocess);

    // Connections are ordered CP-major (all connections for CP=0, then all for CP=1, etc.) from targetIRanks().
    auto const connectionsPerCPDomain = session.getConnections().size() / cPDomainSize;
    TLLM_CHECK_WITH_INFO(connectionsPerCPDomain > 0, "connectionsPerCPDomain must be > 0");
    auto getMlaBufferIdx = [&](size_t localIdx)
    {
        auto const processIdx = pickUpConnections.at(localIdx);
        auto const cpDomainIdx = processIdx / connectionsPerCPDomain;
        auto const ppDomainIdx = (processIdx % connectionsPerCPDomain) % pPDomainSize;
        return cpDomainIdx * pPDomainSize + ppDomainIdx;
    };

    sendTransferBuffers(session, deviceId, outputSplitCaches, pickUpConnections, getMlaBufferIdx);
}

void unformat(TransferSession& session, std::vector<runtime::ITensor::SharedPtr> const& outputBuffers,
    std::vector<size_t> const& pickUpConnections, CacheState const& destConfig, CacheState const& selfConfig,
    int selfIdx, bool transferIndexerKCache, CacheTransBufferManager& transferBufferManager,
    std::vector<size_t> const& bufferEleSizes, int deviceId)
{
    auto& bufferManager = session.getBufferManager();
    auto const& llmRequest = session.getLlmRequest();
    auto recvBufferLease = transferBufferManager.assignBufferIndexForRecvLease();
    auto cacheBufferId = recvBufferLease.get();
    auto result = transferBufferManager.getOrAllocateRecvBuffers(
        cacheBufferId, static_cast<int>(pickUpConnections.size()), bufferEleSizes, bufferManager);
    auto& recvSplitCaches = std::get<0>(result);
    auto const bufferCoverTargetNum = std::get<1>(result);
    auto const onlyUseDynamicBuffer = std::get<2>(result);
    TLLM_CHECK_WITH_INFO(bufferCoverTargetNum == pickUpConnections.size() && !onlyUseDynamicBuffer,
        "MLA CPU KV cache transfer mode requires full persistent pinned CPU transfer buffers.");
    session.setTime(TransferSession::kTimePreprocess);

    auto recvBufferFun = [&](int deviceId, size_t processIdx)
    {
        NVTX3_SCOPED_RANGE(recvBufferFun);
        TLLM_CUDA_CHECK(cudaSetDevice(deviceId));
        auto startTime = LlmRequest::getSteadyClockNow();
        auto& buffer = recvSplitCaches.at(processIdx);
        llmRequest.updateKvCacheSize(buffer->getSizeInBytes());
        session.recv(pickUpConnections.at(processIdx), buffer->data(), buffer->getSizeInBytes());
        auto endTime = LlmRequest::getSteadyClockNow();
        session.appendMeasure(startTime, endTime, buffer->getSizeInBytes());
    };

    runIndexedTransfers(
        pickUpConnections.size(), pickUpConnections.size(),
        [&](size_t processIdx) { recvBufferFun(deviceId, processIdx); }, /*rebalanceFinalBatch=*/true);
    session.setTime(TransferSession::kTimeTransmissions);

    concatTransferBufferToKVCache(
        recvSplitCaches, outputBuffers, destConfig, selfConfig, selfIdx, bufferManager, transferIndexerKCache);
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager::mla_cache_formatter_cpu
