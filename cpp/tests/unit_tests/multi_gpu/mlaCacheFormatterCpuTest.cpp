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

#include "tensorrt_llm/batch_manager/cacheTransBuffer.h"
#include "tensorrt_llm/batch_manager/cacheTransceiver.h"
#include "tensorrt_llm/batch_manager/dataTransceiver.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/mlaCacheFormatter.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"

#include "cacheTransceiverTestAccessor.h"

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace tr = tensorrt_llm::runtime;
using SizeType32 = tensorrt_llm::runtime::SizeType32;
using LlmRequest = tensorrt_llm::batch_manager::LlmRequest;
using namespace tensorrt_llm::batch_manager;
using namespace tensorrt_llm::batch_manager::kv_cache_manager;
namespace texec = tensorrt_llm::executor;

namespace
{

constexpr char const* kMlaCpuBufferEnv{"TRTLLM_MLA_KVCACHE_TRANSFER_USE_CPU_BUFFER"};

class RecordingConnection : public texec::kv_cache::Connection
{
public:
    void send(texec::kv_cache::DataContext const& ctx, void const* data, size_t size) const override
    {
        (void) ctx;
        (void) size;
        mSendMemoryTypes.push_back(tr::IBuffer::memoryType(data));
        mSendPtrs.push_back(data);
    }

    void recv(texec::kv_cache::DataContext const& ctx, void* data, size_t size) const override
    {
        (void) ctx;
        auto const memoryType = tr::IBuffer::memoryType(data);
        mRecvMemoryTypes.push_back(memoryType);
        mRecvPtrs.push_back(data);
        if (memoryType == tr::MemoryType::kCPU || memoryType == tr::MemoryType::kPINNED
            || memoryType == tr::MemoryType::kPINNEDPOOL)
        {
            std::memset(data, 0, size);
            return;
        }
        TLLM_CUDA_CHECK(cudaMemset(data, 0, size));
    }

    mutable std::vector<tr::MemoryType> mSendMemoryTypes;
    mutable std::vector<tr::MemoryType> mRecvMemoryTypes;
    mutable std::vector<void const*> mSendPtrs;
    mutable std::vector<void const*> mRecvPtrs;
};

class ScopedEnvVar
{
public:
    ScopedEnvVar(char const* name, std::optional<std::string> value)
        : mName{name}
    {
        char const* oldValue = std::getenv(name);
        if (oldValue != nullptr)
        {
            mOldValue = std::string{oldValue};
        }

        if (value.has_value())
        {
            setenv(mName.c_str(), value->c_str(), /*overwrite=*/1);
        }
        else
        {
            unsetenv(mName.c_str());
        }
    }

    ~ScopedEnvVar()
    {
        if (mOldValue.has_value())
        {
            setenv(mName.c_str(), mOldValue->c_str(), /*overwrite=*/1);
        }
        else
        {
            unsetenv(mName.c_str());
        }
    }

private:
    std::string mName;
    std::optional<std::string> mOldValue;
};

auto makeMlaCacheManagerAndState()
{
    constexpr SizeType32 numLayers{1};
    constexpr SizeType32 numHeads{1};
    constexpr SizeType32 sizePerHead{16};
    constexpr SizeType32 tokensPerBlock{4};
    constexpr SizeType32 maxBlocksPerSeq{2};
    constexpr SizeType32 maxBeamWidth{1};
    constexpr SizeType32 sinkTokenLength{0};
    constexpr SizeType32 maxNumSequences{1};
    constexpr SizeType32 blocksInSecondaryPool{0};
    constexpr bool enableBlockReuse{false};
    constexpr bool onboardBlocks{true};
    constexpr bool enablePartialReuse{true};
    constexpr bool copyOnPartialReuse{true};
    constexpr bool enableIndexerKCache{false};
    constexpr nvinfer1::DataType dataType{nvinfer1::DataType::kFLOAT};
    auto const stream = std::make_shared<tr::CudaStream>();

    auto const maxNumTokens = tokensPerBlock * maxBlocksPerSeq;
    using BlocksPerWindow = std::map<SizeType32, std::tuple<SizeType32, SizeType32>>;
    auto const blocksPerWindow
        = BlocksPerWindow{{maxNumTokens, {maxNumSequences * maxBlocksPerSeq, blocksInSecondaryPool}}};

    auto manager = std::make_unique<KVCacheManager>(numLayers, numHeads, sizePerHead, tokensPerBlock, blocksPerWindow,
        maxNumSequences, maxBeamWidth, std::vector<BlockManager::SizeType32>{maxNumTokens}, std::nullopt, dataType,
        sinkTokenLength, stream, maxNumTokens, enableBlockReuse, onboardBlocks, CacheType::kSELF, std::nullopt, nullptr,
        enablePartialReuse, copyOnPartialReuse, /*enableTpMlaReplicatedHostOffload=*/false, std::vector<SizeType32>{},
        /*kvCacheConnectorManager=*/nullptr, enableIndexerKCache);
    manager->allocatePools(/*useUvm=*/false);

    std::vector<SizeType32> attentionLayerNumPerPP{numLayers};
    auto cacheState = std::make_unique<texec::kv_cache::CacheState>(numLayers, numHeads, sizePerHead, tokensPerBlock,
        /*tensorParallelism=*/1, /*pipelineParallelism=*/1, /*contextParallelism=*/1, attentionLayerNumPerPP, dataType,
        texec::kv_cache::CacheState::AttentionType::kMLA, /*kvFactor=*/2, /*enableAttentionDP=*/false, /*DPrank=*/0,
        /*DPsize=*/1, /*enableBlockReuse=*/false, enableIndexerKCache);

    return std::make_pair(std::move(manager), std::move(cacheState));
}

std::shared_ptr<LlmRequest> makeMlaTransferRequest(
    LlmRequest::RequestIdType requestId, texec::kv_cache::CacheState const& cacheState)
{
    constexpr SizeType32 promptLength{1};
    constexpr SizeType32 maxNewTokens{1};
    texec::Request request{VecTokens(promptLength, promptLength), maxNewTokens};

    auto state = std::make_unique<texec::DataTransceiverState>();
    state->setCommState(texec::kv_cache::CommState{std::vector<SizeType32>{0}, /*selfIdx=*/0});
    state->setCacheState(cacheState);
    auto contextPhaseParams = texec::ContextPhaseParams({}, requestId, state.release(), std::nullopt);
    request.setContextPhaseParams(std::move(contextPhaseParams));

    return std::make_shared<LlmRequest>(requestId, std::move(request));
}

struct TransferRecord
{
    std::vector<tr::MemoryType> mMemoryTypes;
    std::vector<void const*> mPtrs;
};

enum class MlaTransferDirection : uint8_t
{
    kFormat,
    kUnformat
};

std::unique_ptr<CacheTransBufferManager> makeMlaTransferBufferManager(KVCacheManager* manager)
{
    auto const memoryType = tensorrt_llm::common::getEnvMLAKVCacheTransferUseCpuBuffer() ? tr::MemoryType::kPINNEDPOOL
                                                                                         : tr::MemoryType::kGPU;
    auto const maxNumTokens = static_cast<size_t>(manager->getBlockManager().getTokensPerBlock() * 2);
    return std::make_unique<CacheTransBufferManager>(
        manager, maxNumTokens, /*transferIndexerKCache=*/false, memoryType);
}

void expectFirstKvCacheBlockIsNonFlat(KVCacheManager& manager, LlmRequest const& llmRequest)
{
    auto blockRange = BlockRange::fromAllBlockIds(manager, llmRequest.mRequestId);
    auto const windowSize = manager.getBlockManager().getPoolWindowSize(0);
    auto blockRangeForWindow = blockRange.getBlockRangeForWindow(windowSize);
    ASSERT_GT(blockRangeForWindow.size(), 0);

    auto blockIt = blockRangeForWindow.begin();
    auto const& shape = blockIt->getShape();
    ASSERT_GT(shape.nbDims, 1);
    ASSERT_GT(shape.d[0], 0);
    EXPECT_GT(blockIt->getSize(), static_cast<size_t>(shape.d[0]));
}

TransferRecord recordMlaTransfers(MlaTransferDirection direction, size_t transferCount = 1)
{
    auto [manager, cacheState] = makeMlaCacheManagerAndState();
    auto llmRequest = makeMlaTransferRequest(/*requestId=*/0, *cacheState);
    manager->addSequence(llmRequest->mRequestId, llmRequest->getNumTokens(/*beam=*/0), /*beamWidth=*/1, llmRequest);
    expectFirstKvCacheBlockIsNonFlat(*manager, *llmRequest);

    std::vector<std::unique_ptr<CacheTransBufferManager>> transBufferManagers;
    std::vector<CacheTransBufferManager*> bufferManagers;
    transBufferManagers.emplace_back(makeMlaTransferBufferManager(manager.get()));
    bufferManagers.push_back(transBufferManagers.back().get());
    MLACacheFormatter formatter{manager.get(), bufferManagers};

    RecordingConnection connection;
    std::vector<texec::kv_cache::Connection const*> connections{&connection};
    texec::DataTransceiverState selfState;
    selfState.setCacheState(*cacheState);
    selfState.setCommState(texec::kv_cache::CommState{std::vector<SizeType32>{0}, /*selfIdx=*/0});
    texec::DataTransceiverState destState{selfState};
    tr::BufferManager bufferManager{std::make_shared<tr::CudaStream>()};
    for (size_t i = 0; i < transferCount; i++)
    {
        TransferSession session{connections, texec::kv_cache::DataContext{/*tag=*/static_cast<int>(43 + i)},
            std::vector<SizeType32>{0}, selfState, destState, bufferManager, /*indexFromEnd=*/0, BlockKey{},
            llmRequest.get()};

        if (direction == MlaTransferDirection::kFormat)
        {
            formatter.format(session);
        }
        else
        {
            formatter.unformat(session);
        }
    }

    if (direction == MlaTransferDirection::kFormat)
    {
        return TransferRecord{connection.mSendMemoryTypes, connection.mSendPtrs};
    }
    return TransferRecord{connection.mRecvMemoryTypes, connection.mRecvPtrs};
}

std::string directionName(testing::TestParamInfo<MlaTransferDirection> const& info)
{
    return info.param == MlaTransferDirection::kFormat ? "Format" : "Unformat";
}

bool hasSingleMpiRank()
{
    tensorrt_llm::mpi::initialize(tensorrt_llm::mpi::MpiThreadSupport::THREAD_MULTIPLE);
    return tensorrt_llm::mpi::MpiComm::world().getSize() == 1;
}

struct TransferBufferManagerInfo
{
    size_t mCount;
    std::optional<tr::MemoryType> mFirstMemoryType;
};

TransferBufferManagerInfo createCacheTransceiverAndGetKvTransferManagerInfo(
    texec::kv_cache::CacheState::AttentionType attentionType)
{
    auto [manager, cacheState] = makeMlaCacheManagerAndState();
    (void) cacheState;

    auto config
        = texec::CacheTransceiverConfig{texec::CacheTransceiverConfig::BackendType::MPI, static_cast<size_t>(0)};
    auto worldConfig = tr::WorldConfig{/*tensorParallelism=*/1, /*pipelineParallelism=*/1,
        /*contextParallelism=*/1};
    auto attentionLayerNumPerPP = std::vector<SizeType32>{1};
    auto transceiver = CacheTransceiver{manager.get(), std::vector<SizeType32>{1}, /*sizePerHead=*/16,
        /*tokensPerBlock=*/4, worldConfig, attentionLayerNumPerPP, nvinfer1::DataType::kFLOAT, attentionType, config};
    auto const managerCount
        = tensorrt_llm::batch_manager::CacheTransceiverTestAccessor::getKvTransferBufferManagerCount(transceiver);
    auto const firstMemoryType = managerCount > 0
        ? std::make_optional(
            tensorrt_llm::batch_manager::CacheTransceiverTestAccessor::getKvTransferBufferManagerMemoryType(
                transceiver, 0))
        : std::nullopt;
    return TransferBufferManagerInfo{managerCount, firstMemoryType};
}

using MLACacheFormatterCpuTest = testing::TestWithParam<MlaTransferDirection>;

TEST_P(MLACacheFormatterCpuTest, UsesGpuTransferBufferByDefaultForNonAgentConnection)
{
    ScopedEnvVar env{kMlaCpuBufferEnv, std::nullopt};
    auto record = recordMlaTransfers(GetParam());

    ASSERT_FALSE(record.mMemoryTypes.empty());
    EXPECT_THAT(record.mMemoryTypes, testing::Each(tr::MemoryType::kGPU));
}

TEST_P(MLACacheFormatterCpuTest, UsesPinnedCpuTransferBufferWhenEnabledForNonAgentConnection)
{
    ScopedEnvVar env{kMlaCpuBufferEnv, std::optional<std::string>{"1"}};
    auto record = recordMlaTransfers(GetParam());

    ASSERT_FALSE(record.mMemoryTypes.empty());
    EXPECT_THAT(record.mMemoryTypes, testing::Each(tr::MemoryType::kPINNEDPOOL));
}

TEST_P(MLACacheFormatterCpuTest, ReusesPersistentPinnedCpuTransferBufferWhenEnabledForNonAgentConnection)
{
    ScopedEnvVar env{kMlaCpuBufferEnv, std::optional<std::string>{"1"}};
    auto record = recordMlaTransfers(GetParam(), /*transferCount=*/2);

    ASSERT_EQ(record.mPtrs.size(), 2);
    EXPECT_EQ(record.mPtrs.at(0), record.mPtrs.at(1));
}

INSTANTIATE_TEST_SUITE_P(FormatAndUnformat, MLACacheFormatterCpuTest,
    testing::Values(MlaTransferDirection::kFormat, MlaTransferDirection::kUnformat), directionName);

struct CacheTransceiverBufferCase
{
    char const* mName;
    texec::kv_cache::CacheState::AttentionType mAttentionType;
    std::optional<std::string> mEnvValue;
    tr::MemoryType mExpectedMemoryType;
};

using CacheTransceiverBufferTest = testing::TestWithParam<CacheTransceiverBufferCase>;

TEST_P(CacheTransceiverBufferTest, AllocatesExpectedTransferBufferManagerForMpi)
{
    if (!hasSingleMpiRank())
    {
        GTEST_SKIP() << "Single-rank MPI is required for constructor-only CacheTransceiver assertions.";
    }

    auto const& param = GetParam();
    ScopedEnvVar env{kMlaCpuBufferEnv, param.mEnvValue};
    auto managerInfo = createCacheTransceiverAndGetKvTransferManagerInfo(param.mAttentionType);

    EXPECT_EQ(managerInfo.mCount, 1);
    ASSERT_TRUE(managerInfo.mFirstMemoryType.has_value());
    EXPECT_EQ(managerInfo.mFirstMemoryType.value(), param.mExpectedMemoryType);
}

std::string cacheTransceiverBufferCaseName(testing::TestParamInfo<CacheTransceiverBufferCase> const& info)
{
    return info.param.mName;
}

INSTANTIATE_TEST_SUITE_P(MlaCpuBuffers, CacheTransceiverBufferTest,
    testing::Values(CacheTransceiverBufferCase{"MlaDefault", texec::kv_cache::CacheState::AttentionType::kMLA,
                        std::nullopt, tr::MemoryType::kGPU},
        CacheTransceiverBufferCase{"MlaCpuBufferEnabled", texec::kv_cache::CacheState::AttentionType::kMLA,
            std::optional<std::string>{"1"}, tr::MemoryType::kPINNEDPOOL},
        CacheTransceiverBufferCase{"DefaultAttentionWithCpuBufferEnabled",
            texec::kv_cache::CacheState::AttentionType::kDEFAULT, std::optional<std::string>{"1"},
            tr::MemoryType::kGPU}),
    cacheTransceiverBufferCaseName);

} // namespace
