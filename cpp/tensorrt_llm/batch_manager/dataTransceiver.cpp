/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dataTransceiver.h"

#include "tensorrt_llm/batch_manager/cacheFormatter.h"
#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/batch_manager/kvCacheUtils.h"
#include "tensorrt_llm/batch_manager/perRequestActivityLog.h"
#include "tensorrt_llm/batch_manager/runtimeBuffers.h"
#include "tensorrt_llm/batch_manager/utils/cacheTransceiverDiagnostics.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/tllmException.h"
#include "tensorrt_llm/common/utils.h"
#include "tensorrt_llm/executor/cache_transmission/agent_utils/connection.h"
#include "tensorrt_llm/executor/cache_transmission/cacheSplitConcat.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <future>
#include <map>
#include <memory>
#include <optional>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace tensorrt_llm::batch_manager
{

using BlockRange = tensorrt_llm::batch_manager::kv_cache_manager::BlockRange;

std::vector<Connection const*> const& TransferSession::getConnections() const
{
    return mConnections;
}

void TransferSession::setConnection(size_t idx, Connection const* conn)
{
    mConnections.at(idx) = conn;
}

DataContext const& TransferSession::getDataContext() const
{
    return mDataContext;
}

void TransferSession::terminateTransfer() noexcept
{
    if (mTransferTerminateFlag != nullptr)
    {
        mTransferTerminateFlag->store(true);
    }
}

executor::DataTransceiverState const& TransferSession::getSelfState() const
{
    return *mSelfState;
}

executor::DataTransceiverState const& TransferSession::getOtherState() const
{
    return mOtherState;
}

runtime::BufferManager const& TransferSession::getBufferManager() const
{
    return *mBufferManager;
}

void TransferSession::send(size_t idx, void const* data, size_t size)
{
    try
    {
        mConnections.at(idx)->send(mDataContext, data, size);
        // Count only successful transfers; a failed send throws and skips this.
        mTotalBytesSent->fetch_add(size, std::memory_order_relaxed);
    }
    catch (std::exception const& e)
    {
        terminateTransfer();
        throw common::RequestSpecificException(
            __FILE__, __LINE__, e.what(), mRequest->mRequestId, common::RequestErrorCode::kNETWORK_ERROR);
    }
}

void TransferSession::recv(size_t idx, void* data, size_t size)
{
    try
    {
        mConnections.at(idx)->recv(mDataContext, data, size);
        // Count only successful transfers; a failed recv throws and skips this.
        mTotalBytesReceived->fetch_add(size, std::memory_order_relaxed);
    }
    catch (std::exception const& e)
    {
        terminateTransfer();
        throw common::RequestSpecificException(
            __FILE__, __LINE__, e.what(), mRequest->mRequestId, common::RequestErrorCode::kNETWORK_ERROR);
    }
}

LlmRequest const& TransferSession::getLlmRequest() const
{
    TLLM_CHECK(mRequest != nullptr);
    return *mRequest;
}

void TransferSession::setLlmRequest(LlmRequest const& llmRequest)
{
    mRequest = &llmRequest;
}

void TransferSession::setTime(TimeNames name)
{
    if (mTimes)
    {
        mTimes->times.at(name) = LlmRequest::getSteadyClockNow();
    }
}

void TransferSession::appendMeasure(LlmRequest::TimePoint start, LlmRequest::TimePoint end, size_t size)
{
    if (mTimes)
    {
        mTimes->measures.emplace_back(Measure{start, end, size});
    }
}

void TransferSession::exportMeasure(std::ofstream& outFile, bool isContext) const
{
    if (!mTimes || mTimes->measures.empty())
    {
        return;
    }
    // write header if not exist
    if (outFile.tellp() == 0)
    {
        outFile << "RequestID,RequestInfo,Preparation,Preprocess,Transmissions,Postprocess";
        for (size_t i = 0; i < mTimes->measures.size(); i++)
        {
            outFile << ",Delay,Duration,Bandwidth(Gbps)";
        }
        outFile << '\n';
    }
    auto transferStart = mRequest->getPerfMetrics().timingMetrics.kvCacheTransferStart;
    using Milliseconds = std::chrono::duration<double, std::milli>;

    // write measures, time is in milliseconds
    TLLM_CHECK(isContext || mRequest->getContextPhaseParams().has_value());
    auto reqId = isContext ? mRequest->mRequestId : mRequest->getContextPhaseParams().value().getReqId();
    outFile << reqId;
    auto previousTime = transferStart;
    for (auto time : mTimes->times)
    {
        if (time == LlmRequest::TimePoint())
        {
            // timepoint is unset, skip
            outFile << ",0.0";
            continue;
        }
        double delay = Milliseconds(time - previousTime).count();
        previousTime = time;
        outFile << "," << delay;
    }
    previousTime = mTimes->times[kTimePreprocess];
    for (auto const& measure : mTimes->measures)
    {
        double delay = Milliseconds(measure.start - previousTime).count();
        double duration = Milliseconds(measure.end - measure.start).count();
        double bandwidth = static_cast<double>(measure.size) * 8.0 / duration / 1e6; // byte, ms => Gbps
        outFile << "," << delay << "," << duration << "," << bandwidth;
    }
    outFile << '\n' << std::flush;
}

using runtime::SizeType32;
using AgentConnectionManager = tensorrt_llm::executor::kv_cache::AgentConnectionManager;
using DataContext = tensorrt_llm::executor::kv_cache::DataContext;

namespace
{

constexpr int32_t kTAG_TYPE_BITS{8};
constexpr int32_t kTAG_TYPE_MASK{(1 << kTAG_TYPE_BITS) - 1};
constexpr int32_t kREQUEST_TAG_BITS{23};
constexpr uint64_t kREQUEST_TAG_MASK{(uint64_t{1} << kREQUEST_TAG_BITS) - 1};

int32_t makeRequestScopedTag(LlmRequest::RequestIdType requestId, int32_t tagType)
{
    return static_cast<int32_t>(((requestId & kREQUEST_TAG_MASK) << kTAG_TYPE_BITS) | (tagType & kTAG_TYPE_MASK));
}

int32_t tagFromRequestId(LlmRequest::RequestIdType requestId)
{
    constexpr int32_t kDATA_TAG{43};
    return makeRequestScopedTag(requestId, kDATA_TAG);
}

int32_t readyTagFromRequestId(LlmRequest::RequestIdType requestId)
{
    return makeRequestScopedTag(requestId, TransceiverTag::kREADY_SIGNAL_TAG);
}

int32_t readyTagFromDataTag(int32_t dataTag)
{
    return (dataTag & ~kTAG_TYPE_MASK) | (TransceiverTag::kREADY_SIGNAL_TAG & kTAG_TYPE_MASK);
}

std::filesystem::path getTransferOutputPath(char const* tag)
{
    namespace fs = std::filesystem;
    auto outputPath = common::getEnvKVCacheTimeOutputPath();
    if (!outputPath.empty())
    {
        auto rank = mpi::MpiComm::world().getRank();
        auto path = fs::path(outputPath);
        fs::create_directories(path);
        return path / ("rank_" + std::to_string(rank) + "_" + tag + ".csv");
    }
    return {};
}

} // namespace

struct ReceiveCacheResource
{
    runtime::BufferManager mBufferManager;
    runtime::CudaEvent mCudaEvent;

    ReceiveCacheResource(runtime::BufferManager&& bufferManager, runtime::CudaEvent cudaEvent)
        : mBufferManager(std::move(bufferManager))
        , mCudaEvent(std::move(cudaEvent))
    {
    }
};

RequestInfo::RequestInfo(LlmRequest::RequestIdType requestId, executor::DataTransceiverState transState)
    : mRequestId{requestId}
    , mTransState{std::move(transState)}
{
}

RequestInfo::RequestInfo(LlmRequest::RequestIdType requestId, executor::DataTransceiverState transState,
    int32_t indexFromEnd, BlockKey const& lastBlockKey)
    : mRequestId{requestId}
    , mIndexFromEnd{indexFromEnd}
    , mLastBlockKey{lastBlockKey}
    , mTransState{std::move(transState)}
{
}

bool RequestInfo::operator==(RequestInfo const& rhs) const
{
    return mRequestId == rhs.mRequestId && mIndexFromEnd == rhs.mIndexFromEnd && mLastBlockKey == rhs.mLastBlockKey
        && mTransState == rhs.mTransState;
}

LlmRequest::RequestIdType RequestInfo::getRequestId() const noexcept
{
    return mRequestId;
}

executor::DataTransceiverState const& RequestInfo::getTransState() const noexcept
{
    return mTransState;
}

void RequestInfo::serialize(RequestInfo const& requestInfo, std::ostream& os)
{
    namespace su = executor::serialize_utils;
    su::serialize(requestInfo.mRequestId, os);
    su::serialize(requestInfo.mIndexFromEnd, os);
    su::serialize(requestInfo.mLastBlockKey, os);
    su::serialize(requestInfo.mTransState, os);
}

RequestInfo RequestInfo::deserialize(std::istream& is)
{
    namespace su = executor::serialize_utils;
    auto requestId = su::deserialize<decltype(mRequestId)>(is);
    auto indexFromEnd = su::deserialize<decltype(mIndexFromEnd)>(is);
    auto lastBlockKey = su::deserialize<decltype(mLastBlockKey)>(is);
    auto transState = su::deserialize<decltype(mTransState)>(is);
    return RequestInfo{requestId, std::move(transState), indexFromEnd, lastBlockKey};
}

std::size_t RequestInfo::serializedSize(RequestInfo const& requestInfo)
{
    namespace su = executor::serialize_utils;
    std::size_t totalSize = 0;
    totalSize += su::serializedSize(requestInfo.mRequestId);
    totalSize += su::serializedSize(requestInfo.mIndexFromEnd);
    totalSize += su::serializedSize(requestInfo.mLastBlockKey);
    totalSize += su::serializedSize(requestInfo.mTransState);
    return totalSize;
}

class CacheSender::Impl
{
public:
    using RequestIdType = LlmRequest::RequestIdType;
    static constexpr char const* kCancelledWithoutSession = "cancelled_without_session";

    Impl(executor::kv_cache::ConnectionManager* manager, SizeType32 selfIndex, CacheTransferLayer cacheLayer)
        : mManager{manager}
        , mSelfState{cacheLayer.getCacheState(), executor::kv_cache::CommState{manager->getCommState()}}
        , mCacheTransferLayer{std::move(cacheLayer)}
        , mBufferManager{std::make_shared<runtime::CudaStream>()}
    {
        TLLM_CHECK(mManager);
        TLLM_CHECK(mManager->getCommState().getSelfIdx() == selfIndex);
        PerRequestActivityLog::instance().setRank(static_cast<int>(selfIndex));
        TLLM_CUDA_CHECK(cudaGetDevice(&mDeviceId));
        mCurrentRequest = std::nullopt;
        mResponseFuture = std::async(std::launch::async, &Impl::response, this);
        int asyncSendThreadNum = common::getEnvKVCacheSendMaxConcurrenceNum();
        for (int i = 0; i < asyncSendThreadNum; i++)
        {
            mAsyncSendFutures.emplace_back(
                std::async(std::launch::async, &Impl::handleAsyncSend, this, std::ref(mAsyncSendResource)));
        }
    }

    [[nodiscard]] std::future<void> sendAsync(std::shared_ptr<LlmRequest> const& llmRequest)
    {
        TLLM_CHECK(llmRequest != nullptr);
        std::promise<void> promise;
        auto future = promise.get_future();
        llmRequest->setKvCacheTransferStart(LlmRequest::getSteadyClockNow());
        getOrCreateRequestCancelFlag(llmRequest->mRequestId);
        {
            bool duplicateRequestId = false;
            size_t readyResponsesSize = 0;
            int64_t currentRequestId = -1;
            {
                std::scoped_lock lkResp(mSenderMutex);
                duplicateRequestId = mReadyResponses.find(llmRequest->mRequestId) != mReadyResponses.end();
                readyResponsesSize = mReadyResponses.size();
                currentRequestId = mCurrentRequest.has_value() ? static_cast<int64_t>(*mCurrentRequest) : -1L;
                if (!duplicateRequestId)
                {
                    mReadyResponses.emplace(llmRequest->mRequestId, Response{llmRequest, std::move(promise)});
                }
            }
            if (duplicateRequestId)
            {
                TLLM_LOG_REQ_ERROR(llmRequest->mRequestId,
                    "Duplicate ready response current_request=%ld ready_responses=%zu", currentRequestId,
                    readyResponsesSize);
                auto err = TLLM_REQUEST_EXCEPTION(llmRequest->mRequestId, common::RequestErrorCode::kUNKNOWN_ERROR,
                    "Duplicate ready response insertion for request %zu", llmRequest->mRequestId);
                promise.set_exception(std::make_exception_ptr(err));
                return future;
            }
            std::unique_lock lkCond(mCondMutex);
            mAnyReady = true;
        }
        mSenderCv.notify_all();
        return future;
    }

    [[nodiscard]] executor::kv_cache::CommState const& getCommState() const
    {
        return mSelfState.getCommState().value();
    }

    void setCommState(executor::kv_cache::CommState commState)
    {
        mSelfState.setCommState(std::move(commState));
    }

    [[nodiscard]] std::shared_ptr<std::atomic<bool>> getOrCreateRequestCancelFlag(LlmRequest::RequestIdType requestId)
    {
        std::scoped_lock<std::mutex> lock(mMtxForMap);
        auto& cancelFlag = mRequestCancelFlags[requestId];
        if (cancelFlag == nullptr)
        {
            // Keep cancellation scoped to one request so a timed-out transfer does not poison later sessions.
            cancelFlag = std::make_shared<std::atomic<bool>>(false);
        }
        return cancelFlag;
    }

    [[nodiscard]] bool cancelInFlightRequest(LlmRequest::RequestIdType requestId)
    {
        std::scoped_lock<std::mutex> lock(mMtxForMap);
        auto it = mRequestCancelFlags.find(requestId);
        if (it == mRequestCancelFlags.end())
        {
            return false;
        }
        it->second->store(true);
        return true;
    }

    void cancelAllInFlightRequests()
    {
        std::scoped_lock<std::mutex> lock(mMtxForMap);
        for (auto const& [requestId, cancelFlag] : mRequestCancelFlags)
        {
            static_cast<void>(requestId);
            if (cancelFlag != nullptr)
            {
                cancelFlag->store(true);
            }
        }
    }

    [[nodiscard]] size_t getCounterpartsCount(LlmRequest::RequestIdType requestId)
    {
        std::unique_lock<std::mutex> lock(mMtxForMap);
        auto it = mRequestToSession.find(requestId);
        if (it == mRequestToSession.end())
        {
            TLLM_THROW("getCounterpartsCount: session not found in mRequestToSession; %s\n%s",
                utils::formatSessionNotFoundDiagnostic(
                    requestId, mRequestToSession, mRequestCancelFlags, mSenderMutex, mCancelledRequests)
                    .c_str(),
                PerRequestActivityLog::instance().dump(requestId).c_str());
        }
        return it->second.getConnections().size();
    }

    [[nodiscard]] bool takeContextKvTransferEventReport(RequestIdType requestId)
    {
        std::scoped_lock lock(mMtxForMap);
        return mReportableContextKvTransferRequestIds.erase(requestId) > 0;
    }

    void recordContextKvTransferEventReportUnlocked(RequestIdType requestId, TransferSession const& session)
    {
        if (mCacheTransferLayer.shouldReportKvCacheTransferEvent(session))
        {
            mReportableContextKvTransferRequestIds.insert(requestId);
        }
    }

    void recordReadyResponseContextFailureReportsUnlocked()
    {
        // A response-thread failure can happen before RequestInfo creates a TransferSession, so there is no
        // needSendCache-based rank report bit yet. Report the ready requests that will receive this exception.
        for (auto const& readyResponse : mReadyResponses)
        {
            if (mCancelledRequests.find(readyResponse.first) != mCancelledRequests.end())
            {
                continue;
            }
            mReportableContextKvTransferRequestIds.insert(readyResponse.first);
        }
    }

    /// `reason` is caller attribution for diagnostics (e.g. "send_complete", "cancel").
    void release(LlmRequest::RequestIdType requestId, char const* reason = "unspecified")
    {
        std::unique_lock<std::mutex> lk(mMtxForMap);
        auto it = mRequestToSession.find(requestId);
        if (it == mRequestToSession.end())
        {
            TLLM_THROW(
                "release: session not found in mRequestToSession (likely already released by another path "
                "such as cancel/timeout). reason=\"%s\" %s\n%s",
                reason,
                utils::formatSessionNotFoundDiagnostic(
                    requestId, mRequestToSession, mRequestCancelFlags, mSenderMutex, mCancelledRequests)
                    .c_str(),
                PerRequestActivityLog::instance().dump(requestId).c_str());
        }
        if (!common::getEnvKVCacheTimeOutputPath().empty())
        {
            if (!mMeasuresFile.is_open())
            {
                auto outputPath = getTransferOutputPath("send");
                mMeasuresFile.open(outputPath);
                TLLM_CHECK_WITH_INFO(
                    mMeasuresFile.is_open(), "Failed to open transfer output file: %s", outputPath.string().c_str());
            }
            it->second.exportMeasure(mMeasuresFile, true);
        }
        mRequestToSession.erase(it);
        mRequestCancelFlags.erase(requestId);
        lk.unlock();

        {
            std::scoped_lock lkResp(mSenderMutex);
            mCancelledRequests.erase(requestId);
            mPendingRequestInfoRejectCounts.erase(requestId);
            mAnyReady = hasReceiveWorkUnlocked();
        }

        // Record a tombstone but intentionally do NOT drop the activity log
        // here: the failure this log targets is a *later* double-release /
        // session-not-found for the same id, whose dump must still show that
        // this request was already released (and when). The per-request slot is
        // reclaimed by the global cap (kMaxTrackedRequests) or an explicit
        // PerRequestActivityLog::release() (e.g. from the Python binding).
        PerRequestActivityLog::instance().record(requestId, "session_released", "CacheSender::release");
    }

    [[nodiscard]] std::optional<RequestInfo> recvRequestInfo()
    {
        auto* agentConnectionManager = dynamic_cast<executor::kv_cache::AgentConnectionManager*>(mManager);
        bool isAgent = agentConnectionManager != nullptr;

        TransceiverTag::Id id;
        RequestInfo info;
        executor::kv_cache::Connection const* connection = nullptr;
        if (isAgent)
        {
            connection = agentConnectionManager->recvConnectionAndRequestInfo(info, mTerminate);
        }
        else
        {
            // Tag 19 does not carry a request id yet. Drain the request-info envelope and reject stale requests after
            // decoding the id instead of canceling this anonymous receive for per-request cleanup.
            connection = mManager->recvConnect(DataContext{TransceiverTag::kID_TAG, mTerminate}, &id, sizeof(id));
        }
        if (connection == nullptr)
        {
            return std::nullopt;
        }

        if (!isAgent)
        {
            TLLM_CHECK(id == TransceiverTag::Id::REQUEST_SEND);
            std::uint64_t infoSize{0};
            std::string serializedInfo;
            try
            {
                connection->recv(DataContext{TransceiverTag::kINFO_SIZE_TAG, mTerminate}, &infoSize, sizeof(infoSize));
                serializedInfo.resize(infoSize);
                connection->recv(DataContext{TransceiverTag::kINFO_TAG, mTerminate}, serializedInfo.data(), infoSize);
            }
            catch (std::exception const&)
            {
                if (mTerminate.load())
                {
                    return std::nullopt;
                }
                throw;
            }
            std::istringstream iss(serializedInfo);
            info = RequestInfo::deserialize(iss);
        }

        auto requestId = info.getRequestId();
        try
        {
            if (!isAgent)
            {
                char const* discardReason{nullptr};
                {
                    std::scoped_lock lk(mSenderMutex, mMtxForMap);
                    discardReason = getRequestInfoDiscardReasonUnlocked(requestId);
                }
                if (discardReason != nullptr)
                {
                    auto const rejectCount = getRequestInfoRejectCount(info);
                    logDiscardedRequestInfo(requestId, discardReason);
                    sendRequestInfoRejectReadySignal(connection, requestId, discardReason);
                    finishDiscardedRequestInfo(requestId, discardReason, rejectCount);
                    return std::nullopt;
                }
            }
            mCacheTransferLayer.validateSupport(info.getTransState());

            auto allCounterparts = mCacheTransferLayer.computeCounterparts(
                mSelfState.getCommState().value().getSelfIdx(), info.getTransState());

            auto peerSelfIdx = info.getTransState().getCommState()->getSelfIdx();
            int peerIdx = std::distance(
                allCounterparts.begin(), std::find(allCounterparts.begin(), allCounterparts.end(), peerSelfIdx));

            TLLM_CHECK_WITH_INFO(peerIdx < static_cast<int>(allCounterparts.size()),
                "Peer rank %d not found in expected counterparts", peerSelfIdx);
            auto const peerOffset = static_cast<size_t>(peerIdx);
            char const* discardReason{nullptr};
            bool duplicatePeerConnection{false};
            {
                std::scoped_lock lk(mSenderMutex, mMtxForMap);
                auto it = mRequestToSession.find(requestId);
                if (it == mRequestToSession.end())
                {
                    if (!isAgent)
                    {
                        discardReason = getRequestInfoDiscardReasonUnlocked(requestId);
                    }
                    if (discardReason == nullptr)
                    {
                        auto& requestCancelFlag = mRequestCancelFlags[requestId];
                        if (requestCancelFlag == nullptr)
                        {
                            requestCancelFlag = std::make_shared<std::atomic<bool>>(false);
                        }
                        auto session = TransferSession(std::vector<Connection const*>(allCounterparts.size(), nullptr),
                            DataContext{tagFromRequestId(requestId), *requestCancelFlag}, allCounterparts, mSelfState,
                            info.getTransState(), mBufferManager, info.getIndexFromEnd(), info.getLastBlockKey(),
                            nullptr, !common::getEnvKVCacheTimeOutputPath().empty(), requestCancelFlag);
                        session.setTime(TransferSession::kTimeRequestInfo);
                        it = mRequestToSession.emplace(requestId, std::move(session)).first;
                        PerRequestActivityLog::instance().record(
                            requestId, "session_added", "CacheSender::recvRequestInfo");
                    }
                }
                if (discardReason == nullptr && !isAgent && it->second.getConnections().at(peerOffset) != nullptr)
                {
                    duplicatePeerConnection = true;
                }
                if (discardReason == nullptr && !duplicatePeerConnection)
                {
                    mCurrentRequest = requestId;
                    it->second.setConnection(peerOffset, connection);
                    recordContextKvTransferEventReportUnlocked(requestId, it->second);
                }
            }
            if (duplicatePeerConnection)
            {
                char const* reason = "duplicate_peer_connection";
                logDiscardedRequestInfo(requestId, reason);
                sendRequestInfoRejectReadySignal(connection, requestId, reason);
                return std::nullopt;
            }
            if (discardReason != nullptr)
            {
                logDiscardedRequestInfo(requestId, discardReason);
                sendRequestInfoRejectReadySignal(connection, requestId, discardReason);
                finishDiscardedRequestInfo(requestId, discardReason, allCounterparts.size());
                return std::nullopt;
            }
            return info;
        }
        catch (std::exception const& e)
        {
            TLLM_LOG_REQ_ERROR(requestId, "Exception while handling request info: %s", e.what());
            if (!isAgent)
            {
                sendRequestInfoRejectReadySignal(connection, requestId, "request_info_failure");
            }
            failAndRemoveResponseById(requestId, "request_info_failure", std::current_exception());
            return std::nullopt;
        }
        catch (...)
        {
            TLLM_LOG_REQ_ERROR(requestId, "Exception while handling request info");
            if (!isAgent)
            {
                sendRequestInfoRejectReadySignal(connection, requestId, "request_info_failure");
            }
            failAndRemoveResponseById(requestId, "request_info_failure", std::current_exception());
            return std::nullopt;
        }
    }

    void sendSync(LlmRequest const& llmRequest)
    {
        PerRequestActivityLog::instance().record(llmRequest.mRequestId, "send_started", "CacheSender::sendSync");
        TransferSession* session = nullptr;
        {
            std::unique_lock<std::mutex> lk(mMtxForMap);
            auto it = mRequestToSession.find(llmRequest.mRequestId);
            if (it == mRequestToSession.end())
            {
                TLLM_THROW("CacheSender::sendSync: session for request %zu missing from mRequestToSession.\n%s",
                    static_cast<size_t>(llmRequest.mRequestId),
                    PerRequestActivityLog::instance().dump(llmRequest.mRequestId).c_str());
            }
            session = std::addressof(it->second);
            recordContextKvTransferEventReportUnlocked(llmRequest.mRequestId, *session);
        }

        session->setLlmRequest(llmRequest);
        mCacheTransferLayer.format(*session);
        llmRequest.setKvCacheTransferEnd(LlmRequest::getSteadyClockNow());
        {
            char detail[PerRequestActivityLog::kDetailLen];
            std::snprintf(detail, sizeof(detail), "bytes=%zu connections=%zu", session->getTotalBytesSent(),
                session->getConnections().size());
            PerRequestActivityLog::instance().recordDetail(
                llmRequest.mRequestId, "send_completed", "CacheSender::sendSync", detail);
        }
    }

    bool cancelRequest(LlmRequest const& llmRequest)
    {
        PerRequestActivityLog::instance().record(
            llmRequest.mRequestId, "cancel_requested", "CacheSender::cancelRequest");
        bool isCancelled = false;
        std::optional<CancelledResponse> cancelledResponse;
        {
            std::scoped_lock lkResp(mSenderMutex, mMtxForMap);
            auto it = mReadyResponses.find(llmRequest.mRequestId);
            // If the request is not the current request and already in the ready queue, we can cancel it.
            if (it != mReadyResponses.end()
                && (!mCurrentRequest.has_value() || getCurrentRequestId() != llmRequest.mRequestId))
            {
                if (isAgentConnectionManager())
                {
                    mCancelledRequests.insert(llmRequest.mRequestId);
                    mAnyReady = true;
                }
                else
                {
                    std::vector<Connection const*> connectedConnections;
                    if (auto sessionIt = mRequestToSession.find(llmRequest.mRequestId);
                        sessionIt != mRequestToSession.end())
                    {
                        recordContextKvTransferEventReportUnlocked(llmRequest.mRequestId, sessionIt->second);
                        connectedConnections = getConnectedConnections(sessionIt->second);
                        mRequestToSession.erase(sessionIt);
                    }
                    // A late request-info envelope will be rejected as missing_ready_response. Do not leave a
                    // cancellation tombstone that can make the response thread wait on anonymous tag 19.
                    mCancelledRequests.erase(llmRequest.mRequestId);
                    mPendingRequestInfoRejectCounts.erase(llmRequest.mRequestId);
                    cancelledResponse.emplace(CancelledResponse{
                        llmRequest.mRequestId, std::move(it->second), std::move(connectedConnections)});
                    mReadyResponses.erase(it);
                    mRemainSendCount.erase(llmRequest.mRequestId);
                    mRequestCancelFlags.erase(llmRequest.mRequestId);
                    mAnyReady = hasReceiveWorkUnlocked();
                }
                isCancelled = true;
            }
        }

        if (!isCancelled && cancelInFlightRequest(llmRequest.mRequestId))
        {
            std::scoped_lock lkResp(mSenderMutex);
            mCancelledRequests.insert(llmRequest.mRequestId);
            isCancelled = true;
            mAnyReady = hasReceiveWorkUnlocked();
        }

        if (isCancelled)
        {
            std::scoped_lock lk(mMtxForMap);
            auto it = mRequestToSession.find(llmRequest.mRequestId);
            if (it != mRequestToSession.end())
            {
                recordContextKvTransferEventReportUnlocked(llmRequest.mRequestId, it->second);
            }
        }
        if (isCancelled)
        {
            mSenderCv.notify_all();
        }
        if (cancelledResponse.has_value())
        {
            completeCancelledResponse(*cancelledResponse);
        }
        if (!isCancelled)
        {
            TLLM_LOG_REQ_WARNING(llmRequest.mRequestId, "Cannot cancel request");
        }
        return isCancelled;
    }

    void sendReadySignal(LlmRequest::RequestIdType requestId, bool isReady)
    {
        TransferSession* session = nullptr;
        {
            std::unique_lock<std::mutex> lock(mMtxForMap);
            auto it = mRequestToSession.find(requestId);
            TLLM_CHECK(it != mRequestToSession.end());
            session = std::addressof(it->second);
        }
        auto const& connections = session->getConnections();
        for (size_t i = 0; i < connections.size(); i++)
        {
            auto* agentConnectionManager = dynamic_cast<executor::kv_cache::AgentConnectionManager*>(mManager);
            if (agentConnectionManager)
            {
                auto* agentConnection = dynamic_cast<executor::kv_cache::AgentConnection const*>(connections.at(i));
                TLLM_CHECK(agentConnection);
                agentConnection->sendReadySignal(
                    executor::kv_cache::DataContext{TransceiverTag::kREADY_SIGNAL_TAG}, isReady);
            }
            else
            {
                connections.at(i)->send(
                    executor::kv_cache::DataContext{readyTagFromRequestId(requestId)}, &isReady, sizeof(isReady));
            }
        }
    }

    ~Impl()
    {
        terminate();
    }

private:
    struct Response
    {
        // shared_ptr so this struct co-owns the request until the promise resolves;
        // protects worker-side dereferences and the promise itself from premature destruction.
        std::shared_ptr<LlmRequest> mRequest;
        std::promise<void> mPromise;
    };

    struct CancelledResponse
    {
        RequestIdType mRequestId;
        Response mResponse;
        std::vector<Connection const*> mConnectedConnections;
    };

    struct AsyncSendResource
    {
        std::deque<Response> mSendQueue;
        std::mutex mMtxForQueue;
        std::condition_variable mCVforQueue;
        std::atomic<bool> mTerminate{false};
    };

    [[nodiscard]] bool isAgentConnectionManager() const
    {
        return dynamic_cast<executor::kv_cache::AgentConnectionManager*>(mManager) != nullptr;
    }

    [[nodiscard]] bool hasReceiveWorkUnlocked() const
    {
        return !mReadyResponses.empty();
    }

    [[nodiscard]] bool hasReceiveWork()
    {
        std::scoped_lock lk(mSenderMutex);
        return hasReceiveWorkUnlocked();
    }

    void setResponsePromiseValue(RequestIdType requestId, Response& response) noexcept
    {
        try
        {
            response.mPromise.set_value();
        }
        catch (std::exception const& e)
        {
            TLLM_LOG_REQ_WARNING(requestId, "Failed to set response promise value: %s", e.what());
        }
        catch (...)
        {
            TLLM_LOG_REQ_WARNING(requestId, "Failed to set response promise value");
        }
    }

    void setResponsePromiseException(
        RequestIdType requestId, Response& response, std::exception_ptr responseException) noexcept
    {
        try
        {
            response.mPromise.set_exception(std::move(responseException));
        }
        catch (std::exception const& e)
        {
            TLLM_LOG_REQ_WARNING(requestId, "Failed to set response promise exception: %s", e.what());
        }
        catch (...)
        {
            TLLM_LOG_REQ_WARNING(requestId, "Failed to set response promise exception");
        }
    }

    [[nodiscard]] size_t getRequestInfoRejectCount(RequestInfo const& info) const
    {
        auto allCounterparts = mCacheTransferLayer.computeCounterparts(
            mSelfState.getCommState().value().getSelfIdx(), info.getTransState());
        return std::max<size_t>(allCounterparts.size(), 1);
    }

    [[nodiscard]] char const* getRequestInfoDiscardReasonUnlocked(RequestIdType requestId) const
    {
        if (mCancelledRequests.find(requestId) != mCancelledRequests.end()
            && mRequestToSession.find(requestId) == mRequestToSession.end())
        {
            return kCancelledWithoutSession;
        }
        if (mReadyResponses.find(requestId) == mReadyResponses.end())
        {
            return "missing_ready_response";
        }
        return nullptr;
    }

    void logDiscardedRequestInfo(RequestIdType requestId, char const* reason) const
    {
        TLLM_LOG_REQ_WARNING(requestId, "Discarding stale context transfer request info reason=%s", reason);
    }

    void failAndRemoveResponseById(
        RequestIdType requestId, char const* reason, std::exception_ptr responseException) noexcept
    {
        std::optional<Response> response;
        {
            std::scoped_lock lk(mSenderMutex, mMtxForMap);
            auto it = mReadyResponses.find(requestId);
            if (it == mReadyResponses.end())
            {
                return;
            }
            response.emplace(std::move(it->second));
            mReadyResponses.erase(it);
            mCancelledRequests.erase(requestId);
            mRemainSendCount.erase(requestId);
            if (mCurrentRequest.has_value() && getCurrentRequestId() == requestId)
            {
                mCurrentRequest = std::nullopt;
            }
            mRequestToSession.erase(requestId);
            mRequestCancelFlags.erase(requestId);
            mReportableContextKvTransferRequestIds.insert(requestId);
            mPendingRequestInfoRejectCounts.erase(requestId);
            mAnyReady = hasReceiveWorkUnlocked();
        }

        PerRequestActivityLog::instance().record(requestId, "request_info_failed", "CacheSender::recvRequestInfo");
        static_cast<void>(reason);
        setResponsePromiseException(requestId, *response, std::move(responseException));
    }

    void failAndRemoveAllReadyResponses(char const* reason, std::exception_ptr responseException) noexcept
    {
        std::vector<std::pair<RequestIdType, Response>> responses;
        {
            std::scoped_lock lk(mSenderMutex, mMtxForMap);
            try
            {
                recordReadyResponseContextFailureReportsUnlocked();
            }
            catch (std::exception const& e)
            {
                TLLM_LOG_WARNING("Failed to record CacheSender response failure events: %s", e.what());
            }
            catch (...)
            {
                TLLM_LOG_WARNING("Failed to record CacheSender response failure events");
            }
            for (auto it = mReadyResponses.begin(); it != mReadyResponses.end();)
            {
                auto const requestId = it->first;
                responses.emplace_back(requestId, std::move(it->second));
                it = mReadyResponses.erase(it);
                mCancelledRequests.erase(requestId);
                mPendingRequestInfoRejectCounts.erase(requestId);
                mRemainSendCount.erase(requestId);
                mRequestToSession.erase(requestId);
                mRequestCancelFlags.erase(requestId);
                mReportableContextKvTransferRequestIds.insert(requestId);
            }
            mCurrentRequest = std::nullopt;
            mAnyReady = hasReceiveWorkUnlocked();
        }

        for (auto& [requestId, response] : responses)
        {
            PerRequestActivityLog::instance().record(requestId, reason, "CacheSender::response");
            setResponsePromiseException(requestId, response, responseException);
        }
    }

    void sendRequestInfoRejectReadySignal(
        Connection const* connection, RequestIdType requestId, char const* reason) noexcept
    {
        if (connection == nullptr)
        {
            return;
        }
        try
        {
            bool const isReady{false};
            connection->send(executor::kv_cache::DataContext{readyTagFromRequestId(requestId), mTerminate}, &isReady,
                sizeof(isReady));
        }
        catch (std::exception const& e)
        {
            TLLM_LOG_REQ_WARNING(
                requestId, "Failed to reject stale request info reason=%s error=%s", reason, e.what());
        }
        catch (...)
        {
            TLLM_LOG_REQ_WARNING(requestId, "Failed to reject stale request info reason=%s", reason);
        }
    }

    void finishDiscardedRequestInfo(RequestIdType requestId, char const* reason, size_t rejectCount) noexcept
    {
        if (reason != kCancelledWithoutSession)
        {
            return;
        }

        std::optional<Response> response;
        {
            std::scoped_lock lk(mSenderMutex, mMtxForMap);
            auto& remainingRejectCount = mPendingRequestInfoRejectCounts[requestId];
            if (remainingRejectCount == 0)
            {
                remainingRejectCount = std::max<size_t>(rejectCount, 1);
            }
            --remainingRejectCount;
            if (auto it = mReadyResponses.find(requestId); it != mReadyResponses.end())
            {
                response.emplace(std::move(it->second));
                mReadyResponses.erase(it);
            }
            if (remainingRejectCount == 0)
            {
                mPendingRequestInfoRejectCounts.erase(requestId);
                mCancelledRequests.erase(requestId);
                mRequestCancelFlags.erase(requestId);
                mRemainSendCount.erase(requestId);
                if (mCurrentRequest.has_value() && getCurrentRequestId() == requestId)
                {
                    mCurrentRequest = std::nullopt;
                }
            }
            mAnyReady = hasReceiveWorkUnlocked();
        }
        if (response.has_value())
        {
            setResponsePromiseValue(requestId, *response);
        }
    }

    void handleAsyncSend(AsyncSendResource& resource)
    {
        tensorrt_llm::common::setThreadName("dataTransAsyncSend");
        while (!resource.mTerminate)
        {
            Response resp;
            {
                std::unique_lock lk(resource.mMtxForQueue);
                resource.mCVforQueue.wait(
                    lk, [&resource] { return !resource.mSendQueue.empty() || resource.mTerminate; });
                if (resource.mTerminate)
                {
                    if (!resource.mSendQueue.empty())
                    {
                        TLLM_LOG_WARNING("There are still %zu requests in the mSendQueue, but encountered terminate.",
                            resource.mSendQueue.size());
                    }
                    break;
                }
                resp = std::move(resource.mSendQueue.front());
                resource.mSendQueue.pop_front();
            }
            // Sequence the read before the move: argument initializations
            // are indeterminately sequenced, so inlining resp.mRequest->...
            // alongside std::move(resp) is UB once mRequest is a shared_ptr.
            TLLM_CHECK(resp.mRequest != nullptr);
            auto const reqId = resp.mRequest->mRequestId;
            sendAndRemoveResponse(reqId, std::move(resp));
        }
    }

    void sendAndRemoveResponse(RequestIdType id, Response resp) noexcept
    {
        auto releaseOnFailure = [this, id]() noexcept
        {
            try
            {
                release(id, "send_failure");
            }
            catch (std::exception const& e)
            {
                TLLM_LOG_REQ_WARNING(id, "Failed to release transfer session after send failure: %s", e.what());
            }
            catch (...)
            {
                TLLM_LOG_REQ_WARNING(id, "Failed to release transfer session after send failure");
            }
        };

        try
        {
            TLLM_CUDA_CHECK(cudaSetDevice(mDeviceId));
            sendSync(*resp.mRequest);
            release(id, "send_complete");
            resp.mPromise.set_value();
        }
        catch (tensorrt_llm::common::RequestSpecificException const& e)
        {
            PerRequestActivityLog::instance().record(id, "send_failed", "CacheSender::sendAndRemoveResponse");
            TLLM_LOG_REQ_ERROR(id, "Exception in sendAndRemoveResponse: %s", e.what());
            releaseOnFailure();
            auto new_exception = TLLM_REQUEST_EXCEPTION(id, e.getErrorCode(), "%s", e.what());
            resp.mPromise.set_exception(std::make_exception_ptr(new_exception));
        }
        catch (std::exception const& e)
        {
            PerRequestActivityLog::instance().record(id, "send_failed", "CacheSender::sendAndRemoveResponse");
            TLLM_LOG_REQ_ERROR(id, "Exception in sendAndRemoveResponse: %s", e.what());
            releaseOnFailure();
            resp.mPromise.set_exception(std::current_exception());
        }
    }

    void asyncSendAndRemoveResponse(RequestIdType id, Response resp) noexcept
    {
        std::unique_lock lk(mAsyncSendResource.mMtxForQueue);
        mAsyncSendResource.mSendQueue.emplace_back(std::move(resp));
        mAsyncSendResource.mCVforQueue.notify_one();
    }

    void sendResponse(std::map<RequestIdType, CacheSender::Impl::Response>::iterator it)
    {
        auto reqId = mCurrentRequest.value();
        int count{0};
        {
            std::scoped_lock lk(mSenderMutex);
            count = --mRemainSendCount[reqId];
        }
        TLLM_CHECK(count >= 0);
        if (count > 0 && hasCancelledRequest(reqId))
        {
            auto cancelledResponse = std::move(it->second);
            removeResponse(it);
            rejectCancelledCurrentRequest(reqId);
            setResponsePromiseValue(reqId, cancelledResponse);
            mCurrentRequest = std::nullopt;
            return;
        }
        if (count == 0)
        {
            {
                std::scoped_lock lk(mSenderMutex);
                mRemainSendCount.erase(reqId);
            }

            // Check if the request is cancelled
            bool isReady = true;
            {
                std::scoped_lock lk(mSenderMutex);
                if (mCancelledRequests.find(reqId) != mCancelledRequests.end())
                {
                    isReady = false;
                }
            }
            sendReadySignal(reqId, isReady);

            if (isReady)
            {
                if (dynamic_cast<executor::kv_cache::AgentConnectionManager*>(mManager) != nullptr)
                {
                    // our nixl impl seems only support recv and send in the same thread
                    //  if we use zmq as control path, we may avoid this issue
                    sendAndRemoveResponse(it->first, std::move(it->second));
                }
                else
                {
                    // if we send data in another thread, multiple rank may send data for different requests at the same
                    // time with gen DP case.
                    asyncSendAndRemoveResponse(it->first, std::move(it->second));
                }
                removeResponse(it);
            }
            else
            {
                auto const cancelledReqId = reqId;
                auto cancelledResponse = std::move(it->second);
                removeResponse(it);
                {
                    std::scoped_lock lkResp(mSenderMutex);
                    mCancelledRequests.erase(cancelledReqId);
                    mPendingRequestInfoRejectCounts.erase(cancelledReqId);
                    mAnyReady = hasReceiveWorkUnlocked();
                }
                try
                {
                    release(cancelledReqId, "cancel");
                }
                catch (std::exception const& e)
                {
                    TLLM_LOG_WARNING(
                        "Failed to release cancelled transfer session for request %zu: %s", cancelledReqId, e.what());
                }
                catch (...)
                {
                    TLLM_LOG_WARNING("Failed to release cancelled transfer session for request %zu", cancelledReqId);
                }
                try
                {
                    cancelledResponse.mPromise.set_exception(std::make_exception_ptr(
                        TLLM_REQUEST_EXCEPTION(cancelledReqId, common::RequestErrorCode::kNETWORK_ERROR,
                            "KV cache transfer for request %zu was cancelled", cancelledReqId)));
                }
                catch (std::future_error const&)
                {
                    // Promise already satisfied; nothing to do.
                }
                mCurrentRequest = std::nullopt;
            }
        }
        mCurrentRequest = std::nullopt;
    }

    void response() noexcept
    {
        try
        {
            tensorrt_llm::common::setThreadName("dataTransResp");
            TLLM_CUDA_CHECK(cudaSetDevice(mDeviceId));
            while (!mTerminate || !mAnyReady)
            {
                if (!mAnyReady)
                {
                    std::unique_lock lk(mCondMutex);
                    mSenderCv.wait(lk, [this]() { return (mAnyReady || mTerminate); });
                }
                if (mTerminate)
                {
                    break;
                }
                bool const isAgent = isAgentConnectionManager();
                if (isAgent)
                {
                    completeCancelledReadyResponses();
                }
                if (!hasReceiveWork())
                {
                    continue;
                }

                if (isAgent)
                {
                    completeCancelledReadyResponses();
                    if (!hasReceiveWork())
                    {
                        continue;
                    }
                }

                std::optional<RequestInfo> requestInfo;
                try
                {
                    requestInfo = recvRequestInfo();
                }
                catch (std::exception const& e)
                {
                    auto const responseException = std::current_exception();
                    TLLM_LOG_ERROR("Exception while receiving CacheSender request info: %s", e.what());
                    if (mTerminate || !mManager->isRunning())
                    {
                        return;
                    }
                    if (!isAgent)
                    {
                        failAndRemoveAllReadyResponses("request_info_receive_failed", responseException);
                    }
                    if (isAgent)
                    {
                        completeCancelledReadyResponses();
                    }
                    continue;
                }
                catch (...)
                {
                    auto const responseException = std::current_exception();
                    TLLM_LOG_ERROR("Exception while receiving CacheSender request info");
                    if (mTerminate || !mManager->isRunning())
                    {
                        return;
                    }
                    if (!isAgent)
                    {
                        failAndRemoveAllReadyResponses("request_info_receive_failed", responseException);
                    }
                    if (isAgent)
                    {
                        completeCancelledReadyResponses();
                    }
                    continue;
                }
                if (!requestInfo.has_value())
                {
                    if (mTerminate || !mManager->isRunning())
                    {
                        return;
                    }
                    if (isAgent)
                    {
                        completeCancelledReadyResponses();
                    }
                    continue;
                }
                if (mTerminate || !mManager->isRunning())
                {
                    return;
                }

                auto reqId = requestInfo->getRequestId();
                {
                    std::scoped_lock lk(mSenderMutex);
                    mCurrentRequest = reqId;
                }

                bool needsRemainSendCount = false;
                {
                    std::scoped_lock lk(mSenderMutex);
                    needsRemainSendCount = mRemainSendCount.find(reqId) == mRemainSendCount.end();
                }
                if (needsRemainSendCount)
                {
                    auto const counterpartsCount = getCounterpartsCount(reqId);
                    std::scoped_lock lk(mSenderMutex);
                    mRemainSendCount.emplace(reqId, counterpartsCount);
                }

                while (!hasReadyResponse(reqId) && !hasCancelledRequest(reqId))
                {
                    std::unique_lock lk(mCondMutex);
                    mSenderCv.wait(lk,
                        [this, reqId]() { return mTerminate || hasReadyResponse(reqId) || hasCancelledRequest(reqId); });
                    if (mTerminate)
                    {
                        break;
                    }
                }
                if (mTerminate)
                {
                    break;
                }
                if (!hasReadyResponse(reqId) && hasCancelledRequest(reqId))
                {
                    rejectCancelledCurrentRequest(reqId);
                    continue;
                }
                auto it = getCurrentResponse();
                if (it != mReadyResponses.end())
                {
                    sendResponse(it);
                }
            }
        }
        catch (std::exception const& err)
        {
            auto const responseException = std::current_exception();
            TLLM_LOG_ERROR("Exception in CacheSender response: %s", err.what());
            std::scoped_lock lk(mSenderMutex, mMtxForMap);
            try
            {
                recordReadyResponseContextFailureReportsUnlocked();
            }
            catch (std::exception const& reportErr)
            {
                TLLM_LOG_WARNING("Failed to record CacheSender response failure events: %s", reportErr.what());
            }
            catch (...)
            {
                TLLM_LOG_WARNING("Failed to record CacheSender response failure events");
            }
            for (auto& it : mReadyResponses)
            {
                try
                {
                    it.second.mPromise.set_exception(responseException);
                }
                catch (std::exception const& promiseErr)
                {
                    TLLM_LOG_WARNING("Failed to set CacheSender response exception for request %zu: %s", it.first,
                        promiseErr.what());
                }
                catch (...)
                {
                    TLLM_LOG_WARNING("Failed to set CacheSender response exception for request %zu", it.first);
                }
            }
        }
    }

    void terminate()
    {
        {
            std::unique_lock lk(mCondMutex);
            mTerminate = true;
        }
        cancelAllInFlightRequests();
        // We don't have to wait for the future. If another thread is sending data, it won't pay attention
        // to the terminate flag.
        mSenderCv.notify_all();
        mAsyncSendResource.mTerminate = true;
        mAsyncSendResource.mCVforQueue.notify_all();
        for (auto& future : mAsyncSendFutures)
        {
            future.get();
        }
        if (mResponseFuture.valid())
        {
            mResponseFuture.get();
        }
    }

    void removeResponse(std::map<RequestIdType, Response>::iterator it)
    {
        {
            std::scoped_lock lkResp(mSenderMutex);
            mReadyResponses.erase(it);
            mAnyReady = hasReceiveWorkUnlocked();
        }
    }

    [[nodiscard]] RequestIdType getCurrentRequestId() const
    {
        return mCurrentRequest.value();
    }

    [[nodiscard]] std::map<RequestIdType, Response>::iterator getCurrentResponse()
    {
        std::scoped_lock lk(mSenderMutex);
        return mReadyResponses.find(getCurrentRequestId());
    }

    [[nodiscard]] bool hasReadyResponse(RequestIdType requestId)
    {
        std::scoped_lock lk(mSenderMutex);
        return mReadyResponses.find(requestId) != mReadyResponses.end();
    }

    [[nodiscard]] bool hasCancelledRequest(RequestIdType requestId)
    {
        std::scoped_lock lk(mSenderMutex);
        return mCancelledRequests.find(requestId) != mCancelledRequests.end();
    }

    [[nodiscard]] static std::vector<Connection const*> getConnectedConnections(TransferSession const& session)
    {
        std::vector<Connection const*> connectedConnections;
        for (auto const* connection : session.getConnections())
        {
            if (connection != nullptr)
            {
                connectedConnections.push_back(connection);
            }
        }
        return connectedConnections;
    }

    static void sendReadySignalToConnectedNonAgentPeers(
        RequestIdType requestId, std::vector<Connection const*> const& connectedConnections, bool isReady) noexcept
    {
        auto readySignalContext = executor::kv_cache::DataContext{readyTagFromRequestId(requestId)};
        for (auto const* connection : connectedConnections)
        {
            try
            {
                connection->send(readySignalContext, &isReady, sizeof(isReady));
            }
            catch (std::exception const& e)
            {
                TLLM_LOG_WARNING(
                    "Failed to send cancellation ready signal request_id=%zu error=%s", requestId, e.what());
            }
        }
    }

    void completeCancelledResponse(CancelledResponse& cancelledResponse) noexcept
    {
        bool const isReady = false;
        sendReadySignalToConnectedNonAgentPeers(
            cancelledResponse.mRequestId, cancelledResponse.mConnectedConnections, isReady);
        setResponsePromiseValue(cancelledResponse.mRequestId, cancelledResponse.mResponse);
    }

    void rejectCancelledCurrentRequest(RequestIdType requestId) noexcept
    {
        std::vector<Connection const*> connectedConnections;
        {
            std::scoped_lock lk(mSenderMutex, mMtxForMap);
            if (auto sessionIt = mRequestToSession.find(requestId); sessionIt != mRequestToSession.end())
            {
                recordContextKvTransferEventReportUnlocked(requestId, sessionIt->second);
                connectedConnections = getConnectedConnections(sessionIt->second);
                mRequestToSession.erase(sessionIt);
            }
            mCancelledRequests.erase(requestId);
            mPendingRequestInfoRejectCounts.erase(requestId);
            mRemainSendCount.erase(requestId);
            mRequestCancelFlags.erase(requestId);
            if (mCurrentRequest.has_value() && getCurrentRequestId() == requestId)
            {
                mCurrentRequest = std::nullopt;
            }
            mAnyReady = hasReceiveWorkUnlocked();
        }
        bool const isReady = false;
        sendReadySignalToConnectedNonAgentPeers(requestId, connectedConnections, isReady);
    }

    void completeCancelledReadyResponses()
    {
        std::vector<CancelledResponse> cancelledResponses;
        {
            std::scoped_lock lk(mSenderMutex, mMtxForMap);
            for (auto it = mReadyResponses.begin(); it != mReadyResponses.end();)
            {
                auto const requestId = it->first;
                if (mCancelledRequests.find(requestId) == mCancelledRequests.end())
                {
                    ++it;
                    continue;
                }
                std::vector<Connection const*> connectedConnections;
                if (auto sessionIt = mRequestToSession.find(requestId); sessionIt != mRequestToSession.end())
                {
                    connectedConnections = getConnectedConnections(sessionIt->second);
                    mRequestToSession.erase(sessionIt);
                    mRemainSendCount.erase(requestId);
                }
                if (mCurrentRequest.has_value() && getCurrentRequestId() == requestId)
                {
                    mCurrentRequest = std::nullopt;
                }
                cancelledResponses.emplace_back(
                    CancelledResponse{requestId, std::move(it->second), std::move(connectedConnections)});
                it = mReadyResponses.erase(it);
                mCancelledRequests.erase(requestId);
                mPendingRequestInfoRejectCounts.erase(requestId);
                mRequestCancelFlags.erase(requestId);
            }
            mAnyReady = hasReceiveWorkUnlocked();
        }

        for (auto& cancelledResponse : cancelledResponses)
        {
            completeCancelledResponse(cancelledResponse);
        }
    }

public:
    void setRnnConfig(executor::kv_cache::CacheState::RnnModelConfig rnnModelConfig,
        std::vector<SizeType32> rnnLayerNumPerPP, nvinfer1::DataType convStateDataType,
        nvinfer1::DataType ssmStateDataType)
    {
        mCacheTransferLayer.setRnnConfig(rnnModelConfig, rnnLayerNumPerPP, convStateDataType, ssmStateDataType);
        mSelfState.setCacheState(mCacheTransferLayer.getCacheState());
    }

private:
    std::optional<RequestIdType> mCurrentRequest;
    std::set<LlmRequest::RequestIdType> mCancelledRequests;
    std::unordered_map<RequestIdType, size_t> mPendingRequestInfoRejectCounts;
    std::map<RequestIdType, Response> mReadyResponses;
    std::mutex mSenderMutex, mCondMutex;
    std::atomic<bool> mAnyReady{false}, mTerminate{false};
    std::condition_variable mSenderCv, mResponderCv;
    std::future<void> mResponseFuture;
    std::unordered_map<LlmRequest::RequestIdType, int> mRemainSendCount;
    // Sessions poll these flags without holding mMtxForMap, so the flag object must outlive map updates.
    std::unordered_map<LlmRequest::RequestIdType, std::shared_ptr<std::atomic<bool>>> mRequestCancelFlags;
    AsyncSendResource mAsyncSendResource;
    // Inserted when this rank owns a reportable context KV transfer event. The bit is consumed when
    // checkContextTransferStatus observes completion/error, or by CacheTransceiver after cancellation succeeds.
    std::unordered_set<RequestIdType> mReportableContextKvTransferRequestIds;
    std::vector<std::future<void>> mAsyncSendFutures;
    int mDeviceId{-1};

    executor::kv_cache::ConnectionManager* mManager;
    std::map<LlmRequest::RequestIdType, TransferSession> mRequestToSession;
    executor::DataTransceiverState mSelfState;
    CacheTransferLayer mCacheTransferLayer;
    std::mutex mMtxForMap;
    runtime::BufferManager mBufferManager;
    std::ofstream mMeasuresFile;
};

class CacheReceiver::Impl
{
public:
    Impl(executor::kv_cache::ConnectionManager* manager, SizeType32 selfIndex, CacheTransferLayer cacheLayer)
        : mManager{manager}
        , mSelfState{cacheLayer.getCacheState(), executor::kv_cache::CommState{manager->getCommState()}}
        , mCacheTransferLayer{std::move(cacheLayer)}
        , mBufferManager{std::make_shared<runtime::CudaStream>()}
    {
        TLLM_CHECK(mManager);
        TLLM_CHECK(mManager->getCommState().getSelfIdx() == selfIndex);
        PerRequestActivityLog::instance().setRank(static_cast<int>(selfIndex));
        TLLM_CUDA_CHECK(cudaGetDevice(&mDeviceId));
    }

    [[nodiscard]] std::shared_ptr<std::atomic<bool>> getOrCreateRequestCancelFlag(LlmRequest::RequestIdType requestId)
    {
        std::scoped_lock<std::mutex> lock(mRequestCancelFlagsMutex);
        auto& cancelFlag = mRequestCancelFlags[requestId];
        if (cancelFlag == nullptr)
        {
            cancelFlag = std::make_shared<std::atomic<bool>>(false);
        }
        return cancelFlag;
    }

    void clearRequestCancelFlag(LlmRequest::RequestIdType requestId)
    {
        std::scoped_lock<std::mutex> lock(mRequestCancelFlagsMutex);
        mRequestCancelFlags.erase(requestId);
    }

    [[nodiscard]] bool cancelInFlightRequest(LlmRequest::RequestIdType requestId)
    {
        std::scoped_lock<std::mutex> lock(mRequestCancelFlagsMutex);
        auto it = mRequestCancelFlags.find(requestId);
        if (it == mRequestCancelFlags.end())
        {
            return false;
        }
        it->second->store(true);
        return true;
    }

    void cancelAllInFlightRequests()
    {
        std::scoped_lock<std::mutex> lock(mRequestCancelFlagsMutex);
        for (auto const& [requestId, cancelFlag] : mRequestCancelFlags)
        {
            static_cast<void>(requestId);
            if (cancelFlag != nullptr)
            {
                cancelFlag->store(true);
            }
        }
    }

    [[nodiscard]] std::future<void> receiveAsync(std::shared_ptr<LlmRequest> const& llmRequest)
    {
        TLLM_CHECK(llmRequest != nullptr);
        auto statusFuture = requestAndReceiveAsyncMultiThreads(llmRequest);
        return std::move(statusFuture.future);
    }

    [[nodiscard]] TransferStatusFuture requestAndReceiveAsyncMultiThreads(std::shared_ptr<LlmRequest> const& llmRequest)
    {
        TLLM_CHECK(llmRequest != nullptr);
        try
        {
            getOrCreateRequestCancelFlag(llmRequest->mRequestId);
            auto hasError = std::make_shared<std::atomic<bool>>(false);
            auto promise = std::make_unique<std::promise<void>>();
            auto future = promise->get_future();
            TLLM_CHECK(llmRequest->getDataTransceiverState().getCommState().has_value());
            std::string processInfo = kDefaultProcessInfo;
            if (common::getEnvRequestKVCacheConcurrent())
            {
                processInfo = llmRequest->getDataTransceiverState().getCommState()->toString();
            }
            if (mInstanceToAsyncResource.find(processInfo) == mInstanceToAsyncResource.end())
            {

                mInstanceToAsyncResource.emplace(processInfo, std::make_unique<AsyncResource>());
                auto requestFuture = std::async(std::launch::async, &CacheReceiver::Impl::request, this,
                    std::ref(*mInstanceToAsyncResource.at(processInfo)));
                mRequestFutures.emplace_back(std::move(requestFuture));
            }
            auto& asyncResource = mInstanceToAsyncResource.at(processInfo);
            {
                std::unique_lock<std::mutex> lck(asyncResource->mMtxForQueue);
                asyncResource->mRequestsQueue.emplace_back(llmRequest, std::move(promise), hasError);
            }
            asyncResource->mCVforQueue.notify_all();
            return TransferStatusFuture{std::move(future), std::move(hasError)};
        }
        catch (std::exception const& e)
        {
            TLLM_THROW("%s", e.what());
        }
    }

    void receiveSync(TransferSession& session)
    {
        mCacheTransferLayer.unformat(session);
        if (!common::getEnvKVCacheTimeOutputPath().empty())
        {
            std::unique_lock<std::mutex> lock(mMeasuresFileMutex);
            if (!mMeasuresFile.is_open())
            {
                auto outputPath = getTransferOutputPath("recv");
                mMeasuresFile.open(outputPath);
                TLLM_CHECK_WITH_INFO(
                    mMeasuresFile.is_open(), "Failed to open transfer output file: %s", outputPath.string().c_str());
            }
            session.exportMeasure(mMeasuresFile, false);
        }
    }

    TransferSession sendRequestInfo(LlmRequest const& llmRequest)
    {
        uint64_t requestId = llmRequest.getContextPhaseParams().value().getReqId();
        auto const& contextState = llmRequest.getDataTransceiverState();
        auto const& commState = contextState.getCommState().value();
        auto const& destCacheState = contextState.getCacheState().value();
        mCacheTransferLayer.validateSupport(contextState);
        auto requestCancelFlag = getOrCreateRequestCancelFlag(llmRequest.mRequestId);

        RequestInfo requestInfo(requestId, mSelfState);

        if (!mCacheTransferLayer.getCacheManager()->getBlockManager().isVariableWindow())
        {
            auto* cacheManager = mCacheTransferLayer.getCacheManager();
            auto beam = 0;
            auto const srcPpSize = destCacheState.getParallelConfig().mPipelineParallelism;
            auto requestedBlockRange = getBlockRangeForReceiving(cacheManager, llmRequest,
                destCacheState.getEnableBlockReuse(), destCacheState.getEnablePartialReuse(),
                /*recvSideHasCP=*/false, srcPpSize);

            auto const& uniqueTokens = llmRequest.getUniqueTokens(beam);
            auto lastBlockKey
                = BlockKey(llmRequest.getInputTokensExtraIds().has_value(), llmRequest.getLoraTaskId(), uniqueTokens);
            auto tokensPerBlock = cacheManager->getBlockManager().getTokensPerBlock();
            SizeType32 startTokenIdx = static_cast<SizeType32>(uniqueTokens.size() / tokensPerBlock) * tokensPerBlock;
            SizeType32 endTokenIdx = static_cast<SizeType32>(uniqueTokens.size());
            auto extraKeys = kv_cache_manager::generateBlockHashExtraKeys(llmRequest, startTokenIdx, endTokenIdx);
            lastBlockKey.extraKeys = std::move(extraKeys);
            // Compute indexFromEnd from the number of requested blocks
            int32_t requestedBlockSize = requestedBlockRange.getBlockIdsPerWindow().begin()->second.size();
            TLLM_CHECK_WITH_INFO(requestedBlockSize > 0, "requestedBlockSize must be > 0");
            int32_t indexFromEnd = requestedBlockSize - 1;

            requestInfo = RequestInfo(requestId, mSelfState, indexFromEnd, lastBlockKey);
        }

        auto* agentConnectionManager = dynamic_cast<executor::kv_cache::AgentConnectionManager*>(mManager);
        std::vector<std::optional<size_t>> cacheBufferIds;
        if (agentConnectionManager)
        {
            for (auto& cacheTransBufferManager : agentConnectionManager->getCacheTransBufferManagers())
            {
                cacheBufferIds.push_back(cacheTransBufferManager->assignBufferIndexForRecv());
            }
            TLLM_CHECK(!cacheBufferIds.empty());
        }

        auto allCounterparts
            = mCacheTransferLayer.computeCounterparts(mSelfState.getCommState().value().getSelfIdx(), contextState);

        auto kvCounterParts = mCacheTransferLayer.getKvFormatter()->getCounterparts(
            mCacheTransferLayer.getCacheState(), mSelfState.getCommState().value().getSelfIdx(), destCacheState);

        bool hasRnn = mCacheTransferLayer.getCacheState().hasRnnConfig() && destCacheState.hasRnnConfig();

        std::vector<SizeType32> rnnCounterParts;
        if (hasRnn)
        {
            rnnCounterParts = executor::kv_cache::targetIRanksForRnn(
                destCacheState, mCacheTransferLayer.getCacheState(), mSelfState.getCommState().value().getSelfIdx())
                                  .mIRanks;
        }

        auto connections = mManager->getConnections(commState);
        std::vector<executor::kv_cache::Connection const*> allConnections;
        for (auto index : allCounterparts)
        {
            auto const* connection = connections.at(index);
            allConnections.emplace_back(connection);
        }

        for (size_t ci = 0; ci < allCounterparts.size(); ci++)
        {
            auto rank = allCounterparts[ci];
            auto const* connection = connections.at(rank);

            bool isKvCounterpart
                = std::find(kvCounterParts.begin(), kvCounterParts.end(), rank) != kvCounterParts.end();
            bool isRnnCounterpart
                = hasRnn && std::find(rnnCounterParts.begin(), rnnCounterParts.end(), rank) != rnnCounterParts.end();

            if (agentConnectionManager)
            {
                auto idsForRank = cacheBufferIds;
                auto const& managers = agentConnectionManager->getCacheTransBufferManagers();
                for (size_t i = 0; i < idsForRank.size(); i++)
                {
                    auto kind = managers[i]->getBufferKind();
                    bool include = (kind != BufferKind::kRNN) ? isKvCounterpart : isRnnCounterpart;
                    if (!include)
                    {
                        idsForRank[i] = std::nullopt;
                    }
                }

                int validConnectionIdx = 0;
                if (isKvCounterpart)
                {
                    auto kvCpIdx
                        = std::find(kvCounterParts.begin(), kvCounterParts.end(), rank) - kvCounterParts.begin();
                    auto [pickUpIdx, localRankIdx] = mCacheTransferLayer.getKvFormatter()->pickRecvConnections(
                        allCounterparts.size(), mSelfState.getCacheState().value(),
                        mSelfState.getCommState().value().getSelfIdx(), destCacheState, allCounterparts);
                    validConnectionIdx
                        = std::find(localRankIdx.begin(), localRankIdx.end(), kvCpIdx) - localRankIdx.begin();
                }
                else if (isRnnCounterpart)
                {
                    auto rnnTargetInfo = executor::kv_cache::targetIRanksForRnn(destCacheState,
                        mCacheTransferLayer.getCacheState(), mSelfState.getCommState().value().getSelfIdx());
                    auto rnnCpIdx
                        = std::find(rnnCounterParts.begin(), rnnCounterParts.end(), rank) - rnnCounterParts.begin();
                    auto [pickUpIdx, localRankIdx] = cache_formatter_utils::pickRecvConnections(rnnCounterParts.size(),
                        mCacheTransferLayer.getCacheState(), mSelfState.getCommState().value().getSelfIdx(),
                        destCacheState, rnnCounterParts, rnnTargetInfo);
                    validConnectionIdx
                        = std::find(localRankIdx.begin(), localRankIdx.end(), rnnCpIdx) - localRankIdx.begin();
                }

                auto* agentConnection = dynamic_cast<executor::kv_cache::AgentConnection const*>(connection);
                TLLM_CHECK(agentConnection != nullptr);

                const_cast<executor::kv_cache::AgentConnection*>(agentConnection)
                    ->sendRequestAndBufferInfo(requestInfo, idsForRank, validConnectionIdx);
            }
            else
            {
                sendRequestInfo(connection, requestInfo);
            }
        }
        auto const& resource = getReceiveCacheResource(llmRequest);
        return TransferSession(std::move(allConnections), DataContext{tagFromRequestId(requestId), *requestCancelFlag},
            std::move(allCounterparts), mSelfState, contextState, resource->mBufferManager,
            requestInfo.getIndexFromEnd(), requestInfo.getLastBlockKey(), &llmRequest,
            !common::getEnvKVCacheTimeOutputPath().empty(), requestCancelFlag);
    }

    std::unique_ptr<ReceiveCacheResource> const& getReceiveCacheResource(LlmRequest const& llmRequest)
    {
        std::scoped_lock<std::mutex> lock(mProcessIoResouceMutex);
        TLLM_CHECK(llmRequest.getDataTransceiverState().getCommState().has_value());
        std::string processString = kDefaultProcessInfo;
        if (common::getEnvRequestKVCacheConcurrent())
        {
            processString = llmRequest.getDataTransceiverState().getCommState()->toString();
        }
        if (mProcessToResources.find(processString) == mProcessToResources.end())
        {
            mProcessToResources.emplace(processString,
                std::make_unique<ReceiveCacheResource>(
                    runtime::BufferManager{std::make_shared<runtime::CudaStream>()}, runtime::CudaEvent{}));
        }
        return mProcessToResources.at(processString);
    }

    void sendRequestInfo(executor::kv_cache::Connection const* connection, RequestInfo const& info)
    {
        std::ostringstream oss;
        RequestInfo::serialize(info, oss);
        auto const& serializedInfo = oss.str();
        std::size_t const infoSize = serializedInfo.size();
        TransceiverTag::Id id{TransceiverTag::Id::REQUEST_SEND};
        auto& connectionMutex = getConnectionSendMutex(connection);
        // Keep the request-info envelope contiguous on this connection so concurrent requests cannot interleave.
        std::scoped_lock<std::mutex> lock(connectionMutex);
        connection->send(DataContext{TransceiverTag::kID_TAG}, &id, sizeof(id));
        connection->send(DataContext{TransceiverTag::kINFO_SIZE_TAG}, &infoSize, sizeof(infoSize));
        connection->send(DataContext{TransceiverTag::kINFO_TAG}, serializedInfo.data(), infoSize);
    }

    bool cancelRequest(LlmRequest const& llmRequest)
    {
        // Key by the context (transfer) request id to match requestSync and the
        // sender side; fall back to the local id if context params are absent.
        auto const transferReqId = llmRequest.getContextPhaseParams().has_value()
            ? llmRequest.getContextPhaseParams().value().getReqId()
            : llmRequest.mRequestId;
        PerRequestActivityLog::instance().record(transferReqId, "cancel_requested", "CacheReceiver::cancelRequest",
            "gen_request_id", static_cast<std::int64_t>(llmRequest.mRequestId));

        std::string processInfo = kDefaultProcessInfo;
        if (common::getEnvRequestKVCacheConcurrent())
        {
            processInfo = llmRequest.getDataTransceiverState().getCommState()->toString();
        }

        bool isCancelled = false;
        auto& asyncResource = mInstanceToAsyncResource.at(processInfo);
        {
            std::unique_lock<std::mutex> lck(asyncResource->mMtxForQueue);
            auto it = std::find_if(asyncResource->mRequestsQueue.begin(), asyncResource->mRequestsQueue.end(),
                [&llmRequest](RequestAndPromise const& requestAndPromise)
                { return requestAndPromise.mRequest->mRequestId == llmRequest.mRequestId; });
            if (it != asyncResource->mRequestsQueue.end())
            {
                it->mRequest->setState(LlmRequestState::kDISAGG_TRANS_ERROR);
                if (it->mHasError != nullptr)
                {
                    it->mHasError->store(true);
                }
                // Resolve the promise before erasing so the future returned by
                // receiveAsync surfaces a structured cancellation error rather
                // than std::future_error: Broken promise from the destroyed promise.
                if (it->mPromise)
                {
                    try
                    {
                        it->mPromise->set_exception(std::make_exception_ptr(
                            TLLM_REQUEST_EXCEPTION(llmRequest.mRequestId, common::RequestErrorCode::kNETWORK_ERROR,
                                "Generation KV cache request cancelled before send for request %zu",
                                llmRequest.mRequestId)));
                    }
                    catch (std::future_error const&)
                    {
                        // Promise already satisfied; nothing to do.
                    }
                }
                asyncResource->mRequestsQueue.erase(it);
                clearRequestCancelFlag(llmRequest.mRequestId);
                isCancelled = true;
            }
        }

        if (!isCancelled)
        {
            isCancelled = cancelInFlightRequest(llmRequest.mRequestId);
        }

        if (!isCancelled)
        {
            TLLM_LOG_REQ_WARNING(llmRequest.mRequestId, "Cannot cancel request");
        }
        return isCancelled;
    }

    bool receiveReadySignal(TransferSession& session)
    {
        bool isReadyFinal = true;
        bool isReady = false;
        auto const& connections = session.getConnections();
        auto* agentConnectionManager = dynamic_cast<executor::kv_cache::AgentConnectionManager*>(mManager);
        auto const readySignalTag = agentConnectionManager != nullptr
            ? TransceiverTag::kREADY_SIGNAL_TAG
            : readyTagFromDataTag(session.getDataContext().getTag());
        auto readySignalContext = executor::kv_cache::DataContext{readySignalTag, session.getTransferTerminate()};

        for (size_t i = 0; i < connections.size(); i++)
        {
            if (agentConnectionManager)
            {
                auto* agentConnection = dynamic_cast<executor::kv_cache::AgentConnection const*>(connections.at(i));
                TLLM_CHECK(agentConnection);
                isReady = agentConnection->recvReadySignal(readySignalContext);
            }
            else
            {
                connections.at(i)->recv(readySignalContext, &isReady, sizeof(isReady));
            }
            isReadyFinal &= isReady;
        }

        return isReadyFinal;
    }

    ~Impl()
    {
        cancelAllInFlightRequests();
        mTerminate.store(true);
        for (auto&& [processInfo, asyncResource] : mInstanceToAsyncResource)
        {
            asyncResource->mTerminate = true;
            asyncResource->mCVforQueue.notify_all();
        }
        for (auto&& future : mRequestFutures)
        {
            future.get();
        }
    }

private:
    void requestSync(LlmRequest& llmRequest)
    {
        // Key activity-log events by the context (transfer) request id so they
        // correlate with the sender side, which keys its sessions by
        // RequestInfo::getRequestId() == this same context request id. The
        // local generation request id is recorded as the event payload.
        auto const transferReqId = llmRequest.getContextPhaseParams().value().getReqId();
        auto const genReqId = static_cast<std::int64_t>(llmRequest.mRequestId);
        TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
            "Start calling requestSync for request ID: %zu, context request ID: %zu.", llmRequest.mRequestId,
            transferReqId);
        PerRequestActivityLog::instance().record(
            transferReqId, "recv_request_started", "CacheReceiver::requestSync", "gen_request_id", genReqId);
        llmRequest.setKvCacheTransferStart(std::chrono::steady_clock::now());
        TLLM_CUDA_CHECK(cudaSetDevice(mDeviceId));
        auto requestCancelFlag = getOrCreateRequestCancelFlag(llmRequest.mRequestId);
        if (requestCancelFlag->load())
        {
            PerRequestActivityLog::instance().record(transferReqId, "recv_request_cancelled_early",
                "CacheReceiver::requestSync", "gen_request_id", genReqId);
            llmRequest.setState(LlmRequestState::kDISAGG_TRANS_ERROR);
            llmRequest.setKvCacheTransferEnd(std::chrono::steady_clock::now());
            return;
        }
        auto session = sendRequestInfo(llmRequest);
        session.setTime(TransferSession::kTimeRequestInfo);
        bool isReady = receiveReadySignal(session);
        if (!isReady)
        {
            PerRequestActivityLog::instance().record(
                transferReqId, "recv_ready_signal_failed", "CacheReceiver::requestSync", "gen_request_id", genReqId);
            // Reuse the error state for the cancelled request.
            llmRequest.setState(LlmRequestState::kDISAGG_TRANS_ERROR);
            llmRequest.setKvCacheTransferEnd(std::chrono::steady_clock::now());
            return;
        }
        receiveSync(session);
        llmRequest.setKvCacheTransferEnd(std::chrono::steady_clock::now());
        {
            char detail[PerRequestActivityLog::kDetailLen];
            std::snprintf(detail, sizeof(detail), "gen_request_id=%llu bytes=%zu connections=%zu",
                static_cast<unsigned long long>(genReqId), session.getTotalBytesReceived(),
                session.getConnections().size());
            PerRequestActivityLog::instance().recordDetail(
                transferReqId, "recv_request_completed", "CacheReceiver::requestSync", detail);
        }

        TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
            "End calling requestSync for request ID: %zu, context request ID: %zu.", llmRequest.mRequestId,
            llmRequest.getContextPhaseParams().value().getReqId());
    }

    struct RequestAndPromise
    {
        // shared_ptr so this struct co-owns the request until the promise resolves;
        // protects worker-side dereferences and the promise itself from premature destruction.
        std::shared_ptr<LlmRequest> mRequest;
        std::unique_ptr<std::promise<void>> mPromise;
        std::shared_ptr<std::atomic<bool>> mHasError;

        RequestAndPromise()
            : mRequest(nullptr)
            , mPromise(nullptr)
            , mHasError(nullptr)
        {
        }

        RequestAndPromise(std::shared_ptr<LlmRequest> request, std::unique_ptr<std::promise<void>>&& promise,
            std::shared_ptr<std::atomic<bool>> hasError)
            : mRequest(std::move(request))
            , mPromise(std::move(promise))
            , mHasError(std::move(hasError))
        {
        }

        RequestAndPromise(RequestAndPromise const&) = delete;

        RequestAndPromise(RequestAndPromise&& other) noexcept
            : mRequest(std::move(other.mRequest))
            , mPromise(std::move(other.mPromise))
            , mHasError(std::move(other.mHasError))
        {
        }

        RequestAndPromise& operator=(RequestAndPromise&& other) noexcept
        {
            if (this != &other)
            {
                mRequest.reset();
                if (mPromise)
                {
                    mPromise.reset();
                }

                mRequest = std::move(other.mRequest);
                mPromise = std::move(other.mPromise);
                mHasError = std::move(other.mHasError);
            }
            return *this;
        }
    };

    struct AsyncResource
    {
        std::deque<RequestAndPromise> mRequestsQueue;
        std::mutex mMtxForQueue;
        std::condition_variable mCVforQueue;
        std::atomic<bool> mTerminate{false};
    };

    void request(AsyncResource& resource)
    {
        tensorrt_llm::common::setThreadName("dataTransRequest");
        TLLM_CUDA_CHECK(cudaSetDevice(mDeviceId));

        while (!resource.mTerminate)
        {
            RequestAndPromise requestAndPromise;
            {
                std::unique_lock lck(resource.mMtxForQueue);

                resource.mCVforQueue.wait(
                    lck, [&resource] { return !resource.mRequestsQueue.empty() || resource.mTerminate; });
                if (resource.mTerminate)
                {
                    if (!resource.mRequestsQueue.empty())
                    {
                        TLLM_LOG_WARNING(
                            "There are still %zu requests in the mRequestsQueue, but encountered terminate.",
                            resource.mRequestsQueue.size());
                    }
                    break;
                }
                requestAndPromise = std::move(resource.mRequestsQueue.front());
                resource.mRequestsQueue.pop_front();
            }
            {
                try
                {
                    TLLM_CHECK_WITH_INFO(requestAndPromise.mRequest != nullptr, "requestAndPromise.mRequest is null");
                    requestSync(*requestAndPromise.mRequest);
                    if (requestAndPromise.mRequest->getState() == LlmRequestState::kDISAGG_TRANS_ERROR
                        && requestAndPromise.mHasError != nullptr)
                    {
                        requestAndPromise.mHasError->store(true);
                    }
                    requestAndPromise.mPromise->set_value();
                }
                catch (tensorrt_llm::common::RequestSpecificException const& err)
                {
                    if (requestAndPromise.mHasError != nullptr)
                    {
                        requestAndPromise.mHasError->store(true);
                    }
                    TLLM_LOG_REQ_ERROR(requestAndPromise.mRequest->mRequestId,
                        "Exception in DataRequester request(): context_request_id=%zu: %s",
                        requestAndPromise.mRequest->getContextPhaseParams().value().getReqId(), err.what());
                    auto new_exception = TLLM_REQUEST_EXCEPTION(
                        requestAndPromise.mRequest->mRequestId, err.getErrorCode(), "%s", err.what());
                    requestAndPromise.mPromise->set_exception(std::make_exception_ptr(new_exception));
                }
                catch (std::exception const& err)
                {
                    if (requestAndPromise.mHasError != nullptr)
                    {
                        requestAndPromise.mHasError->store(true);
                    }
                    TLLM_LOG_REQ_ERROR(requestAndPromise.mRequest->mRequestId,
                        "Exception in CacheReceiver request(): context_request_id=%ld: %s",
                        requestAndPromise.mRequest->getContextPhaseParams().value().getReqId(), err.what());
                    requestAndPromise.mPromise->set_exception(std::current_exception());
                }
                clearRequestCancelFlag(requestAndPromise.mRequest->mRequestId);
            }
        }
    }

    std::mutex& getConnectionSendMutex(executor::kv_cache::Connection const* connection)
    {
        TLLM_CHECK(connection != nullptr);
        std::scoped_lock<std::mutex> lock(mConnectionSendMutexesMutex);
        auto it = mConnectionSendMutexes.find(connection);
        if (it == mConnectionSendMutexes.end())
        {
            it = mConnectionSendMutexes.emplace(connection, std::make_unique<std::mutex>()).first;
        }
        return *it->second;
    }

public:
    void setRnnConfig(executor::kv_cache::CacheState::RnnModelConfig rnnModelConfig,
        std::vector<SizeType32> rnnLayerNumPerPP, nvinfer1::DataType convStateDataType,
        nvinfer1::DataType ssmStateDataType)
    {
        mCacheTransferLayer.setRnnConfig(rnnModelConfig, rnnLayerNumPerPP, convStateDataType, ssmStateDataType);
        mSelfState.setCacheState(mCacheTransferLayer.getCacheState());
    }

private:
    int mDeviceId{-1};
    static constexpr char const* kDefaultProcessInfo = "default";
    std::vector<std::future<void>> mRequestFutures;
    std::unordered_map<std::string, std::unique_ptr<AsyncResource>> mInstanceToAsyncResource;
    executor::kv_cache::ConnectionManager* mManager;
    std::mutex mConnectionSendMutexesMutex;
    std::unordered_map<executor::kv_cache::Connection const*, std::unique_ptr<std::mutex>> mConnectionSendMutexes;
    executor::DataTransceiverState mSelfState;
    CacheTransferLayer mCacheTransferLayer;
    std::mutex mRequestCancelFlagsMutex;
    // Transfer sessions poll these flags outside the map mutex while cancellation sets them asynchronously.
    std::unordered_map<LlmRequest::RequestIdType, std::shared_ptr<std::atomic<bool>>> mRequestCancelFlags;
    // Receive resources are keyed by process/comm state and reused until CacheReceiver teardown.
    std::unordered_map<std::string, std::unique_ptr<ReceiveCacheResource>> mProcessToResources;
    std::mutex mProcessIoResouceMutex;
    runtime::BufferManager mBufferManager;
    std::ofstream mMeasuresFile;
    std::mutex mMeasuresFileMutex;
    std::atomic<bool> mTerminate{false};
};

void CacheSender::ImplDeleter::operator()(Impl* ptr)
{
    delete ptr;
}

void CacheReceiver::ImplDeleter::operator()(Impl* ptr)
{
    delete ptr;
}

CacheSender::CacheSender(
    executor::kv_cache::ConnectionManager* manager, SizeType32 selfIndex, CacheTransferLayer cacheLayer)
    : mImpl{std::unique_ptr<Impl, ImplDeleter>(new Impl(manager, selfIndex, std::move(cacheLayer)))}
{
}

std::future<void> CacheSender::sendAsync(std::shared_ptr<LlmRequest> const& llmRequest) const
{
    return mImpl->sendAsync(llmRequest);
}

executor::kv_cache::CommState const& CacheSender::getCommState() const
{
    return mImpl->getCommState();
}

void CacheSender::setCommState(executor::kv_cache::CommState commState)
{
    mImpl->setCommState(std::move(commState));
}

CacheSender::~CacheSender() = default;

void CacheSender::sendSync(LlmRequest const& llmRequest)
{
    mImpl->sendSync(llmRequest);
}

bool CacheSender::takeContextKvTransferEventReport(LlmRequest::RequestIdType requestId)
{
    return mImpl->takeContextKvTransferEventReport(requestId);
}

std::optional<RequestInfo> CacheSender::recvRequestInfo()
{
    return mImpl->recvRequestInfo();
}

bool CacheSender::cancelRequest(LlmRequest const& llmRequest)
{
    return mImpl->cancelRequest(llmRequest);
}

void CacheSender::sendReadySignal(LlmRequest::RequestIdType requestId, bool isReady)
{
    mImpl->sendReadySignal(requestId, isReady);
}

void CacheSender::setRnnConfig(executor::kv_cache::CacheState::RnnModelConfig rnnModelConfig,
    std::vector<SizeType32> rnnLayerNumPerPP, nvinfer1::DataType convStateDataType, nvinfer1::DataType ssmStateDataType)
{
    mImpl->setRnnConfig(std::move(rnnModelConfig), std::move(rnnLayerNumPerPP), convStateDataType, ssmStateDataType);
}

CacheReceiver::CacheReceiver(
    executor::kv_cache::ConnectionManager* manager, SizeType32 selfIndex, CacheTransferLayer cacheLayer)
    : mImpl{std::unique_ptr<Impl, ImplDeleter>(new Impl(manager, selfIndex, std::move(cacheLayer)))}
{
}

std::future<void> CacheReceiver::receiveAsync(std::shared_ptr<LlmRequest> const& llmRequest) const
{
    return mImpl->receiveAsync(llmRequest);
}

TransferStatusFuture CacheReceiver::receiveAsyncWithStatus(std::shared_ptr<LlmRequest> const& llmRequest) const
{
    return mImpl->requestAndReceiveAsyncMultiThreads(llmRequest);
}

CacheReceiver::~CacheReceiver() = default;

TransferSession CacheReceiver::sendRequestInfo(LlmRequest const& llmRequest)
{
    return mImpl->sendRequestInfo(llmRequest);
}

void CacheReceiver::receiveSync(TransferSession& session)
{
    mImpl->receiveSync(session);
}

bool CacheReceiver::cancelRequest(LlmRequest const& llmRequest)
{
    return mImpl->cancelRequest(llmRequest);
}

bool CacheReceiver::receiveReadySignal(TransferSession& session)
{
    return mImpl->receiveReadySignal(session);
}

void CacheReceiver::setRnnConfig(executor::kv_cache::CacheState::RnnModelConfig rnnModelConfig,
    std::vector<SizeType32> rnnLayerNumPerPP, nvinfer1::DataType convStateDataType, nvinfer1::DataType ssmStateDataType)
{
    mImpl->setRnnConfig(std::move(rnnModelConfig), std::move(rnnLayerNumPerPP), convStateDataType, ssmStateDataType);
}

} // namespace tensorrt_llm::batch_manager
