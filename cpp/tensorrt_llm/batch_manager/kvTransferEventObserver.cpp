/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tensorrt_llm/batch_manager/kvTransferEventObserver.h"

#include "tensorrt_llm/batch_manager/cacheTransceiver.h"
#include "tensorrt_llm/batch_manager/dataTransceiver.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"

#include <numeric>

namespace tensorrt_llm::batch_manager
{
namespace
{

std::vector<LlmRequest::RequestIdType> gatherEventValues(
    std::shared_ptr<CacheTransceiverComm> const& comm, std::vector<LlmRequest::RequestIdType> const& values)
{
    int localSize = static_cast<int>(values.size());
    std::vector<int> sizes(comm->getSize());
    std::vector<LlmRequest::RequestIdType> retData;
    if (useMPI())
    {
        comm->allgather(&localSize, sizes.data(), 1, mpi::MpiType::kINT32);
        std::vector<int> displs(comm->getSize());
        size_t totalSize = 0;
        for (int i = 0; i < comm->getSize(); i++)
        {
            displs[i] = totalSize;
            totalSize += sizes[i];
        }
        if (totalSize == 0)
        {
            return {};
        }
        retData.resize(totalSize);
        comm->allgatherv(values.data(), static_cast<int>(values.size()), mpi::MpiType::kUINT64, retData.data(), sizes,
            displs, mpi::MpiType::kUINT64);
    }
    else
    {
        comm->allgather(&localSize, std::ref(sizes), {});
        size_t totalSize = std::accumulate(sizes.begin(), sizes.end(), 0);
        if (totalSize == 0)
        {
            return {};
        }
        retData.resize(totalSize);
        comm->allgatherv(std::ref(values), std::ref(retData), std::cref(sizes), {});
    }
    return retData;
}

void appendFlattenedKvTransferEvents(std::vector<LlmRequest::RequestIdType>& flattenedEvents,
    LlmRequest::RequestIdType status, std::vector<KvTransferEventRecord> const& events)
{
    for (auto const& event : events)
    {
        flattenedEvents.push_back(status);
        flattenedEvents.push_back(static_cast<LlmRequest::RequestIdType>(event.rank));
        flattenedEvents.push_back(event.requestId);
    }
}

void appendPendingErrorEvents(std::vector<KvTransferEventRecord>& events,
    std::unordered_set<LlmRequest::RequestIdType>& pendingRequestIds, int rank)
{
    for (auto const requestId : pendingRequestIds)
    {
        events.push_back({rank, requestId});
    }
    pendingRequestIds.clear();
}

} // namespace

bool KvTransferEventObserver::takeContextEventReport(CacheSender& cacheSender, LlmRequest* llmRequest)
{
    TLLM_CHECK(llmRequest != nullptr);
    return cacheSender.takeContextKvTransferEventReport(llmRequest->mRequestId);
}

void KvTransferEventObserver::recordContextFailure(CacheSender& cacheSender, LlmRequest::RequestIdType requestId)
{
    // Buffer immediately once the sender can confirm rank attribution; otherwise wait for status/cancel.
    if (cacheSender.takeContextKvTransferEventReport(requestId))
    {
        if (mReportedContextErrorEventIds.insert(requestId).second)
        {
            mBufferedContextErrorEvents.push_back({mRank, requestId});
        }
        mPendingContextErrorEventIds.erase(requestId);
        return;
    }
    mPendingContextErrorEventIds.insert(requestId);
}

void KvTransferEventObserver::recordGenerationFailure(LlmRequest::RequestIdType requestId)
{
    mPendingGenerationErrorEventIds.insert(requestId);
}

void KvTransferEventObserver::consumeContextEventReport(RequestStatuses& requestsStatus, CacheSender& cacheSender,
    LlmRequest::RequestIdType requestId, KvTransferResult transferResult, bool collectKvTransferEvents)
{
    // Preserve existing behavior: consume report markers even when stats collection is disabled.
    bool const alreadyReportedKvTransferError = mReportedContextErrorEventIds.erase(requestId) > 0;
    bool const reportKvTransferEvent = cacheSender.takeContextKvTransferEventReport(requestId);
    if (!collectKvTransferEvents || !reportKvTransferEvent || alreadyReportedKvTransferError)
    {
        return;
    }

    bool const pendingKvTransferError = mPendingContextErrorEventIds.erase(requestId) > 0;
    if (pendingKvTransferError || transferResult == KvTransferResult::kFailure)
    {
        requestsStatus.errorKvTransferEvents.push_back({mRank, requestId});
    }
    else
    {
        requestsStatus.completedKvTransferEvents.push_back({mRank, requestId});
    }
}

void KvTransferEventObserver::recordGenerationEvent(
    RequestStatuses& requestsStatus, LlmRequest::RequestIdType requestId, KvTransferResult transferResult,
    bool collectKvTransferEvents) const
{
    if (!collectKvTransferEvents)
    {
        return;
    }
    if (transferResult == KvTransferResult::kFailure)
    {
        requestsStatus.errorKvTransferEvents.push_back({mRank, requestId});
    }
    else
    {
        requestsStatus.completedKvTransferEvents.push_back({mRank, requestId});
    }
}

void KvTransferEventObserver::recordContextCancellation(CacheSender& cacheSender, LlmRequest::RequestIdType requestId)
{
    bool const pendingKvTransferError = mPendingContextErrorEventIds.find(requestId) != mPendingContextErrorEventIds.end();
    bool const reportKvTransferEvent = cacheSender.takeContextKvTransferEventReport(requestId);
    // Keep pending if rank attribution is not available yet; the detached future may provide it later.
    if (pendingKvTransferError && reportKvTransferEvent)
    {
        mPendingContextErrorEventIds.erase(requestId);
        if (mReportedContextErrorEventIds.insert(requestId).second)
        {
            mBufferedContextErrorEvents.push_back({mRank, requestId});
        }
    }
}

void KvTransferEventObserver::flushContextEvents(
    RequestStatuses& requestsStatus, CacheSender& cacheSender, std::vector<detail::TransferFuture> const& senderFutures)
{
    requestsStatus.errorKvTransferEvents.insert(requestsStatus.errorKvTransferEvents.end(),
        mBufferedContextErrorEvents.begin(), mBufferedContextErrorEvents.end());
    mBufferedContextErrorEvents.clear();

    if (mPendingContextErrorEventIds.empty())
    {
        return;
    }

    std::unordered_set<LlmRequest::RequestIdType> activeRequestIds;
    activeRequestIds.reserve(senderFutures.size());
    for (auto const& senderFuture : senderFutures)
    {
        activeRequestIds.insert(senderFuture.requestId);
    }

    for (auto it = mPendingContextErrorEventIds.begin(); it != mPendingContextErrorEventIds.end();)
    {
        auto const requestId = *it;
        if (cacheSender.takeContextKvTransferEventReport(requestId))
        {
            requestsStatus.errorKvTransferEvents.push_back({mRank, requestId});
            mReportedContextErrorEventIds.insert(requestId);
            it = mPendingContextErrorEventIds.erase(it);
        }
        else if (activeRequestIds.find(requestId) == activeRequestIds.end())
        {
            // No transfer session marked this rank reportable before the request left sender futures.
            // Drop it rather than reconstructing rank attribution from request-owned state.
            it = mPendingContextErrorEventIds.erase(it);
        }
        else
        {
            ++it;
        }
    }
}

void KvTransferEventObserver::flushGenerationEvents(RequestStatuses& requestsStatus)
{
    appendPendingErrorEvents(requestsStatus.errorKvTransferEvents, mPendingGenerationErrorEventIds, mRank);
}

void KvTransferEventObserver::gatherIfNeeded(
    RequestStatuses& requestsStatus, std::shared_ptr<CacheTransceiverComm> const& comm, bool enableAttentionDP) const
{
    if (enableAttentionDP || comm == nullptr || comm->getSize() <= 1)
    {
        return;
    }

    static constexpr LlmRequest::RequestIdType kCompletedEvent = 0;
    static constexpr LlmRequest::RequestIdType kErrorEvent = 1;
    std::vector<LlmRequest::RequestIdType> flattenedEvents;
    flattenedEvents.reserve(
        (requestsStatus.completedKvTransferEvents.size() + requestsStatus.errorKvTransferEvents.size()) * 3);
    appendFlattenedKvTransferEvents(flattenedEvents, kCompletedEvent, requestsStatus.completedKvTransferEvents);
    appendFlattenedKvTransferEvents(flattenedEvents, kErrorEvent, requestsStatus.errorKvTransferEvents);

    auto gatheredEvents = gatherEventValues(comm, flattenedEvents);
    if (comm->getRank() != 0)
    {
        requestsStatus.completedKvTransferEvents.clear();
        requestsStatus.errorKvTransferEvents.clear();
        return;
    }

    TLLM_CHECK_WITH_INFO(
        gatheredEvents.size() % 3 == 0, "KV transfer event gather returned an invalid number of values.");

    requestsStatus.completedKvTransferEvents.clear();
    requestsStatus.errorKvTransferEvents.clear();
    for (size_t i = 0; i < gatheredEvents.size(); i += 3)
    {
        auto const status = gatheredEvents[i];
        KvTransferEventRecord record{static_cast<int>(gatheredEvents[i + 1]), gatheredEvents[i + 2]};
        if (status == kCompletedEvent)
        {
            requestsStatus.completedKvTransferEvents.push_back(record);
        }
        else
        {
            TLLM_CHECK_WITH_INFO(status == kErrorEvent, "KV transfer event gather returned an unknown event status.");
            requestsStatus.errorKvTransferEvents.push_back(record);
        }
    }
}

} // namespace tensorrt_llm::batch_manager
