/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "tensorrt_llm/batch_manager/llmRequest.h"

#include <memory>
#include <unordered_set>
#include <vector>

namespace tensorrt_llm::batch_manager
{

class CacheSender;
class CacheTransceiverComm;
struct RequestStatuses;

namespace detail
{
struct TransferFuture;
} // namespace detail

struct KvTransferEventRecord
{
    int rank{-1};
    LlmRequest::RequestIdType requestId{};
};

enum class KvTransferResult
{
    kSuccess,
    kFailure,
};

class KvTransferEventObserver
{
public:
    explicit KvTransferEventObserver(int rank = -1)
        : mRank(rank)
    {
    }

    [[nodiscard]] bool takeContextEventReport(CacheSender& cacheSender, LlmRequest* llmRequest);

    void recordContextFailure(CacheSender& cacheSender, LlmRequest::RequestIdType requestId);

    void recordGenerationFailure(LlmRequest::RequestIdType requestId);

    void consumeContextEventReport(RequestStatuses& requestsStatus, CacheSender& cacheSender,
        LlmRequest::RequestIdType requestId, KvTransferResult transferResult, bool collectKvTransferEvents);

    void recordGenerationEvent(
        RequestStatuses& requestsStatus, LlmRequest::RequestIdType requestId, KvTransferResult transferResult,
        bool collectKvTransferEvents) const;

    void recordContextEvent(
        RequestStatuses& requestsStatus, LlmRequest::RequestIdType requestId, KvTransferResult transferResult,
        bool collectKvTransferEvents) const;

    void recordContextCancellation(CacheSender& cacheSender, LlmRequest::RequestIdType requestId);

    void flushContextEvents(
        RequestStatuses& requestsStatus, CacheSender& cacheSender,
        std::vector<detail::TransferFuture> const& senderFutures);

    void flushGenerationEvents(RequestStatuses& requestsStatus);

    void gatherIfNeeded(
        RequestStatuses& requestsStatus, std::shared_ptr<CacheTransceiverComm> const& comm, bool enableAttentionDP) const;

private:
    std::vector<KvTransferEventRecord> mBufferedContextErrorEvents;
    std::unordered_set<LlmRequest::RequestIdType> mPendingContextErrorEventIds;
    // Kept until the detached sender future drains, so late completion cannot emit a duplicate KV event.
    std::unordered_set<LlmRequest::RequestIdType> mReportedContextErrorEventIds;
    std::unordered_set<LlmRequest::RequestIdType> mPendingGenerationErrorEventIds;
    int mRank{-1};
};

} // namespace tensorrt_llm::batch_manager
