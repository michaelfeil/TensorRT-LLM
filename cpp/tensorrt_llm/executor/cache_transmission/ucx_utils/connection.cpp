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

#include "ucxCacheCommunicator.h"
#if ENABLE_UCX

#include "tensorrt_llm/batch_manager/dataTransceiver.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/tllmException.h"
#include "tensorrt_llm/executor/cache_transmission/kvTransferMetrics.h"
#include "tensorrt_llm/executor/cache_transmission/ucx_utils/connection.h"
#include "tensorrt_llm/executor/cache_transmission/ucx_utils/payloadStaging.h"
#include <cstdint>
#include <cstring>
#include <exception>
#include <future>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace tensorrt_llm::executor::kv_cache
{

namespace
{
constexpr int kDefaultConnectionHandshakeTimeoutMs = 5000;
constexpr char const* kUcxConnectionHandshakeTimeoutMsEnv = "TRTLLM_UCX_CONNECTION_HANDSHAKE_TIMEOUT_MS";
constexpr int32_t kTagTypeBits = 8;
constexpr int32_t kTagTypeMask = (1 << kTagTypeBits) - 1;

bool isHostControlTag(int tag)
{
    using tensorrt_llm::batch_manager::TransceiverTag;
    // Request data tags encode their type in the low bits. Keep ready-signal tags on the host-control path even if a
    // future caller passes an encoded tag instead of the bare kREADY_SIGNAL_TAG value.
    return tag == TransceiverTag::kINFO_SIZE_TAG || tag == TransceiverTag::kINFO_TAG
        || tag == TransceiverTag::kREADY_SIGNAL_TAG || (tag & kTagTypeMask) == TransceiverTag::kREADY_SIGNAL_TAG;
}

int getConnectionHandshakeTimeoutMs(int rank)
{
    auto const timeoutMs = getUcxRequestTimeoutMs(
        rank, kUcxConnectionHandshakeTimeoutMsEnv, kDefaultConnectionHandshakeTimeoutMs, "connection handshake");
    if (timeoutMs == 0)
    {
        TLLM_LOG_WARNING(rank, "%s=0 is unsupported for UCX connection handshakes; using default timeout of %d ms",
            kUcxConnectionHandshakeTimeoutMsEnv, kDefaultConnectionHandshakeTimeoutMs);
        return kDefaultConnectionHandshakeTimeoutMs;
    }
    return timeoutMs;
}

uint64_t makeConnectionHandshakeTag(UcxConnection::ConnectionIdType connectionId, uint64_t tag)
{
    return ((connectionId & 0xFFFFFFFF) << 32) | (tag & 0xFFFFFFFF);
}

void waitForConnectionHandshakeRequest(std::shared_ptr<ucxx::Request> const& req, std::future<void>& future,
    DataContext const& ctx, UcxConnectionManager* manager, char const* operation,
    ucxx::RequestCallbackUserData const& callbackData, ucxx::Endpoint* endpoint)
{
    auto const rank = manager->getRank();
    waitForUcxRequestCompletion(req, future, ctx, rank, operation, true, callbackData,
        sizeof(UcxConnection::ConnectionIdType), getConnectionHandshakeTimeoutMs(rank), endpoint);
    TLLM_CHECK_WITH_INFO(req->isCompleted(), "connection handshake request should be completed");
    req->checkError();
}

UcxConnection::ConnectionIdType recvConnectionHandshake(
    UcxConnectionManager* manager, ucxx::Endpoint& endpoint, ucxx::Tag recvTag, int contextTag)
{
    auto connectionId = std::make_shared<UcxConnection::ConnectionIdType>();
    auto promise = std::make_shared<std::promise<void>>();
    std::future<void> future = promise->get_future();
    auto completionCallback = [promise](ucs_status_t, ucxx::RequestCallbackUserData) -> void { promise->set_value(); };
    ucxx::RequestCallbackUserData callbackData = connectionId;
    std::shared_ptr<ucxx::Request> request = endpoint.tagRecv(connectionId.get(), sizeof(*connectionId), recvTag,
        ucxx::TagMaskFull, false, completionCallback, callbackData);
    if (!request->isCompleted())
    {
        waitForConnectionHandshakeRequest(
            request, future, DataContext{contextTag}, manager, "recvConnectionHandshake", callbackData, &endpoint);
    }
    else
    {
        request->checkError();
    }
    return *connectionId;
}

void sendConnectionHandshake(UcxConnectionManager* manager, ucxx::Endpoint& endpoint,
    UcxConnection::ConnectionIdType connectionId, ucxx::Tag sendTag, int contextTag)
{
    auto connectionIdBuffer = std::make_shared<UcxConnection::ConnectionIdType>(connectionId);
    auto promise = std::make_shared<std::promise<void>>();
    std::future<void> future = promise->get_future();
    auto completionCallback = [promise](ucs_status_t, ucxx::RequestCallbackUserData) -> void { promise->set_value(); };
    ucxx::RequestCallbackUserData callbackData = connectionIdBuffer;
    std::shared_ptr<ucxx::Request> request = endpoint.tagSend(connectionIdBuffer.get(), sizeof(*connectionIdBuffer),
        sendTag, false, completionCallback, callbackData);
    if (!request->isCompleted())
    {
        waitForConnectionHandshakeRequest(
            request, future, DataContext{contextTag}, manager, "sendConnectionHandshake", callbackData, &endpoint);
    }
    else
    {
        request->checkError();
    }
}

void sendPayloadWithoutStaging(ucxx::Endpoint& endpoint, uint64_t sendTag, void const* data, size_t size)
{
    std::promise<void> promise;
    std::future<void> future = promise.get_future();
    auto completionCallback = [&](ucs_status_t, ucxx::RequestCallbackUserData) -> void { promise.set_value(); };
    auto req = endpoint.tagSend(const_cast<void*>(data), size, ucxx::Tag(sendTag), false, completionCallback);
    if (!req->isCompleted())
    {
        future.get();
    }
    TLLM_CHECK_WITH_INFO(req->isCompleted(), "send should be completed");
    req->checkError();
}

void recvPayloadWithoutStaging(ucxx::Endpoint& endpoint, uint64_t recvTag, void* data, size_t size)
{
    std::promise<void> promise;
    std::future<void> future = promise.get_future();
    auto completionCallback = [&](ucs_status_t, ucxx::RequestCallbackUserData) -> void { promise.set_value(); };
    auto req = endpoint.tagRecv(data, size, ucxx::Tag(recvTag), ucxx::TagMaskFull, false, completionCallback);
    if (!req->isCompleted())
    {
        future.get();
    }
    TLLM_CHECK_WITH_INFO(req->isCompleted(), "recv should be completed");
    req->checkError();
}

void logSendEnd(UcxConnection::ConnectionIdType connectionId, UcxConnection::ConnectionIdType connectionIdInPeer,
    bool fromRequester, int rank)
{
    TLLM_LOG_DEBUG(rank, "end UcxConnection::send , mConnectionId: %lu, mConnectionIdInPeer: %lu,fromRequester: %d",
        connectionId, connectionIdInPeer, fromRequester);
}

void logRecvEnd(UcxConnection::ConnectionIdType connectionId, UcxConnection::ConnectionIdType connectionIdInPeer,
    bool fromRequester, int rank)
{
    TLLM_LOG_DEBUG(rank, "end UcxConnection::recv , mConnectionId: %lu, mConnectionIdInPeer: %lu,fromRequester: %d",
        connectionId, connectionIdInPeer, fromRequester);
}

} // namespace

UcxConnection::UcxConnection(ConnectionIdType connectionId, std::shared_ptr<ucxx::Endpoint> endpoint,
    UcxConnectionManager* manager, bool fromRequester, ConnectionIdType requesterConnectionId)
    : mConnectionId(connectionId)
    , mEndpoint(std::move(endpoint))
    , mManager(manager)
    , mFromRequester(fromRequester)
{

    try
    {
        if (mFromRequester)
        {

            // Receive the responder connection id on a requester-unique tag before switching to peer-specific tags.
            auto recvTag = ucxx::Tag(makeConnectionHandshakeTag(requesterConnectionId, ResponserTag));
            mConnectionIdInPeer = recvConnectionHandshake(
                mManager, *mEndpoint, recvTag, static_cast<int>(ResponserTag));

            auto sendTag = ucxx::Tag(makeConnectionHandshakeTag(mConnectionIdInPeer, RequesterTag));
            sendConnectionHandshake(mManager, *mEndpoint, mConnectionId, sendTag, static_cast<int>(RequesterTag));
        }
        else
        {

            // Send the responder connection id first so the requester can reply on a connection-specific tag.
            auto sendTag = ucxx::Tag(makeConnectionHandshakeTag(requesterConnectionId, ResponserTag));
            try
            {
                sendConnectionHandshake(
                    mManager, *mEndpoint, mConnectionId, sendTag, static_cast<int>(ResponserTag));
            }
            catch (...)
            {
                mManager->recordPassiveHandshakeFailureEvent(ResponserTag);
                throw;
            }

            auto recvTag = ucxx::Tag(makeConnectionHandshakeTag(mConnectionId, RequesterTag));
            try
            {
                mConnectionIdInPeer
                    = recvConnectionHandshake(mManager, *mEndpoint, recvTag, static_cast<int>(RequesterTag));
            }
            catch (...)
            {
                mManager->recordPassiveHandshakeFailureEvent(RequesterTag);
                throw;
            }
        }
    }
    catch (std::exception const& e)
    {
        std::string error = std::string("Error in UcxConnection constructor for rank ")
            + std::to_string(mManager->getRank()) + ": " + e.what();
        TLLM_THROW(error);
    }

    mSendTagPrefix = mConnectionIdInPeer;
    mRecvTagPrefix = mConnectionId;

    executor::kv_cache::metrics::recordUcxConnectionEstablished();
    TLLM_LOG_DEBUG(mManager->getRank(),
        "UcxConnection::UcxConnection, mConnectionId: %lu, mConnectionIdInPeer: %lu,fromRequester: %d", mConnectionId,
        mConnectionIdInPeer, mFromRequester);
}

UcxConnection::~UcxConnection()
{

    executor::kv_cache::metrics::recordUcxConnectionClosed();
    TLLM_LOG_DEBUG(mManager->getRank(),
        "UcxConnection::~UcxConnection, mConnectionId: %lu, mConnectionIdInPeer: %lu,fromRequester: %d", mConnectionId,
        mConnectionIdInPeer, mFromRequester);
    // TODO: how to close the endpoint safely?
}

void UcxConnection::sendConnectionId(DataContext const& ctx, void const* data, size_t size) const
{
    TLLM_LOG_DEBUG(mManager->getRank(),
        "start UcxConnection::sendConnectionId , mConnectionId: %lu, mConnectionIdInPeer: %lu,fromRequester: %d",
        mConnectionId, mConnectionIdInPeer, mFromRequester);

    auto promise = std::make_shared<std::promise<void>>();
    std::future<void> future = promise->get_future();
    auto completionCallback = [promise](ucs_status_t, ucxx::RequestCallbackUserData) -> void { promise->set_value(); };

    uint64_t tag = ((mSendTagPrefix & 0xFFFFFFFF) << 32)
        | static_cast<uint64_t>(tensorrt_llm::batch_manager::TransceiverTag::kID_TAG);
    auto buffer = std::make_shared<std::vector<char>>(size + sizeof(mConnectionId));
    memcpy(buffer->data(), data, size);
    memcpy(buffer->data() + size, &mConnectionIdInPeer, sizeof(mConnectionIdInPeer));
    auto req = mEndpoint->tagSend(buffer->data(), buffer->size(), ucxx::Tag(tag), false, completionCallback, buffer);
    if (!req->isCompleted())
    {
        waitForUcxRequestCompletion(req, future, ctx, mManager->getRank(), "sendConnectionId", true, buffer,
            buffer->size(), getUcxHostControlRequestTimeoutMs(mManager->getRank()), mEndpoint.get());
    }
    TLLM_CHECK_WITH_INFO(req->isCompleted(), "sendConnectionId should be completed");
    req->checkError();
    TLLM_LOG_DEBUG(mManager->getRank(),
        "end UcxConnection::sendConnectionId , mConnectionId: %lu, mConnectionIdInPeer: %lu,fromRequester: %d",
        mConnectionId, mConnectionIdInPeer, mFromRequester);
}

void UcxConnection::send(DataContext const& ctx, void const* data, size_t size) const
{
    if (ctx.getTag() == tensorrt_llm::batch_manager::TransceiverTag::kID_TAG)
    {
        sendConnectionId(ctx, data, size);
        return;
    }
    TLLM_LOG_DEBUG(mManager->getRank(),
        "start UcxConnection::send , mConnectionId: %lu, mConnectionIdInPeer: %lu,fromRequester: %d", mConnectionId,
        mConnectionIdInPeer, mFromRequester);

    TLLM_CHECK_WITH_INFO((mEndpoint), "sendBuffer called without established communicator channel.");
    uint64_t sendTag = ((mSendTagPrefix & 0xFFFFFFFF) << 32) | (static_cast<uint64_t>(ctx.getTag()) & (0xFFFFFFFF));
    int const rank = mManager->getRank();
    bool const hostControlTag = isHostControlTag(ctx.getTag());
    if (!hostControlTag)
    {
        if (isPayloadStagingEnabled(rank))
        {
            sendPayloadWithStaging(*mEndpoint, sendTag, ctx, data, size, rank);
        }
        else
        {
            sendPayloadWithoutStaging(*mEndpoint, sendTag, data, size);
        }
        logSendEnd(mConnectionId, mConnectionIdInPeer, mFromRequester, rank);
        return;
    }

    auto promise = std::make_shared<std::promise<void>>();
    std::future<void> future = promise->get_future();
    auto completionCallback = [promise](ucs_status_t, ucxx::RequestCallbackUserData) -> void { promise->set_value(); };
    auto hostControlBuffer = std::make_shared<std::vector<char>>(size);
    memcpy(hostControlBuffer->data(), data, size);
    ucxx::RequestCallbackUserData callbackData = hostControlBuffer;
    auto req = mEndpoint->tagSend(
        hostControlBuffer->data(), size, ucxx::Tag(sendTag), false, completionCallback, callbackData);
    if (!req->isCompleted())
    {
        waitForUcxRequestCompletion(req, future, ctx, rank, "send", true, callbackData, hostControlBuffer->size(),
            getUcxHostControlRequestTimeoutMs(rank), mEndpoint.get());
    }
    TLLM_CHECK_WITH_INFO(req->isCompleted(), "send should be completed");
    req->checkError();

    logSendEnd(mConnectionId, mConnectionIdInPeer, mFromRequester, rank);
}

void UcxConnection::recv(DataContext const& ctx, void* data, size_t size) const
{
    // Guard to ensure CUDA context is initialized for UCX ops
    TLLM_LOG_DEBUG(mManager->getRank(),
        "start UcxConnection::recv , mConnectionId: %lu, mConnectionIdInPeer: %lu,fromRequester: %d", mConnectionId,
        mConnectionIdInPeer, mFromRequester);
    TLLM_CHECK_WITH_INFO((mEndpoint), "recvBuffer called without established communicator channel.");
    uint64_t recvTag = ((mRecvTagPrefix & 0xFFFFFFFF) << 32) | (static_cast<uint64_t>(ctx.getTag()) & (0xFFFFFFFF));
    int const rank = mManager->getRank();
    bool const hostControlTag = isHostControlTag(ctx.getTag());
    if (!hostControlTag)
    {
        if (isPayloadStagingEnabled(rank))
        {
            recvPayloadWithStaging(*mEndpoint, recvTag, ctx, data, size, rank);
        }
        else
        {
            recvPayloadWithoutStaging(*mEndpoint, recvTag, data, size);
        }
        logRecvEnd(mConnectionId, mConnectionIdInPeer, mFromRequester, rank);
        return;
    }

    auto promise = std::make_shared<std::promise<void>>();
    std::future<void> future = promise->get_future();
    auto completionCallback = [promise](ucs_status_t, ucxx::RequestCallbackUserData) -> void { promise->set_value(); };
    auto hostControlBuffer = std::make_shared<std::vector<char>>(size);
    ucxx::RequestCallbackUserData callbackData = hostControlBuffer;
    auto req = mEndpoint->tagRecv(hostControlBuffer->data(), size, ucxx::Tag(recvTag), ucxx::TagMaskFull, false,
        completionCallback, callbackData);
    if (!req->isCompleted())
    {
        waitForUcxRequestCompletion(req, future, ctx, rank, "recv", true, callbackData, hostControlBuffer->size(),
            getUcxHostControlRequestTimeoutMs(rank), mEndpoint.get());
    }
    TLLM_CHECK_WITH_INFO(req->isCompleted(), "recv should be completed");
    req->checkError();
    memcpy(data, hostControlBuffer->data(), size);

    logRecvEnd(mConnectionId, mConnectionIdInPeer, mFromRequester, rank);
}

} // namespace tensorrt_llm::executor::kv_cache

#endif
