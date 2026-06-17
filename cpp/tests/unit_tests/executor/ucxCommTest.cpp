/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#define UCX_WRAPPER_LIB_NAME "tensorrt_llm_ucx_wrapper"

#if defined(_WIN32)
#include <windows.h>
#define dllOpen(name) LoadLibrary(name ".dll")
#define dllClose(handle) FreeLibrary(static_cast<HMODULE>(handle))
#define dllGetSym(handle, name) static_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle), name))
#else // For non-Windows platforms
#include <dlfcn.h>
#define dllOpen(name) dlopen("lib" name ".so", RTLD_LAZY)
#define dllClose(handle) dlclose(handle)
#define dllGetSym(handle, name) dlsym(handle, name)
#endif // defined(_WIN32)

#include "tensorrt_llm/batch_manager/cacheFormatter.h"
#include "tensorrt_llm/batch_manager/cacheTransceiver.h"
#include "tensorrt_llm/batch_manager/dataTransceiver.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/executor/cache_transmission/mpi_utils/connection.h"
#include "tensorrt_llm/executor/dataTransceiverState.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include "gtest/gtest.h"
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <gmock/gmock.h>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <sstream>
#include <tensorrt_llm/batch_manager/mlaCacheFormatter.h>
#include <tensorrt_llm/executor/cache_transmission/cacheSplitConcat.h>
#include <thread>

using SizeType32 = tensorrt_llm::runtime::SizeType32;
using LlmRequest = tensorrt_llm::batch_manager::LlmRequest;
using namespace tensorrt_llm::batch_manager::kv_cache_manager;
using namespace tensorrt_llm::batch_manager;
namespace texec = tensorrt_llm::executor;

namespace
{
std::mutex mDllMutex;

std::unique_ptr<texec::kv_cache::ConnectionManager> makeOneUcxConnectionManager()
{
    std::lock_guard<std::mutex> lock(mDllMutex);
    void* WrapperLibHandle{nullptr};
    WrapperLibHandle = dllOpen(UCX_WRAPPER_LIB_NAME);
    TLLM_CHECK_WITH_INFO(WrapperLibHandle != nullptr, "UCX wrapper library is not open correctly.");
    auto load_sym = [](void* handle, char const* name)
    {
        void* ret = dllGetSym(handle, name);

        TLLM_CHECK_WITH_INFO(ret != nullptr,
            "Unable to load UCX wrapper library symbol, possible cause is that TensorRT LLM library is not "
            "built with UCX support, please rebuild in UCX-enabled environment.");
        return ret;
    };
    std::unique_ptr<tensorrt_llm::executor::kv_cache::ConnectionManager> (*makeUcxConnectionManager)();
    *(void**) (&makeUcxConnectionManager) = load_sym(WrapperLibHandle, "makeUcxConnectionManager");
    return makeUcxConnectionManager();
}

bool isUcxWrapperUnavailable(std::string const& error)
{
    return error.find("UCX wrapper library is not open correctly") != std::string::npos
        || error.find("Unable to load UCX wrapper library symbol") != std::string::npos;
}

class UcxCommTest : public ::testing::Test
{
};

using DataContext = tensorrt_llm::executor::kv_cache::DataContext;
using TransceiverTag = tensorrt_llm::batch_manager::TransceiverTag;

class ScopedEnvVar
{
public:
    ScopedEnvVar(char const* name, char const* value)
        : mName(name)
    {
        char const* oldValue = std::getenv(name);
        if (oldValue != nullptr)
        {
            mHadOldValue = true;
            mOldValue = oldValue;
        }
        TLLM_CHECK_WITH_INFO(setenv(name, value, 1) == 0, "Failed to set %s: %s", name, std::strerror(errno));
    }

    ~ScopedEnvVar() noexcept
    {
        int result = 0;
        if (!mHadOldValue)
        {
            result = unsetenv(mName);
        }
        else
        {
            result = setenv(mName, mOldValue.c_str(), 1);
        }
        if (result != 0)
        {
            ADD_FAILURE() << "Failed to restore " << mName << ": " << std::strerror(errno);
        }
    }

private:
    char const* mName;
    bool mHadOldValue{false};
    std::string mOldValue;
};

bool waitForFuturesUntil(std::vector<std::future<void>>& futures, std::chrono::steady_clock::time_point deadline)
{
    for (auto& future : futures)
    {
        auto const now = std::chrono::steady_clock::now();
        if (now >= deadline || future.wait_until(deadline) != std::future_status::ready)
        {
            return false;
        }
    }
    return true;
}

void collectReadyFutures(std::vector<std::future<void>>& futures)
{
    for (auto& future : futures)
    {
        if (future.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready)
        {
            continue;
        }
        try
        {
            future.get();
        }
        catch (std::exception const&)
        {
        }
    }
}

void runConcurrentConnectionTest(char const* testName,
    std::vector<texec::kv_cache::ConnectionManager*> const& requesters,
    std::vector<texec::kv_cache::ConnectionManager*> const& receivers)
{
    constexpr auto kTestTimeout = std::chrono::seconds(20);
    ScopedEnvVar handshakeTimeout("TRTLLM_UCX_CONNECTION_HANDSHAKE_TIMEOUT_MS", "5000");
    ScopedEnvVar hostControlTimeout("TRTLLM_UCX_HOST_CONTROL_TIMEOUT_MS", "5000");

    ASSERT_EQ(requesters.size(), receivers.size());
    ASSERT_FALSE(requesters.empty());

    std::vector<texec::kv_cache::CommState> receiverStates;
    receiverStates.reserve(receivers.size());
    for (auto* receiver : receivers)
    {
        ASSERT_NE(receiver, nullptr);
        receiverStates.emplace_back(receiver->getCommState());
        ASSERT_TRUE(receiverStates.back().isSocketState());
    }
    for (auto* requester : requesters)
    {
        ASSERT_NE(requester, nullptr);
    }

    std::atomic<bool> terminate{false};
    std::atomic<bool> timedOut{false};
    auto const deadline = std::chrono::steady_clock::now() + kTestTimeout;
    std::mutex errorMutex;
    std::vector<std::string> workerErrors;
    std::mutex watchdogMutex;
    std::condition_variable watchdogCv;
    bool receiverDone{false};

    std::promise<void> startPromise;
    std::shared_future<void> startFuture = startPromise.get_future().share();
    std::vector<std::future<void>> senderFutures;
    senderFutures.reserve(requesters.size());
    for (size_t idx = 0; idx < requesters.size(); idx++)
    {
        auto* requester = requesters.at(idx);
        senderFutures.emplace_back(std::async(std::launch::async,
            [requester, &receiverStates, &startFuture, &terminate, &errorMutex, &workerErrors, idx, testName]()
            {
                try
                {
                    startFuture.wait();
                    auto connections = requester->getConnections(receiverStates.at(idx));
                    TLLM_CHECK_WITH_INFO(
                        connections.size() == 1, "Expected exactly one UCX connection in %s test", testName);
                    uint64_t const id = idx + 1;
                    connections.at(0)->send(DataContext{TransceiverTag::kID_TAG, terminate}, &id, sizeof(id));
                    uint64_t ack{};
                    connections.at(0)->recv(DataContext{TransceiverTag::kINFO_SIZE_TAG, terminate}, &ack, sizeof(ack));
                    TLLM_CHECK_WITH_INFO(
                        ack == id, "Unexpected UCX %s ack: expected %lu, got %lu", testName, id, ack);
                }
                catch (std::exception const& e)
                {
                    {
                        std::scoped_lock lock(errorMutex);
                        workerErrors.emplace_back(e.what());
                    }
                    terminate.store(true);
                    throw;
                }
            }));
    }

    std::thread watchdog(
        [&]()
        {
            std::unique_lock lock(watchdogMutex);
            if (!watchdogCv.wait_for(lock, kTestTimeout, [&receiverDone]() { return receiverDone; }))
            {
                timedOut.store(true);
                terminate.store(true);
            }
        });

    std::set<uint64_t> receivedIds;
    std::string receiverError;
    startPromise.set_value();
    try
    {
        for (auto* receiver : receivers)
        {
            uint64_t receivedId{};
            auto const* connection = receiver->recvConnect(
                DataContext{TransceiverTag::kID_TAG, terminate}, &receivedId, sizeof(receivedId));
            if (connection == nullptr)
            {
                break;
            }
            receivedIds.insert(receivedId);
            connection->send(DataContext{TransceiverTag::kINFO_SIZE_TAG, terminate}, &receivedId, sizeof(receivedId));
        }
    }
    catch (std::exception const& e)
    {
        receiverError = e.what();
        terminate.store(true);
    }

    if (!waitForFuturesUntil(senderFutures, deadline))
    {
        timedOut.store(true);
        terminate.store(true);
        waitForFuturesUntil(senderFutures, std::chrono::steady_clock::now() + std::chrono::seconds(10));
    }

    {
        std::scoped_lock lock(watchdogMutex);
        receiverDone = true;
    }
    watchdogCv.notify_one();
    watchdog.join();

    collectReadyFutures(senderFutures);
    ASSERT_FALSE(timedOut.load()) << "Timed out waiting for concurrent UCX " << testName << " test";
    ASSERT_TRUE(receiverError.empty()) << receiverError;
    if (!workerErrors.empty())
    {
        std::ostringstream errorStream;
        for (auto const& error : workerErrors)
        {
            errorStream << error << '\n';
        }
        FAIL() << errorStream.str();
    }
    ASSERT_EQ(receivedIds.size(), requesters.size());
    for (size_t idx = 0; idx < requesters.size(); idx++)
    {
        EXPECT_NE(receivedIds.find(idx + 1), receivedIds.end());
    }
}

TEST_F(UcxCommTest, Basic)
{

    try
    {
        TransceiverTag::Id id1;
        TransceiverTag::Id id2;

        auto connectionManager1 = makeOneUcxConnectionManager();
        EXPECT_NE(connectionManager1, nullptr);
        auto connectionManager2 = makeOneUcxConnectionManager();
        EXPECT_NE(connectionManager2, nullptr);
        auto CommState1 = connectionManager1->getCommState();
        auto CommState2 = connectionManager2->getCommState();
        ASSERT_EQ(CommState1.isSocketState(), true);
        ASSERT_EQ(CommState2.isSocketState(), true);

        auto connections1 = connectionManager2->getConnections(CommState1);
        ASSERT_EQ(connections1.size(), 1);
        auto connection1 = connections1[0];
        id1 = TransceiverTag::Id::REQUEST_SEND;
        connection1->send(DataContext{TransceiverTag::kID_TAG}, &id1, sizeof(id1));

        auto connection1Peer = connectionManager1->recvConnect(DataContext{TransceiverTag::kID_TAG}, &id2, sizeof(id2));
        ASSERT_EQ(id2, id1);
        constexpr size_t bufferSize = 1024;
        std::vector<char> buffer(bufferSize);
        // Fill buffer with random data
        std::generate(buffer.begin(), buffer.end(), []() { return static_cast<char>(std::rand()); });

        connection1->send(DataContext{0x74}, buffer.data(), buffer.size());

        std::vector<char> recvBuffer(buffer.size());

        connection1Peer->recv(DataContext{0x74}, recvBuffer.data(), recvBuffer.size());

        ASSERT_EQ(memcmp(buffer.data(), recvBuffer.data(), buffer.size()), 0);

        // Test with CUDA memory
        tensorrt_llm::runtime::BufferManager bufferManager{std::make_shared<tensorrt_llm::runtime::CudaStream>()};

        // Create and fill source CUDA buffer with random data
        auto srcBuffer = bufferManager.gpu(buffer.size(), nvinfer1::DataType::kINT8);
        bufferManager.copy(buffer.data(), *srcBuffer);
        bufferManager.getStream().synchronize();

        auto dstBuffer = bufferManager.gpu(buffer.size(), nvinfer1::DataType::kINT8);

        // Send CUDA buffer using connection1
        connection1->send(DataContext{0x75}, srcBuffer->data(), srcBuffer->getSizeInBytes());

        // Receive into CUDA buffer using connection1Peer
        connection1Peer->recv(DataContext{0x75}, dstBuffer->data(), dstBuffer->getSizeInBytes());

        std::vector<char> recvCudaBuffer(buffer.size());
        bufferManager.copy(*dstBuffer, recvCudaBuffer.data(), dstBuffer->getMemoryType());
        bufferManager.getStream().synchronize();

        ASSERT_EQ(memcmp(buffer.data(), recvCudaBuffer.data(), buffer.size()), 0);
    }
    catch (std::exception const& e)
    {
        std::string error = e.what();
        if (isUcxWrapperUnavailable(error))
        {
            GTEST_SKIP() << "UCX wrapper library is not open correctly. Skip this test case.";
        }

        throw;
    }
}

TEST_F(UcxCommTest, multiSend)
{
    try
    {
        TransceiverTag::Id id1;
        TransceiverTag::Id id2;
        TransceiverTag::Id id1Peer;
        TransceiverTag::Id id2Peer;

        auto manager1 = makeOneUcxConnectionManager();
        auto manager2 = makeOneUcxConnectionManager();
        auto managerRecv = makeOneUcxConnectionManager();

        auto connection1 = managerRecv->getConnections(manager1->getCommState())[0];
        auto connection2 = managerRecv->getConnections(manager2->getCommState())[0];
        id1 = TransceiverTag::Id::REQUEST_SEND;
        id2 = TransceiverTag::Id::REQUEST_SEND;
        connection1->send(DataContext{TransceiverTag::kID_TAG}, &id1, sizeof(id1));
        connection2->send(DataContext{TransceiverTag::kID_TAG}, &id2, sizeof(id2));
        auto connection1Peer = manager1->recvConnect(DataContext{TransceiverTag::kID_TAG}, &id1Peer, sizeof(id1Peer));
        auto connection2Peer = manager2->recvConnect(DataContext{TransceiverTag::kID_TAG}, &id2Peer, sizeof(id2Peer));
        ASSERT_EQ(id1Peer, id1);
        ASSERT_EQ(id2Peer, id2);
        constexpr size_t bufferSize = 1024;
        std::vector<char> buffer1(bufferSize);
        std::vector<char> buffer2(bufferSize);
        std::generate(buffer1.begin(), buffer1.end(), []() { return static_cast<char>(std::rand()); });
        std::generate(buffer2.begin(), buffer2.end(), []() { return static_cast<char>(std::rand()); });

        connection1Peer->send(DataContext{0x74}, buffer1.data(), buffer1.size());
        connection2Peer->send(DataContext{0x74}, buffer2.data(), buffer2.size());

        std::vector<char> recvBuffer1(buffer1.size());
        std::vector<char> recvBuffer2(buffer2.size());
        connection2->recv(DataContext{0x74}, recvBuffer2.data(), recvBuffer2.size());
        connection1->recv(DataContext{0x74}, recvBuffer1.data(), recvBuffer1.size());
        ASSERT_EQ(memcmp(buffer1.data(), recvBuffer1.data(), buffer1.size()), 0);
        ASSERT_EQ(memcmp(buffer2.data(), recvBuffer2.data(), buffer2.size()), 0);

        tensorrt_llm::runtime::BufferManager bufferManager{std::make_shared<tensorrt_llm::runtime::CudaStream>()};

        auto srcBuffer1 = bufferManager.gpu(buffer1.size(), nvinfer1::DataType::kINT8);
        auto srcBuffer2 = bufferManager.gpu(buffer2.size(), nvinfer1::DataType::kINT8);
        bufferManager.copy(buffer1.data(), *srcBuffer1);
        bufferManager.copy(buffer2.data(), *srcBuffer2);
        bufferManager.getStream().synchronize();

        auto dstBuffer1 = bufferManager.gpu(buffer1.size(), nvinfer1::DataType::kINT8);
        auto dstBuffer2 = bufferManager.gpu(buffer2.size(), nvinfer1::DataType::kINT8);

        connection1Peer->send(DataContext{0x75}, srcBuffer1->data(), srcBuffer1->getSizeInBytes());
        connection2Peer->send(DataContext{0x75}, srcBuffer2->data(), srcBuffer2->getSizeInBytes());
        connection2->recv(DataContext{0x75}, dstBuffer2->data(), dstBuffer2->getSizeInBytes());

        connection1->recv(DataContext{0x75}, dstBuffer1->data(), dstBuffer1->getSizeInBytes());
        std::vector<char> recvCudaBuffer1(buffer1.size());
        std::vector<char> recvCudaBuffer2(buffer2.size());
        bufferManager.copy(*dstBuffer1, recvCudaBuffer1.data(), dstBuffer1->getMemoryType());
        bufferManager.copy(*dstBuffer2, recvCudaBuffer2.data(), dstBuffer2->getMemoryType());
        bufferManager.getStream().synchronize();

        ASSERT_EQ(memcmp(buffer1.data(), recvCudaBuffer1.data(), buffer1.size()), 0);
        ASSERT_EQ(memcmp(buffer2.data(), recvCudaBuffer2.data(), buffer2.size()), 0);
    }
    catch (std::exception const& e)
    {
        std::string error = e.what();
        if (isUcxWrapperUnavailable(error))
        {
            GTEST_SKIP() << "UCX wrapper library is not open correctly. Skip this test case.";
        }

        throw;
    }
}

TEST_F(UcxCommTest, concurrentRequestersToOneReceiver)
{
    try
    {
        constexpr size_t kRequesterCount = 8;

        auto receiverManager = makeOneUcxConnectionManager();

        std::vector<std::unique_ptr<texec::kv_cache::ConnectionManager>> requesterManagers;
        std::vector<texec::kv_cache::ConnectionManager*> requesters;
        std::vector<texec::kv_cache::ConnectionManager*> receivers;
        requesterManagers.reserve(kRequesterCount);
        requesters.reserve(kRequesterCount);
        receivers.reserve(kRequesterCount);
        for (size_t idx = 0; idx < kRequesterCount; idx++)
        {
            requesterManagers.emplace_back(makeOneUcxConnectionManager());
            requesters.push_back(requesterManagers.back().get());
            receivers.push_back(receiverManager.get());
        }
        runConcurrentConnectionTest("fan-in", requesters, receivers);
    }
    catch (std::exception const& e)
    {
        std::string error = e.what();
        if (isUcxWrapperUnavailable(error))
        {
            GTEST_SKIP() << "UCX wrapper library is not open correctly. Skip this test case.";
        }

        throw;
    }
}

TEST_F(UcxCommTest, concurrentRequesterToMultipleReceivers)
{
    try
    {
        constexpr size_t kReceiverCount = 8;

        auto requesterManager = makeOneUcxConnectionManager();
        std::vector<std::unique_ptr<texec::kv_cache::ConnectionManager>> receiverManagers;
        std::vector<texec::kv_cache::ConnectionManager*> requesters;
        std::vector<texec::kv_cache::ConnectionManager*> receivers;
        receiverManagers.reserve(kReceiverCount);
        requesters.reserve(kReceiverCount);
        receivers.reserve(kReceiverCount);
        for (size_t idx = 0; idx < kReceiverCount; idx++)
        {
            receiverManagers.emplace_back(makeOneUcxConnectionManager());
            requesters.push_back(requesterManager.get());
            receivers.push_back(receiverManagers.back().get());
        }
        runConcurrentConnectionTest("fan-out", requesters, receivers);
    }
    catch (std::exception const& e)
    {
        std::string error = e.what();
        if (isUcxWrapperUnavailable(error))
        {
            GTEST_SKIP() << "UCX wrapper library is not open correctly. Skip this test case.";
        }

        throw;
    }
}

TEST_F(UcxCommTest, CommCache)
{

    try
    {
        TransceiverTag::Id id1;
        TransceiverTag::Id id2;

        auto connectionManager1 = makeOneUcxConnectionManager();
        EXPECT_NE(connectionManager1, nullptr);
        auto connectionManager2 = makeOneUcxConnectionManager();
        EXPECT_NE(connectionManager2, nullptr);
        auto CommState1 = connectionManager1->getCommState();
        auto CommState2 = connectionManager2->getCommState();
        ASSERT_EQ(CommState1.isSocketState(), true);
        ASSERT_EQ(CommState2.isSocketState(), true);

        auto connections1 = connectionManager2->getConnections(CommState1);
        ASSERT_EQ(connections1.size(), 1);
        auto connection1 = connections1[0];
        id1 = TransceiverTag::Id::REQUEST_SEND;
        connection1->send(DataContext{TransceiverTag::kID_TAG}, &id1, sizeof(id1));

        auto connection1Peer = connectionManager1->recvConnect(DataContext{TransceiverTag::kID_TAG}, &id2, sizeof(id2));
        ASSERT_EQ(id2, id1);
        auto connection1Cached = connectionManager2->getConnections(CommState1)[0];
        ASSERT_EQ(connection1Cached, connection1);
        id1 = TransceiverTag::Id::REQUEST_SEND;
        connection1Cached->send(DataContext{TransceiverTag::kID_TAG}, &id1, sizeof(id1));

        auto connection1PeerCached
            = connectionManager1->recvConnect(DataContext{TransceiverTag::kID_TAG}, &id2, sizeof(id2));
        ASSERT_EQ(id2, id1);

        ASSERT_EQ(connection1PeerCached, connection1Peer);
    }
    catch (std::exception const& e)
    {
        std::string error = e.what();
        if (isUcxWrapperUnavailable(error))
        {
            GTEST_SKIP() << "UCX wrapper library is not open correctly. Skip this test case.";
        }

        throw;
    }
}

}; // namespace
