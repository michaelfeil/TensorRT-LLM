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

#include "tensorrt_llm/batch_manager/utils/cacheTransceiverDiagnostics.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>

namespace
{

using tensorrt_llm::batch_manager::LlmRequest;
using tensorrt_llm::batch_manager::utils::formatSessionNotFoundDiagnostic;

// Minimal stand-in for the real TransferSession — the helper only needs the
// map's keys and size(), so the value type can be anything.
struct DummySession
{
    int placeholder{0};
};

using SessionMap = std::unordered_map<LlmRequest::RequestIdType, DummySession>;
using CancelFlagMap = std::unordered_map<LlmRequest::RequestIdType, int>;
using CancelledSet = std::unordered_set<LlmRequest::RequestIdType>;

constexpr LlmRequest::RequestIdType kMissingId = 42;

// --------------------------------------------------------------------------
// Case 1: senderMutex is free — helper acquires it immediately and dumps the
// full diagnostic state including the cancelledRequests bit it reads under
// the lock.
// --------------------------------------------------------------------------
TEST(CacheTransceiverDiagnosticsTest, AcquiresLockAndFormatsAllFields)
{
    SessionMap sessions;
    sessions.emplace(1, DummySession{});
    sessions.emplace(2, DummySession{});
    sessions.emplace(3, DummySession{});

    CancelFlagMap cancelFlags;
    cancelFlags.emplace(kMissingId, 0);

    std::mutex senderMutex;
    CancelledSet cancelledRequests = {kMissingId};

    auto const result
        = formatSessionNotFoundDiagnostic(kMissingId, sessions, cancelFlags, senderMutex, cancelledRequests);

    EXPECT_NE(result.find("requestId=42"), std::string::npos);
    EXPECT_NE(result.find("activeSessions=3"), std::string::npos);
    EXPECT_NE(result.find("inCancelledRequests=true"), std::string::npos);
    EXPECT_NE(result.find("hasCancelFlag=true"), std::string::npos);
    // Active session ids should appear inside the bracketed list.
    auto const bracketStart = result.find('[');
    auto const bracketEnd = result.find(']', bracketStart);
    ASSERT_NE(bracketStart, std::string::npos);
    ASSERT_NE(bracketEnd, std::string::npos);
    auto const bracketed = result.substr(bracketStart, bracketEnd - bracketStart);
    EXPECT_NE(bracketed.find('1'), std::string::npos);
    EXPECT_NE(bracketed.find('2'), std::string::npos);
    EXPECT_NE(bracketed.find('3'), std::string::npos);
}

// --------------------------------------------------------------------------
// Case 2: senderMutex is held by another thread when the helper is called.
// The helper uses std::try_to_lock and must NOT block. It returns quickly,
// reporting inCancelledRequests=unknown because it could not safely read
// mCancelledRequests under contention. This is the production hot-path
// contract: a diagnostic formatter must not stall the thread that is about
// to throw.
// --------------------------------------------------------------------------
TEST(CacheTransceiverDiagnosticsTest, DoesNotBlockWhenLockHeldReportsUnknown)
{
    SessionMap sessions;
    sessions.emplace(10, DummySession{});

    CancelFlagMap cancelFlags;
    std::mutex senderMutex;
    CancelledSet cancelledRequests;

    constexpr auto kHoldDuration = std::chrono::milliseconds(150);
    std::atomic<bool> holderHasLock{false};
    std::atomic<bool> holderReleased{false};

    // Worker thread holds senderMutex for kHoldDuration.
    std::thread holder(
        [&]
        {
            std::scoped_lock lk(senderMutex);
            holderHasLock.store(true, std::memory_order_release);
            std::this_thread::sleep_for(kHoldDuration);
            holderReleased.store(true, std::memory_order_release);
        });

    // Wait until the worker has the lock so the helper call below is
    // guaranteed to be contended.
    while (!holderHasLock.load(std::memory_order_acquire))
    {
        std::this_thread::yield();
    }

    auto const start = std::chrono::steady_clock::now();
    auto const result
        = formatSessionNotFoundDiagnostic(kMissingId, sessions, cancelFlags, senderMutex, cancelledRequests);
    auto const elapsed = std::chrono::steady_clock::now() - start;

    holder.join();
    EXPECT_TRUE(holderReleased.load());

    // Helper must return well before the holder releases its lock; otherwise
    // the throw site stalls behind unrelated sender activity. Use kHoldDuration/5
    // as a generous ceiling so we tolerate scheduler jitter but still catch a
    // regression where the helper waits on a contended mutex.
    EXPECT_LT(elapsed, kHoldDuration / 5);

    // And it should have produced the diagnostic, with inCancelledRequests
    // reported as 'unknown' (try_to_lock failed under contention).
    EXPECT_NE(result.find("requestId=42"), std::string::npos);
    EXPECT_NE(result.find("activeSessions=1"), std::string::npos);
    EXPECT_NE(result.find("inCancelledRequests=unknown"), std::string::npos);
    EXPECT_NE(result.find("hasCancelFlag=false"), std::string::npos);
}

} // namespace
