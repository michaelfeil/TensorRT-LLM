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

#include "tensorrt_llm/batch_manager/perRequestActivityLog.h"

#include <gtest/gtest.h>

#include <atomic>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace
{

using tensorrt_llm::batch_manager::LlmRequest;
using tensorrt_llm::batch_manager::PerRequestActivityLog;

constexpr LlmRequest::RequestIdType kReqA = 1001;
constexpr LlmRequest::RequestIdType kReqB = 1002;

class PerRequestActivityLogTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // The singleton is shared with the rest of the process; clear it
        // and restore default flag/rank state so every test starts known.
        auto& log = PerRequestActivityLog::instance();
        log.clear();
        log.setEmitLiveForTest(false);
        log.setRank(-1);
    }

    void TearDown() override
    {
        PerRequestActivityLog::instance().clear();
    }
};

// --------------------------------------------------------------------------
// Basic record + dump round-trip.
// --------------------------------------------------------------------------
TEST_F(PerRequestActivityLogTest, RecordsAndDumpsEventsInOrder)
{
    auto& log = PerRequestActivityLog::instance();
    log.record(kReqA, "session_added", "test");
    log.record(kReqA, "send_started", "test");
    log.record(kReqA, "send_completed", "test");

    auto const dumped = log.dump(kReqA);
    ASSERT_FALSE(dumped.empty());
    // Every line carries the grep-able prefix and the request id.
    EXPECT_NE(dumped.find(PerRequestActivityLog::kLinePrefix), std::string::npos);
    EXPECT_NE(dumped.find("request=1001"), std::string::npos);
    EXPECT_NE(dumped.find("session_added"), std::string::npos);
    EXPECT_NE(dumped.find("send_started"), std::string::npos);
    EXPECT_NE(dumped.find("send_completed"), std::string::npos);
    // Order should match call sequence.
    EXPECT_LT(dumped.find("session_added"), dumped.find("send_started"));
    EXPECT_LT(dumped.find("send_started"), dumped.find("send_completed"));
}

// --------------------------------------------------------------------------
// Every emitted line (header + events) is prefixed with kLinePrefix and the
// request id, so the dump can be grepped/filtered line-by-line (e.g. in Loki).
// --------------------------------------------------------------------------
TEST_F(PerRequestActivityLogTest, EveryLineCarriesGrepablePrefix)
{
    auto& log = PerRequestActivityLog::instance();
    log.setRank(7);
    // Side is derived from the event origin (CacheSender::* -> 'P').
    log.record(kReqA, "session_added", "CacheSender::recvRequestInfo");
    log.record(kReqA, "send_started", "CacheSender::sendSync");
    log.record(kReqA, "send_completed", "CacheSender::sendSync");

    auto const dumped = log.dump(kReqA);
    ASSERT_FALSE(dumped.empty());

    // Side + rank are part of the grep-able prefix on every line.
    EXPECT_NE(dumped.find("side=P"), std::string::npos);
    EXPECT_NE(dumped.find("rank=7"), std::string::npos);
    std::string const expectedPrefix
        = std::string(PerRequestActivityLog::kLinePrefix) + " side=P" + " rank=7" + " request=" + "1001";
    std::istringstream iss(dumped);
    std::string line;
    std::size_t lineCount = 0;
    while (std::getline(iss, line))
    {
        if (line.empty())
        {
            continue;
        }
        EXPECT_EQ(line.rfind(expectedPrefix, 0), 0U) << "line missing prefix: " << line;
        ++lineCount;
    }
    // 1 header + 3 event lines.
    EXPECT_EQ(lineCount, 4U);
}

// --------------------------------------------------------------------------
// side is derived from the event origin, not a process-global: CacheSender::*
// -> "side=P", CacheReceiver::* -> "side=D", anything else -> "side=?".
// --------------------------------------------------------------------------
TEST_F(PerRequestActivityLogTest, SideDerivedFromOrigin)
{
    auto& log = PerRequestActivityLog::instance();
    log.record(kReqA, "send_started", "CacheSender::sendSync");
    log.record(kReqB, "recv_request_started", "CacheReceiver::requestSync");
    constexpr LlmRequest::RequestIdType kReqC = 1003;
    log.record(kReqC, "evt", "SomethingElse");
    // Formatter format() is a send-side path -> 'P'.
    constexpr LlmRequest::RequestIdType kReqD = 1004;
    log.recordDetail(kReqD, "send_kvcache", "MlaCacheFormatter::format", "blocks=2 tokens=64");

    EXPECT_NE(log.dump(kReqA).find("side=P"), std::string::npos);
    EXPECT_NE(log.dump(kReqB).find("side=D"), std::string::npos);
    EXPECT_NE(log.dump(kReqC).find("side=?"), std::string::npos);
    EXPECT_NE(log.dump(kReqD).find("side=P"), std::string::npos);
}

// --------------------------------------------------------------------------
// dump() of an unknown id returns empty string.
// --------------------------------------------------------------------------
TEST_F(PerRequestActivityLogTest, DumpUnknownIdReturnsEmpty)
{
    EXPECT_EQ(PerRequestActivityLog::instance().dump(kReqA), "");
}

// --------------------------------------------------------------------------
// Payload variant is rendered when present.
// --------------------------------------------------------------------------
TEST_F(PerRequestActivityLogTest, RecordsPayload)
{
    auto& log = PerRequestActivityLog::instance();
    log.record(kReqA, "bytes_sent", "test", 4096);
    auto const dumped = log.dump(kReqA);
    EXPECT_NE(dumped.find("bytes_sent"), std::string::npos);
    EXPECT_NE(dumped.find("payload=4096"), std::string::npos);
}

// --------------------------------------------------------------------------
// Free-form detail is rendered verbatim after the "(origin)" field.
// --------------------------------------------------------------------------
TEST_F(PerRequestActivityLogTest, RecordsDetail)
{
    auto& log = PerRequestActivityLog::instance();
    log.recordDetail(kReqA, "send_kvcache", "test", "tokens=512 layers=[8,16) blocks=4");
    auto const dumped = log.dump(kReqA);
    EXPECT_NE(dumped.find("send_kvcache"), std::string::npos);
    EXPECT_NE(dumped.find("tokens=512 layers=[8,16) blocks=4"), std::string::npos);
}

// --------------------------------------------------------------------------
// Per-rank transfer-bytes detail (the send_completed / recv_request_completed
// format) fits in the detail buffer and appears in the dump.
// --------------------------------------------------------------------------
TEST_F(PerRequestActivityLogTest, BytesDetailAppearsInDump)
{
    auto& log = PerRequestActivityLog::instance();
    log.recordDetail(kReqA, "send_completed", "CacheSender::sendSync", "bytes=44040192 connections=1");
    log.recordDetail(kReqB, "recv_request_completed", "CacheReceiver::requestSync",
        "gen_request_id=8647372539334656 bytes=0 connections=0");

    auto const sent = log.dump(kReqA);
    EXPECT_NE(sent.find("bytes=44040192"), std::string::npos);
    EXPECT_NE(sent.find("connections=1"), std::string::npos);

    auto const recvd = log.dump(kReqB);
    EXPECT_NE(recvd.find("gen_request_id=8647372539334656"), std::string::npos);
    EXPECT_NE(recvd.find("bytes=0"), std::string::npos);
    // Realistic receiver detail must fit without truncation.
    EXPECT_NE(recvd.find("connections=0"), std::string::npos);
}

// --------------------------------------------------------------------------
// Detail longer than the fixed inline buffer is truncated, not overflowed.
// --------------------------------------------------------------------------
TEST_F(PerRequestActivityLogTest, DetailIsTruncated)
{
    auto& log = PerRequestActivityLog::instance();
    std::string const longDetail(PerRequestActivityLog::kDetailLen * 2, 'x');
    log.recordDetail(kReqA, "evt", "test", longDetail.c_str());
    auto const dumped = log.dump(kReqA);
    ASSERT_FALSE(dumped.empty());
    // At most kDetailLen-1 detail chars survive (the buffer is NUL-terminated).
    std::string const fullLen(PerRequestActivityLog::kDetailLen, 'x');
    EXPECT_EQ(dumped.find(fullLen), std::string::npos);
}

// --------------------------------------------------------------------------
// evictionCount() counts entries dropped by the tracked-request cap; the live
// set stays capped at kMaxTrackedRequests.
// --------------------------------------------------------------------------
TEST_F(PerRequestActivityLogTest, EvictionCountTracksDrops)
{
    auto& log = PerRequestActivityLog::instance();
    EXPECT_EQ(log.evictionCount(), 0U);

    constexpr std::size_t kOverflow = 10;
    for (std::size_t i = 0; i < PerRequestActivityLog::kMaxTrackedRequests + kOverflow; ++i)
    {
        log.record(static_cast<LlmRequest::RequestIdType>(100000 + i), "e", "t");
    }

    EXPECT_EQ(log.trackedRequestCount(), PerRequestActivityLog::kMaxTrackedRequests);
    EXPECT_GE(log.evictionCount(), kOverflow);
}

// --------------------------------------------------------------------------
// release(id) drops the entry; subsequent dump() returns empty.
// --------------------------------------------------------------------------
TEST_F(PerRequestActivityLogTest, ReleaseRemovesEntry)
{
    auto& log = PerRequestActivityLog::instance();
    log.record(kReqA, "session_added", "test");
    EXPECT_FALSE(log.dump(kReqA).empty());
    log.release(kReqA);
    EXPECT_TRUE(log.dump(kReqA).empty());
    // release on absent id is idempotent.
    log.release(kReqA);
    EXPECT_TRUE(log.dump(kReqA).empty());
}

// --------------------------------------------------------------------------
// Ring buffer wrap-around: when more than kMaxEventsPerRequest events are
// recorded, the oldest are overwritten but the most recent are still visible.
// --------------------------------------------------------------------------
TEST_F(PerRequestActivityLogTest, RingBufferWrapsAround)
{
    auto& log = PerRequestActivityLog::instance();
    constexpr std::size_t kOverfill = PerRequestActivityLog::kMaxEventsPerRequest + 5;
    for (std::size_t i = 0; i < kOverfill; ++i)
    {
        // Record with a payload that doubles as an identifier so we can check
        // which events are kept after wrap-around.
        log.record(kReqA, "evt", "test", static_cast<std::int64_t>(i));
    }
    auto const dumped = log.dump(kReqA);
    // Header should mark the log as wrapped.
    EXPECT_NE(dumped.find(", wrapped"), std::string::npos);
    // The newest event must be visible.
    EXPECT_NE(dumped.find("payload=" + std::to_string(kOverfill - 1)), std::string::npos);
    // And the oldest events must have been overwritten.
    EXPECT_EQ(dumped.find("payload=0\n"), std::string::npos);
    EXPECT_EQ(dumped.find("payload=1\n"), std::string::npos);
}

// --------------------------------------------------------------------------
// trackedRequestCount tracks unique request ids; release() decrements it.
// --------------------------------------------------------------------------
TEST_F(PerRequestActivityLogTest, TrackedRequestCount)
{
    auto& log = PerRequestActivityLog::instance();
    EXPECT_EQ(log.trackedRequestCount(), 0U);
    log.record(kReqA, "e", "t");
    log.record(kReqB, "e", "t");
    log.record(kReqA, "e", "t"); // same id again, not new
    EXPECT_EQ(log.trackedRequestCount(), 2U);
    log.release(kReqA);
    EXPECT_EQ(log.trackedRequestCount(), 1U);
}

// --------------------------------------------------------------------------
// Thread safety: many threads concurrently recording on the same id must not
// crash, and every event posted must be observed (modulo ring-buffer wrap).
// --------------------------------------------------------------------------
TEST_F(PerRequestActivityLogTest, ConcurrentRecordsAreSafe)
{
    auto& log = PerRequestActivityLog::instance();
    constexpr int kThreadCount = 8;
    constexpr int kPerThread = 64;

    std::atomic<bool> start{false};
    std::vector<std::thread> threads;
    threads.reserve(kThreadCount);
    for (int t = 0; t < kThreadCount; ++t)
    {
        threads.emplace_back(
            [&]
            {
                while (!start.load(std::memory_order_acquire))
                {
                    std::this_thread::yield();
                }
                for (int i = 0; i < kPerThread; ++i)
                {
                    log.record(kReqA, "e", "t");
                }
            });
    }
    start.store(true, std::memory_order_release);
    for (auto& th : threads)
    {
        th.join();
    }
    // The log should still be dumpable and report a wrapped buffer (we
    // recorded 512 events into a 32-slot ring).
    auto const dumped = log.dump(kReqA);
    EXPECT_FALSE(dumped.empty());
    EXPECT_NE(dumped.find(", wrapped"), std::string::npos);
}

// --------------------------------------------------------------------------
// kMaxTrackedRequests eviction: oldest entry (by last-event timestamp) must
// be the one dropped when the singleton is over the soft cap. Verifies the
// contract documented next to kMaxTrackedRequests.
// --------------------------------------------------------------------------
TEST_F(PerRequestActivityLogTest, EvictsOldestByLastEventTimestampWhenOverflowing)
{
    auto& log = PerRequestActivityLog::instance();
    constexpr std::size_t kSentinelCount = 5;

    // Record kSentinelCount ids first with small delays between them so each
    // sentinel has a strictly earlier last-event timestamp than every id
    // recorded in the bulk phase below.
    for (std::size_t i = 0; i < kSentinelCount; ++i)
    {
        log.record(static_cast<LlmRequest::RequestIdType>(1 + i), "old", "test");
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    // Bulk-record kMaxTrackedRequests fresh ids. After this we have
    // kSentinelCount + kMaxTrackedRequests distinct ids, so kSentinelCount
    // evictions should have fired.
    for (std::size_t i = 0; i < PerRequestActivityLog::kMaxTrackedRequests; ++i)
    {
        log.record(static_cast<LlmRequest::RequestIdType>(10000 + i), "new", "test");
    }

    // Sentinels are the oldest by last-event timestamp, so they should have
    // been the ones evicted.
    for (std::size_t i = 0; i < kSentinelCount; ++i)
    {
        EXPECT_TRUE(log.dump(static_cast<LlmRequest::RequestIdType>(1 + i)).empty())
            << "sentinel id " << (1 + i) << " should have been evicted but is still tracked";
    }

    // The freshly-recorded bulk ids should still be there.
    EXPECT_FALSE(log.dump(static_cast<LlmRequest::RequestIdType>(10000)).empty());
    EXPECT_FALSE(
        log.dump(static_cast<LlmRequest::RequestIdType>(10000 + PerRequestActivityLog::kMaxTrackedRequests - 1))
            .empty());

    EXPECT_EQ(log.trackedRequestCount(), PerRequestActivityLog::kMaxTrackedRequests);
}

// --------------------------------------------------------------------------
// Recording a request id after release() must give a fresh slot (writeIdx=0,
// wrapped=false). Otherwise the next failure dump would carry stale state
// from a previous, unrelated request lifecycle that happened to reuse the id.
// --------------------------------------------------------------------------
TEST_F(PerRequestActivityLogTest, RecordAfterReleaseGetsFreshSlot)
{
    auto& log = PerRequestActivityLog::instance();

    // Fill past the per-request ring capacity so the wrapped flag is set.
    for (std::size_t i = 0; i < PerRequestActivityLog::kMaxEventsPerRequest + 5; ++i)
    {
        log.record(kReqA, "fill", "test");
    }
    auto const dumpedFull = log.dump(kReqA);
    ASSERT_NE(dumpedFull.find(", wrapped"), std::string::npos);

    log.release(kReqA);
    EXPECT_TRUE(log.dump(kReqA).empty());

    log.record(kReqA, "fresh", "test");
    auto const dumpedFresh = log.dump(kReqA);
    EXPECT_FALSE(dumpedFresh.empty());
    EXPECT_EQ(dumpedFresh.find(", wrapped"), std::string::npos)
        << "writeIdx and wrapped flag must reset after release; dump was: " << dumpedFresh;
    EXPECT_NE(dumpedFresh.find("fresh"), std::string::npos);
}

// --------------------------------------------------------------------------
// EmitLive does not affect the in-memory ring buffer — record() still writes
// to the buffer regardless. The flag only controls whether a parallel
// KV_TX_STATUS line is emitted through TLLM_LOG_REQ_INFO; capturing that
// stderr output is environment-dependent, so we assert the buffer side here.
// --------------------------------------------------------------------------
TEST_F(PerRequestActivityLogTest, EmitLiveDoesNotAffectBuffer)
{
    auto& log = PerRequestActivityLog::instance();
    log.setEmitLiveForTest(true);
    log.record(kReqA, "e1", "t");
    log.record(kReqA, "e2", "t", 99);
    auto const dumpedOn = log.dump(kReqA);
    EXPECT_NE(dumpedOn.find("e1"), std::string::npos);
    EXPECT_NE(dumpedOn.find("e2"), std::string::npos);

    log.setEmitLiveForTest(false);
    log.record(kReqB, "e3", "t");
    auto const dumpedOff = log.dump(kReqB);
    EXPECT_NE(dumpedOff.find("e3"), std::string::npos);
}

// --------------------------------------------------------------------------
// Labeled payload: the 5-arg overload should render the payload using the
// caller-supplied label in both dump() output and the live KV_TX_STATUS line.
// Used by CacheReceiver call sites to report "gen_request_id" instead of the
// generic "payload".
// --------------------------------------------------------------------------
TEST_F(PerRequestActivityLogTest, LabeledPayloadAppearsInDump)
{
    auto& log = PerRequestActivityLog::instance();
    log.record(kReqA, "recv_request_started", "CacheReceiver::requestSync", "gen_request_id", 42);
    auto const dumped = log.dump(kReqA);
    EXPECT_NE(dumped.find("gen_request_id=42"), std::string::npos);
    // Must NOT carry the default "payload=" prefix when a custom label was used.
    EXPECT_EQ(dumped.find(" payload="), std::string::npos);
}

} // namespace
