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

#include "tensorrt_llm/executor/cache_transmission/kvTransferMetrics.h"

#include <gtest/gtest.h>

namespace kvc = tensorrt_llm::executor::kv_cache;

// The counters are a process-global singleton with no reset hook, so every
// assertion is on the *delta* between two snapshots taken around the calls
// under test. This keeps the cases order-independent and tolerant of state
// left by other tests in the same binary.

TEST(KvTransferMetricsTest, StagingBufferAcquireWaitGuardAndRelease)
{
    auto const before = kvc::getKvTransferMetricsSnapshot();
    kvc::metrics::recordStagingBufferAcquire(123);
    kvc::metrics::recordStagingBufferAcquire(0); // waitMicros <= 0 must not add to the wait total
    auto const acquired = kvc::getKvTransferMetricsSnapshot();
    EXPECT_EQ(acquired.stagingBufferAcquireTotal - before.stagingBufferAcquireTotal, 2);
    EXPECT_EQ(acquired.stagingBufferInUse - before.stagingBufferInUse, 2);
    EXPECT_EQ(acquired.stagingBufferWaitMicrosTotal - before.stagingBufferWaitMicrosTotal, 123);

    kvc::metrics::recordStagingBufferRelease();
    kvc::metrics::recordStagingBufferRelease();
    auto const released = kvc::getKvTransferMetricsSnapshot();
    // in_use is a gauge: back to baseline after matching releases.
    EXPECT_EQ(released.stagingBufferInUse - before.stagingBufferInUse, 0);
    // acquire total is cumulative: not decremented by release.
    EXPECT_EQ(released.stagingBufferAcquireTotal - before.stagingBufferAcquireTotal, 2);
}

TEST(KvTransferMetricsTest, ExhaustedCounterAndPoolSizeGauge)
{
    auto const before = kvc::getKvTransferMetricsSnapshot();
    kvc::metrics::recordStagingBufferExhausted();
    kvc::metrics::setStagingBufferPoolSize(32);
    auto const after = kvc::getKvTransferMetricsSnapshot();
    EXPECT_EQ(after.stagingBufferExhaustedTotal - before.stagingBufferExhaustedTotal, 1);
    EXPECT_EQ(after.stagingBufferPoolSize, 32); // gauge is set to the absolute value
}

TEST(KvTransferMetricsTest, QuarantineEnqueueReclaimAndUnsafeReclaim)
{
    auto const before = kvc::getKvTransferMetricsSnapshot();

    kvc::metrics::recordQuarantineEnqueue(1000);
    auto const enqueued = kvc::getKvTransferMetricsSnapshot();
    EXPECT_EQ(enqueued.stagingQuarantineSize - before.stagingQuarantineSize, 1);
    EXPECT_EQ(enqueued.stagingQuarantineBytes - before.stagingQuarantineBytes, 1000);
    EXPECT_EQ(enqueued.stagingQuarantineTotal - before.stagingQuarantineTotal, 1);

    kvc::metrics::recordQuarantineReclaim(1000);
    auto const reclaimed = kvc::getKvTransferMetricsSnapshot();
    // size/bytes are gauges (decremented); total is cumulative (not decremented).
    EXPECT_EQ(reclaimed.stagingQuarantineSize - before.stagingQuarantineSize, 0);
    EXPECT_EQ(reclaimed.stagingQuarantineBytes - before.stagingQuarantineBytes, 0);
    EXPECT_EQ(reclaimed.stagingQuarantineTotal - before.stagingQuarantineTotal, 1);

    kvc::metrics::recordQuarantineEnqueue(500);
    kvc::metrics::recordQuarantineUnsafeReclaim(500);
    auto const unsafe = kvc::getKvTransferMetricsSnapshot();
    EXPECT_EQ(unsafe.stagingQuarantineUnsafeReclaimTotal - before.stagingQuarantineUnsafeReclaimTotal, 1);
    // enqueue + unsafe reclaim net to zero live size/bytes.
    EXPECT_EQ(unsafe.stagingQuarantineSize - before.stagingQuarantineSize, 0);
    EXPECT_EQ(unsafe.stagingQuarantineBytes - before.stagingQuarantineBytes, 0);
}

TEST(KvTransferMetricsTest, CancelCountedByReason)
{
    auto const before = kvc::getKvTransferMetricsSnapshot();
    kvc::metrics::recordUcxCancel(kvc::UcxCancelReason::kTransferTerminated);
    kvc::metrics::recordUcxCancel(kvc::UcxCancelReason::kTransferTerminated);
    kvc::metrics::recordUcxCancel(kvc::UcxCancelReason::kOther);
    kvc::metrics::recordUcxCancelGraceTimeout();
    kvc::metrics::recordUcxOperationTimeout();
    auto const after = kvc::getKvTransferMetricsSnapshot();

    auto const tt = static_cast<int>(kvc::UcxCancelReason::kTransferTerminated);
    auto const other = static_cast<int>(kvc::UcxCancelReason::kOther);
    auto const opTimeout = static_cast<int>(kvc::UcxCancelReason::kOperationTimeout);
    EXPECT_EQ(after.ucxCancelTotal[tt] - before.ucxCancelTotal[tt], 2);
    EXPECT_EQ(after.ucxCancelTotal[other] - before.ucxCancelTotal[other], 1);
    EXPECT_EQ(after.ucxCancelTotal[opTimeout] - before.ucxCancelTotal[opTimeout], 0);
    EXPECT_EQ(after.ucxCancelGraceTimeoutTotal - before.ucxCancelGraceTimeoutTotal, 1);
    EXPECT_EQ(after.ucxOperationTimeoutTotal - before.ucxOperationTimeoutTotal, 1);
}

TEST(KvTransferMetricsTest, CancelOutOfRangeReasonIsNoOp)
{
    auto const before = kvc::getKvTransferMetricsSnapshot();
    kvc::metrics::recordUcxCancel(static_cast<kvc::UcxCancelReason>(999));
    kvc::metrics::recordUcxCancel(static_cast<kvc::UcxCancelReason>(-1));
    auto const after = kvc::getKvTransferMetricsSnapshot();
    // No in-range counter moves and (importantly) no out-of-bounds write/crash.
    for (int i = 0; i < static_cast<int>(kvc::UcxCancelReason::kReasonCount); ++i)
    {
        EXPECT_EQ(after.ucxCancelTotal[i], before.ucxCancelTotal[i]);
    }
}

TEST(KvTransferMetricsTest, ConnectionGaugeAndSetupErrorByStage)
{
    auto const before = kvc::getKvTransferMetricsSnapshot();
    kvc::metrics::recordUcxConnectionEstablished();
    kvc::metrics::recordUcxConnectionEstablished();
    kvc::metrics::recordUcxConnectionClosed();
    kvc::metrics::recordUcxConnectionSetupError(kvc::UcxConnectionSetupErrorStage::kZmqRecvTimeout);
    auto const after = kvc::getKvTransferMetricsSnapshot();

    EXPECT_EQ(after.ucxConnectionEstablishedTotal - before.ucxConnectionEstablishedTotal, 2);
    // active is a gauge: +2 established, -1 closed => +1.
    EXPECT_EQ(after.ucxActiveConnections - before.ucxActiveConnections, 1);
    auto const stage = static_cast<int>(kvc::UcxConnectionSetupErrorStage::kZmqRecvTimeout);
    EXPECT_EQ(after.ucxConnectionSetupErrorsTotal[stage] - before.ucxConnectionSetupErrorsTotal[stage], 1);
}

TEST(KvTransferMetricsTest, TagOpOkErrorTimeoutByDirection)
{
    auto const before = kvc::getKvTransferMetricsSnapshot();
    auto const s = static_cast<int>(kvc::UcxTagOp::kSend);
    auto const r = static_cast<int>(kvc::UcxTagOp::kRecv);

    kvc::metrics::recordUcxTagOk(kvc::UcxTagOp::kSend, 100);
    kvc::metrics::recordUcxTagOk(kvc::UcxTagOp::kSend, 0); // waitMicros <= 0 must not add
    kvc::metrics::recordUcxTagOk(kvc::UcxTagOp::kRecv, 250);
    kvc::metrics::recordUcxTagError(kvc::UcxTagOp::kRecv);
    kvc::metrics::recordUcxTagTimeout(kvc::UcxTagOp::kSend);
    auto const after = kvc::getKvTransferMetricsSnapshot();

    EXPECT_EQ(after.ucxTagOkTotal[s] - before.ucxTagOkTotal[s], 2);
    EXPECT_EQ(after.ucxTagWaitMicrosTotal[s] - before.ucxTagWaitMicrosTotal[s], 100);
    EXPECT_EQ(after.ucxTagOkTotal[r] - before.ucxTagOkTotal[r], 1);
    EXPECT_EQ(after.ucxTagWaitMicrosTotal[r] - before.ucxTagWaitMicrosTotal[r], 250);
    EXPECT_EQ(after.ucxTagErrorTotal[r] - before.ucxTagErrorTotal[r], 1);
    EXPECT_EQ(after.ucxTagErrorTotal[s] - before.ucxTagErrorTotal[s], 0);
    EXPECT_EQ(after.ucxTagTimeoutTotal[s] - before.ucxTagTimeoutTotal[s], 1);
    EXPECT_EQ(after.ucxTagTimeoutTotal[r] - before.ucxTagTimeoutTotal[r], 0);
}

TEST(KvTransferMetricsTest, TagOpOutOfRangeDirectionIsNoOp)
{
    auto const before = kvc::getKvTransferMetricsSnapshot();
    kvc::metrics::recordUcxTagOk(static_cast<kvc::UcxTagOp>(99), 100);
    kvc::metrics::recordUcxTagError(static_cast<kvc::UcxTagOp>(99));
    kvc::metrics::recordUcxTagTimeout(static_cast<kvc::UcxTagOp>(-1));
    auto const after = kvc::getKvTransferMetricsSnapshot();
    for (int i = 0; i < static_cast<int>(kvc::UcxTagOp::kOpCount); ++i)
    {
        EXPECT_EQ(after.ucxTagOkTotal[i], before.ucxTagOkTotal[i]);
        EXPECT_EQ(after.ucxTagErrorTotal[i], before.ucxTagErrorTotal[i]);
        EXPECT_EQ(after.ucxTagTimeoutTotal[i], before.ucxTagTimeoutTotal[i]);
        EXPECT_EQ(after.ucxTagWaitMicrosTotal[i], before.ucxTagWaitMicrosTotal[i]);
    }
}
