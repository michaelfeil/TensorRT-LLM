/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include "tensorrt_llm/executor/cache_transmission/kvTransferMetrics.h"

#include <atomic>

namespace tensorrt_llm::executor::kv_cache
{

namespace
{

/// @brief Process-global counters. All atomics use relaxed ordering; counters
/// do not synchronize other memory and consumers tolerate eventual visibility.
struct Counters
{
    std::atomic<int64_t> stagingBufferInUse{0};
    std::atomic<int64_t> stagingBufferPoolSize{0};
    std::atomic<int64_t> stagingBufferAcquireTotal{0};
    std::atomic<int64_t> stagingBufferExhaustedTotal{0};
    std::atomic<int64_t> stagingBufferWaitMicrosTotal{0};

    std::atomic<int64_t> stagingQuarantineSize{0};
    std::atomic<int64_t> stagingQuarantineBytes{0};
    std::atomic<int64_t> stagingQuarantineTotal{0};
    std::atomic<int64_t> stagingQuarantineUnsafeReclaimTotal{0};

    std::atomic<int64_t> ucxCancelTotal[static_cast<int>(UcxCancelReason::kReasonCount)] = {};
    std::atomic<int64_t> ucxCancelGraceTimeoutTotal{0};
    std::atomic<int64_t> ucxOperationTimeoutTotal{0};

    std::atomic<int64_t> ucxConnectionSetupErrorsTotal[static_cast<int>(UcxConnectionSetupErrorStage::kStageCount)]
        = {};
    std::atomic<int64_t> ucxActiveConnections{0};
    std::atomic<int64_t> ucxConnectionEstablishedTotal{0};

    std::atomic<int64_t> ucxTagOkTotal[static_cast<int>(UcxTagOp::kOpCount)] = {};
    std::atomic<int64_t> ucxTagErrorTotal[static_cast<int>(UcxTagOp::kOpCount)] = {};
    std::atomic<int64_t> ucxTagWaitMicrosTotal[static_cast<int>(UcxTagOp::kOpCount)] = {};
    std::atomic<int64_t> ucxTagTimeoutTotal[static_cast<int>(UcxTagOp::kOpCount)] = {};
};

Counters& getCounters() noexcept
{
    static Counters c;
    return c;
}

constexpr auto kRelaxed = std::memory_order_relaxed;

} // namespace

namespace metrics
{

void recordStagingBufferAcquire(int64_t waitMicros) noexcept
{
    auto& c = getCounters();
    c.stagingBufferInUse.fetch_add(1, kRelaxed);
    c.stagingBufferAcquireTotal.fetch_add(1, kRelaxed);
    if (waitMicros > 0)
    {
        c.stagingBufferWaitMicrosTotal.fetch_add(waitMicros, kRelaxed);
    }
}

void recordStagingBufferRelease() noexcept
{
    getCounters().stagingBufferInUse.fetch_sub(1, kRelaxed);
}

void recordStagingBufferExhausted() noexcept
{
    getCounters().stagingBufferExhaustedTotal.fetch_add(1, kRelaxed);
}

void setStagingBufferPoolSize(int64_t poolSize) noexcept
{
    getCounters().stagingBufferPoolSize.store(poolSize, kRelaxed);
}

void recordQuarantineEnqueue(int64_t bytes) noexcept
{
    auto& c = getCounters();
    c.stagingQuarantineSize.fetch_add(1, kRelaxed);
    c.stagingQuarantineBytes.fetch_add(bytes, kRelaxed);
    c.stagingQuarantineTotal.fetch_add(1, kRelaxed);
}

void recordQuarantineReclaim(int64_t bytes) noexcept
{
    auto& c = getCounters();
    c.stagingQuarantineSize.fetch_sub(1, kRelaxed);
    c.stagingQuarantineBytes.fetch_sub(bytes, kRelaxed);
}

void recordQuarantineUnsafeReclaim(int64_t bytes) noexcept
{
    auto& c = getCounters();
    c.stagingQuarantineUnsafeReclaimTotal.fetch_add(1, kRelaxed);
    c.stagingQuarantineSize.fetch_sub(1, kRelaxed);
    c.stagingQuarantineBytes.fetch_sub(bytes, kRelaxed);
}

void recordUcxCancel(UcxCancelReason reason) noexcept
{
    auto const idx = static_cast<int>(reason);
    if (idx < 0 || idx >= static_cast<int>(UcxCancelReason::kReasonCount))
    {
        return;
    }
    getCounters().ucxCancelTotal[idx].fetch_add(1, kRelaxed);
}

void recordUcxCancelGraceTimeout() noexcept
{
    getCounters().ucxCancelGraceTimeoutTotal.fetch_add(1, kRelaxed);
}

void recordUcxOperationTimeout() noexcept
{
    getCounters().ucxOperationTimeoutTotal.fetch_add(1, kRelaxed);
}

void recordUcxConnectionSetupError(UcxConnectionSetupErrorStage stage) noexcept
{
    auto const idx = static_cast<int>(stage);
    if (idx < 0 || idx >= static_cast<int>(UcxConnectionSetupErrorStage::kStageCount))
    {
        return;
    }
    getCounters().ucxConnectionSetupErrorsTotal[idx].fetch_add(1, kRelaxed);
}

void recordUcxConnectionEstablished() noexcept
{
    auto& c = getCounters();
    c.ucxActiveConnections.fetch_add(1, kRelaxed);
    c.ucxConnectionEstablishedTotal.fetch_add(1, kRelaxed);
}

void recordUcxConnectionClosed() noexcept
{
    getCounters().ucxActiveConnections.fetch_sub(1, kRelaxed);
}

void recordUcxTagOk(UcxTagOp op, int64_t waitMicros) noexcept
{
    auto const idx = static_cast<int>(op);
    if (idx < 0 || idx >= static_cast<int>(UcxTagOp::kOpCount))
    {
        return;
    }
    auto& c = getCounters();
    c.ucxTagOkTotal[idx].fetch_add(1, kRelaxed);
    if (waitMicros > 0)
    {
        c.ucxTagWaitMicrosTotal[idx].fetch_add(waitMicros, kRelaxed);
    }
}

void recordUcxTagError(UcxTagOp op) noexcept
{
    auto const idx = static_cast<int>(op);
    if (idx < 0 || idx >= static_cast<int>(UcxTagOp::kOpCount))
    {
        return;
    }
    getCounters().ucxTagErrorTotal[idx].fetch_add(1, kRelaxed);
}

void recordUcxTagTimeout(UcxTagOp op) noexcept
{
    auto const idx = static_cast<int>(op);
    if (idx < 0 || idx >= static_cast<int>(UcxTagOp::kOpCount))
    {
        return;
    }
    getCounters().ucxTagTimeoutTotal[idx].fetch_add(1, kRelaxed);
}

} // namespace metrics

KvTransferMetricsSnapshot getKvTransferMetricsSnapshot() noexcept
{
    auto const& c = getCounters();
    KvTransferMetricsSnapshot s;
    s.stagingBufferInUse = c.stagingBufferInUse.load(kRelaxed);
    s.stagingBufferPoolSize = c.stagingBufferPoolSize.load(kRelaxed);
    s.stagingBufferAcquireTotal = c.stagingBufferAcquireTotal.load(kRelaxed);
    s.stagingBufferExhaustedTotal = c.stagingBufferExhaustedTotal.load(kRelaxed);
    s.stagingBufferWaitMicrosTotal = c.stagingBufferWaitMicrosTotal.load(kRelaxed);

    s.stagingQuarantineSize = c.stagingQuarantineSize.load(kRelaxed);
    s.stagingQuarantineBytes = c.stagingQuarantineBytes.load(kRelaxed);
    s.stagingQuarantineTotal = c.stagingQuarantineTotal.load(kRelaxed);
    s.stagingQuarantineUnsafeReclaimTotal = c.stagingQuarantineUnsafeReclaimTotal.load(kRelaxed);

    for (int i = 0; i < static_cast<int>(UcxCancelReason::kReasonCount); ++i)
    {
        s.ucxCancelTotal[i] = c.ucxCancelTotal[i].load(kRelaxed);
    }
    s.ucxCancelGraceTimeoutTotal = c.ucxCancelGraceTimeoutTotal.load(kRelaxed);
    s.ucxOperationTimeoutTotal = c.ucxOperationTimeoutTotal.load(kRelaxed);

    for (int i = 0; i < static_cast<int>(UcxConnectionSetupErrorStage::kStageCount); ++i)
    {
        s.ucxConnectionSetupErrorsTotal[i] = c.ucxConnectionSetupErrorsTotal[i].load(kRelaxed);
    }
    s.ucxActiveConnections = c.ucxActiveConnections.load(kRelaxed);
    s.ucxConnectionEstablishedTotal = c.ucxConnectionEstablishedTotal.load(kRelaxed);

    for (int i = 0; i < static_cast<int>(UcxTagOp::kOpCount); ++i)
    {
        s.ucxTagOkTotal[i] = c.ucxTagOkTotal[i].load(kRelaxed);
        s.ucxTagErrorTotal[i] = c.ucxTagErrorTotal[i].load(kRelaxed);
        s.ucxTagWaitMicrosTotal[i] = c.ucxTagWaitMicrosTotal[i].load(kRelaxed);
        s.ucxTagTimeoutTotal[i] = c.ucxTagTimeoutTotal[i].load(kRelaxed);
    }
    return s;
}

} // namespace tensorrt_llm::executor::kv_cache
