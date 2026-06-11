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

#pragma once

#include <cstddef>
#include <cstdint>

namespace tensorrt_llm::executor::kv_cache
{

/// @brief Reasons a UCX request was cancelled. Mirrors the strings passed to
/// `cancelRequestWithLog` so Prometheus consumers can attribute cancellations.
enum class UcxCancelReason : int
{
    kOperationTimeout = 0,      ///< business timeout in waitForUcxRequestCompletion
    kTransferTerminated = 1,    ///< upstream set DataContext::transferTerminate
    kPipelinedChunkAborted = 2, ///< chunked pipeline cancelled in-flight chunks
    kOther = 3,                 ///< catch-all
    kReasonCount = 4
};

/// @brief Reasons a UCX connection bootstrap failed.
enum class UcxConnectionSetupErrorStage : int
{
    kZmqSend = 0,
    kZmqRecvTimeout = 1,
    kZmqRecv = 2,
    kEndpointInit = 3,
    kStageCount = 4
};

/// @brief Direction of a UCX tag operation, for per-direction counters.
enum class UcxTagOp : int
{
    kSend = 0, ///< tagSend (payload + control messages)
    kRecv = 1, ///< tagRecv
    kOpCount = 2
};

/// @brief Plain-old-data snapshot of all KV transfer counters, suitable for
/// nanobind exposure and Python-side polling.
///
/// All fields are cumulative counters except those documented as gauges.
/// Python consumers compute rates with `rate()` / `delta()` in PromQL or
/// track deltas locally.
struct KvTransferMetricsSnapshot
{
    // ---- staging buffer pool -------------------------------------------------
    int64_t stagingBufferInUse{0};           ///< gauge: buffers currently checked out
    int64_t stagingBufferPoolSize{0};        ///< gauge: configured pool capacity
    int64_t stagingBufferAcquireTotal{0};
    int64_t stagingBufferExhaustedTotal{0};  ///< acquire() returned via deadline path
    int64_t stagingBufferWaitMicrosTotal{0}; ///< cumulative wait time inside acquire()

    // ---- quarantine ----------------------------------------------------------
    int64_t stagingQuarantineSize{0};  ///< gauge: live quarantined requests
    int64_t stagingQuarantineBytes{0}; ///< gauge: cumulative bytes held in quarantine
    int64_t stagingQuarantineTotal{0}; ///< how many requests have ever been quarantined
    int64_t stagingQuarantineUnsafeReclaimTotal{0};

    // ---- UCX request lifecycle ----------------------------------------------
    int64_t ucxCancelTotal[static_cast<int>(UcxCancelReason::kReasonCount)] = {};
    int64_t ucxCancelGraceTimeoutTotal{0}; ///< the L1004 throw
    int64_t ucxOperationTimeoutTotal{0};   ///< the L1012 throw

    // ---- UCX connection bootstrap -------------------------------------------
    int64_t ucxConnectionSetupErrorsTotal[static_cast<int>(UcxConnectionSetupErrorStage::kStageCount)] = {};
    int64_t ucxActiveConnections{0}; ///< gauge
    int64_t ucxConnectionEstablishedTotal{0};

    // ---- UCX tag send/recv completions, indexed by UcxTagOp -----------------
    /// Successful (UCS_OK) tag op completions per direction.
    int64_t ucxTagOkTotal[static_cast<int>(UcxTagOp::kOpCount)] = {};
    /// Tag op completions with an error status (non-OK, excluding cancellations,
    /// which are tracked via ucxCancelTotal). Captures transport/endpoint errors.
    int64_t ucxTagErrorTotal[static_cast<int>(UcxTagOp::kOpCount)] = {};
    /// Cumulative wait time of *successful* tag ops, per direction. Average
    /// latency = ucxTagWaitMicrosTotal / ucxTagOkTotal (per op).
    int64_t ucxTagWaitMicrosTotal[static_cast<int>(UcxTagOp::kOpCount)] = {};
    /// Tag ops that exceeded their per-operation timeout, per direction. Subset
    /// of ucxOperationTimeoutTotal, broken down by send/recv.
    int64_t ucxTagTimeoutTotal[static_cast<int>(UcxTagOp::kOpCount)] = {};
};

/// @brief Increment helpers. All are no-throw, thread-safe, and use
/// `std::memory_order_relaxed` internally — counters do not order other memory
/// operations and consumers only care about eventual visibility.
namespace metrics
{

void recordStagingBufferAcquire(int64_t waitMicros) noexcept;
void recordStagingBufferRelease() noexcept;
void recordStagingBufferExhausted() noexcept;
void setStagingBufferPoolSize(int64_t poolSize) noexcept;

void recordQuarantineEnqueue(int64_t bytes) noexcept;
void recordQuarantineReclaim(int64_t bytes) noexcept;
void recordQuarantineUnsafeReclaim(int64_t bytes) noexcept;

void recordUcxCancel(UcxCancelReason reason) noexcept;
void recordUcxCancelGraceTimeout() noexcept;
void recordUcxOperationTimeout() noexcept;

void recordUcxConnectionSetupError(UcxConnectionSetupErrorStage stage) noexcept;
void recordUcxConnectionEstablished() noexcept;
void recordUcxConnectionClosed() noexcept;

/// Record a successful tag op completion and its wait latency (microseconds).
void recordUcxTagOk(UcxTagOp op, int64_t waitMicros) noexcept;
/// Record a tag op that completed with a (non-cancel) error status.
void recordUcxTagError(UcxTagOp op) noexcept;
/// Record a tag op that hit its per-operation timeout.
void recordUcxTagTimeout(UcxTagOp op) noexcept;

} // namespace metrics

/// @brief Atomically snapshot all counters. Cheap (single ~12-load read), safe
/// to call from any thread, including hot iteration loops.
KvTransferMetricsSnapshot getKvTransferMetricsSnapshot() noexcept;

} // namespace tensorrt_llm::executor::kv_cache
