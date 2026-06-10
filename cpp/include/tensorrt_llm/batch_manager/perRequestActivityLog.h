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

#pragma once

#include "tensorrt_llm/batch_manager/llmRequest.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

namespace tensorrt_llm::batch_manager
{

/// Per-request in-memory activity log used to capture the lifecycle of a
/// KV cache transfer request. The log is intended for post-mortem dumping at
/// C++-side failure sites (e.g. `CacheSender::release` when the session has
/// already been released by another path) so we can see *what* happened to
/// that request before the failure, not just the failure itself.
///
/// Bounded ring buffer per request id; older events are overwritten once the
/// buffer fills. Call `release(id)` after the request is fully done to free
/// the per-request slot.
///
/// Recorded event/origin strings are expected to be string literals (no
/// ownership taken). Process-global singleton; thread-safe.
class PerRequestActivityLog
{
public:
    /// Common prefix prepended to every line emitted by `dump()` so the full
    /// activity log for a request can be grepped/filtered line-by-line (e.g.
    /// `|~ "KV_TX_STATUS"` in Loki).
    static constexpr char const* kLinePrefix = "KV_TX_STATUS";

    static constexpr std::size_t kMaxEventsPerRequest = 32;
    /// Soft cap on the number of distinct request ids tracked at once. When
    /// the singleton holds more than this, the oldest entry (by last-event
    /// timestamp) is evicted on the next record() call. This guards against
    /// permanent leaks if some code path forgets to call release(id).
    static constexpr std::size_t kMaxTrackedRequests = 4096;
    /// Capacity of the fixed inline detail buffer (including the NUL
    /// terminator). Free-form details (e.g. "tokens=512 layers=[8,16) blocks=4")
    /// are copied in and truncated to fit; kept small on purpose because every
    /// Event carries one. See the static_assert below before changing.
    static constexpr std::size_t kDetailLen = 96;

    struct Event
    {
        std::chrono::steady_clock::time_point ts;
        char const* event{nullptr};
        char const* origin{nullptr};
        // Optional integer payload + its label (e.g. "bytes", "gen_request_id").
        // Label defaults to "payload" for the legacy 4-arg record() overload.
        char const* payloadLabel{"payload"};
        std::int64_t payload{0};
        bool hasPayload{false};
        // Optional free-form detail, copied into this fixed inline buffer (no
        // heap). Empty (detail[0] == '\0') when unused.
        char detail[kDetailLen]{};
    };

    // The per-request ring buffer is kMaxEventsPerRequest * sizeof(Event), held
    // for up to kMaxTrackedRequests requests. Guard against silent memory bloat
    // when fields are added or kDetailLen is raised: revisit those (and this
    // budget) deliberately rather than letting Event grow unnoticed.
    static_assert(sizeof(Event) <= 160,
        "PerRequestActivityLog::Event exceeded its size budget; reconsider kDetailLen or new fields.");

    [[nodiscard]] static PerRequestActivityLog& instance() noexcept;

    /// Set the process/executor rank shown on every emitted line ("rank=<r>").
    /// Process-global and constant; set once during transceiver init. Reads -1
    /// ("unknown") until set. Safe to call from multiple init paths with the
    /// same value.
    void setRank(int rank) noexcept
    {
        mRank.store(rank, std::memory_order_relaxed);
    }

    [[nodiscard]] int rank() const noexcept
    {
        return mRank.load(std::memory_order_relaxed);
    }

    /// Append an event without an extra integer payload.
    void record(LlmRequest::RequestIdType id, char const* event, char const* origin) noexcept;

    /// Append an event with an extra integer payload. The payload is logged
    /// and dumped as "payload=N" (legacy label kept for backward compatibility
    /// at sender call sites; receiver call sites should use the labeled
    /// overload below).
    void record(LlmRequest::RequestIdType id, char const* event, char const* origin, std::int64_t payload) noexcept;

    /// Append an event with a labeled integer payload. The payload is logged
    /// and dumped as "<payloadLabel>=N". Use for semantically-typed values
    /// (e.g. "bytes", "gen_request_id", "peer") so log readers can grep by
    /// the field name.
    void record(LlmRequest::RequestIdType id, char const* event, char const* origin, char const* payloadLabel,
        std::int64_t payload) noexcept;

    /// Append an event with a free-form detail string for multi-valued context
    /// (e.g. "tokens=512 layers=[8,16) blocks=4"). The detail is copied into a
    /// fixed inline buffer and truncated to kDetailLen-1 chars; it is logged and
    /// dumped verbatim after the "(origin)" field. Distinct from the labeled
    /// overload above so the 4th arg is never mistaken for a payload label.
    void recordDetail(LlmRequest::RequestIdType id, char const* event, char const* origin, char const* detail) noexcept;

    /// Render the activity log for `id` as a multi-line string. Returns an
    /// empty string if no events have been recorded for `id`.
    [[nodiscard]] std::string dump(LlmRequest::RequestIdType id) const;

    /// Drop all events for `id`. Idempotent. Call when the request is fully
    /// done so the singleton does not grow unbounded.
    void release(LlmRequest::RequestIdType id) noexcept;

    /// Drop all events for all requests. Mainly for tests / shutdown.
    void clear() noexcept;

    /// Number of requests currently tracked in the log.
    [[nodiscard]] std::size_t trackedRequestCount() const noexcept;

    /// Cumulative number of per-request entries dropped by the kMaxTrackedRequests
    /// cap (i.e. evicted before an explicit release). A large/growing value means
    /// activity logs are being lost — the cap is too small for the workload, or
    /// some path is not releasing — and dumps for evicted requests will be empty.
    /// Reset by clear().
    [[nodiscard]] std::uint64_t evictionCount() const noexcept
    {
        return mEvictionCount.load(std::memory_order_relaxed);
    }

    /// Whether each successful record() additionally emits a single
    /// KV_TX_STATUS line via TLLM_LOG_REQ_INFO. Set once at construction from
    /// env TRTLLM_ACTLOG_EMIT_LIVE ("1"/"true"/"on" enables, anything else
    /// or unset disables). Immutable thereafter — change requires restart.
    [[nodiscard]] bool isEmitLive() const noexcept
    {
        return mEmitLive.load(std::memory_order_relaxed);
    }

    // Test-only: flip the flag without going through env vars. Not exposed
    // via Python bindings; production code must not call this.
    void setEmitLiveForTest(bool enabled) noexcept
    {
        mEmitLive.store(enabled, std::memory_order_relaxed);
    }

private:
    PerRequestActivityLog();

    struct RequestLog
    {
        std::array<Event, kMaxEventsPerRequest> events{};
        std::size_t writeIdx{0};
        bool wrapped{false};

        [[nodiscard]] std::chrono::steady_clock::time_point lastEventTime() const noexcept
        {
            std::size_t const last = (writeIdx == 0) ? (kMaxEventsPerRequest - 1) : (writeIdx - 1);
            return events[last].ts;
        }
    };

    /// Caller must hold mMutex.
    void evictOldestLocked();

    void recordImpl(LlmRequest::RequestIdType id, char const* event, char const* origin, char const* payloadLabel,
        std::int64_t payload, bool hasPayload, char const* detail) noexcept;

    mutable std::mutex mMutex;
    std::unordered_map<LlmRequest::RequestIdType, RequestLog> mLogs;
    std::atomic<bool> mEmitLive;
    // Process/executor rank shown on every emitted line; -1 until setRank().
    std::atomic<int> mRank{-1};
    // Cumulative count of entries dropped by the tracked-request cap.
    std::atomic<std::uint64_t> mEvictionCount{0};
    // One-shot guards so the eviction / record-failure notices warn at most
    // once per process instead of flooding. Not reset by clear().
    std::atomic<bool> mEvictionWarned{false};
    std::atomic<bool> mRecordFailWarned{false};
};

} // namespace tensorrt_llm::batch_manager
