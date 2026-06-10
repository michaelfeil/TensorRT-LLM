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

#include "tensorrt_llm/common/logger.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>

namespace tensorrt_llm::batch_manager
{

namespace
{
// Defaults baked in for clarity. "0/false/off" disables; anything else
// (including unset for default=true, or any non-disable string for
// default=false) flips toward the documented default.
bool readEnvBool(char const* name, bool defaultValue) noexcept
{
    char const* v = std::getenv(name);
    if (v == nullptr)
    {
        return defaultValue;
    }
    if (std::strcmp(v, "0") == 0 || std::strcmp(v, "false") == 0 || std::strcmp(v, "off") == 0)
    {
        return false;
    }
    if (std::strcmp(v, "1") == 0 || std::strcmp(v, "true") == 0 || std::strcmp(v, "on") == 0)
    {
        return true;
    }
    // Unknown values: keep default to avoid surprising flips on typos.
    return defaultValue;
}

// Derive the PD-disagg side from the event origin: 'P' for prefill
// (CacheSender::*), 'D' for decode (CacheReceiver::*), '?' otherwise. Per-event
// (not a process global) because a single process constructs both a CacheSender
// and a CacheReceiver — the role is determined by which one actually recorded
// the event, i.e. its origin.
char sideFromOrigin(char const* origin) noexcept
{
    if (origin == nullptr)
    {
        return '?';
    }
    // Send/prefill path: CacheSender::* plus the cache formatters' format()
    // (CacheFormatter / MlaCacheFormatter / RnnCacheFormatter), which is only
    // ever invoked on the sending side. Match the "Formatter::format" suffix
    // rather than enumerating each formatter class.
    if (std::strncmp(origin, "CacheSender", 11) == 0 || std::strstr(origin, "Formatter::format") != nullptr)
    {
        return 'P';
    }
    if (std::strncmp(origin, "CacheReceiver", 13) == 0)
    {
        return 'D';
    }
    return '?';
}
} // namespace

PerRequestActivityLog::PerRequestActivityLog()
    : mEmitLive(readEnvBool("TRTLLM_ACTLOG_EMIT_LIVE", /*defaultValue=*/false))
{
}

PerRequestActivityLog& PerRequestActivityLog::instance() noexcept
{
    static PerRequestActivityLog inst;
    return inst;
}

void PerRequestActivityLog::record(LlmRequest::RequestIdType id, char const* event, char const* origin) noexcept
{
    recordImpl(id, event, origin, "payload", 0, /*hasPayload=*/false, /*detail=*/nullptr);
}

void PerRequestActivityLog::record(
    LlmRequest::RequestIdType id, char const* event, char const* origin, std::int64_t payload) noexcept
{
    recordImpl(id, event, origin, "payload", payload, /*hasPayload=*/true, /*detail=*/nullptr);
}

void PerRequestActivityLog::record(LlmRequest::RequestIdType id, char const* event, char const* origin,
    char const* payloadLabel, std::int64_t payload) noexcept
{
    recordImpl(id, event, origin, payloadLabel, payload, /*hasPayload=*/true, /*detail=*/nullptr);
}

void PerRequestActivityLog::recordDetail(
    LlmRequest::RequestIdType id, char const* event, char const* origin, char const* detail) noexcept
{
    recordImpl(id, event, origin, "payload", 0, /*hasPayload=*/false, detail);
}

void PerRequestActivityLog::recordImpl(LlmRequest::RequestIdType id, char const* event, char const* origin,
    char const* payloadLabel, std::int64_t payload, bool hasPayload, char const* detail) noexcept
{
    try
    {
        std::scoped_lock lk(mMutex);
        if (mLogs.find(id) == mLogs.end() && mLogs.size() >= kMaxTrackedRequests)
        {
            evictOldestLocked();
        }
        auto& log = mLogs[id];
        auto& ev = log.events[log.writeIdx];
        ev = Event{std::chrono::steady_clock::now(), event, origin, payloadLabel, payload, hasPayload, {}};
        if (detail != nullptr)
        {
            // Copy + truncate into the fixed inline buffer (snprintf always
            // NUL-terminates). Empty buffer means "no detail".
            std::snprintf(ev.detail, sizeof(ev.detail), "%s", detail);
        }
        log.writeIdx = (log.writeIdx + 1) % kMaxEventsPerRequest;
        if (log.writeIdx == 0)
        {
            log.wrapped = true;
        }
    }
    catch (...)
    {
        // Recording must not throw; diagnostics path is best-effort. Warn once
        // (fixed string, no allocation/formatting in case this is OOM) so a
        // silently broken activity log is still discoverable.
        if (!mRecordFailWarned.exchange(true, std::memory_order_relaxed))
        {
            TLLM_LOG_WARNING("PerRequestActivityLog: recording failed; suppressing further warnings");
        }
    }

    // Live emission: send a single KV_TX_STATUS line to stderr via the
    // request-scoped logger so the event stream shows up in kubectl logs /
    // Loki without needing a dump on throw. Outside the lock so contention
    // on mMutex is not extended by the formatting + logger call.
    if (mEmitLive.load(std::memory_order_relaxed))
    {
        int const rank = mRank.load(std::memory_order_relaxed);
        char const side = sideFromOrigin(origin);
        char const* safeEvent = (event != nullptr) ? event : "<null>";
        char const* safeOrigin = (origin != nullptr) ? origin : "<null>";
        bool const hasDetail = (detail != nullptr && detail[0] != '\0');
        if (hasPayload)
        {
            char const* safeLabel = (payloadLabel != nullptr) ? payloadLabel : "payload";
            TLLM_LOG_REQ_INFO(id, "%s side=%c rank=%d %s %s %s=%ld", kLinePrefix, side, rank, safeEvent, safeOrigin,
                safeLabel, payload);
        }
        else if (hasDetail)
        {
            TLLM_LOG_REQ_INFO(
                id, "%s side=%c rank=%d %s %s %s", kLinePrefix, side, rank, safeEvent, safeOrigin, detail);
        }
        else
        {
            TLLM_LOG_REQ_INFO(id, "%s side=%c rank=%d %s %s", kLinePrefix, side, rank, safeEvent, safeOrigin);
        }
    }
}

void PerRequestActivityLog::evictOldestLocked()
{
    if (mLogs.empty())
    {
        return;
    }
    auto oldestIt = mLogs.begin();
    auto oldestTime = oldestIt->second.lastEventTime();
    for (auto it = std::next(mLogs.begin()); it != mLogs.end(); ++it)
    {
        auto const candidate = it->second.lastEventTime();
        if (candidate < oldestTime)
        {
            oldestTime = candidate;
            oldestIt = it;
        }
    }
    mLogs.erase(oldestIt);
    mEvictionCount.fetch_add(1, std::memory_order_relaxed);
    // Warn once: hitting the cap means activity logs are now being dropped, so
    // dumps for evicted requests will be empty. Per-eviction logging would
    // flood (eviction is the steady-state reclamation path), so notify once.
    if (!mEvictionWarned.exchange(true, std::memory_order_relaxed))
    {
        TLLM_LOG_WARNING(
            "PerRequestActivityLog: tracked-request cap (%zu) reached; oldest activity logs are now being "
            "evicted (diagnostics may be lossy). Suppressing further notices; see evictionCount().",
            kMaxTrackedRequests);
    }
}

std::string PerRequestActivityLog::dump(LlmRequest::RequestIdType id) const
{
    std::scoped_lock lk(mMutex);
    auto it = mLogs.find(id);
    if (it == mLogs.end())
    {
        return {};
    }
    auto const& log = it->second;
    std::size_t const total = log.wrapped ? kMaxEventsPerRequest : log.writeIdx;
    if (total == 0)
    {
        return {};
    }

    // Read events in chronological order. With wrap-around the oldest event
    // sits at writeIdx; without wrap-around the oldest sits at 0.
    std::size_t const start = log.wrapped ? log.writeIdx : 0U;
    auto const firstTs = log.events[start].ts;

    // Every line is prefixed with `kLinePrefix side=<c> rank=<r> request=<id>` so
    // the full activity log for a request can be grepped/filtered line-by-line
    // (e.g. in Loki) even when interleaved with output from other ranks. The
    // side is derived from the event origin; within one process the events for a
    // request all share the same role, so the first event's side applies to all.
    char const side = sideFromOrigin(log.events[start].origin);
    std::string const linePrefix = std::string(kLinePrefix) + " side=" + std::string(1, side)
        + " rank=" + std::to_string(mRank.load(std::memory_order_relaxed)) + " request=" + std::to_string(id);

    std::ostringstream oss;
    oss << linePrefix << " activity log (" << total << "/" << kMaxEventsPerRequest << (log.wrapped ? ", wrapped" : "")
        << "):";
    for (std::size_t i = 0; i < total; ++i)
    {
        auto const idx = (start + i) % kMaxEventsPerRequest;
        auto const& e = log.events[idx];
        auto const deltaMs = std::chrono::duration_cast<std::chrono::milliseconds>(e.ts - firstTs).count();
        oss << "\n"
            << linePrefix << " +" << deltaMs << "ms " << (e.event ? e.event : "<null>") << " ("
            << (e.origin ? e.origin : "<null>") << ")";
        if (e.hasPayload)
        {
            oss << " " << (e.payloadLabel ? e.payloadLabel : "payload") << "=" << e.payload;
        }
        if (e.detail[0] != '\0')
        {
            oss << " " << e.detail;
        }
    }
    return oss.str();
}

void PerRequestActivityLog::release(LlmRequest::RequestIdType id) noexcept
{
    try
    {
        std::scoped_lock lk(mMutex);
        mLogs.erase(id);
    }
    catch (...)
    {
        // Diagnostics path; never throw.
    }
}

void PerRequestActivityLog::clear() noexcept
{
    try
    {
        std::scoped_lock lk(mMutex);
        mLogs.clear();
        mEvictionCount.store(0, std::memory_order_relaxed);
    }
    catch (...)
    {
    }
}

std::size_t PerRequestActivityLog::trackedRequestCount() const noexcept
{
    std::scoped_lock lk(mMutex);
    return mLogs.size();
}

} // namespace tensorrt_llm::batch_manager
