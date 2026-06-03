/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gtest/gtest.h>

#include "tensorrt_llm/common/logger.h"

#include <cstdint>
#include <regex>
#include <sstream>
#include <streambuf>
#include <string>

using tensorrt_llm::common::Logger;

namespace
{

// Captures stdout/stderr by swapping their rdbufs into local stringstreams for
// the lifetime of the object. RAII ensures the originals are always restored
// even if an assertion failure unwinds the stack mid-test.
class StreamCapture
{
public:
    StreamCapture()
        : mOutBuf(std::cout.rdbuf(mOut.rdbuf()))
        , mErrBuf(std::cerr.rdbuf(mErr.rdbuf()))
    {
    }

    ~StreamCapture()
    {
        std::cout.rdbuf(mOutBuf);
        std::cerr.rdbuf(mErrBuf);
    }

    std::string out() const
    {
        return mOut.str();
    }

    std::string err() const
    {
        return mErr.str();
    }

private:
    std::stringstream mOut;
    std::stringstream mErr;
    std::streambuf* mOutBuf;
    std::streambuf* mErrBuf;
};

// Logger is a process-wide singleton, so tests must explicitly set the level
// they need and restore it. Without RAII restoration, ordering between tests
// becomes load-bearing (e.g. an ERROR-only test would silence later DEBUG/INFO
// tests). Always wrap the StreamCapture inside a ScopedLogLevel.
class ScopedLogLevel
{
public:
    explicit ScopedLogLevel(Logger::Level level)
        : mPrevious(Logger::getLogger()->getLevel())
    {
        Logger::getLogger()->setLevel(level);
    }

    ~ScopedLogLevel()
    {
        Logger::getLogger()->setLevel(mPrevious);
    }

private:
    Logger::Level mPrevious;
};

// Smoke test: INFO-level macro emits to stdout with the [request_id=<id>]
// prefix and forwards trailing printf args through to the message body.
TEST(LoggerMacrosTest, ReqInfoPrefixesRequestId)
{
    ScopedLogLevel scopedLevel(Logger::INFO);
    StreamCapture capture;

    std::uint64_t const requestId = 42;
    TLLM_LOG_REQ_INFO(requestId, "hello %s", "world");

    auto const out = capture.out();
    EXPECT_NE(out.find("[INFO]"), std::string::npos) << out;
    EXPECT_NE(out.find("[request_id=42]"), std::string::npos) << out;
    EXPECT_NE(out.find("hello world"), std::string::npos) << out;
}

// Sink routing: WARNING is severity-routed to stderr (Logger::log() picks
// cerr for level >= WARNING). Verifies stdout stays empty so future kubectl
// log filters by stream still work after the prefix change.
TEST(LoggerMacrosTest, ReqWarningGoesToStderrWithPrefix)
{
    ScopedLogLevel scopedLevel(Logger::WARNING);
    StreamCapture capture;

    std::int64_t const requestId = 9001;
    TLLM_LOG_REQ_WARNING(requestId, "slow path");

    auto const err = capture.err();
    EXPECT_NE(err.find("[WARNING]"), std::string::npos) << err;
    EXPECT_NE(err.find("[request_id=9001]"), std::string::npos) << err;
    EXPECT_NE(err.find("slow path"), std::string::npos) << err;
    EXPECT_TRUE(capture.out().empty()) << capture.out();
}

// printf arg forwarding: the prefix injects a leading %zu so the remaining
// caller-supplied args must shift by one. This test catches mismatches between
// the macro's static_cast<std::size_t>(requestId) and any subsequent args.
TEST(LoggerMacrosTest, ReqErrorWithMultipleArgs)
{
    ScopedLogLevel scopedLevel(Logger::ERROR);
    StreamCapture capture;

    std::size_t const requestId = 7;
    TLLM_LOG_REQ_ERROR(requestId, "context_request_id=%zu reason=%s", static_cast<std::size_t>(123), "timeout");

    auto const err = capture.err();
    EXPECT_NE(err.find("[request_id=7] context_request_id=123 reason=timeout"), std::string::npos) << err;
}

// Level filter: the underlying TLLM_LOG do-while gates on isEnabled(level),
// so a DEBUG call under an INFO-level logger must produce zero side effects.
// Guards against accidentally evaluating arguments (e.g. costly to_string).
TEST(LoggerMacrosTest, ReqDebugRespectsLogLevel)
{
    ScopedLogLevel scopedLevel(Logger::INFO);
    StreamCapture capture;

    TLLM_LOG_REQ_DEBUG(42, "should-not-appear");

    EXPECT_TRUE(capture.out().empty()) << capture.out();
    EXPECT_TRUE(capture.err().empty()) << capture.err();
}

// Prefix uniqueness: catches accidental nesting (e.g. someone wrapping a
// TLLM_LOG_REQ_* call site with an outer TLLM_LOG that already concatenates
// "[request_id=...]"). One log statement must emit the prefix exactly once.
TEST(LoggerMacrosTest, ReqInfoPrefixIsExactlyOnce)
{
    ScopedLogLevel scopedLevel(Logger::INFO);
    StreamCapture capture;

    TLLM_LOG_REQ_INFO(11, "line");

    auto const out = capture.out();
    std::regex re("\\[request_id=11\\]");
    auto begin = std::sregex_iterator(out.begin(), out.end(), re);
    auto end = std::sregex_iterator();
    EXPECT_EQ(std::distance(begin, end), 1) << out;
}

} // namespace
