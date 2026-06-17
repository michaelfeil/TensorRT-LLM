/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "cacheTransceiver.h"
#include "tensorrt_llm/batch_manager/cacheTransceiver.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/rnnStateManager.h"
#include "tensorrt_llm/common/bindingUtils.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/nanobind/common/customCasters.h"
#include <ATen/ATen.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>
#include <nanobind/trampoline.h>
#include <torch/extension.h>
#include <typeinfo>

using SizeType32 = tensorrt_llm::runtime::SizeType32;

namespace tb = tensorrt_llm::batch_manager;
namespace nb = nanobind;

namespace
{

class PyCacheTransceiver : public tb::BaseCacheTransceiver
{
public:
    // using BaseCacheTransceiver::BaseCacheTransceiver; // Inherit constructors
    NB_TRAMPOLINE(tb::BaseCacheTransceiver, 8);

    void respondAndSendAsync(std::shared_ptr<tb::LlmRequest> llmRequest) override
    {
        NB_OVERRIDE_PURE(respondAndSendAsync, llmRequest);
    }

    void requestAndReceiveSync(std::shared_ptr<tb::LlmRequest> llmRequest) override
    {
        NB_OVERRIDE_PURE(requestAndReceiveSync, llmRequest);
    }

    void requestAndReceiveAsync(std::shared_ptr<tb::LlmRequest> llmRequest) override
    {
        NB_OVERRIDE_PURE(requestAndReceiveAsync, llmRequest);
    }

    tb::RequestStatuses checkContextTransferStatus(
        std::optional<int> const& atLeastRequestNum = std::nullopt, bool markComplete = false,
        bool collectKvTransferEvents = false,
        std::vector<tb::LlmRequest::RequestIdType> const& timedOutContextRequestIds = {}) override
    {
        NB_OVERRIDE_PURE(checkContextTransferStatus, atLeastRequestNum, markComplete, collectKvTransferEvents,
            timedOutContextRequestIds);
    }

    tb::RequestStatuses checkGenTransferStatus(
        std::optional<int> const& atLeastRequestNum = std::nullopt, bool collectKvTransferEvents = false) override
    {
        NB_OVERRIDE_PURE(checkGenTransferStatus, atLeastRequestNum, collectKvTransferEvents);
    }

    bool checkGenTransferComplete() const override
    {
        NB_OVERRIDE_PURE(checkGenTransferComplete);
    }

    bool hasPendingGenTransfer(std::shared_ptr<tb::LlmRequest> llmRequest) const override
    {
        NB_OVERRIDE_PURE(hasPendingGenTransfer, llmRequest);
    }

    bool cancelRequest(std::shared_ptr<tb::LlmRequest> llmRequest) override
    {
        NB_OVERRIDE_PURE(cancelRequest, llmRequest);
    }
};

nb::list kvTransferEventRecordsToList(std::vector<tb::KvTransferEventRecord> const& eventRecords)
{
    nb::list events;
    for (auto const& event : eventRecords)
    {
        events.append(nb::make_tuple(event.rank, static_cast<std::uint64_t>(event.requestId)));
    }
    return events;
}
} // namespace

void tb::CacheTransceiverBindings::initBindings(nb::module_& m)
{
    nb::class_<tb::BaseCacheTransceiver, PyCacheTransceiver>(m, "BaseCacheTransceiver")
        .def("respond_and_send_async", &BaseCacheTransceiver::respondAndSendAsync)
        .def("request_and_receive_sync", &BaseCacheTransceiver::requestAndReceiveSync)
        .def("request_and_receive_async", &BaseCacheTransceiver::requestAndReceiveAsync)
        .def(
            "check_context_transfer_status",
            [](tb::BaseCacheTransceiver& self, std::optional<int> const& atLeastRequestNum, bool markComplete = false,
                bool collectKvTransferEvents = false,
                std::vector<tb::LlmRequest::RequestIdType> const& timedOutContextRequestIds = {})
            {
                RequestStatuses result;
                {
                    nb::gil_scoped_release release;
                    result = self.checkContextTransferStatus(
                        atLeastRequestNum, markComplete, collectKvTransferEvents, timedOutContextRequestIds);
                }

                auto completedRequestIds
                    = std::vector<int64_t>(result.completedRequestIds.begin(), result.completedRequestIds.end());
                auto errorRequestIds
                    = std::vector<int64_t>(result.errorRequestIds.begin(), result.errorRequestIds.end());
                auto completedKvTransferEvents = kvTransferEventRecordsToList(result.completedKvTransferEvents);
                auto errorKvTransferEvents = kvTransferEventRecordsToList(result.errorKvTransferEvents);
                return nb::make_tuple(
                    completedRequestIds, errorRequestIds, completedKvTransferEvents, errorKvTransferEvents);
            },
            nb::arg("at_least_request_num") = std::nullopt, nb::arg("mark_complete") = false,
            nb::arg("collect_kv_transfer_events") = false,
            nb::arg("timed_out_context_request_ids") = std::vector<tb::LlmRequest::RequestIdType>{})
        .def("take_context_kv_transfer_event_report", &BaseCacheTransceiver::takeContextKvTransferEventReport)
        .def(
            "check_gen_transfer_status",
            [](tb::BaseCacheTransceiver& self, std::optional<int> const& atLeastRequestNum,
                bool collectKvTransferEvents = false)
            {
                RequestStatuses result;
                {
                    nb::gil_scoped_release release;
                    result = self.checkGenTransferStatus(atLeastRequestNum, collectKvTransferEvents);
                }

                auto completedRequestIds
                    = std::vector<int64_t>(result.completedRequestIds.begin(), result.completedRequestIds.end());
                auto errorRequestIds
                    = std::vector<int64_t>(result.errorRequestIds.begin(), result.errorRequestIds.end());
                auto completedKvTransferEvents = kvTransferEventRecordsToList(result.completedKvTransferEvents);
                auto errorKvTransferEvents = kvTransferEventRecordsToList(result.errorKvTransferEvents);
                return nb::make_tuple(
                    completedRequestIds, errorRequestIds, completedKvTransferEvents, errorKvTransferEvents);
            },
            nb::arg("at_least_request_num") = std::nullopt, nb::arg("collect_kv_transfer_events") = false)
        .def("check_gen_transfer_complete", &BaseCacheTransceiver::checkGenTransferComplete)
        .def("has_pending_gen_transfer", &BaseCacheTransceiver::hasPendingGenTransfer)
        .def("record_context_kv_transfer_failure_event",
            &BaseCacheTransceiver::recordContextKvTransferFailureEvent)
        .def("record_generation_kv_transfer_failure_event",
            &BaseCacheTransceiver::recordGenerationKvTransferFailureEvent)
        .def("cancel_request", &BaseCacheTransceiver::cancelRequest);

    nb::enum_<executor::kv_cache::CacheState::AttentionType>(m, "AttentionType")
        .value("DEFAULT", executor::kv_cache::CacheState::AttentionType::kDEFAULT)
        .value("MLA", executor::kv_cache::CacheState::AttentionType::kMLA);

    nb::class_<tb::CacheTransceiver, tb::BaseCacheTransceiver>(m, "CacheTransceiver")
        .def(nb::init<tb::kv_cache_manager::BaseKVCacheManager*, std::vector<SizeType32>, SizeType32, SizeType32,
                 runtime::WorldConfig, std::vector<SizeType32>, nvinfer1::DataType,
                 executor::kv_cache::CacheState::AttentionType, std::optional<executor::CacheTransceiverConfig>,
                 tb::rnn_state_manager::RnnStateManager*, std::vector<SizeType32>>(),
            nb::arg("cache_manager"), nb::arg("num_kv_heads_per_layer"), nb::arg("size_per_head"),
            nb::arg("tokens_per_block"), nb::arg("world_config"), nb::arg("attention_layer_num_per_pp"),
            nb::arg("dtype"), nb::arg("attention_type"), nb::arg("cache_transceiver_config") = std::nullopt,
            nb::arg("rnn_state_manager") = nullptr, nb::arg("rnn_layer_num_per_pp") = std::vector<SizeType32>{});

    nb::class_<tb::CacheTransceiverComm>(m, "CacheTransceiverComm")
        .def(
            "__init__",
            [](tb::CacheTransceiverComm* self, nb::object pg_obj, std::string pybind11_abi)
            {
                new (self) tb::CacheTransceiverComm(
                    common::get_intrusive_ptr<c10d::ProcessGroup, nb::python_error>(pg_obj.ptr(), pybind11_abi));
            },
            nb::arg("process_group"), nb::arg("pybind11_abi"))
        .def("get_rank", &tb::CacheTransceiverComm::getRank)
        .def("get_size", &tb::CacheTransceiverComm::getSize)
        .def("split", &tb::CacheTransceiverComm::split, nb::arg("color"), nb::arg("key"))
        .def(
            "allgather",
            [](tb::CacheTransceiverComm const& self, int64_t input)
            {
                std::vector<int64_t> out(static_cast<size_t>(self.getSize()));
                c10d::AllgatherOptions options;
                bool ok = self.allgather(input, std::ref(out), options);
                return nb::make_tuple(ok, out);
            },
            nb::arg("input"))
        .def(
            "allgather",
            [](tb::CacheTransceiverComm const& self, double input)
            {
                std::vector<double> out(static_cast<size_t>(self.getSize()));
                c10d::AllgatherOptions options;
                bool ok = self.allgather(input, std::ref(out), options);
                return nb::make_tuple(ok, out);
            },
            nb::arg("input"))
        .def(
            "allgather",
            [](tb::CacheTransceiverComm const& self, char input)
            {
                std::vector<char> out(static_cast<size_t>(self.getSize()));
                c10d::AllgatherOptions options;
                bool ok = self.allgather(input, std::ref(out), options);
                return nb::make_tuple(ok, out);
            },
            nb::arg("input"))
        .def(
            "allgatherv",
            [](tb::CacheTransceiverComm const& self, std::vector<int64_t> input, std::vector<int> const& sizes)
            {
                int total_size = std::accumulate(sizes.begin(), sizes.end(), 0);
                std::vector<int64_t> output(total_size);
                bool ok = self.allgatherv(std::ref(input), std::ref(output), std::cref(sizes));
                return nb::make_tuple(ok, output);
            },
            nb::arg("input"), nb::arg("sizes"))
        .def(
            "allgatherv",
            [](tb::CacheTransceiverComm const& self, std::vector<double> input, std::vector<int> const& sizes)
            {
                int total_size = std::accumulate(sizes.begin(), sizes.end(), 0);
                std::vector<double> output(total_size);
                bool ok = self.allgatherv(std::ref(input), std::ref(output), std::cref(sizes));
                return nb::make_tuple(ok, output);
            },
            nb::arg("input"), nb::arg("sizes"))
        .def(
            "allgatherv",
            [](tb::CacheTransceiverComm const& self, std::vector<char> input, std::vector<int> const& sizes)
            {
                int total_size = std::accumulate(sizes.begin(), sizes.end(), 0);
                std::vector<char> output(total_size);
                bool ok = self.allgatherv(std::ref(input), std::ref(output), std::cref(sizes));
                return nb::make_tuple(ok, output);
            },
            nb::arg("input"), nb::arg("sizes"));

    nb::class_<tb::kv_cache_manager::CacheTransBufferManager>(m, "CacheTransBufferManager")
        .def(nb::init<tb::kv_cache_manager::BaseKVCacheManager*, std::optional<size_t>>(), nb::arg("cache_manager"),
            nb::arg("max_num_tokens") = std::nullopt)
        .def_static("pre_alloc_buffer_size", &tb::kv_cache_manager::CacheTransBufferManager::preAllocBufferSize,
            nb::arg("cache_size_bytes_per_token_per_window"), nb::arg("tokens_per_block"),
            nb::arg("cache_transceiver_config") = nb::none());
}
