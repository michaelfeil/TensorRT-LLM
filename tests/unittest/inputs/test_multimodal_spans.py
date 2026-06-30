# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
import types
from pathlib import Path
from typing import NamedTuple, Optional, Sequence


MULTIMODAL_PATH = (Path(__file__).parents[3] / "tensorrt_llm" / "inputs" /
                   "multimodal.py")


def _load_helper_module():
    tree = ast.parse(MULTIMODAL_PATH.read_text())
    names = {
        "MultimodalSpanTokenCounts",
        "_overlap_size",
        "count_multimodal_span_tokens",
        "find_multimodal_span_containing_boundary",
    }
    nodes = [
        node for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        and node.name in names
    ]
    path = MULTIMODAL_PATH.with_suffix(".span_test.py")
    module = types.ModuleType("multimodal_span_helpers")
    module.__dict__.update({
        "NamedTuple": NamedTuple,
        "Optional": Optional,
        "Sequence": Sequence,
    })
    exec(compile(ast.Module(nodes, type_ignores=[]), str(path), "exec"),
         module.__dict__)
    return module


def test_counts_multimodal_span_tokens_without_prompt_sized_mask() -> None:
    helper = _load_helper_module()

    counts = helper.count_multimodal_span_tokens(
        multimodal_positions=[10, 30],
        multimodal_lengths=[6, 4],
        begin=12,
        end=32,
    )

    assert counts.num_cached_mm_tokens == 2
    assert counts.num_mm_tokens_in_chunk == 6
    assert counts.total_embeds_in_request == 10


def test_finds_only_boundaries_inside_multimodal_spans() -> None:
    helper = _load_helper_module()

    assert helper.find_multimodal_span_containing_boundary(
        [10, 30], [6, 4], 12) == (10, 16)
    assert helper.find_multimodal_span_containing_boundary(
        [10, 30], [6, 4], 10) is None
    assert helper.find_multimodal_span_containing_boundary(
        [10, 30], [6, 4], 16) is None
