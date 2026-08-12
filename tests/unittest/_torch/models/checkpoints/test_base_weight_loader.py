# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from tensorrt_llm._torch.models.checkpoints.base_weight_loader import ConsumableWeightsDict

WEIGHT_KEYS = [
    "model.layers.0.mlp.gate.weight",
    "model.layers.0.mlp.gate.e_score_correction_bias",
    # Non-dot-boundary sibling of the "gate" prefix: startswith("...mlp.gate")
    # matches it, and prefix_items must reproduce that exactly.
    "model.layers.0.mlp.gateway.weight",
    "model.layers.0.mlp.experts.0.w1.weight",
    "model.layers.0.mlp.experts.0.w1.weight_scale",
    "model.layers.0.mlp.experts.10.w2.weight",
    "model.layers.0.mlp.shared_experts.gate_proj.weight",
    "model.layers.1.self_attn.kv_a_proj_with_mqa.weight",
    "model.embed_tokens.weight",
    "model.norm.weight",
    "lm_head.weight",
]


def _reference_filter(prefix: str, weights: dict) -> dict:
    return {k[len(prefix) + 1 :]: v for k, v in weights.items() if k.startswith(prefix)}


def _make() -> tuple[ConsumableWeightsDict, dict]:
    reference = {k: i for i, k in enumerate(WEIGHT_KEYS)}
    return ConsumableWeightsDict(dict(reference)), reference


@pytest.mark.parametrize(
    "prefix",
    [
        "model.layers.0.mlp.gate",
        "model.layers.0.mlp.gate.",
        "model.layers.0.mlp.experts.0",
        "model.layers.0",
        "model.embed_tokens",
        "model.norm",
        "does.not.exist",
        "",
    ],
)
def test_prefix_items_matches_startswith_semantics(prefix):
    weights, reference = _make()
    got = {k[len(prefix) + 1 :]: v for k, v in weights.prefix_items(prefix)}
    assert got == _reference_filter(prefix, reference)


def test_mark_consumed_deletes_dotted_prefix_only():
    weights, reference = _make()
    deleted = weights.mark_consumed("model.layers.0.mlp.gate")
    expected = [k for k in reference if k.startswith("model.layers.0.mlp.gate.")]
    assert deleted == len(expected)
    for key in expected:
        assert key not in weights
    # startswith-but-not-dotted sibling must survive.
    assert "model.layers.0.mlp.gateway.weight" in weights
    assert len(weights) == len(reference) - deleted


def test_index_stays_consistent_across_mutations():
    weights, reference = _make()
    # Build the index, then mutate through every code path.
    assert weights.prefix_items("model.layers.0")

    weights.mark_consumed("model.layers.0.mlp.experts.0.w1")
    for key in [k for k in reference if k.startswith("model.layers.0.mlp.experts.0.w1.")]:
        del reference[key]

    del weights["model.norm.weight"]
    del reference["model.norm.weight"]

    weights["model.layers.2.new.weight"] = 99
    reference["model.layers.2.new.weight"] = 99

    weights.update({"aa.head.weight": 1, "zz.tail.weight": 2})
    reference.update({"aa.head.weight": 1, "zz.tail.weight": 2})

    for prefix in ["model.layers.0", "model.layers.2", "model.norm", "aa", "zz", ""]:
        got = {k[len(prefix) + 1 :]: v for k, v in weights.prefix_items(prefix)}
        assert got == _reference_filter(prefix, reference), prefix
    assert len(weights) == len(reference)


def test_delete_then_reinsert_keeps_index_consistent():
    weights, _ = _make()
    weights.prefix_items("model.norm")  # build the index
    del weights["model.norm.weight"]
    weights["model.norm.weight"] = 0
    assert [k for k, _ in weights.prefix_items("model.norm")] == ["model.norm.weight"]


def test_filter_weights_uses_prefix_index():
    from tensorrt_llm._torch.models.modeling_utils import filter_weights

    weights, reference = _make()
    prefix = "model.layers.0.mlp.experts.0"
    assert filter_weights(prefix, weights) == _reference_filter(prefix, reference)
    # Plain dicts must keep the fallback scan path.
    assert filter_weights(prefix, reference) == _reference_filter(prefix, reference)
