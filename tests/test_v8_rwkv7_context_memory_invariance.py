"""RWKV-7 activation planning must not scale with the requested context."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "version" / "v8" / "scripts" / "build_ir_v8.py"
CIRCUIT = ROOT / "version" / "v8" / "circuits" / "rwkv7.json"

TRAINED_CONTEXT = 25600
HARD_CONTEXT_LIMIT = 1048576


def _load_builder():
    sys.path.insert(0, str(BUILDER.parent))
    spec = importlib.util.spec_from_file_location(
        "build_ir_v8_rwkv7_memory_invariance_test", BUILDER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {BUILDER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def build_ir():
    return _load_builder()


def _rwkv7_context(build_ir, context_length: int) -> tuple[dict, dict]:
    circuit = json.loads(CIRCUIT.read_text(encoding="utf-8"))
    config = build_ir._normalize_manifest_config(
        {
            "model": "rwkv7",
            "arch": "rwkv7",
            "embed_dim": 2048,
            "hidden_size": 2048,
            "num_layers": 24,
            "num_heads": 32,
            "head_dim": 64,
            "intermediate_size": 8192,
            "vocab_size": 65536,
            "activation": "squared_relu",
            "rope": "none",
            "layer_kinds": ["layer_0"] + ["layer_gt0"] * 23,
            "external_activation_slots": ["att_shift", "ffn_shift", "wkv", "v_first"],
            "require_declared_activation_slots": True,
            "rms_eps": 1e-5,
            "rope_theta": 10000.0,
            "rotary_dim": 64,
            "num_seqs": 1,
            "context_length": context_length,
            "max_seq_len": context_length,
            "prefill_chunk_length": circuit["contract"]["runtime_defaults"][
                "prefill_chunk_length"
            ],
        }
    )
    config["_template_uses_kv_cache"] = build_ir._template_uses_kv_cache(
        circuit, config
    )
    config["_template_uses_rope"] = build_ir._template_uses_rope(circuit, config)
    return circuit, config


def _activation_specs(build_ir, context_length: int) -> dict[str, dict]:
    circuit, config = _rwkv7_context(build_ir, context_length)
    return {
        mode: build_ir.build_activation_specs(
            config, mode, context_length, template=circuit
        )
        for mode in ("decode", "prefill")
    }


def test_rwkv7_activation_plan_is_context_invariant(build_ir) -> None:
    trained = _activation_specs(build_ir, TRAINED_CONTEXT)
    ceiling = _activation_specs(build_ir, HARD_CONTEXT_LIMIT)

    for mode in ("decode", "prefill"):
        trained_specs = trained[mode]
        ceiling_specs = ceiling[mode]
        assert set(trained_specs) == set(ceiling_specs)
        trained_total = sum(spec["size"] for spec in trained_specs.values())
        ceiling_total = sum(spec["size"] for spec in ceiling_specs.values())
        assert ceiling_total == trained_total
        for name in trained_specs:
            assert ceiling_specs[name]["size"] == trained_specs[name]["size"], name


def test_rwkv7_plan_has_no_token_indexed_caches(build_ir) -> None:
    for mode, specs in _activation_specs(build_ir, HARD_CONTEXT_LIMIT).items():
        assert "kv_cache" not in specs, mode
        assert "rope_cache" not in specs, mode
        # The only persistent sequence-lifetime buffer is the recurrent state.
        assert specs["wkv"]["size"] == 24 * 32 * 64 * 64 * 4


def test_rwkv7_prefill_stays_token_serial_at_the_ceiling(build_ir) -> None:
    circuit, config = _rwkv7_context(build_ir, HARD_CONTEXT_LIMIT)
    assert build_ir._resolve_prefill_extent(config, HARD_CONTEXT_LIMIT) == 1
    specs = build_ir.build_activation_specs(
        config, "prefill", HARD_CONTEXT_LIMIT, template=circuit
    )
    assert specs["embedded_input"]["shape"] == "[1, 2048]"
    assert specs["logits"]["shape"] == "[1, 65536]"
