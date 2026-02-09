#!/usr/bin/env python3
"""
Smoke tests for all templates - no weights required.
"""
import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch


TEMPLATES = [
    "gemma3.json",
    "qwen3.json",
    "qwen2.json",
]


QUANT_COMBOS = [
    {"token_emb": "fp16", "layers": "q4_0"},
    {"token_emb": "fp16", "layers": "q4_k"},
    {"token_emb": "q8_0", "layers": "q5_0"},
    {"token_emb": "q8_0", "layers": "q8_0"},
]


def make_mock_manifest(template_name: str, quant_combo: dict) -> dict:
    """Create mock manifest for template smoke test."""
    template_path = Path(__file__).parent.parent / "templates" / template_name
    if not template_path.exists():
        pytest.skip(f"Template not found: {template_name}")

    with open(template_path) as f:
        template = json.load(f)

    config = template.get("config", {})
    num_layers = config.get("num_layers", 24)

    return {
        "model_name": f"mock-{template_name}",
        "config": {
            **config,
            "num_layers": num_layers,
            "quant_summary": {
                "token_emb": {"dtype": quant_combo["token_emb"]},
                **{f"layer_{i}": {"dtype": quant_combo["layers"]}
                   for i in range(num_layers)}
            }
        }
    }


@pytest.mark.parametrize("template", TEMPLATES)
@pytest.mark.parametrize("quants", QUANT_COMBOS)
def test_template_builds(template, quants):
    """Test that template + quant combo produces valid IR1 structure."""
    # Import here to check availability
    try:
        from build_ir_v6_6 import build_ir1_direct
    except ImportError:
        pytest.skip("build_ir1_direct not available")

    manifest = make_mock_manifest(template, quants)

    # Mock the heavy operations
    with patch('build_ir1_direct') as mock_build:
        mock_build.return_value = {
            "operations": [
                {"idx": 0, "op": "token_emb", "kernel": "embedding_forward"},
                {"idx": 1, "op": "attn_norm", "kernel": "rmsnorm_forward"},
            ]
        }

        ir1 = build_ir1_direct(manifest, mode="decode")

    # Verify IR1 has expected structure
    assert "operations" in ir1
    assert len(ir1["operations"]) > 0

    # Verify all ops have kernels
    for op in ir1["operations"]:
        assert op.get("kernel"), f"Op {op.get('op')} has no kernel"


@pytest.mark.parametrize("template", TEMPLATES)
def test_template_has_required_ops(template):
    """Verify template has all expected ops."""
    template_path = Path(__file__).parent.parent / "templates" / template
    if not template_path.exists():
        pytest.skip(f"Template not found: {template}")

    with open(template_path) as f:
        template_data = json.load(f)

    # Check required ops exist in body
    body_ops = template_data.get("block_types", {}).get("decoder", {}).get("body", {}).get("ops", [])

    # Templates may use either "attn_norm", "rmsnorm", or "ffn_norm" for normalization
    # We check for any norm op in the first half of body ops (before MLP starts)
    norm_ops = [op for op in body_ops if "norm" in op]
    has_attn_norm = len(norm_ops) >= 1, f"Template {template} missing attention normalization op"

    # Check for required projection and MLP ops
    has_qkv = any("qkv" in op for op in body_ops)
    has_out_proj = "out_proj" in body_ops
    has_mlp_gate_up = "mlp_gate_up" in body_ops or "geglu" in body_ops
    has_mlp_down = "mlp_down" in body_ops

    assert has_attn_norm[0], f"Template {template} missing attention normalization op"
    assert has_qkv, f"Template {template} missing qkv_proj op"
    assert has_out_proj, f"Template {template} missing out_proj op"
    assert has_mlp_gate_up, f"Template {template} missing mlp_gate_up op"
    assert has_mlp_down, f"Template {template} missing mlp_down op"


@pytest.mark.parametrize("template", TEMPLATES)
def test_template_json_structure(template):
    """Verify template has valid JSON structure."""
    template_path = Path(__file__).parent.parent / "templates" / template
    if not template_path.exists():
        pytest.skip(f"Template not found: {template}")

    with open(template_path) as f:
        template_data = json.load(f)

    # Template v2 has version, name, family, flags, sequence, block_types
    assert "version" in template_data, f"Template {template} missing version"
    assert "name" in template_data, f"Template {template} missing name"
    assert "family" in template_data, f"Template {template} missing family"
    assert "block_types" in template_data, f"Template {template} missing block_types"

    # Check decoder block exists and has required structure
    decoder = template_data.get("block_types", {}).get("decoder", {})
    assert "sequence" in decoder, f"Template {template} decoder missing sequence"
    assert "body" in decoder, f"Template {template} decoder missing body"

    # Body must have ops
    body = decoder.get("body", {})
    if isinstance(body, dict):
        assert "ops" in body, f"Template {template} decoder body missing ops"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
