#!/usr/bin/env python3
"""
Test that --debug flag generates debug code in the output.
"""
import os
import sys
import json
import tempfile
from pathlib import Path

# Add scripts/v4 + scripts/v3 to path (v4 codegen, v3 layout helpers)
ROOT = Path(__file__).parent.parent
SCRIPTS_V4 = ROOT / "scripts" / "v4"
SCRIPTS_V3 = ROOT / "scripts" / "v3"
for path in (SCRIPTS_V4, SCRIPTS_V3):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import codegen_v4
import model_layout_v3 as v3

def create_minimal_config():
    """Create minimal model config for Q4_K testing."""
    return {
        "hidden_size": 256,
        "intermediate_size": 512,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "vocab_size": 1000,
        "max_position_embeddings": 512,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "tie_word_embeddings": False,
        "model_type": "qwen2",
    }

def create_minimal_layout(config):
    """Create a minimal ModelLayout for testing."""
    layout = v3.ModelLayout()
    layout.config = config
    layout.sections = []
    layout.model_name = "TestModel"
    layout.total_bytes = 1000000

    # Create minimal section with Q4_K weights
    section = v3.ModelSection(name="main", weight_path="test.bin", base_offset=0)

    # Add globals
    for name in ["token_emb", "embedded_input", "rope_cos_cache", "rope_sin_cache",
                 "ln_gamma", "lm_head", "logits"]:
        tensor = v3.ModelTensor(
            name=name,
            shape=[256, 256] if "weight" in name or name in ["token_emb", "lm_head"] else [256],
            dtype="q4_k" if name in ["token_emb", "lm_head"] else "fp32",
            offset=0,
            byte_size=1024,
            quant_type="q4_k" if name in ["token_emb", "lm_head"] else None,
        )
        section.tensors.append(tensor)

    # Add layers
    layer_tensors = ["ln1_gamma", "ln2_gamma", "wq", "wk", "wv", "wo", "w1", "w2",
                    "ln1_out", "ln2_out", "k", "v", "proj_tmp", "residual1",
                    "fc1_out", "swiglu_out", "mlp_out", "output"]
    for layer_idx in range(2):
        for name in layer_tensors:
            full_name = f"layer_{layer_idx}_{name}"
            is_weight = name in ["wq", "wk", "wv", "wo", "w1", "w2"]
            tensor = v3.ModelTensor(
                name=full_name,
                shape=[256, 256],
                dtype="q4_k" if is_weight else "fp32",
                offset=0,
                byte_size=1024,
                quant_type="q4_k" if is_weight else None,
            )
            section.tensors.append(tensor)

    layout.sections.append(section)
    return layout

def test_debug_codegen():
    """Test that emit_debug=True generates debug code."""
    config = create_minimal_config()
    layout = create_minimal_layout(config)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "generated_test.c")
        header_name = "generated_test.h"

        # Generate with debug=True
        codegen_v4.emit_c_source_v4(
            layout,
            output_path,
            header_name,
            mode="decode",
            emit_main=False,
            emit_debug=True,
            emit_parity=False,
        )

        # Read generated code
        with open(output_path, "r") as f:
            code = f.read()

        # Check for debug functions
        checks = [
            ("debug_check_buffer function", "static void debug_check_buffer("),
            ("debug_check_q4k_weights function", "static void debug_check_q4k_weights("),
            ("layer0_input debug call", '"layer0_input"'),
            ("layer0_wq debug call", '"layer0_wq"'),
            ("layer0_output debug call", '"layer0_output"'),
        ]

        print("Checking generated code for debug instrumentation...")
        all_passed = True
        for name, pattern in checks:
            if pattern in code:
                print(f"  OK: {name}")
            else:
                print(f"  FAIL: {name} not found!")
                all_passed = False

        if all_passed:
            print("\nAll debug checks passed!")

            # Print the relevant section of generated code
            print("\n--- Debug section in generated code ---")
            for line in code.split('\n'):
                if 'layer0_' in line or 'debug_check' in line:
                    print(line)

            return True
        else:
            print("\nSome debug checks failed!")
            print("\n--- Full generated code ---")
            print(code[:5000])  # Print first 5000 chars
            return False

if __name__ == "__main__":
    try:
        success = test_debug_codegen()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
