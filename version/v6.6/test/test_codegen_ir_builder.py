#!/usr/bin/env python3
"""
test_codegen_ir_builder.py - Tests for IR builder and codegen

These tests catch common issues:
1. Wrong weight binding (logits should use token_emb, not final_ln_weight)
2. Wrong kernel selection (FP32 vs quantized activations)
3. Function declaration order issues
4. Missing weights in IR ops
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

# Add scripts to path
SCRIPT_DIR = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

# Test data paths
CACHE_DIR = Path.home() / ".cache/ck-engine-v6.6/models/Qwen--Qwen2-0.5B-Instruct-GGUF"
MANIFEST_PATH = CACHE_DIR / "weights_manifest.json"


def test_logits_has_token_emb():
    """Logits op must use token_emb weight, not final_ln_weight or other weights."""
    # Generate fresh IR1
    result = subprocess.run([
        sys.executable, str(SCRIPT_DIR / "build_ir_v6_6.py"),
        "--manifest", str(MANIFEST_PATH),
        "--mode", "decode",
        "--output", "/tmp/test_ir1.json",
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"FAIL: IR builder failed: {result.stderr}")
        return False

    with open("/tmp/test_ir1.json") as f:
        ir1 = json.load(f)

    # Find logits op
    logits_op = None
    for op in ir1.get("ops", []):
        if op.get("op") == "logits":
            logits_op = op
            break

    if not logits_op:
        print("FAIL: No logits op found in IR1")
        return False

    weights = logits_op.get("weights", {})

    # Must have token_emb
    if "token_emb" not in weights:
        print(f"FAIL: logits op missing token_emb weight. Has: {list(weights.keys())}")
        return False

    # Must NOT have final_ln_weight (that belongs to final_rmsnorm)
    if "final_ln_weight" in weights:
        print("FAIL: logits op has final_ln_weight (should only be in final_rmsnorm)")
        return False

    print("PASS: logits op correctly uses token_emb")
    return True


def test_final_rmsnorm_has_correct_weights():
    """Footer rmsnorm op must use final_ln_weight and final_ln_bias."""
    result = subprocess.run([
        sys.executable, str(SCRIPT_DIR / "build_ir_v6_6.py"),
        "--manifest", str(MANIFEST_PATH),
        "--mode", "decode",
        "--output", "/tmp/test_ir1.json",
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"FAIL: IR builder failed: {result.stderr}")
        return False

    with open("/tmp/test_ir1.json") as f:
        ir1 = json.load(f)

    # Find footer rmsnorm op
    footer_rmsnorm = None
    for op in ir1.get("ops", []):
        if op.get("op") == "rmsnorm" and op.get("section") == "footer":
            footer_rmsnorm = op
            break

    if not footer_rmsnorm:
        print("FAIL: No footer rmsnorm op found in IR1")
        return False

    weights = footer_rmsnorm.get("weights", {})

    if "final_ln_weight" not in weights:
        print(f"FAIL: footer rmsnorm missing final_ln_weight. Has: {list(weights.keys())}")
        return False

    print("PASS: footer rmsnorm correctly uses final_ln_weight")
    return True


def test_gemv_kernels_use_fp32_activations():
    """All gemv kernels should use FP32 activation variants (not q8_0 input)."""
    result = subprocess.run([
        sys.executable, str(SCRIPT_DIR / "build_ir_v6_6.py"),
        "--manifest", str(MANIFEST_PATH),
        "--mode", "decode",
        "--output", "/tmp/test_ir1.json",
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"FAIL: IR builder failed: {result.stderr}")
        return False

    with open("/tmp/test_ir1.json") as f:
        ir1 = json.load(f)

    # Check all gemv kernels
    bad_kernels = []
    for op in ir1.get("ops", []):
        kernel = op.get("kernel", "")
        if kernel.startswith("gemv_") and "_q8_0_q8_0" in kernel:
            # q8_0_q8_0 means quantized input - we want FP32 activation
            bad_kernels.append((op.get("op"), kernel))

    if bad_kernels:
        print(f"FAIL: Found {len(bad_kernels)} gemv ops using Q8_0 activation input:")
        for op, kernel in bad_kernels[:5]:
            print(f"  - {op}: {kernel}")
        return False

    print("PASS: All gemv kernels use FP32 activation input")
    return True


def test_codegen_compiles():
    """Generated C code should compile without errors."""
    # Generate IR and lowered IR
    result = subprocess.run([
        sys.executable, str(SCRIPT_DIR / "build_ir_v6_6.py"),
        "--manifest", str(MANIFEST_PATH),
        "--mode", "decode",
        "--output", "/tmp/test_ir1.json",
        "--layout-output", "/tmp/test_layout.json",
        "--lowered-output", "/tmp/test_lowered.json",
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"FAIL: IR builder failed: {result.stderr}")
        return False

    # Generate C code
    result = subprocess.run([
        sys.executable, str(SCRIPT_DIR / "codegen_v6_6.py"),
        "--ir", "/tmp/test_lowered.json",
        "--layout", "/tmp/test_layout.json",
        "-o", "/tmp/test_codegen.c",
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"FAIL: Codegen failed: {result.stderr}")
        return False

    # Compile
    project_root = Path(__file__).parent.parent.parent.parent
    result = subprocess.run([
        "gcc", "-O0", "-c", "/tmp/test_codegen.c", "-o", "/tmp/test_codegen.o",
        f"-I{project_root}/include",
        f"-I{project_root}/src",
        f"-I{project_root}/src/kernels",
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"FAIL: Compilation failed:\n{result.stderr}")
        return False

    print("PASS: Generated code compiles without errors")
    return True


def test_codegen_logits_uses_token_emb_weight():
    """Codegen should emit token_emb for logits, not wq or other weights."""
    # Generate lowered IR
    result = subprocess.run([
        sys.executable, str(SCRIPT_DIR / "build_ir_v6_6.py"),
        "--manifest", str(MANIFEST_PATH),
        "--mode", "decode",
        "--lowered-output", "/tmp/test_lowered.json",
        "--layout-output", "/tmp/test_layout.json",
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"FAIL: IR builder failed: {result.stderr}")
        return False

    # Generate C code
    result = subprocess.run([
        sys.executable, str(SCRIPT_DIR / "codegen_v6_6.py"),
        "--ir", "/tmp/test_lowered.json",
        "--layout", "/tmp/test_layout.json",
        "-o", "/tmp/test_codegen.c",
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"FAIL: Codegen failed: {result.stderr}")
        return False

    # Check generated code for logits op
    with open("/tmp/test_codegen.c") as f:
        code = f.read()

    # Find the logits op section
    if "Op 338" not in code or "logits" not in code:
        # Find whatever op number logits is
        lines = code.split('\n')
        logits_line = None
        for i, line in enumerate(lines):
            if "(logits)" in line and "Op" in line:
                logits_line = i
                break

        if logits_line is None:
            print("FAIL: Could not find logits op in generated code")
            return False

    # Check that logits uses token_emb
    if "L_HEADER.token_emb" not in code:
        print("FAIL: Generated code does not use L_HEADER.token_emb for logits")
        return False

    # Check it doesn't use L->wq for logits (common bug)
    # Find logits section and check nearby lines
    lines = code.split('\n')
    for i, line in enumerate(lines):
        if "(logits)" in line and "Op" in line:
            # Check next 10 lines for the weight reference
            section = '\n'.join(lines[i:i+10])
            if "L->wq" in section:
                print("FAIL: Logits op incorrectly uses L->wq instead of token_emb")
                return False
            if "L_HEADER.token_emb" in section:
                print("PASS: Logits op correctly uses L_HEADER.token_emb")
                return True

    print("PASS: Logits op uses token_emb")
    return True


def test_all_weights_bound():
    """All weight ops should have weights bound (no empty weights dict)."""
    result = subprocess.run([
        sys.executable, str(SCRIPT_DIR / "build_ir_v6_6.py"),
        "--manifest", str(MANIFEST_PATH),
        "--mode", "decode",
        "--lowered-output", "/tmp/test_lowered.json",
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"FAIL: IR builder failed: {result.stderr}")
        return False

    with open("/tmp/test_lowered.json") as f:
        lowered = json.load(f)

    # Ops that should have weights
    ops_requiring_weights = {
        "dense_embedding_lookup": ["token_emb"],
        "q_proj": ["wq"],
        "k_proj": ["wk"],
        "v_proj": ["wv"],
        "out_proj": ["wo"],
        "mlp_gate_up": ["w1"],
        "mlp_down": ["w2"],
        "logits": ["token_emb"],
    }

    missing_weights = []
    for op in lowered.get("ops", lowered.get("operations", [])):
        op_name = op.get("op", "")
        if op_name in ops_requiring_weights:
            weights = op.get("weights", {})
            required = ops_requiring_weights[op_name]
            for w in required:
                if w not in weights:
                    missing_weights.append((op.get("idx"), op_name, w))

    if missing_weights:
        print(f"FAIL: Found {len(missing_weights)} ops with missing weights:")
        for idx, op, w in missing_weights[:5]:
            print(f"  - Op {idx} ({op}): missing {w}")
        return False

    print("PASS: All weight-requiring ops have weights bound")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("IR Builder and Codegen Tests")
    print("=" * 60)

    if not MANIFEST_PATH.exists():
        print(f"SKIP: Test model not found at {MANIFEST_PATH}")
        print("Run: python3 scripts/convert_gguf_to_bump_v6_6.py to download a test model")
        return 1

    tests = [
        test_logits_has_token_emb,
        test_final_rmsnorm_has_correct_weights,
        test_gemv_kernels_use_fp32_activations,
        test_all_weights_bound,
        test_codegen_compiles,
        test_codegen_logits_uses_token_emb_weight,
    ]

    passed = 0
    failed = 0

    for test in tests:
        print(f"\n[TEST] {test.__name__}")
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"FAIL: Exception: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
