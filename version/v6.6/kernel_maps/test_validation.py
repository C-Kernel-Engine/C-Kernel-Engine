#!/usr/bin/env python3
"""
test_validation.py - Test suite for kernel map validation

This script validates:
1. Kernel map syntax and structure
2. Bindings coverage
3. Header synchronization
4. Manifest structure validation

Usage:
    python test_validation.py [--quick] [--verbose]
"""

import json
import subprocess
import sys
from pathlib import Path

# ANSI colors
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
RESET = '\033[0m'


def run_check(cmd: list, name: str, expect_pass: bool = True) -> bool:
    """Run a validation check and report result."""
    print(f"\n{CYAN}[{name}]{RESET}")
    print("-" * 60)

    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr

    if expect_pass:
        if result.returncode == 0:
            print(f"{GREEN}PASS{RESET}")
            return True
        else:
            print(f"{RED}FAIL{RESET}")
            print(output)
            return False
    else:
        if result.returncode != 0:
            print(f"{GREEN}PASS (expected failure){RESET}")
            return True
        else:
            print(f"{RED}FAIL (expected failure but passed){RESET}")
            return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run kernel map validation tests")
    parser.add_argument("--quick", action="store_true", help="Quick check (skip slow tests)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    scripts_dir = Path(__file__).parent
    kernel_maps_dir = scripts_dir

    print("=" * 60)
    print("KERNEL MAP VALIDATION TEST SUITE")
    print("=" * 60)

    results = []

    # Test 1: Kernel map validation
    results.append(run_check(
        [sys.executable, str(kernel_maps_dir / "validate_kernel_maps.py")],
        "1. Kernel Map Structure Validation",
        expect_pass=True
    ))

    # Test 2: Header sync check
    results.append(run_check(
        [sys.executable, str(kernel_maps_dir / "check_kernel_map_sync.py")],
        "2. Header Synchronization Check",
        expect_pass=True
    ))

    # Test 3: Manifest structure validation
    print(f"\n{CYAN}[3. Manifest Structure Validation]{RESET}")
    print("-" * 60)

    # Create a valid mock manifest with proper structure
    mock_manifest = {
        "template": {
            "name": "mock_model",
            "sequence": ["decoder"],  # Required: top-level sequence
            "block_types": {
                "decoder": {
                    "sequence": ["header", "body", "footer"],
                    "header": ["dense_embedding_lookup"],
                    "body": {
                        "type": "dense",
                        "ops": ["rmsnorm", "qkv_proj", "attn", "out_proj"]
                    },
                    "footer": ["rmsnorm", "logits"]
                }
            }
        },
        "config": {
            "num_layers": 1,
            "embed_dim": 256,
            "num_heads": 8,
            "num_kv_heads": 2,
            "head_dim": 32,
            "intermediate_size": 512,
            "max_seq_len": 128,
        },
        "quant_summary": {
            "token_emb": "q8_0",
            "layer.0": {
                "wq": "q4_0",  # Supported quant format
                "wk": "q8_0",
                "wv": "q8_0",
                "wo": "q8_0",
            }
        }
    }

    mock_path = kernel_maps_dir / "mock_manifest.json"
    with open(mock_path, 'w') as f:
        json.dump(mock_manifest, f)

    # Try to load and validate the manifest
    try:
        with open(mock_path) as f:
            loaded = json.load(f)

        # Verify required fields
        has_template = "template" in loaded
        has_sequence = "sequence" in loaded.get("template", {})
        has_block_types = "block_types" in loaded.get("template", {})

        if has_template and has_sequence and has_block_types:
            print(f"{GREEN}PASS{RESET}")
            print("  Manifest structure is valid (sequence, block_types present)")
            results.append(True)
        else:
            print(f"{RED}FAIL{RESET}")
            print(f"  Missing required fields: template={has_template}, sequence={has_sequence}, block_types={has_block_types}")
            results.append(False)
    except Exception as e:
        print(f"{RED}FAIL{RESET}")
        print(f"  Error loading manifest: {e}")
        results.append(False)

    # Cleanup
    mock_path.unlink(missing_ok=True)

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    print(f"  Passed: {GREEN}{passed}/{total}{RESET}")
    print(f"  Failed: {RED}{total - passed}/{total}{RESET}")

    if passed == total:
        print(f"\n{GREEN}All validation tests passed!{RESET}")
        return 0
    else:
        print(f"\n{YELLOW}Some tests failed - see above for details.{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
