#!/usr/bin/env python3
"""
Compare v6.5 and v6.6 prefill activations step by step.
"""

import subprocess
import numpy as np
import struct
import os
import sys

# Model paths
V65_DIR = os.path.expanduser("~/.cache/ck-engine-v6.5/models/Qwen--Qwen2-0.5B-Instruct-GGUF")
V66_DIR = os.path.expanduser("~/.cache/ck-engine-v6.6/models/Qwen--Qwen2-0.5B-Instruct-GGUF")

def run_v65_prefill(prompt="Hello!", stop_op=None):
    """Run v6.5 inference and return logits."""
    cmd = [
        "python3", "version/v6.5/scripts/ck_run_v6_5.py", "run",
        "Qwen/Qwen2-0.5B-Instruct-GGUF",
        "--weight-dtype=q4_k_m",
        "--context-len=100",
        f"--prompt={prompt}",
        "--max-tokens=1"
    ]
    env = os.environ.copy()
    if stop_op is not None:
        env["CK_STOP_OP"] = str(stop_op)

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return result.stdout, result.stderr

def run_v66_prefill(prompt="Hello!", stop_op=None):
    """Run v6.6 inference and return logits."""
    cmd = [
        "python3", "version/v6.6/scripts/ck_run_v6_6.py", "run",
        "Qwen/Qwen2-0.5B-Instruct-GGUF",
        "--weight-dtype=q4_k_m",
        "--context-len=100",
        f"--prompt={prompt}",
        "--max-tokens=1"
    ]
    env = os.environ.copy()
    if stop_op is not None:
        env["CK_STOP_OP"] = str(stop_op)

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return result.stdout, result.stderr

def compare_outputs():
    """Compare v6.5 and v6.6 outputs."""
    print("=" * 70)
    print("COMPARING V6.5 vs V6.6 PREFILL OUTPUT")
    print("=" * 70)

    # Run both
    print("\nRunning v6.5...")
    v65_out, v65_err = run_v65_prefill()

    print("\nRunning v6.6...")
    v66_out, v66_err = run_v66_prefill()

    print("\n" + "=" * 70)
    print("V6.5 OUTPUT:")
    print("=" * 70)
    # Extract just the response line
    for line in v65_out.split('\n'):
        if 'Response:' in line or 'Hello' in line:
            print(line)

    print("\n" + "=" * 70)
    print("V6.6 OUTPUT:")
    print("=" * 70)
    for line in v66_out.split('\n'):
        if 'Response:' in line or 'Hello' in line:
            print(line)

if __name__ == "__main__":
    compare_outputs()
