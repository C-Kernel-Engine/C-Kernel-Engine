#!/usr/bin/env python3
"""
Debug v6 model output by comparing first token logits
"""

import sys
import json
import subprocess

def run_ck_v6(model_path, prompt):
    """Run C-Kernel-Engine v6 and capture output"""

    cmd = [
        "python", "scripts/v6/ck_run_v6.py", "run",
        f"hf://Qwen/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q4_k_m.gguf",
        "--prompt", prompt,
        "--max-tokens", "3"
    ]

    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

    print("STDOUT:")
    print(result.stdout)

    if result.stderr:
        print("STDERR:")
        print(result.stderr)

    return result.stdout, result.stderr

def run_llama_cpp(model_path, prompt):
    """Run llama.cpp for comparison"""

    # Check if llama.cpp exists
    llama_path = "/home/antshiv/Workspace/C-Kernel-Engine/llama.cpp/build/llama-cli"

    if not Path(llama_path).exists():
        print(f"\nllama.cpp not found at {llama_path}")
        return None, None

    cmd = [
        llama_path,
        "-m", model_path,
        "-p", prompt,
        "-n", "3",
        "--temp", "0.0",
        "-ngl", "0"  # No GPU
    ]

    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

    print("STDOUT:")
    print(result.stdout)

    if result.stderr:
        print("STDERR:")
        print(result.stderr)

    return result.stdout, result.stderr

def main():
    model_path = "/home/antshiv/.cache/ck-engine-v6/models/Qwen--Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q4_k_m.gguf"

    # Test prompts
    prompts = [
        "Hello",
        "The capital of France is",
        "2+2=",
        "Paris is the capital of"
    ]

    print("\n" + "="*60)
    print("C-KERNEL-ENGINE v6 OUTPUT TEST")
    print("="*60)

    for prompt in prompts:
        print(f"\n\n{'#'*60}")
        print(f"# Test Prompt: '{prompt}'")
        print(f"{'#'*60}")

        # Run CK-Engine
        ck_out, ck_err = run_ck_v6(model_path, prompt)

        # Run llama.cpp for comparison (commented out for now)
        # llama_out, llama_err = run_llama_cpp(model_path, prompt)

if __name__ == "__main__":
    main()
