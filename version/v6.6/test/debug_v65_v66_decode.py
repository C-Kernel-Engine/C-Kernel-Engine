#!/usr/bin/env python3
"""
Debug script to compare v6.5 and v6.6 decode step by step.
Run each version with CK_STOP_OP to stop after each op and compare outputs.
"""

import subprocess
import sys
import os

V65_DIR = os.path.expanduser("~/.cache/ck-engine-v6.5/models/Qwen--Qwen2-0.5B-Instruct-GGUF")
V66_DIR = os.path.expanduser("~/.cache/ck-engine-v6.6/models/Qwen--Qwen2-0.5B-Instruct-GGUF")

def run_v65(prompt="Hello!", max_tokens=2):
    """Run v6.5 inference."""
    cmd = [
        "python3", "version/v6.5/scripts/ck_run_v6_5.py", "run",
        "Qwen/Qwen2-0.5B-Instruct-GGUF",
        "--weight-dtype=q4_k_m",
        "--context-len=100",
        f"--prompt={prompt}",
        f"--max-tokens={max_tokens}"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    return result.stdout, result.stderr

def run_v66(prompt="Hello!", max_tokens=2):
    """Run v6.6 inference."""
    cmd = [
        "python3", "version/v6.6/scripts/ck_run_v6_6.py", "run",
        "Qwen/Qwen2-0.5B-Instruct-GGUF",
        "--weight-dtype=q4_k_m",
        "--context-len=100",
        f"--prompt={prompt}",
        f"--max-tokens={max_tokens}"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    return result.stdout, result.stderr

def extract_response(output):
    """Extract the response text from output."""
    for line in output.split('\n'):
        if 'Response:' in line:
            return line.split('Response:')[1].strip()
    return None

def main():
    print("=" * 70)
    print("COMPARING V6.5 vs V6.6 DECODE")
    print("=" * 70)

    # Run with max_tokens=5 to see multiple decode steps
    prompt = "Hello!"
    max_tokens = 5

    print(f"\nPrompt: {prompt}")
    print(f"Max tokens: {max_tokens}")

    print("\n" + "-" * 70)
    print("Running V6.5...")
    v65_out, v65_err = run_v65(prompt, max_tokens)
    v65_response = extract_response(v65_out)
    print(f"V6.5 Response: {v65_response}")

    print("\n" + "-" * 70)
    print("Running V6.6...")
    v66_out, v66_err = run_v66(prompt, max_tokens)
    v66_response = extract_response(v66_out)
    print(f"V6.6 Response: {v66_response}")

    print("\n" + "-" * 70)
    print("COMPARISON:")
    if v65_response == v66_response:
        print("SUCCESS: Responses match!")
    else:
        print("MISMATCH: Responses differ!")
        print(f"  V6.5: {repr(v65_response)}")
        print(f"  V6.6: {repr(v66_response)}")

        # Find first differing position
        if v65_response and v66_response:
            for i, (c1, c2) in enumerate(zip(v65_response, v66_response)):
                if c1 != c2:
                    print(f"  First difference at position {i}: '{c1}' vs '{c2}'")
                    break

if __name__ == "__main__":
    main()
