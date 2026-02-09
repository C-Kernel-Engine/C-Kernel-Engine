#!/usr/bin/env python3
"""
Quick performance profile without sudo.
Parses CK_PROFILE CSV output to generate kernel timing reports.
"""
import subprocess
import sys
import json
import re
import os
from pathlib import Path


def run_profile(model: str, tokens: int = 10) -> dict:
    """Run CK_PROFILE=1 and parse output."""
    env = os.environ.copy()
    env["CK_PROFILE"] = "1"

    ck_run_path = Path(__file__).parent / "ck_run_v6_6.py"

    result = subprocess.run(
        [sys.executable, str(ck_run_path), "run", model,
         "--max-tokens", str(tokens), "--temperature", "0.0"],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(Path(__file__).parent)
    )

    # Parse CK_PROFILE CSV output from stderr
    # Format: mode,kernel,op,layer,time_us,token_id
    kernels = []
    total_time_us = 0.0

    # Combine stdout and stderr for parsing
    all_output = result.stderr + "\n" + result.stdout

    for line in all_output.split('\n'):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Parse CSV format: mode,kernel,op,layer,time_us,token_id
        parts = line.split(',')
        if len(parts) >= 6:
            try:
                mode = parts[0]
                kernel = parts[1]
                op = parts[2]
                layer = int(parts[3])
                time_us = float(parts[4])
                token_id = int(parts[5])

                # Aggregate by kernel name
                total_time_us += time_us

                # Find existing entry or create new
                existing = next((k for k in kernels if k["kernel"] == kernel), None)
                if existing:
                    existing["time_us"] += time_us
                    existing["count"] += 1
                else:
                    kernels.append({
                        "kernel": kernel,
                        "op": op,
                        "mode": mode,
                        "layer": layer,
                        "time_us": time_us,
                        "count": 1
                    })
            except (ValueError, IndexError):
                continue

    # Calculate percentages and convert to ms
    for k in kernels:
        k["time_ms"] = k["time_us"] / 1000.0
        k["percent"] = (k["time_us"] / total_time_us * 100) if total_time_us > 0 else 0

    return {
        "model": model,
        "tokens": tokens,
        "total_time_ms": total_time_us / 1000.0,
        "kernels": sorted(kernels, key=lambda x: -x["time_us"]),
        "profile_format": "csv",
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Quick performance profile without sudo"
    )
    parser.add_argument("--model", required=True, help="Model ID, path, or GGUF file")
    parser.add_argument("--tokens", type=int, default=10, help="Number of tokens to generate")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    args = parser.parse_args()

    print(f"Running profile on model: {args.model}")
    print(f"Tokens to generate: {args.tokens}")
    print("")

    profile = run_profile(args.model, args.tokens)

    with open(args.output, "w") as f:
        json.dump(profile, f, indent=2)

    print(f"\nProfile saved to {args.output}")
    print(f"Total time: {profile['total_time_ms']:.2f} ms")
    print(f"\nTop 10 kernels:")
    for k in sorted(profile["kernels"], key=lambda x: -x["time_ms"])[:10]:
        print(f"  {k['kernel']}: {k['time_ms']:.2f} ms ({k['percent']:.1f}%)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
