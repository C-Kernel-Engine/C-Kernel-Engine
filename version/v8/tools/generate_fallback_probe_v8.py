#!/usr/bin/env python3
"""
Generate a minimal probe_report.json from tokenizer vocab.

This is a *backup* script — the real probe_report.json comes from the
evaluation pipeline (eval_stage_v7.py). Use this only when a run has
trained weights + tokenizer but no probe report, so that export_attention.py
--probe can find sequences to run a forward pass on.

Does NOT modify: export_attention_v8.py, prepare_run_viewer_v8.py, or any core
pipeline script. It just creates a file that the existing --probe flag reads.

Usage:
    python3 version/v8/tools/generate_fallback_probe_v8.py <run_dir>
    python3 version/v8/tools/generate_fallback_probe_v8.py <run_dir> --num-probes 4
    python3 version/v8/tools/generate_fallback_probe_v8.py <run_dir> --force

If probe_report.json already exists, skips unless --force is given.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_probes_from_tokenizer(tok_path: Path, num_probes: int = 6) -> list[dict]:
    """Build representative probe sequences from the tokenizer vocab."""
    tok = json.loads(tok_path.read_text(encoding="utf-8"))
    vocab = tok.get("vocab", tok.get("model", {}).get("vocab", []))

    if isinstance(vocab, dict):
        tokens = list(vocab.keys())
    else:
        tokens = [v if isinstance(v, str) else v[0] for v in vocab]

    # Categorise tokens
    system = {"<|unk|>", "<|bos|>", "<|eos|>", "<|pad|>"}
    control = [t for t in tokens if t not in system and t.startswith("[") and t.endswith("]")]
    output_marker = "[OUT]" if "[OUT]" in tokens else None

    # Group control tokens by role
    task_tokens = [t for t in control if t.startswith("[task:")]
    style_tokens = [t for t in control if any(
        t.startswith(f"[{p}:") for p in ("accent", "bg", "frame", "density", "theme", "layout")
    )]
    content_tokens = [t for t in control if any(
        t.startswith(f"[{p}:") for p in ("topic", "source", "shape", "color", "size")
    )]

    probes = []
    for i in range(num_probes):
        parts = []
        # Always start with a task token if available
        if task_tokens:
            parts.append(task_tokens[0])
        # Rotate through content/style tokens
        if content_tokens:
            parts.append(content_tokens[i % len(content_tokens)])
        if style_tokens and len(style_tokens) > i:
            parts.append(style_tokens[i % len(style_tokens)])
        # Add a couple more control tokens for variety
        remaining = [t for t in control if t not in parts and t != output_marker]
        for extra in remaining[i * 2: i * 2 + 2]:
            parts.append(extra)

        if not parts:
            # No control tokens — use raw vocab tokens
            raw = [t for t in tokens if t not in system and not t.startswith("[")]
            parts = raw[i * 3: i * 3 + 5] if raw else ["hello"]

        prompt = " ".join(parts)
        probes.append({
            "prompt": prompt,
            "expected": "",
            "label": f"fallback_probe_{i}",
        })

    return probes


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate a minimal probe_report.json from tokenizer vocab",
    )
    ap.add_argument("run_dir", type=Path, help="Training run directory")
    ap.add_argument("--num-probes", type=int, default=6, help="Number of probe sequences")
    ap.add_argument("--force", action="store_true", help="Overwrite existing probe_report.json")
    args = ap.parse_args()

    probe_path = args.run_dir / "probe_report.json"
    if probe_path.exists() and not args.force:
        print(f"  ⏭ probe_report.json already exists (use --force to overwrite)")
        return 0

    tok_path = args.run_dir / "tokenizer.json"
    if not tok_path.exists():
        print(f"  ✗ tokenizer.json not found in {args.run_dir}")
        return 1

    probes = build_probes_from_tokenizer(tok_path, args.num_probes)
    if not probes:
        print(f"  ✗ Could not build probes from tokenizer vocab")
        return 1

    report = {
        "source": "generate_fallback_probe_v8.py",
        "description": "Fallback probe report generated from tokenizer vocab for attention export",
        "results": probes,
    }

    probe_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"  ✓ Generated {len(probes)} probes → {probe_path}")
    for p in probes[:3]:
        print(f"    {p['label']}: {p['prompt'][:60]}")
    if len(probes) > 3:
        print(f"    ... +{len(probes) - 3} more")

    return 0


if __name__ == "__main__":
    sys.exit(main())
