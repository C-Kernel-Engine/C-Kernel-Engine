#!/usr/bin/env python3
"""
Resolve a default train manifest path for v7 train-IR targets.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List


def _candidates_from_glob(base: Path) -> List[Path]:
    if not base.exists():
        return []
    paths = []
    for p in sorted(base.glob("*/weights_manifest.json")):
        if p.is_file():
            paths.append(p)
    return paths


def _rank(path: Path) -> int:
    name = str(path).lower()
    score = 0
    if "qwen3" in name:
        score += 30
    if "qwen2" in name:
        score += 20
    if "gemma" in name:
        score += 10
    # Prefer v7 cache over v6.6 if both exist.
    if "ck-engine-v7" in name:
        score += 5
    return score


def resolve(explicit_manifest: str, model_dir: str) -> Path:
    if explicit_manifest:
        p = Path(explicit_manifest).expanduser().resolve()
        if p.exists() and p.is_file():
            return p
        raise FileNotFoundError("manifest not found: %s" % explicit_manifest)

    if model_dir:
        md = Path(model_dir).expanduser().resolve()
        p = md / "weights_manifest.json"
        if p.exists() and p.is_file():
            return p
        raise FileNotFoundError("weights_manifest.json not found under model dir: %s" % md)

    env_manifest = os.getenv("V7_TRAIN_MANIFEST", "").strip()
    if env_manifest:
        p = Path(env_manifest).expanduser().resolve()
        if p.exists() and p.is_file():
            return p

    home = Path.home()
    pools: List[Path] = []
    pools.extend(_candidates_from_glob(home / ".cache" / "ck-engine-v7" / "models"))
    pools.extend(_candidates_from_glob(home / ".cache" / "ck-engine-v6.6" / "models"))

    if not pools:
        raise FileNotFoundError("no weights_manifest.json found in ~/.cache/ck-engine-v7/models or ~/.cache/ck-engine-v6.6/models")

    pools.sort(key=_rank, reverse=True)
    return pools[0]


def main() -> int:
    ap = argparse.ArgumentParser(description="Resolve v7 train manifest path.")
    ap.add_argument("--manifest", default="", help="Explicit manifest path")
    ap.add_argument("--model-dir", default="", help="Model directory containing weights_manifest.json")
    args = ap.parse_args()

    try:
        p = resolve(args.manifest, args.model_dir)
    except Exception as e:
        print("ERROR:", e, file=sys.stderr)
        return 1

    print(str(p))
    return 0


if __name__ == "__main__":
    sys.exit(main())
