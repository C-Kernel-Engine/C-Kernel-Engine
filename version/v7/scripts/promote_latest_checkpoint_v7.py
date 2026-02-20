#!/usr/bin/env python3
"""Promote latest CK train checkpoint to run_dir inference weights."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path


_STEP_RE = re.compile(r"^weights_step_(\d{8})\.bump$")


@dataclass(frozen=True)
class CheckpointPair:
    step: int
    bump: Path
    manifest: Path
    reason: str


def _load_reason(path: Path) -> str:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return "unknown"
    reason = doc.get("reason")
    return str(reason) if isinstance(reason, str) and reason else "unknown"


def _discover_pairs(run_dir: Path) -> list[CheckpointPair]:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        raise SystemExit(f"ERROR: checkpoint directory not found: {ckpt_dir}")

    pairs: list[CheckpointPair] = []
    for bump in ckpt_dir.glob("weights_step_*.bump"):
        m = _STEP_RE.match(bump.name)
        if not m:
            continue
        step = int(m.group(1))
        manifest = ckpt_dir / f"weights_step_{step:08d}_manifest.json"
        if not manifest.exists():
            continue
        pairs.append(
            CheckpointPair(
                step=step,
                bump=bump,
                manifest=manifest,
                reason=_load_reason(manifest),
            )
        )

    if not pairs:
        raise SystemExit(
            "ERROR: no checkpoint pairs found under "
            f"{ckpt_dir}\n"
            "Hint: run training with final checkpoint enabled (default) or set --train-save-every > 0."
        )

    pairs.sort(key=lambda p: p.step)
    return pairs


def _atomic_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=dst.name + ".", suffix=".tmp", dir=str(dst.parent))
    try:
        os.close(fd)
        Path(tmp).unlink(missing_ok=True)
        shutil.copy2(src, tmp)
        Path(tmp).replace(dst)
    finally:
        Path(tmp).unlink(missing_ok=True)


def _select_pair(pairs: list[CheckpointPair], step: int | None) -> CheckpointPair:
    if step is None:
        return pairs[-1]
    for pair in pairs:
        if pair.step == int(step):
            return pair
    available = ", ".join(str(p.step) for p in pairs[-10:])
    raise SystemExit(
        f"ERROR: checkpoint step {step} not found.\n"
        f"  available_steps_tail: {available}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Promote CK train checkpoint to run_dir inference weights")
    ap.add_argument("--run", required=True, help="Run dir containing checkpoints/")
    ap.add_argument("--step", type=int, default=None, help="Optional exact step to promote (default: latest step)")
    ap.add_argument("--dry-run", action="store_true", help="Print selected checkpoint without modifying run_dir")
    args = ap.parse_args()

    run_dir = Path(args.run).expanduser().resolve()
    pairs = _discover_pairs(run_dir)
    pair = _select_pair(pairs, args.step)

    dst_bump = run_dir / "weights.bump"
    dst_manifest = run_dir / "weights_manifest.json"

    print(f"[INFO] run_dir={run_dir}")
    print(f"[INFO] selected_step={pair.step} reason={pair.reason}")
    print(f"[INFO] source_bump={pair.bump}")
    print(f"[INFO] source_manifest={pair.manifest}")
    print(f"[INFO] target_bump={dst_bump}")
    print(f"[INFO] target_manifest={dst_manifest}")

    if args.dry_run:
        print("[OK] dry-run only; no files changed")
        return 0

    _atomic_copy(pair.bump, dst_bump)
    _atomic_copy(pair.manifest, dst_manifest)
    print("[OK] promotion complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
