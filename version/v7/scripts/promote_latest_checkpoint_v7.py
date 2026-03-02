#!/usr/bin/env python3
"""Promote CK train checkpoints into run_dir inference weights.

Supports three selector modes:
1) Default/legacy: latest checkpoint step (or --step N)
2) Run-id mode:    --run-id ascii_bpe_YYYYmmdd_HHMMSS
3) Stage mode:     --stage sft [--stage-pass N]

Stage/run selectors use training_pipeline_latest.json stage_loss_history and
run_ledger.jsonl when available, and prefer per-run snapshots under:
  .ck_pipeline/<run_id>/weights_final.bump
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_STEP_RE = re.compile(r"^weights_step_(\d{8})\.bump$")


@dataclass(frozen=True)
class CheckpointPair:
    step: int
    bump: Path
    manifest: Path
    reason: str


@dataclass(frozen=True)
class RunCheckpointRef:
    run_id: str
    stage: str
    stage_pass: int
    steps: int | None
    run_order: int
    ended_at: str | None
    dataset_name: str | None
    final_loss: float | None
    checkpoint_bump: str | None
    checkpoint_manifest: str | None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_stage(raw: Any) -> str:
    s = str(raw or "").strip().lower()
    aliases = {
        "stage_a": "pretrain",
        "pretrain": "pretrain",
        "stage_b": "midtrain",
        "midtrain": "midtrain",
        "sft": "sft",
        "dpo": "dpo",
        "grpo": "grpo",
        "ppo": "ppo",
        "rlhf": "ppo",
    }
    return aliases.get(s, s or "pretrain")


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


def _select_pair_by_step(pairs: list[CheckpointPair], step: int | None) -> CheckpointPair:
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


def _read_ledger_index(run_dir: Path) -> dict[str, dict[str, Any]]:
    ledger_path = run_dir / "run_ledger.jsonl"
    if not ledger_path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if isinstance(rec, dict) and rec.get("run_id"):
            out[str(rec["run_id"])] = rec
    return out


def _read_stage_history_entries(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "training_pipeline_latest.json"
    if not path.exists():
        return []
    try:
        doc = _load_json(path)
    except Exception:
        return []
    hist = doc.get("stage_loss_history")
    if not isinstance(hist, dict) and isinstance(doc.get("pipeline"), dict):
        hist = doc["pipeline"].get("stage_loss_history")
    entries = hist.get("entries") if isinstance(hist, dict) else None
    if not isinstance(entries, list):
        return []
    return [row for row in entries if isinstance(row, dict)]


def _build_run_refs(run_dir: Path) -> list[RunCheckpointRef]:
    rows = _read_stage_history_entries(run_dir)
    if not rows:
        return []
    ledger_idx = _read_ledger_index(run_dir)

    refs: list[RunCheckpointRef] = []
    for i, row in enumerate(rows):
        run_id = str(row.get("run_id") or "").strip()
        if not run_id:
            continue
        led = ledger_idx.get(run_id, {})

        stage = _normalize_stage(led.get("stage_id") or row.get("stage"))
        run_order_raw = led.get("run_order", row.get("run_order"))
        try:
            run_order = int(run_order_raw)
        except Exception:
            run_order = i

        dataset_name = row.get("dataset_name")
        if not isinstance(dataset_name, str) or not dataset_name.strip():
            dataset_name = led.get("dataset_name")
        if isinstance(dataset_name, str):
            dataset_name = dataset_name.strip() or None
        else:
            dataset_name = None

        steps_raw = led.get("steps", row.get("steps"))
        try:
            steps = int(steps_raw)
        except Exception:
            steps = None

        stage_pass_raw = led.get("stage_pass", row.get("stage_pass"))
        try:
            stage_pass = int(stage_pass_raw)
        except Exception:
            stage_pass = 0

        ended_at = row.get("ended_at")
        if not isinstance(ended_at, str) or not ended_at.strip():
            ended_at = led.get("ended_at")
        ended_at = str(ended_at).strip() if isinstance(ended_at, str) and ended_at else None

        final_loss_raw = row.get("final_loss", led.get("loss_final"))
        try:
            final_loss = float(final_loss_raw)
        except Exception:
            final_loss = None

        ck_bump = led.get("checkpoint_bump")
        ck_manifest = led.get("checkpoint_manifest")
        if not isinstance(ck_bump, str) or not ck_bump.strip():
            local_bump = run_dir / ".ck_pipeline" / run_id / "weights_final.bump"
            if local_bump.exists():
                ck_bump = str(local_bump)
        if not isinstance(ck_manifest, str) or not ck_manifest.strip():
            local_manifest = run_dir / ".ck_pipeline" / run_id / "weights_final_manifest.json"
            if local_manifest.exists():
                ck_manifest = str(local_manifest)

        refs.append(
            RunCheckpointRef(
                run_id=run_id,
                stage=stage,
                stage_pass=stage_pass,
                steps=steps,
                run_order=run_order,
                ended_at=ended_at,
                dataset_name=dataset_name,
                final_loss=final_loss,
                checkpoint_bump=str(ck_bump).strip() if isinstance(ck_bump, str) and ck_bump else None,
                checkpoint_manifest=str(ck_manifest).strip() if isinstance(ck_manifest, str) and ck_manifest else None,
            )
        )

    refs.sort(key=lambda r: r.run_order)
    # Infer stage pass when missing.
    stage_counts: dict[str, int] = {}
    rebuilt: list[RunCheckpointRef] = []
    for ref in refs:
        inferred = ref.stage_pass
        if inferred <= 0:
            stage_counts[ref.stage] = stage_counts.get(ref.stage, 0) + 1
            inferred = stage_counts[ref.stage]
        else:
            stage_counts[ref.stage] = max(stage_counts.get(ref.stage, 0), inferred)
        rebuilt.append(
            RunCheckpointRef(
                run_id=ref.run_id,
                stage=ref.stage,
                stage_pass=inferred,
                steps=ref.steps,
                run_order=ref.run_order,
                ended_at=ref.ended_at,
                dataset_name=ref.dataset_name,
                final_loss=ref.final_loss,
                checkpoint_bump=ref.checkpoint_bump,
                checkpoint_manifest=ref.checkpoint_manifest,
            )
        )
    return rebuilt


def _pick_ref(refs: list[RunCheckpointRef], run_id: str | None, stage: str | None, stage_pass: int | None) -> RunCheckpointRef:
    if run_id:
        for ref in refs:
            if ref.run_id == run_id:
                return ref
        raise SystemExit(f"ERROR: run_id not found in stage history: {run_id}")

    stage_norm = _normalize_stage(stage)
    candidates = [r for r in refs if r.stage == stage_norm]
    if not candidates:
        raise SystemExit(f"ERROR: no runs found for stage '{stage_norm}' in training_pipeline_latest.json")
    if stage_pass is not None:
        for ref in candidates:
            if ref.stage_pass == int(stage_pass):
                return ref
        available = ", ".join(str(r.stage_pass) for r in candidates[-8:])
        raise SystemExit(
            f"ERROR: stage-pass {stage_pass} not found for stage '{stage_norm}'.\n"
            f"  available_passes_tail: {available}"
        )
    return candidates[-1]


def _pair_from_ref(run_dir: Path, pairs: list[CheckpointPair], ref: RunCheckpointRef) -> CheckpointPair:
    if ref.checkpoint_bump and ref.checkpoint_manifest:
        bump = Path(ref.checkpoint_bump).expanduser().resolve()
        manifest = Path(ref.checkpoint_manifest).expanduser().resolve()
        if bump.exists() and manifest.exists():
            is_global_step_path = (
                bump.parent == (run_dir / "checkpoints").resolve()
                and bool(_STEP_RE.match(bump.name))
            )
            if is_global_step_path:
                reason = (
                    f"fallback_step_only: stage={ref.stage} pass={ref.stage_pass} "
                    f"run_id={ref.run_id} step={ref.steps}"
                )
            else:
                reason = f"stage={ref.stage} pass={ref.stage_pass} run_id={ref.run_id}"
            return CheckpointPair(
                step=int(ref.steps or -1),
                bump=bump,
                manifest=manifest,
                reason=reason,
            )
    if ref.steps is None:
        raise SystemExit(
            "ERROR: selected run has no step count and no per-run checkpoint snapshot.\n"
            f"  run_id: {ref.run_id}"
        )
    base = _select_pair_by_step(pairs, int(ref.steps))
    return CheckpointPair(
        step=base.step,
        bump=base.bump,
        manifest=base.manifest,
        reason=(
            f"fallback_step_only: stage={ref.stage} pass={ref.stage_pass} "
            f"run_id={ref.run_id} step={ref.steps} source_reason={base.reason}"
        ),
    )


def _print_run_list(refs: list[RunCheckpointRef]) -> None:
    if not refs:
        print("No stage_loss_history entries found.")
        return
    print("run_order\trun_id\tstage\tstage_pass\tsteps\tfinal_loss\tdataset")
    for r in refs:
        loss_txt = "-" if r.final_loss is None else f"{r.final_loss:.6f}"
        steps_txt = "-" if r.steps is None else str(r.steps)
        ds_txt = r.dataset_name or "-"
        print(f"{r.run_order}\t{r.run_id}\t{r.stage}\t{r.stage_pass}\t{steps_txt}\t{loss_txt}\t{ds_txt}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Promote CK train checkpoint to run_dir inference weights")
    ap.add_argument("--run", required=True, help="Run dir containing checkpoints/")
    ap.add_argument("--step", type=int, default=None, help="Optional exact checkpoint step to promote")
    ap.add_argument("--run-id", default=None, help="Promote by stage-history run_id")
    ap.add_argument("--stage", default=None, help="Promote latest pass for stage (pretrain/midtrain/sft/dpo/grpo/ppo)")
    ap.add_argument("--stage-pass", type=int, default=None, help="Promote exact stage pass number when using --stage")
    ap.add_argument("--list-runs", action="store_true", help="List promotable runs from stage_loss_history and exit")
    ap.add_argument("--dry-run", action="store_true", help="Print selected checkpoint without modifying run_dir")
    args = ap.parse_args()

    run_dir = Path(args.run).expanduser().resolve()
    if args.step is not None and (args.run_id or args.stage):
        raise SystemExit("ERROR: --step cannot be combined with --run-id/--stage selectors.")
    if args.stage_pass is not None and not args.stage:
        raise SystemExit("ERROR: --stage-pass requires --stage.")

    refs = _build_run_refs(run_dir)
    if args.list_runs:
        _print_run_list(refs)
        return 0

    pairs = _discover_pairs(run_dir)
    selected_ref: RunCheckpointRef | None = None
    if args.run_id or args.stage:
        selected_ref = _pick_ref(refs, args.run_id, args.stage, args.stage_pass)
        pair = _pair_from_ref(run_dir, pairs, selected_ref)
    else:
        pair = _select_pair_by_step(pairs, args.step)

    dst_bump = run_dir / "weights.bump"
    dst_manifest = run_dir / "weights_manifest.json"

    print(f"[INFO] run_dir={run_dir}")
    if selected_ref is not None:
        print(
            "[INFO] selector="
            f"run_id={selected_ref.run_id} stage={selected_ref.stage} "
            f"stage_pass={selected_ref.stage_pass} steps={selected_ref.steps}"
        )
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
