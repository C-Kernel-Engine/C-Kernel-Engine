#!/usr/bin/env python3
"""
audit_train_token_provenance_v7.py

Run-level provenance audit for v7 training passes.

Checks, per run_id:
- Human-readable dataset path (raw text file) provenance
- Token file path provenance (train_tokens.txt)
- Cross-artifact consistency: run_ledger vs pipeline_report vs train_token_pack vs train_ck
- File existence and SHA-256 fingerprints
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _read_ledger_latest(ledger_path: Path) -> list[dict[str, Any]]:
    if not ledger_path.exists():
        return []
    latest: dict[str, dict[str, Any]] = {}
    for raw in ledger_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if not isinstance(rec, dict):
            continue
        rid = str(rec.get("run_id") or "").strip()
        if not rid:
            continue
        latest[rid] = rec

    def _order(rec: dict[str, Any]) -> tuple[int, str]:
        try:
            ro = int(rec.get("run_order"))
        except Exception:
            ro = 10**9
        return ro, str(rec.get("run_id") or "")

    return sorted(latest.values(), key=_order)


def _same(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return False
    return str(a) == str(b)


def _canon_path(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return str(Path(s).expanduser().resolve())
    except Exception:
        return s


def _status_chip(ok: bool) -> str:
    return "OK" if ok else "MISMATCH"


def _short(path_like: Any, keep: int = 58) -> str:
    if path_like is None:
        return "-"
    s = str(path_like)
    if len(s) <= keep:
        return s
    return "..." + s[-(keep - 3) :]


def _preview(path: Path, max_lines: int) -> list[str]:
    if max_lines <= 0 or not path.exists() or not path.is_file():
        return []
    out: list[str] = []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if len(out) >= max_lines:
                    break
                row = line.rstrip("\n\r")
                if row.strip():
                    out.append(row[:220])
    except Exception:
        return []
    return out


def audit_run_dir(run_dir: Path, preview_lines: int = 0) -> dict[str, Any]:
    ledger_rows = _read_ledger_latest(run_dir / "run_ledger.jsonl")
    records: list[dict[str, Any]] = []

    for row in ledger_rows:
        run_id = str(row.get("run_id") or "")
        if not run_id:
            continue
        work_dir = Path(
            row.get("work_dir")
            or (run_dir / ".ck_pipeline" / run_id)
        )
        pack = _load_json(work_dir / "train_token_pack.json")
        pipe = _load_json(work_dir / "pipeline_report.json")
        ck = _load_json(work_dir / "train_ck.json")

        ledger_dataset = row.get("dataset")
        pack_dataset = pack.get("dataset")
        pipe_dataset = pipe.get("dataset")
        raw_dataset = pack_dataset or pipe_dataset or ledger_dataset
        raw_dataset_path = Path(raw_dataset).expanduser().resolve() if raw_dataset else None

        pack_token_file = pack.get("token_file")
        ck_token_file = None
        ds = ck.get("data_source")
        if isinstance(ds, dict):
            ck_token_file = ds.get("source_path")
        token_file = pack_token_file or ck_token_file
        token_file_path = Path(token_file).expanduser().resolve() if token_file else None

        pack_token_count = (
            int(pack.get("stats", {}).get("token_file_token_count"))
            if isinstance(pack.get("stats"), dict) and pack.get("stats", {}).get("token_file_token_count") is not None
            else None
        )
        ck_token_count = (
            int(ds.get("token_count"))
            if isinstance(ds, dict) and ds.get("token_count") is not None
            else None
        )

        raw_exists = bool(raw_dataset_path and raw_dataset_path.exists())
        tok_exists = bool(token_file_path and token_file_path.exists())
        raw_sha = _sha256_file(raw_dataset_path) if raw_exists and raw_dataset_path else None
        tok_sha = _sha256_file(token_file_path) if tok_exists and token_file_path else None

        ledger_dataset_c = _canon_path(ledger_dataset)
        pack_dataset_c = _canon_path(pack_dataset)
        pipe_dataset_c = _canon_path(pipe_dataset)
        dataset_paths_consistent = (
            (ledger_dataset_c == pack_dataset_c or pack_dataset_c is None or ledger_dataset_c is None)
            and (ledger_dataset_c == pipe_dataset_c or pipe_dataset_c is None or ledger_dataset_c is None)
        )
        token_paths_consistent = (
            _same(pack_token_file, ck_token_file) or pack_token_file is None or ck_token_file is None
        )
        token_counts_consistent = True
        if pack_token_count is not None and ck_token_count is not None:
            # sample packer writes one trailer token for x/y shift; allow exact match or +1 in pack report
            token_counts_consistent = bool(
                pack_token_count == ck_token_count or pack_token_count == (ck_token_count + 1)
            )

        checks = {
            "dataset_paths": dataset_paths_consistent,
            "token_paths": token_paths_consistent,
            "token_counts": token_counts_consistent,
            "raw_exists": raw_exists,
            "token_exists": tok_exists,
        }
        pass_all = all(checks.values())

        rec: dict[str, Any] = {
            "run_order": row.get("run_order"),
            "run_id": run_id,
            "stage": row.get("stage_id"),
            "stage_pass": row.get("stage_pass"),
            "status": row.get("status"),
            "work_dir": str(work_dir),
            "raw_dataset": str(raw_dataset_path) if raw_dataset_path else None,
            "token_file": str(token_file_path) if token_file_path else None,
            "ledger_dataset": ledger_dataset,
            "pack_dataset": pack_dataset,
            "pipe_dataset": pipe_dataset,
            "pack_token_file": pack_token_file,
            "ck_token_file": ck_token_file,
            "pack_token_count": pack_token_count,
            "ck_token_count": ck_token_count,
            "raw_dataset_sha256": raw_sha,
            "token_file_sha256": tok_sha,
            "checks": checks,
            "pass": pass_all,
            "raw_preview": _preview(raw_dataset_path, preview_lines) if raw_dataset_path else [],
        }
        records.append(rec)

    total = len(records)
    passing = sum(1 for r in records if r.get("pass"))
    return {
        "format": "ck.training_provenance_audit.v1",
        "run_dir": str(run_dir),
        "total_runs": total,
        "passing_runs": passing,
        "failing_runs": total - passing,
        "records": records,
    }


def _print_table(payload: dict[str, Any], *, show_preview: bool = False) -> None:
    rows = payload.get("records") or []
    print(
        "run_order run_id                 stage pass status    checks     "
        "raw_dataset                              token_file"
    )
    for r in rows:
        checks = r.get("checks") or {}
        ok = all(bool(v) for v in checks.values())
        print(
            f"{str(r.get('run_order')).rjust(8)} "
            f"{str(r.get('run_id') or '-')[:22].ljust(22)} "
            f"{str(r.get('stage') or '-').ljust(5)} "
            f"{str(r.get('stage_pass') or '-').rjust(4)} "
            f"{str(r.get('status') or '-').ljust(8)} "
            f"{_status_chip(ok).ljust(9)} "
            f"{_short(r.get('raw_dataset')).ljust(38)} "
            f"{_short(r.get('token_file'))}"
        )
        if show_preview:
            preview_rows = r.get("raw_preview") or []
            for line in preview_rows[:3]:
                print(f"          preview: {line}")
    print(
        f"\nsummary: {payload.get('passing_runs', 0)}/{payload.get('total_runs', 0)} runs passed provenance checks"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit raw dataset ↔ token-file ↔ training-run provenance.")
    ap.add_argument("--run-dir", required=True, type=Path, help="v7 run dir (contains run_ledger.jsonl)")
    ap.add_argument("--json-out", type=Path, default=None, help="Optional output JSON path")
    ap.add_argument("--preview-lines", type=int, default=0, help="Include first N non-empty dataset lines per run")
    args = ap.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    payload = audit_run_dir(run_dir, preview_lines=max(0, int(args.preview_lines)))
    _print_table(payload, show_preview=bool(args.preview_lines > 0))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
