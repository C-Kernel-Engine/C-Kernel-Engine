#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


GIB = 1024 ** 3


def _nearest_existing_path(path: Path) -> Path:
    current = path.expanduser().resolve()
    while not current.exists():
        parent = current.parent
        if parent == current:
            break
        current = parent
    return current


def main() -> int:
    ap = argparse.ArgumentParser(description="Check free disk headroom for a run path.")
    ap.add_argument("--path", required=True, type=Path, help="Path whose backing filesystem should be checked")
    ap.add_argument("--min-free-gb", required=True, type=float, help="Required free space in GiB")
    ap.add_argument("--label", default="disk_headroom", help="Short label for reporting")
    args = ap.parse_args()

    target_path = _nearest_existing_path(args.path)
    usage = shutil.disk_usage(target_path)
    threshold_bytes = int(float(args.min_free_gb) * GIB)
    payload = {
        "label": str(args.label),
        "path": str(target_path),
        "required_free_gib": float(args.min_free_gb),
        "required_free_bytes": threshold_bytes,
        "free_gib": usage.free / GIB,
        "free_bytes": int(usage.free),
        "used_bytes": int(usage.used),
        "total_bytes": int(usage.total),
        "ok": usage.free >= threshold_bytes,
    }
    print(json.dumps(payload, ensure_ascii=True))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
