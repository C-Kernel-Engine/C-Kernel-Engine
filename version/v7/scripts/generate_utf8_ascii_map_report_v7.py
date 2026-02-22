#!/usr/bin/env python3
"""
Generate UTF-8 -> ASCII mapping report for dataset text.

This tool scans non-ASCII characters and reports whether each character has a
keyboard-style replacement in prepare_ascii_dataset_v7.py COMMON_ASCII_MAP.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PREP_ASCII_SCRIPT = ROOT / "version" / "v7" / "scripts" / "prepare_ascii_dataset_v7.py"


def _load_common_ascii_map() -> dict[str, str]:
    spec = importlib.util.spec_from_file_location("prepare_ascii_dataset_v7", PREP_ASCII_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import map source: {PREP_ASCII_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pairs = getattr(module, "COMMON_ASCII_MAP", None)
    if not isinstance(pairs, tuple):
        raise RuntimeError("COMMON_ASCII_MAP not found in prepare_ascii_dataset_v7.py")
    out: dict[str, str] = {}
    for row in pairs:
        if not isinstance(row, tuple) or len(row) != 2:
            continue
        src, dst = row
        if isinstance(src, str) and isinstance(dst, str):
            out[src] = dst
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate UTF-8 -> ASCII replacement coverage report")
    ap.add_argument("--input", required=True, help="Input dataset text file (UTF-8)")
    ap.add_argument("--tsv-out", required=True, help="TSV output path")
    ap.add_argument("--json-out", required=True, help="JSON output path")
    args = ap.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    if not in_path.exists():
        raise SystemExit(f"ERROR: input not found: {in_path}")

    map_dict = _load_common_ascii_map()
    text = in_path.read_text(encoding="utf-8", errors="replace")
    counter = Counter(ch for ch in text if ord(ch) >= 128)

    rows: list[dict] = []
    for ch, cnt in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
        repl = map_dict.get(ch)
        rows.append(
            {
                "char": ch,
                "codepoint": f"U+{ord(ch):04X}",
                "count": int(cnt),
                "ascii_replacement": repl,
                "mapped": bool(repl is not None),
            }
        )

    tsv_path = Path(args.tsv_out).expanduser().resolve()
    json_path = Path(args.json_out).expanduser().resolve()
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with tsv_path.open("w", encoding="utf-8") as f:
        f.write("char\tcodepoint\tcount\tascii_replacement\tmapped\n")
        for row in rows:
            char_cell = str(row["char"]).replace("\t", " ").replace("\n", " ")
            repl = str(row["ascii_replacement"] or "").replace("\t", " ").replace("\n", " ")
            f.write(f"{char_cell}\t{row['codepoint']}\t{row['count']}\t{repl}\t{str(row['mapped']).lower()}\n")

    summary = {
        "format": "utf8_ascii_map_report.v1",
        "source": str(in_path),
        "distinct_non_ascii_chars": int(len(rows)),
        "total_non_ascii_chars": int(sum(row["count"] for row in rows)),
        "mapped_distinct": int(sum(1 for row in rows if row["mapped"])),
        "mapped_total_chars": int(sum(row["count"] for row in rows if row["mapped"])),
        "unmapped_distinct": int(sum(1 for row in rows if not row["mapped"])),
        "unmapped_total_chars": int(sum(row["count"] for row in rows if not row["mapped"])),
        "rows": rows,
    }
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] input:   {in_path}")
    print(f"[OK] tsv:     {tsv_path}")
    print(f"[OK] json:    {json_path}")
    print(f"[INFO] distinct_non_ascii_chars={summary['distinct_non_ascii_chars']}")
    print(f"[INFO] total_non_ascii_chars={summary['total_non_ascii_chars']}")
    print(f"[INFO] mapped_total_chars={summary['mapped_total_chars']} unmapped_total_chars={summary['unmapped_total_chars']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

