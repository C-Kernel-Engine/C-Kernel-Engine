#!/usr/bin/env python3
"""
Build SVG alignment datasets for DPO/GRPO/PPO stages.

This script emits two artifact families:
1) Objective-native JSONL (future true DPO/GRPO/PPO trainers)
2) CE-surrogate text rows (usable today with train_data_pipeline_v7.py)

Input rows are expected to look like:
  <task>...</task><svg ...>...</svg><eos>
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any


LINE_RE = re.compile(
    r"^\s*<task>(?P<task>.*?)</task>(?P<svg><svg.*</svg>)(?:<eos>)?\s*$",
    re.DOTALL,
)
COLORS = [
    "red",
    "orange",
    "gold",
    "green",
    "teal",
    "blue",
    "navy",
    "purple",
    "black",
    "gray",
    "brown",
    "pink",
]


def _assert_ascii(text: str, what: str) -> None:
    try:
        text.encode("ascii")
    except UnicodeEncodeError as exc:
        raise RuntimeError(f"non-ascii content in {what}: {exc}") from exc


def _canon_line(line: str) -> str:
    return " ".join(line.strip().split())


def _mutate_color(svg: str, rng: random.Random) -> str:
    for src in COLORS:
        token = f'"{src}"'
        if token in svg:
            dst = src
            while dst == src:
                dst = COLORS[rng.randrange(len(COLORS))]
            return svg.replace(token, f'"{dst}"', 1)
    return svg


def _make_bad_svg(svg: str, rng: random.Random) -> str:
    # Strong corruption for rejected sample.
    if "</svg>" in svg:
        return svg.replace("</svg>", "", 1)
    if '="' in svg:
        return svg.replace('="', "=", 1)
    if len(svg) > 16:
        cut = rng.randint(8, len(svg) - 1)
        return svg[:cut]
    return svg + "<broken>"


def _make_mild_svg(svg: str, rng: random.Random) -> str:
    # Mild quality degradation while staying parse-like in many cases.
    v = _mutate_color(svg, rng)
    if v != svg:
        return v
    if 'stroke-width="' in svg:
        return svg.replace('stroke-width="', 'stroke-width="1', 1)
    if " fill=" in svg:
        return svg.replace(" fill=", ' fill="none" data-old-fill=', 1)
    return svg


def _parse_rows(paths: list[Path], dedupe: bool) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            m = LINE_RE.match(line)
            if not m:
                continue
            task = m.group("task").strip()
            svg = m.group("svg").strip()
            if not task or not svg:
                continue
            key = _canon_line(task + "||" + svg)
            if dedupe and key in seen:
                continue
            seen.add(key)
            rows.append({"task": task, "svg": svg})
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _write_txt(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build SVG alignment datasets for dpo/grpo/ppo stages")
    ap.add_argument(
        "--instruction-data",
        action="append",
        required=True,
        help="Input instruction dataset file(s). Repeat flag for multiple files.",
    )
    ap.add_argument("--out-dir", default="version/v7/data", help="Output directory")
    ap.add_argument("--prefix", default="svg_alignment", help="Output file prefix")
    ap.add_argument("--max-samples", type=int, default=50000, help="Max aligned rows to emit")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--max-line-length", type=int, default=4096, help="Drop CE rows above this length")
    ap.add_argument("--no-dedupe", action="store_true", help="Disable dedupe by task+svg pair")
    args = ap.parse_args()

    rng = random.Random(int(args.seed))
    out_dir = Path(args.out_dir).expanduser().resolve()
    input_paths = [Path(p).expanduser().resolve() for p in args.instruction_data]
    for p in input_paths:
        if not p.exists():
            raise SystemExit(f"missing --instruction-data file: {p}")

    parsed = _parse_rows(input_paths, dedupe=not bool(args.no_dedupe))
    if not parsed:
        raise SystemExit("no valid <task>...</task><svg...></svg> rows found in --instruction-data")
    rng.shuffle(parsed)
    parsed = parsed[: int(args.max_samples)]

    dpo_pairs: list[dict[str, Any]] = []
    grpo_rollouts: list[dict[str, Any]] = []
    ppo_records: list[dict[str, Any]] = []
    dpo_ce: list[str] = []
    grpo_ce: list[str] = []
    ppo_ce: list[str] = []

    dropped_long = 0

    for i, row in enumerate(parsed):
        task = row["task"]
        chosen = row["svg"]
        mild = _make_mild_svg(chosen, rng)
        bad = _make_bad_svg(chosen, rng)

        if bad == chosen:
            bad = bad + "<broken>"
        if mild == chosen:
            mild = _mutate_color(chosen, rng)

        dpo_pairs.append(
            {
                "id": i,
                "objective": "dpo",
                "prompt": task,
                "chosen": chosen,
                "rejected": bad,
                "source": "synthetic_svg_alignment",
            }
        )
        grpo_rollouts.append(
            {
                "id": i,
                "objective": "grpo",
                "prompt": task,
                "candidates": [
                    {"text": chosen, "reward": 1.00},
                    {"text": mild, "reward": 0.25},
                    {"text": bad, "reward": -0.50},
                ],
                "source": "synthetic_svg_alignment",
            }
        )
        ppo_records.append(
            {
                "id": i,
                "objective": "ppo",
                "prompt": task,
                "response": chosen,
                "reward": 1.00,
                "advantage": 1.00,
                "old_logprob": 0.0,
                "source": "synthetic_svg_alignment",
            }
        )

        dpo_line = (
            f"<task>{task}</task>"
            f"<chosen>{chosen}</chosen>"
            f"<rejected>{bad}</rejected>"
            f"<pick>chosen</pick><eos>"
        )
        grpo_line = (
            f"<task>{task}</task>"
            f"<cand reward=\"1.00\">{chosen}</cand>"
            f"<cand reward=\"0.25\">{mild}</cand>"
            f"<cand reward=\"-0.50\">{bad}</cand>"
            f"<pick>best</pick><eos>"
        )
        ppo_line = (
            f"<task>{task}</task>"
            f"<response reward=\"1.00\" advantage=\"1.00\">{chosen}</response><eos>"
        )
        for line, sink in ((dpo_line, dpo_ce), (grpo_line, grpo_ce), (ppo_line, ppo_ce)):
            _assert_ascii(line, "ce row")
            if len(line) > int(args.max_line_length):
                dropped_long += 1
                continue
            sink.append(line)

    out = {
        "dpo_pairs_jsonl": out_dir / f"{args.prefix}_dpo_pairs.jsonl",
        "grpo_rollouts_jsonl": out_dir / f"{args.prefix}_grpo_rollouts.jsonl",
        "ppo_preferences_jsonl": out_dir / f"{args.prefix}_ppo_preferences.jsonl",
        "dpo_ce_txt": out_dir / f"{args.prefix}_dpo_ce_train.txt",
        "grpo_ce_txt": out_dir / f"{args.prefix}_grpo_ce_train.txt",
        "ppo_ce_txt": out_dir / f"{args.prefix}_ppo_ce_train.txt",
        "manifest_json": out_dir / f"{args.prefix}_alignment_manifest.json",
    }

    _write_jsonl(out["dpo_pairs_jsonl"], dpo_pairs)
    _write_jsonl(out["grpo_rollouts_jsonl"], grpo_rollouts)
    _write_jsonl(out["ppo_preferences_jsonl"], ppo_records)
    _write_txt(out["dpo_ce_txt"], dpo_ce)
    _write_txt(out["grpo_ce_txt"], grpo_ce)
    _write_txt(out["ppo_ce_txt"], ppo_ce)

    manifest = {
        "schema": "ck.svg_alignment_dataset.v1",
        "seed": int(args.seed),
        "max_samples": int(args.max_samples),
        "input_files": [str(p) for p in input_paths],
        "rows_parsed": int(len(parsed)),
        "rows_written": {
            "dpo_pairs": int(len(dpo_pairs)),
            "grpo_rollouts": int(len(grpo_rollouts)),
            "ppo_preferences": int(len(ppo_records)),
            "dpo_ce": int(len(dpo_ce)),
            "grpo_ce": int(len(grpo_ce)),
            "ppo_ce": int(len(ppo_ce)),
        },
        "ce_drop_long_rows": int(dropped_long),
        "objective_note": {
            "dpo": "jsonl is true-pairs format; *_ce_train.txt is current v7 CE-surrogate",
            "grpo": "jsonl has candidates+rewards; *_ce_train.txt is current v7 CE-surrogate",
            "ppo": "jsonl has reward/advantage fields; *_ce_train.txt is current v7 CE-surrogate",
        },
        "artifacts": {k: str(v) for k, v in out.items()},
    }
    out["manifest_json"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("svg alignment dataset build complete")
    print(f"  out_dir: {out_dir}")
    print(f"  rows:    {len(parsed)}")
    print(f"  dpo:     {out['dpo_pairs_jsonl']}")
    print(f"  grpo:    {out['grpo_rollouts_jsonl']}")
    print(f"  ppo:     {out['ppo_preferences_jsonl']}")
    print(f"  manifest:{out['manifest_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

