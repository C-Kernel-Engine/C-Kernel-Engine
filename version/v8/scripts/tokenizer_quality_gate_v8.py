#!/usr/bin/env python3
"""Tokenizer quality gate — flag degenerate vocab before training starts.

Detects:
  - content-embedded mega-tokens (literal topic text baked into single tokens)
  - over-merged structural tokens
  - vocab budget waste from non-generalizable entries
  - imbalanced token-length distributions

Usage:
  python3 version/v8/scripts/tokenizer_quality_gate_v8.py --run <run_dir>
  python3 version/v8/scripts/tokenizer_quality_gate_v8.py --tokenizer <tokenizer.json>

Outputs:
  <run_dir>/tokenizer_quality_gate.json   (when --run)
  stdout summary with PASS / WARN / FAIL verdict
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# ── thresholds ──────────────────────────────────────────────────────
CONTENT_WARN_LEN = 60       # content token > this → warn
CONTENT_FAIL_LEN = 100      # content token > this → fail-grade
STRUCTURAL_WARN_LEN = 140   # structural (@-ref) token > this → warn

# vocab-fraction gates
CONTENT_MEGA_WARN_FRAC = 0.05   # >5% of vocab is content mega → warn
CONTENT_MEGA_FAIL_FRAC = 0.15   # >15% of vocab is content mega → fail

# length-bucket balance
BYTE_LEVEL_WARN_FRAC = 0.40     # >40% byte-level tokens → warn (vocab too raw)
MEGA_TOKEN_WARN_FRAC = 0.10     # >10% tokens >80 bytes → warn


def _load_tokenizer_vocab(tokenizer_path: Path) -> dict[str, int]:
    """Load vocab dict from tokenizer.json (HF format)."""
    doc = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    model = doc.get("model", {})
    vocab = model.get("vocab", {})
    if isinstance(vocab, dict) and vocab:
        return vocab
    # Fallback: added_tokens can contain the full vocab in some formats.
    added = doc.get("added_tokens", [])
    if isinstance(added, list) and added:
        return {str(t.get("content", t.get("id", ""))): int(t.get("id", i)) for i, t in enumerate(added)}
    return {}


def _classify_token(token: str) -> str:
    """Classify a single vocab token into a quality category."""
    byte_len = len(token.encode("utf-8", errors="replace"))
    if byte_len <= 1:
        return "byte"
    if token.startswith("<|") and token.endswith("|>"):
        return "special"
    if token.startswith("[") and token.endswith("]"):
        if "@" in token:
            return "structural"
        body = token[1:-1]
        if ":" not in body:
            return "control_tag"
        # Has colon but no @-refs: check if it's a short control or content-embedded
        if byte_len <= 40:
            return "control_tag"
        return "content_embedded"
    if token.startswith("[") and ":" in token:
        # Partial tag (sometimes BPE splits across the closing bracket)
        if "@" in token:
            return "structural_fragment"
        if byte_len > 40:
            return "content_fragment"
    return "subword"


def _analyze_vocab(vocab: dict[str, int]) -> dict[str, Any]:
    """Run full quality analysis on a vocab."""
    total = len(vocab)
    if total == 0:
        return {"error": "empty vocab", "verdict": "FAIL"}

    classifications: dict[str, list[dict[str, Any]]] = {}
    length_buckets = Counter()
    all_lengths: list[int] = []

    for token, token_id in sorted(vocab.items(), key=lambda x: x[1]):
        byte_len = len(token.encode("utf-8", errors="replace"))
        cls = _classify_token(token)
        all_lengths.append(byte_len)

        # Length bucket
        if byte_len <= 1:
            length_buckets["1_byte"] += 1
        elif byte_len <= 10:
            length_buckets["2_10_bytes"] += 1
        elif byte_len <= 40:
            length_buckets["11_40_bytes"] += 1
        elif byte_len <= 80:
            length_buckets["41_80_bytes"] += 1
        else:
            length_buckets["81_plus_bytes"] += 1

        entry = {
            "id": token_id,
            "token": token[:200],
            "byte_len": byte_len,
        }
        classifications.setdefault(cls, []).append(entry)

    # Flagged tokens
    content_mega = [
        e for e in classifications.get("content_embedded", [])
        if e["byte_len"] > CONTENT_WARN_LEN
    ]
    content_fail = [
        e for e in content_mega
        if e["byte_len"] > CONTENT_FAIL_LEN
    ]
    structural_long = [
        e for e in classifications.get("structural", [])
        if e["byte_len"] > STRUCTURAL_WARN_LEN
    ]

    # Fractions
    content_mega_frac = len(content_mega) / total if total else 0
    byte_frac = length_buckets.get("1_byte", 0) / total if total else 0
    mega_frac = length_buckets.get("81_plus_bytes", 0) / total if total else 0

    # Verdict
    issues: list[dict[str, str]] = []
    verdict = "PASS"

    if content_mega_frac > CONTENT_MEGA_FAIL_FRAC:
        issues.append({
            "severity": "FAIL",
            "code": "CONTENT_MEGA_FRACTION",
            "message": (
                f"{len(content_mega)} content-embedded tokens >{CONTENT_WARN_LEN}B "
                f"= {content_mega_frac:.1%} of vocab (threshold: {CONTENT_MEGA_FAIL_FRAC:.0%}). "
                f"These bake topic-specific text into single tokens and cannot generalize."
            ),
        })
        verdict = "FAIL"
    elif content_mega_frac > CONTENT_MEGA_WARN_FRAC:
        issues.append({
            "severity": "WARN",
            "code": "CONTENT_MEGA_FRACTION",
            "message": (
                f"{len(content_mega)} content-embedded tokens >{CONTENT_WARN_LEN}B "
                f"= {content_mega_frac:.1%} of vocab (warn threshold: {CONTENT_MEGA_WARN_FRAC:.0%})."
            ),
        })
        if verdict == "PASS":
            verdict = "WARN"

    if content_fail:
        sev = "FAIL" if len(content_fail) > 5 else "WARN"
        issues.append({
            "severity": sev,
            "code": "CONTENT_TOKENS_OVER_100B",
            "message": (
                f"{len(content_fail)} content tokens >{CONTENT_FAIL_LEN}B. "
                f"Worst: {content_fail[0]['byte_len']}B — '{content_fail[0]['token'][:80]}...'"
            ),
        })
        if sev == "FAIL":
            verdict = "FAIL"
        elif verdict == "PASS":
            verdict = "WARN"

    if byte_frac > BYTE_LEVEL_WARN_FRAC:
        issues.append({
            "severity": "WARN",
            "code": "HIGH_BYTE_FRACTION",
            "message": (
                f"{byte_frac:.1%} of vocab is single-byte tokens. "
                f"BPE may not have merged enough subwords."
            ),
        })
        if verdict == "PASS":
            verdict = "WARN"

    if mega_frac > MEGA_TOKEN_WARN_FRAC:
        issues.append({
            "severity": "WARN",
            "code": "HIGH_MEGA_FRACTION",
            "message": (
                f"{mega_frac:.1%} of vocab is >80B tokens. "
                f"BPE merged too aggressively on long repeated patterns."
            ),
        })
        if verdict == "PASS":
            verdict = "WARN"

    if structural_long:
        issues.append({
            "severity": "WARN",
            "code": "STRUCTURAL_TOKENS_VERY_LONG",
            "message": (
                f"{len(structural_long)} structural (@-ref) tokens >{STRUCTURAL_WARN_LEN}B. "
                f"Consider splitting into composable sub-tags."
            ),
        })
        if verdict == "PASS":
            verdict = "WARN"

    if not issues:
        issues.append({
            "severity": "PASS",
            "code": "ALL_CLEAR",
            "message": "No degenerate token patterns detected.",
        })

    # Sort flagged tokens by length descending for the report
    content_mega.sort(key=lambda e: e["byte_len"], reverse=True)

    sorted_lengths = sorted(all_lengths)
    p50 = sorted_lengths[len(sorted_lengths) // 2] if sorted_lengths else 0
    p90 = sorted_lengths[int(len(sorted_lengths) * 0.9)] if sorted_lengths else 0
    p99 = sorted_lengths[int(len(sorted_lengths) * 0.99)] if sorted_lengths else 0

    return {
        "schema": "ck.tokenizer_quality_gate.v1",
        "verdict": verdict,
        "vocab_size": total,
        "issues": issues,
        "classification_counts": {
            cls: len(entries) for cls, entries in sorted(classifications.items())
        },
        "length_buckets": dict(length_buckets),
        "length_percentiles": {"p50": p50, "p90": p90, "p99": p99, "max": max(all_lengths) if all_lengths else 0},
        "content_mega_tokens": content_mega[:30],
        "content_mega_count": len(content_mega),
        "content_mega_fraction": round(content_mega_frac, 4),
        "structural_long_count": len(structural_long),
        "thresholds": {
            "content_warn_len": CONTENT_WARN_LEN,
            "content_fail_len": CONTENT_FAIL_LEN,
            "structural_warn_len": STRUCTURAL_WARN_LEN,
            "content_mega_warn_frac": CONTENT_MEGA_WARN_FRAC,
            "content_mega_fail_frac": CONTENT_MEGA_FAIL_FRAC,
        },
    }


def _find_tokenizer_json(run_dir: Path) -> Path | None:
    """Find tokenizer.json in a run directory."""
    for candidate in [
        run_dir / "tokenizer.json",
        run_dir / "dataset" / "tokenizer" / "tokenizer.json",
    ]:
        if candidate.exists():
            return candidate
    return None


def run_gate(*, run_dir: Path | None = None, tokenizer_path: Path | None = None) -> dict[str, Any]:
    """Run the tokenizer quality gate and return the report."""
    if tokenizer_path is None and run_dir is not None:
        tokenizer_path = _find_tokenizer_json(run_dir)
    if tokenizer_path is None or not tokenizer_path.exists():
        return {"error": "tokenizer.json not found", "verdict": "FAIL"}

    vocab = _load_tokenizer_vocab(tokenizer_path)
    report = _analyze_vocab(vocab)
    report["tokenizer_path"] = str(tokenizer_path)

    if run_dir is not None:
        report["run_dir"] = str(run_dir)
        out_path = run_dir / "tokenizer_quality_gate.json"
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        report["output_path"] = str(out_path)

    return report


def _print_summary(report: dict[str, Any]) -> None:
    """Print a human-readable summary to stdout."""
    verdict = report.get("verdict", "UNKNOWN")
    icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}.get(verdict, "❓")
    print(f"\n{icon}  Tokenizer Quality Gate: {verdict}")
    print(f"   Vocab size: {report.get('vocab_size', '?')}")

    cls_counts = report.get("classification_counts", {})
    if cls_counts:
        print(f"   Token classes: {', '.join(f'{k}={v}' for k, v in sorted(cls_counts.items()))}")

    buckets = report.get("length_buckets", {})
    if buckets:
        print(f"   Length buckets: {', '.join(f'{k}={v}' for k, v in sorted(buckets.items()))}")

    pct = report.get("length_percentiles", {})
    if pct:
        print(f"   Length percentiles: p50={pct.get('p50')}B  p90={pct.get('p90')}B  p99={pct.get('p99')}B  max={pct.get('max')}B")

    mega_count = report.get("content_mega_count", 0)
    mega_frac = report.get("content_mega_fraction", 0)
    if mega_count:
        print(f"   Content mega-tokens (>{report.get('thresholds',{}).get('content_warn_len',60)}B, no @-refs): "
              f"{mega_count} = {mega_frac:.1%} of vocab")

    issues = report.get("issues", [])
    if issues:
        print(f"\n   Issues ({len(issues)}):")
        for issue in issues:
            sev = issue.get("severity", "?")
            icon2 = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}.get(sev, "·")
            print(f"     {icon2} [{issue.get('code', '?')}] {issue.get('message', '')}")

    flagged = report.get("content_mega_tokens", [])
    if flagged:
        show = flagged[:8]
        print(f"\n   Worst content-embedded tokens:")
        for entry in show:
            print(f"     [{entry['id']:4d}] {entry['byte_len']:3d}B | {entry['token'][:100]}")
        if len(flagged) > 8:
            print(f"     ... and {len(flagged) - 8} more")

    out = report.get("output_path")
    if out:
        print(f"\n   Report: {out}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenizer quality gate")
    parser.add_argument("--run", type=Path, help="Run directory")
    parser.add_argument("--tokenizer", type=Path, help="Direct path to tokenizer.json")
    parser.add_argument("--json-only", action="store_true", help="Output JSON only, no summary")
    args = parser.parse_args()

    if not args.run and not args.tokenizer:
        parser.error("Provide --run or --tokenizer")

    report = run_gate(run_dir=args.run, tokenizer_path=args.tokenizer)

    if args.json_only:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        _print_summary(report)

    verdict = report.get("verdict", "FAIL")
    if verdict == "FAIL":
        sys.exit(1)
    elif verdict == "WARN":
        sys.exit(0)  # warn but don't block
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
