#!/usr/bin/env python3
"""Compare private same-model corpus token traces without publishing content."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA = "cke.multimodal_corpus_token_parity"
SCHEMA_VERSION = 1


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _case_dirs(root: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in sorted(root.glob("image*/case_result.json")):
        case = _load_json(path)
        index = int(case["image_index"])
        if index in result:
            raise ValueError(f"duplicate image index {index} under {root}")
        result[index] = path
    return result


def _first_divergence(subject: list[int], oracle: list[int]) -> dict[str, int | None] | None:
    for index, (left, right) in enumerate(zip(subject, oracle)):
        if left != right:
            return {"step": index, "subject_token": left, "oracle_token": right}
    if len(subject) != len(oracle):
        index = min(len(subject), len(oracle))
        return {
            "step": index,
            "subject_token": subject[index] if index < len(subject) else None,
            "oracle_token": oracle[index] if index < len(oracle) else None,
        }
    return None


def compare(subject_dir: Path, oracle_dir: Path) -> dict[str, Any]:
    subject_config = _load_json(subject_dir / "config.json")
    oracle_config = _load_json(oracle_dir / "config.json")
    subject_manifest = subject_config.get("manifest_sha256")
    oracle_manifest = oracle_config.get("manifest_sha256")
    if not subject_manifest or subject_manifest != oracle_manifest:
        raise ValueError("subject and oracle must use the identical corpus manifest")

    subject_cases = _case_dirs(subject_dir)
    oracle_cases = _case_dirs(oracle_dir)
    indices = sorted(set(subject_cases) | set(oracle_cases))
    rows: list[dict[str, Any]] = []
    for index in indices:
        subject_path = subject_cases.get(index)
        oracle_path = oracle_cases.get(index)
        if subject_path is None or oracle_path is None:
            rows.append({
                "image_index": index,
                "status": "missing",
                "missing": "subject" if subject_path is None else "oracle",
            })
            continue
        subject = _load_json(subject_path)
        oracle = _load_json(oracle_path)
        if subject.get("image_sha256") != oracle.get("image_sha256"):
            raise ValueError(f"image {index} identity differs between subject and oracle")
        subject_case_config = subject.get("case_config") or {}
        oracle_case_config = oracle.get("case_config") or {}
        if subject_case_config.get("prompt_sha256") != oracle_case_config.get(
            "prompt_sha256"
        ):
            raise ValueError(f"image {index} prompt differs between subject and oracle")
        if subject_case_config.get("max_new_tokens") != oracle_case_config.get(
            "max_new_tokens"
        ):
            raise ValueError(
                f"image {index} generation budget differs between subject and oracle"
            )
        subject_tokens = subject.get("generated_token_ids")
        oracle_tokens = oracle.get("generated_token_ids")
        if not isinstance(subject_tokens, list) or not isinstance(oracle_tokens, list):
            raise ValueError(f"image {index} is missing a private generated token trace")
        left = [int(value) for value in subject_tokens]
        right = [int(value) for value in oracle_tokens]
        divergence = _first_divergence(left, right)
        rows.append({
            "image_index": index,
            "image_sha256": subject["image_sha256"],
            "status": "pass" if divergence is None else "fail",
            "subject_tokens": len(left),
            "oracle_tokens": len(right),
            "subject_trace_sha256": subject.get("token_trace_sha256"),
            "oracle_trace_sha256": oracle.get("token_trace_sha256"),
            "first_divergence": divergence,
        })
    passed = sum(row["status"] == "pass" for row in rows)
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "pass" if passed == len(rows) and rows else "fail",
        "manifest_sha256": subject_manifest,
        "requested": len(rows),
        "passed": passed,
        "failed": sum(row["status"] == "fail" for row in rows),
        "missing": sum(row["status"] == "missing" for row in rows),
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-dir", type=Path, required=True)
    parser.add_argument("--oracle-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = compare(args.subject_dir.resolve(), args.oracle_dir.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"status={report['status']} passed={report['passed']}/{report['requested']} "
        f"report={args.output}"
    )
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
