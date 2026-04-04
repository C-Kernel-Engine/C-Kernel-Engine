#!/usr/bin/env python3
"""Build a generic split-aware probe report for a training run."""

from __future__ import annotations

import argparse
import atexit
import html
import json
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from probe_report_adapters_v7 import (
    apply_output_adapter,
    extract_response_text,
    is_valid_svg,
    normalize_svg,
    normalize_whitespace,
)
from pack_training_tokens_v7 import _TrueBPEHandle


ROOT = Path(__file__).resolve().parents[3]
CK_CHAT_SCRIPT = ROOT / "scripts" / "ck_chat.py"
PROBE_CONTRACT_DIR = ROOT / "version" / "v7" / "data" / "probe_contracts"
CONTRACT_SCHEMA = "ck.probe_report_contract.v1"
REPORT_SCHEMA = "ck.probe_report.v1"
DEFAULT_TOKENIZER_LIB = ROOT / "build" / "libckernel_tokenizer.so"


def _html(text: Any) -> str:
    return html.escape("" if text is None else str(text))


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(raw: str | Path, base_dir: Path) -> Path:
    path = Path(str(raw)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _python_bin() -> str:
    venv = ROOT / ".venv" / "bin" / "python"
    return str(venv) if venv.exists() else sys.executable


class _TokenBudgetAnalyzer:
    def __init__(self, *, handle: _TrueBPEHandle, context_len: int | None, decode_max_tokens: int | None):
        self.handle = handle
        self.context_len = int(context_len) if context_len else None
        self.decode_max_tokens = int(decode_max_tokens) if decode_max_tokens else None

    @classmethod
    def for_run(cls, run_dir: Path, decode_cfg: dict[str, Any]):
        tokenizer_json_candidates = [
            run_dir / "tokenizer.json",
            run_dir / "dataset" / "tokenizer" / "tokenizer.json",
        ]
        tokenizer_bin_candidates = [
            run_dir / "tokenizer_bin",
            run_dir / "dataset" / "tokenizer" / "tokenizer_bin",
        ]
        tokenizer_json = next((p for p in tokenizer_json_candidates if p.exists()), None)
        tokenizer_bin = next((p for p in tokenizer_bin_candidates if p.exists()), None)
        if tokenizer_json is None or tokenizer_bin is None or not DEFAULT_TOKENIZER_LIB.exists():
            return None

        context_len = None
        for candidate in (run_dir / "config.json", run_dir / "weights_manifest.json", run_dir / "train_init_config.json"):
            if not candidate.exists():
                continue
            try:
                doc = _load_json(candidate)
            except Exception:
                continue
            if isinstance(doc, dict):
                if isinstance(doc.get("architecture"), dict):
                    raw = doc["architecture"].get("context_len")
                    if raw:
                        context_len = int(raw)
                        break
                if isinstance(doc.get("config"), dict):
                    raw = doc["config"].get("context_len")
                    if raw:
                        context_len = int(raw)
                        break

        handle = _TrueBPEHandle(DEFAULT_TOKENIZER_LIB, tokenizer_bin, tokenizer_json)
        analyzer = cls(
            handle=handle,
            context_len=context_len,
            decode_max_tokens=int(decode_cfg.get("max_tokens") or 0) or None,
        )
        atexit.register(analyzer.close)
        return analyzer

    def close(self) -> None:
        if getattr(self, "handle", None) is not None:
            self.handle.close()
            self.handle = None

    def count_tokens(self, text: str) -> int:
        if not text or self.handle is None:
            return 0
        return len(self.handle.encode(text))

    def classify(
        self,
        *,
        prompt: str,
        parsed_output: str,
        expected_output: str | None,
        stop_markers: list[str],
        render_error: str | None,
    ) -> dict[str, Any]:
        prompt_tokens = self.count_tokens(prompt)
        parsed_tokens = self.count_tokens(parsed_output)
        expected_tokens = self.count_tokens(expected_output or "")
        decode_threshold = max(1, int(self.decode_max_tokens or 0) - 1) if self.decode_max_tokens else None
        context_threshold = max(1, int(self.context_len or 0) - 1) if self.context_len else None
        hit_decode_ceiling = bool(decode_threshold and parsed_tokens >= decode_threshold)
        hit_context_ceiling = bool(context_threshold and (prompt_tokens + parsed_tokens) >= context_threshold)
        missing_stop_marker = bool(stop_markers) and not any(marker in parsed_output for marker in stop_markers)
        prefix_matches_expected = bool(expected_output) and bool(parsed_output) and str(expected_output).startswith(parsed_output)
        truncated_at_budget = (
            missing_stop_marker
            and prefix_matches_expected
            and ("must end with" in str(render_error or "") or not str(render_error or "").strip())
            and (hit_decode_ceiling or hit_context_ceiling)
        )
        reason_parts: list[str] = []
        if hit_decode_ceiling:
            reason_parts.append("decode_max_tokens")
        if hit_context_ceiling:
            reason_parts.append("context_len")
        return {
            "prompt_tokens": prompt_tokens,
            "parsed_output_tokens": parsed_tokens,
            "expected_output_tokens": expected_tokens,
            "decode_max_tokens": self.decode_max_tokens,
            "context_len": self.context_len,
            "hit_decode_ceiling": hit_decode_ceiling,
            "hit_context_ceiling": hit_context_ceiling,
            "missing_stop_marker": missing_stop_marker,
            "prefix_matches_expected": prefix_matches_expected,
            "truncated_at_budget": truncated_at_budget,
            "truncation_reason": "+".join(reason_parts) if truncated_at_budget and reason_parts else None,
        }


def _candidate_dataset_manifests(dataset_path: Path) -> list[Path]:
    p = dataset_path.expanduser().resolve()
    candidates = [p.with_name(f"{p.stem}_manifest.json")]
    transforms = [
        p.stem.replace("_train", ""),
        p.stem.replace("_holdout", ""),
        p.stem.replace("_all", ""),
        p.stem.replace("_seen_prompts", ""),
        p.stem.replace("_holdout_prompts", ""),
    ]
    seen: set[str] = {str(candidates[0])}
    for stem in transforms:
        if not stem:
            continue
        candidate = p.with_name(f"{stem}_manifest.json")
        key = str(candidate)
        if key not in seen:
            candidates.append(candidate)
            seen.add(key)
    return candidates


def _candidate_contract_stems(dataset_path: Path) -> list[str]:
    stem = dataset_path.expanduser().resolve().stem
    stems = [
        stem,
        stem.replace("_train", ""),
        stem.replace("_holdout", ""),
        stem.replace("_all", ""),
        stem.replace("_seen_prompts", ""),
        stem.replace("_holdout_prompts", ""),
    ]
    out: list[str] = []
    seen: set[str] = set()
    for item in stems:
        key = str(item or "").strip()
        if key and key not in seen:
            out.append(key)
            seen.add(key)
    return out


def _find_dataset_path(run_dir: Path) -> Path | None:
    for name, key in (
        ("dataset_profile.json", "path"),
        ("dataset_qc.json", "path"),
        ("tokenizer_roundtrip.json", "dataset_path"),
    ):
        candidate = run_dir / name
        if not candidate.exists():
            continue
        try:
            doc = _load_json(candidate)
        except Exception:
            continue
        if not isinstance(doc, dict):
            continue
        raw = doc.get(key)
        if isinstance(raw, str) and raw.strip():
            resolved = Path(raw).expanduser().resolve()
            if resolved.exists():
                return resolved

    candidates = sorted(
        run_dir.glob("train*.json"),
        key=lambda path: ("prepare" in path.stem.lower(), path.name),
    )
    for path in candidates:
        try:
            doc = _load_json(path)
        except Exception:
            continue
        if not isinstance(doc, dict):
            continue
        dataset = doc.get("dataset")
        if isinstance(dataset, str) and dataset.strip():
            resolved = Path(dataset).expanduser().resolve()
            if resolved.exists():
                return resolved
    pipeline_path = run_dir / "training_pipeline_latest.json"
    if pipeline_path.exists():
        try:
            doc = _load_json(pipeline_path)
        except Exception:
            doc = None
        stages = (((doc or {}).get("pipeline") or {}).get("stages") or []) if isinstance(doc, dict) else []
        for stage in stages:
            if not isinstance(stage, dict):
                continue
            for dataset in stage.get("datasets") or []:
                if not isinstance(dataset, dict):
                    continue
                raw = dataset.get("path")
                if isinstance(raw, str) and raw.strip():
                    resolved = Path(raw).expanduser().resolve()
                    if resolved.exists():
                        return resolved
    return None


def _autodiscover_contract(run_dir: Path, explicit_contract: str | None) -> tuple[dict[str, Any], str]:
    if explicit_contract:
        path = Path(explicit_contract).expanduser().resolve()
        doc = _load_json(path)
        if not isinstance(doc, dict):
            raise SystemExit(f"invalid contract: {path}")
        return doc, str(path)

    direct = run_dir / "probe_report_contract.json"
    if direct.exists():
        doc = _load_json(direct)
        if isinstance(doc, dict):
            return doc, str(direct)

    dataset_path = _find_dataset_path(run_dir)
    if dataset_path is None:
        raise SystemExit(
            "could not discover a dataset for this run.\n"
            "Pass --contract explicitly or add probe_report_contract metadata next to the dataset manifest."
        )

    for stem in _candidate_contract_stems(dataset_path):
        tracked = PROBE_CONTRACT_DIR / f"{stem}_probe_report_contract.json"
        if tracked.exists():
            contract_doc = _load_json(tracked)
            if isinstance(contract_doc, dict):
                return contract_doc, str(tracked)
        sibling = dataset_path.with_name(f"{stem}_probe_report_contract.json")
        if sibling.exists():
            contract_doc = _load_json(sibling)
            if isinstance(contract_doc, dict):
                return contract_doc, str(sibling)

    for manifest_path in _candidate_dataset_manifests(dataset_path):
        if not manifest_path.exists():
            continue
        doc = _load_json(manifest_path)
        if not isinstance(doc, dict):
            continue
        artifacts = doc.get("artifacts") if isinstance(doc.get("artifacts"), dict) else {}
        contract_raw = None
        if isinstance(artifacts, dict):
            contract_raw = artifacts.get("probe_report_contract")
        if not isinstance(contract_raw, str):
            contract_raw = doc.get("probe_report_contract")
        if isinstance(contract_raw, str) and contract_raw.strip():
            contract_path = _resolve_path(contract_raw, manifest_path.parent)
            contract_doc = _load_json(contract_path)
            if isinstance(contract_doc, dict):
                return contract_doc, str(contract_path)
        if manifest_path.stem.endswith("_manifest"):
            prefix = manifest_path.stem[: -len("_manifest")]
            sibling = manifest_path.with_name(f"{prefix}_probe_report_contract.json")
            if sibling.exists():
                contract_doc = _load_json(sibling)
                if isinstance(contract_doc, dict):
                    return contract_doc, str(sibling)

    raise SystemExit(
        "no probe report contract found.\n"
        f"run={run_dir}\n"
        f"dataset={dataset_path}\n"
        "Expected one of:\n"
        "  - RUN_DIR/probe_report_contract.json\n"
        "  - manifest artifacts.probe_report_contract\n"
        "  - sibling *_probe_report_contract.json next to the dataset manifest\n"
    )


def _parse_prompt_svg_inline(path: Path) -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        start = line.find("<svg")
        end = line.rfind("</svg>")
        if start < 0 or end < 0:
            continue
        end += len("</svg>")
        prompt = line[:start]
        svg_xml = line[start:end]
        catalog[prompt] = {
            "prompt": prompt,
            "expected_output": svg_xml,
            "expected_rendered_output": svg_xml,
            "expected_rendered_mime": "image/svg+xml",
        }
    return catalog


def _load_catalog(contract: dict[str, Any], contract_path: Path) -> dict[str, dict[str, Any]]:
    catalog_cfg = contract.get("catalog")
    if not isinstance(catalog_cfg, dict):
        return {}
    fmt = str(catalog_cfg.get("format") or "").strip().lower()
    raw_path = catalog_cfg.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return {}
    path = _resolve_path(raw_path, contract_path.parent)
    if fmt == "prompt_svg_inline":
        return _parse_prompt_svg_inline(path)
    if fmt != "json_rows":
        raise SystemExit(f"unsupported catalog format: {fmt}")
    doc = _load_json(path)
    if not isinstance(doc, list):
        raise SystemExit(f"catalog must be a JSON list: {path}")
    prompt_key = str(catalog_cfg.get("prompt_key") or "prompt")
    output_key = str(catalog_cfg.get("output_key") or "output_text")
    rendered_key = str(catalog_cfg.get("rendered_key") or catalog_cfg.get("svg_key") or "svg_xml")
    rendered_mime = str(catalog_cfg.get("rendered_mime") or "").strip() or None
    rendered_mime_key = str(catalog_cfg.get("rendered_mime_key") or "").strip() or None
    split_key = str(catalog_cfg.get("split_key") or "split")
    out: dict[str, dict[str, Any]] = {}
    for row in doc:
        if not isinstance(row, dict):
            continue
        prompt = str(row.get(prompt_key) or "").strip()
        if not prompt:
            continue
        out[prompt] = {
            "prompt": prompt,
            "expected_output": str(row.get(output_key) or "").strip() or None,
            "expected_rendered_output": str(row.get(rendered_key) or "").strip() or None,
            "expected_rendered_mime": (
                str(row.get(rendered_mime_key) or "").strip() or None if rendered_mime_key else rendered_mime
            ),
            "catalog_split": str(row.get(split_key) or "").strip() or None,
            "content_json": row.get("content_json") if isinstance(row.get("content_json"), dict) else None,
        }
    return out


def _load_cases(
    contract: dict[str, Any],
    contract_path: Path,
    catalog: dict[str, dict[str, Any]],
    max_per_split: int | None,
) -> list[dict[str, Any]]:
    split_defs = contract.get("splits")
    if not isinstance(split_defs, list) or not split_defs:
        raise SystemExit("probe report contract requires a non-empty splits[] list")
    cases: list[dict[str, Any]] = []
    for split_def in split_defs:
        if not isinstance(split_def, dict):
            continue
        split_name = str(split_def.get("name") or "").strip()
        if not split_name:
            continue
        label = str(split_def.get("label") or split_name.title()).strip()
        limit = max_per_split if max_per_split is not None else split_def.get("limit")
        prompts: list[str] = []
        inline_cases = split_def.get("cases")
        if isinstance(inline_cases, list) and inline_cases:
            for idx, row in enumerate(inline_cases, start=1):
                if not isinstance(row, dict):
                    continue
                prompt = str(row.get("prompt") or "").strip()
                if not prompt:
                    continue
                if isinstance(limit, int) and limit > 0 and len(prompts) >= limit:
                    break
                prompts.append(prompt)
                cases.append(
                    {
                        "id": str(row.get("id") or f"{split_name}_{idx:02d}"),
                        "split": split_name,
                        "split_label": label,
                        "label": str(row.get("label") or f"{label} #{idx}"),
                        "prompt": prompt,
                        "expected_output": str(row.get("expected_output") or "").strip() or None,
                        "expected_rendered_output": str(
                            row.get("expected_rendered_output") or row.get("expected_svg") or ""
                        ).strip()
                        or None,
                        "expected_rendered_mime": (
                            str(row.get("expected_rendered_mime") or "").strip()
                            or ("image/svg+xml" if row.get("expected_svg") else None)
                        ),
                        "content_json": row.get("content_json") if isinstance(row.get("content_json"), dict) else None,
                    }
                )
            continue
        catalog_split = str(split_def.get("catalog_split") or "").strip() or None
        raw_path = split_def.get("path")
        if isinstance(raw_path, str) and raw_path.strip():
            prompts_path = _resolve_path(raw_path, contract_path.parent)
            prompts = [line.strip() for line in prompts_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        elif catalog_split:
            prompts = [
                prompt
                for prompt, row in catalog.items()
                if str((row or {}).get("catalog_split") or "").strip() == catalog_split
            ]
        else:
            continue
        if isinstance(limit, int) and limit > 0:
            prompts = prompts[:limit]
        for idx, prompt in enumerate(prompts, start=1):
            expected = dict(catalog.get(prompt) or {})
            cases.append(
                {
                    "id": f"{split_name}_{idx:02d}",
                    "split": split_name,
                    "split_label": label,
                    "label": f"{label} #{idx}",
                    "prompt": prompt,
                    "expected_output": expected.get("expected_output"),
                    "expected_rendered_output": expected.get("expected_rendered_output"),
                    "expected_rendered_mime": expected.get("expected_rendered_mime"),
                    "content_json": expected.get("content_json") if isinstance(expected.get("content_json"), dict) else None,
                }
            )
    if not cases:
        raise SystemExit("probe report contract resolved zero cases")
    return cases


def _run_prompt(model_dir: Path, prompt: str, decode_cfg: dict[str, Any], adapter_cfg: dict[str, Any]) -> str:
    max_tokens = int(decode_cfg.get("max_tokens") or 64)
    temperature = float(decode_cfg.get("temperature") or 0.0)
    repeat_penalty = float(decode_cfg.get("repeat_penalty") or 1.0)
    raw_stop = decode_cfg.get("stop_on_text")
    stop_markers: list[str] = []
    if isinstance(raw_stop, list):
        stop_markers = [str(marker).strip() for marker in raw_stop if str(marker).strip()]
    else:
        stop_text = str(raw_stop or "").strip()
        if stop_text:
            stop_markers = [stop_text]
    cmd = [
        _python_bin(),
        str(CK_CHAT_SCRIPT),
        "--model-dir",
        str(model_dir),
        "--python-tokenizer",
        "--chat-template",
        "none",
        "--allow-raw-prompt",
        "--prompt",
        prompt,
        "--max-tokens",
        str(max_tokens),
        "--temperature",
        str(temperature),
        "--repeat-penalty",
        str(repeat_penalty),
        "--no-stats",
    ]
    # Control-token visibility must come from the staged tokenizer/template
    # sidecars for the run; the chat runtime stays agnostic to individual specs.
    # The output adapter owns stop-marker truncation so the visible response
    # still includes terminators like [/scene] for parser/render checks.
    if stop_markers:
        for marker in stop_markers:
            cmd.extend(["--stop-on-text", marker])
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    if result.returncode != 0:
        raise SystemExit(
            "ck_chat.py failed while building probe report.\n"
            f"cmd={' '.join(cmd)}\n"
            f"stderr={result.stderr.strip()[:500]}"
        )
    return result.stdout


def _resolve_output_adapter(contract: dict[str, Any]) -> dict[str, Any]:
    def _normalize_name(name: str) -> str:
        lowered = str(name or "").strip().lower()
        if lowered == "svg_dsl":
            return "text_renderer"
        return lowered or "plain_text"

    adapter_cfg = contract.get("output_adapter")
    if isinstance(adapter_cfg, str) and adapter_cfg.strip():
        return {"name": _normalize_name(adapter_cfg)}
    if isinstance(adapter_cfg, dict):
        name = _normalize_name(str(adapter_cfg.get("name") or ""))
        if not name:
            raise SystemExit("output_adapter.name is required when output_adapter is an object")
        normalized = dict(adapter_cfg)
        normalized["name"] = name
        return normalized

    legacy_name = str(contract.get("adapter") or "").strip()
    if legacy_name:
        legacy: dict[str, Any] = {"name": _normalize_name(legacy_name)}
        renderer = str(contract.get("renderer") or "").strip()
        if renderer:
            legacy["renderer"] = renderer
        stop_markers = contract.get("stop_markers")
        if isinstance(stop_markers, list):
            legacy["stop_markers"] = [str(marker).strip() for marker in stop_markers if str(marker).strip()]
        return legacy
    return {"name": "plain_text"}


def _normalize_value(text: Any, mime: str | None = None) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    if str(mime or "").strip().lower() == "image/svg+xml":
        return normalize_svg(raw)
    return normalize_whitespace(raw)


def _normalize_scene_dsl(text: Any) -> str:
    return re.sub(r"\s+", "", str(text or "").strip())


def _build_preview_markup(payload: str | None, mime: str | None, fallback: str) -> str:
    content = str(payload or "").strip()
    mime_name = str(mime or "").strip().lower()
    if not content:
        return f'<div class="empty-preview">{_html(fallback)}</div>'
    if mime_name == "image/svg+xml" and is_valid_svg(content):
        return content
    return f'<div class="empty-preview">{_html(fallback)}</div>'


def _evaluate_case(
    case: dict[str, Any],
    raw_output: str,
    response_text: str,
    adapter_cfg: dict[str, Any],
    *,
    budget_analyzer: _TokenBudgetAnalyzer | None,
) -> dict[str, Any]:
    adapter_name = str(adapter_cfg.get("name") or "plain_text").strip()
    case_adapter_cfg = dict(adapter_cfg)
    content_json = case.get("content_json")
    if isinstance(content_json, dict):
        case_adapter_cfg["content_json"] = content_json
    case_adapter_cfg["prompt"] = str(case.get("prompt") or "")
    actual = apply_output_adapter(adapter_name, response_text, case_adapter_cfg)
    expected_output = str(case.get("expected_output") or "").strip() or None
    expected_rendered_output = str(case.get("expected_rendered_output") or "").strip() or None
    expected_rendered_mime = str(case.get("expected_rendered_mime") or "").strip() or None
    expected_eval = None
    if expected_output:
        expected_adapter_cfg = dict(case_adapter_cfg)
        expected_adapter_cfg.pop("repairer", None)
        expected_adapter_cfg.pop("prompt", None)
        expected_eval = apply_output_adapter(adapter_name, expected_output, expected_adapter_cfg)
    if not expected_rendered_output and isinstance(expected_eval, dict):
        expected_rendered_output = str(expected_eval.get("materialized_output") or "").strip() or None
    if not expected_rendered_mime and isinstance(expected_eval, dict):
        expected_rendered_mime = str(expected_eval.get("materialized_mime") or "").strip() or None

    parsed_output = str(actual.get("parsed_output") or "").strip()
    parsed_output_raw = str(actual.get("parsed_output_raw") or "").strip() or parsed_output
    materialized_output = str(actual.get("materialized_output") or "").strip() or None
    materialized_mime = str(actual.get("materialized_mime") or "").strip() or None
    renderer_name = str(case_adapter_cfg.get("renderer") or "").strip().lower()
    if adapter_name == "text_renderer" and renderer_name.startswith("structured_svg_scene_spec"):
        exact_match = bool(expected_output) and _normalize_scene_dsl(parsed_output) == _normalize_scene_dsl(expected_output)
    else:
        exact_match = bool(expected_output) and _normalize_value(parsed_output) == _normalize_value(expected_output)
    materialized_exact_match = bool(materialized_output and expected_rendered_output) and (
        _normalize_value(materialized_output, materialized_mime or expected_rendered_mime)
        == _normalize_value(expected_rendered_output, expected_rendered_mime or materialized_mime)
    )
    renderable = bool(actual.get("renderable")) and bool(materialized_output)
    valid_svg = (materialized_mime == "image/svg+xml") and is_valid_svg(materialized_output)
    stop_markers = [str(marker).strip() for marker in (adapter_cfg.get("stop_markers") or []) if str(marker).strip()]
    budget_diag = (
        budget_analyzer.classify(
            prompt=str(case.get("prompt") or ""),
            parsed_output=parsed_output,
            expected_output=expected_output,
            stop_markers=stop_markers,
            render_error=actual.get("render_error"),
        )
        if budget_analyzer is not None
        else {
            "prompt_tokens": None,
            "parsed_output_tokens": None,
            "expected_output_tokens": None,
            "decode_max_tokens": int(adapter_cfg.get("max_tokens") or 0) or None,
            "context_len": None,
            "hit_decode_ceiling": False,
            "hit_context_ceiling": False,
            "missing_stop_marker": False,
            "prefix_matches_expected": False,
            "truncated_at_budget": False,
            "truncation_reason": None,
        }
    )

    return {
        "id": case.get("id"),
        "split": case.get("split"),
        "split_label": case.get("split_label"),
        "label": case.get("label"),
        "prompt": case.get("prompt"),
        "expected_output": expected_output,
        "expected_rendered_output": expected_rendered_output,
        "expected_rendered_mime": expected_rendered_mime,
        "expected_svg": expected_rendered_output if expected_rendered_mime == "image/svg+xml" else None,
        "content_json": content_json if isinstance(content_json, dict) else None,
        "raw_output": raw_output.strip(),
        "response_text": response_text.strip(),
        "parsed_output": parsed_output,
        "parsed_output_raw": parsed_output_raw,
        "materialized_output": materialized_output,
        "materialized_mime": materialized_mime,
        "rendered_svg": materialized_output if materialized_mime == "image/svg+xml" else None,
        "render_error": actual.get("render_error"),
        "prefix_text": actual.get("prefix_text"),
        "tail_text": actual.get("tail_text"),
        "repair_applied": bool(actual.get("repair_applied")),
        "repairer": actual.get("repairer"),
        "repair_note": actual.get("repair_note"),
        "repair_diag": actual.get("repair_diag") if isinstance(actual.get("repair_diag"), dict) else None,
        "prompt_tokens": budget_diag.get("prompt_tokens"),
        "parsed_output_tokens": budget_diag.get("parsed_output_tokens"),
        "expected_output_tokens": budget_diag.get("expected_output_tokens"),
        "decode_max_tokens": budget_diag.get("decode_max_tokens"),
        "context_len": budget_diag.get("context_len"),
        "hit_decode_ceiling": bool(budget_diag.get("hit_decode_ceiling")),
        "hit_context_ceiling": bool(budget_diag.get("hit_context_ceiling")),
        "missing_stop_marker": bool(budget_diag.get("missing_stop_marker")),
        "prefix_matches_expected": bool(budget_diag.get("prefix_matches_expected")),
        "truncated_at_budget": bool(budget_diag.get("truncated_at_budget")),
        "truncation_reason": budget_diag.get("truncation_reason"),
        "exact_match": exact_match,
        "materialized_exact_match": materialized_exact_match,
        "svg_exact_match": materialized_exact_match if materialized_mime == "image/svg+xml" else False,
        "valid_svg": valid_svg,
        "renderable": renderable,
        "metrics": {
            "exact_match": 1.0 if exact_match else 0.0,
            "materialized_exact_match": 1.0 if materialized_exact_match else 0.0,
            "svg_exact_match": 1.0 if materialized_mime == "image/svg+xml" and materialized_exact_match else 0.0,
            "valid_svg": 1.0 if valid_svg else 0.0,
            "renderable": 1.0 if renderable else 0.0,
            "truncated_at_budget": 1.0 if budget_diag.get("truncated_at_budget") else 0.0,
        },
    }


def _build_summary(results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        by_split[str(row.get("split") or "unknown")].append(row)
    split_summary: list[dict[str, Any]] = []
    total_exact = 0
    total_renderable = 0
    total_materialized_exact = 0
    total_svg_exact = 0
    total_budget_truncated = 0
    for split_name, rows in by_split.items():
        exact = sum(1 for row in rows if row.get("exact_match"))
        renderable = sum(1 for row in rows if row.get("renderable"))
        materialized_exact = sum(1 for row in rows if row.get("materialized_exact_match"))
        svg_exact = sum(1 for row in rows if row.get("svg_exact_match"))
        budget_truncated = sum(1 for row in rows if row.get("truncated_at_budget"))
        total_exact += exact
        total_renderable += renderable
        total_materialized_exact += materialized_exact
        total_svg_exact += svg_exact
        total_budget_truncated += budget_truncated
        split_summary.append(
            {
                "split": split_name,
                "count": len(rows),
                "exact_rate": exact / len(rows) if rows else 0.0,
                "renderable_rate": renderable / len(rows) if rows else 0.0,
                "materialized_exact_rate": materialized_exact / len(rows) if rows else 0.0,
                "svg_exact_rate": svg_exact / len(rows) if rows else 0.0,
                "budget_truncation_rate": budget_truncated / len(rows) if rows else 0.0,
            }
        )
    split_summary.sort(key=lambda row: str(row.get("split") or ""))
    totals = {
        "count": len(results),
        "exact_rate": total_exact / len(results) if results else 0.0,
        "renderable_rate": total_renderable / len(results) if results else 0.0,
        "materialized_exact_rate": total_materialized_exact / len(results) if results else 0.0,
        "svg_exact_rate": total_svg_exact / len(results) if results else 0.0,
        "budget_truncation_rate": total_budget_truncated / len(results) if results else 0.0,
    }
    return split_summary, totals


def _pct(value: Any) -> str:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{num * 100:.0f}%"


def _metric_class(value: Any) -> str:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "mid"
    if num >= 0.9:
        return "good"
    if num >= 0.5:
        return "mid"
    return "bad"


def _case_card(row: dict[str, Any]) -> str:
    actual_preview = _build_preview_markup(
        row.get("materialized_output"),
        row.get("materialized_mime"),
        str(row.get("render_error") or "No previewable artifact"),
    )
    expected_preview = _build_preview_markup(
        row.get("expected_rendered_output"),
        row.get("expected_rendered_mime"),
        "No expected preview",
    )
    chips = [
        f'<span class="chip {_metric_class(1.0 if row.get("exact_match") else 0.0)}">{"Exact" if row.get("exact_match") else "Drift"}</span>',
        f'<span class="chip {_metric_class(1.0 if row.get("renderable") else 0.0)}">{"Previewable" if row.get("renderable") else "No preview"}</span>',
        f'<span class="chip {_metric_class(1.0 if row.get("materialized_exact_match") else 0.0)}">{"Final exact" if row.get("materialized_exact_match") else "Final drift"}</span>',
        f'<span class="chip split">{_html(row.get("split"))}</span>',
    ]
    if row.get("truncated_at_budget"):
        chips.append('<span class="chip mid">Budget truncation</span>')
    if row.get("tail_text"):
        chips.append('<span class="chip mid">Tail drift</span>')
    if row.get("prefix_text"):
        chips.append('<span class="chip mid">Prefix drift</span>')
    return f"""
    <article class="probe-card">
      <div class="probe-top">
        <div>
          <div class="probe-label">{_html(row.get("label"))}</div>
          <div class="probe-prompt">{_html(row.get("prompt"))}</div>
        </div>
        <div class="chips">{''.join(chips)}</div>
      </div>
      <div class="preview-grid">
        <div class="preview-cell">
          <div class="preview-title">Actual Final Output</div>
          <div class="svg-frame">{actual_preview}</div>
        </div>
        <div class="preview-cell">
          <div class="preview-title">Expected Final Output</div>
          <div class="svg-frame">{expected_preview}</div>
        </div>
      </div>
      <div class="text-grid">
        <div>
          <div class="text-title">Actual Output</div>
          <pre>{_html(row.get("parsed_output") or row.get("response_text") or "")}</pre>
        </div>
        <div>
          <div class="text-title">Expected Output</div>
          <pre>{_html(row.get("expected_output") or "—")}</pre>
        </div>
      </div>
      <div class="text-grid materialized-grid">
        <div>
          <div class="text-title">Actual Materialized</div>
          <pre>{_html(row.get("materialized_output") or "—")}</pre>
        </div>
        <div>
          <div class="text-title">Expected Materialized</div>
          <pre>{_html(row.get("expected_rendered_output") or "—")}</pre>
        </div>
      </div>
      <div class="text-grid materialized-grid">
        <div>
          <div class="text-title">Budget Diagnostics</div>
          <pre>{_html(json.dumps({
              "prompt_tokens": row.get("prompt_tokens"),
              "parsed_output_tokens": row.get("parsed_output_tokens"),
              "expected_output_tokens": row.get("expected_output_tokens"),
              "decode_max_tokens": row.get("decode_max_tokens"),
              "context_len": row.get("context_len"),
              "hit_decode_ceiling": row.get("hit_decode_ceiling"),
              "hit_context_ceiling": row.get("hit_context_ceiling"),
              "missing_stop_marker": row.get("missing_stop_marker"),
              "prefix_matches_expected": row.get("prefix_matches_expected"),
              "truncated_at_budget": row.get("truncated_at_budget"),
              "truncation_reason": row.get("truncation_reason"),
          }, indent=2))}</pre>
        </div>
        <div>
          <div class="text-title">Render Diagnostic</div>
          <pre>{_html(row.get("render_error") or "—")}</pre>
        </div>
      </div>
      <div class="text-grid materialized-grid">
        <div>
          <div class="text-title">Bound content.json</div>
          <pre>{_html(json.dumps(row.get("content_json"), indent=2) if isinstance(row.get("content_json"), dict) else "—")}</pre>
        </div>
        <div>
          <div class="text-title">Compiler Read</div>
          <pre>{_html("The compiler receives the parsed scene DSL plus the bound content.json payload for this case.")}</pre>
        </div>
      </div>
    </article>
    """


def _build_html(report: dict[str, Any]) -> str:
    split_cards = "".join(
        f"""
        <article class="summary-card">
          <div class="k">{_html(row.get("split"))}</div>
          <div class="v">{_pct(row.get("exact_rate"))}</div>
          <div class="n">
            raw exact · {_pct(row.get("materialized_exact_rate"))} final exact · {_pct(row.get("renderable_rate"))} previewable · {_pct(row.get("budget_truncation_rate"))} budget-truncated · {_html(row.get("count"))} cases
          </div>
        </article>
        """
        for row in report.get("split_summary") or []
    )
    sections = []
    ordered_splits = [str(row.get("split")) for row in report.get("split_summary") or []]
    for split_name in ordered_splits:
        rows = [row for row in report.get("results") or [] if str(row.get("split")) == split_name]
        sections.append(
            f"""
            <section class="panel">
              <h2>{_html(split_name.title())}</h2>
              <div class="probe-stack">{''.join(_case_card(row) for row in rows)}</div>
            </section>
            """
        )
    title = str(report.get("title") or report.get("run_name") or "Probe Report")
    subtitle = (
        "Split-aware prompts, raw-target checks, and adapter-backed final artifact previews. "
        "The builder stays generic; the contract declares how outputs are parsed and materialized."
    )
    totals = report.get("totals") or {}
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_html(title)}</title>
  <style>
    :root {{
      --bg: #0b0d12;
      --panel: rgba(255,255,255,0.05);
      --border: rgba(255,255,255,0.10);
      --text: #eef2f7;
      --muted: #98a2b3;
      --good: #39d98a;
      --bad: #ff7b72;
      --mid: #ffb020;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Space Grotesk", "Segoe UI", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(122,162,255,0.14), transparent 28%),
        radial-gradient(circle at top right, rgba(57,217,138,0.10), transparent 24%),
        linear-gradient(180deg, #11141b 0%, #0b0d12 100%);
    }}
    .page {{ width: min(1480px, calc(100vw - 40px)); margin: 22px auto 40px; }}
    .hero, .panel, .summary-card {{ border: 1px solid var(--border); border-radius: 22px; background: var(--panel); box-shadow: 0 24px 60px rgba(0,0,0,0.28); backdrop-filter: blur(10px); }}
    .hero {{ padding: 28px 30px; margin-bottom: 20px; display: grid; gap: 18px; }}
    .eyebrow {{ display: inline-block; padding: 6px 10px; border-radius: 999px; background: rgba(122,162,255,0.16); color: #bfd1ff; text-transform: uppercase; letter-spacing: 0.08em; font-size: 12px; font-weight: 700; }}
    h1 {{ margin: 10px 0 8px; font-size: 38px; line-height: 1.05; }}
    h2 {{ margin: 0 0 10px; font-size: 24px; }}
    p, .meta, .n {{ color: var(--muted); line-height: 1.6; }}
    .meta {{ font-size: 14px; }}
    .summary-strip {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; }}
    .summary-card {{ padding: 16px; }}
    .summary-card .k, .preview-title, .text-title {{ color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; font-size: 12px; font-weight: 700; }}
    .summary-card .v {{ margin-top: 8px; font-size: 28px; font-weight: 800; }}
    .panel {{ padding: 24px 26px; margin-top: 20px; }}
    .probe-stack {{ display: grid; gap: 18px; }}
    .probe-card {{ border: 1px solid rgba(255,255,255,0.08); border-radius: 18px; background: rgba(255,255,255,0.035); padding: 18px; }}
    .probe-top {{ display: flex; justify-content: space-between; gap: 16px; align-items: flex-start; margin-bottom: 14px; }}
    .probe-label {{ font-size: 18px; font-weight: 800; margin-bottom: 6px; }}
    .probe-prompt {{ color: #d4def5; font-family: ui-monospace, monospace; font-size: 13px; line-height: 1.6; word-break: break-word; }}
    .chips {{ display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }}
    .chip {{ display: inline-block; padding: 6px 10px; border-radius: 999px; font-size: 12px; font-weight: 800; letter-spacing: 0.04em; text-transform: uppercase; }}
    .chip.good {{ background: rgba(57,217,138,0.16); color: #b7f3d4; }}
    .chip.mid {{ background: rgba(255,176,32,0.16); color: #ffd48f; }}
    .chip.bad {{ background: rgba(255,123,114,0.16); color: #ffc1bb; }}
    .chip.split {{ background: rgba(122,162,255,0.16); color: #bfd1ff; }}
    .preview-grid, .text-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }}
    .preview-grid {{ margin-bottom: 14px; }}
    .preview-cell {{ border: 1px solid rgba(255,255,255,0.08); background: rgba(255,255,255,0.03); border-radius: 14px; padding: 12px; }}
    .svg-frame {{ min-height: 160px; display: flex; align-items: center; justify-content: center; border-radius: 12px; background: #f8fafc; overflow: hidden; padding: 10px; }}
    .svg-frame svg {{ width: 100%; height: auto; max-width: 320px; display: block; }}
    .empty-preview {{ color: #475569; font-size: 13px; text-align: center; }}
    .materialized-grid {{ margin-top: 14px; }}
    pre {{ margin: 0; white-space: pre-wrap; word-break: break-word; min-height: 96px; border-radius: 12px; padding: 12px; background: #0b1220; color: #d8e4ff; font-family: ui-monospace, monospace; font-size: 12px; line-height: 1.55; border: 1px solid rgba(255,255,255,0.08); }}
    @media (max-width: 980px) {{
      .preview-grid, .text-grid {{ grid-template-columns: 1fr; }}
      .probe-top {{ flex-direction: column; }}
      .chips {{ justify-content: flex-start; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div>
        <span class="eyebrow">Generic Probe Report</span>
        <h1>{_html(title)}</h1>
        <p>{_html(subtitle)}</p>
        <div class="meta">Run: {_html(report.get("run_dir"))} | Output adapter: {_html(report.get("output_adapter"))}</div>
        <div class="meta">Contract: {_html(report.get("contract_source"))}</div>
      </div>
      <div class="summary-strip">
        <article class="summary-card"><div class="k">Exact</div><div class="v">{_pct(totals.get("exact_rate"))}</div><div class="n">{_html(totals.get("count"))} total cases</div></article>
        <article class="summary-card"><div class="k">Previewable</div><div class="v">{_pct(totals.get("renderable_rate"))}</div><div class="n">adapter-backed final artifact preview</div></article>
        <article class="summary-card"><div class="k">Final Exact</div><div class="v">{_pct(totals.get("materialized_exact_rate"))}</div><div class="n">materialized output vs expected final artifact</div></article>
        <article class="summary-card"><div class="k">Budget Truncated</div><div class="v">{_pct(totals.get("budget_truncation_rate"))}</div><div class="n">cases that hit decode/context ceiling before stop marker</div></article>
        {split_cards}
      </div>
    </section>
    {''.join(sections)}
  </div>
</body>
</html>"""


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a generic split-aware probe report for a run")
    ap.add_argument("--run", required=True, help="Run directory containing a built model")
    ap.add_argument("--contract", default=None, help="Optional explicit probe report contract JSON")
    ap.add_argument("--output", default=None, help="Optional HTML output path (default: <run>/probe_report.html)")
    ap.add_argument("--json-out", default=None, help="Optional JSON output path (default: <run>/probe_report.json)")
    ap.add_argument("--max-per-split", type=int, default=None, help="Optional override for per-split case limit")
    args = ap.parse_args()

    run_dir = Path(args.run).expanduser().resolve()
    if not run_dir.exists():
        raise SystemExit(f"run dir not found: {run_dir}")
    model_dir = run_dir / ".ck_build"
    if not model_dir.exists():
        raise SystemExit(f"model dir not found: {model_dir}")

    contract, contract_source = _autodiscover_contract(run_dir, args.contract)
    if str(contract.get("schema") or "").strip() != CONTRACT_SCHEMA:
        raise SystemExit(f"unsupported contract schema: {contract.get('schema')!r}")
    contract_path = Path(contract_source).expanduser().resolve()
    catalog = _load_catalog(contract, contract_path)
    cases = _load_cases(contract, contract_path, catalog, args.max_per_split)

    output_adapter_cfg = _resolve_output_adapter(contract)
    output_adapter = str(output_adapter_cfg.get("name") or "plain_text").strip()
    decode_cfg = contract.get("decode") if isinstance(contract.get("decode"), dict) else {}
    budget_analyzer = _TokenBudgetAnalyzer.for_run(run_dir, decode_cfg)

    results: list[dict[str, Any]] = []
    for case in cases:
        raw_output = _run_prompt(model_dir, str(case["prompt"]), decode_cfg, output_adapter_cfg)
        response_text = extract_response_text(raw_output, str(case["prompt"]))
        results.append(_evaluate_case(case, raw_output, response_text, output_adapter_cfg, budget_analyzer=budget_analyzer))

    split_summary, totals = _build_summary(results)
    report = {
        "schema": REPORT_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "model_dir": str(model_dir),
        "contract_source": str(contract_source),
        "dataset_type": contract.get("dataset_type"),
        "output_adapter": output_adapter,
        "output_adapter_config": output_adapter_cfg,
        "title": contract.get("title") or f"{run_dir.name} Probe Report",
        "split_summary": split_summary,
        "totals": totals,
        "results": results,
    }

    json_out = Path(args.json_out).expanduser().resolve() if args.json_out else (run_dir / "probe_report.json")
    html_out = Path(args.output).expanduser().resolve() if args.output else (run_dir / "probe_report.html")
    json_out.parent.mkdir(parents=True, exist_ok=True)
    html_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    html_out.write_text(_build_html(report), encoding="utf-8")
    print(json_out)
    print(html_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
