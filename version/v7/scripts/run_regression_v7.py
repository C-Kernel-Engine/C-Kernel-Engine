#!/usr/bin/env python3
"""
Model-family regression runner for v7 inference bring-up.

The runner is intentionally data-driven:
- family-specific runtime arguments live in version/v7/regression/families.json
- prompt/coherence expectations live in version/v7/regression/prompts.json

Fast mode:
- build/runtime
- smoke prompts
- coherence scoring
- on failure, quick first-token parity for triage

Full mode:
- fast mode
- on smoke/coherence failure: stitch audit via ir_reverse_validator.py
- on clean stitch path: kernel selection audit
- on clean stitch/kernel path: first-divergence hidden-state trace
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
V7_ROOT = ROOT / "version" / "v7"
REGRESSION_DIR = V7_ROOT / "regression"
CKS_RUN = V7_ROOT / "scripts" / "cks-v7-run"
CK_DUMP_TOKENS = V7_ROOT / "scripts" / "parity" / "ck_dump_tokens.py"
FIRST_TOKEN_PARITY = V7_ROOT / "scripts" / "parity" / "compare_first_token_logits.py"
FIRST_DIVERGENCE = V7_ROOT / "scripts" / "parity" / "check_sequence_hidden_state_vs_llama.py"
IR_REVERSE_VALIDATOR = V7_ROOT / "scripts" / "ir_reverse_validator.py"
KERNEL_REGISTRY = V7_ROOT / "kernel_maps" / "KERNEL_REGISTRY.json"
KERNEL_MAPS_DIR = V7_ROOT / "kernel_maps"

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


@dataclass(frozen=True)
class PromptSpec:
    prompt_id: str
    label: str
    prompt: str
    max_tokens: int
    heuristics: dict[str, Any]


@dataclass(frozen=True)
class FamilySpec:
    family_id: str
    label: str
    model: str
    context_len: int
    runtime_args: list[str]
    smoke_prompts: list[str]
    parity: dict[str, Any]
    response_contract: dict[str, Any]
    coherence_gate: bool
    runtime_expect: dict[str, Any]


def _cache_root() -> Path:
    env = os.environ.get("CK_CACHE_DIR")
    if env:
        base = Path(env).expanduser()
        if base.name == "train":
            return base.parent.parent
        if base.name == "models":
            return base.parent
        return base
    return Path.home() / ".cache" / "ck-engine-v7"


def _default_run_root() -> Path:
    return _cache_root() / "regression" / "runs"


def _default_report_root() -> Path:
    return _cache_root() / "regression" / "reports"


def _run(cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd or ROOT),
        text=True,
        capture_output=True,
        check=False,
    )


def _run_stream(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    prefix: str = "",
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd or ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    captured: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        captured.append(line)
        if prefix:
            print(f"{prefix}{line}", end="", flush=True)
        else:
            print(line, end="", flush=True)
    rc = proc.wait()
    return subprocess.CompletedProcess(cmd, rc, "".join(captured), "")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_prompts(path: Path) -> dict[str, PromptSpec]:
    doc = _load_json(path)
    rows = doc.get("prompts")
    if not isinstance(rows, dict):
        raise ValueError(f"prompts manifest must contain a 'prompts' object: {path}")
    out: dict[str, PromptSpec] = {}
    for prompt_id, row in rows.items():
        if not isinstance(row, dict):
            raise ValueError(f"prompt entry must be an object: {prompt_id}")
        out[str(prompt_id)] = PromptSpec(
            prompt_id=str(prompt_id),
            label=str(row.get("label") or prompt_id),
            prompt=str(row.get("prompt") or ""),
            max_tokens=int(row.get("max_tokens") or 96),
            heuristics=dict(row.get("heuristics") or {}),
        )
    return out


def load_families(path: Path, prompts: dict[str, PromptSpec]) -> list[FamilySpec]:
    doc = _load_json(path)
    rows = doc.get("families")
    if not isinstance(rows, list):
        raise ValueError(f"family manifest must contain a 'families' list: {path}")
    out: list[FamilySpec] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("family entry must be an object")
        if not bool(row.get("enabled", True)):
            continue
        family_id = str(row.get("id") or "").strip()
        if not family_id:
            raise ValueError("family entry missing id")
        if family_id in seen:
            raise ValueError(f"duplicate family id: {family_id}")
        seen.add(family_id)
        model = str(row.get("model") or "").strip()
        if not model:
            raise ValueError(f"family {family_id} missing model")
        smoke_prompts = [str(x) for x in row.get("smoke_prompts") or []]
        for prompt_id in smoke_prompts:
            if prompt_id not in prompts:
                raise ValueError(f"family {family_id} references unknown smoke prompt: {prompt_id}")
        parity = dict(row.get("parity") or {})
        parity_prompt_id = str(parity.get("prompt_id") or "").strip()
        if parity_prompt_id and parity_prompt_id not in prompts:
            raise ValueError(f"family {family_id} references unknown parity prompt: {parity_prompt_id}")
        out.append(
            FamilySpec(
                family_id=family_id,
                label=str(row.get("label") or family_id),
                model=model,
                context_len=int(row.get("context_len") or 1024),
                runtime_args=[str(x) for x in row.get("runtime_args") or []],
                smoke_prompts=smoke_prompts,
                parity=parity,
                response_contract=dict(row.get("response_contract") or {}),
                coherence_gate=bool(row.get("coherence_gate", True)),
                runtime_expect=dict(row.get("runtime_expect") or {}),
            )
        )
    return out


def _extract_assistant_output(stdout: str) -> str:
    text = ANSI_ESCAPE_RE.sub("", str(stdout or ""))
    pattern = re.compile(
        r"(?ms)(?:^|\n)(?:Assistant|Response):\s*(.*?)(?=\n\s*(?:prompt eval:|decode:|sample:|total:)|\nYou:|\Z)"
    )
    matches = list(pattern.finditer(text))
    if matches:
        return matches[-1].group(1).strip()
    return text.strip()


def _strip_think_blocks(text: str) -> str:
    out = re.sub(r"<think>\s*.*?\s*</think>\s*", "", text, flags=re.S | re.I)
    return re.sub(r"<think>\s*.*\Z", "", out, flags=re.S | re.I)


def normalize_assistant_output(text: str, response_contract: dict[str, Any]) -> str:
    out = ANSI_ESCAPE_RE.sub("", str(text or ""))
    contract = dict(response_contract or {})
    if bool(contract.get("strip_think_blocks")):
        out = _strip_think_blocks(out)

    for marker in contract.get("stop_text_markers") or []:
        marker_text = str(marker or "")
        if not marker_text:
            continue
        hit = out.find(marker_text)
        if hit >= 0:
            out = out[:hit]
            break

    if bool(contract.get("strip_trailing_metrics", True)):
        out = re.sub(r"\n(?:prompt eval:|decode:|sample:|total:).*$", "", out, flags=re.S)

    if bool(contract.get("trim_whitespace", True)):
        out = out.strip()
    return out


def _resolve_gguf_path(model_spec: str) -> Path | None:
    text = str(model_spec or "").strip()
    if not text:
        return None
    if text.startswith("hf://"):
        payload = text[len("hf://") :]
        if "/" not in payload:
            return None
        repo = payload.rsplit("/", 1)[0]
        filename = payload.rsplit("/", 1)[1]
        repo_dir = repo.replace("/", "--")
        roots = [
            _cache_root() / "models" / repo_dir,
            _cache_root() / repo_dir,
        ]
        for cache_dir in roots:
            candidate = cache_dir / filename
            if candidate.exists():
                return candidate
        return None
    candidate = Path(text).expanduser()
    return candidate if candidate.exists() else None


def _tokenize_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_+#.-]+", text.lower())


def _collect_keyword_hits(text: str, expected_keywords: list[str]) -> list[str]:
    text_lower = text.lower()
    token_set = {
        token.strip(".,!?;:()[]{}\"'`")
        for token in _tokenize_words(text)
        if token.strip(".,!?;:()[]{}\"'`")
    }
    hits: list[str] = []
    for kw in expected_keywords:
        kw_norm = str(kw).lower()
        if not kw_norm:
            continue
        if re.fullmatch(r"[a-z0-9_+#.-]+", kw_norm):
            if kw_norm in token_set:
                hits.append(kw_norm)
        elif kw_norm in text_lower:
            hits.append(kw_norm)
    return hits


def coherence_metrics(text: str, heuristics: dict[str, Any]) -> dict[str, Any]:
    text = str(text or "")
    stripped = text.strip()
    words = _tokenize_words(text)
    chars = len(text)
    printable = sum(1 for ch in text if ch.isprintable() or ch in "\n\t")
    printable_ratio = float(printable / chars) if chars else 0.0
    replacement_chars = text.count("\ufffd")

    ngram_counts: Counter[tuple[str, ...]] = Counter()
    if len(words) >= 4:
        ngram_counts.update(tuple(words[i : i + 4]) for i in range(len(words) - 3))
    repeated_4gram = max(ngram_counts.values(), default=0)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    duplicate_lines = max(0, len(lines) - len(set(lines)))
    line_count = len(lines)

    expected_keywords = [str(x).lower() for x in heuristics.get("expected_keywords") or []]
    keyword_hits = _collect_keyword_hits(text, expected_keywords)
    required_markers = [str(x) for x in heuristics.get("required_substrings_any_of") or []]
    required_marker_hits = [marker for marker in required_markers if marker and marker in text]

    score = 1.0
    if replacement_chars:
        score -= min(0.4, 0.15 * replacement_chars)
    if repeated_4gram > 1:
        score -= min(0.35, 0.10 * float(repeated_4gram - 1))
    if duplicate_lines:
        score -= min(0.2, 0.08 * float(duplicate_lines))
    score -= max(0.0, float(heuristics.get("min_printable_ratio", 0.0)) - printable_ratio)
    score = max(0.0, min(1.0, score))

    return {
        "text": text,
        "chars": chars,
        "words": len(words),
        "printable_ratio": printable_ratio,
        "replacement_chars": replacement_chars,
        "repeated_4gram": repeated_4gram,
        "duplicate_lines": duplicate_lines,
        "line_count": line_count,
        "keyword_hits": keyword_hits,
        "required_marker_hits": required_marker_hits,
        "score": score,
        "preview": stripped[:200],
    }


def assess_coherence(text: str, heuristics: dict[str, Any]) -> dict[str, Any]:
    metrics = coherence_metrics(text, heuristics)
    reasons: list[str] = []

    min_chars = int(heuristics.get("min_chars") or 0)
    min_words = int(heuristics.get("min_words") or 0)
    max_chars = int(heuristics.get("max_chars") or 0)
    max_words = int(heuristics.get("max_words") or 0)
    max_lines = int(heuristics.get("max_lines") or 0)
    min_printable_ratio = float(heuristics.get("min_printable_ratio") or 0.0)
    max_replacement_chars = int(heuristics.get("max_replacement_chars") or 0)
    max_repeated_4gram = int(heuristics.get("max_repeated_4gram") or 999999)
    max_duplicate_lines = int(heuristics.get("max_duplicate_lines") or 999999)
    min_keyword_hits = int(heuristics.get("min_keyword_hits") or 0)
    min_required_marker_hits = int(heuristics.get("min_required_substrings_any_of_hits") or 0)

    if metrics["chars"] < min_chars:
        reasons.append(f"too_short:{metrics['chars']}<{min_chars}")
    if metrics["words"] < min_words:
        reasons.append(f"too_few_words:{metrics['words']}<{min_words}")
    if max_chars and metrics["chars"] > max_chars:
        reasons.append(f"too_many_chars:{metrics['chars']}>{max_chars}")
    if max_words and metrics["words"] > max_words:
        reasons.append(f"too_many_words:{metrics['words']}>{max_words}")
    if max_lines and metrics["line_count"] > max_lines:
        reasons.append(f"too_many_lines:{metrics['line_count']}>{max_lines}")
    if metrics["printable_ratio"] < min_printable_ratio:
        reasons.append(f"printable_ratio:{metrics['printable_ratio']:.3f}<{min_printable_ratio:.3f}")
    if metrics["replacement_chars"] > max_replacement_chars:
        reasons.append(f"replacement_chars:{metrics['replacement_chars']}>{max_replacement_chars}")
    if metrics["repeated_4gram"] > max_repeated_4gram:
        reasons.append(f"repeated_4gram:{metrics['repeated_4gram']}>{max_repeated_4gram}")
    if metrics["duplicate_lines"] > max_duplicate_lines:
        reasons.append(f"duplicate_lines:{metrics['duplicate_lines']}>{max_duplicate_lines}")
    if len(metrics["keyword_hits"]) < min_keyword_hits:
        reasons.append(f"keyword_hits:{len(metrics['keyword_hits'])}<{min_keyword_hits}")
    if len(metrics["required_marker_hits"]) < min_required_marker_hits:
        reasons.append(
            f"required_markers:{len(metrics['required_marker_hits'])}<{min_required_marker_hits}"
        )

    return {
        "status": PASS if not reasons else FAIL,
        "metrics": metrics,
        "reasons": reasons,
    }


def _resolve_runtime_dir(run_dir: Path) -> Path:
    candidates = [run_dir / ".ck_build", run_dir / "ck_build", run_dir]
    for candidate in candidates:
        if (candidate / "libmodel.so").exists() and (candidate / "weights.bump").exists():
            return candidate
    return run_dir


def _resolve_artifact(run_dir: Path, runtime_dir: Path, name: str) -> Path | None:
    for base in (run_dir, runtime_dir):
        path = base / name
        if path.exists():
            return path
    return None


_MISSING = object()


def _lookup_path(doc: Any, dotted_path: str) -> Any:
    current = doc
    for part in str(dotted_path or "").split("."):
        if not part:
            continue
        if not isinstance(current, dict) or part not in current:
            return _MISSING
        current = current[part]
    return current


def _load_kernel_registry() -> set[str]:
    doc = _load_json(KERNEL_REGISTRY)
    kernels = doc.get("kernels")
    if not isinstance(kernels, list):
        return set()
    known: set[str] = set()
    for row in kernels:
        if not isinstance(row, dict):
            continue
        for value in (row.get("id"), row.get("name")):
            text = str(value or "").strip()
            if text:
                known.add(text)
        impl = row.get("impl")
        if isinstance(impl, dict):
            for key in ("function", "symbol"):
                text = str(impl.get(key) or "").strip()
                if text:
                    known.add(text)
    return known


def audit_kernel_selection(lowered_path: Path) -> dict[str, Any]:
    lowered = _load_json(lowered_path)
    operations = lowered.get("operations")
    if not isinstance(operations, list):
        return {"status": FAIL, "reason": "missing operations array", "unknown_kernels": [], "suspicious_kernels": []}

    known = _load_kernel_registry()
    allowed_builtins = {"memcpy"}
    unknown: list[str] = []
    suspicious: list[str] = []
    for op in operations:
        if not isinstance(op, dict):
            continue
        kernel = str(op.get("kernel") or op.get("function") or "").strip()
        if not kernel:
            unknown.append("<missing>")
            continue
        lowered_kernel = kernel.lower()
        if any(token in lowered_kernel for token in ("fallback", "todo", "stub", "placeholder")):
            suspicious.append(kernel)
        if kernel not in known and kernel not in allowed_builtins and not kernel.startswith("tokenizer_"):
            unknown.append(kernel)
    return {
        "status": PASS if not unknown and not suspicious else FAIL,
        "unknown_kernels": sorted(set(unknown)),
        "suspicious_kernels": sorted(set(suspicious)),
    }


def run_stitch_audit(run_dir: Path, runtime_dir: Path, report_path: Path) -> dict[str, Any]:
    lowered = _resolve_artifact(run_dir, runtime_dir, "lowered_decode_call.json")
    manifest = _resolve_artifact(run_dir, runtime_dir, "weights_manifest.json")
    if lowered is None:
        return {"status": SKIP, "reason": "lowered_decode_call.json not found"}
    print(f"[stitch] validating lowered IR for {run_dir.name}", flush=True)
    cmd = [
        sys.executable,
        str(IR_REVERSE_VALIDATOR),
        "--lowered",
        str(lowered),
        "--kernel-maps",
        str(KERNEL_MAPS_DIR),
        "--json",
    ]
    if manifest is not None:
        cmd.extend(["--manifest", str(manifest)])
    proc = _run(cmd)
    report = {"status": PASS if proc.returncode == 0 else FAIL, "stdout": proc.stdout, "stderr": proc.stderr}
    try:
        payload = json.loads(proc.stdout)
        report["checks"] = payload.get("checks", [])
        report["passed"] = bool(payload.get("passed", proc.returncode == 0))
    except Exception:
        report["checks"] = []
        report["passed"] = proc.returncode == 0
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run_token_dump(runtime_dir: Path, prompt: str) -> list[int]:
    cmd = [
        sys.executable,
        str(CK_DUMP_TOKENS),
        "--model-dir",
        str(runtime_dir),
        "--prompt",
        prompt,
    ]
    proc = _run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"ck_dump_tokens failed\n{proc.stdout}\n{proc.stderr}")
    match = re.search(r"\[CK\]\s+tokens\s+\(\d+\):\s+(\[[^\]]*\])", proc.stdout)
    if not match:
        raise RuntimeError(f"unable to parse token dump output: {proc.stdout}")
    values = ast.literal_eval(match.group(1))
    return [int(x) for x in values]


def run_first_token_parity(
    runtime_dir: Path,
    gguf_path: Path | None,
    parity_cfg: dict[str, Any],
    prompt: PromptSpec,
    report_path: Path,
) -> dict[str, Any]:
    tokens = run_token_dump(runtime_dir, prompt.prompt)
    print(f"[parity] first-token prompt={prompt.prompt_id} tokens={len(tokens)}", flush=True)
    cmd = [
        sys.executable,
        str(FIRST_TOKEN_PARITY),
        "--model-dir",
        str(runtime_dir),
        "--tokens",
        ",".join(str(x) for x in tokens),
        "--ctx-len",
        str(int(parity_cfg.get("ctx_len") or 256)),
        "--top-k",
        str(int(parity_cfg.get("top_k") or 16)),
        "--min-topk-overlap",
        str(float(parity_cfg.get("min_topk_overlap") or 0.5)),
        "--json-out",
        str(report_path),
    ]
    if gguf_path is not None:
        cmd.extend(["--gguf", str(gguf_path)])
    if bool(parity_cfg.get("require_top1_match", True)):
        cmd.append("--require-top1-match")
    else:
        cmd.append("--no-require-top1-match")
    proc = _run(cmd)
    if report_path.exists():
        payload = _load_json(report_path)
    else:
        payload = {"status": FAIL, "stdout": proc.stdout, "stderr": proc.stderr}
    payload.setdefault("status", PASS if proc.returncode == 0 else FAIL)
    if isinstance(payload.get("status"), str):
        payload["status"] = str(payload["status"]).upper()
    payload["returncode"] = proc.returncode
    payload["tokens"] = tokens
    return payload


def run_first_divergence(
    runtime_dir: Path,
    gguf_path: Path | None,
    parity_cfg: dict[str, Any],
    prompt: PromptSpec,
    report_path: Path,
) -> dict[str, Any]:
    tokens = run_token_dump(runtime_dir, prompt.prompt)
    print(f"[parity] divergence prompt={prompt.prompt_id} tokens={len(tokens)}", flush=True)
    cmd = [
        sys.executable,
        str(FIRST_DIVERGENCE),
        "--model-dir",
        str(runtime_dir),
        "--tokens",
        ",".join(str(x) for x in tokens),
        "--ctx-len",
        str(int(parity_cfg.get("ctx_len") or 256)),
        "--top-k",
        str(int(parity_cfg.get("top_k") or 8)),
        "--report-json",
        str(report_path),
    ]
    if gguf_path is not None:
        cmd.extend(["--gguf", str(gguf_path)])
    proc = _run(cmd)
    if report_path.exists():
        payload = _load_json(report_path)
    else:
        payload = {"all_ok": False, "stdout": proc.stdout, "stderr": proc.stderr}
    first_failure = payload.get("first_stable_failure") or payload.get("first_failure")
    payload["status"] = PASS if proc.returncode == 0 else FAIL
    payload["returncode"] = proc.returncode
    payload["summary"] = first_failure
    return payload


def run_prompt(
    family: FamilySpec,
    prompt: PromptSpec,
    run_dir: Path,
    *,
    force_rebuild: bool,
) -> dict[str, Any]:
    cached_gguf = _resolve_gguf_path(family.model)
    model_arg = str(cached_gguf) if cached_gguf is not None else family.model
    cmd = [
        str(CKS_RUN),
        "run",
        model_arg,
        "--run",
        str(run_dir),
        "--context-len",
        str(family.context_len),
        "--prompt",
        prompt.prompt,
        "--max-tokens",
        str(prompt.max_tokens),
    ]
    if force_rebuild:
        cmd.extend(["--force-convert", "--force-compile"])
    cmd.extend(family.runtime_args)
    print(f"[{family.family_id}] smoke prompt={prompt.prompt_id} run={run_dir}", flush=True)
    proc = _run_stream(cmd, prefix=f"[{family.family_id}] ")
    assistant_raw = _extract_assistant_output(proc.stdout)
    assistant = normalize_assistant_output(assistant_raw, family.response_contract)
    coherence = (
        assess_coherence(assistant, prompt.heuristics)
        if proc.returncode == 0
        else {"status": SKIP, "metrics": {}, "reasons": []}
    )
    return {
        "status": PASS if proc.returncode == 0 else FAIL,
        "command": cmd,
        "model_arg": model_arg,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "assistant_raw": assistant_raw,
        "assistant": assistant,
        "coherence": coherence,
    }


def audit_runtime_contract(
    run_dir: Path,
    runtime_dir: Path,
    prompt_rows: list[dict[str, Any]],
    expectations: dict[str, Any],
    report_path: Path,
) -> dict[str, Any]:
    expect = dict(expectations or {})
    if not expect:
        return {"status": SKIP}

    reasons: list[str] = []
    details: dict[str, Any] = {}
    stdout = "\n".join(str(row.get("stdout") or "") for row in prompt_rows)

    required_stdout = [str(x) for x in expect.get("stdout_contains") or []]
    forbidden_stdout = [str(x) for x in expect.get("stdout_not_contains") or []]
    for needle in required_stdout:
        if needle and needle not in stdout:
            reasons.append(f"stdout_missing:{needle}")
    for needle in forbidden_stdout:
        if needle and needle in stdout:
            reasons.append(f"stdout_forbidden:{needle}")

    config_checks = dict(expect.get("config") or {})
    manifest_checks = dict(expect.get("manifest") or {})
    lowered_checks = list(expect.get("lowered_ops") or [])

    config_path = _resolve_artifact(run_dir, runtime_dir, "config.json")
    manifest_path = _resolve_artifact(run_dir, runtime_dir, "weights_manifest.json")
    lowered_path = _resolve_artifact(run_dir, runtime_dir, "lowered_decode_call.json")

    if config_checks:
        if config_path is None:
            reasons.append("missing_artifact:config.json")
        else:
            config_doc = _load_json(config_path)
            details["config_path"] = str(config_path)
            for dotted_path, expected in config_checks.items():
                actual = _lookup_path(config_doc, str(dotted_path))
                if actual is _MISSING:
                    reasons.append(f"config_missing:{dotted_path}")
                elif actual != expected:
                    reasons.append(f"config_mismatch:{dotted_path}:{actual!r}!={expected!r}")

    if manifest_checks:
        if manifest_path is None:
            reasons.append("missing_artifact:weights_manifest.json")
        else:
            manifest_doc = _load_json(manifest_path)
            details["manifest_path"] = str(manifest_path)
            for dotted_path, expected in manifest_checks.items():
                actual = _lookup_path(manifest_doc, str(dotted_path))
                if actual is _MISSING:
                    reasons.append(f"manifest_missing:{dotted_path}")
                elif actual != expected:
                    reasons.append(f"manifest_mismatch:{dotted_path}:{actual!r}!={expected!r}")

    if lowered_checks:
        if lowered_path is None:
            reasons.append("missing_artifact:lowered_decode_call.json")
        else:
            lowered_doc = _load_json(lowered_path)
            operations = lowered_doc.get("operations")
            if not isinstance(operations, list):
                reasons.append("lowered_missing:operations")
            else:
                details["lowered_path"] = str(lowered_path)
                lowered_summary: list[dict[str, Any]] = []
                for row in lowered_checks:
                    if not isinstance(row, dict):
                        continue
                    op_name = str(row.get("op") or "").strip()
                    function_prefix = str(row.get("function_prefix") or "").strip()
                    min_matches = int(row.get("min_matches") or 1)
                    matches = [
                        str(op.get("function") or op.get("kernel") or "")
                        for op in operations
                        if str(op.get("op") or op.get("name") or "") == op_name
                    ]
                    lowered_summary.append({"op": op_name, "functions": sorted(set(matches))})
                    if len(matches) < min_matches:
                        reasons.append(f"lowered_missing_op:{op_name}:{len(matches)}<{min_matches}")
                        continue
                    if function_prefix:
                        bad = sorted({fn for fn in matches if not fn.startswith(function_prefix)})
                        if bad:
                            reasons.append(f"lowered_function_mismatch:{op_name}:{bad!r}")
                details["lowered_summary"] = lowered_summary

    result = {
        "status": PASS if not reasons else FAIL,
        "reasons": reasons,
        "expectations": expect,
        "stdout_contains": required_stdout,
        "stdout_not_contains": forbidden_stdout,
    }
    result.update(details)
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def classify_family_result(
    *,
    build_status: str,
    smoke_status: str,
    coherence_status: str,
    coherence_gate: bool,
    contract_result: dict[str, Any],
    stitch_result: dict[str, Any],
    kernel_result: dict[str, Any],
    first_token_result: dict[str, Any],
    divergence_result: dict[str, Any],
    failure_reason: str,
) -> tuple[str, str]:
    if build_status != PASS:
        return "build_failure", failure_reason or "build/runtime command failed"
    if smoke_status != PASS:
        return "smoke_failure", failure_reason or "prompt execution failed"
    if contract_result.get("status") == FAIL:
        return "contract_failure", "runtime contract/artifact audit failed"
    if stitch_result.get("status") == FAIL:
        return "stitch_failure", "lowered IR handoff validation failed"
    if kernel_result.get("status") == FAIL:
        return "kernel_selection_failure", "kernel registry/binding audit failed"
    if first_token_result.get("status") == FAIL or divergence_result.get("status") == FAIL:
        return "parity_divergence", "llama.cpp parity diverged"
    if coherence_gate and coherence_status != PASS:
        return "coherence_failure", failure_reason or "generated text failed coherence heuristics"
    return "pass", ""


def _display_rows(rows: list[tuple[str, ...]]) -> str:
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) for row in rows)


def render_terminal_matrix(summary: dict[str, Any]) -> str:
    rows: list[tuple[str, ...]] = [
        ("Family", "Build", "Smoke", "Coherence", "Contract", "Stitch", "Kernel", "1stTok", "Status/Class")
    ]
    for family in summary.get("families", []):
        rows.append(
            (
                str(family.get("family_id", "")),
                str(family.get("build_status", SKIP)),
                str(family.get("smoke_status", SKIP)),
                str(family.get("coherence_status", SKIP)),
                str(family.get("contract_status", SKIP)),
                str(family.get("stitch_status", SKIP)),
                str(family.get("kernel_status", SKIP)),
                str(family.get("first_token_status", SKIP)),
                f"{family.get('status', SKIP)}/{family.get('failure_class', 'pass')}",
            )
        )
    return _display_rows(rows)


def render_markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# v7 Regression Report",
        "",
        f"- mode: `{summary.get('mode')}`",
        f"- generated_at: `{summary.get('generated_at')}`",
        f"- overall: `{summary.get('status')}`",
        f"- failure_classes: `{summary.get('failure_classes')}`",
        "",
        "| Family | Build | Smoke | Coherence | Contract | Stitch | Kernel | 1stTok | Status |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for family in summary.get("families", []):
        lines.append(
            "| {family_id} | {build_status} | {smoke_status} | {coherence_status} | {contract_status} | {stitch_status} | {kernel_status} | {first_token_status} | {status} |".format(
                **family
            )
        )
    for family in summary.get("families", []):
        lines.extend(
            [
                "",
                f"## {family['family_id']}",
                "",
                f"- model: `{family['model']}`",
                f"- run_dir: `{family['run_dir']}`",
                f"- status: `{family['status']}`",
                f"- failure_class: `{family.get('failure_class', 'pass')}`",
                f"- contract_status: `{family.get('contract_status', SKIP)}`",
            ]
        )
        if family.get("failure_reason"):
            lines.append(f"- failure_reason: `{family['failure_reason']}`")
        if family.get("failure_detail"):
            lines.append(f"- failure_detail: `{family['failure_detail']}`")
        for prompt_row in family.get("prompts", []):
            coherence = prompt_row.get("coherence") or {}
            metrics = coherence.get("metrics") or {}
            lines.append(
                f"- prompt `{prompt_row['prompt_id']}`: smoke=`{prompt_row['status']}` coherence=`{coherence.get('status', SKIP)}` score=`{float(metrics.get('score', 0.0)):.2f}`"
            )
        parity = family.get("first_token")
        if isinstance(parity, dict) and parity:
            compare = parity.get("compare", {})
            if isinstance(compare, dict):
                lines.append(
                    f"- first-token parity: status=`{parity.get('status')}` cosine=`{compare.get('cosine')}` top1_match=`{compare.get('top1_match')}`"
                )
    return "\n".join(lines) + "\n"


def run_family(
    family: FamilySpec,
    prompts: dict[str, PromptSpec],
    *,
    mode: str,
    run_root: Path,
    report_dir: Path,
    force_rebuild: bool,
) -> dict[str, Any]:
    print(f"\n=== [{family.family_id}] {family.label} ===", flush=True)
    run_dir = run_root / family.family_id
    run_dir.mkdir(parents=True, exist_ok=True)
    family_report_dir = report_dir / family.family_id
    family_report_dir.mkdir(parents=True, exist_ok=True)

    prompt_rows: list[dict[str, Any]] = []
    build_status = PASS
    smoke_status = PASS
    coherence_status = PASS
    failure_reason = ""

    for idx, prompt_id in enumerate(family.smoke_prompts):
        prompt = prompts[prompt_id]
        row = run_prompt(family, prompt, run_dir, force_rebuild=bool(force_rebuild and idx == 0))
        row["prompt_id"] = prompt_id
        prompt_rows.append(row)
        if row["status"] != PASS:
            if idx == 0:
                build_status = FAIL
            smoke_status = FAIL
            failure_reason = f"smoke_failed:{prompt_id}:rc={row['returncode']}"
            break
        if row["coherence"]["status"] != PASS:
            coherence_status = FAIL
            reasons = ",".join(row["coherence"].get("reasons") or [])
            failure_reason = f"coherence_failed:{prompt_id}:{reasons}"

    runtime_dir = _resolve_runtime_dir(run_dir)
    contract_result: dict[str, Any] = {"status": SKIP}
    stitch_result: dict[str, Any] = {"status": SKIP}
    kernel_result: dict[str, Any] = {"status": SKIP}
    first_token_result: dict[str, Any] = {"status": SKIP}
    divergence_result: dict[str, Any] = {"status": SKIP}
    parity_cfg = dict(family.parity or {})
    always_run_parity = bool(parity_cfg.get("always_run"))
    if build_status == PASS and smoke_status == PASS:
        contract_result = audit_runtime_contract(
            run_dir,
            runtime_dir,
            prompt_rows,
            family.runtime_expect,
            family_report_dir / "contract_audit.json",
        )
    needs_debug = build_status == PASS and (
        smoke_status != PASS or (family.coherence_gate and coherence_status != PASS)
    )

    if mode == "full" and needs_debug:
        stitch_result = run_stitch_audit(run_dir, runtime_dir, family_report_dir / "stitch_audit.json")
        lowered = _resolve_artifact(run_dir, runtime_dir, "lowered_decode_call.json")
        if lowered is not None:
            kernel_result = audit_kernel_selection(lowered)
            (family_report_dir / "kernel_audit.json").write_text(json.dumps(kernel_result, indent=2), encoding="utf-8")

    parity_prompt_id = str(parity_cfg.get("prompt_id") or "").strip()
    kernel_path_clean = kernel_result.get("status") in {SKIP, PASS}
    stitch_path_clean = stitch_result.get("status") in {SKIP, PASS}
    if parity_prompt_id and (needs_debug or always_run_parity) and stitch_path_clean and kernel_path_clean:
        gguf_path = _resolve_gguf_path(family.model)
        first_token_result = run_first_token_parity(
            runtime_dir,
            gguf_path,
            parity_cfg,
            prompts[parity_prompt_id],
            family_report_dir / "first_token_parity.json",
        )
        if mode == "full" and first_token_result.get("status") != PASS:
            gguf_path = _resolve_gguf_path(family.model)
            divergence_result = run_first_divergence(
                runtime_dir,
                gguf_path,
                parity_cfg,
                prompts[parity_prompt_id],
                family_report_dir / "first_divergence.json",
            )

    failure_class, failure_detail = classify_family_result(
        build_status=build_status,
        smoke_status=smoke_status,
        coherence_status=coherence_status,
        coherence_gate=family.coherence_gate,
        contract_result=contract_result,
        stitch_result=stitch_result,
        kernel_result=kernel_result,
        first_token_result=first_token_result,
        divergence_result=divergence_result,
        failure_reason=failure_reason,
    )
    overall = PASS if failure_class == "pass" else FAIL

    family_result = {
        "family_id": family.family_id,
        "label": family.label,
        "model": family.model,
        "run_dir": str(run_dir),
        "runtime_dir": str(runtime_dir),
        "build_status": build_status,
        "smoke_status": smoke_status,
        "coherence_status": coherence_status,
        "contract_status": contract_result.get("status", SKIP),
        "stitch_status": stitch_result.get("status", SKIP),
        "kernel_status": kernel_result.get("status", SKIP),
        "first_token_status": first_token_result.get("status", SKIP),
        "divergence_status": divergence_result.get("status", SKIP),
        "failure_reason": failure_reason,
        "failure_class": failure_class,
        "failure_detail": failure_detail,
        "coherence_gate": family.coherence_gate,
        "prompts": prompt_rows,
        "contract_audit": contract_result,
        "stitch_audit": stitch_result,
        "kernel_audit": kernel_result,
        "first_token": first_token_result,
        "first_divergence": divergence_result,
        "status": overall,
    }
    (family_report_dir / "family_summary.json").write_text(json.dumps(family_result, indent=2), encoding="utf-8")
    return family_result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run v7 model-family regression checks")
    parser.add_argument("--mode", choices=["fast", "full"], default="fast")
    parser.add_argument("--family", action="append", default=[], help="Limit to one or more family ids")
    parser.add_argument("--families-manifest", type=Path, default=REGRESSION_DIR / "families.json")
    parser.add_argument("--prompts-manifest", type=Path, default=REGRESSION_DIR / "prompts.json")
    parser.add_argument("--run-root", type=Path, default=_default_run_root())
    parser.add_argument("--report-root", type=Path, default=_default_report_root())
    parser.add_argument("--force-rebuild", action="store_true", help="Force convert/compile on the first smoke prompt")
    args = parser.parse_args()

    prompts = load_prompts(args.prompts_manifest)
    families = load_families(args.families_manifest, prompts)
    selected = {str(x).strip() for x in args.family if str(x).strip()}
    if selected:
        families = [family for family in families if family.family_id in selected]
    if not families:
        raise SystemExit("no families selected")

    report_dir = args.report_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir.mkdir(parents=True, exist_ok=True)
    run_root = args.run_root
    run_root.mkdir(parents=True, exist_ok=True)

    family_results = [
        run_family(
            family,
            prompts,
            mode=args.mode,
            run_root=run_root,
            report_dir=report_dir,
            force_rebuild=bool(args.force_rebuild),
        )
        for family in families
    ]

    status = PASS if all(row.get("status") == PASS for row in family_results) else FAIL
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": args.mode,
        "status": status,
        "run_root": str(run_root),
        "report_dir": str(report_dir),
        "families": family_results,
        "failure_classes": dict(Counter(row.get("failure_class", "pass") for row in family_results)),
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (report_dir / "summary.md").write_text(render_markdown_report(summary), encoding="utf-8")

    print("=" * 72)
    print("C-Kernel-Engine v7 Regression")
    print("=" * 72)
    print(f"mode      : {args.mode}")
    print(f"run_root  : {run_root}")
    print(f"report_dir: {report_dir}")
    print("")
    print(render_terminal_matrix(summary))
    print("")
    print(f"overall   : {status}")
    print(f"summary   : {report_dir / 'summary.json'}")

    return 0 if status == PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
