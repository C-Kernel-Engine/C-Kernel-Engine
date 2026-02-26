#!/usr/bin/env python3
"""
Automated parity autopsy driver for v7.

Workflow (strict order, mixed-quant safe):
1) template-audit (template/kernel/IR-stitch sanity)
2) per-layer quantization + call-IR stitching checks
3) layer-by-layer qkv contract checks
4) (Optional) Run ck_run_v7.py with detailed llama.cpp parity dumps
5) Run parity_test.py --json and locate first meaningful divergence
6) Write JSON + markdown report
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = Path(__file__).resolve().parent
V7_SCRIPT_DIR = SCRIPT_DIR.parent
CK_RUN = V7_SCRIPT_DIR / "ck_run_v7.py"
PARITY_TEST = V7_SCRIPT_DIR / "parity_test.py"
PROJ_CHECK = SCRIPT_DIR / "check_qproj_contract.py"
LAYER0_QKV_CHECK = SCRIPT_DIR / "check_layer0_qkv_contract.py"
TOKEN_REPLAY_CHECK = SCRIPT_DIR / "compare_first_token_logits.py"
SUPPORTED_OPS = ("q_proj", "k_proj", "v_proj")
LAYER_WEIGHTED_OPS: tuple[tuple[str, str], ...] = (
    ("q_proj", "wq"),
    ("k_proj", "wk"),
    ("v_proj", "wv"),
    ("out_proj", "wo"),
    ("mlp_gate_up", "w1"),
    ("mlp_down", "w2"),
)
PARITY_DUMP_INTEGRITY_PATTERNS: tuple[str, ...] = (
    "Resyncing dump parse: bad header",
    "Could not decode header at offset",
    "CKDMP dtype/data-size mismatch",
    "Unexpected EOF reading",
)


FAMILY_PROFILE = {
    "gemma": {
        "parity_model": "gemma",
        "llama_filter": "inp_embd,attn_norm,attn_q,attn_k,attn_v,Qcur,Kcur,Vcur,__fattn__,kqv_out,ffn_norm,ffn_out,l_out",
    },
    "qwen2": {
        "parity_model": "qwen2",
        "llama_filter": "inp_embd,rms_norm,attn_q,attn_k,attn_v,Qcur,Kcur,Vcur,__fattn__,kqv_out,ffn_norm,ffn_out,l_out",
    },
    "qwen3": {
        "parity_model": "qwen3",
        "llama_filter": "inp_embd,rms_norm,attn_q,attn_k,attn_v,Qcur,Kcur,Vcur,__fattn__,kqv_out,ffn_norm,ffn_out,l_out",
    },
    "llama": {
        "parity_model": "llama",
        "llama_filter": "inp_embd,rms_norm,attn_q,attn_k,attn_v,Qcur,Kcur,Vcur,__fattn__,kqv_out,ffn_norm,ffn_out,l_out",
    },
    "mistral": {
        "parity_model": "mistral",
        "llama_filter": "inp_embd,rms_norm,attn_q,attn_k,attn_v,Qcur,Kcur,Vcur,__fattn__,kqv_out,ffn_norm,ffn_out,l_out",
    },
}


def run_cmd(cmd: list[str], cwd: Path | None = None, check: bool = False) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd or ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(shlex.quote(x) for x in cmd)}\n"
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
    return proc


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy scalars/containers to plain Python JSON-safe values."""
    # Primitive fast path.
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    # Dict/list/tuple recursion.
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    # numpy scalar/array support without hard dependency at import time.
    if hasattr(obj, "item") and callable(getattr(obj, "item", None)):
        try:
            return to_jsonable(obj.item())
        except Exception:
            pass
    if hasattr(obj, "tolist") and callable(getattr(obj, "tolist", None)):
        try:
            return to_jsonable(obj.tolist())
        except Exception:
            pass
    # Fallback string.
    return str(obj)


def run_parity_direct(
    ck_dump: Path,
    ref_dump: Path,
    model_family: str,
    pass_name: str,
) -> tuple[int, list[dict[str, Any]]]:
    """Call parity_test.run_parity_test directly (avoids --json serialization bugs)."""
    spec = importlib.util.spec_from_file_location("ck_v7_parity_test", str(PARITY_TEST))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed loading module from {PARITY_TEST}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    exit_code, results = mod.run_parity_test(
        ck_dump_path=ck_dump,
        ref_dump_path=ref_dump,
        atol=1e-4,
        rtol=1e-3,
        verbose=False,
        model_family=model_family,
        pass_filter=pass_name,
    )
    # Keep as Python-native structures for reporting.
    return int(exit_code), to_jsonable(results)


def parse_json_blob(text: str) -> Any:
    text = text.strip()
    if not text:
        return None
    # Fast path.
    try:
        return json.loads(text)
    except Exception:
        pass
    # Find first JSON array/object in noisy output.
    m = re.search(r"(\[\s*\]|\[\s*\{.*\}\s*\]|\{.*\})", text, flags=re.S)
    if not m:
        return None
    return json.loads(m.group(1))


def detect_dump_integrity_issues(text: str) -> list[str]:
    issues: list[str] = []
    if not text:
        return issues
    for pat in PARITY_DUMP_INTEGRITY_PATTERNS:
        if pat in text:
            issues.append(pat)
    return issues


def audit_ref_dump_tokens(index_path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "ok": False,
        "message": "",
        "entries": 0,
        "unique_token_count": 0,
        "unique_tokens_head": [],
    }
    if not index_path.exists():
        out["message"] = f"missing {index_path.name}"
        return out
    try:
        obj = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception as e:
        out["message"] = f"index parse error: {e}"
        return out
    if not isinstance(obj, list):
        out["message"] = "index is not a list"
        return out
    token_ids: list[int] = []
    for row in obj:
        if not isinstance(row, dict):
            continue
        if "token_id" not in row:
            continue
        try:
            token_ids.append(int(row.get("token_id", 0)))
        except Exception:
            continue
    if not token_ids:
        out["message"] = "index has no token_id entries"
        return out
    uniq = sorted(set(token_ids))
    out["entries"] = len(token_ids)
    out["unique_token_count"] = len(uniq)
    out["unique_tokens_head"] = uniq[:16]
    collapsed_zero = len(token_ids) >= 8 and len(uniq) == 1 and uniq[0] == 0
    if collapsed_zero:
        out["message"] = f"collapsed token ids (all zero across {len(token_ids)} dumps)"
        out["ok"] = False
        return out
    out["ok"] = True
    out["message"] = f"entries={len(token_ids)} unique_tokens={uniq[:16]}"
    return out


def _parse_tokens_csv_strict(text: str) -> list[int]:
    vals: list[int] = []
    for part in str(text or "").split(","):
        p = part.strip()
        if not p:
            continue
        vals.append(int(p))
    if not vals:
        raise ValueError("token CSV is empty")
    return vals


def _token_replay_tokens_from_index(index_path: Path, max_unique: int) -> dict[str, Any]:
    out: dict[str, Any] = {
        "ok": False,
        "reason": "",
        "tokens": [],
        "index_path": str(index_path),
    }
    if max_unique <= 0:
        out["reason"] = "max_unique must be > 0"
        return out
    if not index_path.exists():
        out["reason"] = "index.json missing"
        return out
    try:
        obj = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception as e:
        out["reason"] = f"index parse error: {e}"
        return out
    if not isinstance(obj, list):
        out["reason"] = "index is not a list"
        return out

    tokens: list[int] = []
    seen: set[int] = set()
    for row in obj:
        if not isinstance(row, dict):
            continue
        if "token_id" not in row:
            continue
        try:
            tok = int(row.get("token_id", 0))
        except Exception:
            continue
        if tok in seen:
            continue
        seen.add(tok)
        tokens.append(tok)
        if len(tokens) >= int(max_unique):
            break

    if not tokens:
        out["reason"] = "no token_id entries in index"
        return out
    out["ok"] = True
    out["tokens"] = tokens
    out["reason"] = "token_id list extracted from index"
    return out


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_weight_refs_from_call_op(op: dict[str, Any]) -> set[str]:
    refs: set[str] = set()

    def _push(value: Any) -> None:
        if isinstance(value, str):
            v = value.strip()
            if v:
                refs.add(v)

    _push(op.get("weight_ref"))
    nested = op.get("weight_refs")
    if isinstance(nested, list):
        for value in nested:
            _push(value)

    args = op.get("args")
    arg_items: list[Any]
    if isinstance(args, list):
        arg_items = args
    elif isinstance(args, dict):
        arg_items = [args]
    else:
        arg_items = []
    for item in arg_items:
        if not isinstance(item, dict):
            continue
        src = item.get("source")
        if isinstance(src, str) and ":" in src:
            prefix, value = src.split(":", 1)
            if prefix.strip().lower() in {"weight", "manifest", "weight_f"}:
                _push(value)
        _push(item.get("weight_ref"))
        inner = item.get("weight_refs")
        if isinstance(inner, list):
            for value in inner:
                _push(value)
    return refs


def load_op_order(model_dir: Path, pass_name: str) -> dict[tuple[int, str], int]:
    lowered = model_dir / f"lowered_{pass_name}.json"
    if not lowered.exists():
        lowered = model_dir / "lowered_decode.json"
    if not lowered.exists():
        return {}
    try:
        obj = json.loads(lowered.read_text(encoding="utf-8"))
    except Exception:
        return {}
    ops = obj.get("operations", [])
    order: dict[tuple[int, str], int] = {}
    for i, op in enumerate(ops):
        layer = int(op.get("layer", -1))
        name = str(op.get("op", ""))
        key = (layer, name)
        if key not in order:
            order[key] = i
    return order


def _is_alignment_ambiguous_issue(row: dict[str, Any]) -> bool:
    if bool(row.get("alignment_ambiguous")):
        return True
    ck_n = row.get("ck_candidates")
    ref_n = row.get("ref_candidates")
    try:
        if ck_n is not None and int(ck_n) > 1:
            return True
        if ref_n is not None and int(ref_n) > 1:
            return True
    except Exception:
        pass
    return False


def pick_first_issue(
    results: list[dict[str, Any]],
    order: dict[tuple[int, str], int],
    *,
    prefer_unambiguous: bool = True,
) -> dict[str, Any] | None:
    bad = [r for r in results if r.get("status") in ("FAIL", "ERROR")]
    if not bad:
        return None

    def rank(r: dict[str, Any]) -> tuple[int, int, int]:
        layer = int(r.get("layer", 10**9))
        op = str(r.get("op", ""))
        seq = order.get((layer, op), 10**9)
        token = int(r.get("token", 10**9))
        return (layer, seq, token)
    ranked = sorted(bad, key=rank)
    raw_first = ranked[0]
    if not prefer_unambiguous:
        return raw_first
    if str(raw_first.get("op")) == "token_embedding" and _is_alignment_ambiguous_issue(raw_first):
        for row in ranked[1:]:
            if not _is_alignment_ambiguous_issue(row):
                promoted = dict(row)
                promoted["promoted_after_ambiguous_token_embedding"] = True
                promoted["promotion_reason"] = "next_unambiguous_fail"
                promoted["raw_first_issue"] = {
                    "layer": raw_first.get("layer"),
                    "op": raw_first.get("op"),
                    "token": raw_first.get("token"),
                    "status": raw_first.get("status"),
                }
                return promoted
        for row in ranked[1:]:
            if str(row.get("op")) != "token_embedding":
                promoted = dict(row)
                promoted["promoted_after_ambiguous_token_embedding"] = True
                promoted["promotion_reason"] = "next_non_token_embedding_fail"
                promoted["raw_first_issue"] = {
                    "layer": raw_first.get("layer"),
                    "op": raw_first.get("op"),
                    "token": raw_first.get("token"),
                    "status": raw_first.get("status"),
                }
                return promoted
    return raw_first


def pick_layer_first_issues(
    results: list[dict[str, Any]],
    order: dict[tuple[int, str], int],
) -> list[dict[str, Any]]:
    bad = [r for r in results if r.get("status") in ("FAIL", "ERROR")]
    if not bad:
        return []

    def rank(r: dict[str, Any]) -> tuple[int, int]:
        layer = int(r.get("layer", 10**9))
        op = str(r.get("op", ""))
        seq = order.get((layer, op), 10**9)
        token = int(r.get("token", 10**9))
        return (seq, token)

    per_layer: dict[int, dict[str, Any]] = {}
    for row in bad:
        layer = int(row.get("layer", -1))
        cur = per_layer.get(layer)
        if cur is None or rank(row) < rank(cur):
            per_layer[layer] = row
    return [per_layer[k] for k in sorted(per_layer.keys())]


def parse_projection_metrics(output: str) -> dict[str, Any]:
    max_m = re.search(r"max_diff\s*:\s*([0-9.eE+\-]+)", output)
    mean_m = re.search(r"mean_diff\s*:\s*([0-9.eE+\-]+)", output)
    finite_m = re.search(r"y_ref:\s*finite=(\d+)/(\d+)", output)
    max_diff = float(max_m.group(1)) if max_m else None
    mean_diff = float(mean_m.group(1)) if mean_m else None
    finite = None
    if finite_m:
        finite = (int(finite_m.group(1)), int(finite_m.group(2)))
    return {
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "y_ref_finite": finite,
        "pass_contract": (max_diff is not None and max_diff < 1e-3),
    }


def _is_projection_checker_unsupported(stderr_tail: str) -> bool:
    return "unsupported projection dtype for this checker" in (stderr_tail or "")


def infer_model_dir(output_dir: Path | None, model_uri: str) -> Path:
    if output_dir is not None:
        return output_dir
    if model_uri.startswith("hf://"):
        repo = model_uri[len("hf://") :].rsplit("/", 1)[0]
        key = repo.replace("/", "--")
        return Path.home() / ".cache" / "ck-engine-v7" / "models" / key / "ck_build"
    p = Path(model_uri).expanduser().resolve()
    if p.is_dir():
        return p
    return p.parent / "ck_build"


def infer_family(model_uri: str, model_dir: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    for cfg_path in (
        model_dir / "config.json",
        model_dir.parent / "config.json",
    ):
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            except Exception:
                cfg = {}
            model_type = str(cfg.get("model_type", "")).strip().lower()
            if model_type in FAMILY_PROFILE:
                return model_type
            if model_type == "llama":
                return "llama"
    hay = f"{model_uri} {model_dir}".lower()
    if "qwen3" in hay:
        return "qwen3"
    if "qwen2" in hay or "qwen-" in hay:
        return "qwen2"
    if "gemma" in hay:
        return "gemma"
    if "mistral" in hay:
        return "mistral"
    if "llama" in hay or "nanbeige" in hay:
        return "llama"
    # Keep prior behavior as final fallback.
    return "gemma"

def infer_run_dir_from_output_dir(output_dir: Path | None) -> Path | None:
    """Map a provided output dir to ck_run_v7 --run dir.

    parity_autopsy accepts ck_build paths, while ck_run_v7 expects run dirs.
    """
    if output_dir is None:
        return None
    p = output_dir.expanduser().resolve()
    if p.name in {"ck_build", ".ck_build"}:
        return p.parent
    return p

def pick_artifact_root(model_dir: Path, run_dir: Path | None) -> Path:
    """Choose the directory that actually holds parity dump artifacts.

    Some pipelines emit artifacts at run root, others under ck_build/.ck_build.
    """
    candidates: list[Path] = []
    for c in (model_dir, run_dir, model_dir.parent):
        if c is None:
            continue
        c = c.expanduser().resolve()
        if c not in candidates:
            candidates.append(c)
        # Fresh parity artifacts are frequently emitted under .ck_build/ck_build.
        for child in (c / ".ck_build", c / "ck_build"):
            if child not in candidates:
                candidates.append(child)

    def score(root: Path) -> int:
        s = 0
        for rel in ("ck_parity_dumps/dump.bin", "llama_parity_dumps/dump.bin"):
            p = root / rel
            if p.exists():
                s += 1
                try:
                    if p.stat().st_size > 0:
                        s += 4
                except OSError:
                    pass
        # Prefer token-aware llama references when available.
        idx = root / "llama_parity_dumps" / "index.json"
        if idx.exists():
            s += 2
            try:
                if idx.stat().st_size > 0:
                    s += 2
            except OSError:
                pass
        for rel in ("lowered_prefill.json", "lowered_decode.json", "weights_manifest.json"):
            if (root / rel).exists():
                s += 1
        return s

    best = model_dir
    best_score = -1
    for c in candidates:
        sc = score(c)
        if sc > best_score:
            best = c
            best_score = sc
    return best


def _template_audit_gate(model_input: str, run_dir: Path | None, context_len: int | None) -> dict[str, Any]:
    cmd = [sys.executable, str(CK_RUN), "template-audit", model_input]
    model_input_path: Path | None = None
    try:
        p = Path(model_input).expanduser()
        if p.exists():
            model_input_path = p.resolve()
    except Exception:
        model_input_path = None

    same_input_and_run = (
        run_dir is not None
        and model_input_path is not None
        and model_input_path == run_dir.expanduser().resolve()
    )
    if run_dir is not None and not same_input_and_run:
        cmd.extend(["--run", str(run_dir)])
    if context_len and int(context_len) > 0:
        cmd.extend(["--context-len", str(int(context_len))])
    proc = run_cmd(cmd, check=False)
    report_candidates: list[Path] = []
    if run_dir is not None:
        report_candidates.append(run_dir / "template_audit_latest.json")
        report_candidates.append(run_dir / ".ck_build" / "template_audit_latest.json")
    if model_input_path is not None and model_input_path.is_dir():
        report_candidates.append(model_input_path / "template_audit_latest.json")
        report_candidates.append(model_input_path / ".ck_build" / "template_audit_latest.json")
    existing_reports = [cand for cand in report_candidates if cand.exists()]
    report_path: Path | None = None
    if existing_reports:
        report_path = max(existing_reports, key=lambda p: p.stat().st_mtime)
    elif report_candidates:
        report_path = report_candidates[0]
    report_doc: dict[str, Any] | None = None
    if report_path and report_path.exists():
        try:
            report_doc = _load_json(report_path)
        except Exception:
            report_doc = None
    return {
        "command": cmd,
        "returncode": int(proc.returncode),
        "pass": bool(proc.returncode == 0),
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-1200:],
        "report_path": str(report_path) if report_path else None,
        "report_status": report_doc.get("status") if isinstance(report_doc, dict) else None,
        "report": report_doc,
    }


def _build_layer_index(ops: list[dict[str, Any]]) -> dict[int, dict[str, list[dict[str, Any]]]]:
    out: dict[int, dict[str, list[dict[str, Any]]]] = {}
    for op in ops:
        if not isinstance(op, dict):
            continue
        layer = op.get("layer")
        name = op.get("op")
        if not isinstance(layer, int) or layer < 0 or not isinstance(name, str):
            continue
        out.setdefault(layer, {}).setdefault(name, []).append(op)
    return out


def _per_layer_quant_stitch_gate(artifact_root: Path) -> dict[str, Any]:
    manifest_path = artifact_root / "weights_manifest.json"
    decode_path = artifact_root / "lowered_decode_call.json"
    prefill_path = artifact_root / "lowered_prefill_call.json"
    missing = [str(p) for p in (manifest_path, decode_path, prefill_path) if not p.exists()]
    if missing:
        return {
            "pass": False,
            "error": "missing required artifacts",
            "missing": missing,
        }

    manifest = _load_json(manifest_path)
    decode_doc = _load_json(decode_path)
    prefill_doc = _load_json(prefill_path)
    qsum = manifest.get("quant_summary") if isinstance(manifest.get("quant_summary"), dict) else {}
    entries = manifest.get("entries") if isinstance(manifest.get("entries"), list) else []
    manifest_names = {
        str(e.get("name", "")).strip()
        for e in entries
        if isinstance(e, dict) and str(e.get("name", "")).strip()
    }
    num_layers = int(
        (manifest.get("num_layers") if isinstance(manifest.get("num_layers"), int) else 0)
        or (manifest.get("config", {}).get("num_layers") if isinstance(manifest.get("config"), dict) else 0)
        or 0
    )
    decode_ops = decode_doc.get("operations") if isinstance(decode_doc.get("operations"), list) else []
    prefill_ops = prefill_doc.get("operations") if isinstance(prefill_doc.get("operations"), list) else []
    by_mode = {
        "decode": _build_layer_index(decode_ops),
        "prefill": _build_layer_index(prefill_ops),
    }

    results: list[dict[str, Any]] = []
    failures: list[str] = []
    mixed_quant_layers = 0
    for layer in range(num_layers):
        layer_key = f"layer.{layer}"
        layer_quant = qsum.get(layer_key) if isinstance(qsum.get(layer_key), dict) else {}
        dtypes = sorted({str(v) for v in layer_quant.values() if isinstance(v, str) and v.strip()})
        if len(dtypes) > 1:
            mixed_quant_layers += 1
        layer_row: dict[str, Any] = {
            "layer": layer,
            "quant_layer_key": layer_key,
            "quant": layer_quant,
            "quant_dtypes": dtypes,
            "ops": {},
            "pass": True,
        }
        for op_name, quant_key in LAYER_WEIGHTED_OPS:
            op_report = {
                "quant_key": quant_key,
                "quant_dtype": layer_quant.get(quant_key),
                "modes": {},
            }
            if not isinstance(layer_quant.get(quant_key), str) or not str(layer_quant.get(quant_key)).strip():
                msg = f"layer {layer} missing quant_summary.{layer_key}.{quant_key}"
                failures.append(msg)
                layer_row["pass"] = False
                op_report["quant_missing"] = True
            for mode_name, layer_index in by_mode.items():
                candidates = layer_index.get(layer, {}).get(op_name, [])
                mode_report: dict[str, Any] = {
                    "present": bool(candidates),
                    "count": len(candidates),
                }
                if not candidates:
                    failures.append(f"layer {layer} {mode_name}/{op_name} missing in call-IR")
                    layer_row["pass"] = False
                else:
                    op0 = candidates[0]
                    fn = op0.get("function")
                    errs = op0.get("errors") if isinstance(op0.get("errors"), list) else []
                    refs = sorted(_extract_weight_refs_from_call_op(op0))
                    unknown_refs = [
                        r for r in refs
                        if not r.startswith("_") and r not in manifest_names
                    ]
                    mode_report.update(
                        {
                            "function": fn,
                            "errors": errs,
                            "weight_refs": refs,
                            "unknown_weight_refs": unknown_refs,
                        }
                    )
                    if not isinstance(fn, str) or not fn.strip():
                        failures.append(f"layer {layer} {mode_name}/{op_name} has empty function binding")
                        layer_row["pass"] = False
                    if errs:
                        failures.append(f"layer {layer} {mode_name}/{op_name} has op errors: {errs[:2]}")
                        layer_row["pass"] = False
                    if not refs:
                        failures.append(f"layer {layer} {mode_name}/{op_name} has no weight refs")
                        layer_row["pass"] = False
                    if unknown_refs:
                        failures.append(
                            f"layer {layer} {mode_name}/{op_name} unknown refs: {', '.join(unknown_refs[:3])}"
                        )
                        layer_row["pass"] = False
                op_report["modes"][mode_name] = mode_report
            layer_row["ops"][op_name] = op_report
        results.append(layer_row)

    top_decode_errors = decode_doc.get("errors") if isinstance(decode_doc.get("errors"), list) else []
    top_prefill_errors = prefill_doc.get("errors") if isinstance(prefill_doc.get("errors"), list) else []
    if top_decode_errors:
        failures.append(f"decode call-IR top-level errors={len(top_decode_errors)}")
    if top_prefill_errors:
        failures.append(f"prefill call-IR top-level errors={len(top_prefill_errors)}")
    return {
        "pass": len(failures) == 0,
        "num_layers": num_layers,
        "mixed_quant_layers": mixed_quant_layers,
        "decode_top_errors": len(top_decode_errors),
        "prefill_top_errors": len(top_prefill_errors),
        "layers": results,
        "failure_count": len(failures),
        "failures": failures[:200],
    }


def _parse_layer_spec(layer_spec: str, num_layers: int) -> list[int]:
    spec = (layer_spec or "all").strip().lower()
    if spec in {"all", "*"}:
        return list(range(max(0, int(num_layers))))
    if spec.startswith("first:"):
        n = int(spec.split(":", 1)[1] or "0")
        return list(range(max(0, min(num_layers, n))))
    layers: set[int] = set()
    for part in spec.split(","):
        p = part.strip()
        if not p:
            continue
        if "-" in p:
            a_s, b_s = p.split("-", 1)
            a = int(a_s)
            b = int(b_s)
            lo, hi = (a, b) if a <= b else (b, a)
            for x in range(lo, hi + 1):
                if 0 <= x < num_layers:
                    layers.add(x)
            continue
        x = int(p)
        if 0 <= x < num_layers:
            layers.add(x)
    return sorted(layers)


def _select_qkv_model_dir(model_dir: Path, artifact_root: Path, run_dir: Path | None) -> Path:
    needed = (
        "weights.bump",
        "weights_manifest.json",
        "libmodel.so",
        "lowered_decode.json",
        "lowered_decode_call.json",
        "lowered_prefill.json",
        "lowered_prefill_call.json",
    )
    candidates: list[Path] = []
    for c in (model_dir, artifact_root, run_dir):
        if c is None:
            continue
        c = c.expanduser().resolve()
        if c not in candidates:
            candidates.append(c)
    for cand in candidates:
        if all((cand / n).exists() for n in needed):
            return cand
    return model_dir


def _layerwise_qkv_gate(
    qkv_model_dir: Path,
    num_layers: int,
    layer_spec: str,
    decode_token: int,
    stop_on_fail: bool,
) -> dict[str, Any]:
    layers = _parse_layer_spec(layer_spec, num_layers)
    results: list[dict[str, Any]] = []
    failures: list[str] = []
    for layer in layers:
        prefill_csv = ",".join(str(x) for x in (1, 2, 3, 4, max(1, int(decode_token))))
        cmd = [
            sys.executable,
            str(LAYER0_QKV_CHECK),
            "--model-dir",
            str(qkv_model_dir),
            "--layer",
            str(layer),
            "--decode-token",
            str(decode_token),
            "--prefill-tokens",
            prefill_csv,
            "--no-fail-fast",
        ]
        proc = run_cmd(cmd, check=False)
        layer_pass = proc.returncode == 0
        results.append(
            {
                "layer": layer,
                "pass": layer_pass,
                "returncode": int(proc.returncode),
                "command": cmd,
                "stdout_tail": proc.stdout[-2000:],
                "stderr_tail": proc.stderr[-1200:],
            }
        )
        if not layer_pass:
            failures.append(f"layer {layer} qkv-contract failed")
            if stop_on_fail:
                break
    return {
        "pass": len(failures) == 0,
        "layers_checked": len(results),
        "layers_total": num_layers,
        "layer_spec": layer_spec,
        "results": results,
        "failures": failures,
    }


def resolve_hf_cache_gguf(model_uri: str) -> Path | None:
    """Resolve hf://repo/path/file.gguf to the local ck cache GGUF path."""
    if not model_uri.startswith("hf://"):
        return None
    body = model_uri[len("hf://") :]
    if "/" not in body:
        return None
    repo_path, filename = body.rsplit("/", 1)
    repo_key = repo_path.replace("/", "--")
    p = Path.home() / ".cache" / "ck-engine-v7" / "models" / repo_key / filename
    return p if p.exists() else None


def ensure_output_dir_has_gguf(model_uri: str, output_dir: Path | None) -> str | None:
    """If using custom output dir + hf URI, place a symlink to GGUF there for llama parity."""
    if output_dir is None:
        return None
    gguf = resolve_hf_cache_gguf(model_uri)
    if gguf is None:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    link = output_dir / gguf.name
    if link.exists():
        return str(link)
    try:
        link.symlink_to(gguf)
        return str(link)
    except Exception:
        return None


def _resolve_gguf_for_token_replay(model_uri: str, model_dir: Path, artifact_root: Path) -> Path | None:
    candidates: list[Path] = []
    if model_uri.startswith("hf://"):
        p = resolve_hf_cache_gguf(model_uri)
        if p is not None:
            candidates.append(p)
    else:
        p = Path(model_uri).expanduser().resolve()
        if p.is_file() and p.suffix.lower() == ".gguf":
            candidates.append(p)

    for root in (artifact_root, model_dir, model_dir.parent):
        if root is None:
            continue
        for gguf in sorted(root.glob("*.gguf")):
            candidates.append(gguf.resolve())

    for c in candidates:
        if c.exists():
            return c
    return None


def _token_replay_gate(
    *,
    model_dir: Path,
    gguf_path: Path | None,
    tokens_csv: str,
    ctx_len: int,
    top_k: int,
    require_top1_match: bool,
    min_topk_overlap: float,
    max_abs_threshold: float,
) -> dict[str, Any]:
    if not TOKEN_REPLAY_CHECK.exists():
        return {
            "pass": False,
            "status": "skip",
            "reason": f"missing script: {TOKEN_REPLAY_CHECK}",
        }
    if gguf_path is None:
        return {
            "pass": False,
            "status": "skip",
            "reason": "gguf unavailable for token replay",
        }
    cmd = [
        sys.executable,
        str(TOKEN_REPLAY_CHECK),
        "--model-dir",
        str(model_dir),
        "--gguf",
        str(gguf_path),
        "--tokens",
        str(tokens_csv),
        "--ctx-len",
        str(int(ctx_len)),
        "--top-k",
        str(int(top_k)),
        "--min-topk-overlap",
        str(float(min_topk_overlap)),
        "--max-abs-threshold",
        str(float(max_abs_threshold)),
    ]
    if require_top1_match:
        cmd.append("--require-top1-match")
    else:
        cmd.append("--no-require-top1-match")

    proc = run_cmd(cmd, check=False)
    payload = parse_json_blob(proc.stdout)
    payload_dict = payload if isinstance(payload, dict) else {}
    status = str(payload_dict.get("status", "")).lower()
    passed = bool(payload_dict.get("pass", False))
    if proc.returncode == 0 and status == "pass":
        passed = True
    return {
        "pass": passed,
        "status": status or ("pass" if passed else "fail"),
        "returncode": int(proc.returncode),
        "command": cmd,
        "report": payload_dict,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-1200:],
    }


def _layer0_qkv_exact_token_check(
    *,
    model_dir: Path,
    decode_token: int,
    layer: int = 0,
) -> dict[str, Any]:
    if not LAYER0_QKV_CHECK.exists():
        return {
            "pass": False,
            "status": "skip",
            "reason": f"missing script: {LAYER0_QKV_CHECK}",
        }
    cmd = [
        sys.executable,
        str(LAYER0_QKV_CHECK),
        "--model-dir",
        str(model_dir),
        "--layer",
        str(int(layer)),
        "--decode-token",
        str(int(decode_token)),
        "--skip-prefill",
        "--no-fail-fast",
    ]
    proc = run_cmd(cmd, check=False)
    return {
        "pass": bool(proc.returncode == 0),
        "status": "pass" if proc.returncode == 0 else "fail",
        "returncode": int(proc.returncode),
        "command": cmd,
        "stdout_tail": proc.stdout[-3000:],
        "stderr_tail": proc.stderr[-1200:],
    }


def build_report_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Parity Autopsy Report")
    lines.append("")
    lines.append(f"- Timestamp: {report['timestamp']}")
    lines.append(f"- Family: `{report['family']}`")
    lines.append(f"- Model URI: `{report['model_uri']}`")
    lines.append(f"- Model dir: `{report['model_dir']}`")
    lines.append("")
    lines.append("## Preflight")
    tpl = report.get("template_audit", {}) or {}
    if tpl:
        lines.append(
            f"- `template-audit`: rc={tpl.get('returncode')} pass={tpl.get('pass')} "
            f"report_status={tpl.get('report_status')}"
        )
    quant_gate = report.get("quant_stitch_gate", {}) or {}
    if quant_gate:
        lines.append(
            f"- `quant+stitch`: pass={quant_gate.get('pass')} "
            f"layers={quant_gate.get('num_layers')} mixed_quant_layers={quant_gate.get('mixed_quant_layers')} "
            f"failures={quant_gate.get('failure_count')}"
        )
    layerwise = report.get("layerwise_qkv_gate", {}) or {}
    if layerwise:
        lines.append(
            f"- `layerwise_qkv`: pass={layerwise.get('pass')} "
            f"checked={layerwise.get('layers_checked')}/{layerwise.get('layers_total')}"
        )
    lines.append("")
    lines.append("## Parity Summary")
    parity = report.get("parity", {}) or {}
    summary = parity.get("summary")
    if summary:
        lines.append(
            f"- PASS={summary.get('PASS', 0)} FAIL={summary.get('FAIL', 0)} "
            f"ERROR={summary.get('ERROR', 0)} WARN={summary.get('WARN', 0)} TOTAL={summary.get('TOTAL', 0)}"
        )
    else:
        lines.append("- Summary unavailable")
    first = parity.get("first_issue")
    first_raw = parity.get("first_issue_raw")
    if isinstance(first, dict):
        lines.append(
            f"- First issue: layer={first.get('layer')} op={first.get('op')} token={first.get('token')} "
            f"status={first.get('status')}"
        )
    else:
        lines.append("- First issue: none")
    if isinstance(first_raw, dict):
        lines.append(
            f"- Raw first issue: layer={first_raw.get('layer')} op={first_raw.get('op')} token={first_raw.get('token')} "
            f"status={first_raw.get('status')}"
        )
    layer_div = parity.get("layer_first_issues")
    if isinstance(layer_div, list) and layer_div:
        lines.append(f"- Layer divergence count: {len(layer_div)}")
    dump_integrity = parity.get("dump_integrity") if isinstance(parity.get("dump_integrity"), dict) else {}
    if dump_integrity:
        lines.append(
            f"- Dump integrity: ok={dump_integrity.get('ok')} issues={len(dump_integrity.get('issues') or [])}"
        )
    token_replay = report.get("token_replay_check", {}) or {}
    if token_replay:
        tr_tokens_meta = report.get("token_replay_tokens") if isinstance(report.get("token_replay_tokens"), dict) else {}
        if tr_tokens_meta:
            lines.append(
                f"- Token replay tokens: source={tr_tokens_meta.get('source')} "
                f"tokens={tr_tokens_meta.get('tokens')}"
            )
        if token_replay.get("status") == "skip":
            lines.append(f"- Token replay gate: skip ({token_replay.get('reason')})")
        else:
            tr_report = token_replay.get("report", {}) if isinstance(token_replay.get("report"), dict) else {}
            tr_cmp = tr_report.get("compare", {}) if isinstance(tr_report.get("compare"), dict) else {}
            lines.append(
                f"- Token replay gate: status={token_replay.get('status')} pass={token_replay.get('pass')} "
                f"top1_match={tr_cmp.get('top1_match')} overlap={tr_cmp.get('topk_overlap_ratio')} "
                f"max_abs={tr_cmp.get('max_abs_diff')}"
            )
    lines.append("")
    if isinstance(layer_div, list) and layer_div:
        lines.append("## Layer Divergence")
        for row in layer_div:
            lines.append(
                f"- layer={row.get('layer')} op={row.get('op')} token={row.get('token')} "
                f"status={row.get('status')}"
            )
        lines.append("")
    lines.append("## Projection Contract Checks")
    for op, c in report["projection_checks"].items():
        lines.append(
            f"- `{op}`: rc={c['returncode']} pass_contract={c['metrics'].get('pass_contract')} "
            f"max_diff={c['metrics'].get('max_diff')} mean_diff={c['metrics'].get('mean_diff')}"
        )
    qkv = report.get("layer0_qkv_contract") or {}
    if qkv:
        lines.append(
            f"- `layer0_qkv_contract`: rc={qkv.get('returncode')} pass={qkv.get('pass')}"
        )
    qkv_exact = report.get("layer0_qkv_exact_token_check") or {}
    if qkv_exact:
        lines.append(
            f"- `layer0_qkv_exact_token_check`: rc={qkv_exact.get('returncode')} "
            f"pass={qkv_exact.get('pass')} token={qkv_exact.get('decode_token')}"
        )
    lines.append("")
    lines.append("## Diagnosis")
    for x in report["diagnosis"]:
        lines.append(f"- {x}")
    lines.append("")
    lines.append("## Next Actions")
    for x in report["next_actions"]:
        lines.append(f"- {x}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Automated parity autopsy (v7)")
    ap.add_argument("--model-uri", required=True, help="GGUF path or hf:// URI")
    ap.add_argument("--family", choices=sorted(FAMILY_PROFILE.keys()), default=None)
    ap.add_argument("--context-len", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=1)
    ap.add_argument("--prompt", default="Hello")
    ap.add_argument("--proj-token-id", type=int, default=5, help="token id for projection contract checks")
    ap.add_argument("--output-dir", type=Path, default=None, help="ck_build output dir for ck_run")
    ap.add_argument("--skip-run", action="store_true", help="reuse existing ck_build dumps")
    ap.add_argument("--pass", dest="pass_name", choices=["prefill", "decode"], default="prefill")
    ap.add_argument("--llama-filter", default=None, help="override llama dump filter")
    ap.add_argument("--llama-stop-after", type=int, default=20)
    ap.add_argument(
        "--llama-require-token-aware-dumps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require token-aware llama CKDMP index (reject collapsed token_id=0 dumps)",
    )
    ap.add_argument(
        "--llama-allow-raw-fallback",
        action="store_true",
        help="Allow LLAMA_DUMP_LAYER0 raw fallback conversion when CKDMP dump is missing/invalid",
    )
    ap.add_argument("--report-prefix", default="autopsy_report", help="basename for .json/.md report files")
    ap.add_argument("--skip-template-audit", action="store_true", help="skip template-audit preflight")
    ap.add_argument("--skip-quant-stitch-check", action="store_true", help="skip per-layer quant/stitch preflight")
    ap.add_argument("--skip-layerwise-qkv-check", action="store_true", help="skip per-layer qkv contract preflight")
    ap.add_argument("--qkv-layers", default="all", help="layers for qkv preflight: all | first:N | 0,1,2 | 0-3")
    ap.add_argument("--qkv-stop-on-fail", action="store_true", help="stop layerwise qkv preflight at first failing layer")
    ap.add_argument("--skip-token-replay-check", action="store_true", help="skip tokenizer-free first-token logits check")
    ap.add_argument("--token-replay-tokens", default="1,2,3,4,5", help="comma-separated token IDs for tokenizer-free replay")
    ap.add_argument("--token-replay-max-index-unique", type=int, default=8, help="max unique token IDs to derive from parity index")
    ap.add_argument("--token-replay-top-k", type=int, default=16, help="top-k for token replay overlap metric")
    ap.add_argument(
        "--token-replay-require-top1-match",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="require top1 token to match in tokenizer-free replay gate",
    )
    ap.add_argument("--token-replay-min-topk-overlap", type=float, default=0.50, help="minimum top-k overlap ratio")
    ap.add_argument("--token-replay-max-abs", type=float, default=1.0e9, help="max absolute logits diff threshold")
    ap.add_argument("--token-replay-strict", action="store_true", help="fail-fast if tokenizer-free token replay gate fails")
    ap.add_argument("--first-divergence", action="store_true", help="emit global first divergence in report")
    ap.add_argument("--layer-divergence", action="store_true", help="emit first divergence per layer in report")
    args = ap.parse_args()

    model_dir = infer_model_dir(args.output_dir, args.model_uri)
    selected_family = infer_family(args.model_uri, model_dir, args.family)
    profile = FAMILY_PROFILE[selected_family]
    parity_model = profile["parity_model"]
    llama_filter = args.llama_filter or profile["llama_filter"]

    report: dict[str, Any] = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "family": selected_family,
        "model_uri": args.model_uri,
        "model_dir": str(model_dir),
        "commands": {},
        "parity": {},
        "projection_checks": {},
        "diagnosis": [],
        "next_actions": [],
        "preflight_order": [
            "template-audit",
            "per-layer-quant-stitch",
            "layerwise-qkv-contract",
            "token-replay-first-token-logits",
            "parity-autopsy",
        ],
    }
    run_dir = infer_run_dir_from_output_dir(args.output_dir)

    # Strict preflight 1/3: template/kernel/stitch sanity.
    if not args.skip_template_audit:
        template_input = str(run_dir if run_dir is not None else args.model_uri)
        t_gate = _template_audit_gate(template_input, run_dir, args.context_len)
        report["commands"]["template_audit"] = t_gate.get("command")
        report["template_audit"] = {k: v for k, v in t_gate.items() if k != "command"}
        if not bool(t_gate.get("pass")):
            report["diagnosis"].append("template-audit failed (template/kernel/stitch preflight)")
            report["next_actions"].append("Fix template-audit gate before running parity autopsy.")
            out_json = model_dir / f"{args.report_prefix}.json"
            out_md = model_dir / f"{args.report_prefix}.md"
            out_json.write_text(json.dumps(to_jsonable(report), indent=2), encoding="utf-8")
            out_md.write_text(build_report_markdown(report), encoding="utf-8")
            print(f"[autopsy] wrote {out_json}")
            print(f"[autopsy] wrote {out_md}")
            return 3

    if not args.skip_run:
        linked_gguf = ensure_output_dir_has_gguf(args.model_uri, args.output_dir)
        if linked_gguf:
            report["linked_gguf"] = linked_gguf
        run_cmd_list = [
            sys.executable,
            str(CK_RUN),
            "run",
            args.model_uri,
            "--force-compile",
            "--context-len",
            str(args.context_len),
            "--max-tokens",
            str(args.max_tokens),
            "--prompt",
            args.prompt,
            "--detailed-llamacpp-parity",
            "--llama-layer",
            "0",
            "--llama-include-global",
            "--llama-filter",
            llama_filter,
            "--llama-stop-after",
            str(args.llama_stop_after),
            "--llama-timeout",
            "0",
        ]
        if args.llama_require_token_aware_dumps:
            run_cmd_list.append("--llama-require-token-aware-dumps")
        if not args.llama_allow_raw_fallback:
            run_cmd_list.append("--llama-no-raw-fallback")
        if run_dir is not None:
            run_cmd_list.extend(["--run", str(run_dir)])
        report["commands"]["ck_run"] = run_cmd_list
        ck_proc = run_cmd(run_cmd_list, check=False)
        report["ck_run_returncode"] = ck_proc.returncode
        report["ck_run_stdout_tail"] = ck_proc.stdout[-4000:]
        report["ck_run_stderr_tail"] = ck_proc.stderr[-2000:]
        if ck_proc.returncode != 0:
            report["diagnosis"].append("ck_run_v7.py failed; parity not executed.")
            out_json = model_dir / f"{args.report_prefix}.json"
            out_md = model_dir / f"{args.report_prefix}.md"
            out_json.write_text(json.dumps(to_jsonable(report), indent=2), encoding="utf-8")
            out_md.write_text(build_report_markdown(report), encoding="utf-8")
            print(f"[autopsy] wrote {out_json}")
            print(f"[autopsy] wrote {out_md}")
            return 1

    artifact_root = pick_artifact_root(model_dir, run_dir)
    report["artifact_root"] = str(artifact_root)

    # Strict preflight 2/3: per-layer mixed-quant + call-IR stitching checks.
    if not args.skip_quant_stitch_check:
        quant_gate = _per_layer_quant_stitch_gate(artifact_root)
        report["quant_stitch_gate"] = quant_gate
        if not bool(quant_gate.get("pass")):
            report["diagnosis"].append("per-layer quant/stitch preflight failed")
            report["next_actions"].append(
                "Fix per-layer quantization or call-IR stitching errors before full parity autopsy."
            )
            out_json = model_dir / f"{args.report_prefix}.json"
            out_md = model_dir / f"{args.report_prefix}.md"
            out_json.write_text(json.dumps(to_jsonable(report), indent=2), encoding="utf-8")
            out_md.write_text(build_report_markdown(report), encoding="utf-8")
            print(f"[autopsy] wrote {out_json}")
            print(f"[autopsy] wrote {out_md}")
            return 4

    # Strict preflight 3/3: layerwise qkv formula checks.
    if not args.skip_layerwise_qkv_check:
        num_layers = int(
            (report.get("quant_stitch_gate", {}).get("num_layers") if isinstance(report.get("quant_stitch_gate"), dict) else 0)
            or 0
        )
        qkv_model_dir = _select_qkv_model_dir(model_dir, artifact_root, run_dir)
        layerwise_gate = _layerwise_qkv_gate(
            qkv_model_dir=qkv_model_dir,
            num_layers=num_layers,
            layer_spec=args.qkv_layers,
            decode_token=args.proj_token_id,
            stop_on_fail=bool(args.qkv_stop_on_fail),
        )
        report["layerwise_qkv_gate"] = layerwise_gate
        if not bool(layerwise_gate.get("pass")):
            report["diagnosis"].append("layerwise qkv contract preflight failed")
            report["next_actions"].append("Fix failing layer qkv contract before running full parity autopsy.")
            out_json = model_dir / f"{args.report_prefix}.json"
            out_md = model_dir / f"{args.report_prefix}.md"
            out_json.write_text(json.dumps(to_jsonable(report), indent=2), encoding="utf-8")
            out_md.write_text(build_report_markdown(report), encoding="utf-8")
            print(f"[autopsy] wrote {out_json}")
            print(f"[autopsy] wrote {out_md}")
            return 5

    checks_model_dir = _select_qkv_model_dir(model_dir, artifact_root, run_dir)
    token_replay_failed = False
    token_replay_failed_token: int | None = None
    token_replay_tokens_for_checks: list[int] = []

    # Strict preflight 4/4: tokenizer-free token replay logits check.
    if not args.skip_token_replay_check:
        replay_gguf = _resolve_gguf_for_token_replay(args.model_uri, model_dir, artifact_root)
        replay_index = artifact_root / "llama_parity_dumps" / "index.json"
        fallback_tokens = _parse_tokens_csv_strict(str(args.token_replay_tokens))
        idx_tokens_meta = _token_replay_tokens_from_index(replay_index, int(args.token_replay_max_index_unique))
        if bool(idx_tokens_meta.get("ok")) and isinstance(idx_tokens_meta.get("tokens"), list):
            replay_tokens = [int(x) for x in idx_tokens_meta.get("tokens", [])]
            replay_source = "parity_index"
        else:
            replay_tokens = fallback_tokens
            replay_source = "cli_fallback"
        replay_tokens_csv = ",".join(str(x) for x in replay_tokens)
        token_replay_tokens_for_checks = replay_tokens
        token_gate = _token_replay_gate(
            model_dir=checks_model_dir,
            gguf_path=replay_gguf,
            tokens_csv=replay_tokens_csv,
            ctx_len=int(args.context_len),
            top_k=int(args.token_replay_top_k),
            require_top1_match=bool(args.token_replay_require_top1_match),
            min_topk_overlap=float(args.token_replay_min_topk_overlap),
            max_abs_threshold=float(args.token_replay_max_abs),
        )
        report["token_replay_check"] = {k: v for k, v in token_gate.items() if k != "command"}
        report["token_replay_gguf"] = str(replay_gguf) if replay_gguf is not None else None
        report["token_replay_tokens"] = {
            "source": replay_source,
            "tokens": replay_tokens,
            "index_path": str(replay_index),
            "index_tokens_meta": idx_tokens_meta,
        }
        if token_gate.get("command"):
            report["commands"]["token_replay_check"] = token_gate.get("command")
        if token_gate.get("status") == "skip":
            report["diagnosis"].append(f"token replay check skipped: {token_gate.get('reason')}")
        elif not bool(token_gate.get("pass")):
            token_replay_failed = True
            tr = token_gate.get("report", {}) if isinstance(token_gate.get("report"), dict) else {}
            tr_tokens = tr.get("tokens")
            if isinstance(tr_tokens, list) and tr_tokens:
                try:
                    token_replay_failed_token = int(tr_tokens[-1])
                except Exception:
                    token_replay_failed_token = None
            if token_replay_failed_token is None and replay_tokens:
                token_replay_failed_token = int(replay_tokens[-1])
            report["diagnosis"].append("token replay first-token logits gate failed")
            report["next_actions"].append(
                "Run compare_first_token_logits.py on the same token IDs to isolate tokenizer-independent runtime mismatch."
            )
            if token_replay_failed_token is not None:
                exact_qkv = _layer0_qkv_exact_token_check(
                    model_dir=checks_model_dir,
                    decode_token=int(token_replay_failed_token),
                    layer=0,
                )
                report["layer0_qkv_exact_token_check"] = {
                    k: v for k, v in exact_qkv.items() if k != "command"
                }
                report["layer0_qkv_exact_token_check"]["decode_token"] = int(token_replay_failed_token)
                if exact_qkv.get("command"):
                    report["commands"]["layer0_qkv_exact_token_check"] = exact_qkv.get("command")
                if exact_qkv.get("status") == "skip":
                    report["diagnosis"].append(
                        f"exact-token layer0 qkv check skipped: {exact_qkv.get('reason')}"
                    )
                elif not bool(exact_qkv.get("pass")):
                    report["diagnosis"].append(
                        f"exact-token layer0 qkv contract failed (token={token_replay_failed_token})"
                    )
                else:
                    report["diagnosis"].append(
                        f"exact-token layer0 qkv contract passed (token={token_replay_failed_token})"
                    )
            if bool(args.token_replay_strict):
                out_json = model_dir / f"{args.report_prefix}.json"
                out_md = model_dir / f"{args.report_prefix}.md"
                out_json.write_text(json.dumps(to_jsonable(report), indent=2), encoding="utf-8")
                out_md.write_text(build_report_markdown(report), encoding="utf-8")
                print(f"[autopsy] wrote {out_json}")
                print(f"[autopsy] wrote {out_md}")
                return 7

    ck_dump = artifact_root / "ck_parity_dumps" / "dump.bin"
    ref_dump = artifact_root / "llama_parity_dumps" / "dump.bin"
    ref_index = artifact_root / "llama_parity_dumps" / "index.json"
    ref_token_audit = audit_ref_dump_tokens(ref_index)
    report["ref_token_audit"] = ref_token_audit
    if args.llama_require_token_aware_dumps and not bool(ref_token_audit.get("ok")):
        report["diagnosis"].append(
            "Token-aware llama reference dump unavailable or invalid; first-divergence attribution is not trustworthy."
        )
        report["next_actions"].append(
            "Regenerate llama CKDMP dumps with token-aware index (non-collapsed token_id set), then rerun autopsy."
        )
        if not args.llama_allow_raw_fallback:
            report["next_actions"].append("Keep raw fallback disabled while validating first divergence.")
        out_json = model_dir / f"{args.report_prefix}.json"
        out_md = model_dir / f"{args.report_prefix}.md"
        out_json.write_text(json.dumps(to_jsonable(report), indent=2), encoding="utf-8")
        out_md.write_text(build_report_markdown(report), encoding="utf-8")
        print(f"[autopsy] wrote {out_json}")
        print(f"[autopsy] wrote {out_md}")
        return 6
    parity_cmd = [
        sys.executable,
        str(PARITY_TEST),
        "--ck-dump",
        str(ck_dump),
        "--ref-dump",
        str(ref_dump),
        "--model",
        parity_model,
        "--pass",
        args.pass_name,
        "--json",
        "--quiet",
    ]
    report["commands"]["parity_test"] = parity_cmd
    parity_proc = run_cmd(parity_cmd, check=False)
    report["parity_test_returncode"] = parity_proc.returncode
    report["parity_test_stderr_tail"] = parity_proc.stderr[-2000:]
    parity_log_text = f"{parity_proc.stdout}\n{parity_proc.stderr}"
    results = parse_json_blob(parity_proc.stdout)
    if not isinstance(results, list):
        # Fallback path: direct call to parity module to avoid CLI --json failure.
        report["diagnosis"].append("parity_test --json parse failed; using direct module fallback.")
        report["parity_raw_stdout"] = parity_proc.stdout[-4000:]
        try:
            rc2, res2 = run_parity_direct(ck_dump, ref_dump, parity_model, args.pass_name)
            report["parity_test_returncode_fallback"] = rc2
            results = res2
        except Exception as e:
            report["diagnosis"].append(f"Direct parity fallback failed: {e}")
            out_json = model_dir / f"{args.report_prefix}.json"
            out_md = model_dir / f"{args.report_prefix}.md"
            out_json.write_text(json.dumps(to_jsonable(report), indent=2), encoding="utf-8")
            out_md.write_text(build_report_markdown(report), encoding="utf-8")
            print(f"[autopsy] wrote {out_json}")
            print(f"[autopsy] wrote {out_md}")
            return 2

    dump_issues = detect_dump_integrity_issues(parity_log_text + "\n" + str(report.get("parity_raw_stdout", "")))
    summary = {
        "PASS": sum(1 for r in results if r.get("status") == "PASS"),
        "FAIL": sum(1 for r in results if r.get("status") == "FAIL"),
        "ERROR": sum(1 for r in results if r.get("status") == "ERROR"),
        "WARN": sum(1 for r in results if r.get("status") == "WARN"),
        "TOTAL": len(results),
    }
    order = load_op_order(model_dir, args.pass_name)
    # Default behavior: if no flag provided, emit both.
    use_first_div = bool(args.first_divergence or (not args.first_divergence and not args.layer_divergence))
    use_layer_div = bool(args.layer_divergence or (not args.first_divergence and not args.layer_divergence))
    first_issue_raw = pick_first_issue(results, order, prefer_unambiguous=False) if use_first_div else None
    first_issue = pick_first_issue(results, order, prefer_unambiguous=True) if use_first_div else None
    layer_first_issues = pick_layer_first_issues(results, order) if use_layer_div else []
    report["parity"] = {
        "summary": summary,
        "first_issue": first_issue,
        "first_issue_raw": first_issue_raw,
        "layer_first_issues": layer_first_issues,
        "divergence_mode": {
            "first": use_first_div,
            "layer": use_layer_div,
        },
        "dump_integrity": {
            "ok": len(dump_issues) == 0,
            "issues": dump_issues,
        },
    }
    ref_dump_exists = ref_dump.exists() and ref_dump.stat().st_size > 0
    report["parity"]["ref_dump_exists"] = ref_dump_exists
    parity_trustworthy = bool(ref_dump_exists and len(dump_issues) == 0)

    # Keep deep-contract checks on the same selected artifact root to avoid
    # mixing stale run-root artifacts with fresh .ck_build outputs.

    # Projection deep checks (always run; cheap and high-signal).
    proj_token_effective = int(token_replay_failed_token) if token_replay_failed_token is not None else int(args.proj_token_id)
    report["projection_token_effective"] = int(proj_token_effective)
    for op in SUPPORTED_OPS:
        cmd = [
            sys.executable,
            str(PROJ_CHECK),
            "--model-dir",
            str(checks_model_dir),
            "--op",
            op,
            "--layer",
            "0",
            "--token",
            str(proj_token_effective),
        ]
        proc = run_cmd(cmd, check=False)
        report["projection_checks"][op] = {
            "returncode": proc.returncode,
            "metrics": parse_projection_metrics(proc.stdout),
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-800:],
        }

    # Dtype-aware contract: for mixed q4_k/q6_k models, rely on layer0_qkv contract checker.
    if token_replay_failed and token_replay_failed_token is not None:
        qkv_cmd = [
            sys.executable,
            str(LAYER0_QKV_CHECK),
            "--model-dir",
            str(checks_model_dir),
            "--layer",
            "0",
            "--decode-token",
            str(int(token_replay_failed_token)),
            "--skip-prefill",
            "--no-fail-fast",
        ]
    else:
        prefill_seed = token_replay_tokens_for_checks if len(token_replay_tokens_for_checks) >= 2 else [1, 2, 3, 4, max(1, int(proj_token_effective))]
        prefill_csv = ",".join(str(x) for x in prefill_seed)
        qkv_cmd = [
            sys.executable,
            str(LAYER0_QKV_CHECK),
            "--model-dir",
            str(checks_model_dir),
            "--layer",
            "0",
            "--decode-token",
            str(int(proj_token_effective)),
            "--prefill-tokens",
            prefill_csv,
            "--no-fail-fast",
        ]
    qkv_proc = run_cmd(qkv_cmd, check=False)
    qkv_pass = qkv_proc.returncode == 0
    report["commands"]["layer0_qkv_contract"] = qkv_cmd
    report["layer0_qkv_contract"] = {
        "returncode": qkv_proc.returncode,
        "pass": qkv_pass,
        "stdout_tail": qkv_proc.stdout[-2500:],
        "stderr_tail": qkv_proc.stderr[-1200:],
    }

    proj_results = report["projection_checks"].values()
    proj_supported = [c for c in proj_results if not _is_projection_checker_unsupported(c.get("stderr_tail", ""))]
    proj_supported_pass = bool(proj_supported) and all(c.get("metrics", {}).get("pass_contract") for c in proj_supported)
    proj_all_unsupported = bool(proj_results) and all(_is_projection_checker_unsupported(c.get("stderr_tail", "")) for c in proj_results)
    proj_all_pass = proj_supported_pass or (proj_all_unsupported and qkv_pass)

    if first_issue is None:
        report["diagnosis"].append("No FAIL/ERROR in parity output.")
    else:
        if first_issue_raw and first_issue_raw != first_issue:
            if str(first_issue.get("promotion_reason", "")) == "next_non_token_embedding_fail":
                report["diagnosis"].append(
                    "Raw first divergence is alignment-ambiguous (token_embedding); report promotes next non-token_embedding failing op."
                )
            else:
                report["diagnosis"].append(
                    "Raw first divergence is alignment-ambiguous (token_embedding); report promotes first unambiguous failing op."
                )
        if parity_trustworthy:
            report["diagnosis"].append(
                f"First parity issue: layer={first_issue.get('layer')} op={first_issue.get('op')} status={first_issue.get('status')}"
            )
        else:
            report["diagnosis"].append(
                "First parity issue exists but dump integrity is degraded; divergence location is not yet trustworthy."
            )

    if not ref_dump_exists:
        report["diagnosis"].append("llama reference dump missing/empty; WARN-only parity is not actionable.")
        report["next_actions"].append(
            "Re-run with a GGUF path visible in output dir (or without custom --output-dir) so llama parity can dump references."
        )
    if dump_issues:
        report["diagnosis"].append(
            "Dump integrity issue detected in parity dumps (parse resync/dtype-size mismatch); fix dump pipeline before trusting divergence."
        )
        report["next_actions"].append(
            "Fix CKDMP writer/reader alignment first (header parse + dtype/elem-size consistency), then rerun autopsy."
        )
    elif summary["PASS"] == 0 and summary["WARN"] > 0 and summary["FAIL"] == 0 and summary["ERROR"] == 0:
        report["diagnosis"].append("Parity produced WARN-only results (coverage gap).")
        report["next_actions"].append("Expand --llama-filter/--llama-stop-after to include expected layer-0 ops.")

    if proj_all_pass:
        if parity_trustworthy:
            report["diagnosis"].append(
                "Projection contracts (q/k/v) match local reference; likely divergence is after projections."
            )
            report["next_actions"].append(
                "Focus on qk_norm / rope_qk / attention (sliding or decode kernel) and post-attention dataflow."
            )
        else:
            report["diagnosis"].append(
                "Projection contracts (q/k/v) pass, but parity dump integrity issues block reliable stage attribution."
            )
    else:
        report["diagnosis"].append("At least one projection contract check failed.")
        report["next_actions"].append("Fix failing projection contract before deeper attention debugging.")

    report["next_actions"].append(
        "Ensure llama dump coverage includes the op where first divergence appears (otherwise rerun with broader --llama-filter)."
    )
    report["next_actions"].append(
        "Keep Qwen regression gates separate from Gemma debugging path."
    )

    out_json = model_dir / f"{args.report_prefix}.json"
    out_md = model_dir / f"{args.report_prefix}.md"
    out_json.write_text(json.dumps(to_jsonable(report), indent=2), encoding="utf-8")
    out_md.write_text(build_report_markdown(report), encoding="utf-8")
    print(f"[autopsy] wrote {out_json}")
    print(f"[autopsy] wrote {out_md}")
    print(f"[autopsy] parity summary: {summary}")
    if first_issue:
        print(
            f"[autopsy] first issue: layer={first_issue.get('layer')} op={first_issue.get('op')} status={first_issue.get('status')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
