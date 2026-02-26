#!/usr/bin/env python3
"""
Standalone detailed parity analysis for v7.

This script does NOT patch ck_run_v7.py. Instead, it:
1) Optionally calls ck_run_v7.py with --detailed-llamacpp-parity
2) Runs an exhaustive llama.cpp CKDMP pass (all layers, no default filter cap)
3) Audits dump coverage, weight coverage, and first divergence
4) Writes JSON + Markdown reports to the run directory
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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
CK_RUN = SCRIPT_DIR / "ck_run_v7.py"
PARITY_TEST = SCRIPT_DIR / "parity_test.py"

PARITY_MODEL_MAP = {
    "gemma": "gemma",
    "llama": "llama",
    "qwen": "qwen",
    "qwen2": "qwen2",
    "qwen3": "qwen3",
    "mistral": "mistral",
}


def run_cmd(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd or ROOT),
        env=env,
        timeout=timeout,
        stdin=subprocess.DEVNULL,
        text=True,
        capture_output=True,
        check=False,
    )


def infer_model_dir(model_uri: str, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir
    if model_uri.startswith("hf://"):
        repo = model_uri[len("hf://") :].rsplit("/", 1)[0]
        key = repo.replace("/", "--")
        return Path.home() / ".cache" / "ck-engine-v7" / "models" / key
    p = Path(model_uri).expanduser().resolve()
    if p.is_dir():
        return p
    return p.parent


def resolve_hf_gguf(model_uri: str) -> Path | None:
    if not model_uri.startswith("hf://"):
        return None
    body = model_uri[len("hf://") :]
    if "/" not in body:
        return None
    repo, filename = body.rsplit("/", 1)
    repo_key = repo.replace("/", "--")
    p = Path.home() / ".cache" / "ck-engine-v7" / "models" / repo_key / filename
    return p if p.exists() else None


def find_gguf(model_dir: Path, model_uri: str) -> Path | None:
    uri_path = Path(model_uri).expanduser()
    if uri_path.is_file() and uri_path.suffix.lower() == ".gguf":
        return uri_path.resolve()

    hf_path = resolve_hf_gguf(model_uri)
    if hf_path is not None:
        return hf_path
    for pat in ("*.gguf", "*/*.gguf"):
        found = next(iter(model_dir.glob(pat)), None)
        if found:
            return found
    parent_found = next(iter(model_dir.parent.glob("*.gguf")), None)
    if parent_found:
        return parent_found
    return None

def infer_run_dir_from_output_dir(output_dir: Path | None) -> Path | None:
    """Map a provided output dir to ck_run_v7 --run dir.

    detailed_parity_analysis accepts ck_build paths, while ck_run_v7 expects run dirs.
    """
    if output_dir is None:
        return None
    p = output_dir.expanduser().resolve()
    if p.name in {"ck_build", ".ck_build"}:
        return p.parent
    return p


def find_llama_binary() -> Path | None:
    p = ROOT / "build" / "llama-parity"
    if p.exists():
        return p
    p = ROOT / "llama.cpp" / "build" / "bin" / "llama-completion"
    if p.exists():
        return p
    p = ROOT / "llama.cpp" / "build" / "bin" / "llama-cli"
    if p.exists():
        return p
    p = ROOT / "llama.cpp" / "main"
    if p.exists():
        return p
    p = ROOT / "llama.cpp" / "build" / "bin" / "main"
    if p.exists():
        return p
    return None


def token_audit_from_index(index_path: Path) -> tuple[bool, str]:
    if not index_path.exists():
        return False, "missing index.json"
    try:
        obj = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception as e:
        return False, f"index parse error: {e}"
    if not isinstance(obj, list):
        return False, "index is not a list"
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
        return False, "index has no token_id entries"
    unique = sorted(set(token_ids))
    collapsed_zero = len(token_ids) >= 8 and len(unique) == 1 and unique[0] == 0
    if collapsed_zero:
        return False, f"collapsed token ids (all zero across {len(token_ids)} dumps)"
    return True, f"entries={len(token_ids)} unique_tokens={unique[:16]}"


def load_parity_module():
    spec = importlib.util.spec_from_file_location("v7_parity_test", str(PARITY_TEST))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed loading module: {PARITY_TEST}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if hasattr(obj, "item") and callable(getattr(obj, "item", None)):
        try:
            return jsonable(obj.item())
        except Exception:
            pass
    if hasattr(obj, "tolist") and callable(getattr(obj, "tolist", None)):
        try:
            return jsonable(obj.tolist())
        except Exception:
            pass
    return str(obj)


def summarize_dump(dumps: list[Any]) -> dict[str, Any]:
    by_op = Counter(d.op_name for d in dumps)
    by_layer = Counter(int(d.layer_id) for d in dumps)
    tokens = sorted(set(int(d.token_id) for d in dumps))
    nan_tensors = 0
    inf_tensors = 0
    total_elems = 0
    top_absmax: list[tuple[float, int, str, int, tuple[int, ...]]] = []
    for d in dumps:
        arr = d.data
        total_elems += int(arr.size)
        has_nan = bool(np.any(np.isnan(arr)))
        has_inf = bool(np.any(np.isinf(arr)))
        if has_nan:
            nan_tensors += 1
        if has_inf:
            inf_tensors += 1
        try:
            absmax = float(np.nanmax(np.abs(arr)))
        except Exception:
            absmax = float("inf")
        top_absmax.append((absmax, int(d.layer_id), d.op_name, int(d.token_id), tuple(int(x) for x in arr.shape)))
    top_absmax.sort(reverse=True, key=lambda x: x[0])
    return {
        "tensor_count": len(dumps),
        "unique_ops": len(by_op),
        "ops": dict(sorted(by_op.items())),
        "layers": dict(sorted(by_layer.items())),
        "tokens": tokens,
        "nan_tensors": nan_tensors,
        "inf_tensors": inf_tensors,
        "total_elements": total_elems,
        "top_absmax_tensors": [
            {
                "absmax": x[0],
                "layer": x[1],
                "op": x[2],
                "token": x[3],
                "shape": list(x[4]),
            }
            for x in top_absmax[:25]
        ],
    }


def parse_expected_ck_dump_points(model_c: Path) -> set[tuple[int, str]]:
    if not model_c.exists():
        return set()
    txt = model_c.read_text(encoding="utf-8", errors="ignore")
    pat = re.compile(r'ck_dump_tensor(?:_2d)?\([^,]+,\s*(-?\d+)\s*,\s*"([^"]+)"')
    out: set[tuple[int, str]] = set()
    for m in pat.finditer(txt):
        out.add((int(m.group(1)), m.group(2)))
    return out


def ck_coverage_report(expected: set[tuple[int, str]], dumps: list[Any]) -> dict[str, Any]:
    captured = set((int(d.layer_id), d.op_name) for d in dumps)
    missing = sorted(expected - captured)
    unexpected = sorted(captured - expected)
    ratio = 1.0
    if expected:
        ratio = (len(expected) - len(missing)) / float(len(expected))
    return {
        "expected_points": len(expected),
        "captured_points": len(captured),
        "coverage_ratio": ratio,
        "missing_count": len(missing),
        "unexpected_count": len(unexpected),
        "missing_examples": [{"layer": x[0], "op": x[1]} for x in missing[:80]],
        "unexpected_examples": [{"layer": x[0], "op": x[1]} for x in unexpected[:80]],
    }


def load_lowered_order(model_dir: Path, pass_name: str) -> dict[tuple[int, str], int]:
    p = model_dir / f"lowered_{pass_name}.json"
    if not p.exists():
        return {}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    ops = obj.get("operations", [])
    order: dict[tuple[int, str], int] = {}
    for idx, op in enumerate(ops):
        key = (int(op.get("layer", -1)), str(op.get("op", "")))
        if key not in order:
            order[key] = idx
    return order


def first_issue(results: list[dict[str, Any]], order: dict[tuple[int, str], int]) -> dict[str, Any] | None:
    bad = [r for r in results if r.get("status") in ("FAIL", "ERROR")]
    if not bad:
        return None

    def rk(r: dict[str, Any]) -> tuple[int, int, int]:
        l = int(r.get("layer", 10**9))
        op = str(r.get("op", ""))
        seq = order.get((l, op), 10**9)
        t = int(r.get("token", 10**9))
        return (l, seq, t)

    return sorted(bad, key=rk)[0]


def check_weights(model_dir: Path, family: str) -> dict[str, Any]:
    manifest_path = model_dir / "weights_manifest.json"
    bump_path = model_dir / "weights.bump"
    if not manifest_path.exists():
        return {"ok": False, "error": f"missing manifest: {manifest_path}"}

    obj = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = obj.get("entries", [])
    names = [e.get("name", "") for e in entries]
    name_set = set(names)
    dtypes = Counter(str(e.get("dtype", "unknown")) for e in entries)
    dupes = [k for k, v in Counter(names).items() if v > 1]

    # Basic offset sanity.
    offsets_ok = True
    prev_end = -1
    overlap_count = 0
    for e in sorted(entries, key=lambda x: int(x.get("file_offset", 0))):
        off = int(e.get("file_offset", 0))
        sz = int(e.get("size", 0))
        if off < prev_end:
            overlap_count += 1
            offsets_ok = False
        prev_end = max(prev_end, off + sz)

    bump_size = bump_path.stat().st_size if bump_path.exists() else 0
    overflow = prev_end > bump_size if bump_size > 0 else False

    num_layers = int(obj.get("num_layers", 0))
    required_global = {"token_emb", "final_ln_weight", "final_ln_bias"}
    required_layer_common = {"ln1_gamma", "ln2_gamma", "wq", "wk", "wv", "wo", "w1", "w2"}
    required_layer_gemma = {"q_norm", "k_norm", "post_attention_norm", "post_ffn_norm"}
    required_layer = set(required_layer_common)
    if family == "gemma":
        required_layer |= required_layer_gemma

    missing_required: list[str] = []
    for g in sorted(required_global):
        if g not in name_set:
            missing_required.append(g)
    for i in range(num_layers):
        for s in sorted(required_layer):
            k = f"layer.{i}.{s}"
            if k not in name_set:
                missing_required.append(k)

    return {
        "ok": len(missing_required) == 0 and not overflow and offsets_ok,
        "entries": len(entries),
        "num_layers": num_layers,
        "missing_required_count": len(missing_required),
        "missing_required_examples": missing_required[:120],
        "duplicate_name_count": len(dupes),
        "duplicate_names": dupes[:50],
        "dtype_histogram": dict(sorted(dtypes.items())),
        "offsets_ok": offsets_ok,
        "overlap_count": overlap_count,
        "bump_size_bytes": bump_size,
        "max_manifest_end": prev_end,
        "manifest_exceeds_bump": overflow,
    }


def llama_mapping_gaps(parity_mod: Any, ref_dumps: list[Any], family: str) -> dict[str, Any]:
    unknown = Counter()
    mapped = Counter()
    mapper = getattr(parity_mod, "map_llama_to_ck_name", None)
    normalizer = getattr(parity_mod, "_normalize_layer_and_op", None)
    for d in ref_dumps:
        if callable(mapper):
            mapped_name = mapper(d.op_name, family)
        elif callable(normalizer):
            try:
                _layer, mapped_name = normalizer(int(getattr(d, "layer_id", -1)), str(d.op_name))
            except Exception:
                mapped_name = str(d.op_name)
        else:
            mapped_name = str(d.op_name)
        base = d.op_name.split(" (", 1)[0]
        if mapped_name == d.op_name:
            unknown[base] += 1
        else:
            mapped[base] += 1
    return {
        "unknown_raw_name_count": len(unknown),
        "unknown_raw_names_top": unknown.most_common(80),
        "mapped_raw_name_count": len(mapped),
    }


def build_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Detailed Parity Analysis")
    lines.append("")
    lines.append(f"- Timestamp: {report['timestamp']}")
    lines.append(f"- Model: `{report['model_uri']}`")
    lines.append(f"- Family: `{report['family']}`")
    lines.append(f"- run_dir: `{report['model_dir']}`")
    lines.append("")
    lines.append("## Command Status")
    lines.append(f"- ck_run rc: {report['commands'].get('ck_run_rc')}")
    lines.append(f"- llama exhaustive rc: {report['commands'].get('llama_exhaustive_rc')}")
    lines.append("")
    lines.append("## Dump Coverage")
    ck_cov = report.get("ck_coverage")
    if ck_cov is not None:
        lines.append(
            f"- CK dump points: expected={ck_cov['expected_points']} captured={ck_cov['captured_points']} "
            f"coverage={100.0 * ck_cov['coverage_ratio']:.1f}% missing={ck_cov['missing_count']}"
        )
    ck_dump = report.get("ck_dump")
    if ck_dump is not None:
        lines.append(
            f"- CK tensors={ck_dump['tensor_count']} unique_ops={ck_dump['unique_ops']} "
            f"NaN tensors={ck_dump['nan_tensors']} Inf tensors={ck_dump['inf_tensors']}"
        )
    llama_dump = report.get("llama_dump")
    if llama_dump is not None:
        lines.append(
            f"- llama tensors={llama_dump['tensor_count']} unique_ops={llama_dump['unique_ops']} "
            f"NaN tensors={llama_dump['nan_tensors']} Inf tensors={llama_dump['inf_tensors']}"
        )
    lines.append("")
    lines.append("## Weights Audit")
    w = report.get("weights")
    if w is not None:
        lines.append(
            f"- entries={w.get('entries')} missing_required={w.get('missing_required_count')} "
            f"duplicates={w.get('duplicate_name_count')} offsets_ok={w.get('offsets_ok')} "
            f"manifest_exceeds_bump={w.get('manifest_exceeds_bump')}"
        )
    else:
        lines.append("- not available")
    lines.append("")
    lines.append("## First Divergence")
    parity = report.get("parity")
    if parity is not None:
        pf = parity.get("prefill_first_issue")
        dc = parity.get("decode_first_issue")
        lines.append(f"- prefill: {pf if pf else 'none'}")
        lines.append(f"- decode: {dc if dc else 'none'}")
    else:
        lines.append("- not available")
    lines.append("")
    lines.append("## Notes")
    for n in report.get("notes", []):
        lines.append(f"- {n}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Detailed parity analysis (wrapper around ck_run + exhaustive llama dump)")
    ap.add_argument("--model-uri", required=True, help="GGUF path or hf:// URI")
    ap.add_argument("--family", choices=sorted(PARITY_MODEL_MAP.keys()), default="gemma")
    ap.add_argument("--prompt", default="Hello")
    ap.add_argument("--context-len", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=1)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--force-compile", action="store_true")
    ap.add_argument("--skip-ck-run", action="store_true", help="Reuse current run-dir outputs")
    ap.add_argument("--skip-exhaustive-llama", action="store_true", help="Skip second brute-force llama dump pass")
    ap.add_argument("--llama-timeout", type=int, default=0, help="0 disables timeout")
    ap.add_argument("--llama-filter", default=None, help="Optional explicit CKDMP_FILTER for exhaustive pass")
    ap.add_argument("--llama-layer", type=int, default=None, help="Optional CKDMP_LAYER for exhaustive pass")
    ap.add_argument("--llama-stop-after", type=int, default=None, help="Optional CKDMP_STOP_AFTER for exhaustive pass")
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
    ap.add_argument("--ck-run-arg", action="append", default=[], help="Extra arg to pass through to ck_run (repeatable)")
    ap.add_argument("--report-prefix", default="detailed_parity_analysis")
    # Unknown args are forwarded to ck_run, which makes this wrapper easy to use:
    #   python .../detailed_parity_analysis.py --model-uri ... -- --llama-filter attn_norm,Qcur
    args, passthrough_args = ap.parse_known_args()
    if passthrough_args and passthrough_args[0] == "--":
        passthrough_args = passthrough_args[1:]

    model_dir = infer_model_dir(args.model_uri, args.output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    parity_family = PARITY_MODEL_MAP[args.family]

    report: dict[str, Any] = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model_uri": args.model_uri,
        "family": args.family,
        "model_dir": str(model_dir),
        "commands": {},
        "notes": [],
    }

    if not args.skip_ck_run:
        ck_cmd = [
            sys.executable,
            str(CK_RUN),
            "run",
            args.model_uri,
            "--detailed-llamacpp-parity",
            "--prompt",
            args.prompt,
            "--max-tokens",
            str(args.max_tokens),
            "--context-len",
            str(args.context_len),
        ]
        if args.llama_require_token_aware_dumps:
            ck_cmd.append("--llama-require-token-aware-dumps")
        if not args.llama_allow_raw_fallback:
            ck_cmd.append("--llama-no-raw-fallback")
        if args.force_compile:
            ck_cmd.append("--force-compile")
        run_dir = infer_run_dir_from_output_dir(args.output_dir)
        if run_dir is not None:
            ck_cmd.extend(["--run", str(run_dir)])
        for x in args.ck_run_arg:
            ck_cmd.append(x)
        for x in passthrough_args:
            ck_cmd.append(x)
        report["commands"]["ck_run_cmd"] = ck_cmd
        ck_proc = run_cmd(ck_cmd)
        report["commands"]["ck_run_rc"] = ck_proc.returncode
        report["commands"]["ck_run_stdout_tail"] = ck_proc.stdout[-5000:]
        report["commands"]["ck_run_stderr_tail"] = ck_proc.stderr[-3000:]
        if ck_proc.returncode != 0:
            report["notes"].append("ck_run failed")
            out_json = model_dir / f"{args.report_prefix}.json"
            out_md = model_dir / f"{args.report_prefix}.md"
            out_json.write_text(json.dumps(jsonable(report), indent=2), encoding="utf-8")
            out_md.write_text(build_markdown(report), encoding="utf-8")
            print(f"[analysis] wrote {out_json}")
            print(f"[analysis] wrote {out_md}")
            return ck_proc.returncode
    else:
        report["commands"]["ck_run_rc"] = None
        report["notes"].append("Skipped ck_run; used existing dumps.")

    if not args.skip_exhaustive_llama:
        llama_bin = find_llama_binary()
        gguf = find_gguf(model_dir, args.model_uri)
        if llama_bin is None or gguf is None:
            report["commands"]["llama_exhaustive_rc"] = None
            report["notes"].append("Skipped exhaustive llama pass (llama binary or GGUF not found).")
        else:
            ref_dir = model_dir / "llama_parity_dumps"
            ref_dir.mkdir(parents=True, exist_ok=True)
            for f in (ref_dir / "dump.bin", ref_dir / "index.json"):
                if f.exists():
                    f.unlink()
            env = os.environ.copy()
            env["CKDMP_DIR"] = str(ref_dir)
            env["CKDMP_ALL_LAYERS"] = "1"
            env["CKDMP_INCLUDE_GLOBAL"] = "1"
            if args.llama_filter is None:
                env.pop("CKDMP_FILTER", None)
            else:
                env["CKDMP_FILTER"] = args.llama_filter
            if args.llama_layer is None:
                env.pop("CKDMP_LAYER", None)
            else:
                env["CKDMP_LAYER"] = str(args.llama_layer)
            if args.llama_stop_after is None:
                env.pop("CKDMP_STOP_AFTER", None)
            else:
                env["CKDMP_STOP_AFTER"] = str(args.llama_stop_after)

            llama_cmd = [
                str(llama_bin),
                "-m",
                str(gguf),
                "-p",
                args.prompt,
            ]
            exe_name = llama_bin.name.lower()
            if "llama-cli" in exe_name:
                llama_cmd.extend(["-no-cnv", "--single-turn", "--simple-io"])
            elif ("llama-completion" in exe_name) or (exe_name == "main") or ("llama-parity" in exe_name):
                llama_cmd.extend(["-no-cnv", "--simple-io"])
            else:
                llama_cmd.extend(["--simple-io"])
            llama_cmd.extend(
                [
                    "--no-warmup",
                    "--temp",
                    "0",
                    "-n",
                    str(args.max_tokens),
                ]
            )
            if args.context_len > 0:
                llama_cmd.extend(["--ctx-size", str(args.context_len)])
            timeout = None if args.llama_timeout <= 0 else args.llama_timeout
            report["commands"]["llama_exhaustive_cmd"] = llama_cmd
            llama_proc = run_cmd(llama_cmd, cwd=model_dir, env=env, timeout=timeout)
            report["commands"]["llama_exhaustive_rc"] = llama_proc.returncode
            report["commands"]["llama_exhaustive_stdout_tail"] = llama_proc.stdout[-3000:]
            report["commands"]["llama_exhaustive_stderr_tail"] = llama_proc.stderr[-5000:]
            ref_dump = ref_dir / "dump.bin"
            ref_index = ref_dir / "index.json"
            has_ckdmp_dump = ref_dump.exists() and ref_dump.stat().st_size > 0
            token_aware_ok = False
            token_audit_msg = "no CKDMP dump"
            if has_ckdmp_dump:
                token_aware_ok, token_audit_msg = token_audit_from_index(ref_index)
            report["llama_token_audit"] = {
                "require_token_aware_dumps": bool(args.llama_require_token_aware_dumps),
                "ok": bool(token_aware_ok),
                "message": token_audit_msg,
            }

            needs_fallback = (not has_ckdmp_dump) or (
                bool(args.llama_require_token_aware_dumps) and not token_aware_ok
            )
            if needs_fallback and not args.llama_allow_raw_fallback:
                report["notes"].append(
                    "Token-aware llama CKDMP dump required and raw fallback disabled."
                )
            elif needs_fallback and args.llama_allow_raw_fallback:
                # Optional fallback for local llama.cpp builds with LLAMA_DUMP_LAYER0 raw dumps.
                raw_dir = model_dir / "llama_dump"
                for stale in raw_dir.glob("*.bin"):
                    try:
                        stale.unlink()
                    except OSError:
                        pass
                env_raw = os.environ.copy()
                env_raw["LLAMA_DUMP_LAYER0"] = "1"
                llama_raw_proc = run_cmd(llama_cmd, cwd=model_dir, env=env_raw, timeout=timeout)
                report["commands"]["llama_raw_rc"] = llama_raw_proc.returncode
                report["commands"]["llama_raw_stdout_tail"] = llama_raw_proc.stdout[-3000:]
                report["commands"]["llama_raw_stderr_tail"] = llama_raw_proc.stderr[-5000:]

                raw_bins = sorted(raw_dir.glob("*.bin"))
                converter = SCRIPT_DIR / "parity" / "llama_to_ckdmp_converter.py"
                if raw_bins and converter.exists():
                    conv_cmd = [
                        sys.executable,
                        str(converter),
                        "-i",
                        str(raw_dir),
                        "-o",
                        str(ref_dir / "dump.bin"),
                        "--model",
                        parity_family,
                        "--index",
                    ]
                    conv_proc = run_cmd(conv_cmd, cwd=ROOT)
                    report["commands"]["llama_raw_convert_cmd"] = conv_cmd
                    report["commands"]["llama_raw_convert_rc"] = conv_proc.returncode
                    report["commands"]["llama_raw_convert_stdout_tail"] = conv_proc.stdout[-2000:]
                    report["commands"]["llama_raw_convert_stderr_tail"] = conv_proc.stderr[-2000:]
                    if (ref_dir / "dump.bin").exists() and (ref_dir / "dump.bin").stat().st_size > 0:
                        ok2, msg2 = token_audit_from_index(ref_index)
                        report["llama_token_audit_after_fallback"] = {
                            "ok": bool(ok2),
                            "message": msg2,
                        }

            ref_dump = ref_dir / "dump.bin"
            if llama_proc.returncode != 0 and not (ref_dump.exists() and ref_dump.stat().st_size > 0):
                report["notes"].append("Exhaustive llama pass failed.")
    else:
        report["commands"]["llama_exhaustive_rc"] = None
        report["notes"].append("Skipped exhaustive llama pass by request.")

    ck_dump = model_dir / "ck_parity_dumps" / "dump.bin"
    ref_dump = model_dir / "llama_parity_dumps" / "dump.bin"
    if not ck_dump.exists():
        report["notes"].append(f"Missing CK dump: {ck_dump}")
    if not ref_dump.exists():
        report["notes"].append(f"Missing llama dump: {ref_dump}")

    parity_mod = load_parity_module()
    ck_dumps = parity_mod.read_dump_file(ck_dump) if ck_dump.exists() else []
    ref_dumps = parity_mod.read_dump_file(ref_dump) if ref_dump.exists() else []

    report["ck_dump"] = summarize_dump(ck_dumps)
    report["llama_dump"] = summarize_dump(ref_dumps)

    model_c = model_dir / "model_v7.c"
    if not model_c.exists():
        alt_model_c = model_dir / ".ck_build" / "model_v7.c"
        if alt_model_c.exists():
            model_c = alt_model_c
    expected = parse_expected_ck_dump_points(model_c)
    report["ck_coverage"] = ck_coverage_report(expected, ck_dumps)

    report["weights"] = check_weights(model_dir, args.family)
    report["llama_name_mapping"] = llama_mapping_gaps(parity_mod, ref_dumps, parity_family)

    prefill_order = load_lowered_order(model_dir, "prefill")
    decode_order = load_lowered_order(model_dir, "decode")
    prefill_rc, prefill_results = parity_mod.run_parity_test(
        ck_dump_path=ck_dump,
        ref_dump_path=ref_dump,
        atol=1e-4,
        rtol=1e-3,
        verbose=False,
        model_family=parity_family,
        pass_filter="prefill",
    ) if (ck_dump.exists() and ref_dump.exists()) else (1, [])
    decode_rc, decode_results = parity_mod.run_parity_test(
        ck_dump_path=ck_dump,
        ref_dump_path=ref_dump,
        atol=1e-4,
        rtol=1e-3,
        verbose=False,
        model_family=parity_family,
        pass_filter="decode",
    ) if (ck_dump.exists() and ref_dump.exists()) else (1, [])

    def sum_status(rows: list[dict[str, Any]]) -> dict[str, int]:
        c = Counter(str(r.get("status", "")) for r in rows)
        return {k: int(v) for k, v in sorted(c.items())}

    report["parity"] = {
        "prefill_rc": int(prefill_rc),
        "decode_rc": int(decode_rc),
        "prefill_summary": sum_status(prefill_results),
        "decode_summary": sum_status(decode_results),
        "prefill_first_issue": first_issue(prefill_results, prefill_order),
        "decode_first_issue": first_issue(decode_results, decode_order),
        "prefill_fail_examples": [r for r in prefill_results if r.get("status") in ("FAIL", "ERROR")][:40],
        "decode_fail_examples": [r for r in decode_results if r.get("status") in ("FAIL", "ERROR")][:40],
    }

    out_json = model_dir / f"{args.report_prefix}.json"
    out_md = model_dir / f"{args.report_prefix}.md"
    out_json.write_text(json.dumps(jsonable(report), indent=2), encoding="utf-8")
    out_md.write_text(build_markdown(report), encoding="utf-8")

    print("[analysis] Detailed parity analysis complete")
    print(f"[analysis] JSON: {out_json}")
    print(f"[analysis] Markdown: {out_md}")
    print(
        "[analysis] CK coverage: "
        f"{report['ck_coverage']['captured_points']}/{report['ck_coverage']['expected_points']} "
        f"({100.0 * report['ck_coverage']['coverage_ratio']:.1f}%)"
    )
    print(
        "[analysis] Prefill first issue: "
        f"{report['parity']['prefill_first_issue'] if report['parity']['prefill_first_issue'] else 'none'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
