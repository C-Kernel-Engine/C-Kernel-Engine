#!/usr/bin/env python3
"""
Automated parity autopsy driver for v6.6.

Workflow:
1) (Optional) Run ck_run_v6_6.py with detailed llama.cpp parity dumps.
2) Run parity_test.py --json (prefill by default).
3) Find first meaningful divergence using lowered op order.
4) Run projection contract checks (q/k/v) and summarize likely root-cause stage.
5) Write JSON + markdown report.
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
V66_SCRIPT_DIR = SCRIPT_DIR.parent
CK_RUN = V66_SCRIPT_DIR / "ck_run_v6_6.py"
PARITY_TEST = V66_SCRIPT_DIR / "parity_test.py"
PROJ_CHECK = SCRIPT_DIR / "check_qproj_contract.py"
SUPPORTED_OPS = ("q_proj", "k_proj", "v_proj")


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
        "parity_model": "qwen2",  # reuse qwen2 mapping in parity_test.py
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
    spec = importlib.util.spec_from_file_location("ck_v66_parity_test", str(PARITY_TEST))
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


def pick_first_issue(results: list[dict[str, Any]], order: dict[tuple[int, str], int]) -> dict[str, Any] | None:
    bad = [r for r in results if r.get("status") in ("FAIL", "ERROR")]
    if not bad:
        return None

    def rank(r: dict[str, Any]) -> tuple[int, int, int]:
        layer = int(r.get("layer", 10**9))
        op = str(r.get("op", ""))
        seq = order.get((layer, op), 10**9)
        token = int(r.get("token", 10**9))
        return (layer, seq, token)

    return sorted(bad, key=rank)[0]


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


def infer_model_dir(output_dir: Path | None, model_uri: str) -> Path:
    if output_dir is not None:
        return output_dir
    if model_uri.startswith("hf://"):
        repo = model_uri[len("hf://") :].rsplit("/", 1)[0]
        key = repo.replace("/", "--")
        return Path.home() / ".cache" / "ck-engine-v6.6" / "models" / key / "ck_build"
    p = Path(model_uri).expanduser().resolve()
    if p.is_dir():
        return p
    return p.parent / "ck_build"


def resolve_hf_cache_gguf(model_uri: str) -> Path | None:
    """Resolve hf://repo/path/file.gguf to the local ck cache GGUF path."""
    if not model_uri.startswith("hf://"):
        return None
    body = model_uri[len("hf://") :]
    if "/" not in body:
        return None
    repo_path, filename = body.rsplit("/", 1)
    repo_key = repo_path.replace("/", "--")
    p = Path.home() / ".cache" / "ck-engine-v6.6" / "models" / repo_key / filename
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


def build_report_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Parity Autopsy Report")
    lines.append("")
    lines.append(f"- Timestamp: {report['timestamp']}")
    lines.append(f"- Family: `{report['family']}`")
    lines.append(f"- Model URI: `{report['model_uri']}`")
    lines.append(f"- Model dir: `{report['model_dir']}`")
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
    if isinstance(first, dict):
        lines.append(
            f"- First issue: layer={first.get('layer')} op={first.get('op')} token={first.get('token')} "
            f"status={first.get('status')}"
        )
    else:
        lines.append("- First issue: none")
    lines.append("")
    lines.append("## Projection Contract Checks")
    for op, c in report["projection_checks"].items():
        lines.append(
            f"- `{op}`: rc={c['returncode']} pass_contract={c['metrics'].get('pass_contract')} "
            f"max_diff={c['metrics'].get('max_diff')} mean_diff={c['metrics'].get('mean_diff')}"
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
    ap = argparse.ArgumentParser(description="Automated parity autopsy (v6.6)")
    ap.add_argument("--model-uri", required=True, help="GGUF path or hf:// URI")
    ap.add_argument("--family", choices=sorted(FAMILY_PROFILE.keys()), default="gemma")
    ap.add_argument("--context-len", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=1)
    ap.add_argument("--prompt", default="Hello")
    ap.add_argument("--proj-token-id", type=int, default=5, help="token id for projection contract checks")
    ap.add_argument("--output-dir", type=Path, default=None, help="ck_build output dir for ck_run")
    ap.add_argument("--skip-run", action="store_true", help="reuse existing ck_build dumps")
    ap.add_argument("--pass", dest="pass_name", choices=["prefill", "decode"], default="prefill")
    ap.add_argument("--llama-filter", default=None, help="override llama dump filter")
    ap.add_argument("--llama-stop-after", type=int, default=20)
    ap.add_argument("--report-prefix", default="autopsy_report", help="basename for .json/.md report files")
    args = ap.parse_args()

    profile = FAMILY_PROFILE[args.family]
    parity_model = profile["parity_model"]
    llama_filter = args.llama_filter or profile["llama_filter"]
    model_dir = infer_model_dir(args.output_dir, args.model_uri)

    report: dict[str, Any] = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "family": args.family,
        "model_uri": args.model_uri,
        "model_dir": str(model_dir),
        "commands": {},
        "parity": {},
        "projection_checks": {},
        "diagnosis": [],
        "next_actions": [],
    }

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
        if args.output_dir is not None:
            run_cmd_list.extend(["--output-dir", str(args.output_dir)])
        report["commands"]["ck_run"] = run_cmd_list
        ck_proc = run_cmd(run_cmd_list, check=False)
        report["ck_run_returncode"] = ck_proc.returncode
        report["ck_run_stdout_tail"] = ck_proc.stdout[-4000:]
        report["ck_run_stderr_tail"] = ck_proc.stderr[-2000:]
        if ck_proc.returncode != 0:
            report["diagnosis"].append("ck_run_v6_6.py failed; parity not executed.")
            out_json = model_dir / f"{args.report_prefix}.json"
            out_md = model_dir / f"{args.report_prefix}.md"
            out_json.write_text(json.dumps(to_jsonable(report), indent=2), encoding="utf-8")
            out_md.write_text(build_report_markdown(report), encoding="utf-8")
            print(f"[autopsy] wrote {out_json}")
            print(f"[autopsy] wrote {out_md}")
            return 1

    ck_dump = model_dir / "ck_parity_dumps" / "dump.bin"
    ref_dump = model_dir / "llama_parity_dumps" / "dump.bin"
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

    summary = {
        "PASS": sum(1 for r in results if r.get("status") == "PASS"),
        "FAIL": sum(1 for r in results if r.get("status") == "FAIL"),
        "ERROR": sum(1 for r in results if r.get("status") == "ERROR"),
        "WARN": sum(1 for r in results if r.get("status") == "WARN"),
        "TOTAL": len(results),
    }
    order = load_op_order(model_dir, args.pass_name)
    first_issue = pick_first_issue(results, order)
    report["parity"] = {
        "summary": summary,
        "first_issue": first_issue,
    }
    ref_dump_exists = ref_dump.exists() and ref_dump.stat().st_size > 0
    report["parity"]["ref_dump_exists"] = ref_dump_exists

    # Projection deep checks (always run; cheap and high-signal).
    for op in SUPPORTED_OPS:
        cmd = [
            sys.executable,
            str(PROJ_CHECK),
            "--model-dir",
            str(model_dir),
            "--op",
            op,
            "--layer",
            "0",
            "--token",
            str(args.proj_token_id),
        ]
        proc = run_cmd(cmd, check=False)
        report["projection_checks"][op] = {
            "returncode": proc.returncode,
            "metrics": parse_projection_metrics(proc.stdout),
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-800:],
        }

    proj_all_pass = all(
        c.get("metrics", {}).get("pass_contract") for c in report["projection_checks"].values()
    )

    if first_issue is None:
        report["diagnosis"].append("No FAIL/ERROR in parity output.")
    else:
        report["diagnosis"].append(
            f"First parity issue: layer={first_issue.get('layer')} op={first_issue.get('op')} status={first_issue.get('status')}"
        )

    if not ref_dump_exists:
        report["diagnosis"].append("llama reference dump missing/empty; WARN-only parity is not actionable.")
        report["next_actions"].append(
            "Re-run with a GGUF path visible in output dir (or without custom --output-dir) so llama parity can dump references."
        )
    elif summary["PASS"] == 0 and summary["WARN"] > 0:
        report["diagnosis"].append("Parity produced WARN-only results (coverage gap).")
        report["next_actions"].append("Expand --llama-filter/--llama-stop-after to include expected layer-0 ops.")

    if proj_all_pass:
        report["diagnosis"].append(
            "Projection contracts (q/k/v) match local reference; likely divergence is after projections."
        )
        report["next_actions"].append(
            "Focus on qk_norm / rope_qk / attention (sliding or decode kernel) and post-attention dataflow."
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
