#!/usr/bin/env python3
"""
Unified model-family autocheck for v6.6 (Gemma/Qwen/Qwen3/Mixtral/Llama/Mistral).

Pipeline:
1) E2E run check: does C kernel generate tokens?
2) If tokens generated: simple garble check
3) Detailed parity autopsy (CK vs llama.cpp): first divergence + coverage + weights audit
4) Targeted contract probes based on first failing op
5) Stitched JSON + Markdown report
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from parity.probe_defaults import (
    getenv_autocheck_probe_token_id,
    getenv_prefill_tokens_csv,
)


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
CK_RUN = SCRIPT_DIR / "ck_run_v6_6.py"
DETAIL = SCRIPT_DIR / "detailed_parity_analysis.py"
QKV_LAYOUT = SCRIPT_DIR / "check_qkv_layout_match.py"
CLI_BIN = ROOT / "build" / "ck-cli-v6.6"


def _repo_script_path(name: str) -> Path:
    new_path = SCRIPT_DIR / "parity" / name
    if new_path.exists():
        return new_path
    # Backward compatibility for pre-move local trees.
    old_gemma = ROOT / "scripts" / "gemma" / name
    if old_gemma.exists():
        return old_gemma
    return ROOT / "scripts" / name


QPROJ_CONTRACT = _repo_script_path("check_qproj_contract.py")
FFN_NORM_CONTRACT = _repo_script_path("check_ffn_norm_contract.py")
POST_ATTN_CHAIN = _repo_script_path("check_post_attn_chain.py")
POST_ATTN_CHAIN_PREFILL = _repo_script_path("check_post_attn_chain_prefill.py")


SUPPORTED_FAMILIES = {
    "gemma": "gemma",
    "qwen": "qwen",
    "qwen2": "qwen2",
    "qwen3": "qwen3",
    "mixtral": "mistral",
    "mistral": "mistral",
    "llama": "llama",
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
        text=True,
        capture_output=True,
        check=False,
    )


def strip_ansi(s: str) -> str:
    return re.sub(r"\x1B\[[0-9;]*[A-Za-z]", "", s)


def infer_model_dir(model_uri: str, output_dir: Path | None) -> Path:
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


def extract_generated_text(raw: str) -> str:
    text = strip_ansi(raw)
    lines = [ln.rstrip() for ln in text.splitlines()]
    out: list[str] = []
    skip_prefixes = (
        "C-Kernel-Engine",
        "Loading:",
        "Initializing",
        "Ready!",
        "Type /help",
        "[Runtime]",
        "[OpenMP]",
        "[Tokenizer]",
        "prefill ",
        "decode ",
        "sample ",
        "total ",
        "prompt eval:",
    )
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if s.startswith(skip_prefixes):
            continue
        if s.startswith("[") and s.endswith("]"):
            continue
        if s.startswith("You:") or s.startswith("Assistant:"):
            continue
        out.append(s)
    return "\n".join(out).strip()


def garble_check(text: str) -> dict[str, Any]:
    if not text:
        return {"garbled": False, "reason": "empty-text", "confidence": 0.0}

    chars = [c for c in text if not c.isspace()]
    if not chars:
        return {"garbled": False, "reason": "whitespace-only", "confidence": 0.0}

    n = len(chars)
    non_printable = sum(1 for c in chars if not c.isprintable())
    replacement = text.count("\ufffd")
    alnum = sum(1 for c in chars if c.isalnum())
    symbol = n - alnum

    longest_run = 1
    run = 1
    for i in range(1, n):
        if chars[i] == chars[i - 1]:
            run += 1
            longest_run = max(longest_run, run)
        else:
            run = 1

    unique_ratio = len(set(chars)) / float(n)
    non_print_ratio = non_printable / float(n)
    symbol_ratio = symbol / float(n)

    flags = []
    if replacement > 0:
        flags.append("replacement-char")
    if non_print_ratio > 0.08:
        flags.append("non-printable")
    if longest_run >= 20:
        flags.append("long-char-run")
    if n >= 40 and symbol_ratio > 0.85:
        flags.append("mostly-symbols")
    if n >= 40 and unique_ratio < 0.05:
        flags.append("very-low-entropy")

    score = 0.0
    score += 0.6 if replacement > 0 else 0.0
    score += min(0.4, non_print_ratio * 2.0)
    score += 0.2 if longest_run >= 20 else 0.0
    score += 0.2 if (n >= 40 and symbol_ratio > 0.85) else 0.0
    score += 0.2 if (n >= 40 and unique_ratio < 0.05) else 0.0
    score = min(1.0, score)

    return {
        "garbled": bool(flags),
        "reason": ",".join(flags) if flags else "looks-readable",
        "confidence": score,
        "stats": {
            "length": n,
            "non_printable_ratio": non_print_ratio,
            "replacement_chars": replacement,
            "symbol_ratio": symbol_ratio,
            "unique_ratio": unique_ratio,
            "longest_run": longest_run,
        },
    }


def parse_decode_tokens(combined_output: str) -> int:
    txt = strip_ansi(combined_output)
    # ck-cli stats: "... | decode X tok ..."
    m = re.search(r"\bdecode\s+(\d+)\s+tok\b", txt)
    if m:
        return int(m.group(1))
    # fallback wording
    m = re.search(r"Generated\s+(\d+)\s+tokens", txt)
    if m:
        return int(m.group(1))
    return 0


def run_ck_cli_once(model_dir: Path, prompt: str, max_tokens: int, timeout: int) -> dict[str, Any]:
    lib = model_dir / "libmodel.so"
    weights = model_dir / "weights.bump"
    if not CLI_BIN.exists():
        return {"ok": False, "error": f"missing CLI binary: {CLI_BIN}"}
    if not lib.exists():
        return {"ok": False, "error": f"missing model library: {lib}"}
    if not weights.exists():
        return {"ok": False, "error": f"missing weights: {weights}"}

    env = os.environ.copy()
    ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{ROOT / 'build'}:{model_dir}:{ld}"
    cmd = [
        str(CLI_BIN),
        "--lib",
        str(lib),
        "--weights",
        str(weights),
        "--prompt",
        prompt,
        "--max-tokens",
        str(max_tokens),
        "--temperature",
        "0",
        "--top-p",
        "1.0",
        "--no-chat-template",
        "--timing",
    ]
    proc = run_cmd(cmd, env=env, timeout=None if timeout <= 0 else timeout)
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    decode_tokens = parse_decode_tokens(combined)
    generated_text = extract_generated_text(proc.stdout or "")
    g = garble_check(generated_text)
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "cmd": cmd,
        "decode_tokens": decode_tokens,
        "generated_text": generated_text,
        "generated_text_preview": generated_text[:400],
        "garble": g,
        "stdout_tail": (proc.stdout or "")[-5000:],
        "stderr_tail": (proc.stderr or "")[-5000:],
    }


def parse_float_metric(text: str, key: str) -> float | None:
    m = re.search(rf"{re.escape(key)}\s*:\s*([0-9.eE+\-]+)", text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def run_probe(cmd: list[str], timeout: int = 120) -> dict[str, Any]:
    proc = run_cmd(cmd, timeout=timeout)
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "max_diff": parse_float_metric(out, "max_diff"),
        "mean_diff": parse_float_metric(out, "mean_diff"),
        "stdout_tail": (proc.stdout or "")[-3000:],
        "stderr_tail": (proc.stderr or "")[-2000:],
    }


def summarize_root_cause(report: dict[str, Any]) -> list[str]:
    hints: list[str] = []
    e2e = report["e2e"]
    parity = report.get("parity_detail", {})

    if not e2e.get("ok", False):
        hints.append("E2E CLI run failed: runtime/library issue before parity.")
        return hints

    if e2e.get("decode_tokens", 0) <= 0:
        hints.append("No generated tokens: likely C-kernel execution bug in prefill/decode path.")
        return hints

    if e2e.get("garble", {}).get("garbled"):
        hints.append(f"Garbled output detected ({e2e['garble']['reason']}).")

    weights = parity.get("weights", {})
    if weights and not weights.get("ok", True):
        hints.append("Weights/layout audit failed: likely manifest/layout/dtype wiring issue.")

    first = parity.get("parity", {}).get("prefill_first_issue")
    if first:
        op = str(first.get("op", ""))
        hints.append(f"First numerical divergence at op={op}, layer={first.get('layer')}.")
        if op in ("q_proj", "k_proj", "v_proj"):
            hints.append("Projection first-fail: check kernel input/output contract and QKV layout mapping.")
        elif op in ("attn_proj", "attn_sliding", "qk_norm", "rope_qk"):
            hints.append("Attention stack first-fail: projection may pass, inspect qk_norm/rope/attention kernel path.")
        elif op in ("ffn_norm", "mlp_gate", "mlp_down"):
            hints.append("FFN stack first-fail: inspect post-attn residual + ffn_norm + MLP contracts.")
        elif op in ("final_norm", "logits"):
            hints.append("Footer first-fail: likely earlier hidden-state drift or output projection mismatch.")

    probes = report.get("probes", {})
    q = probes.get("q_proj_contract", {})
    k = probes.get("k_proj_contract", {})
    v = probes.get("v_proj_contract", {})
    if any(p.get("returncode", 1) == 0 for p in (q, k, v)):
        means = [
            p.get("mean_diff")
            for p in (q, k, v)
            if p.get("returncode", 1) == 0 and p.get("mean_diff") is not None
        ]
        if means and max(means) < 1e-3:
            hints.append("Projection contracts look good: divergence likely after projections.")

    layout = probes.get("qkv_layout", {})
    if layout.get("returncode", 1) == 0 and "likely layout mismatch" in layout.get("stdout_tail", ""):
        hints.append("QKV layout probe suggests layout mismatch against llama dump ordering.")

    if not hints:
        hints.append("No hard root-cause bucket found; inspect detailed parity fail examples.")
    return hints


def build_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Model Autocheck Report")
    lines.append("")
    lines.append(f"- Timestamp: {report['timestamp']}")
    lines.append(f"- Model URI: `{report['model_uri']}`")
    lines.append(f"- Family: `{report['family']}`")
    lines.append(f"- Model dir: `{report['model_dir']}`")
    lines.append("")
    lines.append("## E2E")
    e2e = report["e2e"]
    lines.append(f"- CLI return code: {e2e.get('returncode')}")
    lines.append(f"- Decode tokens: {e2e.get('decode_tokens')}")
    lines.append(f"- Garbled: {e2e.get('garble', {}).get('garbled')} ({e2e.get('garble', {}).get('reason')})")
    lines.append("")
    lines.append("## Parity")
    p = report.get("parity_detail", {})
    first = p.get("parity", {}).get("prefill_first_issue")
    cov = p.get("ck_coverage", {})
    lines.append(f"- CK coverage: {cov.get('captured_points')}/{cov.get('expected_points')} ({100.0 * cov.get('coverage_ratio', 0.0):.1f}%)")
    lines.append(f"- First prefill issue: {first if first else 'none'}")
    w = p.get("weights", {})
    if w:
        lines.append(f"- Weights ok: {w.get('ok')} (missing_required={w.get('missing_required_count')})")
    lines.append("")
    lines.append("## Root Cause Hints")
    for h in report.get("root_cause_hints", []):
        lines.append(f"- {h}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Family-agnostic v6.6 model autocheck")
    ap.add_argument("--model-uri", required=True, help="GGUF path or hf:// URI")
    ap.add_argument("--family", required=True, choices=sorted(SUPPORTED_FAMILIES.keys()))
    ap.add_argument("--prompt", default="Hello")
    ap.add_argument("--context-len", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--force-compile", action="store_true")
    ap.add_argument("--skip-ck-run", action="store_true")
    ap.add_argument("--skip-detail", action="store_true")
    ap.add_argument("--timeout-cli", type=int, default=120)
    ap.add_argument("--report-prefix", default="model_autocheck")
    ap.add_argument(
        "--probe-token",
        type=int,
        default=getenv_autocheck_probe_token_id(),
        help="decode token used by targeted contract probes",
    )
    ap.add_argument(
        "--probe-prefill-tokens",
        default=getenv_prefill_tokens_csv(),
        help="comma-separated token ids used for prefill probe contracts",
    )
    ap.add_argument("--ck-run-arg", action="append", default=[], help="extra arg forwarded to ck_run (repeatable)")
    args, passthrough = ap.parse_known_args()
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    family_norm = SUPPORTED_FAMILIES[args.family]
    model_dir = infer_model_dir(args.model_uri, args.output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model_uri": args.model_uri,
        "family": args.family,
        "family_norm": family_norm,
        "model_dir": str(model_dir),
        "commands": {},
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
            "--context-len",
            str(args.context_len),
            "--max-tokens",
            str(max(1, args.max_tokens)),
        ]
        if args.force_compile:
            ck_cmd.append("--force-compile")
        if args.output_dir is not None:
            ck_cmd.extend(["--output-dir", str(args.output_dir)])
        ck_cmd.extend(args.ck_run_arg)
        ck_cmd.extend(passthrough)
        report["commands"]["ck_run"] = ck_cmd
        ck_proc = run_cmd(ck_cmd)
        report["commands"]["ck_run_rc"] = ck_proc.returncode
        report["commands"]["ck_run_stdout_tail"] = (ck_proc.stdout or "")[-4000:]
        report["commands"]["ck_run_stderr_tail"] = (ck_proc.stderr or "")[-3000:]
        if ck_proc.returncode != 0:
            report["status"] = "ck_run_failed"
            out_json = model_dir / f"{args.report_prefix}.json"
            out_md = model_dir / f"{args.report_prefix}.md"
            out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
            out_md.write_text(build_markdown(report), encoding="utf-8")
            print(f"[autocheck] wrote {out_json}")
            print(f"[autocheck] wrote {out_md}")
            return ck_proc.returncode
    else:
        report["commands"]["ck_run_rc"] = None

    e2e = run_ck_cli_once(model_dir, args.prompt, args.max_tokens, args.timeout_cli)
    report["e2e"] = e2e

    if args.skip_detail:
        report["parity_detail"] = {}
    else:
        detail_cmd = [
            sys.executable,
            str(DETAIL),
            "--model-uri",
            args.model_uri,
            "--family",
            family_norm,
            "--prompt",
            args.prompt,
            "--context-len",
            str(args.context_len),
            "--max-tokens",
            str(max(1, args.max_tokens)),
            "--skip-ck-run",
        ]
        if args.output_dir is not None:
            detail_cmd.extend(["--output-dir", str(args.output_dir)])
        report["commands"]["detail"] = detail_cmd
        d_proc = run_cmd(detail_cmd)
        report["commands"]["detail_rc"] = d_proc.returncode
        report["commands"]["detail_stdout_tail"] = (d_proc.stdout or "")[-4000:]
        report["commands"]["detail_stderr_tail"] = (d_proc.stderr or "")[-3000:]

        detail_json = model_dir / "detailed_parity_analysis.json"
        if detail_json.exists():
            report["parity_detail"] = json.loads(detail_json.read_text(encoding="utf-8"))
        else:
            report["parity_detail"] = {}

    first_op = None
    pfirst = report.get("parity_detail", {}).get("parity", {}).get("prefill_first_issue")
    if isinstance(pfirst, dict):
        first_op = str(pfirst.get("op"))

    probes: dict[str, Any] = {}
    if first_op in ("q_proj", "k_proj", "v_proj", "attn_proj", "attn_sliding", "qk_norm", "rope_qk"):
        probe_tok = str(args.probe_token)
        for op in ("q_proj", "k_proj", "v_proj"):
            cmd = [
                sys.executable,
                str(QPROJ_CONTRACT),
                "--model-dir",
                str(model_dir),
                "--op",
                op,
                "--layer",
                "0",
                "--token",
                probe_tok,
            ]
            probes[f"{op}_contract"] = run_probe(cmd)
        if QKV_LAYOUT.exists():
            cmd = [sys.executable, str(QKV_LAYOUT), "--ck-dump", str(model_dir / "ck_parity_dumps/dump.bin"), "--ref-dump", str(model_dir / "llama_parity_dumps/dump.bin"), "--layer", "0", "--token", "0"]
            probes["qkv_layout"] = run_probe(cmd)

    if first_op in ("ffn_norm", "mlp_gate", "mlp_down"):
        if FFN_NORM_CONTRACT.exists():
            cmd = [
                sys.executable,
                str(FFN_NORM_CONTRACT),
                "--model-dir",
                str(model_dir),
                "--layer",
                "0",
                "--token",
                str(args.probe_token),
            ]
            probes["ffn_norm_contract"] = run_probe(cmd)

    if first_op in ("post_attention_norm", "layer_out", "attn_proj"):
        if POST_ATTN_CHAIN.exists():
            cmd = [
                sys.executable,
                str(POST_ATTN_CHAIN),
                "--model-dir",
                str(model_dir),
                "--layer",
                "0",
                "--token",
                str(args.probe_token),
            ]
            probes["post_attn_chain_decode"] = run_probe(cmd)
        if POST_ATTN_CHAIN_PREFILL.exists():
            cmd = [
                sys.executable,
                str(POST_ATTN_CHAIN_PREFILL),
                "--model-dir",
                str(model_dir),
                "--layer",
                "0",
                "--tokens",
                args.probe_prefill_tokens,
            ]
            probes["post_attn_chain_prefill"] = run_probe(cmd)

    report["probes"] = probes
    report["root_cause_hints"] = summarize_root_cause(report)
    report["status"] = "ok"

    out_json = model_dir / f"{args.report_prefix}.json"
    out_md = model_dir / f"{args.report_prefix}.md"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(build_markdown(report), encoding="utf-8")

    print("[autocheck] complete")
    print(f"[autocheck] report: {out_json}")
    print(f"[autocheck] summary: {out_md}")
    print(f"[autocheck] e2e decode_tokens={e2e.get('decode_tokens')} garbled={e2e.get('garble', {}).get('garbled')}")
    if pfirst:
        print(f"[autocheck] first divergence: {pfirst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
