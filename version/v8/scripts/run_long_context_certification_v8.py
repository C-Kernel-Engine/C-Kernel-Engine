#!/usr/bin/env python3
"""Run a resumable, capacity-aware long-context model certification sweep."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import re
import shlex
import statistics
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
DEFAULT_CATALOG = SCRIPT_DIR.parent / "regression" / "long_context_models.json"
DEFAULT_PROMPTS = SCRIPT_DIR.parent / "test_assets" / "long_context_quality_prompts.json"
DEFAULT_OUTPUT = ROOT / "build" / "v8-long-context-certification"
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
CK_TIMING_RE = re.compile(
    r"prefill\s+(?P<prompt_tokens>\d+)\s+tok.*?"
    r"(?P<prompt_ms>[0-9.]+)\s+ms\s+(?P<prompt_tok_s>[0-9.]+)\s+tok/s.*?"
    r"decode\s+(?P<decode_tokens>\d+)\s+tok\s+"
    r"(?P<decode_ms>[0-9.]+)\s+ms\s+(?P<decode_tok_s>[0-9.]+)\s+tok/s",
    re.S,
)
RSS_RE = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_schema(path: Path, expected: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != expected or int(payload.get("schema_version", 0)) != 1:
        raise ValueError(f"invalid {expected} document: {path}")
    return payload


def parse_contexts(value: str) -> list[int]:
    contexts = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not contexts or any(item <= 0 for item in contexts):
        raise ValueError("contexts must be positive integers")
    return contexts


def resolve_model(row: dict[str, Any], allow_download: bool) -> tuple[str, str]:
    env_name = str(row.get("model_env", ""))
    configured = os.environ.get(env_name, "").strip() if env_name else ""
    if configured:
        return configured, f"environment:{env_name}"
    default = str(row.get("default_model", "")).strip()
    if default and allow_download:
        return default, "catalog_default"
    return "", f"unset:{env_name or 'model'}"


def resolve_gguf(row: dict[str, Any], model: str, runtime_dir: Path) -> Path | None:
    env_name = str(row.get("gguf_env", ""))
    configured = os.environ.get(env_name, "").strip() if env_name else ""
    if configured:
        path = Path(configured).expanduser()
        return path if path.is_file() else None
    model_path = Path(model).expanduser()
    if model_path.is_file() and model_path.suffix.lower() == ".gguf":
        return model_path
    candidates = sorted(runtime_dir.glob("*.gguf"))
    return candidates[0] if candidates else None


def token_sha256(token_id: int, count: int) -> str:
    value = int(token_id).to_bytes(4, "little", signed=True)
    digest = hashlib.sha256()
    for _ in range(count):
        digest.update(value)
    return digest.hexdigest()


def parse_timing(output: str) -> dict[str, float]:
    clean = ANSI_RE.sub("", output).replace("\r", "")
    match = CK_TIMING_RE.search(clean)
    if not match:
        raise ValueError("native CKE timing line is missing")
    return {
        "prompt_tokens": int(match.group("prompt_tokens")),
        "prompt_ms": float(match.group("prompt_ms")),
        "prompt_tok_s": float(match.group("prompt_tok_s")),
        "decode_tokens": int(match.group("decode_tokens")),
        "decode_ms": float(match.group("decode_ms")),
        "decode_tok_s": float(match.group("decode_tok_s")),
    }


def parse_peak_rss(stderr: str) -> int | None:
    match = RSS_RE.search(stderr)
    return int(match.group(1)) if match else None


def run_command(
    command: list[str],
    *,
    timeout: int,
    env: dict[str, str],
    output_dir: Path,
    name: str,
    timed: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    executed = ["/usr/bin/time", "-v", *command] if timed else command
    started = dt.datetime.now(dt.timezone.utc)
    try:
        completed = subprocess.run(
            executed,
            cwd=ROOT,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
        error = None
    except subprocess.TimeoutExpired as exc:
        returncode = 124
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        error = f"timeout after {timeout}s"
    stdout_path = output_dir / f"{name}.stdout.log"
    stderr_path = output_dir / f"{name}.stderr.log"
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    return {
        "command": command,
        "command_shell": shlex.join(command),
        "returncode": returncode,
        "error": error,
        "started_at": started.isoformat(),
        "elapsed_seconds": (dt.datetime.now(dt.timezone.utc) - started).total_seconds(),
        "peak_rss_kib": parse_peak_rss(stderr) if timed else None,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "stdout": stdout,
        "stderr": stderr,
    }


def provider_summary(runtime_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for phase, filename in (("prefill", "lowered_prefill.json"), ("decode", "lowered_decode.json")):
        path = runtime_dir / filename
        if not path.is_file():
            result[phase] = {"status": "missing", "path": str(path)}
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        operations = [row for row in payload.get("operations", []) if isinstance(row, dict)]
        counts = Counter(str(row.get("kernel", "unknown")) for row in operations)
        selections: Counter[tuple[str, str, str]] = Counter()
        for row in operations:
            contract = row.get("resolved_contract")
            if not isinstance(contract, dict):
                continue
            selections[(
                str(row.get("kernel", "unknown")),
                str(contract.get("contract_id", "")),
                json.dumps(contract.get("selector"), sort_keys=True),
            )] += 1
        result[phase] = {
            "status": "available",
            "operation_count": len(operations),
            "kernel_counts": dict(sorted(counts.items())),
            "contract_selections": [
                {"kernel": kernel, "contract_id": contract_id,
                 "selector": json.loads(selector), "count": count}
                for (kernel, contract_id, selector), count in sorted(selections.items())
            ],
            "path": str(path),
        }
    return result


def host_fingerprint() -> dict[str, Any]:
    cpu_model = ""
    flags: list[str] = []
    try:
        payload = json.loads(subprocess.run(
            ["lscpu", "-J"], text=True, capture_output=True, check=True
        ).stdout)
        fields = {str(row.get("field", "")).rstrip(":"): str(row.get("data", ""))
                  for row in payload.get("lscpu", [])}
        cpu_model = fields.get("Model name", "")
        flags = fields.get("Flags", "").split()
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        pass
    relevant = sorted(flag for flag in flags if flag.startswith(("avx", "amx", "fma")))
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": cpu_model,
        "logical_cpus": os.cpu_count(),
        "isa_flags": relevant,
    }


def median_timing(samples: list[dict[str, Any]]) -> dict[str, float] | None:
    timings = [sample["timing"] for sample in samples if isinstance(sample.get("timing"), dict)]
    if not timings:
        return None
    return {
        key: statistics.median(float(timing[key]) for timing in timings)
        for key in timings[0]
    }


def parse_llama_bench(stdout: str) -> dict[str, float]:
    rows = json.loads(stdout)
    prompt = next((row for row in rows if int(row.get("n_prompt", 0)) > 0), None)
    decode = next((row for row in rows if int(row.get("n_gen", 0)) > 0), None)
    if not prompt or not decode:
        raise ValueError("llama-bench did not return prompt and decode rows")
    return {
        "prompt_tokens": int(prompt["n_prompt"]),
        "prompt_ms": float(prompt["avg_ns"]) / 1_000_000.0,
        "prompt_tok_s": float(prompt["avg_ts"]),
        "decode_tokens": int(decode["n_gen"]),
        "decode_ms": float(decode["avg_ns"]) / 1_000_000.0,
        "decode_tok_s": float(decode["avg_ts"]),
    }


def extract_generated(stdout: str) -> str:
    text = ANSI_RE.sub("", stdout).replace("\r", "")
    marker = "Type /help for commands, Ctrl+C to stop generation"
    if marker in text:
        text = text.split(marker, 1)[1]
    text = re.split(r"\nprefill\s+\d+\s+tok\b", text, maxsplit=1)[0]
    return text.strip()


def code_quality(text: str) -> dict[str, Any]:
    printable = sum(character.isprintable() or character in "\n\r\t" for character in text)
    ratio = printable / max(len(text), 1)
    lowered = text.lower()
    languages = {name: token in lowered for name, token in {
        "c": "#include", "python": "python", "sql": "select"
    }.items()}
    passed = len(text) >= 256 and ratio >= 0.96 and all(languages.values())
    return {
        "pass": passed,
        "characters": len(text),
        "printable_ratio": ratio,
        "language_markers": languages,
    }


def svg_quality(text: str) -> dict[str, Any]:
    sys.path.insert(0, str(SCRIPT_DIR))
    from certify_text_prompt_parity_v8 import evaluate_quality_contract
    return evaluate_quality_contract(text, {
        "kind": "standalone_svg.v1",
        "min_graphic_elements": 8,
        "required_labels": ["tokenize", "prefill", "decode", "detokenize"],
    })


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# v8 Long-Context Certification",
        "",
        f"Generated: `{report['updated_at']}`",
        "",
        "## Performance",
        "",
        "| Model | Context | Status | CKE prefill | llama.cpp prefill | Relative | CKE decode | Peak RAM | First-logit repeatable | Numerical parity |",
        "|---|---:|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in report.get("performance", []):
        cke = row.get("cke") or {}
        llama = row.get("llama_cpp") or {}
        cke_timing = cke.get("timing") or {}
        llama_timing = llama.get("timing") or {}
        ratio = row.get("relative_prefill")
        peak = cke.get("peak_rss_kib")
        lines.append(
            f"| {row['model']} | {row['context_tokens']} | {row['status']} | "
            f"{cke_timing.get('prompt_tok_s', '-')} | {llama_timing.get('prompt_tok_s', '-')} | "
            f"{f'{ratio:.2f}x' if isinstance(ratio, float) else '-'} | "
            f"{cke_timing.get('decode_tok_s', '-')} | "
            f"{f'{peak / 1048576:.2f} GiB' if isinstance(peak, int) else '-'} | "
            f"{cke.get('first_logits_repeatable', '-')} | "
            f"{(row.get('numerical_parity') or {}).get('status', '-')} |"
        )
    lines += ["", "## Quality", "", "| Model | Prompt | Status | Prompt tokens | Output |", "|---|---|---|---:|---|"]
    for row in report.get("quality", []):
        lines.append(
            f"| {row['model']} | {row['prompt_id']} | {row['status']} | "
            f"{(row.get('timing') or {}).get('prompt_tokens', '-')} | `{row.get('output_path', '-')}` |"
        )
    lines += ["", "## Skips And Failures", ""]
    for row in report.get("events", []):
        if row.get("status") in {"SKIP", "FAIL"}:
            lines.append(f"- **{row.get('status')}** `{row.get('model_id')}`: {row.get('reason', 'unspecified')}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def publish(report: dict[str, Any], output_dir: Path) -> None:
    report["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    atomic_json(output_dir / "summary.json", report)
    write_markdown(report, output_dir / "summary.md")


def build_runtime(
    model: str,
    runtime_dir: Path,
    context_tokens: int,
    decode_tokens: int,
    args: argparse.Namespace,
    env: dict[str, str],
    evidence_dir: Path,
) -> dict[str, Any]:
    runtime_context = context_tokens + decode_tokens + 8
    command = [
        sys.executable,
        str(SCRIPT_DIR / "ck_run_v8.py"),
        "run",
        model,
        "--run",
        str(runtime_dir),
        "--context-len",
        str(runtime_context),
        "--prefill-chunk-len",
        str(min(args.prefill_chunk_tokens, context_tokens)),
        "--logits-layout",
        "last",
        "--generate-only",
    ]
    return run_command(
        command, timeout=args.build_timeout, env=env, output_dir=evidence_dir,
        name="build-runtime", timed=False,
    )


def classify_build_failure(build: dict[str, Any]) -> tuple[str, str]:
    detail = str(build.get("error") or build.get("stderr") or "runtime build failed")
    lowered = detail.lower()
    capacity_markers = (
        "cannot allocate memory",
        "failed to allocate",
        "out of memory",
        "context length exceeds",
        "context-len exceeds",
        "maximum context",
    )
    if int(build.get("returncode", 1)) in {-9, 137} or any(marker in lowered for marker in capacity_markers):
        return "SKIP", f"host/model capacity limit: {detail[-2000:]}"
    return "FAIL", detail[-2000:]


def run_cke_perf(
    runtime_dir: Path,
    context_tokens: int,
    args: argparse.Namespace,
    env: dict[str, str],
    evidence_dir: Path,
) -> dict[str, Any]:
    token_csv = ",".join([str(args.token_id)] * context_tokens)
    samples = []
    for repetition in range(args.repetitions):
        trace_path = evidence_dir / f"cke-{repetition + 1}.trace.json"
        command = [
            str(args.ck_cli),
            "--lib", str(runtime_dir / "libmodel.so"),
            "--weights", str(runtime_dir / "weights.bump"),
            "--manifest", str(runtime_dir / "weights_manifest.map"),
            "--prompt-tokens", token_csv,
            "--max-tokens", str(args.decode_tokens + 1),
            "--context", str(context_tokens + args.decode_tokens + 8),
            "--temperature", "0",
            "--ignore-eos",
            "--quiet-output",
            "--no-chat-template",
            "--no-stream",
            "--timing",
            "--token-trace-json", str(trace_path),
        ]
        result = run_command(
            command, timeout=args.run_timeout, env=env, output_dir=evidence_dir,
            name=f"cke-{repetition + 1}", timed=True,
        )
        sample: dict[str, Any] = {
            "returncode": result["returncode"],
            "elapsed_seconds": result["elapsed_seconds"],
            "peak_rss_kib": result["peak_rss_kib"],
            "stdout_path": result["stdout_path"],
            "stderr_path": result["stderr_path"],
        }
        if result["returncode"] == 0:
            sample["timing"] = parse_timing(result["stdout"] + result["stderr"])
            sample["trace"] = json.loads(trace_path.read_text(encoding="utf-8"))
        else:
            sample["error"] = result["error"] or result["stderr"][-2000:]
        samples.append(sample)
    successful = [sample for sample in samples if "timing" in sample]
    hashes = [sample["trace"].get("first_logits_fnv1a64") for sample in successful]
    consumed = all(
        int(sample["trace"].get("prompt_tokens", -1)) == context_tokens
        and int(sample["timing"].get("prompt_tokens", -1)) == context_tokens
        for sample in successful
    ) and len(successful) == args.repetitions
    return {
        "status": "PASS" if consumed and len(set(hashes)) == 1 and hashes[0] else "FAIL",
        "samples": samples,
        "timing": median_timing(successful),
        "peak_rss_kib": max((sample.get("peak_rss_kib") or 0 for sample in samples), default=0),
        "first_logits_hashes": hashes,
        "first_logits_repeatable": bool(hashes and len(hashes) == args.repetitions and len(set(hashes)) == 1),
        "consumed_token_count_verified": consumed,
        "input_token_id": args.token_id,
        "input_sha256": token_sha256(args.token_id, context_tokens),
    }


def run_numerical_parity(
    gguf: Path | None,
    runtime_dir: Path,
    context_tokens: int,
    args: argparse.Namespace,
    env: dict[str, str],
    evidence_dir: Path,
) -> dict[str, Any]:
    if args.no_llama:
        return {"status": "SKIP", "reason": "disabled by --no-llama"}
    if gguf is None or not gguf.is_file():
        return {"status": "SKIP", "reason": "no local GGUF numerical oracle"}
    if context_tokens != args.parity_context:
        return {"status": "SKIP", "reason": f"parity is sampled at {args.parity_context} tokens"}
    report_path = evidence_dir / "numerical-parity.json"
    token_csv = ",".join([str(args.token_id)] * context_tokens)
    parity_env = dict(env)
    parity_env["CK_LLAMA_CPP_ROOT"] = str(args.llama_root)
    command = [
        sys.executable,
        str(SCRIPT_DIR / "compare_multitoken_logits_v8.py"),
        "--model-dir", str(runtime_dir),
        "--gguf", str(gguf),
        "--tokens", token_csv,
        "--max-new-tokens", str(args.parity_new_tokens),
        "--ctx-len", str(context_tokens + args.parity_new_tokens + 8),
        "--threads", str(args.threads),
        "--append-on-divergence", "stop",
        "--json-out", str(report_path),
        "--summary",
    ]
    result = run_command(
        command, timeout=args.run_timeout, env=parity_env, output_dir=evidence_dir,
        name="numerical-parity", timed=True,
    )
    if result["returncode"] != 0 or not report_path.is_file():
        return {"status": "FAIL", "reason": result["error"] or result["stderr"][-2000:]}
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    steps = payload.get("steps") if isinstance(payload.get("steps"), list) else []
    top1 = sum(bool(step.get("top1_match")) for step in steps)
    bit_exact = sum(bool(step.get("bit_exact", float(step.get("max_abs_diff", 1.0)) == 0.0)) for step in steps)
    passed = bool(steps and payload.get("first_divergence") is None and top1 == len(steps))
    return {
        "status": "PASS" if passed else "FAIL",
        "compared_rows": len(steps),
        "top1_exact_rows": top1,
        "bit_exact_rows": bit_exact,
        "first_divergence": payload.get("first_divergence"),
        "report_path": str(report_path),
        "peak_rss_kib": result["peak_rss_kib"],
    }


def run_llama_perf(
    gguf: Path | None,
    context_tokens: int,
    args: argparse.Namespace,
    env: dict[str, str],
    evidence_dir: Path,
) -> dict[str, Any]:
    binary = args.llama_root / "build" / "bin" / "llama-bench"
    if args.no_llama:
        return {"status": "SKIP", "reason": "disabled by --no-llama"}
    if gguf is None or not gguf.is_file():
        return {"status": "SKIP", "reason": "no local GGUF oracle artifact"}
    if not binary.is_file():
        return {"status": "SKIP", "reason": f"llama-bench is missing: {binary}"}
    command = [
        str(binary), "-m", str(gguf), "-p", str(context_tokens),
        "-n", str(args.decode_tokens), "-t", str(args.threads), "-ngl", "0",
        "-r", "1", "-o", "json",
    ]
    result = run_command(
        command, timeout=args.run_timeout, env=env, output_dir=evidence_dir,
        name="llama-bench", timed=True,
    )
    if result["returncode"] != 0:
        return {"status": "FAIL", "reason": result["error"] or result["stderr"][-2000:]}
    try:
        timing = parse_llama_bench(result["stdout"])
    except (ValueError, json.JSONDecodeError) as exc:
        return {"status": "FAIL", "reason": str(exc)}
    return {
        "status": "PASS",
        "timing": timing,
        "peak_rss_kib": result["peak_rss_kib"],
        "consumed_token_count_verified": int(timing["prompt_tokens"]) == context_tokens,
        "comparison_scope": "matched token count performance; not a token-ID numerical parity claim",
    }


def run_quality(
    row: dict[str, Any],
    runtime_dir: Path,
    runtime_context: int,
    prompts: list[dict[str, Any]],
    args: argparse.Namespace,
    env: dict[str, str],
    output_dir: Path,
) -> list[dict[str, Any]]:
    results = []
    for prompt in prompts:
        prompt_id = str(prompt["id"])
        result_path = output_dir / f"{prompt_id}.json"
        if args.resume and result_path.is_file():
            results.append(json.loads(result_path.read_text(encoding="utf-8")))
            continue
        trace_path = output_dir / f"{prompt_id}.trace.json"
        command = [
            str(args.ck_cli),
            "--lib", str(runtime_dir / "libmodel.so"),
            "--weights", str(runtime_dir / "weights.bump"),
            "--manifest", str(runtime_dir / "weights_manifest.map"),
            "--prompt", str(prompt["text"]),
            "--max-tokens", str(int(prompt["max_tokens"]) + 1),
            "--context", str(runtime_context),
            "--temperature", "0",
            "--no-stream", "--timing",
            "--token-trace-json", str(trace_path),
        ]
        command_result = run_command(
            command, timeout=args.quality_timeout, env=env, output_dir=output_dir,
            name=prompt_id, timed=True,
        )
        generated = extract_generated(command_result["stdout"])
        output_path = output_dir / (f"{prompt_id}.svg" if prompt_id == "svg" else f"{prompt_id}.txt")
        output_path.write_text(generated, encoding="utf-8")
        quality = svg_quality(generated) if prompt_id == "svg" else code_quality(generated)
        timing = None
        if command_result["returncode"] == 0:
            try:
                timing = parse_timing(command_result["stdout"] + command_result["stderr"])
            except ValueError:
                pass
        result = {
            "model_id": row["id"], "model": row["label"], "prompt_id": prompt_id,
            "status": "PASS" if command_result["returncode"] == 0 and quality["pass"] else "FAIL",
            "quality": quality, "timing": timing, "peak_rss_kib": command_result["peak_rss_kib"],
            "output_path": str(output_path), "trace_path": str(trace_path),
        }
        atomic_json(result_path, result)
        results.append(result)
    return results


def certify_quality_at_context(
    row: dict[str, Any],
    model: str,
    runtime_dir: Path,
    context_tokens: int,
    prompts: list[dict[str, Any]],
    args: argparse.Namespace,
    env: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    model_id = str(row["id"])
    evidence_dir = args.output_dir / "quality" / model_id
    max_tokens = max(int(prompt["max_tokens"]) for prompt in prompts)
    build = build_runtime(
        model, runtime_dir, context_tokens, max_tokens, args, env, evidence_dir
    )
    if build["returncode"] != 0:
        status, reason = classify_build_failure(build)
        return [], {
            "model_id": model_id,
            "status": status,
            "reason": f"quality runtime build failed at {context_tokens} tokens: {reason}",
        }
    quality_context = context_tokens + max_tokens + 8
    return run_quality(
        row, runtime_dir, quality_context, prompts, args, env, evidence_dir
    ), None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--quality-prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--contexts", default="2048,8192,32768,131072")
    parser.add_argument("--models", default="all", help="Comma-separated IDs or all")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--decode-tokens", type=int, default=8)
    parser.add_argument("--parity-context", type=int, default=2048)
    parser.add_argument("--parity-new-tokens", type=int, default=8)
    parser.add_argument("--token-id", type=int, default=100)
    parser.add_argument("--prefill-chunk-tokens", type=int, default=8192)
    parser.add_argument("--build-timeout", type=int, default=14400)
    parser.add_argument("--run-timeout", type=int, default=43200)
    parser.add_argument("--quality-timeout", type=int, default=43200)
    parser.add_argument("--quality-context", type=int, default=8192)
    parser.add_argument("--ck-cli", type=Path, default=ROOT / "build" / "ck-cli-v8")
    parser.add_argument("--llama-root", type=Path, default=ROOT / "llama.cpp")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--no-llama", action="store_true")
    parser.add_argument("--no-quality", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    args = parser.parse_args()

    if args.repetitions < 2:
        raise ValueError("at least two repetitions are required for first-logit repeatability")
    contexts = parse_contexts(args.contexts)
    catalog = load_schema(args.catalog, "cke.v8.long_context_model_catalog")
    prompt_payload = load_schema(args.quality_prompts, "cke.v8.long_context_quality_prompts")
    selected = {item.strip() for item in args.models.split(",") if item.strip()}
    rows = [row for row in catalog["models"] if selected == {"all"} or row["id"] in selected]
    if not rows:
        raise ValueError("no models selected")

    args.output_dir = args.output_dir.expanduser().resolve()
    args.ck_cli = args.ck_cli.expanduser().resolve()
    args.llama_root = args.llama_root.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CK_NUM_THREADS"] = str(args.threads)
    env["OMP_NUM_THREADS"] = "1"
    env["CK_THREADPOOL_PROFILE"] = "1"

    report: dict[str, Any] = {
        "schema": "cke.v8.long_context_certification", "schema_version": 1,
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": host_fingerprint(),
        "config": {"contexts": contexts, "threads": args.threads, "repetitions": args.repetitions,
                   "decode_tokens": args.decode_tokens, "token_id": args.token_id,
                   "parity_context": args.parity_context,
                   "parity_new_tokens": args.parity_new_tokens,
                   "quality_context": args.quality_context},
        "performance": [], "quality": [], "events": [],
    }
    summary_path = args.output_dir / "summary.json"
    if args.resume and summary_path.is_file():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if previous.get("schema") == report["schema"]:
            report = previous

    completed = {
        (row.get("model_id"), int(row.get("context_tokens", 0)))
        for row in report.get("performance", [])
        if row.get("status") == "PASS"
    }
    completed_quality = {
        str(row.get("model_id"))
        for row in report.get("quality", [])
        if row.get("status") in {"PASS", "FAIL"}
    }
    model_runtime: dict[str, Path] = {}
    model_sources: dict[str, tuple[str, str]] = {}
    for row in rows:
        source = resolve_model(row, args.allow_download)
        model_sources[row["id"]] = source
        model_runtime[row["id"]] = args.output_dir / "runtimes" / row["id"]
        if not source[0]:
            report["events"].append({"model_id": row["id"], "status": "SKIP", "reason": source[1]})
    publish(report, args.output_dir)

    for context_tokens in contexts:
        for row in rows:
            model_id = str(row["id"])
            if (model_id, context_tokens) in completed:
                continue
            model, source_reason = model_sources[model_id]
            if not model:
                continue
            runtime_dir = model_runtime[model_id]
            evidence_dir = args.output_dir / "models" / model_id / f"ctx-{context_tokens}"
            build = build_runtime(model, runtime_dir, context_tokens, args.decode_tokens, args, env, evidence_dir)
            if build["returncode"] != 0:
                failure_status, failure_reason = classify_build_failure(build)
                result = {"model_id": model_id, "model": row["label"], "context_tokens": context_tokens,
                          "status": failure_status, "reason": failure_reason}
                report["performance"].append(result)
                report["events"].append(result)
                publish(report, args.output_dir)
                continue
            if not args.ck_cli.is_file():
                raise FileNotFoundError(f"native CLI is missing: {args.ck_cli}")
            gguf = resolve_gguf(row, model, runtime_dir)
            cke = run_cke_perf(runtime_dir, context_tokens, args, env, evidence_dir)
            llama = run_llama_perf(gguf, context_tokens, args, env, evidence_dir)
            parity = run_numerical_parity(
                gguf, runtime_dir, context_tokens, args, env, evidence_dir
            )
            status = "PASS" if (
                cke["status"] == "PASS"
                and llama["status"] in {"PASS", "SKIP"}
                and parity["status"] in {"PASS", "SKIP"}
            ) else "FAIL"
            result = {
                "model_id": model_id, "model": row["label"], "model_source": source_reason,
                "context_tokens": context_tokens, "runtime_context": context_tokens + args.decode_tokens + 8,
                "status": status, "cke": cke, "llama_cpp": llama,
                "numerical_parity": parity,
                "providers": provider_summary(runtime_dir),
            }
            if cke.get("timing") and llama.get("timing"):
                result["relative_prefill"] = (
                    float(cke["timing"]["prompt_tok_s"]) / float(llama["timing"]["prompt_tok_s"])
                )
            report["performance"].append(result)
            if status == "FAIL":
                report["events"].append({"model_id": model_id, "status": "FAIL",
                                         "reason": f"context {context_tokens} certification failed"})
            publish(report, args.output_dir)

            if (
                not args.no_quality
                and status == "PASS"
                and context_tokens == args.quality_context
                and model_id not in completed_quality
            ):
                quality_rows, event = certify_quality_at_context(
                    row, model, runtime_dir, context_tokens,
                    prompt_payload["prompts"], args, env,
                )
                report["quality"].extend(quality_rows)
                completed_quality.add(model_id)
                if event:
                    report["events"].append(event)
                publish(report, args.output_dir)

    if not args.no_quality:
        passing_contexts: dict[str, int] = {}
        for result in report["performance"]:
            if result.get("status") == "PASS":
                model_id = str(result["model_id"])
                passing_contexts[model_id] = max(passing_contexts.get(model_id, 0), int(result["context_tokens"]))
        for row in rows:
            model_id = str(row["id"])
            if model_id not in passing_contexts or model_id in completed_quality:
                continue
            context_tokens = passing_contexts[model_id]
            runtime_dir = model_runtime[model_id]
            quality_rows, event = certify_quality_at_context(
                row, model_sources[model_id][0], runtime_dir, context_tokens,
                prompt_payload["prompts"], args, env,
            )
            report["quality"].extend(quality_rows)
            completed_quality.add(model_id)
            if event:
                report["events"].append(event)
            publish(report, args.output_dir)

    failed = any(row.get("status") == "FAIL" for row in report.get("performance", []))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
