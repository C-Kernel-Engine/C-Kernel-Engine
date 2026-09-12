#!/usr/bin/env python3
"""Run a private multimodal image corpus through exact CK/llama.cpp parity.

The historical Qwen3-VL GGUF/mmproj path remains the default. A prebuilt
encoder runtime can instead supply BF16/safetensors vision prefixes while the
same redacted, resume-safe decoder oracle contract is retained.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
PROFILE_LOADER_PATH = Path(__file__).resolve().with_name("vision_certification_profiles_v8.py")
_PROFILE_SPEC = importlib.util.spec_from_file_location("vision_certification_profiles_v8", PROFILE_LOADER_PATH)
if _PROFILE_SPEC is None or _PROFILE_SPEC.loader is None:
    raise RuntimeError(f"cannot load vision certification profiles: {PROFILE_LOADER_PATH}")
_PROFILE_MODULE = importlib.util.module_from_spec(_PROFILE_SPEC)
_PROFILE_SPEC.loader.exec_module(_PROFILE_MODULE)
BUILTIN_PROFILES = _PROFILE_MODULE.BUILTIN_PROFILES
apply_profile = _PROFILE_MODULE.apply_profile


BRIDGE = ROOT / "version" / "v8" / "scripts" / "run_multimodal_bridge_v8.py"
PARITY = ROOT / "version" / "v8" / "scripts" / "compare_multimodal_multitoken_logits_v8.py"
DEFAULT_NATIVE_CLI = ROOT / "build" / "ck-cli-v8"
PINNED_LLAMA_COMMIT = "f3e182816421c648188b5eab269853bf1531d950"
DEFAULT_PROMPT = "Extract visible form fields as compact JSON."
MODEL_PROFILES = BUILTIN_PROFILES


def _apply_model_profile(args: argparse.Namespace) -> None:
    """Fill architecture controls from one versioned certification profile."""
    apply_profile(args)


def _json_write(path: Path, value: Any, *, private: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if private:
        temporary.chmod(0o600)
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_identity(path: Path, *, hash_content: bool) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    stat = resolved.stat()
    result: dict[str, Any] = {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if hash_content:
        result["sha256"] = _sha256_file(resolved)
    return result


def _load_corpus(manifest_path: Path) -> list[dict[str, Any]]:
    manifest_path = manifest_path.resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = payload.get("samples") if isinstance(payload, dict) else None
    if not isinstance(samples, list) or not samples:
        raise ValueError("corpus manifest must contain a non-empty samples list")
    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        if not isinstance(sample, dict):
            raise ValueError(f"sample {index} is not an object")
        inputs = sample.get("inputs")
        if not isinstance(inputs, list) or len(inputs) != 1 or not isinstance(inputs[0], dict):
            raise ValueError(f"sample {index} must contain exactly one image input")
        raw_path = inputs[0].get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(f"sample {index} has no image path")
        image_path = Path(raw_path).expanduser()
        if not image_path.is_absolute():
            image_path = manifest_path.parent / image_path
        image_path = image_path.resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"sample {index} image is missing: {image_path}")
        rows.append(
            {
                "index": index,
                "image": image_path,
                "image_sha256": _sha256_file(image_path),
            }
        )
    return rows


def _require_corpus_size(rows: list[dict[str, Any]], minimum: int | None) -> None:
    if minimum is None:
        return
    if minimum < 1:
        raise ValueError("--require-images must be positive")
    if len(rows) < minimum:
        raise ValueError(
            f"private corpus contains {len(rows)} images; at least {minimum} are required"
        )


def _git_commit(repo: Path) -> str:
    probe = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    if probe.returncode != 0:
        raise RuntimeError(probe.stderr.strip() or f"cannot resolve git commit for {repo}")
    return probe.stdout.strip()


def _run_logged(
    command: list[str],
    *,
    env: dict[str, str],
    log_path: Path,
    dry_run: bool,
    show_dry_run_command: bool = True,
    accepted_returncodes: tuple[int, ...] = (0,),
) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = shlex.join(command)
    if dry_run:
        print(rendered if show_dry_run_command else "[dry-run] private command redacted")
        return 0.0
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as stream:
        log_path.chmod(0o600)
        stream.write(f"$ {rendered}\n\n")
        stream.flush()
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.perf_counter() - started
    if completed.returncode not in accepted_returncodes:
        raise RuntimeError(
            f"command failed rc={completed.returncode}; inspect private log {log_path}"
        )
    return elapsed


def _bridge_command(
    args: argparse.Namespace,
    *,
    image: Path,
    runtime_dir: Path,
    prefix_path: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(BRIDGE),
        "--decoder-gguf",
        str(args.decoder_gguf),
        "--workdir",
        str(runtime_dir),
        "--prompt",
        args.prompt,
        "--chat-template",
        str(getattr(args, "chat_template", "qwen3vl")),
        "--thinking-mode",
        "suppressed",
        "--image-path",
        str(image),
        "--image-max-tokens",
        str(args.image_max_tokens),
        "--decoder-context-len",
        str(args.context_len),
        "--dump-prefix-f32",
        str(prefix_path),
        "--report-top-k",
        str(args.top_k),
        "--max-tokens",
        "0",
        "--temperature",
        "0",
        "--no-stream-output",
        "--gemm-schedule",
        getattr(args, "gemm_schedule", "auto"),
    ]
    encoder_runtime = getattr(args, "encoder_runtime", None)
    if encoder_runtime is not None:
        command.extend(["--encoder-runtime", str(encoder_runtime)])
    else:
        command.extend(["--encoder-gguf", str(args.mmproj_gguf)])
    composition_circuit = str(getattr(args, "composition_circuit", "") or "").strip()
    if composition_circuit:
        command.extend(["--composition-circuit", composition_circuit])
    return command


def _parity_command(
    args: argparse.Namespace,
    *,
    bridge_report: Path,
    prefix_path: Path,
    workdir: Path,
    report_path: Path,
    runtime_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(PARITY),
        "--bridge-report",
        str(bridge_report),
        "--prefix-f32",
        str(prefix_path),
        "--workdir",
        str(workdir),
        "--reuse-bridge-decoder-runtime-exact",
        "--ck-engine-so",
        str(runtime_dir / "decoder" / "libckernel_engine.so"),
        "--ctx-len",
        str(args.context_len),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--top-k",
        str(args.top_k),
        "--threads",
        str(args.threads),
        "--ck-threads",
        str(args.ck_threads),
        "--gemm-schedule",
        getattr(args, "gemm_schedule", "auto"),
        "--llama-required-isa",
        args.llama_required_isa,
        "--llama-decode-mode",
        "batched",
        "--llama-flash-attention",
        str(getattr(args, "llama_flash_attention", "disabled")),
        "--append-on-divergence",
        str(getattr(args, "append_on_divergence", "stop")),
        "--json-out",
        str(report_path),
    ]
    return command


def _native_cli_command(
    args: argparse.Namespace,
    *,
    bridge_report: Path,
    runtime_dir: Path,
    trace_path: Path,
) -> list[str]:
    decoder_dir = runtime_dir / "decoder"
    return [
        str(args.native_cli),
        "--lib",
        str(decoder_dir / "libdecoder_v8.so"),
        "--weights",
        str(decoder_dir / "weights.bump"),
        "--manifest",
        str(decoder_dir / "weights_manifest.map"),
        "--bridge-report",
        str(bridge_report),
        "--max-tokens",
        str(args.max_new_tokens),
        "--context",
        str(args.context_len),
        "--temperature",
        "0",
        "--top-p",
        "1",
        "--quiet-output",
        "--no-timing",
        "--require-generated-abi",
        "--token-trace-json",
        str(trace_path),
        "--gemm-schedule",
        getattr(args, "gemm_schedule", "auto"),
    ]


def _pre_eos_tokens(report: dict[str, Any], key: str) -> list[int]:
    stop_ids = {int(value) for value in report.get("stop_token_ids") or []}
    tokens: list[int] = []
    for row in report.get("steps") or []:
        if not isinstance(row, dict) or row.get(key) is None:
            continue
        token = int(row[key])
        if token in stop_ids:
            break
        tokens.append(token)
    return tokens


def _common_parity_prefix_tokens(report: dict[str, Any]) -> list[int]:
    """Return equal CKE/llama tokens before EOS or the first divergence."""
    stop_ids = {int(value) for value in report.get("stop_token_ids") or []}
    tokens: list[int] = []
    for row in report.get("steps") or []:
        if not isinstance(row, dict):
            break
        ck_next = row.get("ck_next")
        llama_next = row.get("llama_next")
        if ck_next is None or llama_next is None:
            break
        ck_token = int(ck_next)
        llama_token = int(llama_next)
        if ck_token != llama_token or ck_token in stop_ids:
            break
        tokens.append(ck_token)
    return tokens


def _first_token_divergence(
    left: list[int],
    right: list[int],
    *,
    require_equal_length: bool = True,
) -> dict[str, Any] | None:
    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if left_token != right_token:
            return {"step": index, "native_token": left_token, "reference_token": right_token}
    if require_equal_length and len(left) != len(right):
        index = min(len(left), len(right))
        return {
            "step": index,
            "native_token": left[index] if index < len(left) else None,
            "reference_token": right[index] if index < len(right) else None,
        }
    return None


def _compare_native_trace(
    report: dict[str, Any],
    trace: dict[str, Any],
) -> dict[str, Any]:
    if trace.get("schema") != "cke.native_token_trace" or trace.get("schema_version") != 1:
        raise ValueError("native CLI emitted an unsupported token trace contract")
    native = [int(value) for value in trace.get("token_ids") or []]
    ck = _pre_eos_tokens(report, "ck_next")
    llama = _pre_eos_tokens(report, "llama_next")
    complete = bool(report.get("pass"))
    ck_divergence = _first_token_divergence(
        native,
        ck,
        require_equal_length=complete or len(native) < len(ck),
    )
    llama_divergence = _first_token_divergence(
        native,
        llama,
        require_equal_length=complete or len(native) < len(llama),
    )
    passed = complete and ck_divergence is None and llama_divergence is None
    return {
        "status": "pass" if passed else "fail",
        "pass": passed,
        "native_tokens": len(native),
        "python_ck_tokens": len(ck),
        "llama_tokens": len(llama),
        "native_vs_python_first_divergence": ck_divergence,
        "native_vs_llama_first_divergence": llama_divergence,
        "native_matches_python_captured_prefix": ck_divergence is None,
        "three_way_comparison_complete": complete,
    }


def _runtime_hash(report: dict[str, Any], key: str) -> str | None:
    runtime = report.get("ck_runtime")
    if not isinstance(runtime, dict):
        return None
    item = runtime.get(key)
    return str(item.get("sha256")) if isinstance(item, dict) and item.get("sha256") else None


def _public_provenance(config: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "version",
        "cke_commit",
        "model_profile",
        "model_label",
        "composition_circuit",
        "manifest_sha256",
        "llama_commit",
        "expected_llama_commit",
        "compiler",
        "prompt_sha256",
        "context_len",
        "image_max_tokens",
        "max_new_tokens",
        "require_images",
        "append_on_divergence",
        "chat_template",
        "threads",
        "ck_threads",
        "top_k",
        "llama_required_isa",
        "native_cli_sha256",
    )
    provenance = {key: config[key] for key in keys}
    profile = config.get("profile_contract")
    if isinstance(profile, dict):
        provenance["profile_sha256"] = profile.get("sha256")
        provenance["oracle"] = profile.get("oracle")
    return provenance


def _redacted_row(
    *,
    index: int,
    image_sha256: str,
    prefix_sha256: str,
    report: dict[str, Any],
    elapsed: dict[str, float],
    requested_tokens: int,
    native_comparison: dict[str, Any] | None = None,
) -> dict[str, Any]:
    divergence = report.get("first_divergence")
    first_divergence = None
    if isinstance(divergence, dict):
        first_divergence = {
            "step": divergence.get("step"),
            "ck_next": divergence.get("ck_next"),
            "llama_next": divergence.get("llama_next"),
            "cosine": divergence.get("cosine"),
            "rmse": divergence.get("rmse"),
            "topk_overlap_count": divergence.get("topk_overlap_count"),
        }
    llama = report.get("llama_oracle")
    compiler = report.get("compiler_provenance")
    compiler_summary = None
    if isinstance(compiler, dict):
        compiler_summary = {
            key: compiler.get(key)
            for key in ("status", "decoder_family", "engine_family")
        }
    prefix = report.get("prefix")
    steps = len(report.get("steps") or [])
    # This field describes the common CKE/llama trajectory, not the number of
    # tokens emitted by an independently continued native replay.  The latter
    # can be longer after an early top-1 mismatch and previously made failed
    # rows print misleading values such as matched=128/128.
    matched_tokens = len(_common_parity_prefix_tokens(report))
    prefix_tokens = int(prefix.get("tokens", 0)) if isinstance(prefix, dict) else 0
    prompt_tokens = len(report.get("prompt_tokens_before_image") or []) + len(
        report.get("prompt_tokens_after_image") or []
    )
    prefill_tokens = prefix_tokens + prompt_tokens
    bridge_sec = float(elapsed.get("bridge", 0.0))
    parity_sec = float(elapsed.get("parity", 0.0))
    native_sec = float(elapsed.get("native_cli", 0.0))
    total_sec = bridge_sec + parity_sec + native_sec
    parity_pass = bool(report.get("pass"))
    native_pass = native_comparison is None or bool(native_comparison.get("pass"))
    return {
        "image_index": int(index),
        "image_sha256": image_sha256,
        "prefix_sha256": prefix_sha256,
        "grid": prefix.get("grid") if isinstance(prefix, dict) else None,
        "status": "pass" if parity_pass and native_pass else "fail",
        "steps": steps,
        "matched_tokens": matched_tokens,
        "requested_tokens": requested_tokens,
        "prefix_tokens": prefix_tokens,
        "prompt_tokens": prompt_tokens,
        "prefill_tokens": prefill_tokens,
        "context_capacity": int(report.get("ctx_len", 0)),
        "context_tokens_after_comparison": prefill_tokens + matched_tokens,
        "first_divergence": first_divergence,
        "stop_reason": report.get("stop_reason"),
        "decoder_sha256": _runtime_hash(report, "shared_library"),
        "engine_sha256": _runtime_hash(report, "engine_library"),
        "compiler_provenance": compiler_summary,
        "llama_commit": llama.get("commit") if isinstance(llama, dict) else None,
        "native_cli": native_comparison,
        "elapsed_sec": {
            **elapsed,
            "total": total_sec,
            "comparison_per_token": parity_sec / steps if steps else None,
        },
    }


def _row_total_sec(row: dict[str, Any]) -> float:
    elapsed = row.get("elapsed_sec")
    return float(elapsed.get("total", 0.0)) if isinstance(elapsed, dict) else 0.0


def _timing_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    totals = [_row_total_sec(row) for row in rows if _row_total_sec(row) > 0.0]
    if not totals:
        return {
            "total_sec": 0.0,
            "mean_sec_per_image": 0.0,
            "min_sec_per_image": 0.0,
            "max_sec_per_image": 0.0,
        }
    return {
        "total_sec": sum(totals),
        "mean_sec_per_image": sum(totals) / len(totals),
        "min_sec_per_image": min(totals),
        "max_sec_per_image": max(totals),
    }


def _progress_line(
    row: dict[str, Any],
    *,
    completed: int,
    requested: int,
    resumed: bool = False,
) -> str:
    elapsed = row.get("elapsed_sec")
    elapsed = elapsed if isinstance(elapsed, dict) else {}
    matched = int(row.get("matched_tokens", row.get("steps", 0)))
    target = int(row.get("requested_tokens", matched))
    native = row.get("native_cli")
    native_text = (
        f"{int(native.get('native_tokens', 0))}/{int(native.get('python_ck_tokens', 0))}"
        if isinstance(native, dict)
        else "skipped"
    )
    suffix = " resumed" if resumed else ""
    return (
        f"[{completed}/{requested}] image {int(row['image_index']):02d}: "
        f"{str(row.get('status', 'unknown')).upper()} "
        f"matched={matched}/{target} "
        f"native={native_text} "
        f"prefix={int(row.get('prefix_tokens', 0))} "
        f"prompt={int(row.get('prompt_tokens', 0))} "
        f"prefill={int(row.get('prefill_tokens', 0))} "
        f"context={int(row.get('context_tokens_after_comparison', 0))}/"
        f"{int(row.get('context_capacity', 0))} "
        f"bridge={float(elapsed.get('bridge', 0.0)):.2f}s "
        f"parity={float(elapsed.get('parity', 0.0)):.2f}s "
        f"native={float(elapsed.get('native_cli', 0.0)):.2f}s "
        f"total={_row_total_sec(row):.2f}s "
        f"compare={float(elapsed.get('comparison_per_token') or 0.0):.3f}s/token-pair"
        f"{suffix}"
    )


def _private_console_enabled(args: argparse.Namespace) -> bool:
    configured = getattr(args, "show_private_details", None)
    if configured is not None:
        return bool(configured)
    return bool(sys.stdout.isatty() and not os.environ.get("CI"))


def _load_json_if_present(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _print_private_case_details(
    *,
    sample: dict[str, Any],
    row: dict[str, Any],
    case_dir: Path,
    prompt: str,
    model_label: str = "Qwen3-VL",
) -> None:
    bridge = _load_json_if_present(case_dir / "bridge_report.json")
    parity = _load_json_if_present(case_dir / "parity.json")
    encoder = bridge.get("encoder_report")
    encoder = encoder if isinstance(encoder, dict) else {}
    bridge_timings = bridge.get("timings")
    bridge_timings = bridge_timings if isinstance(bridge_timings, dict) else {}
    source_size = encoder.get("source_image_size")
    source_text = "unknown"
    if isinstance(source_size, list) and len(source_size) == 2:
        source_text = f"{int(source_size[0])}x{int(source_size[1])}"
    image_width = int(encoder.get("image_width", 0) or 0)
    image_height = int(encoder.get("image_height", 0) or 0)
    processed_text = (
        f"{image_width}x{image_height}" if image_width > 0 and image_height > 0 else "unknown"
    )
    grid = row.get("grid")
    grid_text = "unknown"
    if isinstance(grid, list) and len(grid) == 2:
        grid_text = f"{int(grid[0])}x{int(grid[1])}"

    matched = int(row.get("matched_tokens", row.get("steps", 0)))
    requested = int(row.get("requested_tokens", matched))
    exact = str(row.get("status", "")).lower() == "pass"
    label = "EXACT MATCH" if exact else str(row.get("status", "unknown")).upper()
    elapsed = row.get("elapsed_sec")
    elapsed = elapsed if isinstance(elapsed, dict) else {}
    encoder_sec = float(bridge_timings.get("encoder_execute_ms", 0.0) or 0.0) / 1000.0
    prefill_sec = float(bridge_timings.get("decoder_forward_mixed_ms", 0.0) or 0.0) / 1000.0

    print()
    print("=" * 88)
    print(f"{model_label.upper()} PRIVATE PARITY | IMAGE {int(sample['index']):02d} | {label}")
    print("-" * 88)
    print(f"Image       : {sample['image']}")
    print(f"Image SHA256: {sample['image_sha256']}")
    print(
        f"Geometry    : source {source_text} -> processed {processed_text} | "
        f"grid {grid_text} | vision tokens {int(row.get('prefix_tokens', 0))}"
    )
    print(f"Prompt      : {prompt}")
    print(
        f"Context     : prefill {int(row.get('prefill_tokens', 0))} + "
        f"compared {matched} = {int(row.get('context_tokens_after_comparison', 0))}/"
        f"{int(row.get('context_capacity', 0))}"
    )
    comparison_runtimes = (
        "native/Python/llama" if isinstance(row.get("native_cli"), dict) else "Python/llama"
    )
    print(
        f"Comparison  : {matched}/{requested} exact {comparison_runtimes} pre-EOS tokens | "
        f"stop={row.get('stop_reason') or 'token limit'}"
    )
    print(
        f"Timing      : encoder={encoder_sec:.2f}s mixed-prefill={prefill_sec:.2f}s "
        f"bridge={float(elapsed.get('bridge', 0.0)):.2f}s "
        f"parity-pair={float(elapsed.get('parity', 0.0)):.2f}s "
        f"native-cli={float(elapsed.get('native_cli', 0.0)):.2f}s "
        f"total={_row_total_sec(row):.2f}s"
    )

    shared_text = str(parity.get("generated_shared_text", "") or "")
    print("-" * 88)
    if exact and isinstance(row.get("native_cli"), dict):
        print("Output (native CLI == Python CKE == llama.cpp for every compared token):")
    elif exact:
        print("Output (Python CKE == llama.cpp for every compared token):")
    else:
        print("Shared output before the first divergence:")
    print(shared_text if shared_text else "<no decoded text>")

    divergence = parity.get("first_divergence")
    if isinstance(divergence, dict):
        print("-" * 88)
        print(
            f"First divergence at step {divergence.get('step')}: "
            f"CK={divergence.get('ck_next')} {divergence.get('ck_next_text')!r} | "
            f"llama.cpp={divergence.get('llama_next')} "
            f"{divergence.get('llama_next_text')!r}"
        )
        print(
            f"cosine={divergence.get('cosine')} rmse={divergence.get('rmse')} "
            f"top-k overlap={divergence.get('topk_overlap_count')}"
        )
    print("=" * 88)


def _summary(
    *,
    selected: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    by_index = {int(row["image_index"]): row for row in rows}
    ordered = [by_index[int(sample["index"])] for sample in selected if int(sample["index"]) in by_index]
    passed = sum(row.get("status") == "pass" for row in ordered)
    failed = sum(row.get("status") != "pass" for row in ordered)
    completed = len(ordered)
    if failed:
        status = "fail"
    elif completed == len(selected):
        status = "pass"
    else:
        status = "incomplete"
    localization_only = bool(config.get("skip_native_cli"))
    return {
        "status": status,
        "certification_scope": "localization" if localization_only else "full",
        "comparison": (
            "Python CKE and llama.cpp pre-EOS greedy token parity; native CLI not run"
            if localization_only
            else "exact native CLI, Python CKE, and llama.cpp pre-EOS greedy token parity"
        ),
        "requested": len(selected),
        "completed": completed,
        "passed": passed,
        "failed": failed,
        "max_new_tokens": config["max_new_tokens"],
        "config_sha256": _sha256_json(config),
        "provenance": _public_provenance(config),
        "timing": _timing_summary(ordered),
        "rows": ordered,
    }


def _case_config(
    *,
    global_config_sha256: str,
    sample: dict[str, Any],
) -> dict[str, Any]:
    return {
        "global_config_sha256": global_config_sha256,
        "image_index": int(sample["index"]),
        "image_sha256": str(sample["image_sha256"]),
    }


def _resumed_row(
    case_result: Path,
    expected_config: dict[str, Any],
    *,
    require_native: bool = True,
) -> dict[str, Any] | None:
    if not case_result.is_file():
        return None
    payload = json.loads(case_result.read_text(encoding="utf-8"))
    if payload.get("case_config") != expected_config:
        return None
    row = payload.get("redacted_row")
    if not isinstance(row, dict) or row.get("status") != "pass":
        return None
    if require_native:
        native = row.get("native_cli")
        if not isinstance(native, dict) or not native.get("pass"):
            return None
        native_comparison = _load_json_if_present(case_result.parent / "native_comparison.json")
        if not native_comparison or not native_comparison.get("pass"):
            return None
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-profile",
        default="qwen3vl",
        help="built-in multimodal profile id (default: qwen3vl)",
    )
    parser.add_argument(
        "--profile-file",
        type=Path,
        help="versioned multimodal certification profile; unknown fields fail closed",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--decoder-gguf", type=Path, required=True)
    encoder_source = parser.add_mutually_exclusive_group(required=True)
    encoder_source.add_argument("--mmproj-gguf", type=Path)
    encoder_source.add_argument(
        "--encoder-runtime",
        type=Path,
        help="Provenance-complete prebuilt encoder runtime; uses its fixed image geometry",
    )
    parser.add_argument("--llama-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--runtime-workdir",
        type=Path,
        help="Reuse a shared bridge workdir containing decoder/ instead of storing it under --output-dir",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--model-label")
    parser.add_argument("--expected-llama-commit", default=PINNED_LLAMA_COMMIT)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--require-images",
        type=int,
        help="fail unless the private manifest contains at least this many images",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--chat-template",
        default=None,
        help=(
            "decoder chat contract passed to the multimodal bridge; use auto for "
            "cross-model corpus runs (default preserves the certified Qwen3-VL lane)"
        ),
    )
    parser.add_argument(
        "--composition-circuit",
        help=(
            "explicit multimodal composition circuit passed to the bridge; "
            "Qwen3.6-VL certification should use qwen36vl so architecture and "
            "stitch policy cannot fall back to runtime inference"
        ),
    )
    parser.add_argument(
        "--append-on-divergence",
        choices=("stop", "llama", "ck"),
        default="stop",
        help=(
            "trajectory after the first top-1 mismatch: stop for exact certification, "
            "or continue with llama/CK tokens for long teacher-forced diagnostics"
        ),
    )
    parser.add_argument("--context-len", type=int, default=4096)
    parser.add_argument("--image-max-tokens", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--ck-threads", type=int, default=20)
    parser.add_argument(
        "--gemm-schedule",
        choices=("auto", "static", "dynamic"),
        default="auto",
        help="CK independent GEMM tile scheduling policy (default: auto/dynamic)",
    )
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--llama-required-isa", choices=("auto", "avx2", "avx512"), default="avx2")
    parser.add_argument("--compiler", default="gcc")
    parser.add_argument("--native-cli", type=Path, default=DEFAULT_NATIVE_CLI)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-prefixes", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument(
        "--skip-native-cli",
        action="store_true",
        help=(
            "compare Python CKE directly with llama.cpp without the redundant native-CLI "
            "replay; intended for broad failure-localization scans"
        ),
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="ignore matching completed case results and execute the selected cases again",
    )
    private_console = parser.add_mutually_exclusive_group()
    private_console.add_argument(
        "--show-private-details",
        dest="show_private_details",
        action="store_true",
        default=None,
        help="print private image paths, prompt, decoded text, and detailed timings",
    )
    private_console.add_argument(
        "--redacted-console",
        dest="show_private_details",
        action="store_false",
        help="print only redacted progress, even in an interactive local terminal",
    )
    args = parser.parse_args()
    _apply_model_profile(args)

    os.umask(0o077)
    args.manifest = args.manifest.expanduser().resolve()
    args.decoder_gguf = args.decoder_gguf.expanduser().resolve()
    args.mmproj_gguf = args.mmproj_gguf.expanduser().resolve() if args.mmproj_gguf is not None else None
    args.encoder_runtime = args.encoder_runtime.expanduser().resolve() if args.encoder_runtime is not None else None
    args.llama_root = args.llama_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.runtime_workdir = args.runtime_workdir.expanduser().resolve() if args.runtime_workdir is not None else None
    args.native_cli = args.native_cli.expanduser().resolve()
    for required in (args.decoder_gguf,):
        if not required.is_file():
            raise FileNotFoundError(required)
    if args.mmproj_gguf is not None and not args.mmproj_gguf.is_file():
        raise FileNotFoundError(args.mmproj_gguf)
    if args.encoder_runtime is not None and not args.encoder_runtime.is_dir():
        raise FileNotFoundError(args.encoder_runtime)
    if args.runtime_workdir is not None and not (args.runtime_workdir / "decoder").is_dir():
        raise FileNotFoundError(f"shared bridge workdir has no decoder runtime: {args.runtime_workdir}")
    if not args.skip_native_cli and not args.dry_run and not args.native_cli.is_file():
        raise FileNotFoundError(f"native CLI is missing: {args.native_cli}")
    if not (args.llama_root / "build" / "bin" / "libllama.so").is_file():
        raise FileNotFoundError(f"llama.cpp build is missing libllama.so: {args.llama_root}")
    llama_commit = _git_commit(args.llama_root)
    if args.expected_llama_commit != "any" and llama_commit != args.expected_llama_commit:
        raise RuntimeError(
            "llama.cpp oracle commit mismatch: "
            f"expected={args.expected_llama_commit} actual={llama_commit}"
        )
    corpus = _load_corpus(args.manifest)
    _require_corpus_size(corpus, args.require_images)
    if args.start_index < 1:
        raise ValueError("--start-index must be at least 1")
    selected = [row for row in corpus if int(row["index"]) >= args.start_index]
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        selected = selected[: args.limit]
    if not selected:
        raise ValueError("the requested corpus range is empty")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.chmod(0o700)
    runtime_dir = args.runtime_workdir or (args.output_dir / "runtime")
    config = {
        "version": 2,
        "model_profile": args.model_profile,
        "profile_contract": args.profile_contract,
        "cke_commit": _git_commit(ROOT),
        "manifest_sha256": _sha256_file(args.manifest),
        "decoder": _file_identity(args.decoder_gguf, hash_content=False),
        "encoder_source": (
            {"kind": "prebuilt_runtime", "path": str(args.encoder_runtime)}
            if args.encoder_runtime is not None
            else {"kind": "mmproj_gguf", **_file_identity(args.mmproj_gguf, hash_content=False)}
        ),
        "llama_root": str(args.llama_root),
        "llama_commit": llama_commit,
        "expected_llama_commit": args.expected_llama_commit,
        "compiler": args.compiler,
        "prompt_sha256": hashlib.sha256(args.prompt.encode("utf-8")).hexdigest(),
        "model_label": str(args.model_label),
        "runtime_workdir": str(runtime_dir),
        "context_len": args.context_len,
        "image_max_tokens": args.image_max_tokens,
        "max_new_tokens": args.max_new_tokens,
        "require_images": args.require_images,
        "append_on_divergence": args.append_on_divergence,
        "chat_template": args.chat_template,
        "composition_circuit": args.composition_circuit,
        "threads": args.threads,
        "ck_threads": args.ck_threads,
        "gemm_schedule": args.gemm_schedule,
        "top_k": args.top_k,
        "llama_required_isa": args.llama_required_isa,
        "native_cli": (
            None
            if args.skip_native_cli
            else _file_identity(args.native_cli, hash_content=True)
        ),
        "native_cli_sha256": None if args.skip_native_cli else _sha256_file(args.native_cli),
        "skip_native_cli": bool(args.skip_native_cli),
    }
    config_sha256 = _sha256_json(config)
    _json_write(args.output_dir / "run_config.json", config)
    env = os.environ.copy()
    env.update(
        {
            "CK_LLAMA_CPP_ROOT": str(args.llama_root),
            "CK_V8_COMPILER": args.compiler,
            "CK_V7_COMPILER": args.compiler,
            "CK_NUM_THREADS": str(args.ck_threads),
            "OMP_NUM_THREADS": str(args.ck_threads),
        }
    )
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        filter(
            None,
            (
                str((ROOT / "build").resolve()),
                str((runtime_dir / "decoder").resolve()),
                env.get("LD_LIBRARY_PATH", ""),
            ),
        )
    )

    rows: list[dict[str, Any]] = []
    for sample in selected:
        index = int(sample["index"])
        case_dir = args.output_dir / f"image{index:02d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        case_dir.chmod(0o700)
        result_path = case_dir / "case_result.json"
        case_config = _case_config(
            global_config_sha256=config_sha256,
            sample=sample,
        )
        resumed = (
            None
            if args.force_rerun
            else _resumed_row(
                result_path,
                case_config,
                require_native=not bool(args.skip_native_cli),
            )
        )
        if resumed is not None:
            resumed_report = _load_json_if_present(case_dir / "parity.json")
            resumed_native = _load_json_if_present(case_dir / "native_comparison.json")
            if resumed_report:
                resumed = _redacted_row(
                    index=index,
                    image_sha256=sample["image_sha256"],
                    prefix_sha256=str(resumed.get("prefix_sha256", "")),
                    report=resumed_report,
                    elapsed=dict(resumed.get("elapsed_sec") or {}),
                    requested_tokens=args.max_new_tokens,
                    native_comparison=resumed_native or None,
                )
            rows.append(resumed)
            print(
                _progress_line(
                    resumed,
                    completed=len(rows),
                    requested=len(selected),
                    resumed=True,
                )
            )
            if _private_console_enabled(args):
                _print_private_case_details(
                    sample=sample,
                    row=resumed,
                    case_dir=case_dir,
                    prompt=args.prompt,
                    model_label=args.model_label,
                )
            _json_write(args.output_dir / "summary.json", _summary(selected=selected, rows=rows, config=config))
            continue

        prefix_path = case_dir / "prefix.f32"
        bridge_report = case_dir / "bridge_report.json"
        parity_report = case_dir / "parity.json"
        native_trace = case_dir / "native_token_trace.json"
        native_comparison_path = case_dir / "native_comparison.json"
        elapsed: dict[str, float] = {}
        stage = "bridge"
        try:
            elapsed["bridge"] = _run_logged(
                _bridge_command(
                    args,
                    image=sample["image"],
                    runtime_dir=runtime_dir,
                    prefix_path=prefix_path,
                ),
                env=env,
                log_path=case_dir / "bridge.log",
                dry_run=args.dry_run,
                show_dry_run_command=_private_console_enabled(args),
            )
            if args.dry_run:
                _run_logged(
                    _parity_command(
                        args,
                        bridge_report=bridge_report,
                        prefix_path=prefix_path,
                        workdir=case_dir / "parity_work",
                        report_path=parity_report,
                        runtime_dir=runtime_dir,
                    ),
                    env=env,
                    log_path=case_dir / "parity.log",
                    dry_run=True,
                    show_dry_run_command=_private_console_enabled(args),
                )
                if not args.skip_native_cli:
                    _run_logged(
                        _native_cli_command(
                            args,
                            bridge_report=bridge_report,
                            runtime_dir=runtime_dir,
                            trace_path=native_trace,
                        ),
                        env=env,
                        log_path=case_dir / "native_cli.log",
                        dry_run=True,
                        show_dry_run_command=_private_console_enabled(args),
                    )
                continue
            source_report = runtime_dir / "bridge_report.json"
            if not source_report.is_file():
                raise FileNotFoundError(f"bridge did not produce {source_report}")
            shutil.copy2(source_report, bridge_report)
            bridge_report.chmod(0o600)
            prefix_sha256 = _sha256_file(prefix_path)
            stage = "numerical_parity"
            elapsed["parity"] = _run_logged(
                _parity_command(
                    args,
                    bridge_report=bridge_report,
                    prefix_path=prefix_path,
                    workdir=case_dir / "parity_work",
                    report_path=parity_report,
                    runtime_dir=runtime_dir,
                ),
                env=env,
                log_path=case_dir / "parity.log",
                dry_run=False,
                accepted_returncodes=(0, 3),
            )
            report = json.loads(parity_report.read_text(encoding="utf-8"))
            native_comparison = None
            if not args.skip_native_cli:
                stage = "native_cli"
                elapsed["native_cli"] = _run_logged(
                    _native_cli_command(
                        args,
                        bridge_report=bridge_report,
                        runtime_dir=runtime_dir,
                        trace_path=native_trace,
                    ),
                    env=env,
                    log_path=case_dir / "native_cli.log",
                    dry_run=False,
                )
                trace = json.loads(native_trace.read_text(encoding="utf-8"))
                native_comparison = _compare_native_trace(report, trace)
                _json_write(native_comparison_path, native_comparison)
            row = _redacted_row(
                index=index,
                image_sha256=sample["image_sha256"],
                prefix_sha256=prefix_sha256,
                report=report,
                elapsed=elapsed,
                requested_tokens=args.max_new_tokens,
                native_comparison=native_comparison,
            )
            rows.append(row)
            _json_write(
                result_path,
                {
                    "case_config": case_config,
                    "redacted_row": row,
                    "private_artifacts": {
                        "bridge_report": str(bridge_report),
                        "parity_report": str(parity_report),
                        "bridge_log": str(case_dir / "bridge.log"),
                        "parity_log": str(case_dir / "parity.log"),
                        **(
                            {
                                "native_trace": str(native_trace),
                                "native_comparison": str(native_comparison_path),
                                "native_cli_log": str(case_dir / "native_cli.log"),
                            }
                            if not args.skip_native_cli
                            else {}
                        ),
                    },
                },
            )
            if row["status"] == "pass" and not args.keep_prefixes:
                prefix_path.unlink(missing_ok=True)
            print(_progress_line(row, completed=len(rows), requested=len(selected)))
            if _private_console_enabled(args):
                _print_private_case_details(
                    sample=sample,
                    row=row,
                    case_dir=case_dir,
                    prompt=args.prompt,
                    model_label=args.model_label,
                )
            if row["status"] != "pass" and not args.continue_on_failure:
                break
        except Exception as exc:
            row = {
                "image_index": index,
                "image_sha256": sample["image_sha256"],
                "status": "error",
                "failure_stage": stage,
                "error_type": type(exc).__name__,
                "error_sha256": hashlib.sha256(str(exc).encode("utf-8")).hexdigest(),
                "elapsed_sec": elapsed,
            }
            rows.append(row)
            _json_write(
                result_path,
                {
                    "case_config": case_config,
                    "redacted_row": row,
                    "private_error": str(exc),
                },
            )
            print(
                f"[{len(rows)}/{len(selected)}] image {index:02d}: "
                f"ERROR {type(exc).__name__}; inspect the local case result",
                file=sys.stderr,
            )
            # --continue-on-failure applies to completed numerical comparisons.
            # Execution/setup errors are global until proven otherwise; fail
            # immediately instead of repeating an expensive broken run.
            break
        finally:
            _json_write(args.output_dir / "summary.json", _summary(selected=selected, rows=rows, config=config))

    if args.dry_run:
        return 0
    summary = _summary(selected=selected, rows=rows, config=config)
    _json_write(args.output_dir / "summary.json", summary)
    print(
        f"status={summary['status']} completed={summary['completed']}/{summary['requested']} "
        f"passed={summary['passed']} failed={summary['failed']} "
        f"total={summary['timing']['total_sec']:.2f}s "
        f"mean={summary['timing']['mean_sec_per_image']:.2f}s/image "
        f"report={args.output_dir / 'summary.json'}"
    )
    return 0 if summary["status"] == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
