#!/usr/bin/env python3
"""Certify a prebuilt v8 multimodal runtime on a ground-truth OCR corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
BRIDGE = ROOT / "version" / "v8" / "scripts" / "run_multimodal_bridge_v8.py"
SCHEMA = "cke.multimodal_ocr_corpus_certification"
SCHEMA_VERSION = 1
DEFAULT_PROMPT = (
    "Extract every visible form field as one compact JSON object. Preserve the requested "
    "field names exactly. Use an empty string for a blank field and selected or unselected "
    "for checkboxes. Return JSON only, without Markdown. Fields: {fields}"
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.chmod(0o600)
    os.replace(temporary, path)


def _resolve_file(base: Path, raw: Any, label: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{label} path is missing")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    return path


def _load_samples(manifest_path: Path) -> list[dict[str, Any]]:
    manifest_path = manifest_path.resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = payload.get("samples") if isinstance(payload, dict) else None
    if not isinstance(samples, list) or not samples:
        raise ValueError("corpus manifest must contain a non-empty samples list")
    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        if not isinstance(sample, dict):
            raise ValueError(f"sample {index} must be an object")
        inputs = sample.get("inputs")
        truths = sample.get("groundTruth")
        if not isinstance(inputs, list) or len(inputs) != 1 or not isinstance(inputs[0], dict):
            raise ValueError(f"sample {index} must have exactly one image input")
        if not isinstance(truths, list) or len(truths) != 1 or not isinstance(truths[0], dict):
            raise ValueError(f"sample {index} must have exactly one ground-truth JSON input")
        image = _resolve_file(manifest_path.parent, inputs[0].get("path"), "image")
        truth_path = _resolve_file(manifest_path.parent, truths[0].get("path"), "ground truth")
        image_sha256 = _sha256_file(image)
        truth_sha256 = _sha256_file(truth_path)
        pinned_image_sha256 = inputs[0].get("sha256")
        pinned_truth_sha256 = truths[0].get("sha256")
        if pinned_image_sha256 is not None and pinned_image_sha256 != image_sha256:
            raise ValueError(
                f"sample {index} image SHA-256 mismatch: "
                f"expected {pinned_image_sha256}, got {image_sha256}"
            )
        if pinned_truth_sha256 is not None and pinned_truth_sha256 != truth_sha256:
            raise ValueError(
                f"sample {index} ground-truth SHA-256 mismatch: "
                f"expected {pinned_truth_sha256}, got {truth_sha256}"
            )
        truth = json.loads(truth_path.read_text(encoding="utf-8"))
        if not isinstance(truth, dict):
            raise ValueError(f"sample {index} ground truth must be a JSON object")
        prompt = sample.get("prompt")
        if prompt is not None and (not isinstance(prompt, str) or not prompt.strip()):
            raise ValueError(f"sample {index} prompt must be a non-empty string")
        comparison = sample.get("comparison") or {}
        if not isinstance(comparison, dict):
            raise ValueError(f"sample {index} comparison must be an object")
        max_new_tokens = comparison.get("max_new_tokens")
        if max_new_tokens is not None and (
            not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens <= 0
        ):
            raise ValueError(f"sample {index} comparison.max_new_tokens must be positive")
        rows.append(
            {
                "index": index,
                "id": str(sample.get("id") or f"case-{index:03d}"),
                "image": image,
                "image_sha256": image_sha256,
                "truth_path": truth_path,
                "truth_sha256": truth_sha256,
                "truth": truth,
                "prompt": prompt,
                "comparison": comparison,
            }
        )
    return rows


def _extract_json_object(text: str) -> dict[str, Any] | None:
    candidates = [text.strip()]
    candidates.extend(re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.I | re.S))
    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
        for match in re.finditer(r"\{", candidate):
            try:
                value, _ = decoder.raw_decode(candidate[match.start() :])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    return None


def _normalized(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        value = str(value)
    elif isinstance(value, (dict, list)):
        value = json.dumps(value, sort_keys=True, separators=(",", ":"))
    text = str(value).strip().casefold()
    return re.sub(r"[^a-z0-9]+", "", text)


def _accepted_values(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def _score(expected: dict[str, Any], actual: dict[str, Any] | None) -> dict[str, Any]:
    if actual is None:
        return {
            "json_valid": False,
            "expected_fields": len(expected),
            "present_fields": 0,
            "exact_fields": 0,
            "nonempty_expected_fields": sum(_normalized(value) != "" for value in expected.values()),
            "nonempty_exact_fields": 0,
            "field_accuracy": 0.0,
            "nonempty_field_accuracy": 0.0,
            "missing_fields": sorted(expected),
            "mismatched_fields": [],
            "extra_fields": [],
        }
    exact: list[str] = []
    mismatched: list[str] = []
    missing: list[str] = []
    nonempty_expected = 0
    nonempty_exact = 0
    for key, expected_value in expected.items():
        accepted = {_normalized(value) for value in _accepted_values(expected_value)}
        is_nonempty = any(accepted)
        nonempty_expected += int(is_nonempty)
        if key not in actual:
            missing.append(key)
            continue
        if _normalized(actual[key]) in accepted:
            exact.append(key)
            nonempty_exact += int(is_nonempty)
        else:
            mismatched.append(key)
    total = len(expected)
    return {
        "json_valid": True,
        "expected_fields": total,
        "present_fields": sum(key in actual for key in expected),
        "exact_fields": len(exact),
        "nonempty_expected_fields": nonempty_expected,
        "nonempty_exact_fields": nonempty_exact,
        "field_accuracy": len(exact) / total if total else 1.0,
        "nonempty_field_accuracy": nonempty_exact / nonempty_expected if nonempty_expected else 1.0,
        "missing_fields": missing,
        "mismatched_fields": mismatched,
        "extra_fields": sorted(set(actual) - set(expected)),
    }


def _runtime_identity(path: Path, role: str) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(path)
    if role == "encoder":
        libraries = sorted(path.glob("lib*encoder*.so"))
        if len(libraries) != 1:
            raise ValueError(f"encoder runtime requires exactly one model library: {path}")
        model_library = libraries[0]
    elif role == "decoder":
        model_library = path / "libmodel.so"
    else:
        raise ValueError(f"unknown runtime role: {role}")
    engine = path / "libckernel_engine.so"
    if not model_library.is_file() or not engine.is_file():
        raise FileNotFoundError(f"{role} runtime requires model and engine libraries: {path}")
    files: dict[str, str] = {}
    for name in ("weights.bump", "weights_manifest.map"):
        candidate = path / name
        if candidate.is_file():
            files[name] = _sha256_file(candidate)
    return {
        "path": str(path),
        "model_library": {"path": str(model_library.resolve()), "sha256": _sha256_file(model_library)},
        "engine_library": {"path": str(engine.resolve()), "sha256": _sha256_file(engine)},
        "files": files,
    }


def _verify_bridge_execution(
    report: dict[str, Any], sample: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    if report.get("status") != "ok" or report.get("prefix_source") != "encoder":
        raise ValueError("OCR requires a successful image-fed encoder prefix")
    encoder = report.get("encoder_report")
    if not isinstance(encoder, dict) or encoder.get("image_source") != "file":
        raise ValueError("OCR requires an encoder report from an image file")
    if Path(str(encoder.get("image_path") or "")).resolve() != sample["image"].resolve():
        raise ValueError("encoder image path does not match the corpus case")
    if _sha256_file(sample["image"]) != sample["image_sha256"]:
        raise ValueError("encoder image bytes changed during OCR execution")
    prefix_tokens = report.get("prefix_tokens")
    if (not isinstance(prefix_tokens, int) or isinstance(prefix_tokens, bool)
            or prefix_tokens <= 0 or encoder.get("prefix_tokens") != prefix_tokens):
        raise ValueError("encoder prefix token count is missing or inconsistent")
    libraries: dict[str, str] = {}
    for role in ("encoder", "decoder"):
        expected = config[f"{role}_runtime"]
        observed = report.get(f"{role}_runtime")
        if not isinstance(observed, dict) or observed.get("source") != "prebuilt":
            raise ValueError(f"OCR requires the declared prebuilt {role} runtime")
        if Path(str(observed.get("workdir") or "")).resolve() != Path(expected["path"]):
            raise ValueError(f"{role} runtime directory differs from the pinned runtime")
        library = expected["model_library"]
        if Path(str(observed.get("so_path") or "")).resolve() != Path(library["path"]):
            raise ValueError(f"{role} model library differs from the pinned runtime")
        if _sha256_file(Path(library["path"])) != library["sha256"]:
            raise ValueError(f"{role} model library changed during OCR execution")
        engine = expected["engine_library"]
        if _sha256_file(Path(engine["path"])) != engine["sha256"]:
            raise ValueError(f"{role} engine library changed during OCR execution")
        libraries[role] = library["sha256"]
    # The bridge reports selected paths; these hashes do not prove which engine ELF was loaded.
    return {
        "evidence_kind": "bridge_reported_paths_and_artifact_hashes",
        "loaded_engine_verified": False,
        "prefix_source": "encoder",
        "prefix_tokens": prefix_tokens,
        "model_library_sha256": libraries,
    }


def _build_prompt(
    template: str,
    truth: dict[str, Any],
    sample_prompt: str | None = None,
) -> str:
    if sample_prompt is not None:
        return sample_prompt
    fields = ", ".join(sorted(truth))
    return template.format(fields=fields)


def _max_new_tokens(default: int, sample: dict[str, Any]) -> int:
    return int(sample.get("comparison", {}).get("max_new_tokens", default))


def _bridge_command(
    args: argparse.Namespace,
    sample: dict[str, Any],
    case_dir: Path,
    prompt: str,
) -> list[str]:
    command = [
        sys.executable,
        str(BRIDGE),
        "--decoder-runtime",
        str(args.decoder_runtime),
        "--encoder-runtime",
        str(args.encoder_runtime),
        "--composition-circuit",
        args.composition_circuit,
        "--workdir",
        str(case_dir / "runtime"),
        "--image-path",
        str(sample["image"]),
        "--prompt",
        prompt,
        "--chat-template",
        args.chat_template,
        "--thinking-mode",
        args.thinking_mode,
        "--decoder-context-len",
        str(args.context_len),
        "--max-tokens",
        str(_max_new_tokens(args.max_new_tokens, sample)),
        "--temperature",
        "0",
        "--top-p",
        "1",
        "--repeat-penalty",
        "1",
        "--no-stream-output",
        "--generation-progress-every",
        str(args.generation_progress_every),
    ]
    if args.adapt_encoder_geometry:
        command.extend(
            [
                "--encoder-geometry-cache-dir",
                str(args.output_dir.resolve() / "encoder_geometry_cache"),
            ]
        )
    return command


def _run(command: list[str], log_path: Path, env: dict[str, str]) -> float:
    started = time.perf_counter()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        log_path.chmod(0o600)
        stream.write(f"$ {shlex.join(command)}\n\n")
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
    if completed.returncode:
        raise RuntimeError(f"bridge failed rc={completed.returncode}; inspect {log_path}")
    return elapsed


def _case_config(
    global_hash: str,
    sample: dict[str, Any],
    prompt: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    return {
        "global_config_sha256": global_hash,
        "image_index": sample["index"],
        "image_sha256": sample["image_sha256"],
        "truth_sha256": sample["truth_sha256"],
        "prompt_sha256": _sha256_bytes(prompt.encode("utf-8")),
        "max_new_tokens": max_new_tokens,
    }


def _load_resumed(
    path: Path, expected: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    evidence = value.get("execution_evidence")
    if not isinstance(evidence, dict):
        return None
    hashes = evidence.get("model_library_sha256")
    prefix_tokens = evidence.get("prefix_tokens")
    if (value.get("case_config") != expected or value.get("status") != "complete"
            or value.get("image_index") != expected["image_index"]
            or value.get("image_sha256") != expected["image_sha256"]
            or value.get("truth_sha256") != expected["truth_sha256"]
            or evidence.get("evidence_kind") != "bridge_reported_paths_and_artifact_hashes"
            or evidence.get("loaded_engine_verified") is not False
            or evidence.get("prefix_source") != "encoder"
            or not isinstance(prefix_tokens, int) or isinstance(prefix_tokens, bool)
            or prefix_tokens <= 0 or not isinstance(hashes, dict)
            or any(hashes.get(role) != config[f"{role}_runtime"]["model_library"]["sha256"]
                   for role in ("encoder", "decoder"))):
        return None
    return value


def _public_row(case: dict[str, Any]) -> dict[str, Any]:
    return {
        key: case[key]
        for key in (
            "image_index",
            "image_sha256",
            "truth_sha256",
            "status",
            "output_sha256",
            "token_trace_sha256",
            "stop_reason",
            "generated_tokens",
            "timings",
            "metrics",
            "repeatability",
            "execution_evidence",
        )
        if key in case
    }


def _aggregate(rows: list[dict[str, Any]], requested: int) -> dict[str, Any]:
    completed = [row for row in rows if row.get("status") == "complete"]
    repeated = [row for row in completed if isinstance(row.get("repeatability"), dict)]
    expected = sum(int(row["metrics"]["expected_fields"]) for row in completed)
    exact = sum(int(row["metrics"]["exact_fields"]) for row in completed)
    nonempty_expected = sum(int(row["metrics"]["nonempty_expected_fields"]) for row in completed)
    nonempty_exact = sum(int(row["metrics"]["nonempty_exact_fields"]) for row in completed)
    return {
        "requested": requested,
        "completed": len(completed),
        "errors": len(rows) - len(completed),
        "json_valid": sum(bool(row["metrics"]["json_valid"]) for row in completed),
        "expected_fields": expected,
        "exact_fields": exact,
        "field_accuracy": exact / expected if expected else 0.0,
        "nonempty_expected_fields": nonempty_expected,
        "nonempty_exact_fields": nonempty_exact,
        "nonempty_field_accuracy": nonempty_exact / nonempty_expected if nonempty_expected else 0.0,
        "total_wall_sec": sum(float(row["timings"].get("wall_sec", 0.0)) for row in completed),
        "generated_tokens": sum(int(row.get("generated_tokens", 0)) for row in completed),
        "repeatability_cases": len(repeated),
        "repeatable_cases": sum(bool(row["repeatability"].get("exact")) for row in repeated),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--decoder-runtime", type=Path, required=True)
    parser.add_argument("--encoder-runtime", type=Path, required=True)
    parser.add_argument("--composition-circuit", required=True)
    parser.add_argument("--adapter-id", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-template", default=DEFAULT_PROMPT)
    parser.add_argument("--chat-template", default="auto")
    parser.add_argument("--thinking-mode", choices=("auto", "visible", "suppressed"), default="suppressed")
    parser.add_argument("--context-len", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=896)
    parser.add_argument("--threads", type=int, default=0, help="CK_THREADS override; 0 uses runtime auto-detection")
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--require-images", type=int)
    parser.add_argument("--generation-progress-every", type=int, default=128)
    parser.add_argument(
        "--adapt-encoder-geometry",
        action="store_true",
        help="Use the encoder's declared native-resolution geometry contract",
    )
    parser.add_argument("--oracle-id", default="none")
    parser.add_argument("--oracle-status", choices=("unsupported", "not_configured", "available"), default="not_configured")
    parser.add_argument("--oracle-note", default="")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.context_len <= 0 or args.max_new_tokens <= 0:
        parser.error("context and generation limits must be positive")
    samples = _load_samples(args.manifest)
    if args.require_images is not None and len(samples) < args.require_images:
        parser.error(f"manifest has {len(samples)} images; {args.require_images} required")
    selected = [row for row in samples if row["index"] >= args.start_index]
    if args.limit is not None:
        selected = selected[: args.limit]
    if not selected:
        parser.error("selection contains no samples")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.chmod(0o700)
    config = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "adapter_id": args.adapter_id,
        "model_label": args.model_label,
        "manifest_sha256": _sha256_file(args.manifest.resolve()),
        "decoder_runtime": _runtime_identity(args.decoder_runtime, "decoder"),
        "encoder_runtime": _runtime_identity(args.encoder_runtime, "encoder"),
        "composition_circuit": args.composition_circuit,
        "chat_template": args.chat_template,
        "thinking_mode": args.thinking_mode,
        "context_len": args.context_len,
        "max_new_tokens": args.max_new_tokens,
        "prompt_template_sha256": _sha256_bytes(args.prompt_template.encode("utf-8")),
        "threads": args.threads,
        "adapt_encoder_geometry": bool(args.adapt_encoder_geometry),
        "oracle": {
            "id": args.oracle_id,
            "status": args.oracle_status,
            "note": args.oracle_note,
            "comparison": "task_metric",
        },
    }
    global_hash = _sha256_json(config)
    _write_json(output_dir / "config.json", {**config, "config_sha256": global_hash})

    env = os.environ.copy()
    if args.threads > 0:
        env["CK_THREADS"] = str(args.threads)
    rows: list[dict[str, Any]] = []
    stop_after_failure = False
    for completed, sample in enumerate(selected, start=1):
        case_dir = output_dir / f"image{sample['index']:02d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        case_dir.chmod(0o700)
        prompt = _build_prompt(args.prompt_template, sample["truth"], sample["prompt"])
        max_new_tokens = _max_new_tokens(args.max_new_tokens, sample)
        case_config = _case_config(global_hash, sample, prompt, max_new_tokens)
        case_result = case_dir / "case_result.json"
        resumed = None if args.force_rerun else _load_resumed(case_result, case_config, config)
        if resumed is not None:
            rows.append(resumed)
            print(f"[{completed}/{len(selected)}] image {sample['index']:02d}: resumed")
            continue
        if args.dry_run:
            print(shlex.join(_bridge_command(args, sample, case_dir, prompt)))
            continue
        try:
            bridge_path = case_dir / "runtime" / "bridge_report.json"
            bridge_path.unlink(missing_ok=True)
            wall_sec = _run(_bridge_command(args, sample, case_dir, prompt), case_dir / "bridge.log", env)
            report = json.loads(bridge_path.read_text(encoding="utf-8"))
            execution_evidence = _verify_bridge_execution(report, sample, config)
            generated_text = str(report.get("generated_text", ""))
            generated_tokens = [int(value) for value in report.get("generated_token_ids") or []]
            parsed = _extract_json_object(generated_text)
            metrics = _score(sample["truth"], parsed)
            timings = dict(report.get("timings") or {})
            timings["wall_sec"] = wall_sec
            row = {
                "schema": "cke.multimodal_ocr_case",
                "schema_version": 1,
                "case_config": case_config,
                "image_index": sample["index"],
                "image_id": sample["id"],
                "image_path": str(sample["image"]),
                "image_sha256": sample["image_sha256"],
                "truth_path": str(sample["truth_path"]),
                "truth_sha256": sample["truth_sha256"],
                "status": "complete",
                "execution_evidence": execution_evidence,
                "prompt": prompt,
                "generated_text": generated_text,
                "parsed_output": parsed,
                "output_sha256": _sha256_bytes(generated_text.encode("utf-8")),
                "token_trace_sha256": _sha256_json(generated_tokens),
                "generated_token_ids": generated_tokens,
                "generated_tokens": len(generated_tokens),
                "stop_reason": report.get("generation_stop_reason"),
                "timings": timings,
                "metrics": metrics,
            }
            _write_json(case_result, row)
            rows.append(row)
            print(
                f"[{completed}/{len(selected)}] image {sample['index']:02d}: "
                f"json={'yes' if metrics['json_valid'] else 'no'} "
                f"fields={metrics['exact_fields']}/{metrics['expected_fields']} "
                f"nonempty={metrics['nonempty_exact_fields']}/{metrics['nonempty_expected_fields']} "
                f"tokens={len(generated_tokens)} wall={wall_sec:.2f}s"
            )
        except Exception as exc:
            row = {
                "case_config": case_config,
                "image_index": sample["index"],
                "image_sha256": sample["image_sha256"],
                "truth_sha256": sample["truth_sha256"],
                "status": "error",
                "error": str(exc),
            }
            _write_json(case_result, row)
            rows.append(row)
            print(f"[{completed}/{len(selected)}] image {sample['index']:02d}: ERROR {exc}")
            if not args.continue_on_failure:
                stop_after_failure = True

        public_rows = [_public_row(row) for row in rows if row.get("status") == "complete"]
        aggregate = _aggregate(rows, len(selected))
        _write_json(
            output_dir / "summary.json",
            {
                "schema": SCHEMA,
                "schema_version": SCHEMA_VERSION,
                "status": "complete" if aggregate["completed"] == len(selected) else "incomplete",
                "config_sha256": global_hash,
                "adapter_id": args.adapter_id,
                "model_label": args.model_label,
                "comparison": "task_metric",
                "oracle": config["oracle"],
                "aggregate": aggregate,
                "rows": public_rows,
            },
        )
        if stop_after_failure:
            break

    if args.dry_run:
        return 0
    aggregate = _aggregate(rows, len(selected))
    return 0 if aggregate["completed"] == len(selected) else 1


if __name__ == "__main__":
    raise SystemExit(main())
