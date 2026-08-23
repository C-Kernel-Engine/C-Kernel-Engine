#!/usr/bin/env python3
from __future__ import annotations

"""Certify production-formatted text prompts against a pinned llama.cpp oracle."""

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path[:0] = [str(SCRIPT_DIR), str(REPO_ROOT / "scripts")]

from compare_multitoken_logits_v8 import (  # type: ignore  # noqa: E402
    run_multitoken_trajectory_parity,
    run_multitoken_trajectory_parity_streaming,
)
from gguf_tokenizer import GGUFTokenizer  # type: ignore  # noqa: E402
from run_multimodal_bridge_v8 import (  # type: ignore  # noqa: E402
    _encode_prompt_segment,
    _format_prompt_with_chat_contract,
    _resolve_decoder_chat_contract,
)
from xray_text_recurrent_v8 import capture_and_analyze  # type: ignore  # noqa: E402


CORRUPTION_MARKERS = (
    "\\uFFFD",
    "\ufffd",
    "\u00c3",
    "\u00c2",
    "\u00e2\u20ac",
    "\u00f0\u0178",
    "\ufffd\u0141",
)

STREAMING_TRAJECTORY_BYTES = 256 * 1024 * 1024
SVG_GRAPHIC_ELEMENTS = {
    "circle", "ellipse", "g", "image", "line", "path", "polygon",
    "polyline", "rect", "text", "use",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_prompt_set(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("prompt set schema_version must be 1")
    stages = [int(value) for value in payload.get("stages", [])]
    if not stages or stages != sorted(set(stages)) or any(value <= 0 for value in stages):
        raise ValueError("prompt stages must be unique, positive, and increasing")
    prompts = payload.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("prompt set must contain prompts")
    ids = [str(row.get("id", "")) for row in prompts]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("prompt IDs must be non-empty and unique")
    xray = payload.get("xray")
    if xray is not None:
        if not isinstance(xray, dict):
            raise ValueError("xray contract must be an object")
        if int(xray.get("stage", 0)) not in stages:
            raise ValueError("xray stage must be one of the prompt stages")
        layers = xray.get("layers")
        if not isinstance(layers, list) or not layers or any(int(value) < 0 for value in layers):
            raise ValueError("xray layers must be a non-empty list of non-negative integers")
    return payload


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def split_visible_reasoning(text: str) -> tuple[str, str, bool]:
    stripped = text.strip()
    if not stripped.lower().startswith("<think>"):
        return "", stripped, True
    end = stripped.lower().find("</think>")
    if end < 0:
        return stripped[len("<think>") :].strip(), "", False
    reasoning = stripped[len("<think>") : end].strip()
    answer = stripped[end + len("</think>") :].strip()
    return reasoning, answer, True


def evaluate_quality_contract(text: str, contract: dict[str, Any] | None) -> dict[str, Any]:
    if not contract:
        return {"pass": True, "kind": "none"}
    kind = str(contract.get("kind", ""))
    if kind != "standalone_svg.v1":
        raise ValueError(f"unsupported quality contract: {kind!r}")

    reasoning, stripped, reasoning_complete = split_visible_reasoning(text)
    lowered = stripped.lower()
    start = lowered.find("<svg")
    end = lowered.rfind("</svg>")
    markup = stripped[start : end + len("</svg>")] if start >= 0 and end >= start else ""
    result: dict[str, Any] = {
        "kind": kind,
        "pass": False,
        "reasoning_present": bool(reasoning),
        "reasoning_complete": reasoning_complete,
        "reasoning_characters": len(reasoning),
        "answer_characters": len(stripped),
        "output_only": bool(markup and stripped == markup),
        "xml_parseable": False,
        "root_is_svg": False,
        "has_viewbox": False,
        "has_title": False,
        "has_description": False,
        "graphic_element_count": 0,
        "required_labels_present": False,
        "missing_labels": [],
        "has_arrow_marker": False,
        "has_script": False,
        "has_external_reference": False,
    }
    if not markup:
        return result
    try:
        root = ET.fromstring(markup)
    except ET.ParseError as exc:
        result["parse_error"] = str(exc)
        return result

    result["xml_parseable"] = True
    result["root_is_svg"] = _local_name(root.tag) == "svg"
    viewbox = str(root.attrib.get("viewBox", root.attrib.get("viewbox", ""))).split()
    result["has_viewbox"] = len(viewbox) == 4
    elements = list(root.iter())
    names = [_local_name(element.tag) for element in elements]
    result["has_title"] = "title" in names
    result["has_description"] = "desc" in names
    result["graphic_element_count"] = sum(name in SVG_GRAPHIC_ELEMENTS for name in names)
    visible_text = " ".join(
        " ".join(element.itertext()) for element in elements if _local_name(element.tag) == "text"
    ).lower()
    required_labels = [str(value).strip().lower() for value in contract.get("required_labels", [])]
    result["missing_labels"] = [value for value in required_labels if value not in visible_text]
    result["required_labels_present"] = not result["missing_labels"]
    marker_ids = {
        str(element.attrib.get("id", ""))
        for element in elements
        if _local_name(element.tag) == "marker" and str(element.attrib.get("id", ""))
    }
    result["has_arrow_marker"] = bool(marker_ids) and any(
        any(f"url(#{marker_id})" in str(value) for marker_id in marker_ids)
        for element in elements
        for attribute, value in element.attrib.items()
        if _local_name(attribute) in {"marker-end", "marker-start", "marker-mid"}
    )
    result["has_script"] = "script" in names
    for element in elements:
        for attribute, value in element.attrib.items():
            if _local_name(attribute) == "href" and str(value).strip() and not str(value).strip().startswith("#"):
                result["has_external_reference"] = True
            if "url(" in str(value).lower() and "http" in str(value).lower():
                result["has_external_reference"] = True
    checks = (
        result["output_only"],
        result["reasoning_complete"],
        result["xml_parseable"],
        result["root_is_svg"],
        result["has_viewbox"],
        result["has_title"],
        result["has_description"],
        int(result["graphic_element_count"]) >= int(contract.get("min_graphic_elements", 1)),
        result["required_labels_present"],
        result["has_arrow_marker"] if contract.get("require_arrow_marker") else True,
        not result["has_script"],
        not result["has_external_reference"],
    )
    result["pass"] = all(checks)
    return result


def xray_report_is_complete(report: dict[str, Any]) -> bool:
    rows = report.get("rows")
    return bool(
        report.get("first_divergence") is None
        and isinstance(rows, list)
        and rows
        and all(row.get("status") == "exact" for row in rows)
    )


def trajectory_numerical_contract(report: dict[str, Any], require_bit_exact: bool) -> dict[str, Any]:
    rows = report.get("steps") if isinstance(report.get("steps"), list) else []
    exact_rows = sum(
        bool(row.get("bit_exact", float(row.get("max_abs_diff", 1.0)) == 0.0))
        for row in rows
    )
    top1_rows = sum(bool(row.get("top1_match")) for row in rows)
    passed = bool(
        rows
        and report.get("first_divergence") is None
        and top1_rows == len(rows)
        and (not require_bit_exact or exact_rows == len(rows))
    )
    return {
        "pass": passed,
        "require_bit_exact": bool(require_bit_exact),
        "compared_rows": len(rows),
        "top1_exact_rows": top1_rows,
        "bit_exact_rows": exact_rows,
    }


def trajectory_uses_streaming(model_dir: Path, stage: int) -> bool:
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    vocab_size = int(config.get("vocab_size", 0))
    return vocab_size > 0 and vocab_size * int(stage) * 4 > STREAMING_TRAJECTORY_BYTES


def runtime_context_capacity(model_dir: Path) -> int:
    runtime = ctypes.CDLL(str(model_dir / "libmodel.so"))
    try:
        getter = runtime.ck_model_get_context_window
    except AttributeError as exc:
        raise ValueError("model runtime does not expose its context capacity") from exc
    getter.argtypes = []
    getter.restype = ctypes.c_int
    capacity = int(getter())
    if capacity <= 0:
        raise ValueError(f"model runtime reported invalid context capacity: {capacity}")
    return capacity


def format_and_tokenize_prompts(
    prompt_set: dict[str, Any], gguf_path: Path
) -> list[dict[str, Any]]:
    tokenizer = GGUFTokenizer.from_gguf(str(gguf_path))
    contract = _resolve_decoder_chat_contract(
        gguf_path, chat_template_mode=str(prompt_set.get("chat_template_mode", "auto"))
    )
    rows: list[dict[str, Any]] = []
    for source in prompt_set["prompts"]:
        formatted = _format_prompt_with_chat_contract(
            str(source["text"]),
            contract,
            thinking_mode=str(prompt_set.get("thinking_mode", "auto")),
        )
        tokens = _encode_prompt_segment(tokenizer, formatted, add_bos=True)
        expected = [int(value) for value in source.get("tokens", [])]
        if tokens != expected:
            raise ValueError(
                f"production prompt tokens changed for {source['id']}: "
                f"expected={expected} actual={tokens}"
            )
        rows.append({**source, "formatted": formatted, "tokens": tokens})
    return rows


def decoded_text_is_clean(text: str) -> bool:
    return not any(marker in text for marker in CORRUPTION_MARKERS)


def report_satisfies_stage(report: dict[str, Any], stage: int) -> bool:
    trajectory_pass = (
        report.get("first_divergence") is None
        if "first_divergence" in report
        else bool(report.get("pass"))
    )
    if not trajectory_pass:
        return False
    if report.get("matched_stop_token") is not None:
        return True
    return len(report.get("steps", [])) >= int(stage)


def reusable_report_path(output_dir: Path, prompt_id: str, stages: list[int], stage: int) -> Path | None:
    for previous_stage in reversed([value for value in stages if value < stage]):
        candidate = output_dir / f"{prompt_id}-{previous_stage}.json"
        if not candidate.exists():
            continue
        report = json.loads(candidate.read_text(encoding="utf-8"))
        if report_satisfies_stage(report, stage):
            return candidate
    return None


def xray_handoff(
    model_dir: Path, gguf_path: Path, parity_report: Path, output_root: Path, threads: int
) -> str:
    capture_root = output_root / f"{parity_report.stem}-xray"
    return (
        "python3 version/v8/scripts/xray_text_recurrent_v8.py "
        f"--model-dir {model_dir} --gguf {gguf_path} "
        f"--parity-report {parity_report} --capture-root {capture_root} "
        f"--output {capture_root / 'report.json'} --ctx-len 1034 "
        f"--threads {threads} --ck-prefill-mode hybrid"
    )


def git_head(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--gguf", required=True, type=Path)
    parser.add_argument("--prompt-set", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ctx-len", type=int, default=1034)
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    gguf_path = args.gguf.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_set = load_prompt_set(args.prompt_set.resolve())
    prompts = format_and_tokenize_prompts(prompt_set, gguf_path)
    required_context = max(
        len(prompt["tokens"]) + int(prompt_set["stages"][-1])
        for prompt in prompts
    )
    runtime_capacity = runtime_context_capacity(model_dir)
    if int(args.ctx_len) < required_context:
        raise ValueError(
            f"requested context {args.ctx_len} cannot hold the longest certification "
            f"trajectory ({required_context} tokens)"
        )
    if runtime_capacity < required_context:
        raise ValueError(
            f"model runtime capacity {runtime_capacity} cannot hold the longest "
            f"certification trajectory ({required_context} tokens)"
        )

    llama_root_value = os.environ.get("CK_LLAMA_CPP_ROOT", "").strip()
    llama_root = Path(llama_root_value).resolve() if llama_root_value else None
    expected_llama_commit = str(prompt_set.get("llama_cpp_commit", ""))
    actual_llama_commit = git_head(llama_root) if llama_root else ""
    if not args.dry_run and actual_llama_commit != expected_llama_commit:
        raise RuntimeError(
            f"llama.cpp oracle commit mismatch: expected={expected_llama_commit} "
            f"actual={actual_llama_commit or 'unavailable'}"
        )

    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": "pass",
        "prompt_set": str(prompt_set["name"]),
        "stages": [int(value) for value in prompt_set["stages"]],
        "prompts": [],
        "xray": [],
        "provenance": {
            "cke_commit": git_head(REPO_ROOT),
            "llama_cpp_commit": actual_llama_commit,
            "gguf_sha256": sha256_file(gguf_path),
            "engine_sha256": sha256_file(model_dir / "libckernel_engine.so"),
            "model_runtime_sha256": sha256_file(model_dir / "libmodel.so"),
            "threads": int(args.threads),
            "requested_context": int(args.ctx_len),
            "runtime_context_capacity": runtime_capacity,
        },
    }
    if args.dry_run:
        summary["prompts"] = prompts
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return 0

    tokenizer = GGUFTokenizer.from_gguf(str(gguf_path))
    stop_tokens = {int(value) for value in prompt_set.get("stop_token_ids", [])}
    require_bit_exact = bool(prompt_set.get("require_bit_exact", False))
    final_stage = int(summary["stages"][-1])
    xray_contract = prompt_set.get("xray") or {}
    xray_prompt_ids = {str(value) for value in xray_contract.get("prompt_ids", [])}
    for stage in summary["stages"]:
        for prompt in prompts:
            report_path = output_dir / f"{prompt['id']}-{stage}.json"
            if report_path.exists():
                existing = json.loads(report_path.read_text(encoding="utf-8"))
                if report_satisfies_stage(existing, int(stage)):
                    report = existing
                else:
                    report = None
            else:
                report = None
            if report is None:
                reusable = reusable_report_path(
                    output_dir, str(prompt["id"]), summary["stages"], int(stage)
                )
                if reusable is not None:
                    report = json.loads(reusable.read_text(encoding="utf-8"))
                    report["certified_stage"] = int(stage)
                    report["reused_eos_report"] = str(reusable)
                else:
                    runner = (
                        run_multitoken_trajectory_parity_streaming
                        if trajectory_uses_streaming(model_dir, int(stage))
                        else run_multitoken_trajectory_parity
                    )
                    report = runner(
                        model_dir=model_dir,
                        gguf_path=gguf_path,
                        prompt_tokens=[int(value) for value in prompt["tokens"]],
                        max_new_tokens=int(stage),
                        ctx_len=int(args.ctx_len),
                        top_k=int(args.top_k),
                        threads=int(args.threads),
                        llama_no_repack=False,
                        stop_token_ids=stop_tokens,
                    )
            generated = report["final_prefix"][len(prompt["tokens"]) :]
            decoded = tokenizer.decode(generated, skip_special=True)
            report["prompt_id"] = str(prompt["id"])
            report["prompt_text"] = str(prompt["text"])
            report["formatted_prompt"] = str(prompt["formatted"])
            report["decoded_text"] = decoded
            report["utf8_clean"] = decoded_text_is_clean(decoded)
            report["numerical"] = trajectory_numerical_contract(report, require_bit_exact)
            report["trajectory_storage"] = (
                "stream" if trajectory_uses_streaming(model_dir, int(stage)) else "memory"
            )
            if int(stage) == final_stage:
                report["quality"] = evaluate_quality_contract(
                    decoded, prompt.get("quality_contract")
                )
            report["xray_handoff"] = xray_handoff(
                model_dir, gguf_path, report_path, output_dir, int(args.threads)
            )
            report["pass"] = bool(
                report["numerical"]["pass"]
                and report["utf8_clean"]
                and bool((report.get("quality") or {"pass": True})["pass"])
            )
            report["status"] = "pass" if report["pass"] else "fail"
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            if not report["pass"]:
                summary["status"] = "fail"
                summary["first_failure"] = {
                    "prompt_id": prompt["id"],
                    "stage": int(stage),
                    "report": str(report_path),
                    "xray_handoff": report["xray_handoff"],
                }
                (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
                print(json.dumps(summary, indent=2))
                return 3

            if (
                xray_contract
                and int(stage) == int(xray_contract["stage"])
                and (not xray_prompt_ids or str(prompt["id"]) in xray_prompt_ids)
            ):
                for layer in [int(value) for value in xray_contract["layers"]]:
                    xray_output = output_dir / f"{prompt['id']}-{stage}-xray-layer-{layer}.json"
                    capture_root = output_dir / f"{prompt['id']}-{stage}-xray-layer-{layer}"
                    if xray_output.exists():
                        xray_report = json.loads(xray_output.read_text(encoding="utf-8"))
                    else:
                        xray_report = capture_and_analyze(
                            model_dir,
                            gguf_path,
                            report_path,
                            capture_root,
                            layer,
                            int(args.ctx_len),
                            int(args.threads),
                            str(xray_contract.get("ck_prefill_mode", "hybrid")),
                        )
                        xray_output.write_text(
                            json.dumps(xray_report, indent=2), encoding="utf-8"
                        )
                    xray_pass = xray_report_is_complete(xray_report)
                    summary["xray"].append({
                        "prompt_id": str(prompt["id"]),
                        "stage": int(stage),
                        "layer": layer,
                        "layer_kind": xray_report.get("layer_kind"),
                        "rows": len(xray_report.get("rows") or []),
                        "status": "pass" if xray_pass else "fail",
                        "report": str(xray_output),
                    })
                    if not xray_pass:
                        summary["status"] = "fail"
                        summary["first_failure"] = {
                            "prompt_id": prompt["id"],
                            "stage": int(stage),
                            "layer": layer,
                            "report": str(xray_output),
                        }
                        (output_dir / "summary.json").write_text(
                            json.dumps(summary, indent=2), encoding="utf-8"
                        )
                        print(json.dumps(summary, indent=2))
                        return 3

    for prompt in prompts:
        final_path = output_dir / f"{prompt['id']}-{summary['stages'][-1]}.json"
        final_report = json.loads(final_path.read_text(encoding="utf-8"))
        summary["prompts"].append(
            {
                "id": prompt["id"],
                "status": final_report["status"],
                "steps": len(final_report.get("steps", [])),
                "matched_stop_token": final_report.get("matched_stop_token"),
                "utf8_clean": bool(final_report.get("utf8_clean")),
                "quality": final_report.get("quality"),
                "report": str(final_path),
            }
        )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
