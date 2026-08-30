#!/usr/bin/env python3
"""Run bounded Qwen3-VL BF16 PyTorch-vs-CK numerical X-ray diagnosis."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import build_xray_checkpoint_manifest_v8 as manifest_builder
import xray_numerical_parity_v8 as xray


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_CAPTURE_SCRIPT = SCRIPT_DIR / "compare_qwen3vl_bf16_vision_hidden_v8.py"
DEFAULT_PROFILE = SCRIPT_DIR.parent / "parity_profiles" / "qwen3vl_pytorch_bf16_v1.json"


def _call_checkpoints(call_ir: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for operation in call_ir.get("operations", []):
        for checkpoint in operation.get("semantic_checkpoints", []):
            checkpoint_id = str(checkpoint["id"])
            if checkpoint_id in result:
                raise RuntimeError(f"duplicate call-IR checkpoint: {checkpoint_id}")
            result[checkpoint_id] = checkpoint
    if not result:
        raise RuntimeError("call IR contains no semantic checkpoint ABI")
    return result


def _selector(checkpoint: Dict[str, Any]) -> str:
    layer = int(checkpoint.get("layer", -1))
    return f"{checkpoint['tensor']}@{layer}" if layer >= 0 else str(checkpoint["tensor"])


def _bounded_order(
    requested: list[str],
    available: Dict[str, Dict[str, Any]],
    lower_bound: str | None = None,
) -> list[str]:
    result = []
    if lower_bound and lower_bound != "<input>" and lower_bound in available:
        result.append(lower_bound)
    for checkpoint_id in requested:
        if checkpoint_id in available and checkpoint_id not in result:
            result.append(checkpoint_id)
    return result


def _apply_observed_storage(manifest: Dict[str, Any], observed_storage: Dict[str, Any]) -> None:
    """Apply parity-profile storage semantics per semantic checkpoint."""
    default = str(observed_storage["default"])
    overrides = observed_storage.get("checkpoints") or {}
    for checkpoint in manifest.get("checkpoints", []):
        checkpoint_id = str(checkpoint["checkpoint_id"])
        checkpoint["storage_dtype"] = str(overrides.get(checkpoint_id, default))


def _capture_command(
    args: argparse.Namespace,
    selectors: list[str],
    round_dir: Path,
) -> list[str]:
    return [
        str(args.oracle_python), str(args.capture_script),
        "--checkpoint", str(args.checkpoint),
        "--runtime-dir", str(args.runtime_dir),
        "--weights-bump", str(args.weights_bump),
        "--image", str(args.image),
        "--out-dir", str(round_dir / "capture"),
        "--threads", str(args.threads),
        "--attn-implementation", args.attn_implementation,
        "--architecture", args.architecture,
        "--model-so-name", args.model_so_name,
    ]


def _capture_environment(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["CK_NUM_THREADS"] = str(args.threads)
    env["OMP_NUM_THREADS"] = str(args.threads)
    python_paths = [str(path) for path in args.oracle_pythonpath]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    if python_paths:
        env["PYTHONPATH"] = os.pathsep.join(python_paths)
    return env


def _run_capture(args: argparse.Namespace, selectors: list[str], round_dir: Path) -> Path:
    cmd = _capture_command(args, selectors, round_dir)
    if args.torch_prefix:
        cmd.extend(["--torch-prefix", str(args.torch_prefix)])
    if args.ck_import_layer_input is not None:
        cmd.extend([
            "--ck-import-layer-input", str(args.ck_import_layer_input),
            "--ck-import-layer", str(args.ck_import_layer),
            "--ck-import-checkpoint", str(args.ck_import_checkpoint),
        ])
    for selector in selectors:
        cmd.extend(["--selector", selector])
    env = _capture_environment(args)
    log_path = round_dir / "capture.log"
    round_dir.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log:
        completed = subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        raise RuntimeError(f"bounded capture failed rc={completed.returncode}; see {log_path}")
    report = round_dir / "capture" / "report.json"
    if not report.is_file():
        raise RuntimeError(f"capture did not produce {report}")
    return report


def run(args: argparse.Namespace) -> Dict[str, Any]:
    profile = xray.load_json(args.profile)
    call_ir = xray.load_json(args.call_ir)
    available = _call_checkpoints(call_ir)
    requested = [name for name in profile["checkpoint_order"] if name in available]
    if not requested:
        raise RuntimeError("profile sparse checkpoints do not exist in call IR")
    observed_storage = profile["observed_storage"]
    for checkpoint_id in requested:
        checkpoint = available[checkpoint_id]
        oracle_storage = observed_storage["checkpoints"].get(checkpoint_id, observed_storage["default"])
        if checkpoint["storage_dtype"] != oracle_storage:
            divergence = {
                "checkpoint_id": checkpoint_id,
                "status": "fail",
                "classification": "STORAGE_CONTRACT_MISMATCH",
                "field": "storage_dtype",
                "subject": checkpoint["storage_dtype"],
                "oracle": oracle_storage,
                "recommended_action": xray.REMEDIATIONS["STORAGE_CONTRACT_MISMATCH"],
            }
            report = {
                "schema": "cke.xray_numerical_report", "schema_version": 1,
                "subject_backend": "ck", "oracle_backend": profile["backend"], "status": "fail",
                "comparisons": [divergence], "first_divergence": divergence,
                "last_passing_checkpoint": None, "unresolved_contract_checkpoints": [],
                "ranking_divergence": None,
                "next_plan": {"status": "attributed_interval", "first_failure": checkpoint_id, "next_checkpoints": []},
            }
            result = {"schema": "cke.xray_orchestration_report", "schema_version": 1,
                      "status": "fail", "rounds": [], "preflight": report, "final_report": report}
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir / "xray_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return result
    rounds = []
    lower_bound = None
    final_report = None
    for round_index in range(args.max_rounds):
        active = _bounded_order(requested, available, lower_bound)
        if not active:
            break
        selectors = [_selector(available[name]) for name in active]
        round_dir = args.output_dir / f"round_{round_index:02d}"
        tensor_report_path = _run_capture(args, selectors, round_dir)
        tensor_report = xray.load_json(tensor_report_path)
        ck_manifest = manifest_builder.build_manifest(
            backend="ck", call_ir=call_ir, tensor_report=tensor_report,
            model=args.model, source=str(args.runtime_dir), phase="prefill", requested=set(active),
        )
        torch_manifest = manifest_builder.build_manifest(
            backend="pytorch", call_ir=call_ir, tensor_report=tensor_report,
            model=args.model, source=str(args.checkpoint), phase="prefill",
            storage_dtype_override=observed_storage["default"], requested=set(active),
        )
        _apply_observed_storage(torch_manifest, observed_storage)
        ck_path = round_dir / "ck.checkpoints.json"
        torch_path = round_dir / "pytorch.checkpoints.json"
        ck_path.write_text(json.dumps(ck_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        torch_path.write_text(json.dumps(torch_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        report = xray.compare_manifests(
            ck_manifest, torch_manifest, profile, checkpoint_order=active,
        )
        if report["status"] == "pass" and args.ranking_report:
            report = xray.compare_manifests(
                ck_manifest, torch_manifest, profile,
                ranking_report=xray.load_json(args.ranking_report), checkpoint_order=active,
            )
        report_path = round_dir / "xray_report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        rounds.append({"round": round_index, "checkpoints": active, "selectors": selectors, "report": str(report_path), "status": report["status"]})
        final_report = report
        divergence = report.get("first_divergence") or {}
        classification = divergence.get("classification")
        next_plan = report.get("next_plan") or {}
        next_requested = list(next_plan.get("next_checkpoints") or [])
        if report["status"] == "pass" or not next_requested:
            break
        if classification not in {
            "KERNEL_IMPLEMENTATION_DIVERGENCE",
            "NONFINITE_OUTPUT",
            "RANKING_DIVERGENCE",
        }:
            break
        lower_bound = next_plan.get("passing_lower_bound")
        requested = next_requested
    result = {
        "schema": "cke.xray_orchestration_report", "schema_version": 1,
        "status": final_report["status"] if final_report else "error",
        "rounds": rounds, "preflight": None, "final_report": final_report,
        "oracle_runtime": {
            "python": str(args.oracle_python),
            "pythonpath": [str(path) for path in args.oracle_pythonpath],
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "xray_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Qwen3-VL safetensors checkpoint directory")
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--weights-bump", type=Path, required=True)
    parser.add_argument("--call-ir", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--torch-prefix", type=Path)
    parser.add_argument("--ck-import-layer-input", type=Path)
    parser.add_argument("--ck-import-layer", type=int)
    parser.add_argument("--ck-import-checkpoint", choices=("layer_input", "after_attn"), default="layer_input")
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--capture-script", type=Path, default=DEFAULT_CAPTURE_SCRIPT)
    parser.add_argument(
        "--oracle-python",
        default=sys.executable,
        help="Python executable for the isolated PyTorch capture process",
    )
    parser.add_argument(
        "--oracle-pythonpath",
        type=Path,
        action="append",
        default=[],
        help="Prepend a source checkout or dependency directory to the oracle PYTHONPATH",
    )
    parser.add_argument("--model", default="qwen3vl")
    parser.add_argument("--architecture", choices=("qwen3vl", "cohere_compass"), default="qwen3vl")
    parser.add_argument("--model-so-name", default="libqwen3vl_bf16_encoder_v8.so")
    parser.add_argument("--ranking-report", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("build/xray/qwen3vl_bf16"))
    parser.add_argument("--threads", type=int, default=int(os.environ.get("CK_NUM_THREADS", "20")))
    parser.add_argument("--attn-implementation", choices=("auto", "eager", "sdpa"), default="eager")
    parser.add_argument("--max-rounds", type=int, default=3)
    args = parser.parse_args(argv)
    if (args.ck_import_layer_input is None) != (args.ck_import_layer is None):
        parser.error("--ck-import-layer-input and --ck-import-layer must be provided together")
    result = run(args)
    divergence = (result.get("final_report") or {}).get("first_divergence") or {}
    print(f"status={result['status']} rounds={len(result['rounds'])}")
    if divergence:
        print(f"fail_at={divergence.get('checkpoint_id', divergence.get('position'))}")
        print(f"class={divergence.get('classification')}")
    print(f"report={args.output_dir / 'xray_summary.json'}")
    return 1 if result["status"] != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
