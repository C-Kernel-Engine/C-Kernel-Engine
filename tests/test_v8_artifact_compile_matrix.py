from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "version" / "v8" / "scripts"
FIXTURES = ROOT / "tests" / "fixtures" / "v8" / "artifact_manifests"


CASES = (
    {
        "id": "laguna-xs-2.1-q4_k_m",
        "fixture": "laguna-xs-2.1-q4_k_m.json",
        "model": "laguna",
        "source_arch": "laguna",
        "entry_count": 684,
        "weight_formats": {"q4_k", "q6_k"},
        "layer_kinds": {
            "dense_global_attention",
            "moe_global_attention",
            "moe_sliding_attention",
        },
    },
    {
        "id": "nemotron-nano-9b-v2-q4_k_m",
        "fixture": "nemotron-nano-9b-v2-q4_k_m.json",
        "model": "nemotron_h",
        "source_arch": "nemotron_h",
        "entry_count": 347,
        "weight_formats": {"q4_k", "q5_0", "q8_0"},
        "layer_kinds": {"attention", "mamba", "mlp"},
    },
)


def _load_ck_run_v8():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    path = SCRIPTS / "ck_run_v8.py"
    spec = importlib.util.spec_from_file_location("artifact_matrix_ck_run_v8", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CK_RUN_V8 = _load_ck_run_v8()


def _run(command: list[str]) -> None:
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_real_manifest_compiles_through_both_execution_phases(
    case: dict[str, object], tmp_path: Path
) -> None:
    source = FIXTURES / str(case["fixture"])
    manifest = json.loads(source.read_text(encoding="utf-8"))
    config = manifest["config"]
    assert manifest["model"] == case["model"]
    assert manifest["source_arch"] == case["source_arch"]
    assert len(manifest["entries"]) == case["entry_count"]
    assert case["weight_formats"] <= {
        str(entry.get("dtype")) for entry in manifest["entries"]
    }
    assert case["layer_kinds"] <= set(config["layer_kinds"])

    work = tmp_path / str(case["id"])
    work.mkdir()
    manifest_path = work / "weights_manifest.json"
    shutil.copy2(source, manifest_path)

    # Cached manifests embed the circuit used during conversion. Exercise the
    # production refresh path so current graph policy, not stale policy, is
    # what this compile gate certifies.
    mutable = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutable["template"] = {"name": case["model"], "version": 0}
    manifest_path.write_text(json.dumps(mutable), encoding="utf-8")
    assert CK_RUN_V8._refresh_manifest_circuit_snapshot(manifest_path)

    outputs: dict[str, Path] = {}
    for mode in ("prefill", "decode"):
        mode_dir = work / mode
        mode_dir.mkdir()
        args = [
            sys.executable,
            str(SCRIPTS / "build_ir_v8.py"),
            "--manifest",
            str(manifest_path),
            "--mode",
            mode,
            "--context-len",
            "32",
            "--output",
            str(mode_dir / "ir.json"),
            "--layout-output",
            str(mode_dir / "layout.json"),
            "--lowered-output",
            str(mode_dir / "lowered.json"),
            "--call-output",
            str(mode_dir / "call.json"),
            "--init-output",
            str(mode_dir / "init.json"),
        ]
        if mode == "prefill":
            args.extend(("--prefill-chunk-len", "32"))
        _run(args)
        outputs[f"{mode}_call"] = mode_dir / "call.json"
        outputs[f"{mode}_layout"] = mode_dir / "layout.json"
        if mode == "decode":
            outputs["init_call"] = mode_dir / "init_call.json"

    generated = work / "model_v8.c"
    _run(
        [
            sys.executable,
            str(SCRIPTS / "codegen_v8.py"),
            "--ir",
            str(outputs["decode_call"]),
            "--layout",
            str(outputs["decode_layout"]),
            "--prefill",
            str(outputs["prefill_call"]),
            "--prefill-layout",
            str(outputs["prefill_layout"]),
            "--init",
            str(outputs["init_call"]),
            "--output",
            str(generated),
            "--strict-contracts",
        ]
    )
    _run(
        [
            "cc",
            "-std=c11",
            "-fopenmp",
            "-Iinclude",
            "-Iversion/v8/src",
            "-fsyntax-only",
            str(generated),
        ]
    )
