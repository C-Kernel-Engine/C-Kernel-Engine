#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TARGET = ROOT / "version" / "v7" / "scripts" / "test_visualizer_generated_e2e_v7.py"

XRAY_FIXTURES = ROOT / "version" / "v8" / "tests" / "fixtures" / "xray"
XRAY_EXPECTED_KEYS = [
    "xray_whisper_encoder",
    "xray_ranking",
    "xray_execution_trace",
    "xray_execution_state",
    "xray_decoder_pytorch",
    "xray_qwen3vl_pytorch",
    "xray_qwen3vl_llamacpp",
    "xray_monotonic",
    "lowered_decode_call",
]
XRAY_EXPECTED_SCHEMAS = [
    "cke.whisper_encoder_pytorch_xray",
    "cke.xray_ranking_report",
    "cke.xray_execution_trace",
    "cke.xray_execution_state_report",
    "cke.xray.decoder_pytorch",
    "cke.xray_orchestration_report",
    "cke.xray_monotonic_provider_gate",
]
XRAY_RUNBOOK_MARKERS = [
    "No X-ray artifacts loaded",
    "compare_whisper_encoder_pytorch_v8.py",
    "xray_execution_state_v8.py",
    "xray_numerical_parity_v8.py",
    "test-bf16-xray",
]
XRAY_VIEW_MARKERS = [
    "Circuit X-Ray",
    "X-Ray Operator Runbook",
    "backend selection works",
]


def _generate(open_viz: Path, run_dir: Path, out: Path) -> str | None:
    cmd = [
        sys.executable, str(open_viz),
        "--generate", "--run", str(run_dir),
        "--html-only", "--output", str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0 or not out.exists():
        print(proc.stdout[-3000:])
        print(proc.stderr[-3000:])
        return None
    return out.read_text(encoding="utf-8")


def run_xray_stage() -> int:
    """v8-only stage: fixture X-ray artifacts must embed into the generated report,
    and the X-Ray tab must render its empty-state runbook without them."""
    open_viz = ROOT / "version" / "v8" / "tools" / "open_ir_visualizer_v8.py"
    failures: list[str] = []

    with tempfile.TemporaryDirectory(prefix="ck-viz-xray-") as tmp:
        # 1) Run dir populated with fixture X-ray artifacts -> keys embedded.
        run_dir = Path(tmp) / "run"
        run_dir.mkdir()
        for fixture in sorted(XRAY_FIXTURES.glob("*.json")):
            shutil.copy(fixture, run_dir / fixture.name)
        html = _generate(open_viz, run_dir, run_dir / "ir_report.html")
        if html is None:
            failures.append("generate_with_fixtures")
        else:
            for key in XRAY_EXPECTED_KEYS:
                if f'"{key}"' not in html:
                    failures.append(f"missing_key:{key}")
            for marker in XRAY_EXPECTED_SCHEMAS:
                if marker not in html:
                    failures.append(f"missing_schema:{marker}")
            for marker in XRAY_VIEW_MARKERS:
                if marker not in html:
                    failures.append(f"missing_view:{marker}")

        # 2) Empty run dir -> empty-state runbook commands present.
        empty_dir = Path(tmp) / "empty"
        empty_dir.mkdir()
        html_empty = _generate(open_viz, empty_dir, empty_dir / "ir_report.html")
        if html_empty is None:
            failures.append("generate_empty")
        else:
            for marker in XRAY_RUNBOOK_MARKERS:
                if marker not in html_empty:
                    failures.append(f"missing_runbook:{marker}")
            for marker in XRAY_VIEW_MARKERS:
                if marker not in html_empty:
                    failures.append(f"missing_view_empty:{marker}")

    if failures:
        for f in failures:
            print(f"  ✗ {f}")
        print(f"L3_xray_embed  max_diff={len(failures):.2e}  tol=1e+00  [FAIL]")
        return 1
    print(f"  ✓ xray embed: {len(XRAY_EXPECTED_KEYS)} keys + {len(XRAY_EXPECTED_SCHEMAS)} schemas embedded from fixtures")
    print(f"  ✓ xray empty-state runbook: {len(XRAY_RUNBOOK_MARKERS)} markers present")
    print("L3_xray_embed  max_diff=0.00e+00  tol=1e+00  [PASS]")
    return 0


EXPLAIN_FIXTURES = ROOT / "version" / "v8" / "tests" / "fixtures" / "explain"
EXPLAIN_VIEW_MARKERS = [
    "Explain This Operation",
    "explainOpSelect",
    "explainProvenance",
    "explainReports",
    "kernel-maps.html#resolution",
    "codegen.html",
]
EXPLAIN_SUCCESS_MARKERS = [
    '"explain_reports"',
    '"explain_provenance"',
    "rmsnorm_forward_llama_production",
    "rmsnorm_llama_cpu_production_fp32_output",
    "rmsnorm_forward_llama_production.json",
]
EXPLAIN_FAILURE_MARKERS = [
    "HARD KERNEL RESOLUTION FAULT",
    "status_not_production:observed",
    "weight_dtype_mismatch",
    "HARD CIRCUIT DATAFLOW FAULT",
    "unmatched=['x']",
    "not_generated",
    "reconstructed",
    "369032982",  # Laguna bring-up commit (PR #404) provenance
    "b9712c383",  # Nemotron #412-era provenance
    "tests/test_v8_laguna_contract.py",
    "tests/test_v8_nemotron_state_shape.py",
]


def _sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_explain_stage() -> int:
    """v8-only stage: explain fixtures embed, failure reports surface their
    rejection reasons, and bundle-stamp mismatch is disclosed as stale."""
    open_viz = ROOT / "version" / "v8" / "tools" / "open_ir_visualizer_v8.py"
    failures: list[str] = []

    with tempfile.TemporaryDirectory(prefix="ck-viz-explain-") as tmp:
        # 1) Run dir with the full fixture set: success chain + both failure reports.
        run_dir = Path(tmp) / "run"
        run_dir.mkdir()
        for fixture in sorted(EXPLAIN_FIXTURES.glob("*.json")):
            shutil.copy(fixture, run_dir / fixture.name)
        html = _generate(open_viz, run_dir, run_dir / "ir_report.html")
        if html is None:
            failures.append("generate_with_fixtures")
        else:
            for marker in EXPLAIN_VIEW_MARKERS + EXPLAIN_SUCCESS_MARKERS + EXPLAIN_FAILURE_MARKERS:
                if marker not in html:
                    failures.append(f"missing_marker:{marker}")

        # 2) Partial build: only ir1 -> later stage artifacts are not loaded,
        # and the provenance table says so instead of inventing a chain.
        partial_dir = Path(tmp) / "partial"
        partial_dir.mkdir()
        shutil.copy(EXPLAIN_FIXTURES / "ir1_decode.json", partial_dir / "ir1_decode.json")
        html_partial = _generate(open_viz, partial_dir, partial_dir / "ir_report.html")
        if html_partial is None:
            failures.append("generate_partial")
        else:
            if '"ir1_decode": {' not in html_partial:
                failures.append("partial_missing_ir1")
            if '"lowered_decode_call": {' in html_partial:
                failures.append("partial_filled_lowered_from_elsewhere")
            if '"status": "not_loaded"' not in html_partial:
                failures.append("partial_missing_not_loaded_provenance")

        # 3) Stale artifacts: a bundle stamp whose recorded sha256 disagrees
        # with the file on disk must be disclosed, never silently used.
        stale_dir = Path(tmp) / "stale"
        stale_dir.mkdir()
        for name in ("ir1_decode.json", "lowered_decode_call.json"):
            shutil.copy(EXPLAIN_FIXTURES / name, stale_dir / name)
        good = _sha256(stale_dir / "ir1_decode.json")
        bad = "0" * 64
        stamp = {
            "inputs": {"schema": "ck-v8-ir-bundle-v1"},
            "outputs": {
                "decode_ir": {"path": str(stale_dir / "ir1_decode.json"), "sha256": good, "size": 1},
                "decode_call": {"path": str(stale_dir / "lowered_decode_call.json"), "sha256": bad, "size": 1},
            },
        }
        (stale_dir / ".ck_ir_bundle.json").write_text(json.dumps(stamp), encoding="utf-8")
        html_stale = _generate(open_viz, stale_dir, stale_dir / "ir_report.html")
        if html_stale is None:
            failures.append("generate_stale")
        else:
            if '"status": "stale"' not in html_stale:
                failures.append("stale_not_disclosed")
            if '"status": "match"' not in html_stale:
                failures.append("match_not_recorded")

    if failures:
        for f in failures:
            print(f"  ✗ {f}")
        print(f"L3_explain_embed  max_diff={len(failures):.2e}  tol=1e+00  [FAIL]")
        return 1
    print(f"  ✓ explain embed: {len(EXPLAIN_SUCCESS_MARKERS)} chain + {len(EXPLAIN_FAILURE_MARKERS)} failure markers embedded from fixtures")
    print("  ✓ explain partial build: later stages not loaded, no cross-run fill")
    print("  ✓ explain staleness: bundle-stamp mismatch disclosed as stale")
    print("L3_explain_embed  max_diff=0.00e+00  tol=1e+00  [PASS]")
    return 0



if __name__ == "__main__":
    os.environ.setdefault("CK_VIS_VERSION", "v8")
    os.environ.setdefault("CK_VIS_MODELS_ROOT", str(Path.home() / ".cache" / "ck-engine-v8" / "models"))
    os.environ.setdefault("CK_VIS_HEALTH_SCRIPT", str(ROOT / "version" / "v8" / "scripts" / "test_visualizer_health_v8.py"))
    os.environ.setdefault("CK_VIS_OPEN_IR_VIZ", str(ROOT / "version" / "v8" / "tools" / "open_ir_visualizer_v8.py"))
    os.environ.setdefault("CK_VIS_PREPARE_VIEWER", str(ROOT / "version" / "v8" / "tools" / "prepare_run_viewer_v8.py"))
    os.environ.setdefault("CK_VIS_OPEN_IR_HUB", str(ROOT / "version" / "v8" / "tools" / "open_ir_hub_v8.py"))
    sys.argv[0] = str(Path(__file__).resolve())

    base_code = 0
    try:
        runpy.run_path(str(TARGET), run_name="__main__")
    except SystemExit as exc:
        if isinstance(exc.code, int):
            base_code = exc.code
        elif exc.code is not None:
            base_code = 1

    xray_code = run_xray_stage()
    explain_code = run_explain_stage()
    sys.exit(0 if (base_code == 0 and xray_code == 0 and explain_code == 0) else 1)
