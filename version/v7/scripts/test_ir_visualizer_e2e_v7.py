#!/usr/bin/env python3
"""
v7 IR visualizer E2E regression gate.

Validates runbook-style flow:
1) ck_run_v7.py run ... --generate-visualizer
2) make profile-v7-decode V7_MODEL=<resolved_run_dir>
3) regenerate ir_report.html from explicit --run path
4) verify embedded report metadata/artifacts are coherent

Optional full mode:
5) compile a tiny CK train runtime fixture (dummy weights + libtrain.so)
6) replay ASan command block from the visualizer Profile tab
7) verify train-only profile artifacts are loaded from that fixture run-dir
"""

from __future__ import annotations

import argparse
import json
import shutil
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
CK_RUN = ROOT / "version" / "v7" / "scripts" / "ck_run_v7.py"
RESOLVE_MODEL_DIR = ROOT / "version" / "v7" / "scripts" / "resolve_model_dir_v7.py"


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


SUPPORTED_ARTIFACT_FILES: dict[str, tuple[str, ...]] = {
    "ir1_decode": ("ir1_decode.json",),
    "ir1_prefill": ("ir1_prefill.json",),
    "layout_decode": ("layout_decode.json",),
    "layout_prefill": ("layout_prefill.json",),
    "lowered_decode_call": ("lowered_decode_call.json", "lowered_decode.json"),
    "lowered_prefill_call": ("lowered_prefill_call.json", "lowered_prefill.json"),
    "ir1_train": ("ir1_train_forward.json", "ir1_train.json"),
    "ir2_train": ("ir2_train_backward.json", "ir2_train.json"),
    "ir_train_invariants": ("ir_train_invariants.json",),
    "ir2_train_summary": ("ir2_train_summary.json",),
    "layout_train": ("layout_train.json", "layout_train_latest.json"),
    "layout_train_audit": ("layout_train_audit.json", "layout_train_audit_latest.json"),
    "train_exec_plan": ("train_exec_plan.json", "train_exec_plan_latest.json"),
    "memory_diagnostic": ("memory_diagnostic_latest.json", "memory_diagnostic.json"),
    "memory_verification": ("memory_verification_latest.json", "memory_verification.json"),
    "generated_train_runtime_summary": ("generated_train_runtime_summary_v7.json", "generated_train_runtime_summary.json"),
    "training_loss_curve": ("training_loss_curve.json", "training_loss_curve_latest.json"),
    "training_grad_norms": ("training_grad_norms.json", "training_grad_norms_latest.json"),
    "training_parity": ("training_parity.json", "training_parity_latest.json"),
    "training_step_profile": ("training_step_profile.json", "training_step_profile_latest.json"),
    "training_checkpoint_policy": ("training_checkpoint_policy.json", "training_checkpoint_policy_latest.json"),
    "training_pipeline": ("training_pipeline.json", "training_pipeline_latest.json"),
    "dataset_qc": ("dataset_qc.json",),
    "dataset_profile": ("dataset_profile.json",),
    "tokenizer_roundtrip": ("tokenizer_roundtrip.json",),
    "post_train_eval": ("post_train_eval.json",),
    "training_epoch_sweep": ("training_epoch_sweep.json", "training_epoch_sweep_latest.json"),
    "train_e2e": ("train_e2e.json", "train_e2e_latest.json"),
    "run_config": ("config.json",),
    "sanity_overfit": ("sanity_overfit.json",),
    "parity_report": ("parity_report.json",),
    "profile_latest": ("profile_latest.json",),
    "contract_report": ("contract_report_latest.json",),
    "parity_1token": ("parity_1token_latest.json",),
    "qk_norm_backward_parity": ("qk_norm_backward_parity_latest.json",),
    "fd_gradients": ("fd_gradients_latest.json",),
    "train_parity_epochs_3": ("train_parity_epochs_3_latest.json",),
    "train_parity_epochs_5": ("train_parity_epochs_5_latest.json",),
    "train_runtime_parity_realistic": ("train_runtime_parity_realistic_latest.json",),
    "train_runtime_parity_stress": ("train_runtime_parity_stress_latest.json",),
    "replay_determinism": ("replay_determinism_latest.json",),
    "backprop_stitch_runtime": ("backprop_stitch_runtime_latest.json", "backprop_stitch_runtime.json"),
    "manifest": ("weights_manifest.json",),
    "profile_summary": ("profile_summary.json",),
    "perf_stat_summary": ("perf_stat_summary.json",),
    "flamegraph_manifest": ("flamegraph_manifest.json",),
    "cachegrind_summary": ("cachegrind_summary.json",),
    "asan_summary": ("asan_summary.json",),
    "vtune_summary": ("vtune_summary.json",),
    "advisor_summary": ("advisor_summary.json",),
    "memory_signoff": ("memory_signoff.json",),
    "perf_gate_report": ("perf_gate_report.json",),
    "regression_ledger": ("regression_ledger.json", "REGRESSION_LEDGER.json"),
}


def _run(cmd: list[str], cwd: Path = ROOT) -> None:
    print("[run]", " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _capture(cmd: list[str], cwd: Path = ROOT) -> str:
    print("[run]", " ".join(shlex.quote(c) for c in cmd))
    out = subprocess.check_output(cmd, cwd=str(cwd), text=True)
    return out.strip()


def _extract_embedded_data(report_html: Path) -> dict[str, Any]:
    html = report_html.read_text(encoding="utf-8", errors="replace")
    prefix = "window.EMBEDDED_IR_DATA = "
    suffix = ";window.dispatchEvent(new Event('ckEmbeddedDataLoaded'));"
    start = html.find(prefix)
    if start < 0:
        raise RuntimeError(f"Embedded data prefix not found in {report_html}")
    start += len(prefix)
    end = html.find(suffix, start)
    if end < 0:
        raise RuntimeError(f"Embedded data suffix not found in {report_html}")
    payload = html[start:end]
    return json.loads(payload)


def _profile_csv_exists(run_dir: Path) -> bool:
    return (run_dir / "profile_decode.csv").exists() or (run_dir / ".ck_build" / "profile_decode.csv").exists()


def _train_runtime_exists(run_dir: Path) -> bool:
    return (run_dir / "libtrain.so").exists() or (run_dir / ".ck_build" / "libtrain.so").exists()


def _is_renderable_profile_summary(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    if payload.get("embedded") is True:
        return False
    return "by_op" in payload and "entries" in payload


def _record(checks: list[CheckResult], name: str, passed: bool, detail: str) -> None:
    checks.append(CheckResult(name=name, passed=passed, detail=detail))
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}: {detail}")


def _check_decode_core_files(report_data: dict[str, Any]) -> list[CheckResult]:
    checks: list[CheckResult] = []
    loaded_paths = report_data.get("meta", {}).get("loaded_paths", {})
    for key in ("ir1_decode", "layout_decode", "lowered_decode_call"):
        _record(
            checks,
            f"decode_artifact_{key}",
            key in loaded_paths,
            f"loaded_paths has {key}={key in loaded_paths}",
        )
    return checks


def _artifact_roots_from_report(report_data: dict[str, Any], run_dir: Path) -> list[Path]:
    roots: list[Path] = [run_dir, run_dir / "ck_build", run_dir / ".ck_build"]
    meta = report_data.get("meta", {})
    for path_str in meta.get("profile_roots", []) or []:
        if not path_str:
            continue
        p = Path(path_str)
        roots.extend([p, p / "ck_build", p / ".ck_build"])

    out: list[Path] = []
    seen: set[str] = set()
    for r in roots:
        try:
            key = str(r.resolve())
        except Exception:
            key = str(r)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _path_under_roots(path: Path, roots: list[Path]) -> bool:
    try:
        resolved = path.resolve()
    except Exception:
        resolved = path
    for root in roots:
        try:
            base = root.resolve()
        except Exception:
            base = root
        if resolved == base or base in resolved.parents:
            return True
    return False


def _check_strict_run_scoping(report_data: dict[str, Any], run_dir: Path) -> list[CheckResult]:
    checks: list[CheckResult] = []
    meta = report_data.get("meta", {}) or {}
    loaded_paths = meta.get("loaded_paths", {}) or {}
    roots = _artifact_roots_from_report(report_data, run_dir)
    strict_flag = bool(meta.get("strict_run_artifacts"))
    _record(
        checks,
        "strict_run_artifacts_enabled",
        strict_flag,
        f"strict_run_artifacts={meta.get('strict_run_artifacts')}",
    )

    # These are intentionally repo/global-scoped inputs rather than per-run artifacts.
    exempt = {"grad_rules", "kernel_registry", "regression_ledger"}
    off_root: list[str] = []
    for key, raw in loaded_paths.items():
        if key in exempt:
            continue
        p = Path(str(raw))
        if not _path_under_roots(p, roots):
            off_root.append(f"{key}:{p}")

    _record(
        checks,
        "strict_run_scoped_loaded_paths",
        not off_root,
        f"off_root={off_root[:8]}",
    )
    return checks


def _check_artifact_load_coverage(report_data: dict[str, Any], run_dir: Path) -> list[CheckResult]:
    checks: list[CheckResult] = []
    roots = _artifact_roots_from_report(report_data, run_dir)
    loaded_paths = report_data.get("meta", {}).get("loaded_paths", {}) or {}

    present_keys: dict[str, list[Path]] = {}
    for key, names in SUPPORTED_ARTIFACT_FILES.items():
        matches: list[Path] = []
        for root in roots:
            for name in names:
                p = root / name
                if p.exists():
                    matches.append(p.resolve())
        if matches:
            dedup: list[Path] = []
            seen: set[str] = set()
            for m in matches:
                mk = str(m)
                if mk in seen:
                    continue
                seen.add(mk)
                dedup.append(m)
            present_keys[key] = dedup

    missing_keys: list[str] = []
    mismatched_keys: list[str] = []
    for key, present_paths in sorted(present_keys.items()):
        loaded_raw = loaded_paths.get(key)
        if not loaded_raw:
            missing_keys.append(key)
            continue
        loaded_path = Path(str(loaded_raw))
        if not loaded_path.exists():
            mismatched_keys.append(f"{key}:loaded_path_missing")
            continue
        loaded_resolved = loaded_path.resolve()
        if loaded_resolved not in set(present_paths):
            mismatched_keys.append(f"{key}:loaded_from_unexpected_path")

    _record(
        checks,
        "artifact_load_coverage",
        not missing_keys and not mismatched_keys,
        (
            f"present_keys={len(present_keys)} "
            f"missing={missing_keys[:8]} "
            f"mismatched={mismatched_keys[:8]}"
        ),
    )
    return checks


def _check_vtune_path_resolution(report_data: dict[str, Any]) -> list[CheckResult]:
    checks: list[CheckResult] = []
    files = report_data.get("files", {})
    vtune = files.get("vtune_summary")
    if not isinstance(vtune, dict):
        _record(
            checks,
            "vtune_path_resolution",
            True,
            "vtune_summary not loaded; check skipped",
        )
        return checks

    def _is_relative(p: Any) -> bool:
        if not isinstance(p, str):
            return False
        if p.startswith(("/", "http://", "https://", "file://")):
            return False
        if len(p) >= 3 and p[1] == ":" and p[2] in ("\\", "/"):
            return False
        return True

    unresolved: list[str] = []
    for key in ("result_dir", "report_path", "csv_path"):
        raw = vtune.get(key)
        if _is_relative(raw):
            resolved = vtune.get(f"{key}_resolved")
            if not isinstance(resolved, str) or not resolved.startswith("/"):
                unresolved.append(key)

    analyses = vtune.get("analyses")
    if isinstance(analyses, list):
        for idx, entry in enumerate(analyses):
            if not isinstance(entry, dict):
                continue
            for key in ("result_dir", "report_text", "report_csv"):
                raw = entry.get(key)
                if _is_relative(raw):
                    resolved = entry.get(f"{key}_resolved")
                    if not isinstance(resolved, str) or not resolved.startswith("/"):
                        unresolved.append(f"analyses[{idx}].{key}")

    _record(
        checks,
        "vtune_path_resolution",
        not unresolved,
        f"unresolved={unresolved[:8]}",
    )
    return checks


def _run_operator_profile_compile_check_cmd(run_dir: Path) -> None:
    """
    Replay the operator-facing profile compile-check block from ir_visualizer.html.
    This is intentionally kept aligned with profileCompileCheckCmd in the UI.
    """
    report_model_dir = shlex.quote(str(run_dir))
    script = f"""
REPORT_MODEL_DIR={report_model_dir}
echo "report_model_dir=$REPORT_MODEL_DIR"
PROFILE_C_PATH="$REPORT_MODEL_DIR/model_v7.c"
if [ -f "$REPORT_MODEL_DIR/.ck_build/model_v7.c" ]; then PROFILE_C_PATH="$REPORT_MODEL_DIR/.ck_build/model_v7.c"; fi
echo "profile_c_path=$PROFILE_C_PATH"
rg -n "CK_PROFILE" "$PROFILE_C_PATH" || true
# If output is empty, generated model_v7.c has zero CK_PROFILE sections and decode CSV cannot be emitted.
make profile-v7-decode V7_MODEL="$REPORT_MODEL_DIR" V7_PERF_RUNTIME=cli V7_PREP_WITH_PYTHON=1 V7_FORCE_COMPILE=1
python3 version/v7/tools/open_ir_visualizer.py --generate --run "$REPORT_MODEL_DIR" --html-only --output "$REPORT_MODEL_DIR/ir_report.html"
"""
    _run(["bash", "-lc", script], cwd=ROOT)


def _ensure_train_runtime(run_dir: Path, python_exec: str) -> None:
    """
    Build/refresh libtrain.so in the same run-dir with a tiny CK-only train pass.
    This mirrors operator expectations for train-only profile artifacts.
    """
    json_out = run_dir / "train_e2e_latest.json"
    cmd = [
        python_exec,
        str(CK_RUN),
        "train",
        "--run",
        str(run_dir),
        "--backend",
        "ck",
        "--prompt",
        "hello",
        "--train-epochs",
        "1",
        "--train-seq-len",
        "8",
        "--train-total-tokens",
        "64",
        "--train-grad-accum",
        "8",
        "--analysis-checkpoints",
        "off",
        "--no-train-save-final",
        "--train-json-out",
        str(json_out),
    ]
    _run(cmd)


def _prepare_tiny_train_fixture(run_dir: Path, python_exec: str) -> None:
    """
    Build a tiny train-capable run-dir so train-only profile artifacts
    are validated without depending on large inference checkpoints.
    """
    if run_dir.exists():
        shutil.rmtree(run_dir)
    init_cmd = [
        python_exec,
        str(CK_RUN),
        "init",
        "--run",
        str(run_dir),
        "--layers",
        "2",
        "--vocab-size",
        "256",
        "--embed-dim",
        "64",
        "--hidden-dim",
        "128",
        "--num-heads",
        "4",
        "--num-kv-heads",
        "2",
        "--context-len",
        "128",
        "--train-seed",
        "7",
    ]
    _run(init_cmd)
    _ensure_train_runtime(run_dir, python_exec)


def _run_operator_asan_capture_cmd(run_dir: Path) -> None:
    """
    Replay ASan command block from ir_visualizer Profile tab using explicit run-dir.
    """
    report_model_dir = shlex.quote(str(run_dir))
    report_run_dir = shlex.quote(str(run_dir))
    report_html = shlex.quote(str(run_dir / "ir_report.html"))
    token_path = run_dir / ".ck_profile_tokens_viz.txt"
    token_literal = repr(str(token_path))

    script = f"""
REPORT_MODEL_DIR={report_model_dir}
echo "report_model_dir=$REPORT_MODEL_DIR"
if [ ! -f "$REPORT_MODEL_DIR/libtrain.so" ] && [ ! -f "$REPORT_MODEL_DIR/.ck_build/libtrain.so" ]; then
  echo "WARN: train runtime is missing (no libtrain.so in run dir).";
  echo "WARN: skipping ASan capture because this flow requires training runtime artifacts.";
else
  python3 - <<'PY'
from pathlib import Path
p = Path({token_literal})
p.write_text(' '.join(str(i % 1024) for i in range(8192)))
print(p)
PY
  make --no-print-directory ck-cli-v7
  CK_NUM_THREADS=8 ./build/ck-cli-v7 profile --run {report_run_dir} --tool asan --train-token-file {shlex.quote(str(token_path))} --train-epochs 1 --train-seq-len 8 --train-total-tokens 2048 --train-grad-accum 8 --threads 8
  python3 version/v7/tools/open_ir_visualizer.py --generate --run {report_run_dir} --html-only --output {report_html}
fi
"""
    _run(["bash", "-lc", script], cwd=ROOT)


def main() -> int:
    ap = argparse.ArgumentParser(description="v7 IR visualizer E2E regression")
    ap.add_argument("--model-input", required=True, help="Model input accepted by ck_run_v7.py run")
    ap.add_argument("--context-len", type=int, default=1024, help="Context length for ck_run_v7.py run")
    ap.add_argument("--prompt", default="hi", help="Smoke prompt")
    ap.add_argument("--max-tokens", type=int, default=1, help="Smoke max tokens")
    ap.add_argument("--force-compile", action="store_true", help="Pass --force-compile to ck_run_v7.py run")
    ap.add_argument("--force-convert", action="store_true", help="Pass --force-convert to ck_run_v7.py run")
    ap.add_argument("--skip-profile", action="store_true", help="Skip decode profile generation check")
    ap.add_argument(
        "--with-train-runtime",
        action="store_true",
        help="Also exercise training-runtime Profile tab flow (tiny fixture + libtrain + ASan summary)",
    )
    ap.add_argument(
        "--train-fixture-run-dir",
        default="/tmp/v7_ir_visualizer_train_fixture",
        help="Run-dir used for optional tiny training-runtime fixture",
    )
    ap.add_argument("--json-out", default=None, help="Optional JSON report path")
    args = ap.parse_args()

    py = sys.executable
    checks: list[CheckResult] = []

    run_cmd = [
        py,
        str(CK_RUN),
        "run",
        args.model_input,
        "--context-len",
        str(args.context_len),
        "--chat-template",
        "none",
        "--generate-only",
        "--prompt",
        args.prompt,
        "--max-tokens",
        str(args.max_tokens),
        "--generate-visualizer",
    ]
    if args.force_compile:
        run_cmd.append("--force-compile")
    if args.force_convert:
        run_cmd.append("--force-convert")
    _run(run_cmd)

    resolved_model_dir = Path(
        _capture([py, str(RESOLVE_MODEL_DIR), "--model-input", args.model_input])
    ).resolve()
    _record(checks, "resolved_model_dir_exists", resolved_model_dir.exists(), str(resolved_model_dir))

    report_path = resolved_model_dir / "ir_report.html"
    _record(checks, "initial_report_exists", report_path.exists(), str(report_path))
    if not report_path.exists():
        return 1

    initial_data = _extract_embedded_data(report_path)
    initial_meta = initial_data.get("meta", {})
    _record(
        checks,
        "embedded_run_dir_matches",
        initial_meta.get("run_dir") == str(resolved_model_dir),
        f"embedded={initial_meta.get('run_dir')} expected={resolved_model_dir}",
    )
    checks.extend(_check_decode_core_files(initial_data))
    checks.extend(_check_strict_run_scoping(initial_data, resolved_model_dir))

    if not args.skip_profile:
        _run_operator_profile_compile_check_cmd(resolved_model_dir)
        _record(
            checks,
            "operator_profile_compile_check_cmd_replay",
            True,
            "replayed profileCompileCheckCmd from ir_visualizer.html with explicit REPORT_MODEL_DIR",
        )
        _record(
            checks,
            "decode_profile_csv_exists",
            _profile_csv_exists(resolved_model_dir),
            f"csv under {resolved_model_dir} or {resolved_model_dir / '.ck_build'}",
        )

        refreshed = _extract_embedded_data(report_path)
        refreshed_paths = refreshed.get("meta", {}).get("loaded_paths", {})
        _record(
            checks,
            "profile_summary_loaded_in_report",
            "profile_summary" in refreshed_paths,
            f"profile_summary path={refreshed_paths.get('profile_summary')}",
        )
        _record(
            checks,
            "regression_ledger_loaded_in_report",
            "regression_ledger" in refreshed_paths,
            f"regression_ledger path={refreshed_paths.get('regression_ledger')}",
        )
        profile_summary = refreshed.get("files", {}).get("profile_summary")
        _record(
            checks,
            "profile_summary_renderable_payload",
            _is_renderable_profile_summary(profile_summary),
            "profile_summary has by_op+entries and is not embedded marker stub",
        )
        checks.extend(_check_vtune_path_resolution(refreshed))
        checks.extend(_check_strict_run_scoping(refreshed, resolved_model_dir))
        checks.extend(_check_artifact_load_coverage(refreshed, resolved_model_dir))

        if args.with_train_runtime:
            fixture_run_dir = Path(args.train_fixture_run_dir).expanduser().resolve()
            _prepare_tiny_train_fixture(fixture_run_dir, py)
            _record(checks, "train_fixture_dir_exists", fixture_run_dir.exists(), str(fixture_run_dir))
            _record(
                checks,
                "train_fixture_libtrain_exists",
                _train_runtime_exists(fixture_run_dir),
                f"libtrain under {fixture_run_dir} or {fixture_run_dir / '.ck_build'}",
            )
            if _train_runtime_exists(fixture_run_dir):
                _run_operator_asan_capture_cmd(fixture_run_dir)
                _record(
                    checks,
                    "operator_asan_cmd_replay",
                    True,
                    "replayed ASan profile command block with explicit REPORT_MODEL_DIR",
                )

            fixture_report = fixture_run_dir / "ir_report.html"
            _record(checks, "train_fixture_report_exists", fixture_report.exists(), str(fixture_report))
            if fixture_report.exists():
                fixture_data = _extract_embedded_data(fixture_report)
                fixture_meta = fixture_data.get("meta", {})
                _record(
                    checks,
                    "train_fixture_embedded_run_dir_matches",
                    fixture_meta.get("run_dir") == str(fixture_run_dir),
                    f"embedded={fixture_meta.get('run_dir')} expected={fixture_run_dir}",
                )
                fixture_paths = fixture_meta.get("loaded_paths", {})
                _record(
                    checks,
                    "train_fixture_asan_summary_loaded",
                    "asan_summary" in fixture_paths,
                    f"asan_summary path={fixture_paths.get('asan_summary')}",
                )
                _record(
                    checks,
                    "train_fixture_asan_summary_payload_present",
                    isinstance(fixture_data.get("files", {}).get("asan_summary"), dict),
                    "asan_summary payload is a JSON object",
                )
                checks.extend(_check_strict_run_scoping(fixture_data, fixture_run_dir))
                checks.extend(_check_artifact_load_coverage(fixture_data, fixture_run_dir))

    ok = all(c.passed for c in checks)
    report = {
        "format": "v7-ir-visualizer-e2e",
        "ok": ok,
        "model_input": args.model_input,
        "resolved_model_dir": str(resolved_model_dir),
        "report_path": str(report_path),
        "checks": [asdict(c) for c in checks],
    }

    if args.json_out:
        out_path = Path(args.json_out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[out] {out_path}")

    if ok:
        print("PASS: v7 IR visualizer E2E")
        return 0
    print("FAIL: v7 IR visualizer E2E")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
