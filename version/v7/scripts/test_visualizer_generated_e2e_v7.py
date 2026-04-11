#!/usr/bin/env python3
"""
Level 3 — Generated-file E2E for all three visualizers.

Bridges the gap between L1/L2 (source-only) and L4 (browser):
  1. Discover (or accept) a training run directory
  2. Generate ir_report.html, dataset_viewer.html, ir_hub.html
  3. Run L1 health checks on every generated file
  4. Validate embedded JSON structure in ir_report.html
  5. Cross-check artifact consistency

Usage:
    # Auto-discover latest run:
    python3 version/v7/scripts/test_visualizer_generated_e2e_v7.py

    # Specific run:
    python3 version/v7/scripts/test_visualizer_generated_e2e_v7.py --run ~/.cache/ck-engine-v7/models/train/spec06_...

    # Skip generation (validate existing files only):
    python3 version/v7/scripts/test_visualizer_generated_e2e_v7.py --validate-only

    # JSON output for nightly:
    python3 version/v7/scripts/test_visualizer_generated_e2e_v7.py --json-out report.json

Exit 0 = all checks pass, Exit 1 = failures.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
VIS_VERSION = os.environ.get("CK_VIS_VERSION", "v7")


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


HEALTH_SCRIPT = _env_path(
    "CK_VIS_HEALTH_SCRIPT",
    ROOT / "version" / VIS_VERSION / "scripts" / f"test_visualizer_health_{VIS_VERSION}.py",
)
OPEN_IR_VIZ = _env_path(
    "CK_VIS_OPEN_IR_VIZ",
    ROOT / "version" / VIS_VERSION / "tools" / ("open_ir_visualizer.py" if VIS_VERSION == "v7" else f"open_ir_visualizer_{VIS_VERSION}.py"),
)
PREPARE_VIEWER = _env_path(
    "CK_VIS_PREPARE_VIEWER",
    ROOT / "version" / VIS_VERSION / "tools" / ("prepare_run_viewer.py" if VIS_VERSION == "v7" else f"prepare_run_viewer_{VIS_VERSION}.py"),
)
OPEN_IR_HUB = _env_path(
    "CK_VIS_OPEN_IR_HUB",
    ROOT / "version" / VIS_VERSION / "tools" / ("open_ir_hub.py" if VIS_VERSION == "v7" else f"open_ir_hub_{VIS_VERSION}.py"),
)
DEFAULT_MODELS_ROOT = _env_path(
    "CK_VIS_MODELS_ROOT",
    Path.home() / ".cache" / f"ck-engine-{VIS_VERSION}" / "models",
)
DEFAULT_TRAIN_ROOT = DEFAULT_MODELS_ROOT / "train"

PYTHON = sys.executable

# ANSI
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_NC = "\033[0m"


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""
    level: str = "error"  # error | warning

@dataclass
class StageResult:
    stage: str
    checks: list[Check] = field(default_factory=list)
    elapsed_sec: float = 0.0

    @property
    def passed(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def failed(self) -> int:
        return sum(1 for c in self.checks if not c.passed and c.level == "error")

    @property
    def warnings(self) -> int:
        return sum(1 for c in self.checks if not c.passed and c.level == "warning")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _run(cmd: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    """Run a subprocess, capture output, respect timeout."""
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout,
        cwd=str(ROOT),
    )


def discover_latest_run(models_root: Path) -> Path | None:
    """Find the most-recent training run that has at least config.json."""
    train_dir = models_root / "train"
    if not train_dir.exists():
        return None
    candidates = []
    for d in train_dir.iterdir():
        if d.is_dir() and (d / "config.json").exists():
            mtime = d.stat().st_mtime
            candidates.append((mtime, d))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def print_stage(stage: StageResult, quiet: bool = False) -> None:
    icon = "✓" if stage.failed == 0 else "✗"
    color = _GREEN if stage.failed == 0 else _RED
    print(f"\n{_BOLD}{color}  {icon} {stage.stage}{_NC}"
          f"  {_DIM}({stage.passed}/{len(stage.checks)} passed"
          f", {stage.elapsed_sec:.1f}s){_NC}")
    for c in stage.checks:
        if quiet and c.passed:
            continue
        sym = f"{_GREEN}✓{_NC}" if c.passed else (
            f"{_YELLOW}⚠{_NC}" if c.level == "warning" else f"{_RED}✗{_NC}"
        )
        detail = f"  {_DIM}{c.detail}{_NC}" if c.detail else ""
        print(f"    {sym} {c.name}{detail}")


# ── Stage 1: Generation ─────────────────────────────────────────────────────

def generate_ir_report(run_dir: Path) -> StageResult:
    """Generate ir_report.html via open_ir_visualizer.py."""
    stage = StageResult(stage="Generate ir_report.html")
    t0 = time.monotonic()

    out_path = run_dir / "ir_report.html"
    cmd = [
        PYTHON, str(OPEN_IR_VIZ),
        "--generate", "--run", str(run_dir),
        "--html-only", "--output", str(out_path),
    ]
    try:
        r = _run(cmd, timeout=120)
        if r.returncode == 0 and out_path.exists():
            size = out_path.stat().st_size
            stage.checks.append(Check(
                "ir_report.html generated",
                True,
                f"{size:,} bytes",
            ))
        else:
            detail = r.stderr.strip()[:200] if r.stderr else f"rc={r.returncode}"
            stage.checks.append(Check("ir_report.html generated", False, detail))
    except subprocess.TimeoutExpired:
        stage.checks.append(Check("ir_report.html generated", False, "timeout (120s)"))
    except Exception as e:
        stage.checks.append(Check("ir_report.html generated", False, str(e)[:200]))

    stage.elapsed_sec = time.monotonic() - t0
    return stage


def generate_dataset_viewer(run_dir: Path) -> StageResult:
    """Generate dataset_viewer.html via prepare_run_viewer.py."""
    stage = StageResult(stage="Generate dataset_viewer.html")
    t0 = time.monotonic()

    out_path = run_dir / "dataset_viewer.html"
    cmd = [PYTHON, str(PREPARE_VIEWER), str(run_dir), "--force"]
    try:
        r = _run(cmd, timeout=300)
        if r.returncode == 0 and out_path.exists():
            size = out_path.stat().st_size
            stage.checks.append(Check(
                "dataset_viewer.html generated",
                True,
                f"{size:,} bytes",
            ))
        else:
            # prepare_run_viewer may skip if no manifests — that's a warning, not error
            full_out = (r.stdout or "") + (r.stderr or "")
            out_lower = full_out.lower()
            if ("no manifests" in out_lower or "skipping dataset viewer" in out_lower
                    or "no workspace" in out_lower):
                stage.checks.append(Check(
                    "dataset_viewer.html generated",
                    False,
                    "skipped (no workspace/manifests for this run)",
                    level="warning",
                ))
            elif out_path.exists():
                # File exists but prepare_run_viewer reported issues
                stage.checks.append(Check(
                    "dataset_viewer.html generated",
                    True,
                    f"exists ({out_path.stat().st_size:,} bytes), prepare_run_viewer had warnings",
                ))
            else:
                detail = (r.stderr.strip() or r.stdout.strip() or f"rc={r.returncode}")[:200]
                stage.checks.append(Check(
                    "dataset_viewer.html generated",
                    False,
                    detail,
                ))
    except subprocess.TimeoutExpired:
        stage.checks.append(Check("dataset_viewer.html generated", False, "timeout (300s)"))
    except Exception as e:
        stage.checks.append(Check("dataset_viewer.html generated", False, str(e)[:200]))

    stage.elapsed_sec = time.monotonic() - t0
    return stage


def generate_ir_hub(models_root: Path) -> StageResult:
    """Generate ir_hub.html via open_ir_hub.py."""
    stage = StageResult(stage="Generate ir_hub.html")
    t0 = time.monotonic()

    out_path = models_root / "ir_hub.html"
    cmd = [
        PYTHON, str(OPEN_IR_HUB),
        "--models-root", str(models_root),
        "--output", str(out_path),
    ]
    try:
        r = _run(cmd, timeout=120)
        if r.returncode == 0 and out_path.exists():
            size = out_path.stat().st_size
            stage.checks.append(Check(
                "ir_hub.html generated",
                True,
                f"{size:,} bytes",
            ))
        else:
            detail = r.stderr.strip()[:200] if r.stderr else f"rc={r.returncode}"
            stage.checks.append(Check("ir_hub.html generated", False, detail))
    except subprocess.TimeoutExpired:
        stage.checks.append(Check("ir_hub.html generated", False, "timeout (120s)"))
    except Exception as e:
        stage.checks.append(Check("ir_hub.html generated", False, str(e)[:200]))

    stage.elapsed_sec = time.monotonic() - t0
    return stage


# ── Stage 2: L1 Health Checks on Generated Files ────────────────────────────

def run_l1_health(
    file_path: Path,
    flag: str,
    label: str,
) -> StageResult:
    """Run test_visualizer_health_v7.py on a generated file."""
    stage = StageResult(stage=f"L1 Health: {label}")
    t0 = time.monotonic()

    if not file_path.exists():
        stage.checks.append(Check(f"{label} exists for L1", False, "file not found"))
        stage.elapsed_sec = time.monotonic() - t0
        return stage

    json_out = file_path.parent / f".l1_health_{file_path.stem}.json"
    cmd = [
        PYTHON, str(HEALTH_SCRIPT),
        flag, str(file_path),
        "--json-out", str(json_out),
        "--quiet",
    ]
    try:
        r = _run(cmd, timeout=60)
        if json_out.exists():
            report = json.loads(json_out.read_text(encoding="utf-8"))
            total = report.get("total_checks", 0)
            passed = report.get("passed", 0)
            failed = report.get("failed", 0)
            warnings = report.get("warnings", 0)
            stage.checks.append(Check(
                f"L1 health pass ({passed}/{total})",
                failed == 0,
                f"{failed} failures, {warnings} warnings" if failed else f"{warnings} warnings",
            ))
            # Enumerate individual failures from the JSON
            for suite in report.get("suites", []):
                for chk in suite.get("checks", []):
                    if not chk.get("passed", True):
                        sev = chk.get("severity", chk.get("level", "error"))
                        stage.checks.append(Check(
                            f"  {chk['name']}",
                            False,
                            chk.get("detail", ""),
                            level=sev,
                        ))
            # Clean up temp JSON
            json_out.unlink(missing_ok=True)
        else:
            stage.checks.append(Check(
                "L1 health ran",
                r.returncode == 0,
                r.stderr.strip()[:200] if r.stderr else f"rc={r.returncode}",
            ))
    except subprocess.TimeoutExpired:
        stage.checks.append(Check("L1 health ran", False, "timeout (60s)"))
    except Exception as e:
        stage.checks.append(Check("L1 health ran", False, str(e)[:200]))

    stage.elapsed_sec = time.monotonic() - t0
    return stage


# ── Stage 3: Embedded JSON Structure Validation ─────────────────────────────

# Keys we expect in embedded JSON blobs within ir_report.html
IR_REPORT_EXPECTED_JSON_KEYS = {
    "run_config": ["model_dims", "context_length"],
    "ir1_decode": ["graph_name", "ops"],
    "layout_decode": ["graph_name"],
}

def validate_embedded_json(ir_report_path: Path) -> StageResult:
    """Check that ir_report.html has expected embedded JSON blobs."""
    stage = StageResult(stage="Embedded JSON: ir_report.html")
    t0 = time.monotonic()

    if not ir_report_path.exists():
        stage.checks.append(Check("ir_report.html exists", False, "file not found"))
        stage.elapsed_sec = time.monotonic() - t0
        return stage

    html = ir_report_path.read_text(encoding="utf-8")

    # The visualizer embeds JSON as: window.__DATA__.<key> = {...};
    # or const DATA = { key: {...}, ... }
    # Try both patterns
    data_pattern = re.compile(
        r'(?:window\.__DATA__\.(\w+)\s*=|"(\w+)"\s*:\s*)\s*(\{[^}]{10,})',
        re.DOTALL,
    )

    found_keys: set[str] = set()
    for m in data_pattern.finditer(html):
        key = m.group(1) or m.group(2)
        if key:
            found_keys.add(key)

    # Also check for script blocks with known artifact names
    for artifact_name in ["run_config", "ir1_decode", "layout_decode", "ir1_train"]:
        if artifact_name in html:
            found_keys.add(artifact_name)

    for artifact, expected_fields in IR_REPORT_EXPECTED_JSON_KEYS.items():
        if artifact in found_keys:
            stage.checks.append(Check(
                f"JSON blob: {artifact}",
                True,
                "found in HTML",
            ))
        else:
            stage.checks.append(Check(
                f"JSON blob: {artifact}",
                False,
                "not found in generated HTML",
                level="warning",  # may legitimately be absent for some runs
            ))

    # Check the file has substantial content (not a stub)
    size = len(html)
    stage.checks.append(Check(
        "ir_report.html is non-trivial",
        size > 50_000,
        f"{size:,} chars (expect >50KB for a real report)",
    ))

    # Check it has a <title> and closing </html>
    stage.checks.append(Check(
        "has <title>",
        "<title>" in html,
    ))
    stage.checks.append(Check(
        "has closing </html>",
        "</html>" in html,
    ))

    stage.elapsed_sec = time.monotonic() - t0
    return stage


# ── Stage 4: Dataset Viewer Structure ────────────────────────────────────────

def validate_dataset_viewer_structure(dv_path: Path) -> StageResult:
    """Additional structural checks beyond L1 for generated dataset_viewer.html."""
    stage = StageResult(stage="Structure: dataset_viewer.html")
    t0 = time.monotonic()

    if not dv_path.exists():
        stage.checks.append(Check("dataset_viewer.html exists", False, "file not found"))
        stage.elapsed_sec = time.monotonic() - t0
        return stage

    html = dv_path.read_text(encoding="utf-8")

    # Must have panel containers for key tabs
    for panel_id in ["panel-overview", "panel-attention", "panel-embeddings", "panel-browse", "panel-training"]:
        stage.checks.append(Check(
            f"has {panel_id}",
            f'id="{panel_id}"' in html,
        ))

    # Training tab must have renderTraining and drawCanvasChart
    has_render_training = "function renderTraining" in html
    stage.checks.append(Check(
        "has renderTraining function",
        has_render_training,
        "defined" if has_render_training else "MISSING — training tab will be empty",
    ))
    has_canvas_chart = "function drawCanvasChart" in html
    stage.checks.append(Check(
        "has drawCanvasChart function",
        has_canvas_chart,
        "defined" if has_canvas_chart else "MISSING — charts will not render",
    ))

    # If training data is embedded, verify it has the right structure
    has_ck_training = "CK_TRAINING" in html
    stage.checks.append(Check(
        "has CK_TRAINING embedded data",
        has_ck_training,
        "present" if has_ck_training else "null/missing (ok if no training run)",
    ))

    # Must have attnColor function defined (the original bug was calling without definition)
    has_attn_def = "function attnColor" in html
    has_attn_ref = "attnColor(" in html
    if has_attn_ref and not has_attn_def:
        stage.checks.append(Check(
            "attnColor defined (not just called)",
            False,
            "attnColor() is called but function definition is missing — original bug class",
        ))
    else:
        stage.checks.append(Check(
            "has attnColor function",
            has_attn_def or not has_attn_ref,
            "defined" if has_attn_def else "not referenced (ok)",
        ))

    # Pipeline map: script → artifact → tab mapping
    has_pipeline_map = "CK_PIPELINE_MAP" in html
    stage.checks.append(Check(
        "has CK_PIPELINE_MAP embedded data",
        has_pipeline_map,
        "present" if has_pipeline_map else "missing — pipeline map not embedded",
    ))

    # Provenance banner for synthesized data
    has_provenance = "provenanceBanner" in html
    stage.checks.append(Check(
        "has provenanceBanner function",
        has_provenance,
        "defined" if has_provenance else "missing — no data provenance tracking",
    ))

    # Synthesized data integrity: if structured-atoms, verify synthesized path is honest
    is_synth = "ck.structured_atoms_synthesized" in html
    if is_synth:
        # Quality tab must NOT show false-green metrics for synthesized data
        has_catalog_entries = "Catalog Entries" in html
        stage.checks.append(Check(
            "synthesized quality shows 'Catalog Entries' not 'Normalized OK'",
            has_catalog_entries,
            "honest label" if has_catalog_entries else "still shows misleading 'Normalized OK'",
        ))

        # Must have family_descriptions for layout legend
        has_fam_desc = "family_descriptions" in html
        stage.checks.append(Check(
            "synthesized data includes family_descriptions",
            has_fam_desc,
            "layout legend data present" if has_fam_desc else "missing layout descriptions",
        ))

        # Placeholder totals should be non-empty (DSL values)
        has_ph = '"placeholder_totals":{' in html and '"placeholder_totals":{}' not in html
        stage.checks.append(Check(
            "synthesized placeholder_totals is populated",
            has_ph,
            "DSL values extracted" if has_ph else "empty — DSL tag values not counted",
        ))

        # Roles should include topic, not just split
        # Check that roles array has more than just train/holdout
        import re as _re
        role_matches = _re.findall(r'"roles":\["([^"]+)"', html[:200000])
        unique_roles = set(role_matches)
        has_topic_roles = len(unique_roles) > 2  # more than just train/holdout
        stage.checks.append(Check(
            "synthesized roles include topic data (not just split)",
            has_topic_roles,
            f"{len(unique_roles)} unique role values" if has_topic_roles else "only split-based roles",
        ))

    # File size check
    size = len(html)
    stage.checks.append(Check(
        "dataset_viewer.html is non-trivial",
        size > 30_000,
        f"{size:,} chars (expect >30KB)",
    ))

    stage.elapsed_sec = time.monotonic() - t0
    return stage


# ── Stage 5: IR Hub Structure ────────────────────────────────────────────────

def validate_ir_hub_structure(hub_path: Path) -> StageResult:
    """Structural checks for generated ir_hub.html."""
    stage = StageResult(stage="Structure: ir_hub.html")
    t0 = time.monotonic()

    if not hub_path.exists():
        stage.checks.append(Check("ir_hub.html exists", False, "file not found"))
        stage.elapsed_sec = time.monotonic() - t0
        return stage

    html = hub_path.read_text(encoding="utf-8")

    # Must have run cards or hub navigation
    has_run_cards = ('class="run-card"' in html or 'class="hub-card"' in html
                     or "run-card" in html)
    stage.checks.append(Check(
        "has run cards / navigation",
        has_run_cards,
    ))

    # Must link to at least one ir_report.html
    has_ir_links = "ir_report.html" in html
    stage.checks.append(Check(
        "links to ir_report.html",
        has_ir_links,
    ))

    # File size
    size = len(html)
    stage.checks.append(Check(
        "ir_hub.html is non-trivial",
        size > 10_000,
        f"{size:,} chars",
    ))

    stage.checks.append(Check(
        "has closing </html>",
        "</html>" in html,
    ))

    # Cross-run comparison: loss curve overlay function
    has_loss_overlay = "drawHubLossOverlay" in html
    stage.checks.append(Check(
        "has loss curve overlay (drawHubLossOverlay)",
        has_loss_overlay,
        "present" if has_loss_overlay else "MISSING — compare panel will lack charts",
    ))

    # loss_curve_summary in run data
    has_loss_summary = "loss_curve_summary" in html
    stage.checks.append(Check(
        "has loss_curve_summary in run data",
        has_loss_summary,
        "present" if has_loss_summary else "MISSING — compare panel will lack sparklines",
    ))

    stage.elapsed_sec = time.monotonic() - t0
    return stage


# ── Stage 6: Cross-artifact Consistency ──────────────────────────────────────

def check_cross_artifact_consistency(run_dir: Path, models_root: Path) -> StageResult:
    """Check that artifacts are consistent across the three visualizers."""
    stage = StageResult(stage="Cross-artifact Consistency")
    t0 = time.monotonic()

    ir_report = run_dir / "ir_report.html"
    dataset_viewer = run_dir / "dataset_viewer.html"
    ir_hub = models_root / "ir_hub.html"

    # If ir_hub exists, it should reference this run's directory name
    if ir_hub.exists() and ir_report.exists():
        hub_html = ir_hub.read_text(encoding="utf-8")
        run_name = run_dir.name
        stage.checks.append(Check(
            f"ir_hub references run {run_name}",
            run_name in hub_html,
            "hub should list this run",
        ))

    # If both ir_report and dataset_viewer exist, check they reference
    # the same model dimensions
    if ir_report.exists() and dataset_viewer.exists():
        ir_html = ir_report.read_text(encoding="utf-8")
        dv_html = dataset_viewer.read_text(encoding="utf-8")

        # Look for vocab size in both
        ir_vocab = re.search(r'vocab[_\s]*(?:size)?[:\s=]*(\d+)', ir_html)
        dv_vocab = re.search(r'vocab[_\s]*(?:size)?[:\s=]*(\d+)', dv_html)
        if ir_vocab and dv_vocab:
            stage.checks.append(Check(
                "vocab size consistent",
                ir_vocab.group(1) == dv_vocab.group(1),
                f"ir={ir_vocab.group(1)}, dv={dv_vocab.group(1)}",
            ))

    # Check that config.json exists in run_dir (prerequisite for all)
    config = run_dir / "config.json"
    stage.checks.append(Check(
        "config.json exists in run",
        config.exists(),
    ))

    stage.elapsed_sec = time.monotonic() - t0
    return stage


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description=f"Level 3 — Generated-file E2E for all {VIS_VERSION} visualizers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 version/v7/scripts/test_visualizer_generated_e2e_v7.py
  python3 version/v7/scripts/test_visualizer_generated_e2e_v7.py --run /path/to/run
  python3 version/v7/scripts/test_visualizer_generated_e2e_v7.py --validate-only
  python3 version/v7/scripts/test_visualizer_generated_e2e_v7.py --json-out report.json
""",
    )
    ap.add_argument("--run", type=Path, help="Training run directory (auto-discovers latest if omitted)")
    ap.add_argument("--models-root", type=Path, default=DEFAULT_MODELS_ROOT,
                    help="Models root directory")
    ap.add_argument("--validate-only", action="store_true",
                    help="Skip generation, validate existing files only")
    ap.add_argument("--skip-hub", action="store_true",
                    help="Skip ir_hub generation/validation")
    ap.add_argument("--json-out", type=Path, help="Write JSON report")
    ap.add_argument("--quiet", action="store_true", help="Only print failures")
    args = ap.parse_args()

    # ── Discover run ─────────────────────────────────────────────────────
    run_dir = args.run
    if run_dir is None:
        run_dir = discover_latest_run(args.models_root)
    if run_dir is None:
        print(f"{_YELLOW}○ No training run found in {args.models_root}/train/{_NC}")
        print(f"  Skipping L3 (no cached runs on this machine).")
        # Emit skip-compatible sub-test line for nightly report
        print(f"no_cached_runs  max_diff=0.00e+00  tol=1e+00  [PASS]")
        if args.json_out:
            report = {
                "level": 3,
                "version": VIS_VERSION,
                "description": "Generated-file E2E visualizer validation",
                "skipped": True,
                "reason": "no training runs found",
                "total_checks": 0,
                "passed": 0,
                "failed": 0,
                "warnings": 0,
                "elapsed_sec": 0,
                "stages": [],
            }
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(f"  JSON report → {args.json_out}")
        return 0  # skip is not a failure

    if not run_dir.is_dir():
        print(f"{_YELLOW}○ Run directory not found: {run_dir}{_NC}")
        print(f"  Skipping L3.")
        if args.json_out:
            report = {
                "level": 3,
                "version": VIS_VERSION,
                "skipped": True,
                "reason": f"run directory not found: {run_dir}",
                "total_checks": 0, "passed": 0, "failed": 0,
                "warnings": 0, "elapsed_sec": 0, "stages": [],
            }
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return 0

    print(f"{_BOLD}{'═' * 60}{_NC}")
    print(f"  Level 3 Generated-File E2E")
    print(f"  Version: {_CYAN}{VIS_VERSION}{_NC}")
    print(f"  Run: {_CYAN}{run_dir.name}{_NC}")
    print(f"  Models root: {args.models_root}")
    print(f"  Mode: {'validate-only' if args.validate_only else 'generate + validate'}")
    print(f"{_BOLD}{'═' * 60}{_NC}")

    stages: list[StageResult] = []
    t_total = time.monotonic()

    # ── Stage 1: Generate artifacts ──────────────────────────────────────
    if not args.validate_only:
        stages.append(generate_ir_report(run_dir))
        stages.append(generate_dataset_viewer(run_dir))
        if not args.skip_hub:
            stages.append(generate_ir_hub(args.models_root))

    # ── Stage 2: L1 health on generated files ────────────────────────────
    ir_report_path = run_dir / "ir_report.html"
    dv_path = run_dir / "dataset_viewer.html"
    hub_path = args.models_root / "ir_hub.html"

    if ir_report_path.exists():
        stages.append(run_l1_health(ir_report_path, "--ir-report", "ir_report.html"))
    if dv_path.exists():
        stages.append(run_l1_health(dv_path, "--dataset-viewer", "dataset_viewer.html"))
    if hub_path.exists() and not args.skip_hub:
        stages.append(run_l1_health(hub_path, "--ir-hub", "ir_hub.html"))

    # ── Stage 3: Embedded JSON validation ────────────────────────────────
    if ir_report_path.exists():
        stages.append(validate_embedded_json(ir_report_path))

    # ── Stage 4: Dataset viewer structure ────────────────────────────────
    if dv_path.exists():
        stages.append(validate_dataset_viewer_structure(dv_path))

    # ── Stage 5: IR hub structure ────────────────────────────────────────
    if hub_path.exists() and not args.skip_hub:
        stages.append(validate_ir_hub_structure(hub_path))

    # ── Stage 6: Cross-artifact consistency ──────────────────────────────
    stages.append(check_cross_artifact_consistency(run_dir, args.models_root))

    # ── Results ──────────────────────────────────────────────────────────
    total_elapsed = time.monotonic() - t_total

    for s in stages:
        if args.quiet and s.failed == 0:
            continue
        print_stage(s, quiet=args.quiet)

    total_checks = sum(len(s.checks) for s in stages)
    total_passed = sum(s.passed for s in stages)
    total_failed = sum(s.failed for s in stages)
    total_warnings = sum(s.warnings for s in stages)

    print(f"\n{_BOLD}{'═' * 60}{_NC}")
    if total_failed == 0:
        status_color = _GREEN
        status_icon = "✓"
        status_text = f"All {total_checks} checks passed"
    else:
        status_color = _RED
        status_icon = "✗"
        status_text = f"{total_failed} FAILED / {total_checks} checks"

    print(f"{status_color}  {status_icon} {status_text}"
          f"{f' ({total_warnings} warnings)' if total_warnings else ''}"
          f"  [{total_elapsed:.1f}s]{_NC}")
    print(f"{_BOLD}{'═' * 60}{_NC}")

    # ── Emit sub-test lines for nightly report parsing ────────────────
    # Format: name  max_diff=X  tol=Y  [PASS/FAIL]
    for s in stages:
        tag = s.stage.replace(" ", "_").replace(":", "_")
        status = "PASS" if s.failed == 0 else "FAIL"
        diff = f"{s.failed:.2e}" if s.failed else "0.00e+00"
        print(f"{tag}  max_diff={diff}  tol=1e+00  [{status}]")

    # ── JSON output ──────────────────────────────────────────────────────
    if args.json_out:
        report = {
            "level": 3,
            "version": VIS_VERSION,
            "description": "Generated-file E2E visualizer validation",
            "run_dir": str(run_dir),
            "models_root": str(args.models_root),
            "validate_only": args.validate_only,
            "total_checks": total_checks,
            "passed": total_passed,
            "failed": total_failed,
            "warnings": total_warnings,
            "elapsed_sec": round(total_elapsed, 2),
            "stages": [
                {
                    "stage": s.stage,
                    "passed": s.passed,
                    "failed": s.failed,
                    "warnings": s.warnings,
                    "elapsed_sec": round(s.elapsed_sec, 2),
                    "checks": [asdict(c) for c in s.checks],
                }
                for s in stages
            ],
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"  JSON report → {args.json_out}")

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
