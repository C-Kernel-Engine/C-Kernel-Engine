#!/usr/bin/env python3
"""
Contract-driven JS unit-test harness for visualizer pure functions.

Reads test vectors from JSON contract files, runs them against standalone
JS fixture files via Node.js. No regex extraction, no brittle parsing.

Usage:
    python3 version/v7/scripts/test_visualizer_js_units_v7.py
    python3 version/v7/scripts/test_visualizer_js_units_v7.py --json-out report.json
    python3 version/v7/scripts/test_visualizer_js_units_v7.py --quiet

No npm packages required — just Python 3.8+ and Node.js.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
VIS_VERSION = os.environ.get("CK_VIS_VERSION", "v7")
CONTRACTS_DIR = Path(
    os.environ.get(
        "CK_VIS_CONTRACTS_DIR",
        str(ROOT / "version" / VIS_VERSION / "tests" / "contracts"),
    )
).expanduser()
IR_CONTRACT = CONTRACTS_DIR / "ir_visualizer_contract.json"
DS_CONTRACT = CONTRACTS_DIR / "dataset_viewer_contract.json"

# ── colours ──────────────────────────────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
DIM = "\033[2m"
RESET = "\033[0m"
TICK = "✓"
CROSS = "✗"

# ── result tracking ──────────────────────────────────────────────────
results: list[dict[str, Any]] = []
fail_count = 0


def record(name: str, passed: bool, detail: str = "") -> None:
    global fail_count
    results.append({"name": name, "passed": passed, "detail": detail})
    if not passed:
        fail_count += 1


# ── Contract loading ─────────────────────────────────────────────────

def load_contract(path: Path) -> dict:
    """Load a visualizer contract JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


# ── Node.js runner ───────────────────────────────────────────────────

def run_js(code: str, *, timeout: int = 10) -> tuple[bool, str]:
    """Run JS code via Node.js, return (success, stdout_or_stderr)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False, encoding="utf-8") as f:
        f.write(code)
        f.flush()
        tmp = f.name
    try:
        r = subprocess.run(
            ["node", tmp],
            capture_output=True, text=True, timeout=timeout,
        )
        if r.returncode == 0:
            return True, r.stdout.strip()
        return False, (r.stderr or r.stdout).strip()
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except FileNotFoundError:
        return False, "node not found"
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


# ── Build test harness from contract ─────────────────────────────────

def build_harness_from_contract(contract: dict, fixture_path: Path) -> str:
    """Build a Node.js test script from a contract + fixture file."""
    prefix = contract["name"]
    lines = [
        "// Auto-generated contract-driven test harness",
        "const __results = [];",
        "",
        f"// ── Load fixture: {fixture_path.name} ──",
        f"const fns = require({json.dumps(str(fixture_path))});",
        "// Expose all functions globally for test expressions",
        "for (const [k, v] of Object.entries(fns)) { global[k] = v; }",
        "",
        "// ── Test helpers ──",
        "function assertApprox(a, b, tol = 1e-6) {",
        "  if (typeof a === 'number' && typeof b === 'number') return Math.abs(a - b) < tol;",
        "  return false;",
        "}",
        "function assertDeepEq(a, b) {",
        "  return JSON.stringify(a) === JSON.stringify(b);",
        "}",
        "function test(name, fn) {",
        "  try {",
        "    const result = fn();",
        "    if (result === true || result === undefined) {",
        "      __results.push({ name, passed: true });",
        "    } else {",
        "      __results.push({ name, passed: false, detail: 'returned ' + JSON.stringify(result) });",
        "    }",
        "  } catch (e) {",
        "    __results.push({ name, passed: false, detail: String(e) });",
        "  }",
        "}",
        "",
        "// ── Tests from contract ──",
    ]

    # Verify fixture exports all needed functions
    fn_spec = contract.get("pure_function_tests", {})
    for fn_group in fn_spec.get("functions", []):
        fn_name = fn_group["name"]
        lines.append(f"test('{prefix}:fixture:{fn_name}', () => typeof fns.{fn_name} === 'function' || (() => {{ throw 'not exported' }})());")

    # Generate test cases from contract
    for fn_group in fn_spec.get("functions", []):
        fn_name = fn_group["name"]
        for tc in fn_group.get("tests", []):
            tc_name = f"{prefix}:{fn_name}:{tc['name']}"
            call = tc["call"]
            if "expect" in tc:
                expect_json = json.dumps(tc["expect"])
                lines.append(f"test({json.dumps(tc_name)}, () => assertDeepEq({call}, {expect_json}));")
            elif "expect_approx" in tc:
                expect_val = tc["expect_approx"]
                lines.append(f"test({json.dumps(tc_name)}, () => assertApprox({call}, {expect_val}));")
            elif "expect_fn" in tc:
                fn_body = tc["expect_fn"]
                lines.append(f"test({json.dumps(tc_name)}, () => {{ const r = {call}; return ({fn_body})(r); }});")

    lines.append("")
    lines.append("console.log(JSON.stringify(__results));")
    return "\n".join(lines)


# ── Fixture sync check ───────────────────────────────────────────────

def check_fixture_sync(contract: dict, fixture_path: Path, source_path: Path) -> None:
    """Verify the fixture file exports all functions declared in the contract."""
    prefix = contract["name"]
    if not fixture_path.exists():
        record(f"{prefix}:fixture_exists", False, f"missing: {fixture_path}")
        return
    record(f"{prefix}:fixture_exists", True)

    if not source_path.exists():
        record(f"{prefix}:source_exists", False, f"missing: {source_path}")
        return
    record(f"{prefix}:source_exists", True)

    # Quick sanity: ensure fixture is loadable via node
    ok, out = run_js(f"""
const fns = require({json.dumps(str(fixture_path))});
console.log(JSON.stringify(Object.keys(fns)));
""")
    if not ok:
        record(f"{prefix}:fixture_loadable", False, out[:200])
        return
    record(f"{prefix}:fixture_loadable", True)

    try:
        exported = json.loads(out)
    except json.JSONDecodeError:
        record(f"{prefix}:fixture_loadable", False, f"invalid JSON: {out[:100]}")
        return

    # Check all contract functions are exported
    fn_spec = contract.get("pure_function_tests", {})
    for fn_group in fn_spec.get("functions", []):
        fn_name = fn_group["name"]
        if fn_name in exported:
            record(f"{prefix}:exported:{fn_name}", True)
        else:
            record(f"{prefix}:exported:{fn_name}", False, f"{fn_name} not in fixture exports")


# ── Main ─────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description=f"Contract-driven JS unit tests for visualizer pure functions ({VIS_VERSION})"
    )
    ap.add_argument("--json-out", type=str, help="Write JSON report to file")
    ap.add_argument("--quiet", action="store_true", help="Only print failures and summary")
    args = ap.parse_args()

    # Check Node.js available
    try:
        r = subprocess.run(["node", "--version"], capture_output=True, text=True, timeout=5)
        if r.returncode != 0:
            print(f"{RED}  {CROSS} node not available{RESET}")
            return 1
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print(f"{RED}  {CROSS} node not found — install Node.js to run JS unit tests{RESET}")
        return 1

    # Load contracts
    contracts = []
    for cpath in sorted(CONTRACTS_DIR.glob("*_contract.json")):
        try:
            c = load_contract(cpath)
            contracts.append(c)
        except (json.JSONDecodeError, KeyError) as e:
            record(f"contract:{cpath.name}", False, str(e))

    if not contracts:
        print(f"{RED}  {CROSS} No contracts found in {CONTRACTS_DIR}{RESET}")
        return 1

    for contract in contracts:
        prefix = contract["name"]
        fn_spec = contract.get("pure_function_tests", {})
        fixture_rel = fn_spec.get("fixture_file", "")
        fixture_path = ROOT / fixture_rel if fixture_rel else None
        source_rel = contract.get("source", "")
        source_path = ROOT / source_rel if source_rel else None

        print(f"\n{'━' * 3} {prefix} JS Units {'━' * 3}")

        # 1. Check fixture sync
        if fixture_path and source_path:
            check_fixture_sync(contract, fixture_path, source_path)

        # 2. Run contract-driven tests
        if fixture_path and fixture_path.exists():
            harness = build_harness_from_contract(contract, fixture_path)
            ok, output = run_js(harness)
            if ok:
                try:
                    jr = json.loads(output)
                    for t in jr:
                        record(t["name"], t["passed"], t.get("detail", ""))
                except json.JSONDecodeError:
                    record(f"{prefix}:harness_parse", False, f"invalid JSON: {output[:200]}")
            else:
                record(f"{prefix}:harness_run", False, output[:300])

    # ── Print results ──
    passed = sum(1 for r in results if r["passed"])
    failed = sum(1 for r in results if not r["passed"])
    total = len(results)

    for r in results:
        if r["passed"] and not args.quiet:
            print(f"  {GREEN}{TICK}{RESET} {r['name']}")
        elif not r["passed"]:
            detail = f" — {r['detail']}" if r.get("detail") else ""
            print(f"  {RED}{CROSS}{RESET} {r['name']}{detail}")

    # ── Summary ──
    print()
    bar = "═" * 60
    if failed == 0:
        print(f"{bar}\n  {GREEN}{TICK} All {total} JS unit tests passed{RESET}\n{bar}")
    else:
        print(f"{bar}\n  {RED}{CROSS} {failed}/{total} JS unit tests FAILED{RESET}\n{bar}")

    # ── Emit sub-test lines for nightly report parsing ────────────────
    # Group by component (ir_visualizer / dataset_viewer)
    component_pass: dict[str, int] = {}
    component_fail: dict[str, int] = {}
    for r in results:
        comp = r["name"].split(":")[0] if ":" in r["name"] else "js_units"
        if r["passed"]:
            component_pass[comp] = component_pass.get(comp, 0) + 1
        else:
            component_fail[comp] = component_fail.get(comp, 0) + 1
    for comp in sorted(set(list(component_pass) + list(component_fail))):
        f = component_fail.get(comp, 0)
        status = "PASS" if f == 0 else "FAIL"
        diff = f"{f:.2e}" if f else "0.00e+00"
        print(f"L2_{comp}  max_diff={diff}  tol=1e+00  [{status}]")

    # ── JSON output ──
    if args.json_out:
        outpath = Path(args.json_out)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "tool": f"test_visualizer_js_units_{VIS_VERSION}",
            "version": VIS_VERSION,
            "total": total,
            "passed": passed,
            "failed": failed,
            "tests": results,
        }
        outpath.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n  JSON report: {outpath}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
