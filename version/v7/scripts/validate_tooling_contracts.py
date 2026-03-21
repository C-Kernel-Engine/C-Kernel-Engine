#!/usr/bin/env python3
"""
Validate v7 tooling contracts across IR/codegen/parity orchestration.

This is a static contract checker intended to run before heavy parity/E2E jobs.
It reuses existing scripts as sources of truth and reports drift early.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
V7 = ROOT / "version" / "v7"
SCRIPTS = V7 / "scripts"
PARITY_DIR = SCRIPTS / "parity"


@dataclass
class CheckRow:
    layer: str
    handoff: str
    contract: str
    status: str
    detail: str


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _parse_ast(path: Path) -> ast.AST:
    return ast.parse(_read_text(path), filename=str(path))


def _extract_dict_constant(path: Path, name: str) -> dict[str, Any] | None:
    tree = _parse_ast(path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    try:
                        value = ast.literal_eval(node.value)
                    except Exception:
                        return None
                    if isinstance(value, dict):
                        return value
    return None


def _extract_function_args(path: Path, function_name: str) -> set[str]:
    tree = _parse_ast(path)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            args = {a.arg for a in node.args.args}
            args.update(a.arg for a in node.args.kwonlyargs)
            if node.args.vararg is not None:
                args.add(node.args.vararg.arg)
            if node.args.kwarg is not None:
                args.add(node.args.kwarg.arg)
            return args
    return set()


def _extract_run_parity_call_kwargs(path: Path) -> set[str]:
    tree = _parse_ast(path)
    kwargs: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        fn_name = None
        if isinstance(fn, ast.Name):
            fn_name = fn.id
        elif isinstance(fn, ast.Attribute):
            fn_name = fn.attr
        if fn_name != "run_parity_test":
            continue
        for kw in node.keywords:
            if kw.arg:
                kwargs.add(kw.arg)
    return kwargs


def _extract_add_argument_flags(path: Path) -> set[str]:
    tree = _parse_ast(path)
    flags: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, ast.Attribute) and fn.attr == "add_argument"):
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                if arg.value.startswith("--"):
                    flags.add(arg.value)
            elif isinstance(arg, ast.Tuple):
                for elem in arg.elts:
                    if isinstance(elem, ast.Constant) and isinstance(elem.value, str) and elem.value.startswith("--"):
                        flags.add(elem.value)
    return flags


def _extract_add_argument_choices(path: Path, flag: str) -> set[str]:
    tree = _parse_ast(path)
    choices: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, ast.Attribute) and fn.attr == "add_argument"):
            continue
        has_flag = False
        for arg in node.args:
            if isinstance(arg, ast.Constant) and arg.value == flag:
                has_flag = True
                break
        if not has_flag:
            continue
        for kw in node.keywords:
            if kw.arg != "choices":
                continue
            try:
                literal = ast.literal_eval(kw.value)
            except Exception:
                literal = None
            if isinstance(literal, (list, tuple)):
                for item in literal:
                    if isinstance(item, str):
                        choices.add(item)
    return choices


def _extract_codegen_dump_op_keys(path: Path) -> set[str]:
    tree = _parse_ast(path)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        target_names = [
            t.id for t in node.targets if isinstance(t, ast.Name)
        ]
        if "dump_op_map" not in target_names:
            continue
        try:
            literal = ast.literal_eval(node.value)
        except Exception:
            return set()
        if isinstance(literal, dict):
            return {k for k in literal.keys() if isinstance(k, str)}
    return set()


def _extract_template_names(templates_dir: Path) -> set[str]:
    out: set[str] = set()
    for p in sorted(templates_dir.glob("*.json")):
        try:
            obj = json.loads(_read_text(p))
        except Exception:
            continue
        name = obj.get("name")
        if isinstance(name, str) and name.strip():
            out.add(name.strip().lower())
    return out


def _summarize(values: set[str], max_items: int = 5) -> str:
    if not values:
        return "-"
    items = sorted(values)
    if len(items) <= max_items:
        return ",".join(items)
    return f"{','.join(items[:max_items])},+{len(items) - max_items}"


def _status_for(*parts: str) -> str:
    if "FAIL" in parts:
        return "FAIL"
    if "WARN" in parts:
        return "WARN"
    return "PASS"


def _render_table(rows: list[CheckRow]) -> str:
    headers = ["Layer", "Handoff", "Contract", "Status", "Notes"]
    matrix = [headers] + [
        [r.layer, r.handoff, r.contract, r.status, r.detail] for r in rows
    ]
    widths = [max(len(str(row[i])) for row in matrix) for i in range(len(headers))]

    def fmt_row(row: list[str]) -> str:
        return " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers)))

    sep = "-+-".join("-" * w for w in widths)
    out = [fmt_row(headers), sep]
    for r in rows:
        out.append(fmt_row([r.layer, r.handoff, r.contract, r.status, r.detail]))
    return "\n".join(out)


def run_checks() -> list[CheckRow]:
    rows: list[CheckRow] = []

    parity_test_py = SCRIPTS / "parity_test.py"
    detailed_py = SCRIPTS / "detailed_parity_analysis.py"
    autocheck_py = SCRIPTS / "model_autocheck.py"
    autopsy_py = PARITY_DIR / "parity_autopsy.py"
    converter_py = PARITY_DIR / "llama_to_ckdmp_converter.py"
    codegen_py = SCRIPTS / "codegen_v7.py"
    gemma_runner_sh = SCRIPTS / "run_gemma_parity_min.sh"

    # L1: Template -> IR/Parity model-family contract
    required_families = {"gemma", "qwen2", "qwen3", "llama", "mistral"}
    template_names = _extract_template_names(V7 / "templates")
    supported_present = autocheck_py.exists()
    supported = _extract_dict_constant(autocheck_py, "SUPPORTED_FAMILIES") or {}
    parity_map = _extract_dict_constant(detailed_py, "PARITY_MODEL_MAP") or {}
    profile = _extract_dict_constant(autopsy_py, "FAMILY_PROFILE") or {}
    profile_keys = {k for k in profile.keys() if isinstance(k, str)}
    converter_choices = _extract_add_argument_choices(converter_py, "--model")

    missing_supported = required_families - set(str(k) for k in supported.keys())
    missing_parity_map = required_families - set(str(k) for k in parity_map.keys())
    missing_profile = required_families - profile_keys
    missing_converter = required_families - converter_choices
    aliases: list[str] = []
    for fam in sorted(required_families):
        mapped = parity_map.get(fam)
        if isinstance(mapped, str) and mapped != fam:
            aliases.append(f"{fam}->{mapped}")

    fam_status = "PASS"
    fam_notes: list[str] = []
    if not supported_present:
        fam_notes.append("autocheck-script-absent")
    elif missing_supported:
        fam_status = _status_for(fam_status, "FAIL")
        fam_notes.append(f"autocheck-missing={_summarize(missing_supported)}")
    if missing_parity_map:
        fam_status = _status_for(fam_status, "FAIL")
        fam_notes.append(f"parity-map-missing={_summarize(missing_parity_map)}")
    if missing_profile:
        fam_status = _status_for(fam_status, "FAIL")
        fam_notes.append(f"autopsy-profile-missing={_summarize(missing_profile)}")
    if missing_converter:
        fam_status = _status_for(fam_status, "FAIL")
        fam_notes.append(f"converter-missing={_summarize(missing_converter)}")
    if aliases:
        fam_status = _status_for(fam_status, "WARN")
        fam_notes.append(f"aliases={','.join(aliases)}")
    fam_notes.append(f"templates={_summarize(template_names)}")
    rows.append(
        CheckRow(
            layer="L1",
            handoff="Template -> IR/Parity",
            contract="Family coverage + mapping",
            status=fam_status,
            detail="; ".join(fam_notes),
        )
    )

    # L2: IR ops -> codegen dump contract
    dump_keys = _extract_codegen_dump_op_keys(codegen_py)
    probe_required_ops = {
        "attn_norm",
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        "post_attention_norm",
        "ffn_norm",
        "mlp_down",
    }
    missing_ops = probe_required_ops - dump_keys
    l2_status = "PASS" if not missing_ops else "FAIL"
    l2_note = (
        f"dump-ops-present={len(dump_keys)}"
        if not missing_ops
        else f"missing-dump-ops={_summarize(missing_ops)}"
    )
    rows.append(
        CheckRow(
            layer="L2",
            handoff="IR -> Codegen",
            contract="Probe-required ops mapped for parity dumps",
            status=l2_status,
            detail=l2_note,
        )
    )

    # L3: codegen/parity dump -> parity engine API contract
    parity_args = _extract_function_args(parity_test_py, "run_parity_test")
    caller_kwargs = _extract_run_parity_call_kwargs(detailed_py) | _extract_run_parity_call_kwargs(autopsy_py)
    missing_kwargs = caller_kwargs - parity_args
    l3_status = "PASS" if not missing_kwargs else "FAIL"
    rows.append(
        CheckRow(
            layer="L3",
            handoff="Codegen Dump -> Parity Engine",
            contract="run_parity_test kwargs accepted by parity_test.py",
            status=l3_status,
            detail=(
                "caller-kwargs-compatible"
                if not missing_kwargs
                else f"missing-kwargs={_summarize(missing_kwargs)}"
            ),
        )
    )

    # L4: Gemma harness -> parity CLI contract
    parity_cli_flags = _extract_add_argument_flags(parity_test_py)
    gemma_text = _read_text(gemma_runner_sh)
    gemma_required_flags: set[str] = set()
    for flag in ("--model", "--pass"):
        if flag in gemma_text:
            gemma_required_flags.add(flag)
    missing_cli = gemma_required_flags - parity_cli_flags
    l4_status = "PASS" if not missing_cli else "FAIL"
    rows.append(
        CheckRow(
            layer="L4",
            handoff="Gemma Harness -> Parity CLI",
            contract="run_gemma_parity_min.sh flags supported",
            status=l4_status,
            detail=(
                "gemma-cli-compatible"
                if not missing_cli
                else f"unsupported-flags={_summarize(missing_cli)}"
            ),
        )
    )

    # L5: Autocheck -> targeted probe contract (token defaults)
    token_checks = [
        (PARITY_DIR / "check_post_attn_chain_prefill.py", r'--tokens",\s*default="([^"]+)"'),
        (PARITY_DIR / "check_attn_norm_contract_prefill.py", r'--tokens",\s*default="([^"]+)"'),
        (PARITY_DIR / "check_qproj_contract.py", r'--token",\s*type=int,\s*default=(\d+)'),
        (SCRIPTS / "model_autocheck.py", r'--tokens",\s*"([^"]+)"'),
    ]
    hardcoded_hits: list[str] = []
    for path, pat in token_checks:
        text = _read_text(path)
        m = re.search(pat, text)
        if m:
            hardcoded_hits.append(f"{path.name}:{m.group(1)}")
    defaults_module = PARITY_DIR / "probe_defaults.py"
    defaults_module_used = []
    for path in (
        PARITY_DIR / "check_post_attn_chain_prefill.py",
        PARITY_DIR / "check_attn_norm_contract_prefill.py",
        PARITY_DIR / "check_qproj_contract.py",
        SCRIPTS / "model_autocheck.py",
    ):
        text = _read_text(path)
        if "probe_defaults" in text:
            defaults_module_used.append(path.name)

    l5_status = "WARN" if hardcoded_hits else "PASS"
    rows.append(
        CheckRow(
            layer="L5",
            handoff="Autocheck -> Probe Scripts",
            contract="Probe token defaults externalized",
            status=l5_status,
            detail=(
                f"token defaults externalized via probe_defaults.py; users can set CK_V7_PROBE_* env vars; users={','.join(sorted(defaults_module_used))}"
                if not hardcoded_hits
                else (
                    f"hardcoded-defaults={';'.join(hardcoded_hits)}; "
                    "L5 means probe inputs are embedded in code, which can hide tokenizer/model-specific drift"
                )
            ),
        )
    )

    # L6: Make/CI preflight integration contract
    makefile = _read_text(ROOT / "Makefile")
    has_target = "v7-validate-contracts:" in makefile
    has_preflight_hook = "v7-validate-contracts" in makefile and (
        "ci-local-fast" in makefile or "e2e-v7" in makefile
    )
    l6_status = "PASS" if (has_target and has_preflight_hook) else "WARN"
    notes = []
    if not has_target:
        notes.append("target-missing")
    if not has_preflight_hook:
        notes.append("preflight-hook-missing")
    if not notes:
        notes.append("make-preflight-wired")
    rows.append(
        CheckRow(
            layer="L6",
            handoff="Make/CI -> E2E/Parity",
            contract="Contract validation runs before heavy tests",
            status=l6_status,
            detail=";".join(notes),
        )
    )

    # L7: canonical cache-root contract for IR hub / model artifacts
    readme = _read_text(V7 / "README.md")
    ck_run = _read_text(SCRIPTS / "ck_run_v7.py")
    ir_hub = _read_text(V7 / "tools" / "open_ir_hub.py")
    canonical_models_root = "~/.cache/ck-engine-v7/models"
    canonical_train_root = "~/.cache/ck-engine-v7/models/train"
    path_notes: list[str] = []
    path_status = "PASS"
    if canonical_models_root not in readme:
        path_status = "FAIL"
        path_notes.append("readme-missing-models-root")
    if canonical_train_root not in readme:
        path_status = "FAIL"
        path_notes.append("readme-missing-train-root")
    if canonical_models_root not in ck_run:
        path_status = "FAIL"
        path_notes.append("ck_run-missing-models-root")
    if canonical_models_root not in ir_hub:
        path_status = "FAIL"
        path_notes.append("ir_hub-missing-models-root")
    rows.append(
        CheckRow(
            layer="L7",
            handoff="Run layout -> IR Hub/Visualizer",
            contract="Canonical cache roots stay stable",
            status=path_status,
            detail=";".join(path_notes) if path_notes else "models-root=~/.cache/ck-engine-v7/models;train-root=~/.cache/ck-engine-v7/models/train",
        )
    )

    # L8: visualizer fixture / nightly stabilization path-policy contract
    visualizer_e2e = _read_text(SCRIPTS / "test_ir_visualizer_e2e_v7.py")
    stabilization_py = _read_text(SCRIPTS / "run_v7_stabilization_nightly_v7.py")
    visualizer_allow_count = visualizer_e2e.count("--allow-non-cache-run-dir")
    makefile_text = makefile
    l8_notes: list[str] = []
    l8_status = "PASS"
    if visualizer_allow_count < 2:
        l8_status = "FAIL"
        l8_notes.append(f"visualizer-non-cache-allow-count={visualizer_allow_count}")
    if 'Path("/tmp/v7_stabilization_nightly")' in stabilization_py:
        l8_status = "FAIL"
        l8_notes.append("stabilization-script-hardcodes-/tmp")
    if "V7_STABILIZATION_RUN_ROOT ?= /tmp/v7_stabilization_nightly" in makefile_text:
        l8_status = "FAIL"
        l8_notes.append("makefile-hardcodes-/tmp-run-root")
    if '--run-root "$(V7_STABILIZATION_RUN_ROOT)"' in makefile_text:
        l8_status = "FAIL"
        l8_notes.append("makefile-unconditionally-passes-run-root")
    rows.append(
        CheckRow(
            layer="L8",
            handoff="Fixture policy -> Nightly operator jobs",
            contract="Non-cache fixtures are explicit and nightly training roots stay cache-backed",
            status=l8_status,
            detail=";".join(l8_notes) if l8_notes else "visualizer-fixtures-explicit;stabilization-root-cache-backed",
        )
    )

    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate v7 tooling contracts")
    ap.add_argument("--strict", action="store_true", help="Treat WARN as failure")
    ap.add_argument("--json-out", type=Path, default=None, help="Write machine-readable report JSON")
    args = ap.parse_args()

    rows = run_checks()
    print("=" * 112)
    print("v7 TOOLING CONTRACT REPORT")
    print("=" * 112)
    print(_render_table(rows))
    print("=" * 112)
    print("L5 explanation: probe token defaults should come from shared config/env, not literals in scripts.")
    print("If L5 warns, targeted parity probes may be brittle across model families/tokenizers.")

    counts = {"PASS": 0, "WARN": 0, "FAIL": 0}
    for r in rows:
        counts[r.status] = counts.get(r.status, 0) + 1
    print(f"Summary: PASS={counts.get('PASS', 0)} WARN={counts.get('WARN', 0)} FAIL={counts.get('FAIL', 0)}")

    if args.json_out is not None:
        payload = {
            "summary": counts,
            "rows": [
                {
                    "layer": r.layer,
                    "handoff": r.handoff,
                    "contract": r.contract,
                    "status": r.status,
                    "detail": r.detail,
                }
                for r in rows
            ],
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"JSON: {args.json_out}")

    if counts.get("FAIL", 0) > 0:
        return 1
    if args.strict and counts.get("WARN", 0) > 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
