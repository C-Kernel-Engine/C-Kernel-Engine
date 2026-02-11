# v6.6 Tooling Contracts Handoff (2026-02-11)

## Goal
Add a single preflight validator so IR/codegen/parity scripts fail fast on drift before expensive E2E/parity runs.

## What Was Implemented
- Added `version/v6.6/scripts/validate_tooling_contracts.py`.
- Added `version/v6.6/scripts/validate_model_matrix_v6_6.py` (dynamic 3-model validator).
- Added Make target `v6.6-validate-contracts`.
- Added Make targets `v6.6-validate-matrix` and `v6.6-validate-matrix-smoke`.
- Wired contract checks into:
  - `e2e-v66` (pre + post check)
  - `ci-local`
  - `ci-local-fast`
  - `v6.6-build` dependency chain
- Compatibility fixes so existing scripts work together:
  - `version/v6.6/scripts/parity_test.py` now accepts `--model` and `--pass` and supports `run_parity_test(..., model_family=..., pass_filter=...)`.
  - `version/v6.6/scripts/detailed_parity_analysis.py` uses native `qwen3` mapping.
  - `version/v6.6/scripts/parity/parity_autopsy.py` uses native `qwen3` mapping.
  - `version/v6.6/scripts/parity/llama_to_ckdmp_converter.py` now supports `qwen3`.
  - `version/v6.6/scripts/ck_run_v6_6.py` dummy smoke-weight path now supports both legacy `layout.sections` and new `memory.weights.entries` layouts (fixes `KeyError: 'sections'` in `--test` runs).

## IR Flow Contract Table
| Layer | Handoff | Artifact Boundary | Contract Test | Command |
|---|---|---|---|---|
| L1 | Template -> IR/Parity | `templates/*.json` -> family maps in tooling | Family coverage + mapping consistency | `make v6.6-validate-contracts` |
| L2 | IR -> Codegen | lowered ops -> `dump_op_map` in codegen | Probe-required ops present for dump hooks | `make v6.6-validate-contracts` |
| L3 | Codegen Dump -> Parity Engine | CKDMP dumps -> `run_parity_test` API | Caller kwargs supported by parity engine | `make v6.6-validate-contracts` |
| L4 | Gemma Harness -> Generic Parity | `run_gemma_parity_min.sh` -> parity CLI | CLI flags accepted (`--model`, `--pass`) | `make v6.6-validate-contracts` |
| L5 | Autocheck -> Probe Scripts | `model_autocheck.py` -> targeted probes | Hardcoded token defaults surfaced as warnings | `make v6.6-validate-contracts` |
| L6 | Make/CI -> E2E | preflight gate before heavy jobs | Validator wired before E2E/CI steps | `make ci-local-fast` or `make e2e-v66` |

## Standard Run Order
1. `make v6.6-validate-contracts`
2. `make v6.6-validate-matrix`
3. `make v6.6-test-quick`
4. `make e2e-v66`
5. `make llamacpp-parity` (or `make llamacpp-parity-full`)

## Report Artifacts
- Terminal table: printed by `validate_tooling_contracts.py`
- Machine-readable report: `version/v6.6/tools/contract_report_latest.json`
- Dynamic matrix report: `version/v6.6/tools/model_matrix_report_latest.json`

## Latest Validation Snapshot
- `make v6.6-validate-matrix` -> PASS for Gemma/Qwen2/Qwen3.
- `make v6.6-validate-matrix-smoke` -> PASS for Gemma/Qwen2/Qwen3 after smoke-layout fix.

## Next Suggested Extension
- IR visualizer integration path: load `contract_report_latest.json` in `version/v6.6/tools/ir_visualizer.html` as an overlay panel (stage statuses + drill-down notes).
