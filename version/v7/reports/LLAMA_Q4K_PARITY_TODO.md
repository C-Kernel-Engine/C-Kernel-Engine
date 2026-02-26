# Llama Q4_K Parity TODO (Nanbeige 4.1 3B)

Status date: 2026-02-26
Scope: `hf://mradermacher/Nanbeige4.1-3B-GGUF/Nanbeige4.1-3B.Q4_K_M.gguf`

## Current Findings

1. Llama parity report is not actionable yet (WARN-only path).
- Evidence: `.ck_build/detailed_parity_analysis_latest.json` contains:
  - `"Skipped ck_run; used existing dumps."`
  - `"Missing llama dump: .../llama_parity_dumps/dump.bin"`
- Impact: We do not have CK-vs-llama tensor diffs for prefill/decode.

2. ~~Projection contract checker fails before numeric diff on Q4_K.~~ **FIXED**
- `check_layer0_qkv_contract.py` now includes `q4_k` in the supported dtype list alongside `q6_k`, `q5_0`, `q8_0`.
- Script runs without `unsupported dtype q4_k` on Nanbeige.

3. Existing autopsy points to projection checks as blocker.
- Evidence: `.ck_build/parity_autopsy_latest.md`:
  - `q_proj rc=1 pass_contract=False`
  - `k_proj rc=1 pass_contract=False`
  - `v_proj rc=1 pass_contract=False`
- Impact: attention/rope/debugging is premature until projection contract path is fixed.

4. Llama default context can cause local OOM kill in llama.cpp smoke runs.
- Evidence: `llama-cli --model ...` without `-c` gets `-Killed` on this machine.
- Impact: false "runtime crash" signal if operator does not cap context.

## Priority TODO

## P0: Make parity run produce real llama reference dumps every time — **FIXED**
- [x] Update `ck_run_v7.py` detailed parity flow to fail-fast if llama dump file is absent, not WARN-only.
  - `ck_run_v7.py` line ~9020: changed from `log(..., C_ORANGE)` to `log_error(...)` + `sys.exit(1)`.
- [x] Add a post-check in `detailed_parity_analysis.py`: returns exit code 2 (not 0) when either the llama or CK reference dump is absent, and prints `[analysis] INCOMPLETE: missing ...` before exiting.
- [ ] Ensure `llama_parity_dumps/dump.bin` is created in run dir on every `--detailed-llamacpp-parity` invocation.
  - Remaining: investigate why `_run_llamacpp_parity()` returns False on this model (binary path, GGUF path, or parity binary crash).

Pass criteria:
- `detailed_parity_analysis_latest.json` contains non-empty `prefill_summary`/`decode_summary`.
- `parity_autopsy_latest.md` no longer says "llama reference dump missing/empty".
- `ck_run_v7.py --detailed-llamacpp-parity` exits non-zero when llama dump cannot be produced.

## P0: Add Q4_K support to layer0 QKV contract checker — **FIXED**
- [x] Extended `version/v7/scripts/parity/check_layer0_qkv_contract.py` to handle `dtype=q4_k`.
- [ ] Wire CK and llama parity C symbols for Q4_K projection path (numeric diff still TBD).
- [ ] Emit `max_diff`/`mean_diff` for q_proj/k_proj/v_proj instead of `None`.

Pass criteria:
- Script runs on Nanbeige without `unsupported dtype q4_k`. ✓
- Contract report returns numeric diffs and pass/fail gates for all three projections.

## P1: Lock operator commands to safe context defaults for llama.cpp
- [ ] In runbook snippets, always include `-c 1024` (or explicit bounded `-c`).
- [ ] Add note: model-default context may exceed local RAM and trigger OS kill.

Pass criteria:
- Operator smoke commands do not OOM on default laptop configuration.

## P1: Metadata sanity checks in detailed parity output
- [ ] Ensure `weights`/`ck_coverage` sections are populated (not null) when dumps exist.
- [ ] If null, fail report generation with explicit reason and remediation.

Pass criteria:
- `detailed_parity_analysis_latest.json` has populated `weights` and coverage fields.

## P2: Tokenizer verification for llama-family GGUF runs
- [ ] Add tokenizer sanity step in `template-audit` for llama-family:
  - compare prompt token IDs CK vs llama.cpp on 3 fixed prompts.
- [ ] Emit a compact mismatch report in JSON.

Pass criteria:
- Tokenization mismatches are visible before full decode parity run.

## Suggested Validation Commands

1. Template + structure audit:
```bash
python3 version/v7/scripts/ck_run_v7.py template-audit \
  ~/.cache/ck-engine-v7/models/mradermacher--Nanbeige4.1-3B-GGUF \
  --run /tmp/v7_nanbeige_template_audit \
  --context-len 1024 \
  --force-compile
```

2. Full detailed parity (must produce llama dump; now fails-fast if not):
```bash
python3 version/v7/scripts/ck_run_v7.py run \
  ~/.cache/ck-engine-v7/models/mradermacher--Nanbeige4.1-3B-GGUF \
  --context-len 1024 \
  --detailed-llamacpp-parity \
  --max-tokens 32
```

3. QKV contract check (Q4_K now supported):
```bash
.venv/bin/python version/v7/scripts/parity/check_layer0_qkv_contract.py \
  --model-dir ~/.cache/ck-engine-v7/models/mradermacher--Nanbeige4.1-3B-GGUF/.ck_build \
  --layer 0 --skip-prefill --tol 1e-3
```

4. llama.cpp bounded-memory smoke:
```bash
cd llama.cpp
build/bin/llama-cli \
  --model ~/.cache/ck-engine-v7/models/mradermacher--Nanbeige4.1-3B-GGUF/Nanbeige4.1-3B.Q4_K_M.gguf \
  -c 1024 -n 64 -p "Hello" --temp 0.0
```

5. Standalone detailed_parity_analysis (now returns exit 2 when dump absent):
```bash
python3 version/v7/scripts/detailed_parity_analysis.py \
  --model-dir ~/.cache/ck-engine-v7/models/mradermacher--Nanbeige4.1-3B-GGUF/.ck_build \
  --family llama
echo "exit: $?"
```
