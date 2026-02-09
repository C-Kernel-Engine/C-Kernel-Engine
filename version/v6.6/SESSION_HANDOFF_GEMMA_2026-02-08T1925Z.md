# Gemma Session Handoff (2026-02-08T19:25:16Z)

## Goal
Stabilize Gemma 3 270M on v6.6 without regressing Qwen2/Qwen3.

## Where We Are
- Hard crash path is mostly fixed.
- Gemma now initializes, tokenizes, runs prefill/decode, and returns text.
- Output is still semantically wrong/garbled, but no longer crashing at init.

## Critical Fixes Applied This Session

1) Restored tokenizer stack from stash
- Source used: `stash@{1}`
- Restored files:
  - `Makefile`
  - `include/tokenizer/tokenizer.h`
  - `src/tokenizer/tokenizer.c`
  - `src/tokenizer/tokenizer_spm.c`
- Why: generated Gemma code used SPM APIs/fields not present in current tokenizer ABI.

2) Fixed tokenizer shared-lib symbol collision (partial root cause of segfault)
- File: `Makefile:516`
- Change: tokenizer `.so` now links with `-Wl,-Bsymbolic`.
- Why: both `libckernel_engine.so` and `libckernel_tokenizer.so` export `ck_tokenizer_*`; loader interposed wrong symbols.

3) Forced model compile link order to prefer tokenizer symbols
- File: `version/v6.6/scripts/ck_run_v6_6.py:850`
- Change: link order puts `-lckernel_tokenizer` before `-lckernel_engine`.

4) Fixed Gemma post-norm gamma binding (NULL gamma crash)
- File: `version/v6.6/kernel_maps/kernel_bindings.json:125`
- Change: `rmsnorm_forward.gamma` alt list now includes `post_attention_norm` and `post_ffn_norm`.
- Why: `lowered_prefill_call.json` had `gamma=NULL` for post norms; crashed in `rmsnorm_forward`.

5) Kept Qwen kernel preference on known-good behavior
- File: `version/v6.6/scripts/build_ir_v6_6.py:1479`
- Change: default `prefer_q8_activation` restored to true.

## Crash Progression (what changed)

Before fixes:
- Crash in `ck_model_init` while tokenizer loading.
- gdb frames showed:
  - `ck_tokenizer_add_token` from `libckernel_engine.so`
  - called by `ck_tokenizer_load_binary_with_scores` from tokenizer lib

After symbol + binding fixes:
- Gemma passes init and loads tokenizer:
  - `[TOKENIZER] Loaded 262144 tokens...`
- Then crashed in prefill `rmsnorm_forward` because post-norm gamma was NULL.

After gamma binding fix:
- Gemma run succeeds end-to-end (RC=0).
- Example output now (still wrong quality):
  - Prompt: `Hello`
  - Response: `偶لسللسللسللسللسللسللسل`

## Current Quality Status
- Gemma: runs, but divergent/garbled output.
- Qwen2 sanity still coherent (post-fixes):
  - `Hello -> ", We just got a new"`
- Qwen3 sanity still coherent (post-fixes):
  - `Hello -> "Navigation ..."`

## Commands Used Successfully

Gemma run (now executes, no segfault):
```bash
python version/v6.6/scripts/ck_run_v6_6.py run \
  /home/antshiv/.cache/ck-engine-v6.6/models/unsloth--gemma-3-270m-it-GGUF/gemma-3-270m-it-Q5_K_M.gguf \
  --prompt "Hello" --max-tokens 8 --context-len 100 --force-compile
```

Qwen sanity checks:
```bash
python version/v6.6/scripts/ck_run_v6_6.py run \
  /home/antshiv/.cache/ck-engine-v6.6/models/Qwen--Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q4_k_m.gguf \
  --prompt "Hello" --max-tokens 8 --context-len 1024

python version/v6.6/scripts/ck_run_v6_6.py run \
  /home/antshiv/.cache/ck-engine-v6.6/models/Qwen--Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf \
  --prompt "Hello" --max-tokens 8 --context-len 1024
```

## Remaining Immediate Task
Find first numerical divergence now that runtime is stable.

Script path:
- `version/v6.6/scripts/parity/run_gemma_first_divergence.sh`

Blocked by permissions when cleaning dump dirs under `~/.cache`.
Need elevated run or manual cleanup first.

## Next Session: Exact Steps

1) Run divergence script with permission to clean/write dumps:
```bash
bash version/v6.6/scripts/parity/run_gemma_first_divergence.sh
```

2) If script fails on file permissions, clean manually (with needed perms):
```bash
rm -rf ~/.cache/ck-engine-v6.6/models/unsloth--gemma-3-270m-it-GGUF/ck_build/ck_parity_dumps
rm -rf ~/.cache/ck-engine-v6.6/models/unsloth--gemma-3-270m-it-GGUF/ck_build/llama_parity_dumps
```

3) After first divergent op is known, patch only that op contract/kernel map and rerun:
```bash
python version/v6.6/scripts/ck_run_v6_6.py run \
  /home/antshiv/.cache/ck-engine-v6.6/models/unsloth--gemma-3-270m-it-GGUF/gemma-3-270m-it-Q5_K_M.gguf \
  --prompt "Hello" --max-tokens 8 --context-len 100 --force-compile
```

## Notes
- There is still an OpenMP runtime warning (`OMP: Warning #179`) but it is non-fatal in current runs.
- Main tracked files modified in working tree include:
  - `Makefile`
  - `include/tokenizer/tokenizer.h`
  - `src/tokenizer/tokenizer.c`
  - `version/v6.6/kernel_maps/kernel_bindings.json`
  - `version/v6.6/scripts/build_ir_v6_6.py`
  - `version/v6.6/scripts/ck_run_v6_6.py`

## Update: 2026-02-08 (Later Session)

### What Changed Since Previous Note

1) `run_gemma_first_divergence.sh` is stale against current `ck_run_v6_6.py` CLI.
- Script still passes removed args like `--detailed-llamacpp-parity` and `--llama-layer`.
- Running it currently fails with `unrecognized arguments`.

2) Found and fixed a major false-diagnostic issue in parity scripts.
- Root cause: several scripts computed `CK_STOP_OP` indices from `lowered_*` IR op lists, but runtime stop-op indexing follows the call-ready execution order (`lowered_*_call.operations`).
- Effect: checks were stopping at wrong operations and reporting large mismatches that were not real.

3) Patched parity scripts to use stop indices from `lowered_*_call` execution order.
- Files patched:
  - `version/v6.6/scripts/parity/check_attn_norm_contract_prefill.py`
  - `version/v6.6/scripts/parity/check_post_attn_chain_prefill.py`
  - `version/v6.6/scripts/parity/check_ffn_norm_contract.py`
  - `version/v6.6/scripts/parity/check_post_attn_chain.py`

### Re-Run Results After Stop-Index Fix

- `check_attn_norm_contract_prefill.py`: PASS (tiny diff, max ~3e-5)
- `check_post_attn_chain_prefill.py`: PASS (tiny diff, max ~7.6e-6; residual add exact)
- `check_ffn_norm_contract.py`: PASS (tiny diff, max ~2.8e-6)
- `check_post_attn_chain.py` (decode): PASS (tiny diff, max ~7.6e-6; residual add exact)

This means previously-reported giant mismatches in post-attention chain were false positives from wrong stop-op indexing.

### Current Gemma Status (after these fixes)

- Gemma still runs but text quality is garbled.
- Q/K/V contract checks for layer-0 decode+prefill are good.
- Post-attn chain and FFN norm checks are now good.
- So the remaining divergence is likely deeper in attention path details (layout/transpose/softmax behavior) or another downstream op not yet isolated by first-divergence tooling.

### New High-Value Suspect

In `build_ir_v6_6.py` (`generate_ir_lower_3`), ops like transpose placeholders are skipped (`continue`), and current kernel binding comments imply prefill codegen emits transpose loops itself. Need to verify generated C actually performs the required Q/K/V layout transforms for Gemma path.

### Immediate Next Steps

1) Patch/refresh `version/v6.6/scripts/parity/run_gemma_first_divergence.sh` for current `ck_run_v6_6.py` args.
2) Re-run first-divergence end-to-end with fresh dumps.
3) If first failing op lands in attention, instrument generated prefill attention path to validate Q/K/V memory layout and softmax scaling/ordering directly against captured tensors.
4) Keep Qwen2/Qwen3 hello sanity after each patch to prevent regressions.


## Update: 2026-02-08 (Current Session - Transpose + Tokenizer)

### 1) Confirmed obvious generated-code bug and fixed it

Root cause found in codegen pipeline:
- `lowered_prefill.json` contained transpose ops (`transpose_qkv_to_head_major`, `transpose_kv_to_head_major`, `transpose_attn_out_to_token_major`),
- but `generate_ir_lower_3` dropped them before `lowered_prefill_call.json`,
- so `codegen_prefill_v6_6.py` never emitted the actual transpose loops.

This meant prefill attention could consume token-major Q/K/V as if head-major.

Patch:
- `version/v6.6/scripts/build_ir_v6_6.py`
  - keep transpose placeholder ops in IR Lower 3 call output (do not skip).

Verification:
- regenerated Gemma C now includes transpose loops in prefill (`transpose_q_to_head_major`, `transpose_k_to_head_major`, `transpose_v_to_head_major`, `transpose_attn_out_to_token_major`).

### 2) Found tokenizer mismatch vs llama.cpp and fixed it

Observed mismatch (before fix):
- CK tokenizer ids for `Hello`: `[2, 26352]`
- llama-tokenize ids for `Hello`: `[2, 9259]`

Debug (`CK_DEBUG_SPM_ENCODE=1`) showed CK preprocessed `Hello` to `▁Hello`.
That indicates wrong SPM space-prefix behavior for Gemma.

Patch:
- `version/v6.6/scripts/build_ir_v6_6.py` in `_generate_tokenizer_c_code(..., tokenizer_type='sentencepiece')`
  - default `add_space_prefix` now derived when metadata is missing:
    - llama-style SPM => `true`
    - unigram/non-llama SPM (Gemma) => `false`

After rebuilding:
- CK preprocessed text is now `Hello` (no forced leading ▁)
- CK tokens now match llama-tokenize:
  - CK: `[2, 9259]`
  - llama-tokenize: `[2, 9259]`

### 3) Fixed hidden compile-fallback pitfall in tokenizer init codegen

While rebuilding, found compile failures due undefined vocab macros in generated tokenizer init:
- `W_VOCAB_SCORES`
- `W_VOCAB_TYPES`

This caused `ck_run_v6_6.py` compile step to fail and silently keep stale `libmodel.so`.

Patch:
- `version/v6.6/scripts/build_ir_v6_6.py` tokenizer init now uses:
  
## Update: 2026-02-08 (Embed-Scale Codegen + Prefill Coherence)

### What was fixed in this session

1) Moved Gemma embedding scaling into generator code (no post-patch script needed)
- Added template flag:
  - `version/v6.6/templates/gemma3.json`
  - `flags.scale_embeddings_sqrt_dim = true`
- IR propagation + backward-compatible fallback for older Gemma manifests:
  - `version/v6.6/scripts/build_ir_v6_6.py`
  - sets `config["scale_embeddings_sqrt_dim"]`
  - defaults ON for Gemma family if flag missing
- Decode emission:
  - `version/v6.6/scripts/codegen_v6_6.py`
  - emits `inp_embd * sqrt(EMBED_DIM)` right after embedding op, before layer-0 residual save
- Prefill emission:
  - `version/v6.6/scripts/codegen_prefill_v6_6.py`
  - emits batched scale `num_tokens * EMBED_DIM` at same semantic point

2) Deterministic-run bug fix
- `version/v6.6/scripts/ck_run_v6_6.py`
- `--temperature 0` now forwards correctly (`if args.temperature is not None`)

### Verification performed

1) Regenerated Gemma (`context_len=100`) and checked generated C
- `model_v6_6.c` now contains `Gemma embedding contract` blocks in both:
  - `ck_decode` after embedding op 0
  - `ck_prefill` after embedding op 0
- `inp_scaled` parity dump label is emitted in both paths.

2) Gemma 1-token coherence check
- Command:
  - `python version/v6.6/scripts/ck_run_v6_6.py run ~/.cache/ck-engine-v6.6/models/unsloth--gemma-3-270m-it-GGUF/gemma-3-270m-it-Q5_K_M.gguf --context-len 100 --prompt "Hello" --max-tokens 1 --temperature 0`
- Result:
  - `Response: !`
- Additional sanity:
  - Prompt `The sky is` -> `Response:  a`

3) Focused Gemma prefill parity contracts (layer 0) all pass
- `check_attn_norm_contract_prefill.py` -> tiny diff (`max ~9.15e-05`)
- `check_post_attn_chain_prefill.py` -> exact/tiny (`max 0` for post-attn norm and residual add)
- `check_layer0_qkv_contract.py --skip-decode` -> PASS against llama helper refs

4) Qwen non-regression sanity
- Qwen2 and Qwen3 still run with `Embed sqrt(dim) scale: OFF` during IR build.
- 1-token smoke responses complete without garbled byte output.

### Known script issue still open

- `version/v6.6/scripts/parity/run_gemma_first_divergence.sh` is stale:
  - uses removed `ck_run_v6_6.py` flags (`--detailed-llamacpp-parity`, `--llama-*`)
  - currently fails until script is updated.

### Immediate next steps

1) Update `run_gemma_first_divergence.sh` to current `ck_run_v6_6.py` CLI.
2) Re-run full first-divergence after this embed-scale fix to identify next true mismatch (if any).
3) If mismatch remains, isolate first failing tensor after `sa_out` with a fresh dump path.
  - `ck_tokenizer_load_binary_with_scores(...)` only when both macros are defined,
  - otherwise falls back to `ck_tokenizer_load_binary(...)`.

Result:
- `libmodel.so` now updates successfully (mtime newer than generated `model_v6_6.c`).

### 4) Current runtime status

- Qwen2/Qwen3 still coherent after transpose fix:
  - Qwen2 `Hello` -> `", I'm a 22 year"`
  - Qwen3 `Hello` -> `"est\n\n# Question\n\n# We can"`

- Gemma state is mixed by cache build dir:

A) `/home/antshiv/.cache/ck-engine-v6.6/models/gemma-3-270m-it-Q5_K_M`
- Runs without segfault.
- Tokenization matches (`[2, 9259]`).
- Output still garbled (`速لسل...`).

B) `/home/antshiv/.cache/ck-engine-v6.6/models/unsloth--gemma-3-270m-it-GGUF`
- After latest rebuild, tokenization now matches (`[2, 9259]`).
- But `ck_chat.py` currently segfaults in prefill (`rmsnorm_forward` frame in backtrace).
- Need to treat this as a separate stability issue in that cache artifact path.

### 5) Parity checks after fixes

On stable Gemma build dir (`gemma-3-270m-it-Q5_K_M`):
- `check_attn_norm_contract_prefill.py` PASS (tiny diff)
- `check_post_attn_chain_prefill.py` PASS (tiny diff)
- `check_ffn_norm_contract.py` PASS (exact/tiny)

So major remaining quality issue likely sits in attention/value path details not covered by those norm/residual contract checks.

### 6) Remaining high-priority next steps

1. Unify on one Gemma cache dir (prefer stable non-segfault path) to avoid split diagnostics.
2. Add/repair first-divergence runner to current CLI (script still stale on removed args).
3. Add a direct attention output parity check vs llama helper (post-rope/pre-out-proj), not only norm/residual checks.
4. Investigate and fix unsloth cache prefill segfault separately once stable path parity is resolved.


---

## Session Update (2026-02-08T20:00:23Z)

### What is confirmed now
- Handoff file is current and contains the key Gemma fixes already applied.
- `build_ir_v6_6.py` still contains all three critical patches:
  - preserve prefill transpose ops in IR lower-3 output
  - explicit `add_space_prefix` default by tokenizer model family
  - compile-guarded tokenizer init for `W_VOCAB_SCORES`/`W_VOCAB_TYPES`
- Repo is not clean; there are many modified/untracked files, so keep narrow scope on Gemma path and avoid broad refactors.

### Git snapshot for resume
- `HEAD`: `6a2da9e` (`Updaet fixes and incldues new kernel dtype`)
- Recent relevant commits:
  - `ac4eb6a` Add Gemma kernels and tokenizer SPM parity tests
  - `58c574d` Updated ir to work with Qwen2/Qwen3
- Available stash entries include:
  - `stash@{0}` WIP on `6a2da9e`
  - `stash@{1}` WIP on `cf80032`

### Current blocking behavior
- Gemma tokenizer parity for `Hello` is fixed (token ids now align with llama-tokenize in patched path).
- Remaining blocker is generation quality and/or stability depending on cache artifact:
  - one Gemma cache path runs but outputs garbled text
  - another Gemma cache path can segfault in prefill (`rmsnorm_forward` stack frame observed)

### Fastest next-session execution plan
1. Pick one Gemma cache artifact path and stick to it for the entire session.
2. Rebuild once and verify resulting `libmodel.so` mtime is newer than `model_v6_6.c`.
3. Run a minimal deterministic prompt (`Hello`, fixed seed/temp) and record token ids + first 16 output tokens.
4. Run first-divergence/parity at attention output boundary (post-rope and pre-out-proj), not only norm/residual checks.
5. Fix the first mismatched tensor/op only; rerun the same tiny prompt after each patch.

### Guardrails to avoid regression
- Do not re-open broad Qwen template/tokenizer churn while debugging Gemma.
- Keep Gemma-specific behavior behind template/flags (no silent cross-family behavior changes).
- Any change touching shared runtime (`prefill`, `decode`, tokenizer core) should immediately smoke-test Qwen2 and Qwen3 with `Hello`.

### Handy resume commands
```bash
# Inspect current state
cd /home/antshiv/Workspace/C-Kernel-Engine
git status --short
git stash list | head -n 10

# Confirm the three key codegen patches are present
rg -n "add_space_prefix|W_VOCAB_SCORES|transpose_.*head_major" version/v6.6/scripts/build_ir_v6_6.py

# Re-run minimal model path (example)
python version/v6.6/scripts/ck_run_v6_6.py run \
  hf://unsloth/gemma-3-270m-it-GGUF/gemma-3-270m-it-Q5_K_M.gguf \
  --prompt "Hello" --max-tokens 16 --ctx-size 100
```

---

## Session Update (2026-02-08T20:08:10Z)

### New smoke results (`context-len=100`, prompt=`Hello`)
- Qwen2 (`qwen2-0_5b-instruct-q4_k_m.gguf`): coherent.
  - Response sample: `, Everyone!\n\nI'm a new user on AoPS...`
  - Decode speed: ~`23.96 tok/s`
- Qwen3 (`Qwen3-0.6B-Q8_0.gguf`): coherent.
  - Response sample: `Name, I'm trying to do a simulation...`
  - Decode speed: ~`10.93 tok/s`
- Gemma (`gemma-3-270m-it-Q5_K_M.gguf`): still garbled.
  - Response sample still mixed bytes/non-Latin (`<0xE8>...`).
  - Decode speed: ~`11.8-12.2 tok/s`

### New confirmed generated-code state
- Active Gemma generated C now has:
  - transpose ops present in prefill
  - `add_space_prefix = false`
  - `spm_mode = CK_SPM_MODE_UNIGRAM` (now fixed)
- So tokenizer mode was a real bug, but not the only root cause of garbled generation.

### Patch made in this session (codegen)
- File: `version/v6.6/scripts/build_ir_v6_6.py`
- Change:
  - `_generate_tokenizer_c_code(...)` now accepts model hints and treats Gemma-family SPM as unigram even when GGUF metadata reports `tokenizer_model=llama`.
  - call site now passes `template.get("name")` for family detection (`gemma3`).
- Status:
  - Python syntax check passed (`python -m py_compile ...`).

### Additional diagnostic hook added
- File: `src/kernels/attention_kernels_sliding.c`
- Added env-gated escape hatch:
  - `CK_FORCE_NONSLIDING_ATTN=1` routes sliding attention entry points to non-sliding flash kernels.
- Purpose:
  - Quick A/B isolation: determine whether sliding kernel path is the remaining Gemma divergence source.

### Current blocker for testing that hook
- Full `make -j4` failed in current tree with unrelated quant compile errors (e.g. `block_q8_1` / `block_q5_K` type resolution in q5 kernels).
- Because of this, updated `attention_kernels_sliding.c` has not yet been validated at runtime.

### Immediate next step (high value)
1. Restore a buildable `libckernel_engine.so` state (or compile a minimal override for attention symbols).
2. Run Gemma twice on same prompt:
   - baseline
   - `CK_FORCE_NONSLIDING_ATTN=1`
3. If output improves/coheres only with forced non-sliding:
   - root cause is in sliding kernels; focus there.
4. If no change:
   - root cause is elsewhere (likely attention value/QKV contract or post-attention path).

---

## Session Update (2026-02-08T21:13:41Z)

### Goal addressed
Enable llama parity testing again from the current clean-ish Qwen-working baseline and verify the parity pipeline is operational for Gemma prefill.

### What was patched (from stash to working tree)
1. `version/v6.6/scripts/ck_run_v6_6.py`
- Restored parity/detailed llama parity CLI wiring:
  - `--parity-dump`
  - `--detailed-llamacpp-parity`
  - `--llama-filter`
  - `--llama-layer`
  - `--llama-stop-after`
  - `--llama-include-global`
  - `--llama-timeout`
- Restored parity-aware execution path:
  - codegen receives `--parity-dump`
  - compile adds `-DCK_PARITY_DUMP` when needed
  - CK dump generation step (`ck-cli-v6.6`)
  - llama reference dump generation step (`build/llama-parity` fallback `llama.cpp/main`)
- `--detailed-llamacpp-parity` now implies parity dump mode and runs dump producers instead of regular chat flow.

2. `version/v6.6/scripts/parity/run_gemma_first_divergence.sh`
- Updated call to `parity_test.py` to current CLI:
  - now uses `--ck-dump` and `--ref-dump`
  - removed stale legacy args (`--model gemma --pass prefill`)
- Increased default llama timeout budget:
  - `LLAMA_TIMEOUT_BASE=120`
- Fixed default model/workdir resolution for current cache behavior:
  - prefers local cached Gemma GGUF when present
  - computes work dir same way as current `ck_run_v6_6.py`
  - removes stale assumptions around `/ck_build` layout

### Validation done
- `python -m py_compile version/v6.6/scripts/ck_run_v6_6.py` passed.
- `python version/v6.6/scripts/ck_run_v6_6.py run --help` now includes the restored parity flags.
- Direct detailed parity run successfully produced both dump families:
  - `~/.cache/ck-engine-v6.6/models/gemma-3-270m-it-Q5_K_M/ck_parity_dumps/dump.bin`
  - `~/.cache/ck-engine-v6.6/models/gemma-3-270m-it-Q5_K_M/llama_parity_dumps/dump.bin`
  - `~/.cache/ck-engine-v6.6/models/gemma-3-270m-it-Q5_K_M/llama_parity_dumps/index.json`

### Current blocking failure
- `version/v6.6/scripts/parity_test.py` currently fails on produced dumps with:
  - `struct.error: unpack requires a buffer of 120 bytes`
- Notably both dump files have CKDMP magic (`CKDMP\0\0\0`) and appear non-empty/real.
- This points to a format reader mismatch (record/header size expectations) more than dump generation being absent.

### Why this is still useful
- Llama parity infra is now re-enabled end-to-end at script level.
- We are no longer blocked by missing flags / missing dumps.
- The narrow remaining blocker is the parser/format contract in `parity_test.py`.

### Immediate next-session task (narrow)
1. Patch `version/v6.6/scripts/parity_test.py` to tolerate/handle current CKDMP header/record variant.
2. Re-run:
   - `bash version/v6.6/scripts/parity/run_gemma_first_divergence.sh`
3. Once parser runs, capture first tensor divergence and then target only that Gemma prefill op.

### Useful resume commands
```bash
cd /home/antshiv/Workspace/C-Kernel-Engine

git status --short

python -m py_compile version/v6.6/scripts/ck_run_v6_6.py
python version/v6.6/scripts/ck_run_v6_6.py run --help | rg -n "detailed-llamacpp-parity|llama-layer|llama-filter|parity-dump"

bash version/v6.6/scripts/parity/run_gemma_first_divergence.sh
```

---

## Session Update (2026-02-08T21:18:22Z)

### Parity parser root cause fixed
- File patched: `version/v6.6/scripts/parity_test.py`
- Root cause of crash:
  - Script attempted to unpack a 120-byte struct from 128 bytes (`struct.error: unpack requires a buffer of 120 bytes`).
  - Also had a 7-byte magic constant while dumps use 8-byte magic (`CKDMP\0\0\0`).
- Additional blocker fixed in same file:
  - typo `if notck_dump:` -> `if not ck_dump:`

### What changed in parser
- Added multi-format CKDMP header support with auto-detection:
  - rank_120 (`<8sIi32sII4qIi24x`)
  - rank_124 (`<8sIi32sII4qIi28x`)
  - rank_dump_type_128 (`<8sIi32sII4qIiI28x`)
  - legacy_no_rank_124 (`<8sIi32sI4qIQi24x`)
- Parser now probes up to max header size and rewinds to actual detected header length before reading tensor bytes.
- Added sanity checks for plausible headers and safer shape handling (fallback flatten on reshape mismatch).

### Validation
- `python -m py_compile version/v6.6/scripts/parity_test.py` passed.
- Direct comparison now runs instead of crashing:
  - `python version/v6.6/scripts/parity_test.py --ck-dump ~/.cache/ck-engine-v6.6/models/gemma-3-270m-it-Q5_K_M/ck_parity_dumps/dump.bin --ref-dump ~/.cache/ck-engine-v6.6/models/gemma-3-270m-it-Q5_K_M/llama_parity_dumps/dump.bin`
- Current output shows parity table (no parser exception), though many entries are still `MISSING` because tensor naming/filter coverage alignment remains incomplete.

### Remaining narrow issue after parser fix
- Reference dump parse currently warns at one later offset (`Could not decode header at offset 55808`).
- This no longer blocks parity execution; next improvement is optional resync scan + tighter name mapping/filter alignment so first real FAIL is surfaced earlier.
- Follow-up fix in same file: quiet mode now works (`--quiet` no longer crashes with `UnboundLocalError` on summary counters).

---

## Session Update (2026-02-08T21:23:34Z)

### Follow-up parity hardening done (`parity_test.py`)
- Added dump parser resync behavior:
  - on bad header, scan forward to next CKDMP magic and continue parsing instead of hard stop.
- Added robust element-size inference:
  - choose payload size (4/2/1 bytes) using look-ahead alignment checks.
  - fixes misalignment when header dtype is inconsistent with actual payload bytes.
- Added operation normalization/aliasing for better cross-runtime key matching:
  - strips suffix annotations (`" (reshaped)"`, `" (copy of ...)"`)
  - strips layer suffix (`-0`) and preserves/infers layer id
  - maps llama-style names to canonical names (e.g., `Qcur -> q_proj`, `Kcur -> k_proj`, `Vcur -> v_proj`, `inp_embd -> token_embedding`).

### Concrete issue found in current llama dump
- At offset `54656`, one record (`k_cache_view-0 (copy of Kcur-0)`) advertises fp16 dtype but is laid out like fp32 payload.
- New inference logic handles this and keeps stream alignment.

### Result after hardening
- Reference dump now parses to `38` tensors (was `12` before parse abort).
- CK dump currently still has only `1` tensor (`inp_scaled`) in this run.
- So parity script now executes reliably, but current comparison remains mostly `MISSING` because CK dump coverage is too narrow in this artifact.

---

## Session Update (2026-02-08T22:08:00Z)

### Decode threadpool gap fixed for Gemma kernels
- Files patched:
  - `version/v6.6/src/ck_parallel_decode.h`
  - `version/v6.6/src/ck_parallel_decode.c`
- Root cause:
  - `CK_PARALLEL_DECODE` only redirected q5_0/q4_k/q6_k/q8_0 GEMV paths.
  - Gemma decode IR uses `gemv_q5_1_q8_1`, `gemv_q5_k`, and `gemv_q8_0_q8_0_contract`, so these stayed effectively serial.

### What was added
- New decode dispatch APIs + macro redirects:
  - `gemv_q5_1_q8_1_parallel_dispatch(...)`
  - `gemv_q5_k_parallel_dispatch(...)`
  - `gemv_q8_0_q8_0_contract_parallel_dispatch(...)`
- Added worker functions for q5_1/q5_k that split output rows across threadpool workers.
- Added contract wrapper for q8_0 logits path:
  - quantizes FP32 activation to Q8_0 once
  - reuses existing `gemv_q8_0_q8_0_parallel_dispatch(...)`
- Included a local `Q5_K` row layout struct in `ck_parallel_decode.c` to compute row byte strides safely for row-split offsets.

### Validation run results
- Gemma (`gemma-3-270m-it-Q5_K_M.gguf`) with force compile:
  - command succeeded, coherent text output, no segfault.
  - decode IR still shows expected ops (`gemv_q5_1_q8_1`, `gemv_q5_k`, `gemv_q8_0_q8_0_contract`) now routed by decode macro layer.
- Qwen2 and Qwen3 regression smokes (both force compile) passed:
  - `Qwen2-0.5B-Instruct` generated coherent output.
  - `Qwen3-0.6B-Q8_0` generated coherent output.

### Quick throughput check (Gemma, local GGUF)
- `CK_NUM_THREADS=1`: decode ~`126.33 ms/tok` (`7.92 tok/s`)
- `CK_NUM_THREADS=4`: decode ~`96.38 ms/tok` (`10.38 tok/s`)
- Indicates decode threadpool path now contributes for Gemma decode kernels.

### Notes / next narrow step
- Remaining speed headroom likely in kernel-internal SIMD for q5_1/q5_k paths (current row-split dispatch works but still calls existing serial kernel bodies per shard).
- If needed next: add dedicated `_parallel_simd` implementations for q5_1/q5_k to reduce per-thread quantization overhead.
