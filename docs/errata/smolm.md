• Added a SmolLM per‑stage diff harness so you can pinpoint whether the first divergence happens in RoPE, attention, or later
  MLP/projection stages. It runs the C layer kernel on HF weights and compares to a pure PyTorch reference for the same layer,
  so you get a clean stage‑by‑stage diff without going through the bump converter. I also wired a Make target for it.

  What changed

  - New script scripts/smollm_layer_stage_diff.py: runs one Llama/SmolLM layer, dumps per‑stage diffs (ln1, q/k/v, scores,
    attn_out, proj, ln2, swiglu, mlp, output).
  - Makefile target smollm-layer-diff + variables SMOLLM_LAYER, SMOLLM_STAGE_TOL, SMOLLM_STAGE_DUMP.

  How to use it (this answers “1–5 tokens” + bug hunting)

  - Build a safe lib for your CPU if needed:
    make build/libckernel_engine.so AVX_FLAGS=
  - Run per‑stage diff on layer 0 with a short prompt:

  What to look for

  - If proj is the first jump: output projection (wo packing).
  - If attention path is clean but fc1_out/swiglu/mlp_out jump: MLP packing/activation.

  Note

  - This script uses HF weights directly (no bump conversion). If this passes but full model parity still fails, the bug is
    likely in the HF→bump converter or runtime weight mapping, not in the kernels.

  Next steps (pick one)

  1. Run the new smollm-layer-diff and share the first stage that spikes.
  2. If layer 0 is clean, run with SMOLLM_LAYER=1 to see where drift starts.
  3. If both are clean, I’ll add a “bump‑weights” path to isolate the converter/layout.


› can you explain all the cnges you ahve made or tested. i cant keept rack now. and what are we having issue with? and how we
  are emthodcialy testing ehre now?


• Here’s a clean, current picture of what I changed, what’s failing, and how we’re testing.

  What I changed (recent)

  - Added scripts/smollm_layer_stage_diff.py
    Purpose: per‑stage diffs for one SmolLM layer against a PyTorch reference. It runs the C layer kernel (rmsnorm → qkv →
    rope → attention → proj → rmsnorm → swiglu → mlp → output) and prints diffs for each stage. This lets us pinpoint whether
    the divergence starts in RoPE, attention, proj, or MLP.
  - Added Make target smollm-layer-diff + vars in Makefile
    New vars: SMOLLM_LAYER, SMOLLM_STAGE_TOL, SMOLLM_STAGE_DUMP
    New target: make smollm-layer-diff ... (runs the new script).

  What we are seeing (issue)

  - Full SmolLM forward parity vs PyTorch diverges at later tokens (context‑dependent).
  - In earlier tests the top‑1 predictions match for first few positions, then diverge (ex: position 3–5) with non‑tiny logit
    diffs.
  - That points to a logic/layout bug, not just SIMD drift. Most likely in:
      - RoPE application, or
      - attention path (scores/softmax/weighted sum), or
      - output projection packing.

  Methodical testing plan (current)
  We’re now isolating the failure in layers, not whole‑model:

  1. Per‑layer, per‑stage diff (new script)
      - Command:
        make smollm-layer-diff SMOLLM_CONTEXT=5 SMOLLM_TEXT="Hello world" SMOLLM_LAYER=0 SMOLLM_STAGE_DUMP=1
      - Output shows max diff at each stage.
        The first stage that jumps tells us where the bug starts:
          - If q/k diverge after RoPE → RoPE bug.
          - If scores/attn_out diverge → attention bug.
          - If proj diverges while attn_out is clean → projection/packing bug.
          - If attention clean but mlp diverges → MLP/packing bug.
  If you want, I can also:

  1. add a context sweep (ctx=1..5) to show exactly where top‑1 changes
  2. add a bump‑weights parity path to isolate HF→bump converter issues


RESOLVED: RoPE Rotation Convention Bug (2024-12)

  Root cause

  - The C RoPE kernel used the wrong rotation convention.
  - Original code: even/odd interleaved pairs - (x[0],x[1]), (x[2],x[3]), ...
    This applies cos/sin to adjacent pairs.
  - HF Llama/SmolLM: rotate-half - first half with second half
    This applies cos/sin to (x[0..half-1]) with (x[half..dim-1]).

  What was changed

  - Updated src/kernels/rope_kernels.c to use rotate-half convention:
    Before:  x0 = x_row[2*i],     x1 = x_row[2*i + 1]
    After:   x0 = x_row[i],       x1 = x_row[i + half_dim]

  - Forward and backward passes both updated.

  Verification

  - make smollm-forward now passes with all positions [OK]
  - Logit diffs down to ~1e-4 (numerical precision)
  - Top-1 predictions match PyTorch for all context positions

  Why per-layer tests didn't catch it

  - The layer stage diff script (smollm_layer_stage_diff.py) used a Python reference
    that matched our C kernel's interleaved convention, not HF's rotate-half.
  - So per-layer tests "passed" but full-model parity failed because our reference
    was wrong, not just the C kernel.

  Errata summary (requested)

  What was wrong
  - The full-model mismatch was caused by RoPE rotation convention.
  - Our RoPE was even/odd interleaved (x0,x1)(x2,x3)...
    HF Llama/SmolLM uses rotate-half (first half <-> second half).
  - This made per-layer manual refs "agree" with our C kernel but disagree with HF,
    causing large drift in full-model parity.

  What I changed
  - Switched RoPE kernels to Llama rotate-half:
      - src/kernels/rope_kernels.c
  - Updated all Python references to match rotate-half:
      - unittest/test_rope.py
      - unittest/test_orchestration_layer.py
      - scripts/smollm_layer_stage_diff.py
