  v6.6 + llama.cpp Parity Debugging Guide

  Quick Start

  # Gemma-3 minimal parity test
  bash version/v6.6/scripts/run_gemma_parity_min.sh

  # Full CK + llama.cpp dumps and compare
  python version/v6.6/scripts/parity_test.py --ck-dump ck.bin --ref-dump llama.bin --model gemma

  Workflow: Finding Divergence

  Step 1: Generate Dumps

  # Clean build directory
  rm -rf ~/.cache/ck-engine-v6.6/models/unsloth--gemma-3-270m-it-GGUF/ck_build

  # Generate both CK and llama.cpp dumps
  python version/v6.6/scripts/ck_run_v6_6.py run \
    hf://unsloth/gemma-3-270m-it-GGUF/gemma-3-270m-it-Q5_K_M.gguf \
    --context-len 256 \
    --force-compile \
    --detailed-llamacpp-parity \
    --prompt "Hello" \
    --max-tokens 1 \
    --llama-layer 0 \
    --llama-include-global \
    --llama-filter token_embd,attn_norm \
    --llama-stop-after 2 \
    --llama-timeout 0

  Step 2: Compare Dumps

  python version/v6.6/scripts/parity_test.py \
    --ck-dump ~/.cache/ck-engine-v6.6/models/unsloth--gemma-3-270m-it-GGUF/ck_build/ck_parity_dumps/dump.bin \
    --ref-dump ~/.cache/ck-engine-v6.6/models/unsloth--gemma-3-270m-it-GGUF/ck_build/llama_parity_dumps/dump.bin \
    --model gemma --pass prefill | rg "token_embd|attn_norm|FAIL|ERROR"

  Step 3: Manual llama.cpp Dump (if needed)

  cd ~/.cache/ck-engine-v6.6/models/unsloth--gemma-3-270m-it-GGUF/ck_build

  CKDMP_DIR=./llama_parity_dumps \
  CKDMP_FILTER=token_embd,attn_norm \
  CKDMP_LAYER=0 \
  CKDMP_INCLUDE_GLOBAL=1 \
  CKDMP_STOP_AFTER=2 \
  ./llama.cpp/main \
    -m ./gemma-3-270m-it-Q5_K_M.gguf \
    -p "Hello" -n 0 --ctx-size 256 --temp 0

  ---
  ck_run_v6_6.py Commands

  # Force recompile
  python version/v6.6/scripts/ck_run_v6_6.py run hf://...gemma... --context-len 1024 --force-compile

  # With llama.cpp dumps
  python version/v6.6/scripts/ck_run_v6_6.py run hf://... \
    --detailed-llamacpp-parity \
    --llama-layer 0 \
    --llama-filter token_embd,attn_norm,Qcur,Kcur,Vcur,attn_out,ffn_norm,ffn_gate,ffn_up,ffn_down

  parity_test.py Commands

  # Basic compare
  python version/v6.6/scripts/parity_test.py --ck-dump ck.bin --ref-dump llama.bin --model gemma

  # With tolerance
  python version/v6.6/scripts/parity_test.py --ck-dump ck.bin --ref-dump llama.bin --atol 1e-3

  # Prefill only
  python version/v6.6/scripts/parity_test.py --ck-dump ck.bin --ref-dump llama.bin --model gemma --pass prefill

  ---
  Environment Variables

  CK Runtime
  ┌───────────────────────┬─────────────────┐
  │       Variable        │     Purpose     │
  ├───────────────────────┼─────────────────┤
  │ CK_DUMP_ACTIVATIONS=1 │ Enable dumps    │
  ├───────────────────────┼─────────────────┤
  │ CK_DUMP_DIR=path      │ Dump directory  │
  ├───────────────────────┼─────────────────┤
  │ CK_STOP_OP=N          │ Stop after op N │
  └───────────────────────┴─────────────────┘
  llama.cpp
  ┌─────────────────────────────┬──────────────────────┐
  │          Variable           │       Purpose        │
  ├─────────────────────────────┼──────────────────────┤
  │ CKDMP_DIR=path              │ llama.cpp dump dir   │
  ├─────────────────────────────┼──────────────────────┤
  │ CKDMP_FILTER=token_embd,... │ Which tensors        │
  ├─────────────────────────────┼──────────────────────┤
  │ CKDMP_LAYER=N               │ Layer (-1 for embed) │
  ├─────────────────────────────┼──────────────────────┤
  │ CKDMP_INCLUDE_GLOBAL=1      │ Include globals      │
  └─────────────────────────────┴──────────────────────┘
  Debug
  ┌────────────────────────┬──────────────────┐
  │        Variable        │     Purpose      │
  ├────────────────────────┼──────────────────┤
  │ CK_SKIP_SPM_SPECIALS=1 │ Skip SPM special │
  ├────────────────────────┼──────────────────┤
  │ CK_DEBUG_SPM_ENCODE=1  │ Debug SPM encode │
  └────────────────────────┴──────────────────┘
  ---
  CK CLI Direct Testing

  # With dumps
  CK_DUMP_ACTIVATIONS=1 \
  CK_DUMP_DIR=~/.cache/ck-engine-v6.6/models/unsloth--gemma-3-270m-it-GGUF/ck_build/ck_parity_dumps \
  CK_STOP_OP=26 \
  ./build/ck-cli-v6.6 \
    --lib ~/.cache/ck-engine-v6.6/models/unsloth--gemma-3-270m-it-GGUF/ck_build/libmodel.so \
    --weights ~/.cache/ck-engine-v6.6/models/unsloth--gemma-3-270m-it-GGUF/ck_build/weights.bump \
    --prompt "Hello" --max-tokens 1

  ---
  Common Tasks

  # Clean rebuild
  rm -rf ~/.cache/ck-engine-v6.6/models/unsloth--gemma-3-270m-it-GGUF/ck_build

  # Check dump exists
  ls -la ~/.cache/ck-engine-v6.6/models/unsloth--gemma-3-270m-it-GGUF/ck_build/ck_parity_dumps/dump.bin

  # Find op in generated code
  rg -n "CK_TRACE_OPS" ~/.cache/ck-engine-v6.6/models/unsloth--gemma-3-270m-it-GGUF/ck_build/model_v6_6.c

  # Quick filter
  python version/v6.6/scripts/ck_run_v6_6.py run ... 2>&1 | grep -E "completeness|HARD FAIL|Error"

  ---
  Gemma Filter Sets
  ┌─────────────┬────────────────────────────────────────────────────────────────────────────────┐
  │    Phase    │                                     Filter                                     │
  ├─────────────┼────────────────────────────────────────────────────────────────────────────────┤
  │ Embedding   │ token_embd                                                                     │
  ├─────────────┼────────────────────────────────────────────────────────────────────────────────┤
  │ Layer 0     │ attn_norm,Qcur,Kcur,Vcur,attn_out                                              │
  ├─────────────┼────────────────────────────────────────────────────────────────────────────────┤
  │ Layer 0 FFN │ ffn_norm,ffn_gate,ffn_up,ffn_down                                              │
  ├─────────────┼────────────────────────────────────────────────────────────────────────────────┤
  │ Combined    │ token_embd,attn_norm,Qcur,Kcur,Vcur,attn_out,ffn_norm,ffn_gate,ffn_up,ffn_down │
  └─────────────┴────────────────────────────────────────────────────────────────────────────────┘
  ---
  Shell Scripts

  # Minimal Gemma test
  bash version/v6.6/scripts/run_gemma_parity_min.sh

  # Find first divergence
  bash version/v6.6/scripts/parity/run_gemma_first_divergence.sh hf://... "Hello" 256 1

  ---
  Your Common Commands

  # 1. Full build
  python version/v6.6/scripts/ck_run_v6_6.py run hf://unsloth/gemma-3-270m-it-GGUF/gemma-3-270m-it-Q5_K_M.gguf --context-len 1024 --force-compile

  # 2. With parity dumps
  python version/v6.6/scripts/ck_run_v6_6.py run hf://... --detailed-llamacpp-parity --llama-layer 0 --llama-filter token_embd,attn_norm --llama-stop-after 2

  # 3. Compare
  python version/v6.6/scripts/parity_test.py --ck-dump ck.bin --ref-dump llama.bin --model gemma --pass prefill

  # 4. Or full test
  bash version/v6.6/scripts/run_gemma_parity_min.sh

  ---
  Model Cache Locations
  ┌──────────────┬─────────────────────────────────────────────────────────────────────────┐
  │    Model     │                                  Path                                   │
  ├──────────────┼─────────────────────────────────────────────────────────────────────────┤
  │ Gemma-3-270M │ ~/.cache/ck-engine-v6.6/models/unsloth--gemma-3-270m-it-GGUF/ck_build/  │
  ├──────────────┼─────────────────────────────────────────────────────────────────────────┤
  │ Qwen2-0.5B   │ ~/.cache/ck-engine-v6.6/models/Qwen--Qwen2-0.5B-Instruct-GGUF/ck_build/ │
  ├──────────────┼─────────────────────────────────────────────────────────────────────────┤
  │ Qwen3-0.6B   │ ~/.cache/ck-engine-v6.6/models/Qwen--Qwen3-0.6B-GGUF/ck_build/          │
  └──────────────┴─────────────────────────────────────────────────────────────────────────┘
