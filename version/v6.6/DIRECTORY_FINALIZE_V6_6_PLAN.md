



































































































































































   1895  python version/v6.6/scripts/ck_run_v6_6.py run hf://unsloth/gemma-3-270m-it-GGUF/gemma-3-270m-it-Q5_K_M.gguf     --force-compile     --prompt "Hello"     --llama-filter token_embd     --llama-stop-after 1     --llama-timeout 0                                
   1896  python version/v6.6/scripts/ck_run_v6_6.py run hf://unsloth/gemma-3-270m-it-GGUF/gemma-3-270m-it-Q5_K_M.gguf     --context-len 256     --force-compile     --prompt "Hello"     --llama-layer 0     --llama-include-global     --llama-filter                     
  token_embd,attn_norm     --llama-stop-after 3                                                                                                                                                                                                                            
   1897  CKDMP_DIR=~/.cache/ck-engine-v6.6/models/unsloth--gemma-3-270m-it-GGUF/ck_build/llama_parity_dumps   CKDMP_FILTER=token_embd,attn_norm   CKDMP_LAYER=0   CKDMP_INCLUDE_GLOBAL=1   CKDMP_STOP_AFTER=2   ./llama.cpp/main -m                                        
  ~/.cache/ck-engine-v6.6/models/unsloth--gemma-3-270m-it-GGUF/gemma-3-270m-it-Q5_K_M.gguf     -p "Hello" -n 0 --ctx-size 256 --temp 0                                                                                                                                     
   1898  python version/v6.6/scripts/ck_run_v6_6.py run hf://unsloth/gemma-3-270m-it-GGUF/gemma-3-270m-it-Q5_K_M.gguf     --force-compile     --detailed-llamacpp-parity     --prompt "Hello"     --max-tokens 0     --llama-layer 0     --llama-include-global            
  --llama-filter token_embd,attn_norm     --llama-timeout 0                                                                                                                                                                                                                
   1899  python version/v6.6/scripts/ck_run_v6_6.py run hf://unsloth/gemma-3-270m-it-GGUF/gemma-3-270m-it-Q5_K_M.gguf     --context-len 256     --force-compile     --detailed-llamacpp-parity     --prompt "Hello"     --max-tokens 1     --llama-layer 0                 
  --llama-include-global     --llama-filter token_embd,attn_norm     --llama-stop-after 2     --llama-timeout 0                                                                                                                                                            
   1900  python version/v6.6/scripts/parity_test.py     --ck-dump ~/.cache/ck-engine-v6.6/models/unsloth--gemma-3-270m-it-GGUF/ck_build/ck_parity_dumps/dump.bin     --ref-dump                                                                                            
  ~/.cache/ck-engine-v6.6/models/unsloth--gemma-3-270m-it-GGUF/ck_build/llama_parity_dumps/dump.bin     --model gemma --pass prefill | rg "token_embd|attn_norm|FAIL|ERROR"                                                                                                
   1901  python version/v6.6/scripts/ck_run_v6_6.py run hf://unsloth/gemma-3-270m-it-GGUF/gemma-3-270m-it-Q5_K_M.gguf     --context-len 256     --force-compile     --detailed-llamacpp-parity     --prompt "Hello"     --max-tokens 1     --llama-layer 0                 
  --llama-include-global     --llama-filter token_embd,attn_norm     --llama-stop-after 2     --llama-timeout 0                                                                                                                                                            
   1902  ls ~/.cache/ck-engine-v6.6/models/unsloth--gemma-3-270m-it-GGUF/ck_build/ck_parity_dumps/                                                                                                                                                                         
   1903  /home/antshiv/.cache/ck-engine-v6.6/models/unsloth--gemma-3-270m-it-GGUF/ck_build/llama_parity_dumps/dump.bin                                                                                                                                                     
   1904  file /home/antshiv/.cache/ck-engine-v6.6/models/unsloth--gemma-3-270m-it-GGUF/ck_build/llama_parity_dumps/dump.bin                                                                                                                                                
   1905  python version/v6.6/scripts/ck_run_v6_6.py run hf://unsloth/gemma-3-270m-it-GGUF/gemma-3-270m-it-Q5_K_M.gguf     --context-len 256     --force-compile     --detailed-llamacpp-parity     --prompt "Hello"     --max-tokens 1     --llama-layer 0                 
  --llama-include-global     --llama-filter token_embd,attn_norm     --llama-stop-after 2     --llama-timeout 0                                                                                                                                                            
   1906  ls ~/.cache/ck-engine-v6.6/models/unsloth--gemma-3-270m-it-GGUF/ck_build/ck_parity_dumps/dump.bin                                                                                                                                                                 
   1907  python version/v6.6/scripts/parity_test.py     --ck-dump ~/.cache/ck-engine-v6.6/models/unsloth--gemma-3-270m-it-GGUF/ck_build/ck_parity_dumps/dump.bin     --ref-dump                                                                                            
  ~/.cache/ck-engine-v6.6/models/unsloth--gemma-3-270m-it-GGUF/ck_build/llama_parity_dumps/dump.bin     --model gemma --pass prefill | rg "token_embd|attn_norm|FAIL|ERROR"                                                                                                
   1908  python version/v6.6/scripts/ck_run_v6_6.py run hf://unsloth/gemma-3-270m-it-GGUF/gemma-3-270m-it-Q5_K_M.gguf     --context-len 256     --force-compile     --detailed-llamacpp-parity     --prompt "Hello"     --max-tokens 1     --llama-layer 0                 
  --llama-include-global     --llama-filter token_embd,attn_norm     --llama-stop-after 2     --llama-timeout 0                                                                                                                                                            
   1909  ls -l ~/.cache/ck-engine-v6.6/models/unsloth--gemma-3-270m-it-GGUF/ck_build/ck_parity_dumps/dump.bin                                                                                                                                                              
   1910  python version/v6.6/scripts/parity_test.py     --ck-dump ~/.cache/ck-engine-v6.6/models/unsloth--gemma-3-270m-it-GGUF/ck_build/ck_parity_dumps/dump.bin     --ref-dump                                                                                            
  ~/.cache/ck-engine-v6.6/models/unsloth--gemma-3-270m-it-GGUF/ck_build/llama_parity_dumps/dump.bin     --model gemma --pass prefill | rg "token_embd|attn_norm|FAIL|ERROR"                                                                                                
   1911  python version/v6.6/scripts/ck_run_v6_6.py run hf://unsloth/gemma-3-270m-it-GGUF/gemma-3-270m-it-Q5_K_M.gguf     --detailed-llamacpp-parity     --max-tokens 1     --llama-include-global     --llama-filter token_embd,attn_norm     --llama-stop-after 2        
  --llama-timeout 0                                                                                                                                                                                                                                                        
   1918  python version/v6.6/scripts/parity_test.py     --ck-dump ~/.cache/ck-engine-v6.6/models/unsloth--gemma-3-270m-it-GGUF/ck_build/ck_parity_dumps/dump.bin     --ref-dump                                                                                            
  ~/.cache/ck-engine-v6.6/models/unsloth--gemma-3-270m-it-GGUF/ck_build/llama_parity_dumps/dump.bin     --model gemma --pass prefill | rg "token_embd|attn_norm|FAIL|ERROR"                                                                                                
   1919  python version/v6.6/scripts/ck_run_v6_6.py run hf://unsloth/gemma-3-270m-it-GGUF/gemma-3-270m-it-Q5_K_M.gguf     --detailed-llamacpp-parity     --max-tokens 1     --llama-include-global     --llama-filter token_embd,attn_norm     --llama-stop-after 2        
  --llama-timeout 0                                                                                                                                                                                                                                                        
   1920  python version/v6.6/scripts/ck_run_v6_6.py run hf://unsloth/gemma-3-270m-it-GGUF/gemma-3-270m-it-Q5_K_M.gguf --context-len 1024 --force-compile                                                                                                                   
   1921  python version/v6.6/scripts/parity_test.py     --ck-dump ~/.cache/ck-engine-v6.6/models/unsloth--gemma-3-270m-it-GGUF/ck_build/ck_parity_dumps/dump.bin     --ref-dump                                                                                            
  ~/.cache/ck-engine-v6.6/models/unsloth--gemma-3-270m-it-GGUF/ck_build/llama_parity_dumps/dump.bin     --model gemma --pass prefill | rg "token_embd|attn_norm|FAIL|ERROR"                                                                                                
   1922  python version/v6.6/scripts/ck_run_v6_6.py run hf://unsloth/gemma-3-270m-it-GGUF/gemma-3-270m-it-Q5_K_M.gguf --context-len 1024 --force-compile                                                                                                                   
   1969  histroy | grep gemma                                                                                                                                                                                                                                              
   1970  history | grep gemma                                                                                                                                                                                                                                              
   1971  python version/v6.6/scripts/parity_test.py     --ck-dump ~/.cache/ck-engine-v6.6/models/unsloth--gemma-3-270m-it-GGUF/ck_build/ck_parity_dumps/dump.bin     --ref-dump                                                                                            
  ~/.cache/ck-engine-v6.6/models/unsloth--gemma-3-270m-it-GGUF/ck_build/llama_parity_dumps/dump.bin     --model gemma --pass prefill | rg "token_embd|attn_norm|FAIL|ERROR"                                                                                                
   1977  history | grep gemma                                                                                                                                                                                                                                              
                                                                                                                                                                                                                                                                           

● Let me search both version/ and scripts/ for Gemma and llama.cpp parity related files.

● Explore(Find gemma/llamacpp parity scripts)
  ⎿  Done (40 tool uses · 90.8k tokens · 3m 44s)

● Here's the full inventory of Gemma and llama.cpp parity scripts across the repo:

  ---
  version/v6.6/scripts/ — Primary Parity Pipeline
  ┌─────────────────────────┬───────┬──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │          File           │ Size  │                                                                             Purpose                                                                              │
  ├─────────────────────────┼───────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ ck_run_v6_6.py          │ 116KB │ Main unified runner — download → convert → IR → codegen → compile → run. Has --detailed-llamacpp-parity, --llama-filter, --llama-layer, --llama-stop-after flags │
  ├─────────────────────────┼───────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ parity_test.py          │ 28KB  │ Layer-by-layer numerical comparison between CK and llama.cpp dumps. Supports --model gemma|llama|qwen|mistral, --pass prefill|decode, configurable tolerances    │
  ├─────────────────────────┼───────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ parity_report.py        │ 14KB  │ Generates JSON reports for parity_viewer.html interactive viewer. Works with both CK-only and CK vs llama.cpp comparisons                                        │
  ├─────────────────────────┼───────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ compare_tokens_hf_ck.py │ 4KB   │ Tokenizer parity — CK tokenizer vs HuggingFace tokenizer (Gemma-focused)                                                                                         │
  ├─────────────────────────┼───────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ ck_parity_dump.h        │ 5KB   │ CKDMP binary format definition — header struct for dump files                                                                                                    │
  ├─────────────────────────┼───────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ run_gemma_parity_min.sh │ 4KB   │ Minimal Gemma-3-270M parity script with --phase embed|layer0|both                                                                                                │
  └─────────────────────────┴───────┴──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
  scripts/ — Root-Level Parity Tools
  ┌───────────────────────────────┬──────┬────────────────────────────────────────────────────────────────────────────────────────┐
  │             File              │ Size │                                        Purpose                                         │
  ├───────────────────────────────┼──────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ run_gemma_first_divergence.sh │ 4KB  │ Standalone Gemma divergence debugger — finds first mismatch point                      │
  ├───────────────────────────────┼──────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ run_parity_smoketest.sh       │ 28KB │ Comprehensive CI smoketest — --quick, --kernels, --perf modes. Pins llama.cpp commit   │
  ├───────────────────────────────┼──────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ add_parity_dumps.py           │ 7KB  │ Injects dump_tensor() calls into generated model_v6_6.c for activation capture         │
  ├───────────────────────────────┼──────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ analyze_parity.py             │ 7KB  │ Dump analysis with statistics, NaN/Inf detection, top-k logit reporting                │
  ├───────────────────────────────┼──────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ compare_layer_llamacpp.py     │ 17KB │ Layer-by-layer weight + forward-pass comparison. Supports Q4_K, Q6_K, prefill + decode │
  ├───────────────────────────────┼──────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ benchmark_vs_llamacpp.py      │ 17KB │ Performance comparison — prefill/decode throughput, latency                            │
  └───────────────────────────────┴──────┴────────────────────────────────────────────────────────────────────────────────────────┘
  version/v6.6/test/ — Test Suite
  ┌────────────────────────────────┬──────────────────────────────────────────────────────────────────────┐
  │              File              │                               Purpose                                │
  ├────────────────────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ test_layer0_parity.py          │ Layer 0 full comparison (embed → norm → QKV → attn → MLP → residual) │
  ├────────────────────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ test_layer0_parity_prefill.py  │ Prefill-only parity for layer 0                                      │
  ├────────────────────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ test_layer0_parity_streamed.py │ Streaming parity tests                                               │
  ├────────────────────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ test_out_proj_parity.py        │ Out-projection micro-test (Q5_0 dequant, bias add)                   │
  ├────────────────────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ test_numerical_parity.py       │ Full layer-by-layer numerical validation                             │
  ├────────────────────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ compare_vs_llamacpp.py         │ mega_fused_attention_prefill vs llama.cpp flash attention            │
  ├────────────────────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ test_tokenizer_model_parity.py │ Tokenization correctness                                             │
  ├────────────────────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ verify_numerical_parity.py     │ Numerical verification                                               │
  ├────────────────────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ run_all_v66_tests.py           │ Test runner for all v6.6 tests                                       │
  └────────────────────────────────┴──────────────────────────────────────────────────────────────────────┘
  ---
  Typical Workflow (from your history)

  # 1. Full run with parity dumps
  python version/v6.6/scripts/ck_run_v6_6.py run \
    hf://unsloth/gemma-3-270m-it-GGUF/gemma-3-270m-it-Q5_K_M.gguf \
    --context-len 256 --force-compile \
    --detailed-llamacpp-parity \
    --prompt "Hello" --max-tokens 1 \
    --llama-layer 0 --llama-include-global \
    --llama-filter token_embd,attn_norm

  # 2. Compare dumps
  python version/v6.6/scripts/parity_test.py \
    --ck-dump ~/.cache/ck-engine-v6.6/.../ck_parity_dumps/dump.bin \
    --ref-dump ~/.cache/ck-engine-v6.6/.../llama_parity_dumps/dump.bin \
    --model gemma --pass prefill

  # 3. Quick Gemma divergence finder
  bash scripts/run_gemma_first_divergence.sh

  The parity infrastructure is quite comprehensive — you have binary dump format (CKDMP), instrumentation injection (add_parity_dumps.py), the main runner with llama.cpp integration, and dedicated comparison tools at multiple levels (tokens, weights, activations, full
  layers). The Gemma model family gets the most coverage with dedicated shell scripts.

✻ Cogitated for 4m 24s

❯ how do you suggest we oraize. jsut give em a plan 

● Here's a plan for organizing the scripts and tests:

  ---
  Current Problem

  Scripts and tests are scattered across 5 locations with unclear ownership:

  scripts/                      ← root-level (mix of old + new parity tools)
  src/v6.6/scripts/             ← LEGACY (should be dead)
  version/v6.6/scripts/         ← ACTIVE (52+ files, everything dumped here)
  version/v6.6/test/            ← v6.6 tests
  tests/                        ← root C integration tests
  unittest/                     ← Python kernel parity tests

  Proposed Structure

  version/v6.6/
  ├── scripts/
  │   ├── pipeline/             ← core codegen pipeline (what builds the model)
  │   │   ├── build_ir_v6_6.py
  │   │   ├── codegen_v6_6.py
  │   │   ├── codegen_prefill_v6_6.py
  │   │   ├── memory_planner_v6_6.py
  │   │   ├── parallel_pass.py
  │   │   ├── parallel_planner.py
  │   │   ├── fusion_patterns.py
  │   │   ├── ir_core_v6_6.py
  │   │   ├── ir_types_v6_6.py
  │   │   └── ir_schema.py
  │   │
  │   ├── convert/              ← model download + format conversion
  │   │   ├── convert_gguf_to_bump_v6_6.py
  │   │   ├── convert_hf_to_bump_v6_6.py
  │   │   ├── convert_any_gguf_v6_6.py
  │   │   └── convert_any_model_v6_6.sh
  │   │
  │   ├── parity/               ← all parity/comparison tools in one place
  │   │   ├── parity_test.py
  │   │   ├── parity_report.py
  │   │   ├── ck_parity_dump.h
  │   │   ├── compare_tokens_hf_ck.py
  │   │   ├── compare_layer_llamacpp.py    ← move from scripts/
  │   │   ├── add_parity_dumps.py          ← move from scripts/
  │   │   ├── analyze_parity.py            ← move from scripts/
  │   │   ├── benchmark_vs_llamacpp.py     ← move from scripts/
  │   │   ├── run_gemma_parity_min.sh
  │   │   └── run_gemma_first_divergence.sh ← move from scripts/
  │   │
  │   ├── inspect/              ← debugging + inspection utilities
  │   │   ├── bump_inspect.py
  │   │   ├── inspect_weights_v6_6.py
  │   │   ├── bump_layout_v6_6.py
  │   │   └── profile_inference.py
  │   │
  │   ├── registry/             ← kernel mapping + registry generation
  │   │   ├── kernel_registry.py
  │   │   ├── gen_kernel_map.py
  │   │   ├── gen_kernel_registry.py
  │   │   ├── kernel_source_scan.py
  │   │   ├── validate_kernel_maps.py
  │   │   └── check_kernel_map_sync.py
  │   │
  │   ├── ck_run_v6_6.py        ← top-level entry point (stays at root)
  │   └── ck_model_smoke_v6_6.py
  │
  ├── test/                     ← stays as-is, already well-organized
  │   ├── test_layer0_parity.py
  │   ├── test_numerical_parity.py
  │   ├── compare_vs_llamacpp.py
  │   ├── run_all_v66_tests.py
  │   └── ...
  │
  ├── ci/                       ← CI/nightly scripts
  │   ├── run_parity_smoketest.sh  ← move from scripts/
  │   └── nightly.sh               ← future
  │
  └── docs/                     ← move inline .md files here
      ├── FUSION_STATUS.md
      ├── KERNEL_MAPPING.md
      ├── IR1_ARCHITECTURE.md
      └── OP_BUILDER_DESIGN.md

  What to Do with Each Existing Location
  ┌───────────────────────┬──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │       Location        │                                                                  Action                                                                  │
  ├───────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ src/v6.6/scripts/     │ Delete or archive. Legacy. Add a DEPRECATED.md pointing to version/v6.6/scripts/                                                         │
  ├───────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ scripts/ (root)       │ Migrate parity tools to version/v6.6/scripts/parity/. Delete anything already duplicated. Keep only truly cross-version scripts (if any) │
  ├───────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ version/v6.6/scripts/ │ Reorganize into subdirs as above. The 52-file flat directory is the main pain point                                                      │
  ├───────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ *_legacy.py files     │ Delete. build_ir_v6_6_legacy.py, codegen_v6_6_legacy.py, codegen_v6_6_temp.py — if they're not imported anywhere, remove them            │
  └───────────────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
  Key Principles

  1. One location per concern — all parity tools in parity/, all conversion in convert/
  2. ck_run_v6_6.py stays at the scripts root — it's the top-level entry point, imports from subdirs
  3. Inline docs move to docs/ — 6 markdown files currently mixed in with Python scripts
  4. Kill legacy aggressively — _legacy.py and _temp.py files should be deleted, not kept around
  5. CI scripts get their own dir — smoketests and nightly runs are distinct from dev tools

  Want me to implement this?

✻ Brewed for 32s

❯ give me a detailed prompt how to fix this and make adjustments to make files too?                                                                                                                                                                                        
  lets not kill legacy. legacy is improtant to show how we are evolving.                                                                                                                                                                                                   

● Here's a detailed prompt you can use:

  ---
  Prompt: Reorganize v6.6 Scripts Directory and Update Build System

  Context

  The C-Kernel-Engine project has v6.6 scripts scattered across multiple locations with no clear organization. The main active directory (version/v6.6/scripts/) has 52+ files in a flat structure mixing codegen pipeline, parity tools, conversion scripts, inspection
  utilities, registry generators, and documentation. Root-level scripts/ has duplicate/overlapping parity tools. src/v6.6/scripts/ is legacy but should be preserved.

  Goals

  1. Reorganize version/v6.6/scripts/ into logical subdirectories
  2. Consolidate parity/llama.cpp tools from scripts/ into version/v6.6/scripts/parity/
  3. Preserve all legacy code — never delete, only move to a legacy/ subdirectory with a README explaining its historical purpose
  4. Update all import paths in Python files so nothing breaks
  5. Update the Makefile to reflect new paths
  6. Update ck_run_v6_6.py imports and subprocess calls to reference new locations
  7. Add __init__.py files to new subdirectories so Python imports work
  8. Verify nothing breaks by doing a dry-run of the codegen pipeline

  Directory Reorganization Plan

  Current flat structure: version/v6.6/scripts/ (52+ files)

  Target structure:

  version/v6.6/scripts/
  ├── ck_run_v6_6.py                    ← TOP-LEVEL ENTRY POINT (stays here)
  ├── ck_model_smoke_v6_6.py            ← quick smoke test (stays here)
  │
  ├── pipeline/                         ← Core codegen pipeline
  │   ├── __init__.py
  │   ├── build_ir_v6_6.py              ← IR construction (241KB)
  │   ├── codegen_v6_6.py               ← Code emission (61KB)
  │   ├── codegen_prefill_v6_6.py       ← Prefill codegen (18KB)
  │   ├── ir_core_v6_6.py               ← IR core logic (139KB)
  │   ├── ir_types_v6_6.py              ← IR type definitions (22KB)
  │   ├── ir_schema.py                  ← IR schema (16KB)
  │   ├── memory_planner_v6_6.py        ← Memory planning (20KB)
  │   ├── parallel_pass.py              ← Threading pass (19KB)
  │   ├── parallel_planner.py           ← Threading planner (22KB)
  │   ├── fusion_patterns.py            ← Fusion pattern matching (18KB)
  │   ├── v6_6_ir_lowering.py           ← IR lowering (40KB)
  │   ├── parse_template_v2.py          ← Template parsing (15KB)
  │   ├── op_builders_v6_6.py           ← Op builders (14KB)
  │   ├── op_builders_hybrid_v6_6.py    ← Hybrid op builders (9KB)
  │   ├── op_builders_auto.py           ← Auto op builders (20KB)
  │   ├── gen_layout_header_v6_6.py     ← Layout header gen (15KB)
  │   ├── generate_memory_map_v6_6.py   ← Memory map gen (12KB)
  │   └── bump_layout_v6_6.py           ← Bump layout (11KB)
  │
  ├── convert/                          ← Model download + format conversion
  │   ├── __init__.py
  │   ├── convert_gguf_to_bump_v6_6.py  ← GGUF→BUMP (113KB)
  │   ├── convert_hf_to_bump_v6_6.py    ← HF→BUMP (27KB)
  │   ├── convert_hf_to_bump.py         ← Original HF converter (18KB)
  │   ├── convert_any_gguf_v6_6.py      ← Universal GGUF (5KB)
  │   ├── convert_any_gguf_v6_6.sh      ← Shell wrapper (2KB)
  │   ├── convert_any_model_v6_6.sh     ← Shell wrapper (4KB)
  │   └── quant_types.py                ← Quantization type defs (15KB)
  │
  ├── parity/                           ← All parity + llama.cpp comparison tools
  │   ├── __init__.py
  │   ├── parity_test.py                ← Core parity comparison (28KB)
  │   ├── parity_report.py              ← Report generator (14KB)
  │   ├── ck_parity_dump.h              ← CKDMP binary format header (5KB)
  │   ├── compare_tokens_hf_ck.py       ← Tokenizer parity (4KB)
  │   ├── compare_layer_llamacpp.py     ← MOVE from scripts/ (17KB)
  │   ├── add_parity_dumps.py           ← MOVE from scripts/ (7KB)
  │   ├── analyze_parity.py             ← MOVE from scripts/ (7KB)
  │   ├── benchmark_vs_llamacpp.py      ← MOVE from scripts/ (17KB)
  │   ├── run_gemma_parity_min.sh       ← Gemma parity runner (4KB)
  │   └── run_gemma_first_divergence.sh ← MOVE from scripts/ (4KB)
  │
  ├── inspect/                          ← Debugging + weight inspection
  │   ├── __init__.py
  │   ├── bump_inspect.py               ← BUMP file inspector (8KB)
  │   ├── inspect_weights_v6_6.py       ← Weight inspector (21KB)
  │   ├── profile_inference.py          ← Profiling (5KB)
  │   ├── q4_k_quantize.py              ← Q4_K quantization (5KB)
  │   └── verify_test_bug.py            ← Bug verification (2KB)
  │
  ├── registry/                         ← Kernel mapping + registry
  │   ├── __init__.py
  │   ├── kernel_registry.py            ← Main registry (27KB)
  │   ├── gen_kernel_map.py             ← Map generator (15KB)
  │   ├── gen_kernel_registry.py        ← Registry generator (4KB)
  │   ├── gen_kernel_registry_from_maps.py ← Registry from maps (6KB)
  │   ├── gen_op_builders_from_registry.py ← Op builders gen (14KB)
  │   ├── kernel_source_scan.py         ← Source scanner (7KB)
  │   ├── validate_kernel_maps.py       ← Map validator (14KB)
  │   ├── validate_templates.py         ← Template validator (8KB)
  │   ├── validate_template_v2.py       ← Template v2 validator (14KB)
  │   └── check_kernel_map_sync.py      ← Sync checker (3KB)
  │
  ├── fusion/                           ← Fusion-specific tools
  │   ├── __init__.py
  │   ├── registry_driven_fusion.py     ← Fusion registry (7KB)
  │   ├── add_decode_fusion_patterns.py ← Decode fusion (3KB)
  │   ├── update_fusion_patterns.py     ← Pattern updates (2KB)
  │   └── build_ir2_from_ir1.py         ← IR1→IR2 (9KB)
  │
  ├── training/                         ← Training config (future use)
  │   ├── __init__.py
  │   └── training_config.py            ← Training config (40KB)
  │
  ├── ci/                               ← CI + smoketests
  │   ├── run_parity_smoketest.sh       ← MOVE from scripts/ (28KB)
  │   └── README.md                     ← How to run CI tests
  │
  ├── legacy/                           ← Historical versions (DO NOT USE for new work)
  │   ├── README.md                     ← Explains legacy purpose + evolution history
  │   ├── build_ir_v6_6_legacy.py       ← Original IR builder before refactor
  │   ├── codegen_v6_6_legacy.py        ← Original codegen before modularization
  │   └── codegen_v6_6_temp.py          ← Intermediate version during refactor
  │
  ├── ir_reverse_validator.py           ← IR round-trip validator (stays, cross-cutting)
  │
  └── docs/                             ← Move inline .md out of the script dir
      ├── FUSION_ENABLED.md
      ├── FUSION_STATUS.md
      ├── KERNEL_MAPPING.md
      ├── KERNEL_SELECTION_CORRECTED.md
      ├── IR1_ARCHITECTURE.md
      ├── OP_BUILDER_DESIGN.md
      ├── README_OP_BUILDERS.md
      └── PROFILING_AND_OPTIMIZATION_PLAN.md

  Files to Move FROM scripts/ (root) INTO version/v6.6/scripts/parity/

  scripts/compare_layer_llamacpp.py    → version/v6.6/scripts/parity/compare_layer_llamacpp.py
  scripts/add_parity_dumps.py          → version/v6.6/scripts/parity/add_parity_dumps.py
  scripts/analyze_parity.py            → version/v6.6/scripts/parity/analyze_parity.py
  scripts/benchmark_vs_llamacpp.py     → version/v6.6/scripts/parity/benchmark_vs_llamacpp.py
  scripts/run_gemma_first_divergence.sh → version/v6.6/scripts/parity/run_gemma_first_divergence.sh
  scripts/run_parity_smoketest.sh      → version/v6.6/scripts/ci/run_parity_smoketest.sh

  After moving, leave a forwarding stub in the original scripts/ location so existing workflows don't break silently:

  # scripts/compare_layer_llamacpp.py
  """MOVED: This script has moved to version/v6.6/scripts/parity/compare_layer_llamacpp.py"""
  import sys, os
  sys.exit(f"ERROR: This script has moved to version/v6.6/scripts/parity/{os.path.basename(__file__)}")

  Legacy Handling

  src/v6.6/scripts/ — Mark as legacy, do NOT delete:

  1. Add src/v6.6/scripts/LEGACY_README.md:
  # Legacy v6.6 Scripts (DO NOT USE)

  These scripts are the **original monolithic v6.6 codegen pipeline** from early development.
  They are preserved for historical reference to show the evolution from monolithic → modular design.

  ## Active location
  All v6.6 development now uses: `version/v6.6/scripts/`

  ## Files
  - `v6.6_codegen.py` (106KB) — Original monolithic codegen. Replaced by modular pipeline:
    - `version/v6.6/scripts/pipeline/codegen_v6_6.py` (61KB) — code emission
    - `version/v6.6/scripts/pipeline/build_ir_v6_6.py` (241KB) — IR construction
    - `version/v6.6/scripts/pipeline/ir_core_v6_6.py` (139KB) — IR core
  - `v6.6_download.py` (9KB) — Original model downloader
  - `kernel_selector.py` (7KB) — Original kernel selection

  version/v6.6/scripts/legacy/ — Move *_legacy.py and *_temp.py files:

  Add version/v6.6/scripts/legacy/README.md:
  # Legacy Pipeline Versions

  These files show the evolution of the v6.6 codegen pipeline.
  Preserved for reference — DO NOT import or use in new code.

  ## Timeline
  1. `codegen_v6_6_legacy.py` (110KB) — First modular attempt, still had hardcoded logic
  2. `codegen_v6_6_temp.py` (60KB) — Intermediate cleanup during refactoring
  3. `codegen_v6_6.py` (current, 61KB) — Clean version with IR-driven emission
  4. `build_ir_v6_6_legacy.py` (177KB) — Original IR builder before schema-driven rewrite
  5. `build_ir_v6_6.py` (current, 241KB) — Schema-driven IR with proper validation

  Import Path Updates

  After moving files into subdirectories, update all import and from ... import statements. Key files that import from sibling scripts:

  1. ck_run_v6_6.py — This is the main entry point. It likely imports or subprocess-calls many of the pipeline scripts. Update all references:
    - import codegen_v6_6 → from pipeline import codegen_v6_6
    - import build_ir_v6_6 → from pipeline import build_ir_v6_6
    - import memory_planner_v6_6 → from pipeline import memory_planner_v6_6
    - import parity_test → from parity import parity_test
    - import convert_gguf_to_bump_v6_6 → from convert import convert_gguf_to_bump_v6_6
    - Any subprocess.run(["python", "...path..."]) calls need updated paths
  2. codegen_v6_6.py — May import from ir_core_v6_6, ir_types_v6_6, fusion_patterns, parallel_pass. Update to relative imports within pipeline/.
  3. build_ir_v6_6.py — May import ir_core_v6_6, ir_schema, ir_types_v6_6. Update similarly.
  4. parity_test.py / parity_report.py — Check for imports of dump format or analysis helpers.
  5. Shell scripts (run_gemma_parity_min.sh, run_parity_smoketest.sh) — Update hardcoded python version/v6.6/scripts/parity_test.py paths.

  Strategy for safe import updates:
  - First, grep -rn "^import \|^from " version/v6.6/scripts/*.py to find all current imports
  - For each file being moved, find all files that reference it: grep -rn "filename" version/v6.6/
  - Add sys.path fallback in __init__.py if needed for backwards compatibility
  - Consider adding a top-level version/v6.6/scripts/conftest.py or path setup for test discovery

  Makefile Updates

  Search the Makefile (150KB) for any references to script paths and update:

  grep -n "v6.6/scripts\|scripts/.*parity\|scripts/.*llamacpp\|scripts/.*gemma\|scripts/.*benchmark" Makefile

  Expected changes:
  - make test-parity targets → update script paths
  - make llamacpp-parity-full → update to version/v6.6/scripts/parity/ paths
  - make e2e → if it calls ck_run_v6_6.py, path stays the same
  - make benchmark → update to version/v6.6/scripts/parity/benchmark_vs_llamacpp.py
  - Any PARITY_SCRIPT, CODEGEN_SCRIPT variables → update

  Verification Steps

  After all moves and import updates:

  1. Python import check:
  cd version/v6.6/scripts && python -c "from pipeline import codegen_v6_6; print('OK')"
  cd version/v6.6/scripts && python -c "from parity import parity_test; print('OK')"
  cd version/v6.6/scripts && python -c "from convert import convert_gguf_to_bump_v6_6; print('OK')"
  2. Codegen dry run:
  python version/v6.6/scripts/ck_run_v6_6.py run \
    hf://unsloth/gemma-3-270m-it-GGUF/gemma-3-270m-it-Q5_K_M.gguf \
    --context-len 256 --force-compile --dry-run
  3. Parity test dry run:
  python version/v6.6/scripts/parity/parity_test.py --help
  4. Make targets:
  make test-parity
  make e2e
  5. Shell scripts:
  bash version/v6.6/scripts/parity/run_gemma_parity_min.sh --help

  Files to NOT Move (keep in scripts/ root)

  Keep anything in root scripts/ that is truly cross-version or used by the top-level Makefile for non-v6.6 targets. Audit each remaining file — if it's v6.6-specific, move it. If it's generic tooling, leave it.

  Commit Strategy

  Do this in 3 commits to keep history clean and make bisection possible:

  1. Reorganize v6.6 scripts into logical subdirectories — mkdir, git mv, add __init__.py files, add legacy READMEs
  2. Update import paths and subprocess calls for new script layout — fix all Python imports, shell script paths
  3. Update Makefile targets for reorganized script paths — Makefile path updates, verify make targets work


