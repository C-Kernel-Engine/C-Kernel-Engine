# v6.6 Manual E2E Help

This doc is a quick reference for manual E2E checks on the current v6.6 models.

## HF model URLs

- Gemma3 Q5_K_M  
  `hf://unsloth/gemma-3-270m-it-GGUF/gemma-3-270m-it-Q5_K_M.gguf`
- Qwen3 Q8_0  
  `hf://Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf`
- Qwen2 Q4_K_M  
  `hf://Qwen/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q4_k_m.gguf`

## Single‑model runs

Gemma3:

```bash
python3 version/v6.6/scripts/ck_run_v6_6.py run \
  hf://unsloth/gemma-3-270m-it-GGUF/gemma-3-270m-it-Q5_K_M.gguf \
  --context-len 1024 --force-compile \
  --prompt "Hello! Reply with a single short sentence." \
  --max-tokens 32
```

Qwen3:

```bash
python3 version/v6.6/scripts/ck_run_v6_6.py run \
  hf://Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf \
  --context-len 1024 --force-compile \
  --prompt "Give a tiny C function that returns 42." \
  --max-tokens 32
```

Qwen2:

```bash
python3 version/v6.6/scripts/ck_run_v6_6.py run \
  hf://Qwen/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q4_k_m.gguf \
  --context-len 1024 --force-compile \
  --prompt "Give a tiny SQL query that selects the number 42." \
  --max-tokens 32
```

## Full sweep (all models + prompts)

```bash
./scripts/e2e_manual_v66.sh
```

## Prompt set

- `Hello! Reply with a single short sentence.`
- `Give a tiny C function that returns 42.`
- `Give a tiny Python function that returns 42.`
- `Give a tiny SQL query that selects the number 42.`
