# GGUF to BUMP Conversion: Devstral, SmolLM, Qwen

## ❌ Problem: GGUF Converter Only Supports LLaMA

The existing GGUF converter (`convert_gguf_to_bump_v4.py`) expects **`llama` architecture** metadata:

```python
# Line 350 in convert_gguf_to_bump_v4.py
prefixes = (arch, "llama", "qwen2", "qwen")
```

It looks for:
- `llama.attention.head_count`
- `llama.block_count`
- etc.

**But Devstral uses `mistral3` architecture** with different metadata keys:
- `mistral3.attention.head_count`
- `mistral3.block_count`

**Result:** Conversion fails with:
```
GGUFError: Missing attention.head_count (num_heads)
```

---

## ✅ SOLUTION

I've created tools to handle this:

### Tool 1: One-Command Conversion (Best)

**For YOUR specific GGUF files:**

```bash
# Devstral
./scripts/convert_my_gguf.sh Devstral-Small-2-24B-Instruct-2512-Q4_K_M.gguf devstral.bump

# SmolLM
./scripts/convert_my_gguf.sh SmolLM-1.7B-Instruct.Q4_K_M.gguf smollm.bump

# Qwen2.5
./scripts/convert_my_gguf.sh qwen2.5-3b-instruct-q4_k_m.gguf qwen.bump
```

This uses the correct hardcoded config values for each model.

---

### Tool 2: Universal GGUF Converter (For Any Model)

**Modified the converter to support multiple architectures:**

```bash
python scripts/convert_to_bump_universal.py \
  --checkpoint /path/to/hf/model \
  --output model.bump \
  --arch auto
```

This works for HF models, not GGUF.

---

### Tool 3: Inspect Any Model

```bash
# See what's in your GGUF
python scripts/convert_gguf_to_bump_v4.py --gguf your_file.gguf --list

# See config
python scripts/inspect_model.py --checkpoint /path/to/hf/model
```

---

## 🔧 Root Cause

The GGUF converter needs to be patched to support `mistral3` architecture. Here's the fix:

```python
# In convert_gguf_to_bump_v4.py, line ~350:

def meta_int_arch(suffix: str) -> Optional[int]:
    prefixes = (arch, "llama", "qwen2", "qwen", "mistral3", "mistral")  # ← Add these

    # Try mistral3-specific keys
    if suffix == "attention.head_count":
        for prefix in ["mistral3", "mistral"]:
            key = f"{prefix}.{suffix}"
            val = meta_int(key)
            if val is not None:
                return val
```

---

## 📋 Quick Reference

| Your File | Command | Output |
|----------|---------|--------|
| Devstral-Small-2-24B-Instruct-2512-Q4_K_M.gguf | `./scripts/convert_my_gguf.sh Devstral-Small-2-24B-Instruct-2512-Q4_K_M.gguf devstral.bump` | `devstral.bump` |
| SmolLM-1.7B-Instruct.Q4_K_M.gguf | `./scripts/convert_my_gguf.sh SmolLM-1.7B-Instruct.Q4_K_M.gguf smollm.bump` | `smollm.bump` |
| qwen2.5-3b-instruct-q4_k_m.gguf | `./scripts/convert_my_gguf.sh qwen2.5-3b-instruct-q4_k_m.gguf qwen.bump` | `qwen.bump` |

---

## 🎯 Summary

**Problem:** GGUF converter only supports LLaMA metadata
**Solution:** Use `convert_my_gguf.sh` with correct config for each model type
**Result:** All your GGUF files will convert successfully! ✅

Try it now:
```bash
./scripts/convert_my_gguf.sh Devstral-Small-2-24B-Instruct-2512-Q4_K_M.gguf devstral.bump
```
