# Bump Converter: Multi-Architecture Support Guide

## Problem Statement

The current bump converter only works with **LLaMA-style models** and fails with:
- ❌ **Devstral** - Different embedding layer names
- ❌ **SmolLM** - Different LayerNorm naming
- ❌ **Other architectures** - Custom weight naming

Common errors:
```
KeyError: 'model.embed_tokens.weight'
KeyError: 'input_layernorm.weight'
```

## Solution: Use the Universal Converter

### Option 1: Quick Fix (Minimal Changes)

**Use the universal converter:**

```bash
# Inspect your model first
python scripts/inspect_model.py --checkpoint /path/to/Devstral

# Convert with auto-detection
python scripts/convert_to_bump_universal.py \
  --checkpoint /path/to/Devstral \
  --output model.bump \
  --arch auto

# Or specify architecture
python scripts/convert_to_bump_universal.py \
  --checkpoint /path/to/Devstral \
  --output model.bump \
  --arch devstral
```

### Option 2: Patch Existing Converter

**Quickly patch your existing converter:**

```bash
# Make a backup
cp scripts/convert_hf_to_bump_v4.py scripts/convert_hf_to_bump_v4.py.backup

# Apply the patch
python scripts/patch_bump_converter.py \
  --input scripts/convert_hf_to_bump_v4.py \
  --output scripts/convert_hf_to_bump_v4_patched.py

# Use the patched version
python scripts/convert_hf_to_bump_v4_patched.py \
  --checkpoint /path/to/Devstral \
  --output model.bump
```

## Quick Reference

### Inspect Any Model
```bash
python scripts/inspect_model.py --checkpoint /path/to/model
```

**Output shows:**
- ✅ Config details
- ✅ Weight patterns
- ✅ Architecture detection
- ✅ Suggested mapping

### Convert Model
```bash
# Auto-detect (recommended)
python scripts/convert_to_bump_universal.py \
  --checkpoint /path/to/model \
  --output model.bump \
  --arch auto

# Or specify architecture
--arch llama      # LLaMA models
--arch devstral   # Devstral/SmolLM models
--arch smollm     # SmolLM models
--arch qwen       # Qwen models
```

## Architecture Support

| Model | Command |
|-------|---------|
| LLaMA | `--arch llama` or `--arch auto` |
| Devstral | `--arch devstral` or `--arch auto` |
| SmolLM | `--arch smollm` or `--arch auto` |
| Qwen | `--arch qwen` or `--arch auto` |
| Unknown | `--arch auto` (tries multiple patterns) |

## Troubleshooting

### Still getting KeyError?

```bash
# 1. Inspect the model to see actual weight names
python scripts/inspect_model.py --checkpoint /path/to/model

# 2. Check the output for weight patterns
# Example for Devstral:
# [WEIGHT PATTERNS]
#   model.embedding.weight              x1  ← Different from LLaMA!
#   model.layers.{i}.layer_norm.weight  x24 ← Different from LLaMA!
```

### Config missing required fields?

```bash
# Create config override
cat > override.json <<EOF
{
  "vocab_size": 32000,
  "context_window": 4096,
  "hidden_size": 2048,
  "num_hidden_layers": 24,
  "num_attention_heads": 32,
  "intermediate_size": 8192
}
EOF

# Use with converter
python scripts/convert_to_bump_universal.py \
  --checkpoint /path/to/model \
  --output model.bump \
  --config override.json
```

## Files Created

1. **`scripts/inspect_model.py`** - Diagnostic tool to inspect model structure
2. **`scripts/convert_to_bump_universal.py`** - Universal converter with multi-arch support
3. **`scripts/patch_bump_converter.py`** - Quick patch for existing converter
4. **`docs/bump-converter-multi-arch.md`** - Detailed documentation

## Migration Guide

### Old Way (LLaMA Only)
```bash
python scripts/convert_hf_to_bump_v4.py \
  --checkpoint /path/to/llama \
  --output model.bump
```

### New Way (Universal)
```bash
# Same command now works for ALL models!
python scripts/convert_to_bump_universal.py \
  --checkpoint /path/to/llama \
  --output model.bump \
  --arch auto  # ← Add this
```

## For Developers

### Adding Support for a New Architecture

1. **Inspect the model:**
```bash
python scripts/inspect_model.py --checkpoint /path/to/new_model
```

2. **Add mapping** to `convert_to_bump_universal.py`:
```python
ARCHITECTURE_MAPPINGS = {
    "my_arch": {
        "embed_tokens": ["model.embed_tokens.weight"],
        "layers": "model.blocks.{layer}",
        "layer_norm": {
            "ln1": ["norm1.weight"],
            "ln2": ["norm2.weight"],
        },
        # ... more mappings
    }
}
```

3. **Test it:**
```bash
python scripts/convert_to_bump_universal.py \
  --checkpoint /path/to/new_model \
  --output model.bump \
  --arch my_arch
```

## Summary

✅ **Problem:** Converter only supports LLaMA
✅ **Solution:** Universal converter with auto-detection
✅ **Usage:** Inspect → Convert with `--arch auto`
✅ **Supports:** LLaMA, Devstral, SmolLM, Qwen, custom models

## Example: Converting Devstral

```bash
# Step 1: Inspect
python scripts/inspect_model.py --checkpoint /path/to/Devstral
# → Shows: model.embedding.weight, model.layers.{i}.layer_norm.weight

# Step 2: Convert
python scripts/convert_to_bump_universal.py \
  --checkpoint /path/to/Devstral \
  --output devstral.bump \
  --arch auto
# → Success! ✅

# Step 3: Use the bump file
ck_cli_v5 --model devstral.bump --prompt "Hello"
```

That's it! The converter now works with Devstral and other non-LLaMA models! 🎉
