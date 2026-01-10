# Bump Converter: Multi-Architecture Support

## The Problem: LLaMA-Only Converter

The current bump converter (`convert_hf_to_bump_v4.py`) is **hardcoded for LLaMA-style models** with specific weight naming patterns:

```python
# Hardcoded LLaMA patterns
"model.embed_tokens.weight"           # Embedding layer
"model.layers.{layer}"               # Layer naming
"input_layernorm.weight"            # LayerNorm
"self_attn.q_proj.weight"           # Attention projections
"mlp.gate_proj.weight"               # MLP gates
"model.norm.weight"                 # Final norm
```

This works great for LLaMA, but **fails for other architectures** like:
- **Devstral** - Different embedding layer names
- **SmolLM** - Slightly different layer norm naming
- **Qwen** - Similar to LLaMA but not identical
- **Any custom model** - Different architectures entirely

### Common Error Messages

```
KeyError: 'model.embed_tokens.weight'
KeyError: 'input_layernorm.weight'
KeyError: 'self_attn.q_proj.weight'
```

## The Solution: Universal Converter

### 1. Quick Diagnosis: `inspect_model.py`

First, **inspect your model** to see its actual weight names:

```bash
python scripts/inspect_model.py --checkpoint /path/to/Devstral/model

# Output shows:
# - Config details
# - Weight patterns
# - Architecture detection
# - Suggested mapping
```

Example output for Devstral:
```
[CONFIG]
  Model Type: devstral
  Hidden Size: 896
  Vocab Size: 151936
  Num Layers: 24

[WEIGHT PATTERNS]
  model.embed_tokens.weight              x1
  model.layers.{i}.input_layernorm      x24
  model.layers.{i}.self_attn           x24
  ...

[ARCHITECTURE DETECTION]
  Detected: Devstral/SmolLM

[SUGGESTED MAPPING]
  Use --arch devstral
```

### 2. Convert with Architecture-Specific Mapping

Use the **universal converter** with the `--arch` flag:

```bash
# For Devstral/SmolLM
python scripts/convert_to_bump_universal.py \
  --checkpoint /path/to/Devstral/model \
  --output model.bump \
  --arch devstral

# For SmolLM
python scripts/convert_to_bump_universal.py \
  --checkpoint /path/to/SmolLM/model \
  --output model.bump \
  --arch smollm

# For Qwen
python scripts/convert_to_bump_universal.py \
  --checkpoint /path/to/Qwen/model \
  --output model.bump \
  --arch qwen

# Auto-detect (recommended)
python scripts/convert_to_bump_universal.py \
  --checkpoint /path/to/any/model \
  --output model.bump \
  --arch auto
```

### 3. Architecture Mappings

The universal converter supports multiple architectures:

| Architecture | Detected From | Key Differences |
|------------|---------------|-----------------|
| **llama** | `model_type: llama` or `tok_embeddings` in weights | Standard LLaMA naming |
| **devstral** | `model_type: devstral` or `embedding.weight` | Uses `model.embedding.weight` |
| **smollm** | `model_type: smollm` | Similar to Devstral |
| **qwen** | `model_type: qwen2` | Qwen2-specific naming |
| **auto** | Auto-detect from config and weights | Best for unknown models |

### 4. How It Works

The universal converter uses **flexible weight mapping**:

```python
# Old way (hardcoded)
"model.embed_tokens.weight"

# New way (flexible)
ARCHITECTURE_MAPPINGS = {
    "devstral": {
        "embed_tokens": [
            "model.embed_tokens.weight",
            "model.embedding.weight",  # Try this too!
        ],
        "layer_norm": {
            "ln1": [
                "input_layernorm.weight",
                "layer_norm.weight",  # Alternative name
            ],
        },
        # ... more flexible mappings
    }
}
```

## Troubleshooting

### Error: `KeyError: 'model.embed_tokens.weight'`

**Solution:**
```bash
# Inspect the model first
python scripts/inspect_model.py --checkpoint /path/to/model

# Use the correct architecture
python scripts/convert_to_bump_universal.py \
  --checkpoint /path/to/model \
  --output model.bump \
  --arch auto
```

### Error: `Config missing required fields`

**Solution:**
```bash
# Create a config override
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
  --arch auto \
  --config override.json
```

### Architecture Not Supported

**Solution:**
1. **Inspect the model** to see weight patterns
2. **Create a custom mapping** by extending `ARCHITECTURE_MAPPINGS`
3. **Or use `auto` mode** which tries multiple patterns

## For Developers

### Adding Support for a New Architecture

1. **Inspect the model:**
```bash
python scripts/inspect_model.py --checkpoint /path/to/new_model
```

2. **Add mapping** to `convert_to_bump_universal.py`:
```python
ARCHITECTURE_MAPPINGS = {
    "my_architecture": {
        "embed_tokens": ["model.embed_tokens.weight"],
        "layers": "model.blocks.{layer}",  # Different layer name
        "layer_norm": {
            "ln1": ["norm1.weight"],  # Different LN name
            "ln2": ["norm2.weight"],
        },
        "attention": {
            "q_proj": "attn.q.weight",  # Different projection names
            "k_proj": "attn.k.weight",
            "v_proj": "attn.v.weight",
            "o_proj": "attn.o.weight",
        },
        "mlp": {
            "gate_proj": "ffn.gate.weight",
            "up_proj": "ffn.up.weight",
            "down_proj": "ffn.down.weight",
        },
        "final_norm": "norm.weight",
        "lm_head": "lm_head.weight",
    }
}
```

3. **Test it:**
```bash
python scripts/convert_to_bump_universal.py \
  --checkpoint /path/to/new_model \
  --output model.bump \
  --arch my_architecture
```

## Migration from Old Converter

### Before (LLaMA only)
```bash
python scripts/convert_hf_to_bump_v4.py \
  --checkpoint /path/to/llama/model \
  --output model.bump
```

### After (Universal)
```bash
# Same command works for LLaMA!
python scripts/convert_to_bump_universal.py \
  --checkpoint /path/to/llama/model \
  --output model.bump \
  --arch llama  # or --arch auto
```

## Summary

✅ **Problem:** Converter only works for LLaMA models
✅ **Solution:** Universal converter with architecture detection
✅ **How to use:** Inspect model → Use `--arch` flag → Convert
✅ **Supports:** LLaMA, Devstral, SmolLM, Qwen, and custom models
✅ **Best practice:** Use `--arch auto` for automatic detection

## Quick Reference

```bash
# 1. Inspect any model
python scripts/inspect_model.py --checkpoint /path/to/model

# 2. Convert with auto-detection
python scripts/convert_to_bump_universal.py \
  --checkpoint /path/to/model \
  --output model.bump \
  --arch auto

# 3. Or specify architecture
python scripts/convert_to_bump_universal.py \
  --checkpoint /path/to/model \
  --output model.bump \
  --arch devstral  # or llama, smollm, qwen
```
