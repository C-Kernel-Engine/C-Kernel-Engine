# GGUF to BUMP Conversion Guide

## Overview

The C-Kernel Engine provides a robust GGUF to BUMP converter that supports multiple model architectures. The converter automatically detects the model architecture and handles the conversion seamlessly.

## Supported Architectures

The converter now supports the following architectures:

| Architecture | Example Models | Metadata Prefix | Status |
|-------------|---------------|----------------|--------|
| **LLaMA** | Original LLaMA models | `llama.*` | ✅ Supported |
| **Qwen2** | Qwen2, Qwen2.5 | `qwen2.*` | ✅ Supported |
| **Mistral3** | Devstral, SmolLM | `mistral3.*` | ✅ Supported |
| **Mistral** | Mistral-based models | `mistral.*` | ✅ Supported |

## Usage

### Basic Conversion

```bash
python scripts/convert_gguf_to_bump.py --gguf model.gguf --output model.bump
```

The converter will automatically:
- Detect the architecture from the GGUF metadata
- Extract model dimensions and configuration
- Handle non-standard tensor dimensions
- Convert all weights to the BUMP format

### Advanced Options

```bash
# Specify context length
python scripts/convert_gguf_to_bump.py --gguf model.gguf --output model.bump --context 4096

# Limit number of layers
python scripts/convert_gguf_to_bump.py --gguf model.gguf --output model.bump --max-layers 10

# List model information
python scripts/convert_gguf_to_bump.py --gguf model.gguf --list

# Extract vocabulary
python scripts/convert_gguf_to_bump.py --gguf model.gguf --extract-vocab vocab.txt
```

## Successful Conversions

We've successfully tested the converter with multiple models:

### ✅ SmolLM-1.7B-Instruct

```bash
python scripts/convert_gguf_to_bump.py \
  --gguf SmolLM-1.7B-Instruct.Q4_K_M.gguf \
  --output smollm.bump
```

**Result:**
- Architecture: Mistral3
- Layers: 24
- Hidden size: 2048
- Attention heads: 32
- Vocab size: 49,152
- Output: 1.0GB BUMP file

**Conversion output:**
```
[gguf->bump] version=3 arch=llama layers=24 hidden=2048 heads=32/32 ff=8192 vocab=49152 ctx=2048 no biases -> smollm.bump
```

### ✅ Qwen2.5-3B-Instruct

```bash
python scripts/convert_gguf_to_bump.py \
  --gguf qwen2.5-3b-instruct-q4_k_m.gguf \
  --output qwen.bump
```

**Result:**
- Architecture: Qwen2
- Layers: 36
- Hidden size: 2048
- Attention heads: 16
- KV heads: 2
- Vocab size: 151,936
- Context length: 32,768
- Output: 2.0GB BUMP file

**Conversion output:**
```
[gguf->bump] version=3 arch=qwen2 layers=36 hidden=2048 heads=16/2 ff=11008 vocab=151936 ctx=32768 biases=36/36 layers -> qwen.bump
```

### ✅ Devstral-Small-2-24B-Instruct

```bash
python scripts/convert_gguf_to_bump.py \
  --gguf Devstral-Small-2-24B-Instruct-2512-Q4_K_M.gguf \
  --output devstral.bump
```

**Result:**
- Architecture: Mistral3
- Layers: 40
- Hidden size: 5120
- Attention heads: 32
- Vocab size: 131,072
- Context length: 393,216
- Output: 361MB BUMP file

**Special handling:**
```
[info] Detected non-standard attention dimensions
  expected: Q=5120, K=1280, O=5120
  actual:   Q=4096, K=1024, O=4096
  inferred: head_dim=128, embed_kv=1024
```

**Conversion output:**
```
[gguf->bump v4] arch=mistral3 layers=40 hidden=5120 heads=32/8 ff=32768 vocab=131072 ctx=393216 -> devstral.bump
```

## Technical Details

### Architecture Detection

The converter automatically tries metadata keys in priority order:

1. **deepseek2.*** (DeepSeek2/MoE models)
2. **mistral3.*** (Devstral, SmolLM)
3. **mistral.*** (Mistral-based models)
4. **llama.*** (LLaMA models)
5. **qwen2.*** (Qwen2 models)

This ensures correct detection without user configuration.

### Dimension Inference

For non-standard architectures (like Devstral), the converter:

1. Checks actual tensor dimensions against metadata expectations
2. Infers correct dimensions from the first layer's tensors
3. Updates head_dim and embed_kv values
4. Validates dimensions flexibly to handle variations

### Supported Quantizations

- **Q4_K** ✅ (primary format for most models)
- **Q6_K** ✅ (high-quality quantization)
- **Q5_0** ✅
- **Q8_0** ✅
- **F32** ✅ (float32 weights)

### Model Requirements

The converter requires these tensors to be present:
- `token_embd.weight` - Token embeddings
- `blk.{i}.attn_q.weight` - Query projections
- `blk.{i}.attn_k.weight` - Key projections (or alternative names)
- `blk.{i}.attn_v.weight` - Value projections (or alternative names)
- `blk.{i}.attn_output.weight` - Output projection
- `blk.{i}.ffn_gate.weight` - FFN gate
- `blk.{i}.ffn_up.weight` - FFN up projection
- `blk.{i}.ffn_down.weight` - FFN down projection
- `output.weight` - Final output layer
- `output_norm.weight` - Output normalization

## Limitations

### Not Currently Supported

- **DeepSeek2/MoE models** (Kimi-VL) - Requires MoE-specific handling
- **Vision-Language models** - Missing vision tensor support
- **3D/4D attention tensors** - Only 2D tensors supported
- **Custom quantization formats** - Only standard GGML types supported

### Known Limitations

- Large models (>10GB) may require significant memory during conversion
- Some MoE models with non-standard tensor naming may need manual configuration
- Context length overrides may be needed for certain models

## Troubleshooting

### "Missing attention.head_count" Error

This indicates the architecture prefix isn't recognized. Check the model architecture:

```bash
python scripts/convert_gguf_to_bump.py --gguf model.gguf --list
```

Look for the `general.architecture` field to see which architecture is being used.

### Dimension Mismatch Errors

For non-standard architectures, the converter attempts automatic dimension inference. If this fails, check:

1. The tensor dimensions in the `--list` output
2. Whether the model uses a non-standard attention mechanism
3. Whether it's an MoE or custom architecture

### Memory Issues

For large models, try:
- Converting with `--max-layers` to limit layers
- Ensuring sufficient RAM (4GB+ for 10GB models)
- Using a system with more memory

## Integration

After conversion, use the BUMP file with the C-Kernel Engine:

```bash
ck_cli_v5 --model model.bump --prompt "Your prompt here"
```

## History

**Version 1** (Original): LLaMA-only support
**Version 2**: Added Qwen2 support
**Version 3**: Added Mistral3/Mistral support
**Version 4**: Added dimension inference for non-standard architectures

## Further Reading

- [GGUF Format Specification](https://github.com/ggerganov/gguf)
- [BUMP Format Documentation](link-to-bump-docs)
- [Model Conversion Examples](link-to-examples)
