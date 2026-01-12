#!/bin/bash
# Auto-convert any GGUF file (Devstral, SmolLM, Qwen, LLaMA) to BUMP

set -e

GGUF_FILE="$1"
OUTPUT_FILE="${2:-${GGUF_FILE%.gguf}.bump}"

if [ -z "$GGUF_FILE" ]; then
    echo "Usage: $0 <gguf_file> [output.bump]"
    echo ""
    echo "Examples:"
    echo "  $0 Devstral-Small-2-24B-Instruct-2512-Q4_K_M.gguf"
    echo "  $0 SmolLM-1.7B-Instruct.Q4_K_M.gguf"
    echo "  $0 qwen2.5-3b-instruct-q4_k_m.gguf"
    exit 1
fi

if [ ! -f "$GGUF_FILE" ]; then
    echo "❌ File not found: $GGUF_FILE"
    exit 1
fi

echo "📖 Extracting metadata from $GGUF_FILE..."

# Get key tensors
TENSOR_LIST=$(python scripts/v6/convert_gguf_to_bump_v6.py --gguf "$GGUF_FILE" --list 2>&1)

# Extract values
VOCAB_SIZE=$(echo "$TENSOR_LIST" | grep "token_embd.weight" | sed -n 's/.*dims=([^,]*),.*/\1/p' | head -1)
EMBED_DIM=$(echo "$TENSOR_LIST" | grep "token_embd.weight" | sed -n 's/.*dims=([^,]*), ([^)]*).*/\2/p' | head -1)
NUM_LAYERS=$(echo "$TENSOR_LIST" | grep "^  - blk\." | cut -d'.' -f2 | sort -u | wc -l | tr -d ' ')

# Calculate heads from attention output
ATTN_OUTPUT=$(echo "$TENSOR_LIST" | grep "blk.0.attn_output.weight" | head -1)
HEAD_DIM=$(echo "$ATTN_OUTPUT" | sed -n 's/.*dims=([^,]*),.*/\1/p')
OUTPUT_DIM=$(echo "$ATTN_OUTPUT" | sed -n 's/.*dims=([^,]*), ([^)]*).*/\2/p')

# Calculate num_heads and head_dim
NUM_HEADS=$((OUTPUT_DIM / HEAD_DIM))

echo "   Architecture: $(echo "$TENSOR_LIST" | grep "arch=" | cut -d' ' -f3)"
echo "   Vocab size: $VOCAB_SIZE"
echo "   Hidden size: $EMBED_DIM"
echo "   Layers: $NUM_LAYERS"
echo "   Heads: $NUM_HEADS"
echo "   Head dim: $HEAD_DIM"

# Create config file
cat > /tmp/config.json <<EOF
{
  "vocab_size": $VOCAB_SIZE,
  "context_window": 4096,
  "hidden_size": $EMBED_DIM,
  "num_hidden_layers": $NUM_LAYERS,
  "num_attention_heads": $NUM_HEADS
}
EOF

echo ""
echo "🔄 Converting to BUMP format..."
python scripts/v6/convert_gguf_to_bump_v6.py \
    --gguf "$GGUF_FILE" \
    --output "$OUTPUT_FILE" \
    --config-out /tmp/config.json

rm /tmp/config.json

echo ""
echo "✅ Conversion complete: $OUTPUT_FILE"
echo ""
echo "Next steps:"
echo "  ck-engine-v6 -m $OUTPUT_FILE -p 'Hello'"
