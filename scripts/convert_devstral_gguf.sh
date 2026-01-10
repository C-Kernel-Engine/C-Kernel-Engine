#!/bin/bash
# Simple script to convert Devstral/SmolLM/Qwen GGUF files to BUMP

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

echo "📖 Inspecting GGUF file..."
python scripts/v4/convert_gguf_to_bump_v4.py --gguf "$GGUF_FILE" --list 2>&1 | head -20

# Create a config file with the metadata
cat > /tmp/config.json <<EOF
{
  "vocab_size": 131072,
  "context_window": 4096,
  "hidden_size": 5120,
  "num_hidden_layers": 48,
  "num_attention_heads": 32,
  "intermediate_size": 32768
}
EOF

echo ""
echo "🔄 Converting GGUF to BUMP..."
python scripts/v4/convert_gguf_to_bump_v4.py \
    --gguf "$GGUF_FILE" \
    --output "$OUTPUT_FILE" \
    --config-out /tmp/config.json

echo ""
echo "✅ Conversion complete: $OUTPUT_FILE"
echo ""
echo "Next steps:"
echo "  ck_cli_v5 --model $OUTPUT_FILE --prompt 'Hello'"
