#!/bin/bash
# Convert Devstral, SmolLM, or Qwen GGUF to BUMP

GGUF_FILE="$1"
OUTPUT="${2:-${GGUF_FILE%.gguf}.bump}"

if [ -z "$GGUF_FILE" ]; then
    echo "Usage: $0 <gguf_file> [output.bump]"
    echo ""
    echo "Supported models:"
    echo "  - Devstral-Small-2-24B-Instruct-2512-Q4_K_M.gguf"
    echo "  - SmolLM-1.7B-Instruct.Q4_K_M.gguf"
    echo "  - qwen2.5-3b-instruct-q4_k_m.gguf"
    exit 1
fi

# Get filename without path
BASENAME=$(basename "$GGUF_FILE")

# Auto-detect model type and set config
case "$BASENAME" in
    Devstral*)
        echo "📦 Detected: Devstral"
        CONFIG='{
            "vocab_size": 131072,
            "context_window": 4096,
            "hidden_size": 5120,
            "num_hidden_layers": 40,
            "num_attention_heads": 32
        }'
        ;;
    SmolLM*)
        echo "📦 Detected: SmolLM"
        CONFIG='{
            "vocab_size": 49152,
            "context_window": 8192,
            "hidden_size": 2048,
            "num_hidden_layers": 30,
            "num_attention_heads": 32
        }'
        ;;
    qwen2.5*)
        echo "📦 Detected: Qwen2.5"
        CONFIG='{
            "vocab_size": 152064,
            "context_window": 32768,
            "hidden_size": 4096,
            "num_hidden_layers": 28,
            "num_attention_heads": 32
        }'
        ;;
    *)
        echo "❌ Unknown model: $BASENAME"
        echo "Supported: Devstral, SmolLM, Qwen2.5"
        exit 1
        ;;
esac

echo "🔄 Converting $BASENAME to BUMP..."

# Write config
echo "$CONFIG" > /tmp/config.json

# Convert
python scripts/v4/convert_gguf_to_bump_v4.py \
    --gguf "$GGUF_FILE" \
    --output "$OUTPUT" \
    --config-out /tmp/config.json

rm /tmp/config.json

echo ""
echo "✅ Done! Output: $OUTPUT"
echo ""
echo "Test it:"
echo "  ck_cli_v5 --model $OUTPUT --prompt 'Hello'"
