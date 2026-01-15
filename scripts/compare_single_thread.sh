#!/bin/bash
# Compare single-thread performance: CK-Engine vs llama.cpp

MODEL_DIR="$HOME/.cache/ck-engine-v6.5/models/Qwen--Qwen2-0.5B-Instruct-GGUF"
MODEL_NAME=$(basename "$MODEL_DIR")
GGUF=$(find "$MODEL_DIR" -name "*.gguf" | head -1)
LLAMA_CLI="$HOME/llama.cpp/build/bin/llama-cli"

echo "=== Single-Thread Performance Comparison ==="
echo ""

# Test CK-Engine
echo "1. CK-Engine (OMP_NUM_THREADS=1)"
echo "---"
export OMP_NUM_THREADS=1
time ./build/ck-cli-v6.5 --model "$MODEL_NAME" --max-tokens 100 --prompt "Generate a story" 2>&1 | \
  grep -E "(Hardware|total:|tok/s)"
echo ""

# Test llama.cpp if available
if [ -f "$LLAMA_CLI" ] && [ -n "$GGUF" ]; then
    echo "2. llama.cpp (OMP_NUM_THREADS=1)"
    echo "---"
    export OMP_NUM_THREADS=1
    time $LLAMA_CLI -m "$GGUF" -p "Generate a story" -n 100 --no-display-prompt 2>&1 | \
      grep -E "(time|first token|generation)"
    echo ""
else
    echo "2. llama.cpp not found or GGUF not available"
    echo ""
fi

echo "=== Analysis ==="
echo "Focus on the tok/s or tokens/sec metric"
echo "Lower time = better performance"
echo ""
