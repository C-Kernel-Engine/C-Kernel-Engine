#!/bin/bash
# Debug CLI issues

echo "=== Debug CLI Build ==="
echo ""

echo "1. Check if CLI exists:"
if [ -f "./build/ck-cli-v6.5" ]; then
    echo "✓ CLI exists: ./build/ck-cli-v6.5"
    ls -lh ./build/ck-cli-v6.5
else
    echo "✗ CLI NOT found at ./build/ck-cli-v6.5"
    echo ""
    echo "Build with:"
    echo "  git pull origin main"
    echo "  make clean"
    echo "  make ck-cli-v6.5"
    exit 1
fi

echo ""
echo "2. Check model exists:"
MODEL="qwen2-0_5b-instruct-q4_k_m"
MODEL_DIR="$HOME/.cache/ck-engine-v6.5/models/$MODEL"

if [ -d "$MODEL_DIR" ]; then
    echo "✓ Model directory exists: $MODEL_DIR"
    ls -lh "$MODEL_DIR"
else
    echo "✗ Model NOT found at: $MODEL_DIR"
    echo ""
    echo "Find models:"
    ls -la "$HOME/.cache/ck-engine-v6.5/models/" 2>/dev/null || echo "No models directory"
    echo ""
    echo "Available models:"
    find "$HOME/.cache/ck-engine-v6.5" -name "*.gguf" -o -name "*.so" | head -5
fi

echo ""
echo "3. Test CLI help:"
./build/ck-cli-v6.5 --help 2>&1 | head -20
