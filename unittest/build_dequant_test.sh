#!/bin/bash
# Build the dequantization comparison test

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LLAMA_DIR="$PROJECT_DIR/llama.cpp"

echo "Building dequantization test..."
echo "  Project: $PROJECT_DIR"
echo "  llama.cpp: $LLAMA_DIR"

# Check if llama.cpp exists
if [ ! -d "$LLAMA_DIR" ]; then
    echo "ERROR: llama.cpp not found at $LLAMA_DIR"
    echo "  Clone it with: git clone https://github.com/ggerganov/llama.cpp.git"
    exit 1
fi

# Build llama.cpp if needed
if [ ! -f "$LLAMA_DIR/build/ggml/src/libggml.so" ] && [ ! -f "$LLAMA_DIR/build/src/libllama.so" ]; then
    echo "Building llama.cpp..."
    cd "$LLAMA_DIR"
    cmake -B build -DGGML_NATIVE=OFF
    cmake --build build -j$(nproc)
    cd "$SCRIPT_DIR"
fi

# Find ggml library
GGML_LIB=""
if [ -f "$LLAMA_DIR/build/ggml/src/libggml.so" ]; then
    GGML_LIB="$LLAMA_DIR/build/ggml/src"
elif [ -f "$LLAMA_DIR/build/src/libggml.so" ]; then
    GGML_LIB="$LLAMA_DIR/build/src"
fi

echo "  GGML lib: $GGML_LIB"

# Compile standalone test (no external deps needed - has all dequant code inline)
echo "Compiling standalone dequant test..."
gcc -O2 -Wall -Wextra \
    -o "$SCRIPT_DIR/test_dequant_vs_llamacpp" \
    "$SCRIPT_DIR/test_dequant_vs_llamacpp.c" \
    -lm

echo ""
echo "Build complete! Run with:"
echo "  ./unittest/test_dequant_vs_llamacpp"
