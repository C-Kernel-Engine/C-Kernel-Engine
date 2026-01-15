#!/bin/bash
# Profile single-thread kernel performance

echo "=== Single-Thread Kernel Profiling ==="
echo ""

MODEL_DIR="$HOME/.cache/ck-engine-v6.5/models/Qwen--Qwen2-0.5B-Instruct-GGUF"
MODEL_NAME=$(basename "$MODEL_DIR")

# Test 1: CK-Engine performance
echo "1. CK-Engine Single-Thread Performance"
echo "OMP_NUM_THREADS=1"
export OMP_NUM_THREADS=1
time ./build/ck-cli-v6.5 --model "$MODEL_NAME" --max-tokens 100 --prompt "Generate a short story" 2>&1 | \
  grep -E "(Hardware|total:|tok/s)"
echo ""

# Test 2: llama.cpp single-thread
LLAMA_CLI="$HOME/llama.cpp/build/bin/llama-cli"
if [ -f "$LLAMA_CLI" ]; then
    echo "2. llama.cpp Single-Thread Performance"
    echo "OMP_NUM_THREADS=1"
    GGUF=$(find "$MODEL_DIR" -name "*.gguf" | head -1)
    if [ -n "$GGUF" ]; then
        OMP_NUM_THREADS=1 $LLAMA_CLI -m "$GGUF" -p "Generate a short story" -n 100 --no-display-prompt 2>&1 | \
          grep -E "(time|first token|generation)"
    fi
    echo ""
else
    echo "2. llama.cpp not found"
    echo ""
fi

# Test 3: Check kernel implementation
echo "3. Kernel Analysis"
echo "Looking for gemm_avx512 implementation..."
grep -n "gemm_avx512\|gemv_q4_k_q8_k" "$MODEL_DIR"/ck-kernel-inference.c | head -5
echo ""

# Test 4: Check if Q4_K x Q8_K kernels are used
echo "4. Kernel Dispatch"
echo "Checking which kernels are compiled..."
nm -C ./build/libckernel_engine.so 2>/dev/null | grep -i "gemv_q4_k" | head -5 || \
  nm -C ./build/ck-cli-v6.5 2>/dev/null | grep -i "gemv_q4_k" | head -5

echo ""
echo "=== Summary ==="
echo "Compare the tok/s between CK-Engine and llama.cpp"
echo "Focus on optimizing the slower kernel!"
