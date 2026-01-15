#!/bin/bash

echo "=== Simple Performance Test ==="
echo ""

# Find the actual model name
MODEL_DIR=$(ls -d ~/.cache/ck-engine-v6.5/models/Qwen* 2>/dev/null | head -1)
if [ -z "$MODEL_DIR" ]; then
    echo "Model not found!"
    exit 1
fi

# Extract just the directory name
MODEL_NAME=$(basename "$MODEL_DIR")
echo "Using model: $MODEL_NAME"
echo ""

# Test 1: Basic run
echo "1. Basic run (default threads):"
./build/ck-cli-v6.5 --model "$MODEL_NAME" --max-tokens 10 --prompt "Hello" 2>&1 | grep -E "(Hardware|total:|tok/s)"
echo ""

# Test 2: Single thread
echo "2. Single thread (OMP_NUM_THREADS=1):"
(export OMP_NUM_THREADS=1; ./build/ck-cli-v6.5 --model "$MODEL_NAME" --max-tokens 10 --prompt "Hello" 2>&1) | grep -E "(Hardware|total:|tok/s)"
echo ""

# Test 3: All threads
echo "3. All threads (OMP_NUM_THREADS=$(nproc)):"
(export OMP_NUM_THREADS=$(nproc); ./build/ck-cli-v6.5 --model "$MODEL_NAME" --max-tokens 10 --prompt "Hello" 2>&1) | grep -E "(Hardware|total:|tok/s)"
echo ""

# Test 4: perf if available
if command -v perf &> /dev/null; then
    echo "4. With perf (cycles, instructions, cache misses):"
    perf stat -e cycles,instructions,LLC-load-misses,L1-dcache-load-misses \
        ./build/ck-cli-v6.5 --model "$MODEL_NAME" --max-tokens 20 --prompt "Hello" 2>&1 | \
        grep -E "(Hardware|total:|cycles|LLC-load-misses|L1-dcache-load-misses)"
else
    echo "4. perf not available"
fi
