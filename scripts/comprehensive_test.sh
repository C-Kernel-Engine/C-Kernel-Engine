#!/bin/bash
# Comprehensive performance test

set -e

CLI="./build/ck-cli-v6.5"
MODEL="Qwen--Qwen2-0.5B-Instruct-GGUF"
PROMPT="Hello! How are you?"
TOKENS=50

echo "=========================================="
echo "  Comprehensive Performance Test"
echo "=========================================="
echo ""

# Test 1: Hardware detection
echo "1. Hardware Detection"
echo "---"
$CLI --model $MODEL --max-tokens 1 --prompt "$PROMPT" 2>&1 | grep -E "(Hardware|Loading|Ready)"
echo ""

# Test 2: Single thread
echo "2. Single-thread performance (OMP_NUM_THREADS=1)"
echo "---"
OMP_NUM_THREADS=1 time $CLI --model $MODEL --max-tokens $TOKENS --prompt "$PROMPT" 2>&1 | grep -E "(Hardware|total:|tok/s)"
echo ""

# Test 3: All threads
echo "3. Multi-thread performance (OMP_NUM_THREADS=$(nproc))"
echo "---"
OMP_NUM_THREADS=$(nproc) time $CLI --model $MODEL --max-tokens $TOKENS --prompt "$PROMPT" 2>&1 | grep -E "(Hardware|total:|tok/s)"
echo ""

# Test 4: With perf (if available)
if command -v perf &> /dev/null; then
    echo "4. With perf stat"
    echo "---"
    echo "Running perf stat (cycles, instructions, cache misses)..."
    perf stat -e cycles,instructions,LLC-load-misses,L1-dcache-load-misses \
        $CLI --model $MODEL --max-tokens $TOKENS --prompt "$PROMPT" 2>&1 | \
        grep -E "(Hardware|total:|cycles|LLC-load-misses|L1-dcache-load-misses)"
    echo ""
else
    echo "4. perf not available"
fi

# Test 5: Compare with llama.cpp (if available)
LLAMA_CLI="$HOME/llama.cpp/build/bin/llama-cli"
if [ -f "$LLAMA_CLI" ]; then
    echo "5. llama.cpp comparison"
    echo "---"
    GGUF=$(find ~/.cache/ck-engine-v6.5/models -name "*.gguf" | head -1)
    if [ -n "$GGUF" ]; then
        echo "Using GGUF: $GGUF"
        $LLAMA_CLI -m "$GGUF" -p "$PROMPT" -n $TOKENS --no-display-prompt 2>&1 | \
            grep -E "(time|first token|generation)"
    fi
    echo ""
else
    echo "5. llama.cpp not found"
fi

echo "=========================================="
echo "  Analysis Complete"
echo "=========================================="
echo ""
echo "Key findings:"
echo "  - Hardware: Should show AVX-512"
echo "  - Single vs Multi-thread: Compare tok/s"
echo "  - perf: Look for LLC misses (DRAM access)"
echo ""
