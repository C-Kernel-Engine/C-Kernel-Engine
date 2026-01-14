#!/bin/bash
#
# test_performance.sh - Simple performance testing
#

set -e

CLI="./build/ck-cli-v6.5"
MODEL="qwen2-0_5b-instruct-q4_k_m"
PROMPT="Hello"
TOKENS=50

echo "=========================================="
echo "  Performance Test"
echo "=========================================="
echo ""

# Check if CLI exists
if [ ! -f "$CLI" ]; then
    echo "ERROR: CLI not found. Build first with: make ck-cli-v6.5"
    exit 1
fi

echo "Testing: $MODEL"
echo "Tokens: $TOKENS"
echo "Prompt: $PROMPT"
echo ""

# Test 1: Single thread
echo "1. Single-thread performance (OMP_NUM_THREADS=1)"
echo "---"
OMP_NUM_THREADS=1 $CLI --model $MODEL --max-tokens $TOKENS --prompt "$PROMPT" 2>&1 | grep -E "(Hardware|total:|decode:)"
echo ""

# Test 2: All threads
echo "2. Multi-thread performance (OMP_NUM_THREADS=$(nproc))"
echo "---"
OMP_NUM_THREADS=$(nproc) $CLI --model $MODEL --max-tokens $TOKENS --prompt "$PROMPT" 2>&1 | grep -E "(Hardware|total:|decode:)"
echo ""

# Test 3: If perf available
if command -v perf &> /dev/null; then
    echo "3. Cache analysis with perf"
    echo "---"
    echo "Running perf stat..."
    perf stat -e cycles,instructions,LLC-load-misses,L1-dcache-load-misses \
        $CLI --model $MODEL --max-tokens $TOKENS --prompt "$PROMPT" 2>&1 | \
        grep -E "(Hardware|total:|cycles|LLC-load-misses|L1-dcache-load-misses)"
    echo ""
fi

# Test 4: If VTune available
if command -v vtune &> /dev/null; then
    echo "4. VTune hotspots analysis"
    echo "---"
    echo "Run VTune manually:"
    echo "  vtune -collect hotspots -result-dir vtune_results -- $CLI --model $MODEL --max-tokens $TOKENS --prompt '$PROMPT'"
    echo "  vtune -report -result-dir vtune_results"
    echo ""
fi

echo "=========================================="
echo "  Analysis"
echo "=========================================="
echo ""
echo "Check the output above for:"
echo "  - Hardware: Should show 'AVX-512' not just 'AVX'"
echo "  - Kernel: Should show 'gemv_q4_k_q8_k_vnni' not 'gemm_avx'"
echo "  - tok/s: Compare single vs multi-thread"
echo ""
