#!/bin/bash
#
# benchmark_avx512.sh - Comprehensive AVX-512/AMX benchmark
#
# Tests:
# 1. Kernel dispatch verification (which kernels are being used?)
# 2. Single-core performance baseline
# 3. Multi-core parallel scaling
# 4. Comparison with llama.cpp
# 5. DRAM/cache analysis
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/build"
RESULTS_DIR="$PROJECT_ROOT/test_results"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

mkdir -p "$RESULTS_DIR"

MODEL="qwen2-0_5b-instruct-q4_k_m"
CLI="$BUILD_DIR/ck-cli-v6.5"

# Colors
GREEN='\033[92m'
BLUE='\033[96m'
YELLOW='\033[93m'
CYAN='\033[36m'
BOLD='\033[1m'
RESET='\033[0m'

echo -e "${BOLD}${CYAN}======================================================================${RESET}"
echo -e "${BOLD}${CYAN}      AVX-512/AMX Comprehensive Benchmark${RESET}"
echo -e "${BOLD}${CYAN}======================================================================${RESET}"
echo ""

# =============================================================================
# Check 1: Kernel Dispatch Verification
# =============================================================================
echo -e "${BOLD}1. Kernel Dispatch Verification${RESET}"
echo "Building with verbose kernel logging..."
echo ""

# Rebuild with KERNEL_DEBUG=1 to see dispatch
cd "$PROJECT_ROOT"
make clean >/dev/null 2>&1
make KERNEL_DEBUG=1 ck-cli-v6.5 >/dev/null 2>&1

echo "Testing single token decode (should use VNNI/AMX for Q4_K x Q8_K)..."
echo ""

# Run with KERNEL_DEBUG to see which kernel is called
KERNEL_LOG="$RESULTS_DIR/kernel_dispatch_$TIMESTAMP.txt"
{
    echo "=== Kernel Dispatch Log ==="
    echo ""
    echo "Running: Single token decode"
    echo ""
} > "$KERNEL_LOG"

"$CLI" --model "$MODEL" --max-tokens 1 --prompt "Hi" 2>&1 | tee -a "$KERNEL_LOG"

echo ""
echo -e "${GREEN}Kernel dispatch log saved to: $KERNEL_LOG${RESET}"
echo ""

# =============================================================================
# Check 2: Single-Core Baseline
# =============================================================================
echo -e "${BOLD}2. Single-Core Performance Baseline${RESET}"
echo "Disabling parallelization to measure single-core performance..."
echo ""

# Set OMP_NUM_THREADS=1 for single-core
export OMP_NUM_THREADS=1

SINGLE_CORE_LOG="$RESULTS_DIR/single_core_$TIMESTAMP.txt"
{
    echo "=== Single-Core Performance ==="
    echo "OMP_NUM_THREADS=1"
    echo "Model: $MODEL"
    echo "Timestamp: $TIMESTAMP"
    echo ""
} > "$SINGLE_CORE_LOG"

for tokens in 10 20 50 100; do
    echo "Testing $tokens tokens (single-core)..."
    {
        echo "--- $tokens tokens ---"
    } >> "$SINGLE_CORE_LOG"

    "$CLI" --model "$MODEL" --max-tokens "$tokens" --prompt "Hello" 2>&1 | \
        grep -E "(prompt eval|decode:|total:)" | \
        tee -a "$SINGLE_CORE_LOG"
    echo "" >> "$SINGLE_CORE_LOG"
done

echo -e "${GREEN}Single-core results saved to: $SINGLE_CORE_LOG${RESET}"
echo ""

# =============================================================================
# Check 3: Multi-Core Scaling
# =============================================================================
echo -e "${BOLD}3. Multi-Core Parallel Scaling${RESET}"
echo "Testing with 1, 2, 4, 8, 16, 32 cores..."
echo ""

MULTI_CORE_LOG="$RESULTS_DIR/multi_core_$TIMESTAMP.txt"
{
    echo "=== Multi-Core Scaling Test ==="
    echo "Model: $MODEL"
    echo "Timestamp: $TIMESTAMP"
    echo ""
} > "$MULTI_CORE_LOG"

for cores in 1 2 4 8 16 32; do
    if [ $cores -gt $(nproc) ]; then
        break
    fi

    export OMP_NUM_THREADS=$cores

    echo "Testing with $cores cores..."
    {
        echo "--- OMP_NUM_THREADS=$cores ---"
    } >> "$MULTI_CORE_LOG"

    "$CLI" --model "$MODEL" --max-tokens 50 --prompt "Hello" 2>&1 | \
        grep -E "(prompt eval|decode:|total:)" | \
        tee -a "$MULTI_CORE_LOG"
    echo "" >> "$MULTI_CORE_LOG"
done

echo -e "${GREEN}Multi-core results saved to: $MULTI_CORE_LOG${RESET}"
echo ""

# Reset to all cores
export OMP_NUM_THREADS=$(nproc)

# =============================================================================
# Check 4: Compare with llama.cpp
# =============================================================================
echo -e "${BOLD}4. Performance Comparison with llama.cpp${RESET}"

LLAMA_CLI="$PROJECT_ROOT/llama.cpp/build/bin/llama-cli"
if [ -f "$LLAMA_CLI" ]; then
    echo "llama.cpp found. Running comparison..."
    echo ""

    GGUF_PATH=$(find ~/.cache/ck-engine-v6.5/models -name "*.gguf" -type f 2>/dev/null | head -1)

    COMPARE_LOG="$RESULTS_DIR/compare_$TIMESTAMP.txt"
    {
        echo "=== CK-Engine vs llama.cpp Comparison ==="
        echo "Model: $MODEL"
        echo "GGUF: $GGUF_PATH"
        echo "Timestamp: $TIMESTAMP"
        echo ""
    } > "$COMPARE_LOG"

    # CK-Engine (multi-core)
    echo "CK-Engine (32 cores)..."
    export OMP_NUM_THREADS=32
    {
        echo "--- CK-Engine (OMP_NUM_THREADS=32) ---"
    } >> "$COMPARE_LOG"
    "$CLI" --model "$MODEL" --max-tokens 100 --prompt "Hello" 2>&1 | \
        grep -E "(prompt eval|decode:|total:)" | \
        tee -a "$COMPARE_LOG"
    echo "" >> "$COMPARE_LOG"

    # llama.cpp
    echo "llama.cpp..."
    {
        echo "--- llama.cpp ---"
    } >> "$COMPARE_LOG"
    "$LLAMA_CLI" -m "$GGUF_PATH" -p "Hello" -n 100 --no-display-prompt 2>&1 | \
        grep -E "(time|first token|prompt|generation)" | \
        tee -a "$COMPARE_LOG"

    echo ""
    echo -e "${GREEN}Comparison saved to: $COMPARE_LOG${RESET}"
else
    echo -e "${YELLOW}llama.cpp not found. Skipping comparison.${RESET}"
    echo "To build: cd llama.cpp && mkdir -p build && cd build && cmake .. -DGGML_AVX512=ON && make -j"
    echo ""
fi

# =============================================================================
# Check 5: DRAM/Cache Analysis
# =============================================================================
echo -e "${BOLD}5. DRAM/Cache Analysis${RESET}"
echo "Using perf to measure cache misses and DRAM access..."
echo ""

if command -v perf &> /dev/null; then
    CACHE_LOG="$RESULTS_DIR/cache_analysis_$TIMESTAMP.txt"
    {
        echo "=== Cache/DRAM Analysis ==="
        echo "Model: $MODEL"
        echo "Timestamp: $TIMESTAMP"
        echo ""
    } > "$CACHE_LOG"

    echo "Running perf stat (this may take a while)..."
    perf stat -e \
        cycles,\
        instructions,\
        LLC-loads,\
        LLC-load-misses,\
        LLC-stores,\
        LLC-store-misses,\
        L1-dcache-loads,\
        L1-dcache-load-misses,\
        cache-references,\
        cache-misses \
        -o "$CACHE_LOG" \
        -- "$CLI" --model "$MODEL" --prompt "Hello" --max-tokens 20 2>/dev/null

    echo -e "${GREEN}Cache analysis saved to: $CACHE_LOG${RESET}"
    echo ""
    echo "Key metrics:"
    grep -E "LLC-load-misses|cache-misses" "$CACHE_LOG" || true
else
    echo -e "${YELLOW}perf not found. Skipping cache analysis.${RESET}"
    echo "Install with: sudo apt-get install linux-tools-generic"
    echo ""
fi

# =============================================================================
# Summary Report
# =============================================================================
echo ""
echo -e "${BOLD}${CYAN}======================================================================${RESET}"
echo -e "${BOLD}${CYAN}                      Summary & Next Steps${RESET}"
echo -e "${BOLD}${CYAN}======================================================================${RESET}"
echo ""

echo "All results saved to: $RESULTS_DIR/"
echo ""
echo "Key files to check:"
echo "  - $KERNEL_LOG"
echo "  - $SINGLE_CORE_LOG"
echo "  - $MULTI_CORE_LOG"
if [ -f "$COMPARE_LOG" ]; then
    echo "  - $COMPARE_LOG"
fi
echo "  - $CACHE_LOG"
echo ""

echo "Analysis:"
echo ""
echo "1. If single-core is slow:"
echo "   - Check kernel dispatch (AMX/VNNI should be used)"
echo "   - Profile with VTune: vtune -collect hotspots -- $CLI --model $MODEL ..."
echo "   - Check for quantization overhead"
echo ""
echo "2. If multi-core scaling is poor:"
echo "   - Need to parallelize the decode loop"
echo "   - Current implementation is sequential"
echo "   - Add OpenMP parallel for() loops"
echo ""
echo "3. If llama.cpp is faster:"
echo "   - Their parallelization is mature"
echo "   - We need multi-core support"
echo "   - Focus on parallel decode first"
echo ""

echo -e "${GREEN}Benchmark complete!${RESET}"
echo ""
