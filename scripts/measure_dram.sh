#!/bin/bash
# scripts/measure_dram.sh
#
# Measure DRAM/cache access patterns for fusion validation
#
# Usage:
#   ./scripts/measure_dram.sh              # Measure default decode
#   ./scripts/measure_dram.sh --fused      # Measure with fusion enabled
#   ./scripts/measure_dram.sh --compare    # Side-by-side comparison

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$ROOT_DIR/build"
RESULTS_DIR="$ROOT_DIR/test_results"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

CLI="$BUILD_DIR/ck-cli-v6.5"
MODEL="${MODEL:-qwen2-0_5b-instruct-q4_k_m}"
PROMPT="${PROMPT:-Hello}"
MAX_TOKENS="${MAX_TOKENS:-20}"

mkdir -p "$RESULTS_DIR"

# Check perf availability
if ! command -v perf &> /dev/null; then
    echo "ERROR: perf not found. Install linux-tools-generic"
    exit 1
fi

# Check if we can access perf counters
if ! perf stat -e cycles true 2>/dev/null; then
    echo "WARNING: perf counters may require: sudo sysctl -w kernel.perf_event_paranoid=-1"
fi

measure_cache() {
    local label=$1
    local extra_flags=$2
    local output_file="$RESULTS_DIR/dram_${label}_${TIMESTAMP}.txt"

    echo ""
    echo "=============================================="
    echo "Measuring: $label"
    echo "=============================================="
    echo "Model:     $MODEL"
    echo "Tokens:    $MAX_TOKENS"
    echo "Output:    $output_file"
    echo ""

    # Core cache/memory events
    # LLC = Last Level Cache (L3) - misses here go to DRAM
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
        -o "$output_file" \
        -- "$CLI" --model "$MODEL" --prompt "$PROMPT" --max-tokens "$MAX_TOKENS" $extra_flags 2>/dev/null

    echo ""
    echo "=== Results ==="
    cat "$output_file"

    # Extract key metrics
    local llc_misses=$(grep "LLC-load-misses" "$output_file" | awk '{print $1}' | tr -d ',')
    local llc_loads=$(grep "LLC-loads" "$output_file" | awk '{print $1}' | tr -d ',')
    local cache_misses=$(grep "cache-misses" "$output_file" | awk '{print $1}' | tr -d ',')

    echo ""
    echo "=== Key Metrics ==="
    echo "LLC Load Misses (DRAM reads):  $llc_misses"
    echo "LLC Loads (L3 accesses):       $llc_loads"
    echo "Total Cache Misses:            $cache_misses"

    if [ -n "$llc_misses" ] && [ -n "$llc_loads" ] && [ "$llc_loads" != "0" ]; then
        local miss_rate=$(echo "scale=2; $llc_misses * 100 / $llc_loads" | bc 2>/dev/null || echo "N/A")
        echo "LLC Miss Rate:                 ${miss_rate}%"
    fi
}

run_compare() {
    echo "=== DRAM Access Comparison: Unfused vs Fused ==="
    echo ""

    # Measure unfused
    measure_cache "unfused" ""
    local unfused_file="$RESULTS_DIR/dram_unfused_${TIMESTAMP}.txt"

    # Measure fused (if flag exists)
    measure_cache "fused" "--fused"
    local fused_file="$RESULTS_DIR/dram_fused_${TIMESTAMP}.txt"

    echo ""
    echo "=============================================="
    echo "COMPARISON SUMMARY"
    echo "=============================================="

    local unfused_llc=$(grep "LLC-load-misses" "$unfused_file" 2>/dev/null | awk '{print $1}' | tr -d ',')
    local fused_llc=$(grep "LLC-load-misses" "$fused_file" 2>/dev/null | awk '{print $1}' | tr -d ',')

    if [ -n "$unfused_llc" ] && [ -n "$fused_llc" ] && [ "$unfused_llc" != "0" ]; then
        local reduction=$(echo "scale=2; ($unfused_llc - $fused_llc) * 100 / $unfused_llc" | bc 2>/dev/null || echo "N/A")
        echo "Unfused LLC Misses: $unfused_llc"
        echo "Fused LLC Misses:   $fused_llc"
        echo "DRAM Reduction:     ${reduction}%"

        if [ "$(echo "$reduction > 0" | bc 2>/dev/null)" = "1" ]; then
            echo ""
            echo "SUCCESS: Fusion reduced DRAM access!"
        fi
    else
        echo "Could not compare - check if --fused flag is implemented"
    fi
}

# Print system info
echo "=== System Info ==="
echo "CPU: $(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)"
echo "L3 Cache: $(lscpu | grep "L3 cache" | awk '{print $3}')"

# Check for AVX-512
if grep -q avx512f /proc/cpuinfo; then
    echo "AVX-512: YES"
else
    echo "AVX-512: NO (AVX/AVX2 only)"
fi

case "${1:-baseline}" in
    baseline|base)
        measure_cache "baseline" ""
        ;;
    fused)
        measure_cache "fused" "--fused"
        ;;
    compare|cmp)
        run_compare
        ;;
    *)
        echo ""
        echo "Usage: $0 <mode>"
        echo ""
        echo "  baseline   Measure unfused decode (default)"
        echo "  fused      Measure fused decode"
        echo "  compare    Side-by-side comparison"
        echo ""
        echo "Environment variables:"
        echo "  MODEL=...      Model name (default: qwen2-0_5b-instruct-q4_k_m)"
        echo "  MAX_TOKENS=... Tokens to generate (default: 20)"
        echo "  PROMPT=...     Input prompt (default: Hello)"
        ;;
esac
