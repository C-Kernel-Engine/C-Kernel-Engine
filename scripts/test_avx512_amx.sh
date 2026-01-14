#!/bin/bash
# scripts/test_avx512_amx.sh
#
# AVX-512 and AMX capability test and benchmark
# Run this on the remote AVX-512/AMX machine
#
# Usage:
#   ./scripts/test_avx512_amx.sh          # Full test
#   ./scripts/test_avx512_amx.sh detect   # Just detect capabilities
#   ./scripts/test_avx512_amx.sh bench    # Just benchmark

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$ROOT_DIR/build"
RESULTS_DIR="$ROOT_DIR/test_results"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

mkdir -p "$RESULTS_DIR"

# =============================================================================
# CPU Feature Detection
# =============================================================================

detect_features() {
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║           AVX-512 / AMX Capability Detection             ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo ""

    # CPU Model
    echo "CPU: $(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)"
    echo "Cores: $(nproc)"
    echo ""

    # Cache sizes
    echo "Cache Hierarchy:"
    lscpu | grep -E "L1d|L1i|L2|L3" | while read line; do
        echo "  $line"
    done
    echo ""

    # AVX-512 variants
    echo "AVX-512 Features:"
    AVX512_FEATURES="avx512f avx512dq avx512cd avx512bw avx512vl avx512_vnni avx512_bf16 avx512_vbmi avx512_vbmi2"
    for feat in $AVX512_FEATURES; do
        if grep -q "$feat" /proc/cpuinfo 2>/dev/null; then
            echo "  ✅ $feat"
        else
            echo "  ❌ $feat"
        fi
    done
    echo ""

    # AMX features
    echo "AMX Features:"
    AMX_FEATURES="amx_tile amx_bf16 amx_int8"
    HAS_AMX=0
    for feat in $AMX_FEATURES; do
        if grep -q "$feat" /proc/cpuinfo 2>/dev/null; then
            echo "  ✅ $feat"
            HAS_AMX=1
        else
            echo "  ❌ $feat"
        fi
    done
    echo ""

    # Summarize what kernels will be used
    echo "Kernel Dispatch Summary:"

    if grep -q "avx512_vnni\|avx512vnni" /proc/cpuinfo 2>/dev/null; then
        echo "  Q4_K x Q8_K → VNNI (optimal)"
    elif grep -q "avx512f" /proc/cpuinfo 2>/dev/null; then
        echo "  Q4_K x Q8_K → AVX2 fallback (no VNNI)"
    fi

    if grep -q "avx512f" /proc/cpuinfo 2>/dev/null; then
        echo "  Q5_0 x Q8_0 → AVX-512F (optimal)"
        echo "  Q6_K x Q8_K → AVX-512F (optimal)"
        echo "  Q8_0 x FP32 → AVX-512F (optimal)"
    fi

    if [ "$HAS_AMX" = "1" ]; then
        echo "  AMX available → Can implement tile-based GEMM (8-16x speedup potential)"
    fi
    echo ""

    # Save to file
    local outfile="$RESULTS_DIR/cpu_features_$TIMESTAMP.txt"
    {
        echo "CPU Feature Detection - $TIMESTAMP"
        echo "CPU: $(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)"
        grep -E "avx|amx|vnni|bf16" /proc/cpuinfo | head -1
    } > "$outfile"
    echo "Saved to: $outfile"
}

# =============================================================================
# Kernel Benchmarks
# =============================================================================

run_benchmark() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║              AVX-512 Kernel Benchmark                    ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo ""

    # Check if CLI exists
    CLI="$BUILD_DIR/ck-cli-v6.5"
    if [ ! -f "$CLI" ]; then
        echo "Building CLI..."
        cd "$ROOT_DIR"
        make ck-cli-v6.5
    fi

    MODEL="${MODEL:-qwen2-0_5b-instruct-q4_k_m}"
    MAX_TOKENS="${MAX_TOKENS:-50}"

    echo "Model: $MODEL"
    echo "Tokens: $MAX_TOKENS"
    echo ""

    # Run benchmark
    echo "=== CK-Engine Performance ==="
    "$CLI" --model "$MODEL" --prompt "Hello" --max-tokens "$MAX_TOKENS" 2>&1 | tee "$RESULTS_DIR/ck_bench_$TIMESTAMP.txt"

    # Compare with llama.cpp if available
    LLAMA_CLI="$ROOT_DIR/llama.cpp/build/bin/llama-cli"
    if [ -f "$LLAMA_CLI" ]; then
        echo ""
        echo "=== llama.cpp Performance ==="
        GGUF_PATH=$(find ~/.cache/ck-engine-v6.5/models -name "*.gguf" -type f 2>/dev/null | head -1)
        if [ -n "$GGUF_PATH" ]; then
            "$LLAMA_CLI" -m "$GGUF_PATH" -p "Hello" -n "$MAX_TOKENS" --no-display-prompt 2>&1 | tee "$RESULTS_DIR/llama_bench_$TIMESTAMP.txt"
        fi
    fi

    echo ""
    echo "Results saved to: $RESULTS_DIR/*_bench_$TIMESTAMP.txt"
}

# =============================================================================
# DRAM/Cache Analysis
# =============================================================================

run_dram_analysis() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║              DRAM/Cache Access Analysis                  ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo ""

    CLI="$BUILD_DIR/ck-cli-v6.5"
    MODEL="${MODEL:-qwen2-0_5b-instruct-q4_k_m}"

    if ! command -v perf &> /dev/null; then
        echo "ERROR: perf not found"
        return 1
    fi

    echo "Measuring cache/memory access patterns..."
    echo ""

    # Standard cache events
    perf stat -e \
        cycles,\
        instructions,\
        LLC-loads,\
        LLC-load-misses,\
        LLC-stores,\
        LLC-store-misses,\
        L1-dcache-loads,\
        L1-dcache-load-misses \
        -o "$RESULTS_DIR/perf_cache_$TIMESTAMP.txt" \
        -- "$CLI" --model "$MODEL" --prompt "Hello" --max-tokens 20 2>/dev/null

    echo "=== Cache Analysis Results ==="
    cat "$RESULTS_DIR/perf_cache_$TIMESTAMP.txt"

    # Extract key metrics
    LLC_MISSES=$(grep "LLC-load-misses" "$RESULTS_DIR/perf_cache_$TIMESTAMP.txt" | awk '{print $1}' | tr -d ',')
    echo ""
    echo "Key Metric: LLC-load-misses = $LLC_MISSES"
    echo "(Lower is better - indicates less DRAM access)"
    echo ""
    echo "Results saved to: $RESULTS_DIR/perf_cache_$TIMESTAMP.txt"
}

# =============================================================================
# VTune Analysis (if available)
# =============================================================================

run_vtune() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║              Intel VTune Analysis                        ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo ""

    if ! command -v vtune &> /dev/null; then
        echo "VTune not found. Try:"
        echo "  source /opt/intel/oneapi/setvars.sh"
        return 1
    fi

    # Run VTune hotspots
    ./scripts/vtune_profile.sh hotspots

    # Run VTune memory analysis
    ./scripts/vtune_profile.sh memory-access
}

# =============================================================================
# Full Test Suite
# =============================================================================

run_full() {
    detect_features
    run_benchmark
    run_dram_analysis

    # VTune if available
    if command -v vtune &> /dev/null; then
        run_vtune
    fi

    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║                    TEST COMPLETE                         ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo ""
    echo "All results saved to: $RESULTS_DIR/"
    echo ""
    echo "To commit and push results:"
    echo "  git add test_results/ vtune_results/"
    echo "  git commit -m 'AVX-512/AMX test results $TIMESTAMP'"
    echo "  git push"
}

# =============================================================================
# Main
# =============================================================================

case "${1:-full}" in
    detect|info)
        detect_features
        ;;
    bench|benchmark)
        run_benchmark
        ;;
    dram|cache|memory)
        run_dram_analysis
        ;;
    vtune)
        run_vtune
        ;;
    full|all)
        run_full
        ;;
    *)
        echo "Usage: $0 <command>"
        echo ""
        echo "Commands:"
        echo "  detect    - Detect CPU features (AVX-512, AMX)"
        echo "  bench     - Run kernel benchmarks"
        echo "  dram      - Analyze DRAM/cache access"
        echo "  vtune     - Run VTune analysis (if available)"
        echo "  full      - Run all tests (default)"
        echo ""
        echo "Environment variables:"
        echo "  MODEL=...      Model name"
        echo "  MAX_TOKENS=... Tokens to generate"
        ;;
esac
