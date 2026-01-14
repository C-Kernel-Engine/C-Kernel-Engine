#!/bin/bash
# scripts/vtune_profile.sh
#
# VTune profiling script for CK-Engine performance analysis
#
# Usage:
#   ./scripts/vtune_profile.sh hotspots           # CPU hotspots
#   ./scripts/vtune_profile.sh memory-access      # Memory bandwidth/latency
#   ./scripts/vtune_profile.sh microarchitecture  # uArch stalls
#   ./scripts/vtune_profile.sh compare            # Compare CK vs llama.cpp

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$ROOT_DIR/build"
RESULTS_DIR="$ROOT_DIR/vtune_results"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# Check VTune
if ! command -v vtune &> /dev/null; then
    echo "ERROR: VTune not found. Source Intel oneAPI:"
    echo "  source /opt/intel/oneapi/setvars.sh"
    exit 1
fi

# Check AVX-512
HAS_AVX512=$(grep -q avx512f /proc/cpuinfo 2>/dev/null && echo "yes" || echo "no")
if [ "$HAS_AVX512" = "no" ]; then
    echo "NOTE: AVX-512 not available - profiling AVX/AVX2 kernels"
fi

# Default model and prompt
MODEL="${MODEL:-qwen2-0_5b-instruct-q4_k_m}"
PROMPT="${PROMPT:-Explain the theory of relativity in simple terms.}"
MAX_TOKENS="${MAX_TOKENS:-50}"

CLI="$BUILD_DIR/ck-cli-v6.5"
mkdir -p "$RESULTS_DIR"

run_vtune() {
    local analysis_type=$1
    local result_dir="$RESULTS_DIR/${analysis_type}_${TIMESTAMP}"

    echo ""
    echo "=============================================="
    echo "VTune Analysis: $analysis_type"
    echo "=============================================="
    echo "Model:     $MODEL"
    echo "Tokens:    $MAX_TOKENS"
    echo "Output:    $result_dir"

    vtune -collect "$analysis_type" \
          -result-dir "$result_dir" \
          -quiet \
          -- "$CLI" --model "$MODEL" --prompt "$PROMPT" --max-tokens "$MAX_TOKENS"

    echo ""
    echo "Open with: vtune-gui $result_dir"

    # Generate text report
    vtune -report summary -result-dir "$result_dir" \
          -format text -report-output "${result_dir}_summary.txt" 2>/dev/null || true
    
    if [ -f "${result_dir}_summary.txt" ]; then
        echo ""
        cat "${result_dir}_summary.txt"
    fi
}

run_compare() {
    echo "=== CK-Engine vs llama.cpp Comparison ==="

    LLAMA_CLI="$ROOT_DIR/llama.cpp/build/bin/llama-cli"
    GGUF_PATH=$(find ~/.cache/ck-engine-v6.5/models -name "*.gguf" -type f 2>/dev/null | head -1)

    if [ ! -f "$LLAMA_CLI" ]; then
        echo "ERROR: llama.cpp CLI not found. Run: make llamacpp-parity-rebuild"
        exit 1
    fi

    echo "Profiling CK-Engine..."
    vtune -collect hotspots -result-dir "$RESULTS_DIR/ck_$TIMESTAMP" -quiet \
          -- "$CLI" --model "$MODEL" --prompt "$PROMPT" --max-tokens "$MAX_TOKENS"

    echo "Profiling llama.cpp..."
    vtune -collect hotspots -result-dir "$RESULTS_DIR/llama_$TIMESTAMP" -quiet \
          -- "$LLAMA_CLI" -m "$GGUF_PATH" -p "$PROMPT" -n "$MAX_TOKENS" --no-display-prompt 2>/dev/null

    vtune -report hotspots -result-dir "$RESULTS_DIR/ck_$TIMESTAMP" \
          -format text -report-output "$RESULTS_DIR/ck_hotspots_$TIMESTAMP.txt"
    vtune -report hotspots -result-dir "$RESULTS_DIR/llama_$TIMESTAMP" \
          -format text -report-output "$RESULTS_DIR/llama_hotspots_$TIMESTAMP.txt"

    echo ""
    echo "=== CK-Engine Top 20 ==="
    head -25 "$RESULTS_DIR/ck_hotspots_$TIMESTAMP.txt"
    echo ""
    echo "=== llama.cpp Top 20 ==="
    head -25 "$RESULTS_DIR/llama_hotspots_$TIMESTAMP.txt"
}

echo "CPU: $(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)"
echo "AVX-512: $HAS_AVX512"

case "${1:-help}" in
    hotspots|hot)       run_vtune "hotspots" ;;
    memory-access|mem)  run_vtune "memory-access" ;;
    uarch|micro)        run_vtune "uarch-exploration" ;;
    threading)          run_vtune "threading" ;;
    compare|cmp)        run_compare ;;
    full|all)
        run_vtune "hotspots"
        run_vtune "memory-access"
        run_vtune "uarch-exploration"
        ;;
    *)
        echo "Usage: $0 <analysis-type>"
        echo ""
        echo "  hotspots    CPU hotspots"
        echo "  memory-access  Memory bandwidth analysis"
        echo "  uarch       Microarchitecture stalls"
        echo "  threading   Thread analysis"
        echo "  compare     CK vs llama.cpp comparison"
        echo "  full        All analyses"
        echo ""
        echo "Env: MODEL=... PROMPT=... MAX_TOKENS=..."
        ;;
esac
