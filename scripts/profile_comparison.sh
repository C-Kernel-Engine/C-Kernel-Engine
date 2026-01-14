#!/bin/bash
# Profiling script for C-Kernel-Engine vs llama.cpp
# Uses valgrind/callgrind and gprof for analysis

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CK_DIR="$SCRIPT_DIR"
MODEL_DIR="$HOME/.cache/ck-engine-v6.5/models/Qwen--Qwen2-0.5B-Instruct-GGUF"
LLAMA_DIR="$SCRIPT_DIR/llama.cpp"
PROFILE_DIR="$SCRIPT_DIR/profile_results"

mkdir -p "$PROFILE_DIR"

echo "============================================"
echo "C-Kernel-Engine vs llama.cpp Profiling"
echo "============================================"
echo ""
echo "Model: Qwen2-0.5B-Instruct (Q4_K_M)"
echo "Output: $PROFILE_DIR"
echo ""

# Helper function to run llama.cpp decode
run_llama_decode() {
    local output_file="$1"
    local extra_args="${2:-}"

    echo "  Running llama.cpp decode..."
    cd "$LLAMA_DIR"
    ./build/bin/llama-cli \
        -m "$MODEL_DIR/qwen2-0_5b-instruct-q4_k_m.gguf" \
        -p "Hello, I am" \
        -n 10 \
        --temp 0 \
        --threads 4 \
        $extra_args 2>&1 | tee "$output_file"
}

# Helper function to run CK decode (Python wrapper)
run_ck_decode() {
    local output_file="$1"
    local extra_args="${2:-}"

    echo "  Running C-Kernel-Engine decode..."
    cd "$CK_DIR"
    python scripts/v6.5/ck_run_v6_5.py run \
        "$MODEL_DIR/qwen2-0_5b-instruct-q4_k_m.gguf" \
        --force-compile \
        --threads 4 \
        $extra_args 2>&1 | tee "$output_file"
}

# Run basic timing first
echo "=== Step 1: Basic Timing Comparison ==="
echo ""

echo "--- llama.cpp ---"
time run_llama_decode "$PROFILE_DIR/llama_basic.txt" 2>&1 | tail -5

echo ""
echo "--- C-Kernel-Engine ---"
time run_ck_decode "$PROFILE_DIR/ck_basic.txt" 2>&1 | tail -5

echo ""
echo "=== Step 2: Valgrind/Callgrind Profiling ==="
echo ""

# Profile with valgrind (if available)
if command -v valgrind &> /dev/null; then
    echo "Running valgrind/callgrind on llama.cpp..."
    cd "$LLAMA_DIR"
    valgrind --tool=callgrind --dump-instr=yes --collect-jumps=yes \
        ./build/bin/llama-cli \
        -m "$MODEL_DIR/qwen2-0_5b-instruct-q4_k_m.gguf" \
        -p "Hello" \
        -n 5 \
        --threads 1 \
        2>&1 | tee "$PROFILE_DIR/llama_valgrind.txt"

    echo "Callgrind output: callgrind.out.* in $LLAMA_DIR"

    echo ""
    echo "Running valgrind/callgrind on C-Kernel-Engine..."
    cd "$CK_DIR"
    valgrind --tool=callgrind --dump-instr=yes \
        python scripts/v6.5/ck_run_v6_5.py run \
        "$MODEL_DIR/qwen2-0_5b-instruct-q4_k_m.gguf" \
        --force-compile \
        --threads 1 \
        -n 5 \
        2>&1 | tee "$PROFILE_DIR/ck_valgrind.txt"
else
    echo "Valgrind not available, skipping..."
fi

echo ""
echo "=== Step 3: GPROF Profiling ==="
echo ""

# For gprof, we'd need to rebuild with -pg
# This is optional and takes longer
echo "To use gprof, rebuild with: make CFLAGS='-O2 -pg' LDFLAGS='-pg'"
echo ""

echo "=== Step 4: Summary ==="
echo ""
echo "Profile results saved to: $PROFILE_DIR"
echo ""
echo "To analyze callgrind results:"
echo "  1. Install kcachegrind: apt install kcachegrind"
echo "  2. Open: kcachegrind callgrind.out.*"
echo ""
echo "To generate flamegraph from callgrind:"
echo "  1. git clone https://github.com/jrfonseca/gprof2dot"
echo "  2. python gprof2dot.py -f callgrind callgrind.out.* | dot -Tsvg -o flamegraph.svg"
