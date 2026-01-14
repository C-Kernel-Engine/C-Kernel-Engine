#!/bin/bash
#
# profile_comparison.sh - Profile CK-Engine vs llama.cpp
#
# Generates flamegraphs and performance data for both engines
# to identify bottlenecks and compare performance.
#
# Prerequisites:
#   - perf (linux-tools-generic or perf package)
#   - FlameGraph (auto-cloned if missing)
#   - Model downloaded (run ck_run_v6_5.py first)
#
# Usage:
#   ./scripts/profile_comparison.sh              # Default 100 tokens
#   ./scripts/profile_comparison.sh --tokens 200 # Custom token count
#   ./scripts/profile_comparison.sh --ck-only    # Profile CK-Engine only
#   ./scripts/profile_comparison.sh --llama-only # Profile llama.cpp only
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
FLAMEGRAPH_DIR="$PROJECT_ROOT/FlameGraph"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2-0.5B-Instruct-GGUF}"
MODEL_DIR_NAME=$(echo "$MODEL_NAME" | tr '/' '--')
MODEL_CACHE="$HOME/.cache/ck-engine-v6.5/models/$MODEL_DIR_NAME"

# Settings
NUM_TOKENS=100
PERF_FREQ=99
PROFILE_CK=true
PROFILE_LLAMA=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --tokens)
            NUM_TOKENS="$2"
            shift 2
            ;;
        --ck-only)
            PROFILE_LLAMA=false
            shift
            ;;
        --llama-only)
            PROFILE_CK=false
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --tokens N     Number of tokens to generate (default: 100)"
            echo "  --ck-only      Profile CK-Engine only"
            echo "  --llama-only   Profile llama.cpp only"
            echo "  --help         Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Colors
GREEN='\033[92m'
RED='\033[91m'
CYAN='\033[96m'
YELLOW='\033[93m'
RESET='\033[0m'

echo -e "${CYAN}======================================================================"
echo -e "  Performance Profiling: CK-Engine vs llama.cpp"
echo -e "======================================================================${RESET}"
echo ""

# Check prerequisites
check_prereqs() {
    echo -e "${CYAN}Checking prerequisites...${RESET}"

    # Check perf
    if ! command -v perf &> /dev/null; then
        echo -e "${RED}ERROR: 'perf' not found${RESET}"
        echo "Install with:"
        echo "  Ubuntu/Debian: sudo apt install linux-tools-generic linux-tools-\$(uname -r)"
        echo "  RHEL/CentOS:   sudo dnf install perf"
        exit 1
    fi
    echo -e "  ${GREEN}perf: OK${RESET}"

    # Check/clone FlameGraph
    if [ ! -d "$FLAMEGRAPH_DIR" ]; then
        echo -e "  ${YELLOW}FlameGraph not found, cloning...${RESET}"
        git clone --depth 1 https://github.com/brendangregg/FlameGraph.git "$FLAMEGRAPH_DIR"
    fi
    echo -e "  ${GREEN}FlameGraph: OK${RESET}"

    # Check model
    if [ ! -d "$MODEL_CACHE" ]; then
        echo -e "${RED}ERROR: Model not found at $MODEL_CACHE${RESET}"
        echo "Run first: python3 scripts/v6.5/ck_run_v6_5.py run $MODEL_NAME"
        exit 1
    fi
    echo -e "  ${GREEN}Model: OK${RESET}"

    # Check CK-Engine library
    if [ "$PROFILE_CK" = true ] && [ ! -f "$MODEL_CACHE/ck-kernel-inference.so" ]; then
        echo -e "${YELLOW}CK-Engine not compiled, compiling...${RESET}"
        python3 "$SCRIPT_DIR/v6.5/ck_run_v6_5.py" run "$MODEL_NAME" --force-compile --max-tokens 1
    fi
    [ "$PROFILE_CK" = true ] && echo -e "  ${GREEN}CK-Engine: OK${RESET}"

    # Check llama.cpp
    LLAMA_CLI="$PROJECT_ROOT/llama.cpp/build/bin/llama-cli"
    if [ "$PROFILE_LLAMA" = true ] && [ ! -f "$LLAMA_CLI" ]; then
        echo -e "${YELLOW}llama.cpp not built, building...${RESET}"
        make -C "$PROJECT_ROOT" llamacpp-parity-rebuild
    fi
    [ "$PROFILE_LLAMA" = true ] && echo -e "  ${GREEN}llama.cpp: OK${RESET}"

    echo ""
}

# Create results directory
setup_results() {
    TIMESTAMP=$(date +%Y%m%d-%H%M%S)
    RESULTS_DIR="$PROJECT_ROOT/profile_results/$TIMESTAMP"
    mkdir -p "$RESULTS_DIR"
    echo -e "${CYAN}Results directory: $RESULTS_DIR${RESET}"
    echo ""
}

# Profile CK-Engine
profile_ck_engine() {
    echo -e "${CYAN}======================================================================"
    echo -e "  Profiling CK-Engine ($NUM_TOKENS tokens)"
    echo -e "======================================================================${RESET}"

    # Create profiling script
    cat > /tmp/profile_ck.py << PROFILE_SCRIPT
#!/usr/bin/env python3
import ctypes
import os
import sys
import time

NUM_TOKENS = $NUM_TOKENS
MODEL_DIR = "$MODEL_CACHE"
LIB_PATH = os.path.join(MODEL_DIR, "ck-kernel-inference.so")
WEIGHTS_PATH = os.path.join(MODEL_DIR, "weights.bump")

lib = ctypes.CDLL(LIB_PATH)

lib.ck_model_init.argtypes = [ctypes.c_char_p]
lib.ck_model_init.restype = ctypes.c_int
lib.ck_model_free.argtypes = []
lib.ck_model_free.restype = None
lib.ck_model_forward.argtypes = [ctypes.POINTER(ctypes.c_float)]
lib.ck_model_forward.restype = ctypes.c_int
lib.ck_model_decode.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
lib.ck_model_decode.restype = ctypes.c_int
lib.ck_model_embed_tokens.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
lib.ck_model_embed_tokens.restype = ctypes.c_int
lib.ck_model_kv_cache_enable.argtypes = [ctypes.c_int]
lib.ck_model_kv_cache_enable.restype = ctypes.c_int
lib.ck_model_kv_cache_reset.argtypes = []
lib.ck_model_kv_cache_reset.restype = None
lib.ck_model_sample_argmax.argtypes = []
lib.ck_model_sample_argmax.restype = ctypes.c_int

print(f"Initializing CK-Engine...", flush=True)
ret = lib.ck_model_init(WEIGHTS_PATH.encode())
if ret != 0:
    print(f"Init failed: {ret}")
    sys.exit(1)

# Standard prompt
tokens = [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198,
          151644, 872, 198, 3838, 374, 279, 6864, 315, 9625, 30, 151645, 198,
          151644, 77091, 198]
tokens_array = (ctypes.c_int32 * len(tokens))(*tokens)

lib.ck_model_kv_cache_enable(32768)
lib.ck_model_kv_cache_reset()
lib.ck_model_embed_tokens(tokens_array, len(tokens))

print(f"Prefill {len(tokens)} tokens...", flush=True)
t0 = time.perf_counter()
lib.ck_model_forward(None)
t1 = time.perf_counter()
print(f"Prefill: {(t1-t0)*1000:.1f}ms ({len(tokens)/(t1-t0):.1f} tok/s)", flush=True)

print(f"Generating {NUM_TOKENS} tokens...", flush=True)
t0 = time.perf_counter()
for i in range(NUM_TOKENS):
    next_token = lib.ck_model_sample_argmax()
    ret = lib.ck_model_decode(next_token, None)
    if ret != 0:
        break
t1 = time.perf_counter()
print(f"Decode: {NUM_TOKENS} tokens in {(t1-t0)*1000:.1f}ms ({NUM_TOKENS/(t1-t0):.1f} tok/s)", flush=True)

lib.ck_model_free()
PROFILE_SCRIPT

    # Run with perf
    echo "Running perf record..."
    perf record -F $PERF_FREQ -g --call-graph dwarf -o "$RESULTS_DIR/ck_perf.data" \
        python3 /tmp/profile_ck.py 2>&1 | tee "$RESULTS_DIR/ck_output.txt"

    # Generate flamegraph
    echo "Generating flamegraph..."
    perf script -i "$RESULTS_DIR/ck_perf.data" 2>/dev/null | \
        "$FLAMEGRAPH_DIR/stackcollapse-perf.pl" 2>/dev/null | \
        "$FLAMEGRAPH_DIR/flamegraph.pl" --title "CK-Engine ($NUM_TOKENS tokens)" \
        > "$RESULTS_DIR/ck_flamegraph.svg"

    # Generate perf report
    perf report -i "$RESULTS_DIR/ck_perf.data" --stdio --no-children 2>/dev/null \
        > "$RESULTS_DIR/ck_perf_report.txt" || true

    echo -e "${GREEN}CK-Engine profiling complete${RESET}"
    echo "  Flamegraph: $RESULTS_DIR/ck_flamegraph.svg"
    echo ""
}

# Profile llama.cpp
profile_llamacpp() {
    echo -e "${CYAN}======================================================================"
    echo -e "  Profiling llama.cpp ($NUM_TOKENS tokens)"
    echo -e "======================================================================${RESET}"

    LLAMA_CLI="$PROJECT_ROOT/llama.cpp/build/bin/llama-cli"
    GGUF_FILE=$(find "$MODEL_CACHE" -name "*.gguf" -type f 2>/dev/null | head -1)

    if [ -z "$GGUF_FILE" ]; then
        echo -e "${RED}ERROR: No GGUF file found in $MODEL_CACHE${RESET}"
        return 1
    fi

    PROMPT="<|im_start|>system
You are a helpful assistant.<|im_end|}
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
"

    echo "Running perf record..."
    perf record -F $PERF_FREQ -g --call-graph dwarf -o "$RESULTS_DIR/llama_perf.data" \
        "$LLAMA_CLI" -m "$GGUF_FILE" -p "$PROMPT" -n $NUM_TOKENS \
        --temp 0 -t 1 -ngl 0 2>&1 | tee "$RESULTS_DIR/llama_output.txt"

    # Generate flamegraph
    echo "Generating flamegraph..."
    perf script -i "$RESULTS_DIR/llama_perf.data" 2>/dev/null | \
        "$FLAMEGRAPH_DIR/stackcollapse-perf.pl" 2>/dev/null | \
        "$FLAMEGRAPH_DIR/flamegraph.pl" --title "llama.cpp ($NUM_TOKENS tokens)" \
        > "$RESULTS_DIR/llama_flamegraph.svg"

    # Generate perf report
    perf report -i "$RESULTS_DIR/llama_perf.data" --stdio --no-children 2>/dev/null \
        > "$RESULTS_DIR/llama_perf_report.txt" || true

    echo -e "${GREEN}llama.cpp profiling complete${RESET}"
    echo "  Flamegraph: $RESULTS_DIR/llama_flamegraph.svg"
    echo ""
}

# Show results summary
show_summary() {
    echo -e "${CYAN}======================================================================"
    echo -e "  Summary"
    echo -e "======================================================================${RESET}"

    echo ""
    echo "Results saved to: $RESULTS_DIR/"
    ls -la "$RESULTS_DIR/" 2>/dev/null || true

    echo ""
    echo -e "${CYAN}View flamegraphs:${RESET}"
    [ -f "$RESULTS_DIR/ck_flamegraph.svg" ] && echo "  CK-Engine: firefox $RESULTS_DIR/ck_flamegraph.svg"
    [ -f "$RESULTS_DIR/llama_flamegraph.svg" ] && echo "  llama.cpp: firefox $RESULTS_DIR/llama_flamegraph.svg"

    if [ -f "$RESULTS_DIR/ck_perf_report.txt" ]; then
        echo ""
        echo -e "${CYAN}Top functions (CK-Engine):${RESET}"
        grep -E "^\s+[0-9]" "$RESULTS_DIR/ck_perf_report.txt" | head -15
    fi

    if [ -f "$RESULTS_DIR/llama_perf_report.txt" ]; then
        echo ""
        echo -e "${CYAN}Top functions (llama.cpp):${RESET}"
        grep -E "^\s+[0-9]" "$RESULTS_DIR/llama_perf_report.txt" | head -15
    fi

    echo ""
    echo -e "${GREEN}======================================================================"
    echo -e "  Profiling Complete!"
    echo -e "======================================================================${RESET}"
    echo ""
    echo "What to look for in flamegraphs:"
    echo "  1. Wide bars = functions taking most time"
    echo "  2. Compare kernel widths between CK and llama.cpp"
    echo "  3. Look for unexpected hotspots (memory allocation, etc.)"
    echo ""
}

# Main
check_prereqs
setup_results

[ "$PROFILE_CK" = true ] && profile_ck_engine
[ "$PROFILE_LLAMA" = true ] && profile_llamacpp

show_summary
