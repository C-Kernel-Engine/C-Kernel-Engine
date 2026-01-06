#!/bin/bash
#
# run_full_validation.sh - Complete Parity Validation Pipeline
#
# Validates C-Kernel-Engine against llama.cpp at two levels:
#   1. Kernel-level: Individual ops (dequant, gemv, rmsnorm, rope, swiglu)
#   2. Model-level: Tensor-by-tensor during inference
#
# Usage:
#   ./scripts/run_full_validation.sh --gguf model.gguf
#   ./scripts/run_full_validation.sh --gguf model.gguf --kernels-only
#   ./scripts/run_full_validation.sh --gguf model.gguf --model-only
#

set -e

# Colors
GREEN='\033[92m'
RED='\033[91m'
YELLOW='\033[93m'
BOLD='\033[1m'
RESET='\033[0m'

# Defaults
GGUF=""
PROMPT="Hello"
TOLERANCE="1e-3"
KERNELS_ONLY=false
MODEL_ONLY=false
BUILD_DIR="build"
LLAMA_DUMP_DIR="llama_dump"
CK_DUMP_DIR="parity"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gguf)
            GGUF="$2"
            shift 2
            ;;
        --prompt)
            PROMPT="$2"
            shift 2
            ;;
        --tol|--tolerance)
            TOLERANCE="$2"
            shift 2
            ;;
        --kernels-only)
            KERNELS_ONLY=true
            shift
            ;;
        --model-only)
            MODEL_ONLY=true
            shift
            ;;
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --gguf FILE        GGUF model file (required for model-level tests)"
            echo "  --prompt TEXT      Prompt for inference test (default: 'Hello')"
            echo "  --tol FLOAT        Tolerance for comparisons (default: 1e-3)"
            echo "  --kernels-only     Run only kernel-level tests"
            echo "  --model-only       Run only model-level tests"
            echo "  --build-dir DIR    Build directory (default: build)"
            echo "  -h, --help         Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${BOLD}============================================================${RESET}"
echo -e "${BOLD}    C-Kernel-Engine vs llama.cpp Validation Pipeline${RESET}"
echo -e "${BOLD}============================================================${RESET}"
echo ""

# Step 1: Build parity libraries
if [[ "$MODEL_ONLY" == false ]]; then
    echo -e "${BOLD}[Step 1/5] Building parity testing libraries...${RESET}"
    echo ""

    # Build CK parity library
    echo "Building CK parity library..."
    make libck_parity.so 2>&1 | head -20

    # Check if llama.cpp exists
    if [[ -d "llama.cpp" ]]; then
        echo "Building llama.cpp kernel test library..."

        # Build llama.cpp if needed
        if [[ ! -f "llama.cpp/build/lib/libggml.so" ]]; then
            echo "Building llama.cpp core..."
            (cd llama.cpp && mkdir -p build && cd build && cmake .. && make -j$(nproc) ggml)
        fi

        # Build test library
        if [[ ! -f "llama.cpp/libggml_kernel_test.so" ]]; then
            (cd llama.cpp && g++ -shared -fPIC -o libggml_kernel_test.so \
                tests/test-kernel-parity.cpp \
                -I ggml/include -I ggml/src \
                -L build/lib -lggml -lm -lpthread \
                -Wl,-rpath,$(pwd)/build/lib 2>&1) || {
                echo -e "${YELLOW}Warning: Could not build llama.cpp kernel test library${RESET}"
                echo "Kernel tests may be skipped."
            }
        fi
    else
        echo -e "${YELLOW}Warning: llama.cpp not found. Kernel tests will be skipped.${RESET}"
        echo "Clone with: git clone https://github.com/ggerganov/llama.cpp llama.cpp"
    fi

    echo -e "${GREEN}Libraries built.${RESET}"
    echo ""
fi

# Step 2: Kernel-level tests
if [[ "$MODEL_ONLY" == false ]]; then
    echo -e "${BOLD}[Step 2/5] Running kernel-level parity tests...${RESET}"
    echo ""

    if [[ -f "llama.cpp/libggml_kernel_test.so" ]] && [[ -f "$BUILD_DIR/libck_parity.so" ]]; then
        python3 scripts/test_kernels_vs_llamacpp.py --all --tol "$TOLERANCE" || {
            echo -e "${RED}Kernel tests failed!${RESET}"
            if [[ "$KERNELS_ONLY" == true ]]; then
                exit 1
            fi
        }
    else
        echo -e "${YELLOW}Skipping kernel tests (libraries not available)${RESET}"
    fi

    echo ""
fi

if [[ "$KERNELS_ONLY" == true ]]; then
    echo -e "${GREEN}Kernel-only validation complete.${RESET}"
    exit 0
fi

# Step 3: Convert GGUF to bump format
if [[ -n "$GGUF" ]]; then
    echo -e "${BOLD}[Step 3/5] Converting GGUF to bump format...${RESET}"
    echo ""

    MODEL_NAME=$(basename "$GGUF" .gguf)
    MODEL_DIR="$BUILD_DIR/$MODEL_NAME"
    mkdir -p "$MODEL_DIR"

    WEIGHTS_BUMP="$MODEL_DIR/weights.bump"
    CONFIG_JSON="$MODEL_DIR/config.json"
    MANIFEST_JSON="$MODEL_DIR/weights_manifest.json"

    if [[ ! -f "$WEIGHTS_BUMP" ]]; then
        python3 scripts/convert_gguf_to_bump.py \
            --gguf "$GGUF" \
            --output "$WEIGHTS_BUMP" \
            --config-out "$CONFIG_JSON" \
            --manifest-out "$MANIFEST_JSON" \
            2>&1 | head -30

        echo -e "${GREEN}Converted GGUF to bump format.${RESET}"
    else
        echo "Using existing bump file: $WEIGHTS_BUMP"
    fi

    echo ""
else
    echo -e "${YELLOW}[Step 3/5] Skipping GGUF conversion (no --gguf specified)${RESET}"
    echo ""
fi

# Step 4: Run llama.cpp with tensor dumps
if [[ -n "$GGUF" ]]; then
    echo -e "${BOLD}[Step 4/5] Running llama.cpp to generate tensor dumps...${RESET}"
    echo ""

    # Check for hacked llama.cpp binary
    LLAMA_CLI="llama.cpp/build/bin/llama-cli"

    if [[ -f "$LLAMA_CLI" ]]; then
        mkdir -p "$LLAMA_DUMP_DIR"

        echo "Running: $LLAMA_CLI -m $GGUF -p '$PROMPT' -n 1"
        LD_LIBRARY_PATH=llama.cpp/build/lib "$LLAMA_CLI" \
            -m "$GGUF" \
            -p "$PROMPT" \
            -n 1 \
            2>&1 | head -50

        echo ""
        echo "Tensor dumps generated:"
        ls -la "$LLAMA_DUMP_DIR"/*.bin 2>/dev/null | head -20 || echo "No dumps found"
    else
        echo -e "${YELLOW}Warning: llama.cpp CLI not found at $LLAMA_CLI${RESET}"
        echo "Build with: cd llama.cpp && mkdir build && cd build && cmake .. && make"
    fi

    echo ""
else
    echo -e "${YELLOW}[Step 4/5] Skipping llama.cpp inference (no --gguf specified)${RESET}"
    echo ""
fi

# Step 5: Run CK with parity dumps and compare
if [[ -n "$GGUF" ]]; then
    echo -e "${BOLD}[Step 5/5] Running model parity comparison...${RESET}"
    echo ""

    # Check if we have tensor dumps to compare
    if [[ -d "$LLAMA_DUMP_DIR" ]] && [[ -n "$(ls -A $LLAMA_DUMP_DIR/*.bin 2>/dev/null)" ]]; then
        if [[ -d "$CK_DUMP_DIR" ]] && [[ -n "$(ls -A $CK_DUMP_DIR/*.bin 2>/dev/null)" ]]; then
            python3 scripts/compare_runtime_parity.py \
                --llama-dump "$LLAMA_DUMP_DIR" \
                --ck-dump "$CK_DUMP_DIR" \
                --manifest "$MANIFEST_JSON" \
                --tol "$TOLERANCE"
        else
            echo -e "${YELLOW}No CK parity dumps found in $CK_DUMP_DIR${RESET}"
            echo "Run CK with --parity flag to generate dumps:"
            echo "  python scripts/ck_run_v5.py run $MODEL_DIR --parity"
        fi
    else
        echo -e "${YELLOW}No llama.cpp tensor dumps found in $LLAMA_DUMP_DIR${RESET}"
        echo "Make sure llama.cpp is built with tensor dump hack enabled."
    fi

    echo ""
else
    echo -e "${YELLOW}[Step 5/5] Skipping model comparison (no --gguf specified)${RESET}"
    echo ""
fi

echo -e "${BOLD}============================================================${RESET}"
echo -e "${BOLD}                   Validation Complete${RESET}"
echo -e "${BOLD}============================================================${RESET}"
echo ""
echo "Summary of available tests:"
echo "  make test-kernels              Run kernel-level parity tests"
echo "  make test-kernel-<name>        Run specific kernel test"
echo "  python scripts/compare_runtime_parity.py  Run model-level comparison"
echo ""
