#!/bin/bash
# =============================================================================
# CK-Engine Full Pipeline Validation
# =============================================================================
#
# This script runs ALL 6 layers of testing to pinpoint exactly where failures
# occur in the inference pipeline.
#
# Layers:
#   1. Kernel Parity (llama.cpp) - Do kernels match reference?
#   2. Bump Conversion           - Are weights converted correctly?
#   3. IR Structure              - Is the computation graph correct?
#   4. Codegen                   - Does generated code compile?
#   5. Tensor Flow               - Are dimensions correct throughout?
#   6. E2E Inference             - Does it produce coherent output?
#
# Usage:
#   ./scripts/test_full_pipeline.sh           # Run all layers
#   ./scripts/test_full_pipeline.sh --quick   # Skip slow tests
#   ./scripts/test_full_pipeline.sh --layer 3 # Run specific layer only
#
# =============================================================================

set -e  # Exit on first error (for pipeline mode)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Counters
TOTAL_PASSED=0
TOTAL_FAILED=0
LAYER_RESULTS=()

# Options
QUICK_MODE=false
SPECIFIC_LAYER=0
MAX_LAYER=6
VERBOSE=false
MODEL_DIR=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick|-q)
            QUICK_MODE=true
            shift
            ;;
        --layer|-l)
            SPECIFIC_LAYER=$2
            shift 2
            ;;
        --max-layer|-x)
            MAX_LAYER=$2
            shift 2
            ;;
        --model-dir|-m)
            MODEL_DIR=$2
            shift 2
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --quick, -q       Skip slow tests (kernel parity)"
            echo "  --layer N, -l N   Run specific layer only (1-6)"
            echo "  --max-layer N, -x N  Run layers 1..N only (default: 6)"
            echo "  --model-dir DIR   Use specific model directory"
            echo "  --verbose, -v     Verbose output"
            echo "  --help, -h        Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if ! [[ "$MAX_LAYER" =~ ^[0-9]+$ ]]; then
    echo "Invalid --max-layer: $MAX_LAYER (must be integer 1-6)"
    exit 1
fi
if [ "$MAX_LAYER" -lt 1 ] || [ "$MAX_LAYER" -gt 6 ]; then
    echo "Invalid --max-layer: $MAX_LAYER (must be 1-6)"
    exit 1
fi

# Find model directory if not specified
if [ -z "$MODEL_DIR" ]; then
    for dir in \
        "$HOME/.cache/ck-engine-v6.6/models/"* \
        "$HOME/.cache/ck-engine-v6.5/models/"* \
        "$HOME/.cache/ck-engine-v6/models/"*; do
        if [ -d "$dir" ] && [ -f "$dir/weights_manifest.json" ]; then
            MODEL_DIR="$dir"
            break
        fi
    done
fi

# Header
echo ""
echo -e "${CYAN}================================================================${NC}"
echo -e "${CYAN}  CK-Engine Full Pipeline Validation${NC}"
echo -e "${CYAN}================================================================${NC}"
echo ""
if [ -n "$MODEL_DIR" ]; then
    echo -e "Model: ${BLUE}$(basename $MODEL_DIR)${NC}"
fi
echo ""

run_layer() {
    local layer_num=$1
    local layer_name=$2
    local test_cmd=$3
    local fail_hint=$4

    echo -e "${CYAN}────────────────────────────────────────────────────────────${NC}"
    echo -e "${CYAN}[$layer_num/6] $layer_name${NC}"
    echo -e "${CYAN}────────────────────────────────────────────────────────────${NC}"
    echo ""

    local start_time=$(date +%s)
    local result=0

    # Run the test
    if eval "$test_cmd"; then
        result=0
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        echo ""
        echo -e "${GREEN}Layer $layer_num PASSED${NC} (${duration}s)"
        LAYER_RESULTS+=("$layer_num:PASS")
        ((TOTAL_PASSED++))
    else
        result=1
        echo ""
        echo -e "${RED}Layer $layer_num FAILED${NC}"
        echo -e "${YELLOW}Hint: $fail_hint${NC}"
        LAYER_RESULTS+=("$layer_num:FAIL")
        ((TOTAL_FAILED++))
    fi

    echo ""
    return $result
}

# =============================================================================
# Layer 1: Kernel Parity with llama.cpp
# =============================================================================
run_layer_1() {
    if $QUICK_MODE; then
        echo -e "${YELLOW}Skipping (quick mode)${NC}"
        LAYER_RESULTS+=("1:SKIP")
        return 0
    fi

    cd "$ROOT_DIR"

    # Check if parity test exists
    if [ -f "scripts/test_kernels_vs_llamacpp.py" ]; then
        # Check if libck_parity.so exists
        if [ ! -f "build/libck_parity.so" ] && [ ! -f "libck_parity.so" ]; then
            echo -e "${YELLOW}Parity library not built, skipping kernel parity tests${NC}"
            echo "  Build with: make libck_parity.so"
            LAYER_RESULTS+=("1:SKIP")
            return 0
        fi

        python3 scripts/test_kernels_vs_llamacpp.py --quick 2>&1
    else
        echo -e "${YELLOW}Kernel parity test not found${NC}"
        LAYER_RESULTS+=("1:SKIP")
    fi
}

# =============================================================================
# Layer 2: Bump Conversion
# =============================================================================
run_layer_2() {
    cd "$ROOT_DIR"

    if [ -z "$MODEL_DIR" ]; then
        echo -e "${YELLOW}No model directory found, skipping${NC}"
        return 0
    fi

    # Find GGUF file
    GGUF_FILE=$(find "$MODEL_DIR" -name "*.gguf" -type f 2>/dev/null | head -1)

    if [ -n "$GGUF_FILE" ]; then
        python3 scripts/test_bump_conversion.py \
            --gguf "$GGUF_FILE" \
            --bump "$MODEL_DIR" \
            ${VERBOSE:+--verbose}
    else
        # Just test the bump directory structure
        python3 scripts/test_bump_conversion.py \
            --auto \
            ${VERBOSE:+--verbose}
    fi
}

# =============================================================================
# Layer 3: IR Structure Validation
# =============================================================================
run_layer_3() {
    cd "$ROOT_DIR"

    if [ -z "$MODEL_DIR" ]; then
        echo -e "${YELLOW}No model directory found, skipping${NC}"
        return 0
    fi

    python3 scripts/test_ir_validation.py \
        --model-dir "$MODEL_DIR" \
        ${VERBOSE:+--verbose}
}

# =============================================================================
# Layer 4: Codegen Validation
# =============================================================================
run_layer_4() {
    cd "$ROOT_DIR"

    if [ -z "$MODEL_DIR" ]; then
        echo -e "${YELLOW}No model directory found, skipping${NC}"
        return 0
    fi

    python3 scripts/test_codegen_validation.py \
        --model-dir "$MODEL_DIR" \
        ${VERBOSE:+--verbose}
}

# =============================================================================
# Layer 5: Tensor Flow Validation
# =============================================================================
run_layer_5() {
    cd "$ROOT_DIR"

    if [ -z "$MODEL_DIR" ]; then
        echo -e "${YELLOW}No model directory found, skipping${NC}"
        return 0
    fi

    python3 scripts/test_tensor_flow.py \
        --model-dir "$MODEL_DIR" \
        ${VERBOSE:+--verbose}
}

# =============================================================================
# Layer 6: E2E Inference
# =============================================================================
run_layer_6() {
    cd "$ROOT_DIR"

    if [ -f "scripts/full_integration_testing.sh" ]; then
        bash scripts/full_integration_testing.sh
    else
        echo -e "${YELLOW}E2E test not found${NC}"
        return 0
    fi
}

# =============================================================================
# Main
# =============================================================================

if [ $SPECIFIC_LAYER -gt 0 ]; then
    # Run specific layer only
    case $SPECIFIC_LAYER in
        1)
            run_layer 1 "Kernel Parity (llama.cpp)" "run_layer_1" \
                "Check kernel implementations in src/kernels/"
            ;;
        2)
            run_layer 2 "Bump Conversion" "run_layer_2" \
                "Check scripts/v6/convert_gguf_to_bump_v6.py"
            ;;
        3)
            run_layer 3 "IR Structure Validation" "run_layer_3" \
                "Check scripts/v6/build_ir_v6.py"
            ;;
        4)
            run_layer 4 "Codegen Validation" "run_layer_4" \
                "Check scripts/v6/codegen_v6.py"
            ;;
        5)
            run_layer 5 "Tensor Flow Validation" "run_layer_5" \
                "Check IR shapes vs generated code dimensions"
            ;;
        6)
            run_layer 6 "E2E Inference" "run_layer_6" \
                "All lower layers passed, check runtime"
            ;;
        *)
            echo "Invalid layer: $SPECIFIC_LAYER (must be 1-6)"
            exit 1
            ;;
    esac
else
    # Run layers 1..MAX_LAYER in order
    set +e  # Don't exit on first error for summary

    if [ "$MAX_LAYER" -ge 1 ]; then
        run_layer 1 "Kernel Parity (llama.cpp)" "run_layer_1" \
            "Check kernel implementations in src/kernels/" || true
    else
        LAYER_RESULTS+=("1:SKIP")
    fi

    if [ "$MAX_LAYER" -ge 2 ]; then
        run_layer 2 "Bump Conversion" "run_layer_2" \
            "Check scripts/v6/convert_gguf_to_bump_v6.py" || true
    else
        LAYER_RESULTS+=("2:SKIP")
    fi

    if [ "$MAX_LAYER" -ge 3 ]; then
        run_layer 3 "IR Structure Validation" "run_layer_3" \
            "Check scripts/v6/build_ir_v6.py" || true
    else
        LAYER_RESULTS+=("3:SKIP")
    fi

    if [ "$MAX_LAYER" -ge 4 ]; then
        run_layer 4 "Codegen Validation" "run_layer_4" \
            "Check scripts/v6/codegen_v6.py" || true
    else
        LAYER_RESULTS+=("4:SKIP")
    fi

    if [ "$MAX_LAYER" -ge 5 ]; then
        run_layer 5 "Tensor Flow Validation" "run_layer_5" \
            "Check IR shapes vs generated code dimensions" || true
    else
        LAYER_RESULTS+=("5:SKIP")
    fi

    if [ "$MAX_LAYER" -ge 6 ]; then
        run_layer 6 "E2E Inference" "run_layer_6" \
            "All lower layers passed, check runtime" || true
    else
        LAYER_RESULTS+=("6:SKIP")
    fi
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo -e "${CYAN}================================================================${NC}"
echo -e "${CYAN}  Pipeline Summary${NC}"
echo -e "${CYAN}================================================================${NC}"
echo ""

# Print layer results
for result in "${LAYER_RESULTS[@]}"; do
    layer_num="${result%%:*}"
    status="${result##*:}"

    case $status in
        PASS)
            echo -e "  Layer $layer_num: ${GREEN}PASSED${NC}"
            ;;
        FAIL)
            echo -e "  Layer $layer_num: ${RED}FAILED${NC}"
            ;;
        SKIP)
            echo -e "  Layer $layer_num: ${YELLOW}SKIPPED${NC}"
            ;;
    esac
done

echo ""
echo -e "  ${GREEN}Total Passed:${NC} $TOTAL_PASSED"
echo -e "  ${RED}Total Failed:${NC} $TOTAL_FAILED"
echo ""

if [ $TOTAL_FAILED -eq 0 ]; then
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  ALL PIPELINE TESTS PASSED!${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    exit 0
else
    echo -e "${RED}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${RED}  PIPELINE FAILED - Check failed layers above${NC}"
    echo -e "${RED}═══════════════════════════════════════════════════════════════${NC}"
    exit 1
fi
