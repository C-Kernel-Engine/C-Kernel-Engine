#!/bin/bash
#
# v7_build.sh - Build v7 C-Kernel-Engine Training
#
# Build pipeline:
#   1. Download FP32 weights (if needed)
#   2. Convert to weights.bump (FP32)
#   3. Generate IR with training metadata
#   4. Memory planner
#   5. Generate backprop code
#   6. Compile training binary
#

set -e
shopt -s nullglob

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="$SCRIPT_DIR/../../scripts:$PYTHONPATH"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[BUILD]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

CACHE_DIR="${CACHE_DIR:-$HOME/.cache/ck-engine-v7}"

usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --model <name>        Model name (e.g., smollm-135m, qwen2-0.5b)"
    echo "  --download <preset>   Download model from HuggingFace"
    echo "  --cache-dir <dir>     Cache directory (default: ~/.cache/ck-engine-v7)"
    echo "  --train               Build training binary"
    echo "  -v, --verbose         Verbose output"
    echo "  -h, --help            Show help"
    echo ""
    echo "Presets:"
    echo "  smollm-135m    HuggingFaceTB/SmolLM-135M"
    echo "  smollm-360m    HuggingFaceTB/SmolLM-360M"
    echo "  qwen2-0.5b     Qwen/Qwen2-0.5B-Instruct"
}

# Parse arguments
MODEL_NAME=""
DOWNLOAD_MODEL=""
BUILD_TRAIN=0
VERBOSE=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --download)
            DOWNLOAD_MODEL="$2"
            shift 2
            ;;
        --cache-dir)
            CACHE_DIR="$2"
            shift 2
            ;;
        --train)
            BUILD_TRAIN=1
            shift
            ;;
        -v|--verbose)
            VERBOSE=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Step 0: Download model if requested
if [ -n "$DOWNLOAD_MODEL" ]; then
    log_info "Step 0: Downloading model from HuggingFace..."
    DOWNLOAD_SCRIPT="$SCRIPT_DIR/../scripts/download_model_v7.py"

    if [ ! -f "$DOWNLOAD_SCRIPT" ]; then
        log_error "Download script not found: $DOWNLOAD_SCRIPT"
        exit 1
    fi

    python3 "$DOWNLOAD_SCRIPT" --preset "$DOWNLOAD_MODEL" --output "$CACHE_DIR/models/$DOWNLOAD_MODEL"
    MODEL_NAME="$DOWNLOAD_MODEL"

    log_info "Download complete!"
fi

if [ -z "$MODEL_NAME" ]; then
    log_error "No model specified. Use --model or --download"
    exit 1
fi

MODEL_DIR="$CACHE_DIR/models/$MODEL_NAME"
mkdir -p "$MODEL_DIR"

log_info "Model directory: $MODEL_DIR"

# Step 1: Check prerequisites
log_info "Step 1: Checking prerequisites..."

for cmd in python3 gcc make; do
    if ! command -v $cmd &> /dev/null; then
        log_error "$cmd is required but not installed"
        exit 1
    fi
done

# Step 2: Convert to weights.bump (FP32)
WEIGHTS_BUMP="$MODEL_DIR/weights.bump"
if [ ! -f "$WEIGHTS_BUMP" ]; then
    log_info "Step 2: Converting weights to BUMP format (FP32)..."
    CONVERT_SCRIPT="$SCRIPT_DIR/../scripts/convert_to_bump_v7.py"

    if [ ! -f "$CONVERT_SCRIPT" ]; then
        log_error "Convert script not found: $CONVERT_SCRIPT"
        exit 1
    fi

    python3 "$CONVERT_SCRIPT" --input "$MODEL_DIR" --output "$WEIGHTS_BUMP" --dtype fp32
else
    log_info "Step 2: Using existing weights.bump"
fi

# Step 3: Generate IR with training metadata
IR_FILE="$MODEL_DIR/model_ir.json"
if [ ! -f "$IR_FILE" ]; then
    log_info "Step 3: Generating IR with training metadata..."
    IR_GEN="$SCRIPT_DIR/../scripts/build_ir_v7.py"

    python3 "$IR_GEN" \
        --weights "$WEIGHTS_BUMP" \
        --output "$IR_FILE" \
        --training
else
    log_info "Step 3: Using existing IR"
fi

# Step 4: Memory planner
MEMORY_PLAN="$MODEL_DIR/memory_plan.json"
if [ ! -f "$MEMORY_PLAN" ]; then
    log_info "Step 4: Running memory planner..."
    MEM_PLAN="$SCRIPT_DIR/../scripts/memory_planner_v7.py"

    python3 "$MEM_PLAN" \
        --ir "$IR_FILE" \
        --output "$MEMORY_PLAN" \
        --available-mb 4096
else
    log_info "Step 4: Using existing memory plan"
fi

# Check if training is feasible
if [ -f "$MEMORY_PLAN" ]; then
    FITS=$(python3 -c "import json; print(json.load(open('$MEMORY_PLAN'))['fits_in_memory'])")
    if [ "$FITS" != "True" ]; then
        log_warn "Model may not fit in memory with current settings"
        log_warn "Check $MEMORY_PLAN for details"
    fi
fi

# Step 5: Generate backprop code
BACKPROP_C="$MODEL_DIR/ck-kernel-backprop.c"
if [ $BUILD_TRAIN -eq 1 ]; then
    log_info "Step 5: Generating backprop code..."
    CODEGEN="$SCRIPT_DIR/../scripts/codegen_backprop_v7.py"

    python3 "$CODEGEN" \
        --ir "$IR_FILE" \
        --memory-plan "$MEMORY_PLAN" \
        --output "$BACKPROP_C"
fi

# Step 6: Compile training binary
if [ $BUILD_TRAIN -eq 1 ] && [ -f "$BACKPROP_C" ]; then
    log_info "Step 6: Compiling training binary..."

    PROJECT_ROOT="/home/antshiv/Workspace/C-Kernel-Engine"
    INCLUDE_PATHS="-I$PROJECT_ROOT/include -I$SCRIPT_DIR/include -I$MODEL_DIR"

    CFLAGS="-O3 -march=native -mtune=native -std=c11 -ffast-math -D_GNU_SOURCE"

    # Compile backprop
    BACKPROP_OBJ="$MODEL_DIR/ck-kernel-backprop.o"
    gcc $CFLAGS -fopenmp -c "$BACKPROP_C" -o "$BACKPROP_OBJ" $INCLUDE_PATHS

    # Compile other sources (TODO: create v7 versions)
    log_warn "Training binary compilation requires more v7 sources"
    log_warn "Currently at: $BACKPROP_OBJ"
fi

echo ""
log_info "=========================================="
log_info "v7 Build Summary"
log_info "=========================================="
echo ""
log_info "Model: $MODEL_NAME"
log_info "Cache: $MODEL_DIR"
echo ""

if [ -f "$WEIGHTS_BUMP" ]; then
    log_info "Weights: $WEIGHTS_BUMP"
fi
if [ -f "$IR_FILE" ]; then
    log_info "IR: $IR_FILE"
fi
if [ -f "$MEMORY_PLAN" ]; then
    log_info "Memory plan: $MEMORY_PLAN"
fi
if [ -f "$BACKPROP_C" ]; then
    log_info "Backprop: $BACKPROP_C"
fi

echo ""
log_info "Next steps:"
log_info "  - Complete v7 training sources (v7_train.c, v7_optimizer.c)"
log_info "  - Compile full training binary"
log_info "  - Test training loop"

exit 0
