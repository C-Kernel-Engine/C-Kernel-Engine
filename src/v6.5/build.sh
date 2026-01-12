#!/bin/bash
#
# v6.5_build.sh - Build v6.5 C-Kernel-Engine
#
# Build pipeline:
#   1. Generate IR from config + manifest
#   2. Generate C source from IR (decode + prefill)
#   3. Compile generated code to .o
#   4. Link everything into ck-engine-v6.5
#

set -e  # Exit on error
shopt -s nullglob  # Handle empty glob patterns

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Add parent scripts directory to Python path for imports
export PYTHONPATH="$SCRIPT_DIR/../../scripts/v6:$PYTHONPATH"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[BUILD]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Default paths
MODEL_NAME="${MODEL_NAME:-qwen2-0.5b-instruct}"
CONFIG_FILE="${CONFIG_FILE:-models/$MODEL_NAME/config.json}"
MANIFEST_FILE="${MANIFEST_FILE:-models/$MODEL_NAME/weights_manifest.json}"
WEIGHTS_FILE="${WEIGHTS_FILE:-models/$MODEL_NAME/weights.bump}"
OUTPUT_DIR="${OUTPUT_DIR:-generated}"

# Short model name for file naming (e.g., "qwen2" from "qwen2-0.5b-instruct")
MODEL_SHORT="${MODEL_NAME%%-*}"

# Model-specific settings
EMBED_DIM="${EMBED_DIM:-896}"
NUM_LAYERS="${NUM_LAYERS:-24}"
NUM_HEADS="${NUM_HEADS:-14}"
NUM_KV_HEADS="${NUM_KV_HEADS:-2}"
HEAD_DIM="${HEAD_DIM:-64}"
INTERMEDIATE_SIZE="${INTERMEDIATE_SIZE:-4864}"
VOCAB_SIZE="${VOCAB_SIZE:-128256}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-32768}"

# Build settings
NUM_THREADS="${NUM_THREADS:-$(nproc)}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
VERBOSE="${VERBOSE:-0}"

# ============================================================================
# Parse arguments
# ============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-name)
            MODEL_NAME="$2"
            CONFIG_FILE="models/$MODEL_NAME/config.json"
            MANIFEST_FILE="models/$MODEL_NAME/weights_manifest.json"
            WEIGHTS_FILE="models/$MODEL_NAME/weights.bump"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --manifest)
            MANIFEST_FILE="$2"
            shift 2
            ;;
        --weights)
            WEIGHTS_FILE="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --threads)
            NUM_THREADS="$2"
            shift 2
            ;;
        --dtype)
            DTYPE="$2"
            shift 2
            ;;
        --release)
            BUILD_TYPE="Release"
            shift
            ;;
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        -v|--verbose)
            VERBOSE="1"
            shift
            ;;
        --download)
            DOWNLOAD_MODEL="$2"
            shift 2
            ;;
        --download-repo)
            DOWNLOAD_REPO="$2"
            shift 2
            ;;
        --cache-dir)
            CACHE_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --model-name <name>   Model name (default: qwen2-0.5b-instruct)"
            echo "  --config <file>       Config file"
            echo "  --manifest <file>     Weights manifest file"
            echo "  --weights <file>      Weights BUMP file"
            echo "  --output <dir>        Output directory"
            echo "  --threads <n>         Number of build threads"
            echo "  --dtype <type>        Data type (fp32, bf16)"
            echo "  --release             Release build"
            echo "  --debug               Debug build"
            echo "  --download <preset>   Auto-download model from HuggingFace (e.g., qwen2-0.5b)"
            echo "  --download-repo <repo> Download from HuggingFace repo (e.g., Qwen/Qwen2-0.5B-Instruct)"
            echo "  --cache-dir <dir>     Cache directory (default: ~/.cache/ck-engine-v6.5)"
            echo "  -v, --verbose         Verbose output"
            echo "  -h, --help            Show this help"
            echo ""
            echo "Presets:"
            echo "  qwen2-0.5b    Qwen/Qwen2-0.5B-Instruct"
            echo "  qwen2-1.5b    Qwen/Qwen2-1.5B-Instruct"
            echo "  smollm-135    HuggingFaceTB/SmolLM-135M"
            echo "  smollm-360    HuggingFaceTB/SmolLM-360M"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Default cache directory
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/ck-engine-v6.5}"
TOKENIZER_LIB="$PROJECT_ROOT/build/libckernel_tokenizer.so"
TOKENIZER_LDFLAGS="-L$PROJECT_ROOT/build -lckernel_tokenizer -Wl,-rpath,$PROJECT_ROOT/build"

# ============================================================================
# Step 0: Download model if requested
# ============================================================================
if [ -n "$DOWNLOAD_MODEL" ] || [ -n "$DOWNLOAD_REPO" ]; then
    log_info "Step 0: Downloading model from HuggingFace..."

    DOWNLOAD_SCRIPT="$SCRIPT_DIR/scripts/v6.5_download.py"

    if [ ! -f "$DOWNLOAD_SCRIPT" ]; then
        log_error "Download script not found: $DOWNLOAD_SCRIPT"
        exit 1
    fi

    # Set model-specific paths
    if [ -n "$DOWNLOAD_MODEL" ]; then
        MODEL_NAME="$DOWNLOAD_MODEL"
        CONFIG_FILE="$CACHE_DIR/models/$DOWNLOAD_MODEL/config.json"
        MANIFEST_FILE="$CACHE_DIR/models/$DOWNLOAD_MODEL/weights_manifest.json"
        WEIGHTS_FILE="$CACHE_DIR/models/$DOWNLOAD_MODEL/weights.bump"
    fi

    # Run download
    if [ -n "$DOWNLOAD_MODEL" ]; then
        python3 "$DOWNLOAD_SCRIPT" --preset "$DOWNLOAD_MODEL" --output "$CACHE_DIR/models/$DOWNLOAD_MODEL"
    elif [ -n "$DOWNLOAD_REPO" ]; then
        python3 "$DOWNLOAD_SCRIPT" --repo "$DOWNLOAD_REPO" --output "$CACHE_DIR/models/${DOWNLOAD_REPO##*/}"
    fi

    log_info "Download complete!"
fi

# ============================================================================
# Step 1: Check prerequisites
# ============================================================================
log_info "Step 1: Checking prerequisites..."

# Check for required tools
for cmd in python3 gcc make; do
    if ! command -v $cmd &> /dev/null; then
        log_error "$cmd is required but not installed"
        exit 1
    fi
done

# Check for IR generator (use parent scripts dir)
IR_GEN="$SCRIPT_DIR/../../scripts/v6/build_ir_v6.py"
if [ ! -f "$IR_GEN" ]; then
    log_error "IR generator not found: $IR_GEN"
    exit 1
fi

# Check for codegen
CODEGEN="$SCRIPT_DIR/scripts/v6.5_codegen.py"
if [ ! -f "$CODEGEN" ]; then
    log_error "Codegen not found: $CODEGEN"
    exit 1
fi

# Check for config file
if [ ! -f "$CONFIG_FILE" ]; then
    log_warn "Config file not found: $CONFIG_FILE"
    log_warn "Using default config values"
fi

# ============================================================================
# Step 2: Create output directory
# ============================================================================
mkdir -p "$OUTPUT_DIR"

# ============================================================================
# Step 3: Generate IR Layout (only if C source doesn't exist)
# ============================================================================
DECODE_SOURCE="$OUTPUT_DIR/ck-kernel-inference.c"
PREFILL_SOURCE="$OUTPUT_DIR/ck-kernel-prefill.c"

if [ -f "$DECODE_SOURCE" ] && [ -f "$PREFILL_SOURCE" ]; then
    log_info "Step 2: Using existing generated C source (skipping IR generation)"
else
    log_info "Step 2: Generating IR layout..."

    IR_GEN="$SCRIPT_DIR/../../scripts/v6/build_ir_v6.py"
    export PYTHONPATH="$SCRIPT_DIR/../../scripts/v6:$PYTHONPATH"

    # Check if config exists in different locations
    CONFIG_FILE=""
    for config_path in "models/$MODEL_NAME/config.json" "models/qwen2-0.5b/config.json" "config.json"; do
        if [ -f "$config_path" ]; then
            CONFIG_FILE="$config_path"
            break
        fi
    done

    # Check manifest location
    MANIFEST_FILE=""
    for manifest_path in "models/$MODEL_NAME/weights_manifest.json" "generated/weights_manifest.json" "weights_manifest.json"; do
        if [ -f "$manifest_path" ]; then
            MANIFEST_FILE="$manifest_path"
            break
        fi
    done

    if [ -n "$CONFIG_FILE" ]; then
        log_info "Using config: $CONFIG_FILE"
        python3 "$IR_GEN" \
            --config="$CONFIG_FILE" \
            --weights-manifest="$MANIFEST_FILE" \
            --prefix="$OUTPUT_DIR" \
            --dtype=fp32 \
            --weight-dtype=q4_k_m \
            --modes=prefill,decode \
            --codegen=v6
    elif [ -n "$MANIFEST_FILE" ]; then
        log_info "Using manifest: $MANIFEST_FILE"
        python3 "$IR_GEN" \
            --weights-manifest="$MANIFEST_FILE" \
            --prefix="$OUTPUT_DIR" \
            --dtype=fp32 \
            --weight-dtype=q4_k_m \
            --modes=prefill,decode \
            --codegen=v6
    else
        log_warn "No config or manifest found - using preset"
        python3 "$IR_GEN" \
            --preset=qwen2-0.5b \
            --prefix="$OUTPUT_DIR" \
            --dtype=fp32 \
            --weight-dtype=q4_k_m \
            --modes=prefill,decode \
            --codegen=v6
    fi

    # Check that output files were generated
    if [ ! -f "$OUTPUT_DIR/layout_decode.json" ] || [ ! -f "$OUTPUT_DIR/layout_prefill.json" ]; then
        log_error "IR generation failed - layout files not found"
        exit 1
    fi

    log_info "IR layout generated in: $OUTPUT_DIR"
fi

# ============================================================================
# Step 4: Verify C source files exist
# ============================================================================
log_info "Step 4: Verifying generated C source..."

DECODE_SOURCE="$OUTPUT_DIR/ck-kernel-inference.c"
PREFILL_SOURCE="$OUTPUT_DIR/ck-kernel-prefill.c"

if [ ! -f "$DECODE_SOURCE" ]; then
    log_error "Decode source not found: $DECODE_SOURCE"
    exit 1
fi

if [ ! -f "$PREFILL_SOURCE" ]; then
    log_error "Prefill source not found: $PREFILL_SOURCE"
    exit 1
fi

log_info "Decode source: $DECODE_SOURCE"
log_info "Prefill source: $PREFILL_SOURCE"

# ============================================================================
# Step 5: Compile Generated Code
# ============================================================================
log_info "Step 5: Compiling generated code..."

# Common flags
COMMON_FLAGS="-O3 -march=native -mtune=native -std=c11 -ffast-math -funroll-loops -D_GNU_SOURCE"
DEBUG_FLAGS="-g -O0"
RELEASE_FLAGS="-O3 -DNDEBUG"

if [ "$BUILD_TYPE" = "Debug" ]; then
    CFLAGS="$COMMON_FLAGS $DEBUG_FLAGS"
else
    CFLAGS="$COMMON_FLAGS $RELEASE_FLAGS"
fi

# Common include paths (must be defined before use)
# Use absolute paths for reliability
PROJECT_ROOT="/home/antshiv/Workspace/C-Kernel-Engine"
INCLUDE_PATHS="-I$PROJECT_ROOT/include -I$SCRIPT_DIR/include -I$SCRIPT_DIR/src -I$SCRIPT_DIR/src/tokenizer -I$SCRIPT_DIR/src/kernels -I$OUTPUT_DIR"

# Compile decode source
DECODE_OBJ="$OUTPUT_DIR/ck-kernel-inference.o"
gcc $CFLAGS -c "$DECODE_SOURCE" -o "$DECODE_OBJ" $INCLUDE_PATHS
if [ $? -ne 0 ]; then
    log_error "Decode compilation failed"
    exit 1
fi
log_info "Compiled: $DECODE_OBJ"

# Compile prefill source
PREFILL_OBJ="$OUTPUT_DIR/ck-kernel-prefill.o"
gcc $CFLAGS -fopenmp -c "$PREFILL_SOURCE" -o "$PREFILL_OBJ" $INCLUDE_PATHS
if [ $? -ne 0 ]; then
    log_error "Prefill compilation failed"
    exit 1
fi
log_info "Compiled: $PREFILL_OBJ"

# ============================================================================
# Step 7: Compile Kernels
# ============================================================================
log_info "Step 6: Compiling kernels..."

KERNEL_OBJECTS=""
KERNELS_DIR="/home/antshiv/Workspace/C-Kernel-Engine/src/kernels"
for kernel in "$KERNELS_DIR"/*.c; do
    if [ -f "$kernel" ]; then
        kernel_name=$(basename "$kernel" .c)
        kernel_obj="$OUTPUT_DIR/${kernel_name}.o"
        gcc $CFLAGS -fopenmp -c "$kernel" -o "$kernel_obj" $INCLUDE_PATHS
        if [ $? -ne 0 ]; then
            log_error "Kernel compilation failed: $kernel"
            exit 1
        fi
        KERNEL_OBJECTS="$KERNEL_OBJECTS $kernel_obj"
        log_info "  Compiled: $kernel_name.o"
    fi
done

# ============================================================================
# Step 7b: Compile Utility Sources
# ============================================================================
log_info "Step 6b: Compiling utility sources..."

UTILITY_SOURCES="
    ckernel_strict.c
    cpu_features.c
    ckernel_orchestration.c
    ckernel_model_load_v4.c
    ck_tokenizer.c
"

UTILITY_OBJECTS=""
for src in $UTILITY_SOURCES; do
    src_path="/home/antshiv/Workspace/C-Kernel-Engine/src/$src"
    if [ -f "$src_path" ]; then
        obj_name=$(basename "$src" .c)
        obj_path="$OUTPUT_DIR/${obj_name}.o"
        gcc $CFLAGS -fopenmp -c "$src_path" -o "$obj_path" $INCLUDE_PATHS
        if [ $? -ne 0 ]; then
            log_error "Utility compilation failed: $src"
            exit 1
        fi
        UTILITY_OBJECTS="$UTILITY_OBJECTS $obj_path"
        log_info "  Compiled: $obj_name.o"
    fi
done

# ============================================================================
# Step 8: Compile Inference Binary
# ============================================================================
log_info "Step 7: Compiling inference binary..."

INFER_OBJ="$OUTPUT_DIR/v6.5_inference.o"
gcc $CFLAGS -fopenmp -c "$SCRIPT_DIR/v6.5_inference.c" -o "$INFER_OBJ" $INCLUDE_PATHS
if [ $? -ne 0 ]; then
    log_error "Inference compilation failed"
    exit 1
fi
log_info "Compiled: $INFER_OBJ"

# ============================================================================
# Step 9: Link Final Binary
# ============================================================================
log_info "Step 8: Linking final binary..."

FINAL_BINARY="$SCRIPT_DIR/ck-engine-v6.5"

if [ ! -f "$TOKENIZER_LIB" ]; then
    log_info "Tokenizer library missing; building libckernel_tokenizer.so..."
    (cd "$PROJECT_ROOT" && make tokenizer)
fi

gcc $CFLAGS -fopenmp \
    "$INFER_OBJ" \
    "$DECODE_OBJ" \
    "$PREFILL_OBJ" \
    $KERNEL_OBJECTS \
    $UTILITY_OBJECTS \
    $TOKENIZER_LDFLAGS \
    -lm -lpthread \
    -o "$FINAL_BINARY"

if [ $? -ne 0 ]; then
    log_error "Linking failed"
    exit 1
fi

log_info "Binary created: $FINAL_BINARY"

# ============================================================================
# Summary
# ============================================================================
echo ""
log_info "=========================================="
log_info "Build Complete!"
log_info "=========================================="
echo ""
log_info "Binary: $FINAL_BINARY"
log_info "Model: $MODEL_NAME"
log_info "Build type: $BUILD_TYPE"
log_info "Threads: $NUM_THREADS"
echo ""

if [ -f "$WEIGHTS_FILE" ]; then
    log_info "To run: $FINAL_BINARY $WEIGHTS_FILE"
else
    log_warn "Weights file not found: $WEIGHTS_FILE"
    log_info "To run: $FINAL_BINARY <weights.bump>"
fi

exit 0
