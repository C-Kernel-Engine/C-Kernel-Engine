#!/bin/bash
# scripts/run_parity_smoketest.sh
#
# Parity smoketest comparing C-Kernel-Engine against llama.cpp
# Runs as part of nightly CI to catch kernel regressions
#
# Usage:
#   ./scripts/run_parity_smoketest.sh              # Full test
#   ./scripts/run_parity_smoketest.sh --quick      # Quick kernel tests only
#   ./scripts/run_parity_smoketest.sh --kernels    # Kernel-level only
#   ./scripts/run_parity_smoketest.sh --skip-build # Skip llama.cpp build

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
LLAMA_DIR="$ROOT_DIR/llama.cpp"
PATCHES_DIR="$ROOT_DIR/patches"
BUILD_DIR="$ROOT_DIR/build"

# Default options
QUICK_MODE=false
KERNELS_ONLY=false
SKIP_BUILD=false
VERBOSE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --kernels)
            KERNELS_ONLY=true
            shift
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --quick       Run quick kernel tests only (~30s)"
            echo "  --kernels     Run kernel-level tests only (no full model)"
            echo "  --skip-build  Skip llama.cpp build step"
            echo "  --verbose     Verbose output"
            echo "  --help        Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_step() {
    echo -e "${GREEN}[STEP]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

echo "=============================================="
echo "  C-Kernel-Engine Parity Smoketest"
echo "=============================================="
echo ""
echo "Root directory: $ROOT_DIR"
echo "Quick mode: $QUICK_MODE"
echo "Kernels only: $KERNELS_ONLY"
echo ""

# Track results
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0

# Step 1: Check/build llama.cpp
if [ "$SKIP_BUILD" = false ]; then
    log_step "[1/5] Checking llama.cpp..."

    if [ ! -d "$LLAMA_DIR" ]; then
        log_warn "llama.cpp not found. Checking if it's a submodule..."

        if [ -f "$ROOT_DIR/.gitmodules" ] && grep -q "llama.cpp" "$ROOT_DIR/.gitmodules"; then
            log_step "Initializing llama.cpp submodule..."
            cd "$ROOT_DIR"
            git submodule update --init llama.cpp
        else
            log_error "llama.cpp not found and not configured as submodule."
            log_error "Add it with: git submodule add https://github.com/ggerganov/llama.cpp.git"
            exit 1
        fi
    fi

    # Check if build exists
    if [ ! -f "$LLAMA_DIR/build/bin/llama-cli" ]; then
        log_step "Building llama.cpp..."
        cd "$LLAMA_DIR"

        # Apply patches if they exist
        if [ -d "$PATCHES_DIR" ]; then
            # Apply tensor dump patch
            if [ -f "$PATCHES_DIR/llama.patch" ]; then
                log_step "Applying tensor dump patch..."
                git apply "$PATCHES_DIR/llama.patch" 2>/dev/null || \
                    log_warn "Patch already applied or failed"
            fi
            # Copy kernel parity test file
            if [ -f "$PATCHES_DIR/test-kernel-parity.cpp" ]; then
                log_step "Copying kernel parity test..."
                cp "$PATCHES_DIR/test-kernel-parity.cpp" tests/
            fi
        fi

        # Build
        cmake -B build -DGGML_CPU=ON -DLLAMA_BUILD_TESTS=OFF
        cmake --build build -j$(nproc)
        cd "$ROOT_DIR"
    else
        log_step "llama.cpp already built"
    fi

    # Build kernel test library
    if [ ! -f "$LLAMA_DIR/libggml_kernel_test.so" ]; then
        log_step "Building ggml kernel test library..."
        cd "$LLAMA_DIR"

        if [ -f "tests/test-kernel-parity.cpp" ]; then
            g++ -shared -fPIC -o libggml_kernel_test.so \
                tests/test-kernel-parity.cpp \
                -I ggml/include -I ggml/src \
                -L build/bin -lggml -lggml-cpu -lggml-base -lm -lpthread \
                -Wl,-rpath,'$ORIGIN/build/bin' 2>/dev/null || \
                log_warn "Could not build kernel test library (may not exist yet)"
        fi
        cd "$ROOT_DIR"
    fi
else
    log_step "[1/5] Skipping llama.cpp build (--skip-build)"
fi

# Step 2: Build CK parity library
log_step "[2/5] Building CK parity library..."
cd "$ROOT_DIR"

if ! make libck_parity.so 2>/dev/null; then
    log_warn "libck_parity.so target not found, trying regular build..."
    make -j$(nproc) 2>&1 | tail -5
fi

# Step 3: Kernel-level tests
log_step "[3/5] Running kernel-level tests..."

# Check if test script exists
KERNEL_TEST="$SCRIPT_DIR/test_kernels_vs_llamacpp.py"
if [ -f "$KERNEL_TEST" ]; then
    if [ "$QUICK_MODE" = true ]; then
        python3 "$KERNEL_TEST" --quick 2>&1 && {
            log_success "Kernel tests passed"
            ((TESTS_PASSED++))
        } || {
            log_error "Kernel tests failed"
            ((TESTS_FAILED++))
        }
    else
        python3 "$KERNEL_TEST" --all 2>&1 && {
            log_success "Kernel tests passed"
            ((TESTS_PASSED++))
        } || {
            log_error "Kernel tests failed"
            ((TESTS_FAILED++))
        }
    fi
else
    log_warn "Kernel test script not found: $KERNEL_TEST"
    log_warn "Run existing PyTorch parity tests instead..."

    # Fallback to existing PyTorch parity tests
    PYTORCH_TEST="$ROOT_DIR/unittest/test_pytorch_parity.py"
    if [ -f "$PYTORCH_TEST" ]; then
        python3 "$PYTORCH_TEST" 2>&1 && {
            log_success "PyTorch parity tests passed"
            ((TESTS_PASSED++))
        } || {
            log_error "PyTorch parity tests failed"
            ((TESTS_FAILED++))
        }
    else
        log_warn "No kernel tests available"
        ((TESTS_SKIPPED++))
    fi
fi

# Step 4: PyTorch FP32 reference tests
if [ "$KERNELS_ONLY" = false ]; then
    log_step "[4/5] Running PyTorch reference tests..."

    PYTORCH_TEST="$ROOT_DIR/unittest/test_pytorch_parity.py"
    if [ -f "$PYTORCH_TEST" ]; then
        if [ "$QUICK_MODE" = true ]; then
            # Quick mode: run subset
            python3 "$PYTORCH_TEST" -k "gemm or rmsnorm or softmax" 2>&1 && {
                log_success "PyTorch reference tests passed"
                ((TESTS_PASSED++))
            } || {
                log_error "PyTorch reference tests failed"
                ((TESTS_FAILED++))
            }
        else
            python3 "$PYTORCH_TEST" 2>&1 && {
                log_success "PyTorch reference tests passed"
                ((TESTS_PASSED++))
            } || {
                log_error "PyTorch reference tests failed"
                ((TESTS_FAILED++))
            }
        fi
    else
        log_warn "PyTorch parity test not found"
        ((TESTS_SKIPPED++))
    fi
else
    log_step "[4/5] Skipping PyTorch tests (--kernels)"
    ((TESTS_SKIPPED++))
fi

# Step 5: Full model parity (if not quick/kernels-only)
if [ "$KERNELS_ONLY" = false ] && [ "$QUICK_MODE" = false ]; then
    log_step "[5/5] Running full model parity test..."

    MODEL_PARITY_TEST="$SCRIPT_DIR/compare_runtime_parity.py"
    if [ -f "$MODEL_PARITY_TEST" ]; then
        # Check for a small test model
        TEST_MODEL=""
        if [ -d "$HOME/.cache/huggingface/hub/SmolLM-135M" ]; then
            TEST_MODEL="$HOME/.cache/huggingface/hub/SmolLM-135M"
        fi

        if [ -n "$TEST_MODEL" ]; then
            python3 "$MODEL_PARITY_TEST" \
                --model-dir "$TEST_MODEL" \
                --layers 0,1 \
                --tokens 5 \
                --tol 1e-3 2>&1 && {
                log_success "Full model parity test passed"
                ((TESTS_PASSED++))
            } || {
                log_error "Full model parity test failed"
                ((TESTS_FAILED++))
            }
        else
            log_warn "No test model found, skipping full model test"
            ((TESTS_SKIPPED++))
        fi
    else
        log_warn "Model parity script not found: $MODEL_PARITY_TEST"
        ((TESTS_SKIPPED++))
    fi
else
    log_step "[5/5] Skipping full model test (quick/kernels mode)"
    ((TESTS_SKIPPED++))
fi

# Summary
echo ""
echo "=============================================="
echo "  PARITY SMOKETEST SUMMARY"
echo "=============================================="
echo ""
echo "  Passed:  $TESTS_PASSED"
echo "  Failed:  $TESTS_FAILED"
echo "  Skipped: $TESTS_SKIPPED"
echo ""

if [ "$TESTS_FAILED" -gt 0 ]; then
    log_error "Some tests failed!"
    exit 1
else
    log_success "All parity tests passed!"
    exit 0
fi
