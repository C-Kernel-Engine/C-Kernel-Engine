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
FORCE_REBUILD=false
VERBOSE=false
PERF_MODE=false
PERF_LARGE=false

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
        --force-rebuild)
            FORCE_REBUILD=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --perf)
            PERF_MODE=true
            shift
            ;;
        --perf-large)
            PERF_MODE=true
            PERF_LARGE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --quick         Run quick kernel tests only (~30s)"
            echo "  --kernels       Run kernel-level tests only (no full model)"
            echo "  --perf          Run performance benchmarks (CK vs llama.cpp)"
            echo "  --perf-large    Run performance benchmarks with 7B model dimensions"
            echo "  --skip-build    Skip llama.cpp build step"
            echo "  --force-rebuild Force rebuild of llama.cpp (clean build)"
            echo "  --verbose       Verbose output"
            echo "  --help          Show this help"
            echo ""
            echo "Environment variables:"
            echo "  LLAMA_CPP_COMMIT  Pin to specific llama.cpp commit (default: b4876)"
            echo ""
            echo "CI Usage:"
            echo "  # Fresh clone, checkout, patch, build, and test:"
            echo "  rm -rf llama.cpp && ./scripts/run_parity_smoketest.sh"
            echo ""
            echo "  # Use specific commit:"
            echo "  LLAMA_CPP_COMMIT=abc123 ./scripts/run_parity_smoketest.sh"
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
# Note: Use assignment form to avoid ((0)) returning exit code 1 with set -e
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0

incr_passed()  { TESTS_PASSED=$((TESTS_PASSED + 1)); }
incr_failed()  { TESTS_FAILED=$((TESTS_FAILED + 1)); }
incr_skipped() { TESTS_SKIPPED=$((TESTS_SKIPPED + 1)); }

# Pinned llama.cpp commit for reproducible parity testing
# Update this when upgrading llama.cpp version
LLAMA_CPP_COMMIT="${LLAMA_CPP_COMMIT:-b4876}"
LLAMA_CPP_REPO="https://github.com/ggerganov/llama.cpp.git"

# Step 1: Check/build llama.cpp
if [ "$SKIP_BUILD" = false ]; then
    log_step "[1/5] Checking llama.cpp..."

    # Force rebuild: clean existing build
    if [ "$FORCE_REBUILD" = true ] && [ -d "$LLAMA_DIR/build" ]; then
        log_step "Force rebuild: removing llama.cpp/build..."
        rm -rf "$LLAMA_DIR/build"
        rm -f "$LLAMA_DIR/libggml_kernel_test.so"
    fi

    NEED_CHECKOUT=false
    NEED_PATCHES=false

    if [ ! -d "$LLAMA_DIR" ]; then
        log_warn "llama.cpp not found."

        # Try submodule first
        if [ -f "$ROOT_DIR/.gitmodules" ] && grep -q "llama.cpp" "$ROOT_DIR/.gitmodules"; then
            log_step "Initializing llama.cpp submodule..."
            cd "$ROOT_DIR"
            git submodule update --init llama.cpp
            NEED_CHECKOUT=true
            NEED_PATCHES=true
        else
            # Clone directly (for CI environments without submodule setup)
            log_step "Cloning llama.cpp from $LLAMA_CPP_REPO..."
            cd "$ROOT_DIR"
            git clone --depth 100 "$LLAMA_CPP_REPO" llama.cpp
            NEED_CHECKOUT=true
            NEED_PATCHES=true
        fi
    elif [ ! -f "$LLAMA_DIR/build/bin/llama-cli" ]; then
        # Directory exists but not built - may need patches
        NEED_PATCHES=true
    fi

    # Checkout pinned version for reproducibility
    if [ "$NEED_CHECKOUT" = true ] && [ -d "$LLAMA_DIR" ]; then
        log_step "Checking out llama.cpp commit $LLAMA_CPP_COMMIT..."
        cd "$LLAMA_DIR"

        # Fetch if needed (shallow clone may not have the commit)
        if ! git cat-file -e "$LLAMA_CPP_COMMIT" 2>/dev/null; then
            log_step "Fetching commit $LLAMA_CPP_COMMIT..."
            git fetch --depth 100 origin "$LLAMA_CPP_COMMIT" 2>/dev/null || \
            git fetch origin 2>/dev/null || \
            log_warn "Could not fetch commit, using current HEAD"
        fi

        # Checkout the pinned commit
        git checkout "$LLAMA_CPP_COMMIT" 2>/dev/null || \
            log_warn "Could not checkout $LLAMA_CPP_COMMIT, using current HEAD"

        log_step "llama.cpp at: $(git rev-parse --short HEAD)"
        cd "$ROOT_DIR"
    fi

    # Apply patches if needed
    if [ "$NEED_PATCHES" = true ] && [ -d "$PATCHES_DIR" ] && [ -d "$LLAMA_DIR" ]; then
        cd "$LLAMA_DIR"

        # Apply tensor dump patch
        if [ -f "$PATCHES_DIR/llama.patch" ]; then
            log_step "Applying tensor dump patch..."
            git apply "$PATCHES_DIR/llama.patch" 2>/dev/null || \
                log_warn "Patch already applied or failed to apply"
        fi

        # Copy kernel parity test file
        if [ -f "$PATCHES_DIR/test-kernel-parity.cpp" ]; then
            log_step "Copying kernel parity test..."
            mkdir -p tests
            cp "$PATCHES_DIR/test-kernel-parity.cpp" tests/
        fi

        cd "$ROOT_DIR"
    fi

    # Build llama.cpp if needed
    if [ ! -f "$LLAMA_DIR/build/bin/llama-cli" ]; then
        log_step "Building llama.cpp..."
        cd "$LLAMA_DIR"

        # Configure and build
        cmake -B build \
            -DGGML_CPU=ON \
            -DLLAMA_BUILD_TESTS=OFF \
            -DLLAMA_CURL=OFF \
            -DCMAKE_BUILD_TYPE=Release
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
    # Ensure dependencies are found
    export LD_LIBRARY_PATH="$LLAMA_DIR/build/bin:$BUILD_DIR:$LD_LIBRARY_PATH"
    
    set +e
    if [ "$QUICK_MODE" = true ]; then
        python3 "$KERNEL_TEST" --quick
        RET=$?
    else
        python3 "$KERNEL_TEST" --all
        RET=$?
    fi
    set -e

    if [ $RET -eq 0 ]; then
        log_success "Kernel tests passed"
        incr_passed
    else
        log_error "Kernel tests failed"
        incr_failed
    fi
else
    log_warn "Kernel test script not found: $KERNEL_TEST"
    log_warn "Run existing PyTorch parity tests instead..."

    # Fallback to existing PyTorch parity tests
    PYTORCH_TEST="$ROOT_DIR/unittest/test_pytorch_parity.py"
    if [ -f "$PYTORCH_TEST" ]; then
        python3 "$PYTORCH_TEST" 2>&1 && {
            log_success "PyTorch parity tests passed"
            incr_passed
        } || {
            log_error "PyTorch parity tests failed"
            incr_failed
        }
    else
        log_warn "No kernel tests available"
        incr_skipped
    fi
fi

# Step 3b: Comprehensive GEMV kernel tests (Q4_K, Q5_0, Q8_0)
log_step "[3b/5] Running comprehensive GEMV kernel tests..."
GEMV_TEST="$ROOT_DIR/unittest/test_gemv_kernels_comprehensive.py"
if [ -f "$GEMV_TEST" ]; then
    set +e
    if [ "$QUICK_MODE" = true ]; then
        python3 "$GEMV_TEST" --quick
        RET=$?
    elif [ "$PERF_LARGE" = true ]; then
        python3 "$GEMV_TEST" --large
        RET=$?
    else
        python3 "$GEMV_TEST"
        RET=$?
    fi
    set -e

    if [ $RET -eq 0 ]; then
        log_success "Comprehensive GEMV tests passed"
        incr_passed
    else
        log_error "Comprehensive GEMV tests failed"
        incr_failed
    fi
else
    log_warn "Comprehensive GEMV test not found: $GEMV_TEST"
    incr_skipped
fi

# Step 3c: Performance benchmarks (if --perf specified)
if [ "$PERF_MODE" = true ]; then
    log_step "[3c/5] Running performance benchmarks..."

    if [ -f "$KERNEL_TEST" ]; then
        set +e
        if [ "$PERF_LARGE" = true ]; then
            python3 "$KERNEL_TEST" --perf-large
            RET=$?
        else
            python3 "$KERNEL_TEST" --perf
            RET=$?
        fi
        set -e

        if [ $RET -eq 0 ]; then
            log_success "Performance benchmarks completed"
            incr_passed
        else
            log_error "Performance benchmarks failed"
            incr_failed
        fi
    else
        log_warn "Kernel test script not found, skipping performance benchmarks"
        incr_skipped
    fi
fi

# Step 4: PyTorch FP32 reference tests
if [ "$KERNELS_ONLY" = false ]; then
    log_step "[4/5] Running PyTorch reference tests..."

    # Ensure engine library can be loaded
    export LD_LIBRARY_PATH="$ROOT_DIR/build:$LD_LIBRARY_PATH"

    PYTORCH_TEST="$ROOT_DIR/unittest/test_pytorch_parity.py"
    if [ -f "$PYTORCH_TEST" ]; then
        set +e
        if [ "$QUICK_MODE" = true ]; then
            # Quick mode: run subset
            python3 "$PYTORCH_TEST" --quick
            RET=$?
        else
            python3 "$PYTORCH_TEST"
            RET=$?
        fi
        set -e

        if [ $RET -eq 0 ]; then
            log_success "PyTorch reference tests passed"
            incr_passed
        else
            log_error "PyTorch reference tests failed"
            incr_failed
        fi
    else
        log_warn "PyTorch parity test not found"
        incr_skipped
    fi
else
    log_step "[4/5] Skipping PyTorch tests (--kernels)"
    incr_skipped
fi

# Step 5: Full model parity (if not quick/kernels-only)
if [ "$KERNELS_ONLY" = false ] && [ "$QUICK_MODE" = false ]; then
    log_step "[5/5] Running full model parity test..."

    MODEL_PARITY_TEST="$SCRIPT_DIR/compare_runtime_parity.py"
    LLAMA_DUMP_DIR="$ROOT_DIR/llama_dump"
    CK_DUMP_DIR="$ROOT_DIR/parity"

    if [ -f "$MODEL_PARITY_TEST" ]; then
        # Check if dump directories exist (requires running llama.cpp and CK with parity flags)
        if [ -d "$LLAMA_DUMP_DIR" ] && [ -d "$CK_DUMP_DIR" ]; then
            set +e
            python3 "$MODEL_PARITY_TEST" \
                --llama-dump "$LLAMA_DUMP_DIR" \
                --ck-dump "$CK_DUMP_DIR" \
                --tol 1e-3 2>&1
            RET=$?
            set -e

            if [ $RET -eq 0 ]; then
                log_success "Full model parity test passed"
                incr_passed
            else
                log_error "Full model parity test failed"
                incr_failed
            fi
        else
            log_warn "Tensor dump directories not found, skipping full model parity"
            log_warn "  To generate dumps:"
            log_warn "    1. Run llama.cpp with tensor dump patch: llama.cpp/build/bin/llama-cli -m model.gguf -p 'Hello' -n 1"
            log_warn "    2. Run CK with --parity flag: python scripts/ck_run_v5.py run model_dir --parity"
            incr_skipped
        fi
    else
        log_warn "Model parity script not found: $MODEL_PARITY_TEST"
        incr_skipped
    fi
else
    log_step "[5/5] Skipping full model test (quick/kernels mode)"
    incr_skipped
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
