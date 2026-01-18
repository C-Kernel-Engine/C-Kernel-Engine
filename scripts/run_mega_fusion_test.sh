#!/bin/bash
# scripts/run_mega_fusion_test.sh
#
# Mega-Fusion Attention Test
# Compares CK-Engine's mega_fused_attention_prefill against llama.cpp
#
# PRIORITY: NUMERICAL ACCURACY (Primary) > Performance (Secondary)
#
# Key principles:
# 1. Numerical parity is UTMOST important (not performance)
# 2. Test with REAL weights to avoid "0 × ∞ problem"
# 3. Zero weights give zero output and hide numerical errors
# 4. Performance is secondary to correctness
#
# Usage:
#   ./scripts/run_mega_fusion_test.sh              # Full test
#   ./scripts/run_mega_fusion_test.sh --quick      # Quick test
#   ./scripts/run_mega_fusion_test.sh --skip-build # Skip llama.cpp build
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
LLAMA_DIR="$ROOT_DIR/llama.cpp"
PATCHES_DIR="$ROOT_DIR/patches"
BUILD_DIR="$ROOT_DIR/build"

# Default options
QUICK_MODE=false
SKIP_BUILD=false
SKIP_TESTS=false
VERBOSE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
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
            echo "  --quick         Run quick test (~30s)"
            echo "  --skip-build    Skip llama.cpp build step"
            echo "  --skip-tests    Build only, don't run tests"
            echo "  --verbose       Verbose output"
            echo "  --help          Show this help"
            echo ""
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
echo "  Mega-Fusion Attention Test"
echo "  CK-Engine vs llama.cpp"
echo "=============================================="
echo ""
echo "Root directory: $ROOT_DIR"
echo "Quick mode: $QUICK_MODE"
echo ""

# Track results
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0

incr_passed()  { TESTS_PASSED=$((TESTS_PASSED + 1)); }
incr_failed()  { TESTS_FAILED=$((TESTS_FAILED + 1)); }
incr_skipped() { TESTS_SKIPPED=$((TESTS_SKIPPED + 1)); }

# Step 1: Check/build llama.cpp
if [ "$SKIP_BUILD" = false ]; then
    log_step "[1/4] Checking llama.cpp..."

    # Build kernel test library if needed
    if [ ! -f "$LLAMA_DIR/libggml_kernel_test.so" ]; then
        log_step "Building llama.cpp kernel test library..."

        cd "$LLAMA_DIR"

        # Detect CPU capabilities
        HAS_SSSE3=$(grep -q ssse3 /proc/cpuinfo 2>/dev/null && echo "yes" || echo "no")
        if [ "$HAS_SSSE3" = "yes" ]; then
            ARCH_FLAGS="-mssse3 -march=core2"
        else
            ARCH_FLAGS=""
        fi

        # Build test library
        if [ -f "tests/test-kernel-parity.cpp" ]; then
            export LD_LIBRARY_PATH="$PWD/build/bin:$LD_LIBRARY_PATH"

            if g++ -shared -fPIC $ARCH_FLAGS -o libggml_kernel_test.so \
                tests/test-kernel-parity.cpp \
                -I ggml/include -I ggml/src \
                -L build/bin -lggml -lggml-cpu -lggml-base -lm -lpthread \
                -Wl,-rpath,'$ORIGIN/build/bin' 2>&1; then
                log_step "Built ggml kernel test library"
            else
                log_warn "Could not build kernel test library"
            fi
        else
            log_warn "test-kernel-parity.cpp not found"
        fi
        cd "$ROOT_DIR"
    else
        log_step "llama.cpp kernel test library already exists"
    fi
else
    log_step "[1/4] Skipping llama.cpp build (--skip-build)"
fi

# Step 2: Build CK-Engine
log_step "[2/4] Building CK-Engine..."
cd "$ROOT_DIR"

if make -j4 2>&1; then
    log_step "Built CK-Engine library"
else
    log_error "Failed to build CK-Engine"
    exit 1
fi

# Exit early if --skip-tests was specified
if [ "$SKIP_TESTS" = true ]; then
    log_success "Build complete (--skip-tests specified)"
    echo ""
    echo "Libraries built:"
    [ -f "$LLAMA_DIR/libggml_kernel_test.so" ] && echo "  llama.cpp: $LLAMA_DIR/libggml_kernel_test.so"
    [ -f "$BUILD_DIR/libckernel_engine.so" ] && echo "  CK-Engine: $BUILD_DIR/libckernel_engine.so"
    echo ""
    exit 0
fi

# Step 3: Build CK parity library (needed for attention kernel tests)
log_step "[3/6] Building CK parity library..."

cd "$ROOT_DIR"
if make libck_parity.so 2>&1; then
    log_step "Built CK parity library"
else
    log_warn "Failed to build CK parity library, some tests may be skipped"
fi

# Step 4: Run mega-fusion attention parity tests
log_step "[4/6] Running attention kernel parity tests (CK vs llama.cpp)..."

# Ensure libraries are in path
export LD_LIBRARY_PATH="$LLAMA_DIR/build/bin:$BUILD_DIR:$LD_LIBRARY_PATH"

# Run the attention parity test with numerical accuracy AND performance
set +e
if [ "$QUICK_MODE" = true ]; then
    python3 unittest/fusion/test_mega_fusion_parity.py --quick --perf 2>&1
    RET=$?
else
    python3 unittest/fusion/test_mega_fusion_parity.py --verbose --perf 2>&1
    RET=$?
fi
set -e

if [ $RET -eq 0 ]; then
    log_success "Attention kernel parity test passed"
    incr_passed
else
    log_error "Attention kernel parity test failed"
    incr_failed
fi

# Step 5: Run OutProj+MLP fusion parity test
log_step "[5/6] Running OutProj+MLP fusion parity tests (CK vs llama.cpp)..."

if [ -f "unittest/fusion/test_mega_fusion_outproj_mlp_parity.py" ]; then
    set +e
    if [ "$QUICK_MODE" = true ]; then
        python3 unittest/fusion/test_mega_fusion_outproj_mlp_parity.py --quick 2>&1
        RET=$?
    else
        python3 unittest/fusion/test_mega_fusion_outproj_mlp_parity.py 2>&1
        RET=$?
    fi
    set -e

    if [ $RET -eq 0 ]; then
        log_success "OutProj+MLP fusion parity test passed"
        incr_passed
    else
        log_error "OutProj+MLP fusion parity test failed"
        incr_failed
    fi
else
    log_warn "OutProj+MLP parity test script not found, skipping"
    incr_skipped
fi

# Step 6: Run mega-fusion prefill performance benchmark (CK internal)
log_step "[6/6] Running mega-fusion prefill benchmark..."

if [ -f "scripts/bench_mega_fused_attention_prefill.py" ]; then
    set +e
    python3 scripts/bench_mega_fused_attention_prefill.py 2>&1
    RET=$?
    set -e

    if [ $RET -eq 0 ]; then
        log_success "Mega-fusion prefill benchmark completed"
        incr_passed
    else
        log_warn "Mega-fusion prefill benchmark failed or not available"
        incr_skipped
    fi
else
    log_warn "Benchmark script not found, skipping"
    incr_skipped
fi

# Summary
echo ""
echo "=============================================="
echo "  MEGA-FUSION TEST SUMMARY"
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
    log_success "All mega-fusion tests passed!"
    exit 0
fi
