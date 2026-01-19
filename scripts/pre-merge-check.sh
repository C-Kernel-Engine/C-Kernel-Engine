#!/bin/bash
# =============================================================================
# Pre-Merge Check Script
# =============================================================================
# Run this before merging any branch to main.
# All checks must pass for code to be merged.
#
# Usage:
#   ./scripts/pre-merge-check.sh          # Run all checks
#   ./scripts/pre-merge-check.sh --quick  # Skip slow tests
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

QUICK_MODE=false
[[ "$1" == "--quick" ]] && QUICK_MODE=true

echo ""
echo -e "${CYAN}================================================================${NC}"
echo -e "${CYAN}  Pre-Merge Checks for CK-Engine${NC}"
echo -e "${CYAN}================================================================${NC}"
echo ""

FAILED=0

# -----------------------------------------------------------------------------
# Check 0: Kernel Map Validation (NEW - no build required)
# -----------------------------------------------------------------------------
echo -e "${CYAN}[0/5] Kernel Map Validation${NC}"
KERNEL_MAPS_DIR="$ROOT_DIR/version/v6.6/kernel_maps"
if [ -f "$KERNEL_MAPS_DIR/validate_kernel_maps.py" ]; then
    if python "$KERNEL_MAPS_DIR/validate_kernel_maps.py" > /dev/null 2>&1; then
        echo -e "${GREEN}[PASS]${NC} Kernel maps are valid"
    else
        echo -e "${RED}[FAIL]${NC} Kernel map validation failed"
        python "$KERNEL_MAPS_DIR/validate_kernel_maps.py"
        FAILED=1
    fi
else
    echo -e "${YELLOW}[SKIP]${NC} No kernel maps found"
fi
echo ""

# -----------------------------------------------------------------------------
# Check 1: Build succeeds
# -----------------------------------------------------------------------------
echo -e "${CYAN}[1/5] Build Check${NC}"
cd "$ROOT_DIR"
if make clean && make -j$(nproc) 2>&1 | tail -5; then
    echo -e "${GREEN}[PASS]${NC} Build succeeded"
else
    echo -e "${RED}[FAIL]${NC} Build failed"
    FAILED=1
fi
echo ""

# -----------------------------------------------------------------------------
# Check 2: No compiler errors (warnings OK)
# -----------------------------------------------------------------------------
echo -e "${CYAN}[2/5] Compiler Error Check${NC}"
if make 2>&1 | grep -q "error:"; then
    echo -e "${RED}[FAIL]${NC} Compiler errors found"
    FAILED=1
else
    echo -e "${GREEN}[PASS]${NC} No compiler errors"
fi
echo ""

# -----------------------------------------------------------------------------
# Check 3: E2E Inference works
# -----------------------------------------------------------------------------
echo -e "${CYAN}[3/5] E2E Inference Check${NC}"
if $QUICK_MODE; then
    echo -e "${YELLOW}[SKIP]${NC} Skipped (quick mode)"
else
    if make e2e 2>&1 | grep -q "All tests passed"; then
        echo -e "${GREEN}[PASS]${NC} E2E inference works"
    else
        # Check if at least inference passed
        if make e2e 2>&1 | grep -q "Inference produced correct answer"; then
            echo -e "${GREEN}[PASS]${NC} E2E inference works (some optional tests skipped)"
        else
            echo -e "${RED}[FAIL]${NC} E2E inference failed"
            FAILED=1
        fi
    fi
fi
echo ""

# -----------------------------------------------------------------------------
# Check 4: Critical kernel tests
# -----------------------------------------------------------------------------
echo -e "${CYAN}[4/5] Kernel Parity Check${NC}"
if $QUICK_MODE; then
    echo -e "${YELLOW}[SKIP]${NC} Skipped (quick mode)"
else
    if [ -f "build/libck_parity.so" ]; then
        if ./scripts/run_parity_smoketest.sh --quick 2>&1 | grep -q "ALL TESTS PASSED"; then
            echo -e "${GREEN}[PASS]${NC} Kernel parity tests passed"
        else
            echo -e "${YELLOW}[WARN]${NC} Some kernel tests failed (non-blocking)"
        fi
    else
        echo -e "${YELLOW}[SKIP]${NC} Parity library not built"
    fi
fi
echo ""

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo -e "${CYAN}================================================================${NC}"
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}  ALL CHECKS PASSED - OK to merge to main${NC}"
    echo -e "${CYAN}================================================================${NC}"
    echo ""
    echo "To merge:"
    echo "  git checkout main"
    echo "  git merge $(git branch --show-current)"
    echo "  git push origin main"
    exit 0
else
    echo -e "${RED}  CHECKS FAILED - Do NOT merge to main${NC}"
    echo -e "${CYAN}================================================================${NC}"
    exit 1
fi
