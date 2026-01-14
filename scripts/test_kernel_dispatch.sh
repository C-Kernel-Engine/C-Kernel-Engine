#!/bin/bash
#
# test_kernel_dispatch.sh - Quick test to verify which kernels are being used
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/build"

CLI="$BUILD_DIR/ck-cli-v6.5"
MODEL="qwen2-0_5b-instruct-q4_k_m"

echo "=========================================="
echo "  Kernel Dispatch Verification"
echo "=========================================="
echo ""

# Check if CLI exists
if [ ! -f "$CLI" ]; then
    echo "ERROR: CLI not found at $CLI"
    echo "Build first with: make ck-cli-v6.5"
    exit 1
fi

echo "Testing kernel dispatch with Q4_K x Q8_K..."
echo ""

# Run with verbose output to see kernel calls
echo "Output:"
echo "---"
"$CLI" --model "$MODEL" --max-tokens 1 --prompt "Hi" 2>&1
echo "---"
echo ""

echo "Expected kernels:"
echo "  - Q4_K x Q8_K should use VNNI or AMX"
echo "  - Check for 'gemv_q4_k_q8_k_vnni' or 'gemv_q4_k_q8_k_amx' in output"
echo ""
echo "If you see AVX2 or reference kernels, dispatch is wrong!"
echo ""
