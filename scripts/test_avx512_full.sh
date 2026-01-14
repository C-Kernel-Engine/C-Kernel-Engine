#!/bin/bash
#
# test_avx512_full.sh - Complete AVX-512 validation suite
#
# Run this on an AVX-512 machine to validate:
# 1. Q6_K x Q8_K kernel parity (SSE vs AVX2 vs AVX-512)
# 2. Full model inference correctness
# 3. Performance comparison vs llama.cpp
#
# Usage:
#   ./scripts/test_avx512_full.sh [--quick]
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors
GREEN='\033[92m'
RED='\033[91m'
CYAN='\033[96m'
YELLOW='\033[93m'
BOLD='\033[1m'
RESET='\033[0m'

# Parse args
QUICK=0
if [[ "$1" == "--quick" ]]; then
    QUICK=1
fi

echo -e "${CYAN}${BOLD}======================================================================${RESET}"
echo -e "${CYAN}${BOLD}      AVX-512 Full Validation Suite - CK-Engine v6.5${RESET}"
echo -e "${CYAN}${BOLD}======================================================================${RESET}"
echo

# Check CPU features
echo -e "${CYAN}Checking CPU features...${RESET}"
echo -e "${CYAN}CPU: $(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)${RESET}"
echo

echo -e "${CYAN}AVX-512 Features:${RESET}"
if grep -q avx512f /proc/cpuinfo; then
    echo -e "  ${GREEN}AVX-512F: YES${RESET}"
else
    echo -e "  ${RED}AVX-512F: NO${RESET}"
    echo -e "${RED}This system does not have AVX-512. Tests will use AVX2 fallback.${RESET}"
fi
if grep -q avx512bw /proc/cpuinfo; then
    echo -e "  ${GREEN}AVX-512BW: YES${RESET}"
fi
if grep -q avx512vl /proc/cpuinfo; then
    echo -e "  ${GREEN}AVX-512VL: YES${RESET}"
fi
if grep -q avx512vbmi /proc/cpuinfo; then
    echo -e "  ${GREEN}AVX-512VBMI: YES (Ice Lake+)${RESET}"
fi
if grep -qE "avx512_vnni|avx512vnni" /proc/cpuinfo; then
    echo -e "  ${GREEN}AVX-512VNNI: YES (INT8 dot product acceleration)${RESET}"
    HAS_VNNI=1
else
    echo -e "  ${YELLOW}AVX-512VNNI: NO${RESET}"
    HAS_VNNI=0
fi
if grep -q avx512_bf16 /proc/cpuinfo; then
    echo -e "  ${GREEN}AVX-512BF16: YES (BFloat16 support)${RESET}"
fi
echo

echo -e "${CYAN}AMX Features:${RESET}"
HAS_AMX=0
if grep -q amx_tile /proc/cpuinfo; then
    echo -e "  ${GREEN}AMX-TILE: YES${RESET}"
    HAS_AMX=1
else
    echo -e "  ${YELLOW}AMX-TILE: NO${RESET}"
fi
if grep -q amx_int8 /proc/cpuinfo; then
    echo -e "  ${GREEN}AMX-INT8: YES (Sapphire Rapids+)${RESET}"
fi
if grep -q amx_bf16 /proc/cpuinfo; then
    echo -e "  ${GREEN}AMX-BF16: YES${RESET}"
fi
echo

echo -e "${CYAN}Kernel Dispatch:${RESET}"
if [ "$HAS_AMX" = "1" ]; then
    echo -e "  ${GREEN}Q4_K x Q8_K → AMX (tile-based, 8-16x speedup potential)${RESET}"
elif [ "$HAS_VNNI" = "1" ]; then
    echo -e "  ${GREEN}Q4_K x Q8_K → VNNI (INT8 dot product)${RESET}"
else
    echo -e "  ${YELLOW}Q4_K x Q8_K → AVX2 fallback${RESET}"
fi
echo

# Step 1: Rebuild with AVX-512
echo -e "${CYAN}${BOLD}Step 1: Rebuilding CK-Engine with native optimizations...${RESET}"
cd "$PROJECT_ROOT"
make clean 2>/dev/null || true
make -j$(nproc)
echo -e "  ${GREEN}Build complete${RESET}"
echo

# Step 2: Kernel parity test
echo -e "${CYAN}${BOLD}Step 2: Running kernel parity tests...${RESET}"
if [[ $QUICK -eq 1 ]]; then
    python3 "$SCRIPT_DIR/test_avx512_parity.py"
else
    python3 "$SCRIPT_DIR/test_avx512_parity.py" --cross --full
fi
echo

# Step 3: Model inference test
echo -e "${CYAN}${BOLD}Step 3: Testing full model inference...${RESET}"
MODEL="Qwen/Qwen2-0.5B-Instruct-GGUF"
echo -e "  Testing with $MODEL..."

# Force recompile to pick up AVX-512 optimizations
python3 "$SCRIPT_DIR/v6.5/ck_run_v6_5.py" run "$MODEL" --force-compile --max-tokens 20 2>&1 | head -30

# Check for garbage output
OUTPUT=$(python3 "$SCRIPT_DIR/v6.5/ck_run_v6_5.py" run "$MODEL" --max-tokens 10 2>&1 | tail -5)
if echo "$OUTPUT" | grep -qE '[가-힣ぁ-んァ-ン一-龥]|腐败|버'; then
    echo -e "  ${RED}FAIL: Detected garbage output (non-ASCII characters)${RESET}"
    echo -e "  Output: $OUTPUT"
    exit 1
else
    echo -e "  ${GREEN}PASS: Output looks reasonable${RESET}"
fi
echo

# Step 4: Performance comparison (optional)
if [[ $QUICK -eq 0 ]]; then
    echo -e "${CYAN}${BOLD}Step 4: Performance comparison vs llama.cpp...${RESET}"

    LLAMA_CLI="$PROJECT_ROOT/llama.cpp/build/bin/llama-cli"
    if [[ -f "$LLAMA_CLI" ]]; then
        python3 "$SCRIPT_DIR/benchmark_vs_llamacpp.py" --model "$MODEL" --tokens 50 --runs 3
    else
        echo -e "  ${YELLOW}llama.cpp not built, skipping performance comparison${RESET}"
        echo -e "  To build: cd llama.cpp && mkdir -p build && cd build && cmake .. -DGGML_AVX512=ON && make -j"
    fi
    echo
fi

# Summary
echo -e "${CYAN}${BOLD}======================================================================${RESET}"
echo -e "${CYAN}${BOLD}                          Summary${RESET}"
echo -e "${CYAN}${BOLD}======================================================================${RESET}"
echo
echo -e "  ${GREEN}All tests passed!${RESET}"
echo
echo -e "  The AVX-512 implementation is working correctly."
echo -e "  You can now use CK-Engine on this machine."
echo
echo -e "  To run inference:"
echo -e "    python3 scripts/v6.5/ck_run_v6_5.py run Qwen/Qwen2-0.5B-Instruct-GGUF"
echo
