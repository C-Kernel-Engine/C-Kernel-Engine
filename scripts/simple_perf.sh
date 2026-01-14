#!/bin/bash
# Simple perf test without VTune

CLI="./build/ck-cli-v6.5"
MODEL="qwen2-0_5b-instruct-q4_k_m"
TOKENS=100

if [ ! -f "$CLI" ]; then
    echo "Build first: make ck-cli-v6.5"
    exit 1
fi

echo "Simple Performance Test"
echo "========================"
echo ""

echo "1. Basic run:"
$CLI --model $MODEL --max-tokens 5 --prompt "Hi" 2>&1 | grep -E "(Hardware|total:)"
echo ""

echo "2. With perf (basic):"
if command -v perf &> /dev/null; then
    echo "Running perf stat..."
    perf stat $CLI --model $MODEL --max-tokens $TOKENS --prompt "Hello" 2>&1 | grep -E "(Hardware|total:|cycles|instructions)"
else
    echo "perf not installed"
fi
