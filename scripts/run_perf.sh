#!/bin/bash
# Simple script to run perf on the litmus test
# Usage: sudo scripts/run_perf.sh

set -e

cd /home/antshiv/Workspace/C-Kernel-Engine

# Clean old perf data
rm -f build/*.perf build/*.svg

# Generate test data (same as litmus test)
python3 unittest/test_lm_head_litmus.py --skip-compile --seed 42

# Get the binary name
LITMUS_BIN=$(ls -t build/litmus_generated 2>/dev/null | head -1)

if [ -z "$LITMUS_BIN" ]; then
    echo "Error: litmus binary not found"
    exit 1
fi

echo "Using binary: $LITMUS_BIN"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "This script must be run as root (use sudo)"
    echo "Alternatively, run:"
    echo "  sudo perf record -F 999 -e cycles,cache-misses -g -o build/perf_litmus.perf -- $LITMUS_BIN --litmus --hidden build/litmus_hidden.bin --weights build/litmus_weights.bin --targets build/litmus_targets.bin"
    exit 1
fi

# Run perf record
echo "Recording perf data..."
perf record -F 999 -e cycles,cache-misses -g -o build/perf_litmus.perf \
    -- $LITMUS_BIN --litmus \
    --hidden build/litmus_hidden.bin \
    --weights build/litmus_weights.bin \
    --targets build/litmus_targets.bin \
    --out-logits build/litmus_logits.bin \
    --out-loss build/litmus_loss.bin

echo "Generating flamegraph..."
# Generate flamegraph
perf script -i build/perf_litmus.perf | \
    ~/Programs/FlameGraph/stackcollapse-perf.pl | \
    ~/Programs/FlameGraph/flamegraph.pl > build/perf_litmus.svg

echo "Done! Flamegraph: build/perf_litmus.svg"
echo "Open in browser: file://$(pwd)/build/perf_litmus.svg"
