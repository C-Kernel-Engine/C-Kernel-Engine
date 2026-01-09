#!/bin/bash
# Run litmus in a loop to reduce startup overhead in perf
# Usage: sudo scripts/perf_loop.sh

set -e

cd /home/antshiv/Workspace/C-Kernel-Engine

# Number of iterations
ITERATIONS=${1:-100}

echo "Running litmus test $ITERATIONS times in a loop..."

# Create a wrapper that runs multiple times
cat > /tmp/litmus_loop.c << 'EOF'
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    int iters = (argc > 1) ? atoi(argv[1]) : 100;

    for (int i = 0; i < iters; i++) {
        system("./build/litmus_generated --litmus --hidden build/litmus_hidden.bin --weights build/litmus_weights.bin --targets build/litmus_targets.bin > /dev/null 2>&1");
    }

    return 0;
}
EOF

gcc -O3 -o /tmp/litmus_loop /tmp/litmus_loop.c

# Run perf on the loop
echo "Recording perf (this will take a while)..."
sudo perf record -F 999 -e cycles,cache-misses -g -o build/perf_loop.perf \
    -- /tmp/litmus_loop $ITERATIONS

# Generate flamegraph
echo "Generating flamegraph..."
perf script -i build/perf_loop.perf | \
    ~/Programs/FlameGraph/stackcollapse-perf.pl | \
    ~/Programs/FlameGraph/flamegraph.pl > build/perf_loop.svg

echo "Done! Flamegraph: build/perf_loop.svg"
echo "Flamegraph (litmus only): build/perf_litmus.svg"
