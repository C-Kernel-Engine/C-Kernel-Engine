#!/bin/bash
#
# Mega-Fused Attention: DRAM Pressure Measurement Script
#
# THIS IS THE CRITICAL TEST - Mega-fusion's whole point is reducing DRAM traffic!
#
# This script measures:
# 1. cache-misses (LLC misses = DRAM access)
# 2. LLC-loads/stores (L3 cache traffic)
# 3. L1-dcache-load-misses (L1 misses = L2/L3 access)
# 4. Generates flamegraph for visual confirmation
#
# Expected results:
# - Unfused: ~800KB/token DRAM traffic
# - Mega-Fused: ~8KB/token (100x reduction!)
#

set -e

# Configuration
MODEL="${1:-qwen2-0_5b-instruct-q4_k_m}"
MAX_TOKENS="${2:-50}"
ITERATIONS="${3:-2}"
PROJECT_DIR="/home/antshiv/Workspace/C-Kernel-Engine"
BUILD_DIR="$PROJECT_DIR/build"
TEST_RESULTS_DIR="$PROJECT_DIR/test_results"
FLAMEGRAPH_DIR="$PROJECT_DIR/FlameGraph"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  MEGA-FUSED ATTENTION: DRAM PRESSURE MEASUREMENT${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""
echo -e "${YELLOW}Model:${NC} $MODEL"
echo -e "${YELLOW}Tokens:${NC} $MAX_TOKENS"
echo -e "${YELLOW}Iterations:${NC} $ITERATIONS"
echo ""

# Create results directory
mkdir -p "$TEST_RESULTS_DIR"

# Check if build exists
if [ ! -f "$BUILD_DIR/ck-cli-v6.5" ]; then
    echo -e "${YELLOW}Building C-Kernel-Engine with debug symbols...${NC}"
    cd "$PROJECT_DIR"
    make CK_DEBUG=1 ck-cli-v6.5
fi

# Check for perf
if ! command -v perf &> /dev/null; then
    echo -e "${RED}ERROR: 'perf' not found. Install with:${NC}"
    echo "  sudo apt-get install linux-tools-common linux-tools-$(uname -r)"
    exit 1
fi

# Clone FlameGraph if needed
if [ ! -d "$FLAMEGRAPH_DIR" ]; then
    echo -e "${YELLOW}Cloning FlameGraph...${NC}"
    git clone https://github.com/brendangregg/FlameGraph "$FLAMEGRAPH_DIR" 2>/dev/null || true
fi

# Key perf events for DRAM pressure (available on most systems)
# These measure actual memory traffic, not just cache activity
PERF_EVENTS="cycles,instructions,cache-references,cache-misses,LLC-loads,LLC-stores,L1-dcache-load-misses"

echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  STEP 1: BASELINE (Unfused Attention)${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""
echo "Measuring DRAM pressure for CURRENT (unfused) implementation..."
echo "This shows how much intermediate data is written to DRAM."
echo ""
echo "Perf events: $PERF_EVENTS"
echo ""

# Run baseline multiple times for averaging
for i in $(seq 1 $ITERATIONS); do
    echo -e "${YELLOW}Iteration $i/$ITERATIONS${NC}"

    perf stat -e "$PERF_EVENTS" \
        -o "$TEST_RESULTS_DIR/perf_baseline_run${i}.txt" \
        -- "$BUILD_DIR/ck-cli-v6.5" \
        --model "$MODEL" \
        --max-tokens "$MAX_TOKENS" \
        --prompt "The quick brown fox jumps over the lazy dog. " \
        2>&1 | tee "$TEST_RESULTS_DIR/baseline_output${i}.txt"
done

# Average baseline results
echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  STEP 2: MEGA-FUSED ATTENTION${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""
echo "Measuring DRAM pressure for MEGA-FUSED attention..."
echo "Expecting: 10-100x reduction in DRAM traffic!"
echo ""

# Run mega-fused multiple times
for i in $(seq 1 $ITERATIONS); do
    echo -e "${YELLOW}Iteration $i/$ITERATIONS${NC}"

    perf stat -e "$PERF_EVENTS" \
        -o "$TEST_RESULTS_DIR/perf_megafused_run${i}.txt" \
        -- "$BUILD_DIR/ck-cli-v6.5" \
        --model "$MODEL" \
        --max-tokens "$MAX_TOKENS" \
        --mega-fused \
        --prompt "The quick brown fox jumps over the lazy dog. " \
        2>&1 | tee "$TEST_RESULTS_DIR/megafused_output${i}.txt"
done

echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  STEP 3: GENERATING FLAMEGRAPH${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

# Record memory events for flamegraph (using available event)
echo "Recording memory access patterns for flamegraph..."
perf record -g -e cache-misses \
    -o "$TEST_RESULTS_DIR/memory_flamegraph.data" \
    -- "$BUILD_DIR/ck-cli-v6.5" \
    --model "$MODEL" \
    --max-tokens "$MAX_TOKENS" \
    --mega-fused \
    --prompt "Generate a detailed analysis of CPU architecture considerations for high-performance computing."

# Generate flamegraph
echo "Generating flamegraph..."
"$FLAMEGRAPH_DIR/stackcollapse-perf.pl" "$TEST_RESULTS_DIR/memory_flamegraph.data" | \
    "$FLAMEGRAPH_DIR/flamegraph.pl" \
    --countname="cache misses" \
    --title="Mega-Fused Attention: Cache Misses (Memory Access)" \
    > "$TEST_RESULTS_DIR/mega_fused_flamegraph.svg"

echo -e "${GREEN}Flamegraph written to: $TEST_RESULTS_DIR/mega_fused_flamegraph.svg${NC}"
echo ""
echo "Visual check:"
echo "  - Unfused: Large 'memory' section in flamegraph"
echo "  - Fused: Tiny 'memory' section (fusion working!)"

echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  STEP 4: RESULTS COMPARISON${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

# Parse and compare results
cat > "$TEST_RESULTS_DIR/compare_results.py" << PYEOF
import re
import json
import glob

def parse_perf_file(filepath):
    """Parse perf stat output."""
    results = {}
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    # Try to extract numeric value
                    value_str = parts[0].replace(',', '')
                    if '.' in value_str:
                        value = float(value_str)
                    else:
                        value = int(value_str)
                    # Find the metric name (last word that's not a unit)
                    metric = parts[-1]
                    if metric not in ['seconds', 'msec', 'usec', 'nsec', 'K/sec', 'M/sec']:
                        results[metric] = value
                except (ValueError, IndexError):
                    continue
    return results

def average_results(filepaths):
    """Average results from multiple runs."""
    all_results = [parse_perf_file(f) for f in filepaths]
    if not all_results:
        return {}

    avg = {}
    for key in all_results[0].keys():
        values = [r.get(key, 0) for r in all_results]
        avg[key] = sum(values) / len(values)
    return avg

# Parse baseline
baseline_files = sorted(glob.glob("$TEST_RESULTS_DIR/perf_baseline_run*.txt"))
baseline = average_results(baseline_files)

# Parse mega-fused
megafused_files = sorted(glob.glob("$TEST_RESULTS_DIR/perf_megafused_run*.txt"))
megafused = average_results(megafused_files)

print("=" * 70)
print("DRAM PRESSURE COMPARISON: BASELINE vs MEGA-FUSED")
print("=" * 70)
print()

# Key metrics for DRAM pressure
key_metrics = [
    ('cache-misses', 'Cache Misses', 'M'),
    ('LLC-loads', 'L3 Cache Loads', 'M'),
    ('L1-dcache-load-misses', 'L1 Cache Load Misses', 'M'),
    ('cycles', 'CPU Cycles', 'M'),
]

print(f"{'Metric':<35} {'Baseline':>15} {'Mega-Fused':>15} {'Reduction':>12}")
print("-" * 80)

results = {}
for metric, name, unit in key_metrics:
    b = baseline.get(metric, 0)
    m = megafused.get(metric, 0)

    if b > 0:
        reduction = (b - m) / b * 100
        if unit == 'M':
            b_val = b / 1e6
            m_val = m / 1e6
        else:
            b_val = b
            m_val = m

        print(f"{name:<35} {b_val:>12.2f} {m_val:>12.2f} {reduction:>8.1f}%")

        if reduction > 50:
            print(f"\033[0;32m  {'EXCELLENT! Fusion working!':<35}\033[0m")
        elif reduction > 0:
            print(f"\033[1;33m  {'Some improvement':<35}\033[0m")
        else:
            print(f"\033[0;31m  {'WARNING: No improvement!':<35}\033[0m")

        results[metric] = {'baseline': b, 'megafused': m, 'reduction': reduction}

print()
print("=" * 70)
print("THE CRITICAL METRIC: DRAM Traffic Reduction")
print("=" * 70)
print()

# Calculate actual DRAM traffic reduction
# Cache misses are the primary indicator of DRAM traffic
# Lower cache misses = less DRAM access = fusion working!

if 'cache-misses' in results:
    cm_reduction = results['cache-misses']['reduction']
    print(f"Cache Miss Reduction: {cm_reduction:.1f}%")
    print()
    print("Interpretation:")
    print("  - Cache misses are the primary indicator of memory traffic")
    print("  - Each cache miss may trigger a DRAM access (for L3 miss)")
    print("  - 10% reduction = ~10% less memory bandwidth")
    print("  - 90% reduction = ~90% less memory bandwidth (EXCELLENT)")
    print()
    if cm_reduction > 50:
        print("\033[0;32mMEGA-FUSION IS WORKING! DRAM traffic significantly reduced!\033[0m")
    elif cm_reduction > 0:
        print("\033[1;33mPartial improvement detected. Check implementation.\033[0m")
    else:
        print("\033[0;31mNo improvement. Check mega-fusion implementation.\033[0m")

# Save results
with open("$TEST_RESULTS_DIR/dram_comparison.json", "w") as f:
    json.dump(results, f, indent=2)

print()
print(f"Results saved to: $TEST_RESULTS_DIR/dram_comparison.json")
PYEOF

python3 "$TEST_RESULTS_DIR/compare_results.py"

echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  STEP 5: NEXT STEPS${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""
echo "1. View the flamegraph:"
echo "   firefox $TEST_RESULTS_DIR/mega_fused_flamegraph.svg"
echo ""
echo "2. Check detailed results:"
echo "   cat $TEST_RESULTS_DIR/dram_comparison.json"
echo ""
echo "3. Compare with unfused (generate baseline flamegraph similarly)"
echo ""
echo -e "${GREEN}If DRAM reduction is >50%, mega-fusion is working!${NC}"
echo ""
echo -e "${YELLOW}To improve further:${NC}"
echo "  - Ensure L1/L2 working set fits per-head data"
echo "  - Optimize tile sizes for cache hierarchy"
echo "  - Use AVX-512 for more register storage"
