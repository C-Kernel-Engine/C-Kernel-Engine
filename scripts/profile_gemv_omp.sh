#!/bin/bash
#
# profile_gemv_omp.sh - Comprehensive profiling of serial vs OpenMP GEMV kernels
#
# Runs perf stat, flamegraph, cachegrind, VTune, and massif comparisons
# between the serial and OMP GEMV kernel variants.
#
# Usage:
#   ./scripts/profile_gemv_omp.sh              # All profiling tools
#   ./scripts/profile_gemv_omp.sh perf         # perf stat only
#   ./scripts/profile_gemv_omp.sh flamegraph   # flamegraph only
#   ./scripts/profile_gemv_omp.sh cachegrind   # cachegrind only
#   ./scripts/profile_gemv_omp.sh vtune        # VTune only
#   ./scripts/profile_gemv_omp.sh massif       # heap profiling only
#   ./scripts/profile_gemv_omp.sh all          # everything

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$ROOT_DIR/build"
RESULTS_DIR="$ROOT_DIR/profile_results/gemv_omp"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
RESULT_SUBDIR="$RESULTS_DIR/$TIMESTAMP"
FLAMEGRAPH_DIR="${FLAMEGRAPH_DIR:-$ROOT_DIR/FlameGraph}"
BENCH_BIN="$BUILD_DIR/test_gemv_omp_parity"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

log_step()  { echo -e "${GREEN}[STEP]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_info()  { echo -e "${CYAN}[INFO]${NC} $1"; }

# Parse arguments
MODE="${1:-all}"

echo "======================================================================"
echo "  GEMV Serial vs OpenMP Profiling Suite"
echo "======================================================================"
echo ""
echo "  Mode:       $MODE"
echo "  Output:     $RESULT_SUBDIR"
echo "  Threads:    $(nproc)"
echo ""

mkdir -p "$RESULT_SUBDIR"

# ============================================================================
# Step 0: Build the benchmark binary with frame pointers for profiling
# ============================================================================

build_benchmark() {
    log_step "Building GEMV OMP benchmark with frame pointers..."

    # Rebuild library with frame pointers for accurate profiling
    make -C "$ROOT_DIR" -B "$BUILD_DIR/libckernel_engine.so" \
        CFLAGS="-O3 -fno-omit-frame-pointer -g -fPIC -fopenmp -Wall \
        $(pkg-config --cflags 2>/dev/null || echo '-mavx -march=native') \
        -Iinclude" 2>&1 | tail -5

    # Build the benchmark binary
    local CC="${CC:-gcc}"
    $CC -O3 -fno-omit-frame-pointer -g -march=native -fopenmp -Iinclude \
        "$ROOT_DIR/tests/test_gemv_omp_parity.c" \
        -L"$BUILD_DIR" -lckernel_engine -lm \
        -Wl,-rpath,"$BUILD_DIR" \
        -o "$BENCH_BIN"

    if [ ! -f "$BENCH_BIN" ]; then
        log_error "Failed to build benchmark binary"
        exit 1
    fi
    log_step "Benchmark binary built: $BENCH_BIN"
}

# ============================================================================
# Step 1: perf stat — hardware counters comparison
# ============================================================================

check_perf_permissions() {
    if ! command -v perf &> /dev/null; then
        log_warn "perf not installed"
        return 1
    fi
    # Test if perf can actually record (paranoid setting may block it)
    local paranoid=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo "unknown")
    if [ "$paranoid" != "unknown" ] && [ "$paranoid" -ge 2 ]; then
        log_warn "perf_event_paranoid=$paranoid (too restrictive for perf record/stat)"
        log_warn "Fix with: sudo sysctl kernel.perf_event_paranoid=1"
        log_warn "Or run:   sudo sh -c 'echo 1 > /proc/sys/kernel/perf_event_paranoid'"
        return 1
    fi
    return 0
}

run_perf_stat() {
    if ! check_perf_permissions; then
        log_warn "Skipping perf stat (no permissions). Falling back to timing-only benchmark."
        log_info "Running benchmark directly for timing data..."
        LD_LIBRARY_PATH="$BUILD_DIR:$LD_LIBRARY_PATH" "$BENCH_BIN" --quick \
            > "$RESULT_SUBDIR/benchmark_timing.txt" 2>&1
        cat "$RESULT_SUBDIR/benchmark_timing.txt"
        return
    fi

    log_step "Running perf stat: serial (OMP_NUM_THREADS=1) vs parallel..."

    local PERF_EVENTS="cycles,instructions,cache-references,cache-misses,\
LLC-loads,LLC-load-misses,LLC-stores,LLC-store-misses,\
branches,branch-misses,stalled-cycles-frontend,stalled-cycles-backend"

    # Serial (1 thread)
    log_info "perf stat with OMP_NUM_THREADS=1..."
    OMP_NUM_THREADS=1 LD_LIBRARY_PATH="$BUILD_DIR:$LD_LIBRARY_PATH" \
        perf stat -e "$PERF_EVENTS" -o "$RESULT_SUBDIR/perf_stat_serial.txt" \
        "$BENCH_BIN" --quick 2>&1 | tail -5

    # Parallel (all threads)
    log_info "perf stat with OMP_NUM_THREADS=$(nproc)..."
    OMP_NUM_THREADS=$(nproc) LD_LIBRARY_PATH="$BUILD_DIR:$LD_LIBRARY_PATH" \
        perf stat -e "$PERF_EVENTS" -o "$RESULT_SUBDIR/perf_stat_parallel.txt" \
        "$BENCH_BIN" --quick 2>&1 | tail -5

    echo ""
    echo "=== Serial (1 thread) perf stat ==="
    cat "$RESULT_SUBDIR/perf_stat_serial.txt"
    echo ""
    echo "=== Parallel ($(nproc) threads) perf stat ==="
    cat "$RESULT_SUBDIR/perf_stat_parallel.txt"
    echo ""

    log_step "perf stat results saved to $RESULT_SUBDIR/perf_stat_*.txt"
}

# ============================================================================
# Step 2: Flamegraph — visual hotspot comparison
# ============================================================================

run_flamegraph() {
    if ! check_perf_permissions; then
        log_warn "Skipping flamegraph (perf not available or no permissions)"
        return
    fi

    # Auto-clone FlameGraph if missing
    if [ ! -d "$FLAMEGRAPH_DIR" ]; then
        log_step "Cloning FlameGraph tools..."
        git clone --depth 1 https://github.com/brendangregg/FlameGraph.git "$FLAMEGRAPH_DIR"
    fi

    if [ ! -f "$FLAMEGRAPH_DIR/flamegraph.pl" ]; then
        log_warn "FlameGraph tools not found at $FLAMEGRAPH_DIR, skipping"
        return
    fi

    log_step "Generating flamegraphs..."

    # Serial flamegraph
    log_info "Recording serial (OMP_NUM_THREADS=1)..."
    if OMP_NUM_THREADS=1 LD_LIBRARY_PATH="$BUILD_DIR:$LD_LIBRARY_PATH" \
        perf record -g -F 997 -o "$RESULT_SUBDIR/perf_serial.data" \
        "$BENCH_BIN" --quick 2>"$RESULT_SUBDIR/perf_record_serial.log"; then

        perf script -i "$RESULT_SUBDIR/perf_serial.data" | \
            "$FLAMEGRAPH_DIR/stackcollapse-perf.pl" | \
            "$FLAMEGRAPH_DIR/flamegraph.pl" --title="GEMV Serial (1 thread)" \
            > "$RESULT_SUBDIR/flamegraph_serial.svg"
    else
        log_warn "perf record failed for serial (see $RESULT_SUBDIR/perf_record_serial.log)"
    fi

    # Parallel flamegraph
    log_info "Recording parallel (OMP_NUM_THREADS=$(nproc))..."
    if OMP_NUM_THREADS=$(nproc) LD_LIBRARY_PATH="$BUILD_DIR:$LD_LIBRARY_PATH" \
        perf record -g -F 997 -o "$RESULT_SUBDIR/perf_parallel.data" \
        "$BENCH_BIN" --quick 2>"$RESULT_SUBDIR/perf_record_parallel.log"; then

        perf script -i "$RESULT_SUBDIR/perf_parallel.data" | \
            "$FLAMEGRAPH_DIR/stackcollapse-perf.pl" | \
            "$FLAMEGRAPH_DIR/flamegraph.pl" --title="GEMV OpenMP Parallel ($(nproc) threads)" \
            > "$RESULT_SUBDIR/flamegraph_parallel.svg"
    else
        log_warn "perf record failed for parallel (see $RESULT_SUBDIR/perf_record_parallel.log)"
    fi

    if [ -f "$RESULT_SUBDIR/flamegraph_serial.svg" ] || [ -f "$RESULT_SUBDIR/flamegraph_parallel.svg" ]; then
        log_step "Flamegraphs saved:"
        [ -f "$RESULT_SUBDIR/flamegraph_serial.svg" ] && log_info "  Serial:   $RESULT_SUBDIR/flamegraph_serial.svg"
        [ -f "$RESULT_SUBDIR/flamegraph_parallel.svg" ] && log_info "  Parallel: $RESULT_SUBDIR/flamegraph_parallel.svg"
    fi
}

# ============================================================================
# Step 3: Cachegrind — cache behavior comparison
# ============================================================================

run_cachegrind() {
    if ! command -v valgrind &> /dev/null; then
        log_warn "valgrind not found, skipping cachegrind"
        return
    fi

    log_step "Running cachegrind analysis..."

    # Serial
    log_info "Cachegrind with OMP_NUM_THREADS=1..."
    OMP_NUM_THREADS=1 LD_LIBRARY_PATH="$BUILD_DIR:$LD_LIBRARY_PATH" \
        valgrind --tool=cachegrind \
        --cachegrind-out-file="$RESULT_SUBDIR/cachegrind_serial.out" \
        "$BENCH_BIN" --quick 2>&1 | tail -15 > "$RESULT_SUBDIR/cachegrind_serial_summary.txt"

    # Parallel
    log_info "Cachegrind with OMP_NUM_THREADS=$(nproc)..."
    OMP_NUM_THREADS=$(nproc) LD_LIBRARY_PATH="$BUILD_DIR:$LD_LIBRARY_PATH" \
        valgrind --tool=cachegrind \
        --cachegrind-out-file="$RESULT_SUBDIR/cachegrind_parallel.out" \
        "$BENCH_BIN" --quick 2>&1 | tail -15 > "$RESULT_SUBDIR/cachegrind_parallel_summary.txt"

    # Annotate
    cg_annotate "$RESULT_SUBDIR/cachegrind_serial.out" \
        > "$RESULT_SUBDIR/cachegrind_serial_annotated.txt" 2>/dev/null || true
    cg_annotate "$RESULT_SUBDIR/cachegrind_parallel.out" \
        > "$RESULT_SUBDIR/cachegrind_parallel_annotated.txt" 2>/dev/null || true

    echo ""
    echo "=== Serial Cachegrind Summary ==="
    cat "$RESULT_SUBDIR/cachegrind_serial_summary.txt"
    echo ""
    echo "=== Parallel Cachegrind Summary ==="
    cat "$RESULT_SUBDIR/cachegrind_parallel_summary.txt"
    echo ""

    log_step "Cachegrind results saved to $RESULT_SUBDIR/cachegrind_*.txt"
}

# ============================================================================
# Step 4: VTune — Intel performance analysis
# ============================================================================

run_vtune() {
    if ! command -v vtune &> /dev/null; then
        log_warn "VTune not found. To install: source /opt/intel/oneapi/setvars.sh"
        log_warn "Skipping VTune analysis"
        return
    fi

    log_step "Running VTune hotspots analysis..."

    # Serial
    log_info "VTune hotspots (serial, OMP_NUM_THREADS=1)..."
    OMP_NUM_THREADS=1 LD_LIBRARY_PATH="$BUILD_DIR:$LD_LIBRARY_PATH" \
        vtune -collect hotspots \
        -result-dir "$RESULT_SUBDIR/vtune_serial_hotspots" \
        -quiet \
        -- "$BENCH_BIN" --quick 2>/dev/null || log_warn "VTune serial hotspots failed"

    # Parallel
    log_info "VTune hotspots (parallel, OMP_NUM_THREADS=$(nproc))..."
    OMP_NUM_THREADS=$(nproc) LD_LIBRARY_PATH="$BUILD_DIR:$LD_LIBRARY_PATH" \
        vtune -collect hotspots \
        -result-dir "$RESULT_SUBDIR/vtune_parallel_hotspots" \
        -quiet \
        -- "$BENCH_BIN" --quick 2>/dev/null || log_warn "VTune parallel hotspots failed"

    # Threading analysis (parallel only)
    log_info "VTune threading analysis..."
    OMP_NUM_THREADS=$(nproc) LD_LIBRARY_PATH="$BUILD_DIR:$LD_LIBRARY_PATH" \
        vtune -collect threading \
        -result-dir "$RESULT_SUBDIR/vtune_threading" \
        -quiet \
        -- "$BENCH_BIN" --quick 2>/dev/null || log_warn "VTune threading failed"

    # Memory access analysis
    log_info "VTune memory-access analysis..."
    OMP_NUM_THREADS=$(nproc) LD_LIBRARY_PATH="$BUILD_DIR:$LD_LIBRARY_PATH" \
        vtune -collect memory-access \
        -result-dir "$RESULT_SUBDIR/vtune_memory" \
        -quiet \
        -- "$BENCH_BIN" --quick 2>/dev/null || log_warn "VTune memory-access failed"

    # Generate text reports
    for dir in "$RESULT_SUBDIR"/vtune_*; do
        if [ -d "$dir" ]; then
            local name=$(basename "$dir")
            vtune -report summary -result-dir "$dir" \
                -format text -report-output "$RESULT_SUBDIR/${name}_report.txt" 2>/dev/null || true
        fi
    done

    log_step "VTune results saved to $RESULT_SUBDIR/vtune_*"
    log_info "Open with: vtune-gui $RESULT_SUBDIR/vtune_parallel_hotspots"
}

# ============================================================================
# Step 5: Massif — heap profiling
# ============================================================================

run_massif() {
    if ! command -v valgrind &> /dev/null; then
        log_warn "valgrind not found, skipping massif"
        return
    fi

    log_step "Running Valgrind massif (heap profiling)..."

    # Serial
    log_info "Massif with OMP_NUM_THREADS=1..."
    OMP_NUM_THREADS=1 LD_LIBRARY_PATH="$BUILD_DIR:$LD_LIBRARY_PATH" \
        valgrind --tool=massif --pages-as-heap=yes \
        --massif-out-file="$RESULT_SUBDIR/massif_serial.out" \
        "$BENCH_BIN" --quick 2>/dev/null

    # Parallel
    log_info "Massif with OMP_NUM_THREADS=$(nproc)..."
    OMP_NUM_THREADS=$(nproc) LD_LIBRARY_PATH="$BUILD_DIR:$LD_LIBRARY_PATH" \
        valgrind --tool=massif --pages-as-heap=yes \
        --massif-out-file="$RESULT_SUBDIR/massif_parallel.out" \
        "$BENCH_BIN" --quick 2>/dev/null

    # Generate text summaries
    ms_print "$RESULT_SUBDIR/massif_serial.out" > "$RESULT_SUBDIR/massif_serial_report.txt" 2>/dev/null || true
    ms_print "$RESULT_SUBDIR/massif_parallel.out" > "$RESULT_SUBDIR/massif_parallel_report.txt" 2>/dev/null || true

    log_step "Massif results saved:"
    log_info "  Serial:   ms_print $RESULT_SUBDIR/massif_serial.out"
    log_info "  Parallel: ms_print $RESULT_SUBDIR/massif_parallel.out"
}

# ============================================================================
# Dispatch
# ============================================================================

build_benchmark

case "$MODE" in
    perf|perf-stat)
        run_perf_stat
        ;;
    flamegraph|flame)
        run_flamegraph
        ;;
    cachegrind|cache)
        run_cachegrind
        ;;
    vtune)
        run_vtune
        ;;
    massif|heap|memgraph)
        run_massif
        ;;
    all|full)
        run_perf_stat
        run_flamegraph
        run_cachegrind
        run_vtune
        run_massif
        ;;
    *)
        echo "Usage: $0 [perf|flamegraph|cachegrind|vtune|massif|all]"
        echo ""
        echo "  perf        Hardware counters (cycles, cache misses, LLC)"
        echo "  flamegraph  SVG flamegraphs (serial vs parallel)"
        echo "  cachegrind  Cache behavior analysis"
        echo "  vtune       Intel VTune (hotspots, threading, memory)"
        echo "  massif      Heap memory profiling"
        echo "  all         Run all profiling tools"
        exit 1
        ;;
esac

echo ""
echo "======================================================================"
echo "  Profiling Complete"
echo "======================================================================"
echo ""
echo "  Results in: $RESULT_SUBDIR"
echo ""
echo "  Quick view:"
echo "    perf stat:   cat $RESULT_SUBDIR/perf_stat_*.txt"
echo "    flamegraph:  open $RESULT_SUBDIR/flamegraph_*.svg"
echo "    cachegrind:  cat $RESULT_SUBDIR/cachegrind_*_summary.txt"
echo "    vtune:       vtune-gui $RESULT_SUBDIR/vtune_*"
echo "    massif:      ms_print $RESULT_SUBDIR/massif_*.out"
echo ""
