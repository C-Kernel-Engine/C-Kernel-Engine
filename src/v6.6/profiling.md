  1. Perf (requires sudo for full access)

  # Set up environment first
  source /opt/intel/oneapi/setvars.sh

  # Record profile (sudo with LD_LIBRARY_PATH preserved)
  sudo env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
      perf record -g --call-graph dwarf -o ck-perf.data \
      ./build/ck-cli-v6.6 \
      ~/.cache/ck-engine-v6.6/models/Qwen--Qwen2-0.5B-Instruct-GGUF/ck-kernel-inference.so \
      ~/.cache/ck-engine-v6.6/models/Qwen--Qwen2-0.5B-Instruct-GGUF/weights.bump \
      -p "Your prompt here" --max-tokens 50

  # View top functions
  sudo perf report -f -i ck-perf.data --stdio --no-children | head -100

  # View with call graph
  sudo perf report -f -i ck-perf.data --stdio | head -150

  2. Perf without sudo (limited but works for user-space)

  # Lower paranoid level (one-time, persists across reboots)
  echo 1 | sudo tee /proc/sys/kernel/perf_event_paranoid

  # Then run without sudo
  perf record -g ./build/ck-cli-v6.6 ...
  perf report -i perf.data --stdio

  3. VTune (no sudo needed)

  # Set up Intel oneAPI
  source /opt/intel/oneapi/setvars.sh

  # Hotspots analysis
  vtune -collect hotspots -r vtune_ck_hotspots -- \
      ./build/ck-cli-v6.6 \
      ~/.cache/ck-engine-v6.6/models/Qwen--Qwen2-0.5B-Instruct-GGUF/ck-kernel-inference.so \
      ~/.cache/ck-engine-v6.6/models/Qwen--Qwen2-0.5B-Instruct-GGUF/weights.bump \
      -p "Your prompt here" --max-tokens 50

  # View text report
  vtune -report summary -r vtune_ck_hotspots
  vtune -report hotspots -r vtune_ck_hotspots

  # Microarchitecture analysis (cache misses, branch mispredicts)
  vtune -collect uarch-exploration -r vtune_ck_uarch -- ./build/ck-cli-v6.6 ...

  # Memory access analysis
  vtune -collect memory-access -r vtune_ck_memory -- ./build/ck-cli-v6.6 ...

  4. Flamegraph (from perf data)

  # Generate flamegraph (after perf record)
  sudo perf script -i ck-perf.data | \
      ./FlameGraph/stackcollapse-perf.pl | \
      ./FlameGraph/flamegraph.pl > ck-flamegraph.svg

  # Open in browser
  firefox ck-flamegraph.svg

  5. Quick timing (no profiler)

  # Just timing breakdown (built into CLI)
  ./build/ck-cli-v6.6 \
      ~/.cache/ck-engine-v6.6/models/Qwen--Qwen2-0.5B-Instruct-GGUF/ck-kernel-inference.so \
      ~/.cache/ck-engine-v6.6/models/Qwen--Qwen2-0.5B-Instruct-GGUF/weights.bump \
      -p "Hello" --max-tokens 20 --timing

  Model shortcuts

  # Using --model flag (auto-discovers from cache)
  ./build/ck-cli-v6.6 --model Qwen -p "Hello" --max-tokens 20

  # List available models
  ./build/ck-cli-v6.6 --list

  ---
  Now, let's see your current profile:

  sudo perf report -f -i ck-perf-cli.data --stdio --no-children | head -100

