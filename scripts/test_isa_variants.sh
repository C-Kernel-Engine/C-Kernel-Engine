#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cpu_flags() {
  if command -v lscpu >/dev/null 2>&1; then
    lscpu | awk -F: '/Flags/ {print $2}'
  elif [[ -f /proc/cpuinfo ]]; then
    awk -F: '/flags/ {print $2; exit}' /proc/cpuinfo
  else
    echo ""
  fi
}

have_flag() {
  local flag="$1"
  echo " $CPU_FLAGS " | grep -q " ${flag} "
}

run_variant() {
  local name="$1"
  local flags="$2"
  local build_dir="build_${name}"

  echo "==> ISA variant: ${name} (AVX_FLAGS='${flags}')"
  make -C "${ROOT_DIR}" -B BUILD_DIR="${build_dir}" AVX_FLAGS="${flags}" test-gemm-avx-bench-quick
}

CPU_FLAGS="$(cpu_flags)"
echo "CPU flags:${CPU_FLAGS}"

# Always test AVX and AVX2 variants (works on any AVX2-capable CPU).
run_variant "avx" "-mavx"
run_variant "avx2" "-mavx2"

# AVX-512 if supported by CPU.
if have_flag "avx512f"; then
  run_variant "avx512" "-mavx512f -mavx512bw -mavx512dq"
else
  echo "Skipping AVX-512: CPU does not report avx512f"
fi

echo "ISA variant tests complete."
