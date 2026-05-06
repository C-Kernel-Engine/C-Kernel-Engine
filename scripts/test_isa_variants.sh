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

compiler() {
  if [[ -n "${CC:-}" ]]; then
    echo "${CC}"
  elif command -v icx >/dev/null 2>&1; then
    command -v icx
  elif command -v gcc >/dev/null 2>&1; then
    command -v gcc
  else
    command -v cc
  fi
}

compiler_supports_flags() {
  local flags="$1"
  local cc
  cc="$(compiler)"
  printf 'int main(void){return 0;}\n' | "${cc}" ${flags} -x c - -c -o /tmp/ck_isa_flag_test.o >/dev/null 2>&1
  local rc=$?
  rm -f /tmp/ck_isa_flag_test.o
  return "${rc}"
}

run_variant() {
  local name="$1"
  local flags="$2"
  local build_dir="build_${name}"

  echo "==> ISA variant: ${name} (AVX_FLAGS='${flags}')"
  if ! compiler_supports_flags "${flags}"; then
    echo "Skipping ${name}: compiler '$(compiler)' does not accept these flags"
    return 0
  fi
  make -C "${ROOT_DIR}" -B BUILD_DIR="${build_dir}" AVX_FLAGS="${flags}" test-gemm-avx-bench-quick
}

CPU_FLAGS="$(cpu_flags)"
echo "CPU flags:${CPU_FLAGS}"

# Test the slowest x86 SIMD floor used by the quantized kernels.
if have_flag "sse4_1" && have_flag "ssse3"; then
  run_variant "sse41" "-msse4.1 -mssse3"
else
  echo "Skipping SSE4.1/SSSE3: CPU does not report sse4_1 and ssse3"
fi

if have_flag "avx"; then
  run_variant "avx" "-mavx"
else
  echo "Skipping AVX: CPU does not report avx"
fi

if have_flag "avx2"; then
  run_variant "avx2" "-mavx2 -mfma"
else
  echo "Skipping AVX2: CPU does not report avx2"
fi

# AVX-512 if supported by CPU.
if have_flag "avx512f"; then
  run_variant "avx512" "-mavx512f -mavx512bw -mavx512dq -mavx512vl -mfma"
else
  echo "Skipping AVX-512: CPU does not report avx512f"
fi

if have_flag "avx512f" && { have_flag "avx512_vnni" || have_flag "avx512vnni"; }; then
  run_variant "avx512_vnni" "-mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512vnni -mfma"
else
  echo "Skipping AVX-512 VNNI: CPU does not report avx512_vnni"
fi

if have_flag "avx512_bf16"; then
  run_variant "avx512_bf16" "-mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512bf16 -mfma"
else
  echo "Skipping AVX-512 BF16: CPU does not report avx512_bf16"
fi

if have_flag "amx_tile" && have_flag "amx_int8"; then
  run_variant "amx_int8" "-mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512vnni -mamx-tile -mamx-int8 -mfma"
else
  echo "Skipping AMX INT8: CPU does not report amx_tile and amx_int8"
fi

echo "ISA variant tests complete."
