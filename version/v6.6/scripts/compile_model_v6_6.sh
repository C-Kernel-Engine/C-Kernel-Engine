#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <model_dir> [model_c_path]" >&2
  exit 1
fi

MODEL_DIR=$(cd "$1" && pwd)
MODEL_C=${2:-"$MODEL_DIR/model_v6_6.c"}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
BUILD_DIR="$ROOT/build"
V66_ROOT="$ROOT/version/v6.6"

if [ ! -f "$MODEL_C" ]; then
  echo "Error: model C file not found: $MODEL_C" >&2
  exit 1
fi

if [ ! -f "$BUILD_DIR/libckernel_engine.so" ]; then
  echo "Error: missing $BUILD_DIR/libckernel_engine.so (run make)" >&2
  exit 1
fi
if [ ! -f "$BUILD_DIR/libckernel_tokenizer.so" ]; then
  echo "Error: missing $BUILD_DIR/libckernel_tokenizer.so (run make tokenizer)" >&2
  exit 1
fi

COMPILER=gcc
OMP_FLAG="-fopenmp"
if command -v icx >/dev/null 2>&1; then
  COMPILER=icx
  OMP_FLAG="-qopenmp"
fi

OUT_SO="$MODEL_DIR/libmodel.so"

set -x
"$COMPILER" -shared -fPIC \
  -mcmodel=large -O3 -march=native -std=c11 -fvisibility=default \
  "$OMP_FLAG" \
  -I"$ROOT/include" \
  -I"$V66_ROOT/include" \
  -I"$V66_ROOT/src" \
  -o "$OUT_SO" \
  "$MODEL_C" \
  "$V66_ROOT/src/ckernel_model_load_v6_6.c" \
  "$V66_ROOT/src/ck_parallel_decode.c" \
  "$V66_ROOT/src/ck_parallel_prefill.c" \
  -L"$BUILD_DIR" -L"$MODEL_DIR" \
  -lckernel_tokenizer -lckernel_engine -lm \
  -Wl,-rpath,'$ORIGIN' -Wl,-rpath,"$BUILD_DIR"
set +x

ln -sf libmodel.so "$MODEL_DIR/ck-kernel-inference.so"

echo "Built: $OUT_SO"
