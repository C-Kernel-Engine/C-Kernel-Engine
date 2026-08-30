#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

APT_PACKAGES=(
  build-essential
  cmake
  git
  nodejs
  npm
  python3
  python3-numpy
  python3-venv
  python3-pip
)

need_cmd() {
  command -v "$1" >/dev/null 2>&1
}

have_apt=0
if need_cmd apt-get; then
  have_apt=1
fi

echo "== CK-Engine developer environment setup =="

if [ "$have_apt" -eq 1 ]; then
  missing=()
  for pkg in "${APT_PACKAGES[@]}"; do
    if ! dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null | grep -q "install ok installed"; then
      missing+=("$pkg")
    fi
  done

  if [ "${#missing[@]}" -gt 0 ]; then
    echo "Installing system packages: ${missing[*]}"
    sudo apt-get update
    sudo apt-get install -y "${missing[@]}"
  else
    echo "System packages: OK"
  fi
else
  echo "apt-get not found; skipping system package installation"
fi

if ! need_cmd node; then
  echo "ERROR: node not found. Install Node.js before running pre-push visualizer checks." >&2
  exit 1
fi
if ! need_cmd npm; then
  echo "ERROR: npm not found. Install npm before running pre-push visualizer checks." >&2
  exit 1
fi

python3 - <<'PY'
import numpy
print(f"system python numpy: {numpy.__version__}")
PY

if [ ! -d .venv ]; then
  echo "Creating .venv"
  python3 -m venv .venv
fi

echo "Installing Python requirements into .venv"
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
if [ -f version/v8/requirements.txt ]; then
  .venv/bin/python -m pip install -r version/v8/requirements.txt
fi
if [ -f version/v7/requirements.txt ]; then
  .venv/bin/python -m pip install -r version/v7/requirements.txt
fi

./scripts/setup-hooks.sh

echo "Checking v7 training manifest"
if .venv/bin/python version/v7/scripts/resolve_train_manifest_v7.py >/tmp/ck_v7_train_manifest_path.txt 2>/dev/null; then
  v7_manifest="$(cat /tmp/ck_v7_train_manifest_path.txt)"
  echo "  v7 train manifest: $v7_manifest"
else
  echo "  Initializing v7 tiny training manifest"
  .venv/bin/python version/v7/scripts/ck_run_v7.py init \
    --run-name tiny_init \
    --train-seed 42 \
    --layers 2 \
    --vocab-size 256 \
    --embed-dim 128 \
    --hidden-dim 256 \
    --context-len 128
  v7_manifest="$(.venv/bin/python version/v7/scripts/resolve_train_manifest_v7.py)"
  echo "  v7 train manifest: $v7_manifest"
fi

echo
echo "Dependency check:"
echo "  node: $(node --version)"
echo "  npm:  $(npm --version)"
echo "  python3: $(python3 --version)"
echo "  .venv python: $(.venv/bin/python --version)"
echo "  v7 train manifest: $v7_manifest"

echo
echo "Setup complete."
