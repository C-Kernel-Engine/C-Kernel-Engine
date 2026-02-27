#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
HTML_PATH="${SCRIPT_DIR}/ir_visualizer.html"

if [[ ! -d "${SRC_DIR}" ]]; then
  echo "Missing src dir: ${SRC_DIR}" >&2
  exit 1
fi
if [[ ! -f "${HTML_PATH}" ]]; then
  echo "Missing HTML file: ${HTML_PATH}" >&2
  exit 1
fi

FILES=(
  "utils.js"
  "training_dashboard.js"
  "training_tabs.js"
  "gradient_flow.js"
  "weight_activation.js"
  "attention_inspector.js"
  "run_compare.js"
  "training_memory_diag.js"
  "question_headers.js"
  "mode_ui.js"
  "memory_explorer.js"
  "live_mode.js"
  "main.js"
)

tmp_bundle="$(mktemp)"
tmp_html="$(mktemp)"

{
  echo "/* GENERATED FILE: do not edit by hand."
  echo " * Source modules in version/v7/tools/src/"
  echo " * Built by build_ir_visualizer_bundle.sh"
  echo " */"
  echo "(() => {"
  for rel in "${FILES[@]}"; do
    path="${SRC_DIR}/${rel}"
    if [[ ! -f "${path}" ]]; then
      echo "Missing module: ${path}" >&2
      rm -f "${tmp_bundle}" "${tmp_html}"
      exit 1
    fi
    echo
    echo "/* ===== ${rel} ===== */"
    # Dependency-free bundling:
    # 1) drop static imports (manual file ordering handles deps)
    # 2) strip leading `export` for declarations
    sed -E \
      -e '/^[[:space:]]*import[[:space:]].*;[[:space:]]*$/d' \
      -e 's/^[[:space:]]*export[[:space:]]+//' \
      "${path}"
  done
  echo
  echo "})();"
} > "${tmp_bundle}"

if ! grep -q 'CK_V7_MODULE_BUNDLE_START' "${HTML_PATH}"; then
  echo "Bundle markers not found in ${HTML_PATH}" >&2
  rm -f "${tmp_bundle}" "${tmp_html}"
  exit 1
fi

awk -v bundle_path="${tmp_bundle}" '
  BEGIN {
    start="<!-- CK_V7_MODULE_BUNDLE_START -->";
    end="<!-- CK_V7_MODULE_BUNDLE_END -->";
    replacing=0;
  }
  {
    if (index($0, start) > 0) {
      print $0;
      print "<script type=\"module\">";
      while ((getline line < bundle_path) > 0) print line;
      close(bundle_path);
      print "</script>";
      replacing=1;
      next;
    }
    if (index($0, end) > 0) {
      replacing=0;
      print $0;
      next;
    }
    if (!replacing) print $0;
  }
' "${HTML_PATH}" > "${tmp_html}"

mv "${tmp_html}" "${HTML_PATH}"
rm -f "${tmp_bundle}"

echo "Bundled modules into: ${HTML_PATH}"
