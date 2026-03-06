#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash version/v7/scripts/init_data_workspace_v7.sh [options]

Create a new v7 data workspace scaffold for a dataset family/spec iteration.

Examples:
  bash version/v7/scripts/init_data_workspace_v7.sh --spec spec03 --dataset-type svg
  bash version/v7/scripts/init_data_workspace_v7.sh --workspace version/v7/data/c_spec01 --dataset-type c
  bash version/v7/scripts/init_data_workspace_v7.sh --spec spec04 --dataset-type svg --force

Options:
  --spec NAME           Workspace name under version/v7/data (e.g. spec03)
  --workspace PATH      Explicit workspace path (overrides --spec)
  --dataset-type TYPE   Dataset type: svg, c, python, sql, json, bash, html, css, js
  --goal TEXT           Short end-goal description
  --force               Overwrite stub files if they already exist
  -h, --help            Show this help

Notes:
  - The folder layout is generic.
  - Contract and eval stub content are dataset-type aware.
  - This script does not import data; it only creates the workspace scaffold.
EOF
}

spec=""
workspace=""
dataset_type="svg"
goal=""
force=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --spec)
      spec="${2:-}"
      shift 2
      ;;
    --workspace)
      workspace="${2:-}"
      shift 2
      ;;
    --dataset-type)
      dataset_type="${2:-}"
      shift 2
      ;;
    --goal)
      goal="${2:-}"
      shift 2
      ;;
    --force)
      force=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$workspace" && -z "$spec" ]]; then
  echo "ERROR: pass --spec or --workspace" >&2
  exit 1
fi

case "$dataset_type" in
  svg|c|python|sql|json|bash|html|css|js)
    ;;
  *)
    echo "ERROR: unsupported --dataset-type: $dataset_type" >&2
    exit 1
    ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

if [[ -z "$workspace" ]]; then
  workspace="$ROOT/version/v7/data/$spec"
else
  case "$workspace" in
    /*) ;;
    *) workspace="$ROOT/$workspace" ;;
  esac
fi

workspace="$(cd "$(dirname "$workspace")" && pwd)/$(basename "$workspace")"
workspace_name="$(basename "$workspace")"

if [[ -z "$goal" ]]; then
  case "$dataset_type" in
    svg) goal="Render structured, valid SVG that follows a closed control contract." ;;
    c) goal="Generate compilable C code that follows the repo's kernel and API conventions." ;;
    python) goal="Generate runnable Python that follows task and API constraints." ;;
    sql) goal="Generate executable SQL that matches schema and query intent." ;;
    json) goal="Generate schema-valid JSON with stable field coverage." ;;
    bash) goal="Generate runnable shell commands/scripts that satisfy task constraints." ;;
    html) goal="Generate valid HTML that follows layout/content constraints." ;;
    css) goal="Generate valid CSS that follows design/token constraints." ;;
    js) goal="Generate runnable JavaScript that follows task and API constraints." ;;
    *) goal="Generate outputs that satisfy the dataset contract." ;;
  esac
fi

write_file() {
  local path="$1"
  local content="$2"
  if [[ -f "$path" && "$force" -ne 1 ]]; then
    return 0
  fi
  printf '%s' "$content" > "$path"
}

mkdir -p \
  "$workspace/contracts" \
  "$workspace/raw_assets" \
  "$workspace/normalized" \
  "$workspace/pretrain" \
  "$workspace/midtrain" \
  "$workspace/sft" \
  "$workspace/holdout" \
  "$workspace/tokenizer" \
  "$workspace/manifests"

readme_content=$(cat <<EOF
# $workspace_name

\`$workspace_name\` is a v7 data workspace for the \`$dataset_type\` dataset family.

## Goal

$goal

## Layout

- \`contracts/\`
- \`raw_assets/\`
- \`normalized/\`
- \`pretrain/\`
- \`midtrain/\`
- \`sft/\`
- \`holdout/\`
- \`tokenizer/\`
- \`manifests/\`

## Workflow

1. import and inventory source data into \`raw_assets/\`
2. normalize and placeholderize into \`normalized/\`
3. derive \`pretrain/\`, \`midtrain/\`, and \`sft/\`
4. build tokenizer corpus in \`tokenizer/\`
5. keep fixed evaluation assets/prompts in \`holdout/\`
6. store inventory, dedupe, coverage, and fit reports in \`manifests/\`

## Contract rule

Do not mix incompatible row formats inside one workspace.

Keep:
- one canonical input/output contract
- one tokenizer corpus policy
- one eval contract

Split experimental format changes into a new workspace instead of mutating history in place.
EOF
)

contract_content=$(cat <<EOF
# $workspace_name Contract

Dataset type: \`$dataset_type\`

## Goal

$goal

## Hard rules

- Define one canonical row format for this workspace.
- Do not mix legacy and new conditioning styles.
- Keep eval probes aligned with the same contract used in training.
- Track normalization, dedupe, and holdout policy in \`manifests/\`.

## TODO

- write the canonical row format
- define placeholder policy
- define dedupe policy
- define holdout/canary policy
- define promotion gates
EOF
)

eval_stub=$(cat <<EOF
{
  "schema": "ck.eval_contract.v1",
  "dataset_type": "$dataset_type",
  "goal": "$goal",
  "notes": [
    "Fill in probes and metric thresholds for $workspace_name.",
    "Keep this contract aligned with the workspace training format."
  ],
  "stage_metrics": [],
  "headline_metrics": [],
  "probes": []
}
EOF
)

write_file "$workspace/README.md" "$readme_content"
write_file "$workspace/contracts/WORKSPACE_CONTRACT.md" "$contract_content"
write_file "$workspace/contracts/eval_contract.$dataset_type.v1.json" "$eval_stub"

for d in raw_assets normalized pretrain midtrain sft holdout tokenizer manifests; do
  if [[ ! -f "$workspace/$d/.gitkeep" ]]; then
    : > "$workspace/$d/.gitkeep"
  fi
done

echo "[OK] created data workspace: $workspace"
echo "[OK] dataset_type: $dataset_type"
echo "[OK] contract: $workspace/contracts/WORKSPACE_CONTRACT.md"
echo "[OK] eval stub: $workspace/contracts/eval_contract.$dataset_type.v1.json"
