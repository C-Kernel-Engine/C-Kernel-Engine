# Adapter-Based Deterministic Preflight

This note captures the reusable pattern behind `spec07_preflight_v7.py`.

## Goal

Use the same training-side preflight shape across deterministic domains:

- `svg` scene DSL -> compile -> render -> compare
- `html`/`css` DSL -> parse -> normalize -> render/check invariants
- `python`/`node` code DSL -> parse -> import/build -> test
- `sql` DSL -> parse -> execute on fixture DB -> compare result sets

The reusable rule is:

1. compute real packed-token budgets from the staged tokenizer and dataset
2. validate deterministic catalog hygiene before training
3. run a small balanced canary before spending real compute
4. fail early on structural or oracle regressions

## Generic Contract

The generic contract now lives in:

- `version/v7/scripts/deterministic_preflight_adapters_v7.py`

It exposes:

- `DeterministicPreflightCase`
- `DeterministicPreflightValidation`
- `DeterministicPreflightAdapter`
- `budget_stage(...)`
- `validate_case_collection(...)`
- `summarize_status(...)`

The adapter owns domain-specific logic:

- load catalog rows
- build a balanced canary
- parse one expected output
- materialize it with the deterministic oracle
- decide whether the oracle result matches the expected artifact

The shared core owns dataset-side logic:

- packed one-epoch token accounting
- effective epoch recommendation
- oversize-row audit
- per-group summary accounting
- failure collation
- preflight status synthesis

## Minimal Adapter Shape

```python
class DeterministicPreflightAdapter(Protocol):
    name: str
    group_name: str

    def load_catalog_cases(self, run_dir: Path, prefix: str) -> list[DeterministicPreflightCase]:
        ...

    def build_canary_cases(self, run_dir: Path, prefix: str, per_split: int) -> list[DeterministicPreflightCase]:
        ...

    def validate_case(self, case: DeterministicPreflightCase) -> DeterministicPreflightValidation:
        ...
```

The validation record is intentionally generic:

- `starts_clean`
- `ends_clean`
- `parse_ok`
- `materialized_ok`
- `oracle_exact`

That keeps the core reusable while letting each adapter translate those fields into domain-specific names such as `renderable`, `compiled`, or `tests_passed`.

## Mapping By Domain

### SVG

- group: `layout`
- parse: scene or atom DSL parser
- materialize: deterministic SVG compiler/renderer
- oracle: exact rendered SVG

### HTML

- group: page type or layout family
- parse: HTML/DOM parse
- materialize: canonical DOM or screenshot/invariant pipeline
- oracle: normalized DOM equality or screenshot/invariant pass

### Python / Node

- group: task family or module shape
- parse: AST parse
- materialize: import/build/test harness
- oracle: deterministic test result, fixture output, or normalized AST/format result

### SQL

- group: query family
- parse: SQL parser
- materialize: execute against fixed fixtures
- oracle: result-set equality

## Guardrails

- Do not reuse token budgets across representations without recomputing packed one-epoch totals.
- Do not treat train loss as a promotion gate for DSL tasks.
- Gate by slice (`layout`, `task`, `module_family`, `query_family`), not only aggregate score.
- For new DSLs, treat the first run as a format-control run before a capability run.

## Current Example

`spec07_preflight_v7.py` now uses the generic core with a scene-SVG adapter while keeping its existing report shape for backward compatibility.
