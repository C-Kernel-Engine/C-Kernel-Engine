# SPEC13 Run Prediction Framework 2026-03-24

Purpose: reduce guesswork for future `spec13` runs by turning the empirical
lessons from `spec12` and `spec13a` into a concrete decision checklist.

## Why Prediction Is Still Hard

Training outcomes are still only partly predictable because:

- loss is not the real decision surface
- the compiler contract exposes failure modes that token loss hides
- prompt-side changes can create new attractors without breaking grammar
- renderer rigidity can make the model look weaker than the optimization stack
  actually is

That said, the guesswork is no longer unconstrained. We now have enough data to
predict some failure classes before spending many more rungs.

## What `spec12` Taught Us

Transferred lessons:

- whole-scene rows are safer than fragment repair rows
- hidden eval matters
- warm-starts are worth protecting
- parity and preflight should gate real runs
- training loss can improve while capability collapses

These lessons prevented `spec13a` from repeating the `spec12 r7/r8` collapse.

## What `spec13a` Added

New lesson:

- prompt-to-plan inference introduces a new failure surface that `spec12`
  never had to solve

The line is stable, but family choice is weak.

Observed best-to-latest progression:

- `r2`: `0.596 / 1.000 / 0.596`
- `r3`: `0.533 / 0.967 / 0.533`
- `r4`: `0.536 / 0.964 / 0.554`
- `r5`: `0.304 / 0.911 / 0.304`
- `r6`: `0.333 / 0.979 / 0.333`

This is not noise. It is a clear stall pattern.

## Dominant Failure Pattern

`r6` failures are dominated by:

- `decision_tree -> table_matrix`
- `memory_map -> table_matrix`

That is the central predictive insight for the next phase.

Implication:

- if a proposed rung does not explicitly address family-choice confusion, it is
  unlikely to beat `r2`

## Heuristic Signals To Track Before Training

### 1. Anchor Non-Regression

Question:

- does the seeded model still behave on explicit `spec12`-style anchors?

If not:

- stop before training

### 2. Family Attractor Risk

Question:

- does the prompt mix overweight one family or one prompt surface?

Red flags:

- one family dominates the omitted-field rows
- one prompt surface removes the clearest disambiguating cue
- prompt ordering causes a specific family to become the default

### 3. Prompt-Surface Novelty

Question:

- are we changing too many prompt dimensions at once?

Examples:

- omitted fields
- reordered fields
- plain labels instead of bracketed fields
- bridge prompts with and without goals

If more than one of these shifts at once, outcome prediction gets weaker.

### 4. Hidden-Train Degradation

Question:

- is the hidden-train split falling faster than train/dev/test?

This is now a strong early warning sign that the model is memorizing surface
patterns instead of learning a stable planner.

### 5. Family Confusion Matrix

Question:

- where do wrong cases go?

For `spec13a`, this matters more than raw exact:

- wrong `decision_tree` that becomes `table_matrix`
- wrong `memory_map` that becomes `table_matrix`

is more informative than a flat exact miss.

## Heuristic Signals To Track After Training

1. Exact/renderable/materialized totals
2. Split-wise exact, especially:
   - `hidden_train`
   - `hidden_test`
3. Family confusion matrix
4. Anchor replay exactness
5. Missing-stop or parse noise rate
6. Materialized vs scene exact gap

## Concrete Lessons From `spec13a`

### `r2`

Best current rung.

Interpretation:

- bounded bridge worked somewhat
- family choice is above chance
- contract stayed clean

### `r3`

Widened omission too early.

Interpretation:

- more bridge freedom, less stable planning

### `r4`

Removed goal-less bridge rows but did not fix the core family confusion.

Interpretation:

- goal-less prompts were not the only problem

### `r5`

Plain labeled fields only.

Interpretation:

- overcorrected into a `table_matrix` attractor

### `r6`

Dual prompt surface.

Interpretation:

- recovered renderability
- did not recover family selection

## Predictive Rules For Future Rungs

Use these as hard planning rules.

### Rule 1

If the failure mode is unchanged for two rungs in a row, stop rung-chasing and
change the task decomposition.

### Rule 2

If exact drops while renderability stays high, suspect planning confusion before
suspecting kernel/training instability.

### Rule 3

If hidden-train exact is near zero while train/test stay materially higher,
suspect prompt-surface shortcutting.

### Rule 4

If a new rung changes both prompt schema and task freedom, prediction quality
will be low. Split those changes.

### Rule 5

Do not start a new rung unless you can name the failure class it is meant to
attack and the metric that would prove it helped.

## What This Means For `spec13a`

Another plain prompt-mix rung is low expected value right now.

The next `spec13a` run is justified only if it changes the decomposition, for
example:

- explicit family-router step
- stronger family anchors
- separate family-choice evaluation before full scene exactness

Without that, the most likely outcome is another table-matrix attractor.

## What This Means For `spec13b`

Prediction for `spec13b` is better if the first run proves a contract change,
not a full capability jump.

High-confidence `spec13b r1` target:

- generalized decision-tree IR
- deterministic layered renderer
- backward compatibility with current decision-tree gold scenes

Low-confidence `spec13b r1` target:

- generalized trees + generalized tables + generalized memory maps + weaker
  prompts + assistant behavior

That would be too many moving parts.

## Operational Monitoring Policy

For future `spec13` runs:

1. Keep tmux pollers only for active runs.
2. Log:
   - preflight outcome
   - parity outcome
   - ledger tail
   - live `ck_cli` loss
3. After each rung, immediately compute:
   - split summary
   - family confusion summary
   - anchor non-regression summary
4. If the same confusion pattern persists, stop and redesign.

## Bottom Line

We cannot predict exact capability scores deterministically yet.

But we can now predict with decent confidence:

- when a rung is attacking the wrong problem
- when a rung is likely to repeat an existing attractor
- when a failure is about planning, not optimization

That is enough to reduce wasted iterations, even before the next major leap in
contract design.
