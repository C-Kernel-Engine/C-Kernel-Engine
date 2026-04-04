# Spec18 Routing-First Training Plan

- date: `2026-04-01`
- status: `proposed next branch`
- derived from:
  - `spec16_scene_bundle_l3_d192_h384_ctx768_r9`
  - `spec17_scene_bundle_l3_d192_h384_ctx768_r4`
  - `spec16 r12` and `spec17 r2/r4` autopsies

## Why Spec18 Exists

`spec17` was a good branch in the research sense:

- it proved the bounded-intent bridge is a real capability problem
- it exposed missing scaffold surfaces and weak contrast rows
- it forced the project to add materialized-catalog auditing and cleaner launch guards

But `spec17` is not currently a good promotion line:

- `r4` fixed the process bugs
- `r4` improved renderability
- `r4` still had `0.0` exact on `dev`, `test`, `hidden_train`, and `hidden_test`

So the remaining failure is no longer “we launched the wrong scaffold.” The remaining failure is that the branch is trying to learn too much at once:

- family routing
- sibling-form choice
- style inference
- topology/count inference

`spec18` should split those apart.

## Core Hypothesis

The next useful branch should teach bounded-intent planning in two stages:

1. learn `family + form` routing under a smaller target space
2. add style and topology inference only after routing is stable

If `spec18` still fails after that decomposition, the next lever is likely capacity, not more same-style curriculum churn.

## Output Contract

Keep the frozen `spec16` shared bundle contract:

- exactly one `[bundle] ... [/bundle]`
- no raw SVG
- no free prose
- no topic ids or payload text inside the bundle

The compiler, renderer, and `content.json` boundary stay unchanged.

## Dataset Buckets

`spec18` should use an explicit mixture manifest with named buckets. Do not treat the training text as one undifferentiated corpus.

### Bucket A: Anchor Replay

Purpose:

- preserve the frozen `spec16` contract
- prevent family or syntax collapse

Contents:

- explicit family/form/style/topology bundle prompts from the frozen winner
- clean stop anchors
- exact canonical rows

### Bucket B: Routing Direct

Purpose:

- teach `topic + goal + audience -> family + form`

Contents:

- prompts that require only family and form selection
- targets still emit the full shared bundle
- style and topology remain canonical defaults for the chosen profile

### Bucket C: Form Contrast

Purpose:

- distinguish sibling forms inside the same family

Contents:

- near-neighbor prompts with explicit sibling-form distractor metadata
- balanced across all forms in each family

### Bucket D: Family Contrast

Purpose:

- distinguish neighboring families from similar intent surfaces

Contents:

- cross-family near-neighbor prompts
- explicit distractor family metadata
- routing-only pressure without extra style noise

### Bucket E: Style/Topology Bridge

Purpose:

- add style and count inference only after routing rows exist

Contents:

- bounded hints for `theme`, `tone`, `density`, `background`
- bounded hints for topology/count defaults

### Bucket F: Paraphrase / Surface Robustness

Purpose:

- improve prompt-surface robustness without widening semantics too early

Contents:

- paraphrases
- reordered prompt tags
- lightweight lexical alternations

### Bucket G: Hidden / Holdout

Purpose:

- evaluation only

Contents:

- prompt-surface holdouts
- semantic holdouts
- hidden paraphrase and hidden recombination rows

### Bucket H: Repair Rows

Purpose:

- narrow syntax hygiene only

Contents:

- singleton cleanup
- stop-boundary cleanup
- compiler-valid closure rows

Rules:

- cap tightly
- no warning-language prose
- no `[OUT]` contamination teaching
- no broad “do not do X” rows

## Proposed Blend Ratios

For the first `spec18` canary:

- `30%` Anchor Replay
- `25%` Routing Direct
- `15%` Form Contrast
- `10%` Family Contrast
- `10%` Style/Topology Bridge
- `5%` Paraphrase / Surface Robustness
- `5%` Repair Rows

Notes:

- keep repair rows well below the `10-15%` warning zone from the training-method rules
- keep routing pressure (`Routing Direct + Form Contrast + Family Contrast`) at `50%`
- this branch should spend more mass on choosing the right family/form than on style polish

For a follow-up widened rung only if routing improves:

- `25%` Anchor Replay
- `20%` Routing Direct
- `15%` Form Contrast
- `10%` Family Contrast
- `20%` Style/Topology Bridge
- `5%` Paraphrase / Surface Robustness
- `5%` Repair Rows

## Training Stages

### Stage A: Routing Lock

Goal:

- learn family and form choice with minimal extra variance

Bias:

- heavy Anchor Replay
- heavy Routing Direct
- full contrast coverage
- only light style/topology bridge

### Stage B: Style/Topology Expansion

Goal:

- keep the learned routing while widening inferred bundle detail

Bias:

- maintain routing anchors
- increase style/topology bridge
- preserve contrast rows so routing does not regress while the target space widens

## Determinism Gates

`spec18` should not launch unless all of these pass:

1. Compiler parity:
   - gold scene packs compile to the same SVG as reference assets

2. Tokenizer determinism:
   - frozen tokenizer or versioned tokenizer branch only
   - no accidental vocab drift
   - no `<unk>` on gold rows

3. Dataset QC:
   - row parse pass
   - train/holdout prompt disjointness
   - duplicate accounting
   - bucket counts written to a manifest

4. Materialized blueprint audit:
   - audit the generated render catalog, not just the blueprint JSON

5. Replay determinism:
   - run `check_replay_determinism_v7.py`
   - require repeatable loss/param drift across identical harness runs

6. Training parity regimen:
   - require A1, A4, D1, D2, E1, F1
   - tolerate only the already-whitelisted non-blocking regimen cases

7. Token-budget integrity:
   - selected tokens must match actual processed tokens closely enough to call the run a real canary
   - no one-third-epoch “signal” canaries on a new branch

8. Seed and cache discipline:
   - seed from frozen `spec16 r9`
   - keep run artifacts under `~/.cache/ck-engine-v7/models/train/`

## Promotion Criteria

`spec18 r1` should only promote if all of the following are true:

1. `dev` exact is nonzero
2. `test` exact is nonzero
3. at least one hidden split exact is nonzero
4. no family exact rate regresses relative to the frozen baseline on the explicit-contract slices
5. renderable stays high enough that gains are not coming from syntax collapse
6. the branch beats `spec17 r4` on held-out exactness, not just on renderability

Recommended stronger gate for `r2`:

- hidden exact must improve over `r1`
- family routing errors must clearly shrink in the failure gallery
- no promotion if exact stays `0.0` on both holdout splits, even when renderable improves

## Decision On Spec17

Do not treat `spec17` as “bad” or wasted.

Treat it as:

- a successful diagnostic branch
- a failed promotion branch
- the source of the next branch design

What should stop is not `spec17` thinking. What should stop is more near-identical `spec17` raw rungs on the same problem framing.

The forward move should be:

- freeze `spec17 r4` as the cleanest bounded-intent autopsy
- keep `spec16 r9 + deterministic repair` as the practical default path
- start `spec18` as the routing-first branch

## Immediate Implementation Tasks

1. Add a `spec18` blueprint with explicit bucket ids and stage mixes.
2. Materialize a routing-first dataset generator that emits:
   - routing-direct rows
   - sibling-form contrast rows
   - sibling-family contrast rows
   - light style/topology bridge rows
3. Add bucket-count and token-count manifests.
4. Reuse the current launch guard, parity regimen, and replay determinism checks.
5. Add a stricter post-run decision artifact that blocks promotion when renderability rises but holdout exactness stays at zero.
