# SVG Family Execution Method 2026-03-25

Purpose: define the long-horizon, repeatable method for expanding the SVG DSL
line from `spec12`/`spec13` into a family-by-family system that can eventually
cover most of `docs/site/assets` without collapsing the training signal.

This document is the operating plan for:

- how we decide what family to add next
- when the tokenizer must change
- when the compiler must change first
- what a family `r1` is allowed to prove
- how we poll, monitor, and promote training runs
- how we avoid redoing the entire empirical journey from zero every time

It is meant to be a reference methodology, not just a one-off run note.

---

## Bottom Line

We should **not** try to train one monolithic model to emit every infographic
family at once.

We should instead:

1. add **one SVG family at a time**
2. make the **compiler and renderer robust first**
3. test whether the **gold DSL + content -> SVG** path is already strong
4. only then train the model on that family
5. run a **small number of disciplined rungs**
6. promote only when hidden eval and compiled SVG quality say the family is
   stable

This is the shortest path to a model that can eventually create most of the
infographics we care about.

---

## What Transfers From Earlier Specs

The `spec02 -> spec12` journey was not wasted narrow work. It established the
rules that should continue to govern all later lines:

- loss is not the real decision surface
- compiler-backed scoring is the real target
- whole-scene supervision is safer than fragment repair rows
- hidden eval matters
- parity and preflight must gate real runs
- tokenizer drift must be treated as a first-class intervention
- one rung should answer one question

Those rules should remain in force for every new family.

What does **not** transfer automatically:

- a model that can serialize one fixed family will not automatically infer a
  new family
- a renderer that looks generic at the parser level is often still rigid in the
  actual placement logic
- prompt-side weakening (`spec13a`) creates a different failure surface than
  renderer/IR generalization (`spec13b`)

---

## `spec12 r17` Successor Contract

`spec12 r17` remains the best **method** baseline in the repo.

It is **not** the permanent tokenizer ceiling and it is **not** proof that the
current DSL surface is already broad enough for every later family.

Future family work should preserve what made `r17` strong:

- whole-scene supervision
- compiler-backed scoring
- hidden evaluation
- parity and preflight gating
- small, interpretable `r1` runs
- fixed architecture for the first rung of a new family

Future family work should **not** preserve the narrow parts that were only
acceptable because `spec12` covered a bounded family set:

- asset-specific topology assumptions in the DSL
- a tokenizer surface that cannot cleanly express the next reusable family
- topic or case identity baked into the output contract
- domain facts embedded in the model output instead of external payloads

This is the successor rule:

1. keep the **training method** from `spec12 r17`
2. remove the **non-generalized output contract** when a new family requires it
3. add **one new family capability at a time**
4. redesign the DSL/canonicalizer first so the new family is generic
5. keep visible payload in external `content.json`
6. build the tokenizer from the finalized generic DSL corpus
7. verify `gold DSL + content.json -> SVG` before launching training

In other words:

- preserve the proven curriculum discipline
- do not freeze the representation at the point where `spec12` stopped

---

## Current Status

### `spec13a`

Status:

- useful, but stalled
- best rung remains `r2`
- operational stack is healthy
- capability is not climbing

Best known baseline:

- `spec13a r2`: `0.596 exact / 1.000 renderable / 0.596 materialized`

Latest evidence:

- newer rungs regressed while keeping grammar mostly intact
- dominant failure is family confusion
- `decision_tree -> table_matrix`
- `memory_map -> table_matrix`

Decision:

- pause blind `spec13a` runging
- keep one explicit family-router rung in reserve only if needed later

### `spec13b`

Status:

- useful reference line
- first generalized graph-family baseline exists

Current baseline:

- `spec13b r1`: `0.000 exact / 0.727 renderable / 0.455 materialized`

Interpretation:

- the graph-family compiler target is real
- the model can often emit compiler-usable graph DSL
- exact scene recovery is still weak
- this is a valid `r1`, not a collapse

Decision:

- keep `spec13b` as graph-family reference material
- do not resume a stalled graph rung blindly as the successor mainline

---

## Core Method

Every new family follows the same loop.

### Stage 0: Asset Classification

Before adding a family, classify candidate assets into:

- already-covered by an existing family
- reducible to an existing family with better content packs
- requiring a new renderer family
- not yet worth formalizing

Do not add a new family just because an asset is visually different.
Add one only when the underlying structure is genuinely new.

### Stage 1: Family Design

Define the family contract:

- what the scene IR must represent
- what placement/layout logic must be deterministic
- what semantic payload remains in external `content.json`
- what visual degrees of freedom are controlled by the DSL

The model should learn to emit:

- reusable structure
- reusable refs
- stable control fields

The model should **not** learn:

- asset-specific hacks
- one-off ids
- renderer implementation details that belong in the compiler

### Stage 2: Compiler-First Validation

Before any training:

1. author `3-5` gold scenes for the family
2. author matching `content.json` payloads
3. compile them with the real renderer
4. generate a smoke report

The family is **not train-ready** until:

- all gold scenes compile
- the SVG is semantically correct
- the layout is visually strong enough to be worth learning
- the renderer behaves deterministically

If the compiler cannot already turn gold DSL into good SVG, training is too
early.

### Stage 3: Tokenizer Audit

For every new family, ask:

1. Can this family be expressed with the current reserved output control tokens?
2. If not, can the new structure be expressed with a **small reusable generic**
   token set?
3. Are we about to add asset-specific tokens instead of family-level tokens?

Rules:

- prefer **generic structural tokens**
- for strict visual-language lines, keep domain semantics out of the
  model-facing prompt surface; use routing metadata and external
  `content.json` instead
- never add tokens for one asset or one case id

Tokenizer decisions:

- if current control surface is enough: reuse tokenizer
- if genuinely new structure is required: create a new family tokenizer line
- do not pretend a new output family is warm-start compatible if the control
  surface changed materially

### Stage 4: Dataset and Probe Design

A family dataset should include:

- canonical tagged prompts
- a small number of bridge prompts
- hidden paraphrase prompts
- hidden holdout prompts

The rows should be:

- whole-scene rows only
- compiler-backed
- family-balanced

Avoid by default:

- fragment repairs
- contamination rows
- synthetic corruption rows
- local continuation hacks

Each family needs:

- visible probe
- hidden paraphrase probe
- hidden holdout probe
- prompt-to-SVG report

### Stage 5: Preflight and Parity

Before a real rung:

- materializer workspace check
- tokenizer roundtrip
- compiler smoke report
- family probe contract build
- training parity regimen
- preflight recommended token budget

If parity or compiler readiness is not good, do not launch the rung.

### Stage 6: Small Rung First

Every family starts with a small `r1`.

`r1` should answer:

- can this family be learned at all under the current contract?

`r1` should not try to prove:

- broad semantic mastery
- many unrelated prompt styles
- all assets in the family
- architecture scaling

Keep `r1` narrow and interpretable.

### Stage 7: Polling and Background Monitoring

For every live rung:

- use a dedicated tmux train session
- use a dedicated tmux monitor session
- append a 15-minute polling log

Artifacts:

- train log
- monitor log
- run ledger
- probe report
- prompt-to-SVG report if available

Monitoring should be automatic.
Promotion should not be.

### Stage 8: Post-Run Decision

After each rung, answer:

1. Did exact/renderable/materialized move in the expected direction?
2. Did the family-specific failure mode change?
3. Did hidden eval degrade relative to visible?
4. Did the model produce the right family but the wrong structure?
5. Did the model produce the wrong family entirely?
6. Is the problem in:
   - tokenizer
   - compiler
   - dataset
   - prompt contract
   - family IR
   - capacity

Then choose one of:

- promote the rung
- do one focused repair rung
- stop rung-chasing and redesign the family contract

---

## Promotion Rules

### Promote a Rung When

- exact or materialized quality improves materially
- hidden eval does not collapse
- failure class is narrowing rather than diffusing
- compiler-valid outputs are increasing
- the family contract still looks clean

### Pause a Line When

- the same failure mode repeats for two rungs
- renderability stays high but exact/materialized do not improve
- a prompt or tokenizer change adds ambiguity without new information
- the compiler target itself is still unstable

### Change the Family Contract When

- the renderer is the bottleneck
- multiple assets in the same supposed family need incompatible hacks
- the tokenizer has to grow with every new asset
- exactness remains near zero despite clean renderability

---

## Architecture Policy

Do **not** turn every family `r1` into a model-size experiment.

Default policy:

- keep `3L / 192d / 384ff / ctx768`
- keep CK native train/infer path
- keep whole-scene supervision

Only scale the architecture after:

- compiler is strong
- family IR is stable
- `r1/r2` show the contract is coherent

Otherwise, scale will hide the wrong lesson.

---

## Tokenizer Policy

Tokenizer strategy should be family-level, not asset-level.

### Allowed Reasons To Start a New Tokenizer Line

- a new reusable output family requires new structure tokens
- a new generalized IR cannot be expressed by the old control surface
- we intentionally choose a broader family contract

### Bad Reasons

- one new asset name
- one new topic id
- one new case id
- one special renderer exception

### Preferred Long-Term Policy

- stable generic structure tokens
- prompt-side semantics mostly through ASCII/base pieces
- external `content.json` for asset payloads
- new tokenizer lines only when the output grammar genuinely grows
- keep case ids and asset identity out of the output DSL unless the renderer
  truly needs them
- use the router/data layer to select payloads; use the model to plan family
  structure

### DSL Boundary Rule

When a family is still being stabilized, keep these layers separate:

- output DSL: generic family structure
- model-facing prompt: family-generic design/control wording only
- routing/request metadata: user-visible task wording, domain/topic identity,
  and case selection
- `case_id`: catalog, holdout bookkeeping, routing metadata
- `content.json`: visible copy, numbers, and topic-specific facts

Do not blend asset-specific topics into either the model-side prompt surface or
the model-side output contract just to make one asset work. That trains
memorized labels instead of transferable SVG planning.

### Successor DSL Rule

When a new line grows beyond the `spec12` family set, the correct move is:

- keep legacy family behavior reachable by the compiler
- redesign the DSL around reusable structure tokens
- regenerate the tokenizer from the new generalized DSL corpus
- keep case/topic semantics outside the model output

Do **not** force a wider family into the old control surface just because
`spec12 r17` was the champion.

---

## Family Sequence

This is the recommended sequence to reach broad asset coverage while keeping the
empirical loop deterministic.

### `spec13b`: Graph Family

Scope:

- generalized `decision_tree`
- `flow_graph`

Why first:

- graph-family work is already started
- it covers many current assets
- it is structurally coherent

Success target:

- graph-family hidden exact/materialized become stable enough that adding more
  graph assets is a data problem, not a renderer problem

### `spec14a`: Comparison Board

Scope:

- generalized comparison tables
- multi-card boards
- structured comparison layouts

Why next:

- very high asset coverage
- many current “technical diagrams” are actually structured comparisons

### `spec14b`: Timeline

Scope:

- event sequences
- milestone bars
- grouped eras
- callouts

Why later:

- smaller family
- easy to isolate
- lower coverage leverage than comparison boards

### `spec15a`: Generalized Memory Map

Scope:

- first line to strictly enforce domain-agnostic model input and output
- no topic-bearing prompt text in model training rows
- memory facts and labels selected upstream and bound through
  `content.json` / routing only
- variable segment count
- spans / brackets
- multi-lane memory views
- annotation cards

Why after that:

- important family
- more specialized than graph or board layouts

### `spec15b`: Technical Diagram / System Diagram

Scope:

- composed architecture diagrams
- mixed graph + comparison + memory motifs

Why last:

- this is where multiple mature families should start composing
- do not begin here

---

## Expected Rung Count

We should not think in terms of infinite blind runging.

Reasonable expectation:

- `3-5` disciplined rungs per new family
- possibly fewer if the compiler target is strong before `r1`
- more only if the family definition was too broad

This means:

- `spec13b` graph family probably needs a few more rungs
- later families should be cheaper if we preserve the method

What we want is:

- a small number of high-information rungs
- not endless probe chasing

---

## Background Operations Policy

Automation is appropriate for:

- polling active runs
- collecting logs
- building probe reports
- generating prompt-to-SVG reports
- appending logbook entries
- building compiler smoke reports

Automation is **not** appropriate for:

- automatically launching a new rung after the last one finishes
- silently promoting a rung
- silently changing tokenizer lines
- silently changing family scope

Why:

- each new rung should answer a named question
- family changes are design decisions, not cron jobs

So the operational policy should be:

- automatic polling during an active run
- manual go/no-go between rungs

### Agent-Readable Execution Contract

Any agent may run the full family loop autonomously **only after** the repo has
an explicit execution contract for that family.

That contract must name:

- family scope
- compatibility promise with older families
- DSL boundary rules
- tokenizer policy
- gold compiler-smoke assets
- dataset/materializer requirements
- launcher path
- rung ladder and stop conditions

If that contract is missing, agents should stop after compiler/tokenizer
authoring instead of guessing the next training rung.

Current implementation artifacts:

- policy:
  `version/v7/reports/spec_family_autopilot_policy.json`
- executor:
  `version/v7/scripts/spec_family_autopilot_v7.py`

The autopilot may auto-advance only when:

- the family already has an explicit policy entry
- the next rung intervention is already encoded
- the line stays inside the same family/tokenizer contract

It must stop and emit status instead of guessing when:

- a new family would need to start
- tokenizer policy would need to change
- renderer/IR authoring is still missing
- the family execution contract has not been authored yet

---

## What We Need To Measure Every Time

For every family and rung, record:

- exact rate
- renderable rate
- materialized exact rate
- visible split breakdown
- hidden split breakdown
- family confusion matrix if applicable
- missing stop marker rate
- malformed token/field duplication rate
- compiler error taxonomy
- prompt-to-SVG examples
- prediction vs actual outcome

This is how we replace guesswork with accumulated engineering intuition.

---

## Immediate Next Steps

### Mainline

1. Keep `spec13a` paused at `r2`.
2. Keep `spec12 r17` as the training-method baseline.
3. Treat the next broadening step as a **compiler-first successor line**, not as
   a blind continuation of the latest stalled graph rung.

### Planning

4. Start `spec14a` as the explicit next family:
   - scope: `comparison_board`
   - method: compiler first, tokenizer second, training third
   - contract: `version/v7/reports/SPEC14A_COMPARISON_BOARD_EXECUTION_CONTRACT_2026-03-26.md`
5. Keep `spec14b` scoped to `timeline`.
6. Do not mix `comparison_board` or `timeline` into `spec13b`.

### Method

7. Before the first rung of every new family:
   - compiler smoke first
   - tokenizer audit second
   - preflight/parity third
   - training fourth

---

## Final Principle

The goal is not to randomly see whether a bigger training run works.

The goal is to build a system where:

- the compiler is strong enough that training against it is meaningful
- the tokenizer only changes when the structure really changes
- each spec adds one family
- each rung answers one question
- the model eventually learns to emit beautiful, compiler-backed infographics
  across most of the families we care about

That is the path to broad SVG capability without losing determinism.
