# Spec / Run Discipline And Training Intuition 2026-03-18

This note defines the practical method behind the `spec[x]` and `r[y]`
training line.

The goal is not to make runs look scientific after the fact. The goal is to
turn each run into a controlled question so the project builds real training
intuition over time.

## Why This Exists

Public discussion often overstates one true point into a false conclusion:

- true: no one has a complete first-principles theory of why every capability
  emerges inside large language models
- false: therefore model training is mostly guesswork

That is not how serious model work operates.

The useful middle ground is:

- mechanistic theory is incomplete
- empirical control is real
- system design gives even more control

In practice, the operator can often predict:

- which data mix will strengthen a contract
- whether a failure is syntax, semantics, context, tokenizer, or compiler
- whether more capacity will help or only hide a bad representation
- whether a new spec is needed or a narrow repair run is enough

That is the kind of understanding this method is trying to build.

## What `spec[x]` Means

A `spec` is a new experiment contract.

It should answer a materially different question, such as:

- new representation
- new compiler boundary
- new prompt surface
- new output family
- new token granularity
- new separation between structure and content

A new spec changes the learning problem.

## What `r[y]` Means

A run revision is a controlled iteration inside the same spec.

It should hold the main contract fixed and change one primary axis:

- curriculum pressure
- repair rows
- token budget / effective epochs
- prompt balance
- negative examples
- decode hygiene
- capacity, only after representation is stable

`r[y]` should not silently redefine the task.

## When To Start A New Spec

Start a new spec when the current spec ceiling is real and measured.

Good reasons:

- current DSL is too literal or too loose
- compiler cannot express the target assets
- prompt surface is over-specified and blocks planning
- token budget is structurally wrong for the task
- the model is solving the old benchmark but not the next product need

Do **not** start a new spec just because one run failed.

## When To Stay Inside The Same Spec

Stay in the current spec when:

- the failure is narrow
- the probe shows the right family but wrong closure/order
- renderability is the main gap
- only one or two slices are weak
- the representation still matches the intended product boundary

That is when `r2`, `r3`, or `r4` is the right move.

## What Frontier-Style Intuition Actually Looks Like

The practical skill is not "fully understand intelligence."

It is learning to reason in layers:

1. data distribution
2. interface / representation
3. optimizer and token budget
4. capacity and context
5. evaluation
6. system scaffolding

If a run changes, ask in this order:

1. did the target contract change?
2. did the data pressure change?
3. did the tokenizer surface change?
4. did context or packing change?
5. did the compiler or probe change?
6. only then ask whether model size is the bottleneck

## What Frontier Labs Usually Know

The claim that "no one knows how LLMs work" is too broad to be useful.

A better version is:

- full mechanistic theory is still incomplete
- empirical control is strong
- product and system control are often stronger still

In other words, frontier labs may not be able to explain every internal circuit,
but they can still learn, with real discipline, that:

- this data mixture
- with this architecture
- at this scale
- under this objective
- with these post-training steps
- and these evaluations

tends to produce these kinds of behaviors.

That is not total understanding, but it is real engineering knowledge.

## Why Data Curation Still Matters

Scaling does not erase the training distribution.

The model still compresses what it sees.

So:

- noisy data teaches noisy boundaries
- over-weighted slices distort behavior
- weak holdouts create false confidence
- clean repair rows can fix a narrow failure much faster than more generic data

The project does not need a complete theory of generalization to benefit from
careful curation. It only needs a disciplined loop:

- define the contract
- shape the data to teach that contract
- measure the right behavior
- repair the actual failure layer

## The Failure-To-Repair Loop

In practice, a lot of progress comes from a simple discipline:

1. observe the weakness
2. classify the weakness
3. teach that weakness directly
4. rerun from the best relevant checkpoint if the spec is still the same

That is not the same as "keep adding more data."

The important step is classification.

Before adding rows, decide whether the failure is mainly:

- data
- DSL
- compiler
- tokenizer
- token budget
- decode hygiene
- capacity

Then respond at the right layer.

Examples:

- if the model misses `[/scene]`, add closure and continuation rows and tighten
  stop hygiene
- if the model confuses two families, add contrast rows and family-local anchors
- if the DSL itself is too verbose or ambiguous, start a new spec instead of
  piling on more data
- if the compiler cannot express the target asset, fix the compiler before
  retraining

If the contract is unchanged, continuing from the best current weights is often
the right move. That is what a repair run is for.

Good rule:

- same spec -> usually continue or rerun inside the same family
- new spec -> usually treat it as a new line, not just another repair

This is how failures become supervision rather than wasted compute.

## The Failure Matrix

Use this matrix as the default run read before deciding whether to repair data,
change the DSL, fix the compiler, or scale compute.

| Failure Class | What It Looks Like | What To Track | Likely Cause | Typical Fix |
|---|---|---|---|---|
| `scene_prefix_failure` | missing `[scene]`, missing `[layout:...]`, duplicated top-level attrs | start-valid rate, missing-layout rate, duplicate-attr rate | weak canonical anchors, too much fragment training | add full-scene canonical rows and prefix-only repair rows |
| `scene_suffix_failure` | missing `[/scene]` | close-tag miss rate | weak termination training, decode stop/budget issues | add close-tag rows, verify stop markers, verify decode budget |
| `block_nesting_failure` | wrong closing tag, invalid nesting, repeated block open | nesting error rate by block type | weak block grammar, fragment-heavy repair rows | add balanced block rows, transition rows, canonical block-order rows |
| `budget_truncation` | correct prefix cut off | `truncated_at_budget` rate, prompt/output token counts | decode budget too small or context too small | raise decode budget first, then context only if needed |
| `special_token_leak` | `<|bos|>`, `<|eos|>`, prompt tokens inside scene output | special-token leak rate | tokenizer boundary contamination, bad row boundaries | strip/control special tokens and strengthen scene-only targets |
| `layout_drift` | wrong family or empty layout | per-layout confusion matrix | family overlap, weak family anchors | more direct family rows and layout-class repair rows |
| `theme_tone_drift` | wrong or duplicated theme/tone attrs | theme/tone confusion matrix, duplicate-attr rate | weak top-level canonicalization, over-repair | canonical scene-header rows and dedupe rules |
| `renderable_but_not_exact` | SVG compiles but scene DSL is off | exact vs renderable gap | semantic drift, ordering drift | targeted exactness repair rows |
| `exact_but_not_materialized` | scene matches but final SVG differs | materialized-exact gap | compiler or content-binding bug | fix renderer/probe path, not training |
| `family_imbalance` | one family learns, one collapses | per-family exact/renderable/materialized | data imbalance or family-specific grammar difficulty | family-specific anchors and family-weight tuning |
| `undertraining` | high loss, broad failure everywhere | loss curve, steps per epoch, token budget | too little budget for the grammar difficulty | raise epochs or total tokens |
| `over_repair_fragmentation` | valid local fragments but corrupted full scenes after a repair push | renderable drop after repair-row increase, local grammar error counts | too many fragment rows relative to clean full scenes | reduce fragment ratio and add more clean full-scene anchors |
| `compiler_parity_gap` | model may be fine but target family still looks weak | gold asset parity score | compiler not expressive enough | do a compiler pass before more training |
| `probe_accounting_bug` | obviously good outputs score wrong | mismatch between exact, renderable, and materialized evidence | reporting or probe bug | fix probe/report path first |

## Minimum Run Scoreboard

Every run should publish the same small scoreboard:

| Metric | Why |
|---|---|
| `exact_rate` | scene contract fidelity |
| `renderable_rate` | structural validity |
| `materialized_exact_rate` | final compiler truth |
| `budget_truncation_rate` | separates learning failure from budget failure |
| `missing_scene_start_rate` | top-level grammar health |
| `missing_scene_end_rate` | termination health |
| `duplicate_attr_rate` | canonicalization health |
| `block_nesting_error_rate` | nested grammar health |
| `special_token_leak_rate` | tokenizer/output contamination |
| per-layout exact/renderable/materialized | family-specific diagnosis |
| train/dev/test split rates | overfit detection |
| `gold_asset_parity_score` | compiler readiness |

Simple interpretation rules:

- low exact + high renderable -> semantic or ordering drift
- low renderable + low truncation -> grammar corruption
- high truncation -> budget problem
- high exact + low materialized exact -> compiler or probe bug
- one family bad, others good -> family-specific curriculum problem
- all families bad + high loss -> undertraining
- all families bad after huge repair increase -> over-repair fragmentation

## What To Copy From Frontier Practice

The useful habits are not mystical. They are operational:

- proxy runs before expensive runs
- aggressive evaluation design
- controlled ablations instead of folklore
- explicit failure taxonomies
- careful mixture design
- separation of base learning, post-training, and system scaffolding
- refusal to let one run answer five different questions at once

This is exactly why the spec/run method matters. It turns broad ambition into a
sequence of smaller, interpretable steps.

## The Working Method

Every spec/run should answer one explicit question.

Recommended loop:

1. define the contract
2. build compiler parity and gold assets
3. measure token budgets and preflight gates
4. launch the smallest real run that can answer the question
5. inspect probe, not loss alone
6. classify the failure
7. decide: repair run, new spec, compiler step, tokenizer step, or capacity step

If a run cannot answer a specific question, it is not a useful run.

## A Practical Failure Taxonomy

### Representation failure

Symptoms:

- good loss, bad exact
- wrong families across the board
- prompt-understanding confusion

Action:

- redesign the DSL or prompt contract

### Curriculum failure

Symptoms:

- some families strong, some collapse
- narrow slices regress after edits
- anchor replay too weak

Action:

- repair rows, replay, better balancing

### Grammar failure

Symptoms:

- missing close tags
- invalid nesting
- component leakage across families

Action:

- continuation rows
- stricter canonicalization
- family-local anchors

### Compiler failure

Symptoms:

- model output looks right but renderability is low
- exact vs materialized mismatch

Action:

- fix the compiler/probe path before retraining

### Capacity failure

Symptoms:

- stable grammar
- stable data
- broad semantic underfit remains

Action:

- only now scale layers, width, context, or compute

## Ethical Scaling Path

Going from something small to something bigger should not mean removing
constraints faster than understanding improves.

The safer path is:

1. start with a bounded contract
2. prove syntax, renderability, and evaluation honesty
3. widen the family set only after the previous one is stable
4. add weaker prompts gradually, not all at once
5. keep retrieval, content systems, and compilers deterministic where possible
6. require stronger non-regression gates as capability increases
7. keep humans in the loop for high-stakes domains

In practice, this means:

- do not jump from explicit tags to open-ended language without a bridge
- do not present a compiler-heavy system as if it were free general intelligence
- do not widen product scope until the failure modes are visible and measured
- do not scale compute just to hide unresolved representation problems

Ethical scaling is not only about policy. It is also about research honesty:

- know what the model is really doing
- know what the compiler is doing
- know what the content system is doing
- and report those boundaries clearly

That makes the system safer and the research more believable.

## Bottom Line

The project does not need full mechanistic interpretability before it can train
models methodically.

What it needs is:

- stable contracts
- controlled run revisions
- honest probes
- disciplined failure classification

That is how practical model intuition is built.
