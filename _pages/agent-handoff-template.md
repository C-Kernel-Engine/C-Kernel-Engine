# Agent Handoff Template

Use this file after a reboot, agent crash, or machine restart.

It is intentionally short and copy-paste friendly.

## Read First

1. `docs/site/_pages/spec-training-method.html`
2. `version/v7/reports/SPEC15A_MEMORY_MAP_NEXT_AGENT_BRIEF_2026-03-28.md` when `spec15a` is the active strict-line target
3. `version/v7/reports/SPEC[X]_EXECUTION_CONTRACT_YYYY-MM-DD.md`
4. `version/v7/reports/spec_family_autopilot_policy.json`

## Current Source Of Truth

- The last good `spec[x] rung[y]` is the training-method baseline.
- The last good `spec[x] rung[y]` is not the permanent tokenizer ceiling.
- The next line is `spec[z]`, scoped to one new `family[a]`.
- The next line is compiler-first and tokenizer-audit-first.
- Payload facts, copy, and numbers stay in external `content.json`.
- The model output should contain generic structure and visual control only.
- Do not resume stale or mixed-family runs blindly.

## What The Next Agent Should Do

1. Continue the current execution contract checklist.
2. Author the successor DSL/canonicalizer surface.
3. Implement the new `family[a]` compiler/rendering.
4. Build gold scenes plus `content.json`.
5. Produce compiler smoke outputs.
6. Build the tokenizer corpus from legacy carry-forward families plus the new `family[a]`.
7. Only then write the launcher and autopilot rung ladder.
8. Only then start background training and monitoring.

## Hard Stop Conditions

Do not launch training yet if any of these are missing:

- successor DSL surface
- new family renderer/compiler path
- gold scenes plus payloads
- compiler smoke report
- tokenizer corpus
- launcher script
- autopilot rung interventions

## Copy-Paste Prompt

```text
Read these first and treat them as the live source of truth:

1. docs/site/_pages/spec-training-method.html
2. docs/site/_pages/agent-handoff-template.md
3. version/v7/reports/SPEC15A_MEMORY_MAP_NEXT_AGENT_BRIEF_2026-03-28.md if spec15a is the active target
4. version/v7/reports/SPEC[X]_EXECUTION_CONTRACT_YYYY-MM-DD.md
5. version/v7/reports/spec_family_autopilot_policy.json

Current intent:
- keep the last good spec[x] rung[y] as the training-method baseline, not the tokenizer ceiling
- build the successor DSL/compiler/tokenizer path explicitly
- keep payload/content external to the model DSL
- add one capability at a time
- make spec[z] family[a] the next family-construction line
- do not launch training until the compiler, tokenizer corpus, launcher, and rung policy checklist are complete

Then continue from the current execution contract checklist and update the repo artifacts before starting any background training or autopilot.
```
