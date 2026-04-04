# spec15b Gold Mappings

This directory is the bounded gold workspace for `spec15b`, the strict
`system_diagram` family line.

Initial scope:

- `pipeline-overview.svg`
- `ir-pipeline-flow.svg`
- `kernel-registry-flow.svg`

Rules:

- keep stage ids family-generic (`stage_1`, `stage_2`, ...)
- keep terminal ids family-generic (`terminal`)
- keep visible payload text in `content.json`
- keep output scene DSL free of topic-bearing ids

Authoring order:

1. family-generic `scene.compact.dsl`
2. full `scene.dsl`
3. matching `content.json`
4. compiler smoke parity

Do not train `spec15b` until those steps pass.
