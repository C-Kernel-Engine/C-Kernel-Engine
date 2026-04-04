# spec14b Gold Mappings Workspace

Status: gold mappings authored, compiler smoke passed

Family: `timeline`

Accepted `r1` gold assets:

- `docs/site/assets/ir-v66-evolution-timeline.svg`
- `docs/site/assets/ir-timeline-why.svg`

Rejected for `r1`:

- `docs/site/assets/ir-pipeline-flow.svg`

Reason:

- `ir-v66-evolution-timeline.svg` is a clear milestone timeline with dated nodes
  and one dominant chronological spine.
- `ir-timeline-why.svg` is a compact staged sequence that still fits a bounded
  timeline family.
- `ir-pipeline-flow.svg` is not a clean timeline. It mixes process-flow,
  system-diagram, and explanatory panel behavior. That belongs in a later
  mixed-family line, not `spec14b r1`.

Current workspace files:

- `ir-v66-evolution-timeline.scene.dsl`
- `ir-v66-evolution-timeline.scene.compact.dsl`
- `ir-v66-evolution-timeline.content.json`
- `ir-timeline-why.scene.dsl`
- `ir-timeline-why.scene.compact.dsl`
- `ir-timeline-why.content.json`
- `spec14b_gold_pack_status_20260329.json`

Smoke outputs:

- `version/v7/reports/spec14b_smoke/compiler_smoke_report.json`
- `version/v7/reports/spec14b_smoke/compiler_smoke_report.html`

Immediate next authoring tasks:

1. decide whether a third strict timeline asset is needed for `r1`
2. build the `spec14b` generator/materializer pair
3. define the `spec14b` probe contract and launcher
4. only then consider `spec14b r1` training
