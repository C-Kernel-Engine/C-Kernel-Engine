# v8 Capability Cases

`capability_cases.json` records how a model capability reaches executable
evidence. The registration audit verifies that circuits, evidence files,
model catalog entries, Make targets, and required nightly routes remain
reachable. Registration alone does not claim that evidence passed.

The nightly runner maps the Make and Python entry points it actually executes
onto `cke.v8.capability_evidence_report`. Each current-run case is reported as
`pass`, `fail`, `error`, `timeout`, or `not_tested`. A skipped, unavailable, or
unselected case is always `not_tested`; historical evidence cannot satisfy a
current-commit result.

Existing X-Ray, vision, audio, and long-context runners continue to own their
specialized semantics. The shared envelope reports selection and outcome
without replacing their numerical acceptance criteria.
