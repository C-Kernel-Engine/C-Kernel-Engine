# spec04 Contract

Dataset type: `svg`

## Goal

Render structured, valid SVG that follows a closed control contract.

## Hard rules

- Define one canonical row format for this workspace.
- Do not mix legacy and new conditioning styles.
- Keep eval probes aligned with the same contract used in training.
- Track normalization, dedupe, split policy, and optional holdout/canary policy in `manifests/`.

## TODO

- write the canonical row format
- define placeholder policy
- define dedupe policy
- define train/dev/test split policy
- define holdout/canary policy
- define promotion gates