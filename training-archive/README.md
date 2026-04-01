# Training Archive Mount

This directory is the intended mount point for a future git submodule that holds bulky training artifacts outside the main repo.

Use the split this way:

- Stable operator and method docs stay in `docs/site/_pages/`.
- Compact generated summaries stay in `version/v7/reports/spec_training_manifest.json`.
- Heavy raw evidence can move here:
  - run directories
  - probe report HTML/JSON
  - tested-prompt galleries
  - smoke outputs
  - large model-side artifacts that are useful for audit but noisy in the main repo

The docs site should depend on the compact manifest, not on browsing this archive directly.

Suggested future layout:

- `training-archive/version/v7/runs/`
- `training-archive/version/v7/reports/`
- `training-archive/version/v7/probes/`

Once a dedicated archive repository exists, replace this directory with a submodule mounted at the same path.
