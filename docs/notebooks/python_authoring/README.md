# Python Authoring Notebooks

This folder groups the notebook lanes that sit on top of the existing Python authoring and API surface.

Current lanes:

- `v8_training/`
  Generated-C v8 authoring lane. The notebook reuses `ck.nn`, performs kernel-map
  capability preflight, and launches the certified v8 training-to-inference workflow.
- `v7_training/`
  Ordered v7 notebook lane for training demos, onboarding, dataset preparation, and artifact inspection.

Primary launch path:

```bash
.venv/bin/jupyter lab docs/notebooks/python_authoring/v8_training/
```

The non-interactive notebook companion used by CI is:

```bash
make v8-training-python-authoring-smoke
```

The historical v7 lane remains available at:

```bash
.venv/bin/jupyter lab docs/notebooks/python_authoring/v7_training/
```

Compatibility note:

- `docs/notebooks/v7_training/` remains available as a compatibility alias so existing commands, demos, and links do not break while the notebook surface is being reorganized.
