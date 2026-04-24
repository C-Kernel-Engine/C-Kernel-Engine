# v7 Training Notebooks

This ordered lane now lives under `notebooks/python_authoring/v7_training/`.
Use `notebooks/v7_training/` only as a compatibility alias for older docs and launch commands.


Use this folder as the ordered notebook lane for demos, onboarding, and live walkthroughs.

Recommended reading order:

1. `01_v7_experiment_story_walkthrough.ipynb`
   Summary of the spec02 -> spec19 arc, why each phase mattered, and what the current training stack can do.
2. `02_v7_python_authoring_quickstart.ipynb`
   Smallest end-to-end Python authoring run: materialize, train, and open the viewer/dashboard surface.
3. `03_v7_dsl_dataset_preparation.ipynb`
   Dataset workspace preparation for the SVG/DSL path: manifests, staging, snapshots, and `dataset_viewer.html`.
4. `04_v7_python_authoring_artifact_walkthrough.ipynb`
   Deep inspection of the run-dir artifacts, IR/codegen boundary, and viewer outputs.
5. `05_v7_python_module_api_quickstart.ipynb`
   Thin `ck.nn` module graph -> `ck.v7.compile(...)` adapter -> materialize/train/viewer flow on the same existing v7 pipeline.

Launch examples from the repo root:

```bash
.venv/bin/jupyter lab notebooks/python_authoring/v7_training/01_v7_experiment_story_walkthrough.ipynb
.venv/bin/jupyter lab notebooks/python_authoring/v7_training/02_v7_python_authoring_quickstart.ipynb
.venv/bin/jupyter lab notebooks/python_authoring/v7_training/03_v7_dsl_dataset_preparation.ipynb
.venv/bin/jupyter lab notebooks/python_authoring/v7_training/04_v7_python_authoring_artifact_walkthrough.ipynb
.venv/bin/jupyter lab notebooks/python_authoring/v7_training/05_v7_python_module_api_quickstart.ipynb
```

If you want to browse them as a set:

```bash
.venv/bin/jupyter lab notebooks/python_authoring/v7_training/
.venv/bin/jupyter notebook notebooks/python_authoring/v7_training/
```
