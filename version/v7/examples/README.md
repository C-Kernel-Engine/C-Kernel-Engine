# v7 Python Authoring Examples

These examples sit on top of the current `v7` training pipeline.

What works now:
- [python_authoring_tiny_lm_v7.py](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/examples/python_authoring_tiny_lm_v7.py)
  Python authors the project spec, then hands off to `ck_run_v7.py init/train`, refreshes `ir_report.html`, and regenerates `ir_hub.html`. Run-local `dataset_viewer.html` and `attention.json` stay conditional on dataset/tokenizer artifacts.
- [python_module_api_tiny_lm_v7.py](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/examples/python_module_api_tiny_lm_v7.py)
  Builds a tiny `ck.models.qwen3_tiny(...)` transformer graph, compiles it through `ck.v7.compile(...)`, records compile/pass sidecars, and then uses the same existing v7 materialize/train/viewer flow.
- [01_v7_experiment_story_walkthrough.ipynb](/home/antshiv/Workspace/C-Kernel-Engine/notebooks/python_authoring/v7_training/01_v7_experiment_story_walkthrough.ipynb)
  Presenter-oriented walkthrough of the spec02 -> spec19 training arc, with direct links into the numbered notebook lane and live demo commands.
- [02_v7_python_authoring_quickstart.ipynb](/home/antshiv/Workspace/C-Kernel-Engine/notebooks/python_authoring/v7_training/02_v7_python_authoring_quickstart.ipynb)
  Notebook walkthrough for materialize + train + viewer generation, including a run artifact dashboard that links out to the rest of the v7 artifact surface.
- [03_v7_dsl_dataset_preparation.ipynb](/home/antshiv/Workspace/C-Kernel-Engine/notebooks/python_authoring/v7_training/03_v7_dsl_dataset_preparation.ipynb)
  Notebook scaffold for split-aware SVG/DSL dataset prep: workspace inspection, artifact materialization, run-local staging, `dataset_viewer.html`, and training handoff commands.
- [04_v7_python_authoring_artifact_walkthrough.ipynb](/home/antshiv/Workspace/C-Kernel-Engine/notebooks/python_authoring/v7_training/04_v7_python_authoring_artifact_walkthrough.ipynb)
  Notebook for inspecting the run-dir artifacts, generated-runtime handoff, and viewer outputs.
- [05_v7_python_module_api_quickstart.ipynb](/home/antshiv/Workspace/C-Kernel-Engine/notebooks/python_authoring/v7_training/05_v7_python_module_api_quickstart.ipynb)
  Notebook walkthrough for the thin `ck.nn` graph -> `ck.v7.compile(...)` adapter and the same existing v7 viewer surface.
- [v7_training/README.md](/home/antshiv/Workspace/C-Kernel-Engine/notebooks/python_authoring/v7_training/README.md)
  Folder index for the ordered `01 -> 05` notebook flow.

Starting the notebooks:
- From the repo root, run `.venv/bin/jupyter lab notebooks/python_authoring/v7_training/02_v7_python_authoring_quickstart.ipynb`.
- To open the whole ordered folder in JupyterLab, run `.venv/bin/jupyter lab notebooks/python_authoring/v7_training/`.
- If you prefer the classic UI, run `.venv/bin/jupyter notebook notebooks/python_authoring/v7_training/`.
- Open them from inside the repo checkout so repo-root detection works.
- `notebooks/v7_training/` remains available as a compatibility alias for older launch commands.

Example ladder to build next:
- `linear_regression`
  Use existing `linear`/GEMM kernels plus an MSE loss path.
- `polynomial_regression`
  Keep the same linear kernel, build polynomial features in Python.
- `multiclass_classification`
  Use `linear` plus softmax cross-entropy.
- `tiny_mlp_classifier`
  Add one hidden layer and reuse the same train flow.
- `tiny_lm`
  Current working path.
- `data_and_tokenizer_planning`
  Notebook/API for dataset staging, token streams, tokenizer lineage, and run metadata.

Recommended notebook suite:
- `01_v7_experiment_story_walkthrough`
  Start with the story: summarize the experiment ladder, what changed across the specs, and what to demo live.
- `02_v7_python_authoring_quickstart`
  Start a tiny run and point users to the IR visualizer, IR hub, and dataset viewer when available.
- `03_v7_dsl_dataset_preparation`
  Scaffold SVG/DSL dataset staging, manifest inspection, run-local dataset snapshots, and `dataset_viewer.html`.
- `04_v7_python_authoring_artifact_walkthrough`
  Inspect `python_authoring_plan.json`, manifests, IR, layout, codegen outputs, and reports.
- `05_v7_python_module_api_quickstart`
  Build a tiny `ck.models.qwen3_tiny(...)` graph, compile it through the v7 adapter, and inspect the exported graph/config/pass artifacts beside the normal run outputs.
- `v7_training_operator_workbench`
  Planned notebook for parity, sanity, full train, artifact refresh, and train command surfacing.

Current boundary:
- Python is the UI and planning layer.
- `v7` remains the owner of manifest creation, IR lowering, codegen, compiled generated C runtime, and train execution.
- Python can also call the existing viewer tools so notebook users can refresh the IR report, export embeddings, and regenerate the run hub without dropping to shell commands.
- `dataset_viewer.html` requires dataset manifests or a staged dataset workspace, and `attention.json` requires tokenizer/probe artifacts.
