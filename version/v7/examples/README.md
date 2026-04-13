# v7 Python Authoring Examples

These examples sit on top of the current `v7` training pipeline.

What works now:
- [python_authoring_tiny_lm_v7.py](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/examples/python_authoring_tiny_lm_v7.py)
  Python authors the project spec, then hands off to `ck_run_v7.py init/train`, refreshes `ir_report.html`, and regenerates `ir_hub.html`. Run-local `dataset_viewer.html` and `attention.json` stay conditional on dataset/tokenizer artifacts.
- [v7_python_authoring_quickstart.ipynb](/home/antshiv/Workspace/C-Kernel-Engine/notebooks/v7_python_authoring_quickstart.ipynb)
  Notebook walkthrough for materialize + train + viewer generation, including a run artifact dashboard that links out to the rest of the v7 artifact surface.
- [v7_python_authoring_artifact_walkthrough.ipynb](/home/antshiv/Workspace/C-Kernel-Engine/notebooks/v7_python_authoring_artifact_walkthrough.ipynb)
  Notebook for inspecting the run-dir artifacts, generated-runtime handoff, and viewer outputs.

Starting the notebooks:
- From the repo root, run `jupyter lab notebooks/v7_python_authoring_quickstart.ipynb`.
- If you prefer the classic UI, run `jupyter notebook notebooks/`.
- Open them from inside the repo checkout so repo-root detection works.

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

Current boundary:
- Python is the UI and planning layer.
- `v7` remains the owner of manifest creation, IR lowering, codegen, compiled generated C runtime, and train execution.
- Python can also call the existing viewer tools so notebook users can refresh the IR report, export embeddings, and regenerate the run hub without dropping to shell commands.
- `dataset_viewer.html` requires dataset manifests or a staged dataset workspace, and `attention.json` requires tokenizer/probe artifacts.
