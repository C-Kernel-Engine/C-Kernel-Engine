# v7 Python Authoring Examples

These examples sit on top of the current `v7` training pipeline.

What works now:
- [python_authoring_tiny_lm_v7.py](/home/antshiv/Workspace/C-Kernel-Engine/version/v7/examples/python_authoring_tiny_lm_v7.py)
  Python authors the project spec, then hands off to `ck_run_v7.py init/train`.
- [v7_python_authoring_quickstart.ipynb](/home/antshiv/Workspace/C-Kernel-Engine/notebooks/v7_python_authoring_quickstart.ipynb)
  Notebook walkthrough for materialize + train.
- [v7_python_authoring_artifact_walkthrough.ipynb](/home/antshiv/Workspace/C-Kernel-Engine/notebooks/v7_python_authoring_artifact_walkthrough.ipynb)
  Notebook for inspecting the run-dir artifacts and generated-runtime handoff.

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
