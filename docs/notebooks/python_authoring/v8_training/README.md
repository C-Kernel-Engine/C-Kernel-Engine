# v8 Generated-C Training Notebook

`01_v8_generated_training_quickstart.ipynb` is the first supported v8 Python
authoring path. It constructs a reduced Qwen3 dense/GQA model through the shared
`ck.nn` surface and hands one experiment configuration to the existing v8
generated-C training workflow.

Python owns experiment authoring and presentation. CKE semantic lowering,
kernel maps, generated forward/backward execution, AdamW, checkpointing, and
inference export remain the execution path. There is no Python or PyTorch model
fallback.

The notebook defaults to preflight-only inspection. Set `EXECUTE = True` in the
execution cell to run the bounded frozen-English workflow. CI exercises the same
authoring and preflight path with:

```bash
make v8-training-python-authoring-smoke
```

Current support is deliberately narrow: FP32, 4/6/10-layer reduced Qwen3-style
dense/GQA models, byte or CKE BPE tokenization, and generated-C AdamW. Qwen3.5
DeltaNet modules will be exposed only after their composition-level training
contracts pass. RWKV is outside this workstream. The workflow certifies the
independently generated v8 C inference runtime; a standalone native executable
is still reported as `NOT_CERTIFIED` by the underlying workflow.
