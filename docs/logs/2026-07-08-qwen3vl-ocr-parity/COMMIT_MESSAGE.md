fix(v8): tighten qwen3-vl vision position parity

Summary:
- Match llama.cpp-style float evaluation order in Qwen3-VL tiled position embedding interpolation.
- Keep strict attention oracle score dumps from perturbing the softmax input graph.
- Add a Qwen3-VL OCR parity handoff note with commands, results, and next diagnostic target.

Validation:
- Qwen3-VL bad OCR image `1_81.ppm`: `patch_bias` and `inp_pos_emb` now match llama.cpp exactly.
- Layer-0 Q/K/V and RoPE are within ~1e-6.
- Layer-0 `kqv_out` is within ~1.19e-6.

Remaining:
- Layer-0 q8 out-proj still amplifies tiny `kqv_out` drift to about 5e-4.
- Next work should inspect Q8 activation quantization / q8_0 x q8_0 out-proj parity before rerunning the full 40-image OCR sweep.
