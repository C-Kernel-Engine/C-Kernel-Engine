This version builds on everything we have done from v1-v5.
v1 got fp32 working with pytorch and hugging face safe tensor weights
most of teh code and ir in v1 was explictly written but worked. The kernels were tested robustly and that is the essense of v1. get the kernels to work and have full aprity with pytorch for both inference and back prop.
v2-v5 builds on that. but we wanted a way to create or read either the hugging face config.json file and generate I.R and then generate explcit C code with full memory planning since this is all deterministic. 
We also started to study the gguff file format and quantization and realized how much more powerful the c-kernel-engine could be with quantiszation support. so we built kernel that had the quantization algorithim and wrote test scritps to test agaisnt llama.cpp. This was the main power of v5. and it all worked.
But what is incomplete is this whole piepleline from reading either a gguff file or a config.jos + safe tensor, creating the I.R and generating clean explict code for prefill, decode and backprop if we are training. 
creating a ck-cli to then loads either gguff weights or safetensor convert them to our bump allocator with the tokenizer and load them all in contigious memory
and then run inference or backprop. 

the main goal with what we have is to tie up dense models like devstral, qwen, llama family of models, smolm by giving it a gguff or safe tensor file, and the codegen, not using teh v5 oor ck orchsestrator anymore generate the right files with full memoory planning all in one contigious memory allocator.
we then compile this generated file into a model-specific binary (ck-engine-<model>) and run it.

All generated files can be stored liek we did in v4 and v5 to ~/.cache/ck-engine-v6/models/*

The script ck_run_v6.py or something when we give it a hf link it downlaods and create the stores that model in the above path. v6 shoudl copy all the good parts but now start to ahve all this in c. 
v5 used the scripts/convert_gguf_to_bump.py and we shoudl copy this as v6 and make adjsutments if need be.

Anything we reuse or copy from previosu versions shoudl be copied with the postfix _v6 so that we don't break what v5 or the previous version does. 

What is also not copied are the kernel themselves. They should work acorss all versions unless we change the api. in that case create a new file or new function and expand with proper unit test for performance and numerical aprity with both pytorch and llama.cpp.

this it the goal of this repository. 

Generated code naming + build flow
- Generated files should use fixed names: ck-kernel-prefill.c, ck-kernel-inference.c, ck-kernel-backprop.c (and matching .h/.o/.so).
- Each model gets its own compiled binary: ck-engine-<model> (e.g., ck-engine-qwen2-0.5b)
- NO dlopen: the generated C code is compiled directly into a model-specific binary
- The binary is self-contained: all generated code + kernel calls are linked in
- This makes deployment easy: copy the binary + weights.bump to any system (embedded, server, etc.)
- Kernels are NOT duplicated; they live in src/kernels/ and are linked at compile time
- Build flow per model:
  1. ~/.cache/ck-engine-v6/models/<model>/
  2. Download GGUF/Safetensors → convert weights.bump + tokenizer.json
  3. Generate IR → ck-kernel-prefill.c, ck-kernel-inference.c
  4. Compile: ck-engine-<model> (links generated code + kernels)
  5. Run: ./ck-engine-<model> weights.bump

Cache metadata + invalidation
- Each model cache folder should include a small metadata file (model.json or build.json) with: model ID, revision/hash, config hash, weights hash, tokenizer hash, ABI/version, compiler flags, and build timestamp.
- Use a lock file during conversion/build to avoid partial outputs when two processes run in the same model folder.
- Rebuild if any hash/ABI changes.

C-Kernel format (*.ck)
- Tokenizer must come from the GGUF header or HF/safetensors and be stored in the C-Kernel format file.
- All weights can be stored as *.ck to mark the C-Kernel format explicitly (weights.ck).
- The .ck file is self-contained: it stores all weights, tokenizer, context length, offsets, and alignment so generated code can map it directly into one bump allocator arena.
- The .ck format can be larger than GGUF because it stores cache-aligned tensors and a contiguous layout for fast runtime access.
- The .ck file should include per-layer quantization metadata and a kernel requirements list (copied from GGUF when possible).
- The .ck file can include training metadata (optimizer, batch plan, memory plan hints, CPU feature requirements, etc.) to support backprop and scheduling.
- GGUF -> .ck and safetensors -> .ck converters should compute offsets and alignment up front so IR/codegen can be explicit about memory.

Training (v7 target)
- Same philosophy as inference: explicit codegen, no runtime dispatch
- Generated files: ck-kernel-prefill.c, ck-kernel-inference.c, ck-kernel-backprop.c
- Binary: ck-engine-<model> --train (includes forward + backward + optimizer)
- Optimizer support: Adam (m, v buffers), SGD (momentum buffer)
- Memory planning: all gradients + optimizer states pre-allocated at compile time
- No autograd overhead at runtime - all gradients computed explicitly
- Training flow:
  1. ck-cli-v6 --train <model.gguf>
  2. Generate: ck-kernel-backprop.c (layer-by-layer gradient kernels)
  3. Generate: optimizer kernel (Adam/SGD state updates)
  4. Compile: ck-engine-<model> --train
  5. Run: ./ck-engine-<model> --train weights.bump --data <dataset>

Version bootstrap notes (for v7+)
- Start by copying working pieces into new folders: scripts/vX and src/vX.
- Copy these from the prior version: build script, download script, codegen script, minimal CLI, and any manifest/cache helpers.
- Do not duplicate kernels across versions; they already have extensive unit tests, and duplication adds little value. Versions should iterate on stitching/IR/codegen, not the core compute kernels (unless an API change forces it; then add parity/perf tests).
- Do not copy generated artifacts (generated_*.c/.h/.o, layout json, schedules) into the repo; keep them in ~/.cache/ck-engine-vX/models/*.
- Do not hardcode model names in the inference binary; generate a new binary per model with model-specific function names.
- Keep older versions isolated: never modify v5 paths; always suffix new tools with _v6.
- Make sure the new version can run end-to-end from cache (download -> convert -> IR -> codegen -> build -> run).
