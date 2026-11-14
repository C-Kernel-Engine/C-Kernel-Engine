# Qwen3-TTS shared checked-bundle compilation

The pinned native speech session now uses
`version/v8/scripts/compile_checked_bundle_v8.py` for ordinary IR construction,
lowering, memory planning, code generation and compilation. Its model-specific
authoring still declares the circuit and weight geometry. Neural source files are
derived from the resolved provider maps, replacing the session's independent list.

The two explicitly declared support sources are generic native frame-loop and
frame-to-codec control shells under `src/runtime`. They are not extra neural
providers. The shared builder rejects support files outside that directory and
exports every declared segment plus explicitly requested runtime functions.

The session retains GNU11, OpenMP and fast floating-point contraction, matching
its prior compiler policy. Other shared-builder consumers retain their strict
defaults. Generated C always passes a separate ISO C11 syntax check with warnings
as errors. Legacy provider warnings are recorded in `compiler-diagnostics.txt`;
they are no longer hidden by `-w`. This does not claim those warnings are fixed.

Each build records compiler command/version, selected provider hashes, source
hashes, IR/layout hashes, generated source and library hashes, exports and build
policy in `build-identity.json`. Session metrics include that file's hash.
The identity explicitly leaves embedded runtime build identity NOT_VERIFIED.

## Reproduction

```sh
OMP_NUM_THREADS=1 CK_NUM_THREADS=4 python -m unittest \
  tests.test_v8_checked_bundle_policy \
  tests.test_v8_checked_multi_entry \
  tests.test_v8_qwen3_tts_loop_codec \
  tests.test_v8_qwen3_tts_pinned_session \
  tests.test_v8_clef_flash_complete_head \
  tests.test_v8_clef_flash_head_ingress
```

The synthetic connected loop uses the production build path and control entry.
It forces EOS after two frames, checks 128 valid samples and untouched padding,
and compares the complete valid waveform exactly with a separately compiled
codec. Other loop tests retain the previous compiler as an independent regression
path. This is generated execution evidence, not natural EOS or intelligible
speech evidence.

Full pinned-model synthesis requires `CKE_QWEN3_TTS_MODEL_DIR`. Missing assets
remain SKIP. This cleanup does not recertify full-model numerical parity, listening,
speed, or Kokoro's unresolved connected waveform mismatch. It does not remove
development-time model conversion dependencies or change text preparation.
