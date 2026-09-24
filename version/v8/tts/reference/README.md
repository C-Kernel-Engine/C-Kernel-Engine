# Kokoro v1.0 reference capture

This directory contains a single-utterance PyTorch oracle for native CKE Kokoro
bring-up. The pinned model snapshot, source revisions, voice, text, seed, and
conditioning rule are in `kokoro_v1_reference.json`. `fixture_manifest.json`
records observed assets, dependency versions, phonemes, token IDs, tensor shapes,
and file hashes from the canonical capture. The full `.npy` and WAV artifacts are
currently in `/tmp/cke-kokoro-v1-af-heart-reference`. The script refuses network
access.

Run with an environment containing the pinned Kokoro and Misaki source checkouts
and their resolved Python dependencies. Supply the three files from the pinned
Hugging Face snapshot at the paths listed in the pin manifest:

```sh
python version/v8/tts/reference/capture_kokoro_v1.py \
  --model-dir /path/to/Kokoro-82M-snapshot \
  --output-dir /tmp/cke-kokoro-v1-af-heart-reference
```

`--preprocess-only` needs just `config.json` and captures segmentation, G2P, and
token IDs. Full capture also emits selected style, ALBERT and duration stages,
F0/noise projections, text encoder, decoder, frame-to-token indices, float32
waveform, final ISTFT magnitude and phase, and a 24 kHz mono PCM16 WAV for listening. The `.npy` waveform is the
numerical oracle; the WAV is a playable derivative. Both are checked for finite
samples. Keep the captured fixture beside its generated manifest and check the
source/dependency pins before comparing CKE output.

The first full capture took 5.51 seconds wall time and peaked at 1,134,048 KiB
RSS on this host with one PyTorch thread. A second process produced identical
tensor and WAV hashes with seed 0. Reconstructing from captured ISTFT magnitude
and phase with PyTorch `istft` (`n_fft=20`, `hop=5`, periodic Hann) reproduced the
float32 waveform exactly. `oracle-requirements.lock` records the 89 installed
package versions and source URLs. It does not contain wheel hashes; the asset
SHA-256 hashes are in `fixture_manifest.json`.

The fixture is upstream reference evidence only. Its `evidence` object leaves
native primitive parity, native full waveform parity, human listening, and
application playback as `NOT_TESTED`. Missing dependencies or assets stop capture
with an error; they never count as passing evidence. Each production kernel needs
its own oracle comparison and CI registration outside this reference directory.

The selected style row is `af_heart[len(phonemes)-1]`; the predictor receives its
last 128 columns and the decoder receives its first 128. The generated manifest
marks any missing hook explicitly.
