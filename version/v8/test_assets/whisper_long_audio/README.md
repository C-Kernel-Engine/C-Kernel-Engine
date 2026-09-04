# Whisper long-audio regression corpus

`kernel2_mic1_5min.flac` is the first five minutes of Anthony Shivakumar's
September 4, 2026 Kernel Engineering Part 2 recording. It was captured from
the final MIC1 track, converted to 16 kHz mono PCM, trimmed at exactly 300
seconds, and stored losslessly as FLAC. The repository owner publishes this
recording for CKE regression testing.

The checked-in transcript is human-corrected reference text. Certification is
semantic rather than token-exact because Whisper sizes can make different but
valid punctuation and tokenization choices. `corpus.json` pins the audio and
PCM hashes, model checkpoints, resource tiers, and acceptance thresholds.

Materialize the input accepted by CKE with:

```bash
ffmpeg -hide_banner -loglevel error -y \
  -i version/v8/test_assets/whisper_long_audio/kernel2_mic1_5min.flac \
  -ar 16000 -ac 1 -c:a pcm_s16le build/whisper-long-audio.wav
```

All five sizes currently run as independent hosted-runner jobs. Medium and
Large-v3 may expose a hosted-runner resource limit, but they must not be
silently skipped: each job either publishes a passing report or fails.
