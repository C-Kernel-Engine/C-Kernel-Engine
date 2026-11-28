# Phonon 31M English CPU baseline and generated codec ingress

The pinned publisher runtime `ptts==0.8.0` streamed the current 2,241-word
technical summary on P3's Intel Core i7-14700T. The Q8_0 run took 60.52 seconds
to emit 1,165.04 seconds of 24 kHz PCM. First PCM arrived in 38 ms after
submitting the text to an already loaded model. The process peaked at 414 MiB
RSS. Closing the test stream after ten chunks took 63 ms, and the same
instance served another request. The repeated short request produced the same
duration and a different PCM hash. The elapsed times measure synthesis and
stream arrival. Headset latency, intelligibility, and lexical coverage still
require listening or a separate transcription evaluation.

Use a fresh output directory. The benchmark records the input hash, time
to first PCM, predicted gaps from chunk arrival times, and a WAV for each
request. It never sends the summary to a speech service.

```bash
RAYON_NUM_THREADS=4 uv run --no-project --with ptts==0.8.0 --with numpy \
  python -m version.v8.tts.benchmark_phonon31m_reference \
  --text /path/to/server_summary.txt --output /tmp/phonon-reference
```

CKE's generated C runs the first codec boundary from 32-channel normalized
latents to 512-channel Mimi features. The circuit composes per-channel
multiplication, broadcast addition, and the 1x1 projection using existing
maps and providers. The caller binds the checkpoint's mean as a constant
input. No compiler changes are needed. The released BF16 mean and scale
widen exactly to FP32, and the released projection is FP32.

With three deterministic input frames and the pinned release weights, CKE
denormalization matched PyTorch exactly. The projection's maximum absolute
error was `7.1525574e-7`. The next boundary is the stateful Mimi decoder from
these features to waveform samples. Generating the supplied latents from
text in CKE remains a separate milestone.

```bash
CK_PHONON31M_CHECKPOINT=/path/to/phonon-31m-en \
  python -m pytest -q -s tests/test_v8_phonon31m_codec_ingress.py
```

The released-weight test verifies the Safetensors SHA-256. Without that asset,
nightly runs the synthetic one- and three-frame cases and reports a skip for
the released-weight case.

[Checkpoint 8f7a37c9](https://huggingface.co/gradium/phonon-31m-en/tree/8f7a37c94a4c58b04c8287300c53f11d2e208eae) ·
[Publisher runtime v0.8.0](https://github.com/gradium-ai/xn-ptts/tree/ece09255bee59e877e42ee50884a38ece8867738)
