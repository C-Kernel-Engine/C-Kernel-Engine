# TTS kernel numerical coverage, 2026-09-24

This report concerns two scalar inference providers and a Kokoro-specific extent
preflight. It does **not** certify a generated Kokoro waveform or native text-to-speech.
The provider maps declare backward and training unsupported.

| Provider | Dtype / direction | Independent oracle | Executable test | PR / nightly registration | Observed result |
| --- | --- | --- | --- | --- | --- |
| `audio_duration_expand_channel_major_f32` | FP32 + int32 durations / forward only | PyTorch `torch.repeat_interleave`; committed PyTorch fixture | `tests/test_v8_audio_duration_expand_oracle.py`; invalid extent and stride checks in `tests/test_v8_kokoro_shape_bounds.py` | `.github/workflows/tts-kernels.yml`; `scripts/nightly_runner.py` key `tts_duration_expand_oracle` | PASS: exact values on the fixed fixture, untouched padding, repeated call, and capacity rejection. Live PyTorch fixture check PASS in the local CKE venv. |
| `audio_istft_mag_phase_f32` | FP32 / forward only | PyTorch `torch.istft` with centered periodic Hann; pinned Kokoro magnitude/phase capture | `tests/test_v8_audio_istft_oracle.py`; `version/v8/scripts/compare_kokoro_istft_native_v8.py` | `.github/workflows/tts-kernels.yml`; `scripts/nightly_runner.py` key `tts_istft_oracle` | PASS: synthetic PyTorch fixture and live oracle locally. Pinned Kokoro primitive capture: 61,800 samples, max absolute error 8.4564e-7, mean absolute error 7.2885e-8. |
| `kokoro_shape_bounds` host preflight | Size arithmetic / inference | Independent mathematical boundary cases | `tests/test_v8_kokoro_shape_bounds.c` via `tests/test_v8_kokoro_shape_bounds.py` | `.github/workflows/tts-kernels.yml`; `scripts/nightly_runner.py` key `tts_kokoro_shape_bounds` | PASS: checked limits and overflow cases locally. |
| Generated producer → length validation → consumer | FP32 / forward | Synthetic independent reference | Pending PR A | NOT_TESTED | No generated graph exists yet. |
| Generated Kokoro phoneme → waveform | FP32 / forward | Pinned Kokoro reference capture | Pending PR B | NOT_TESTED | No generated waveform exists yet. |
| TTS backward | FP32 / backward | PyTorch autograd and finite differences | Pending training work | NOT_TESTED | Inference-only provider maps. |

The committed fixtures make cheap PR tests independent of a PyTorch installation.
The nightly runner runs live PyTorch checks when available; if the dependency is
missing, those individual checks are SKIP with a reason. A skipped oracle is not
evidence of parity. The scalar providers have no multithread execution mode, so
single-versus-multithread numerical comparison is not applicable. Performance and
P3/Ryzen certification have not been run.

Reproduce the cheap PR checks:

```sh
python3 -m unittest \
  tests.test_v8_kokoro_shape_bounds \
  tests.test_v8_audio_duration_expand_oracle.AudioDurationExpandOracleTest.test_committed_torch_fixture \
  tests.test_v8_audio_istft_oracle.AudioIstftOracleTest.test_committed_torch_fixture
```

Reproduce live PyTorch checks in an environment with PyTorch and NumPy:

```sh
python3 -m unittest tests.test_v8_audio_duration_expand_oracle tests.test_v8_audio_istft_oracle
```

The pinned-capture comparison requires the reference artifact path described in
`version/v8/tts/reference/README.md`.
