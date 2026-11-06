# Kokoro → Whisper long-reply diagnostic (2026-11-06)

This run used CKE-generated Kokoro audio and CKE-generated Whisper base. It did
not use LinuxUtilities or a reference model to synthesize or transcribe audio.
The input was the committed
`version/v8/test_assets/kokoro_long_reply/technical_summary_20261106.txt`,
a varied 1,133-word technical reply beginning with the then-current 264-word
`server_summary.txt` snapshot. Its raw SHA-256 was
`0bbca9d04ece5e012929adcd8093f6e8117e2c9e93a6489e54b0c4e65eb4bdf7`.

| Boundary | Result |
| --- | --- |
| Pinned Kokoro text planning | 316 checked chunks, 40 IDs or fewer; exact source-span coverage |
| Generated Kokoro WAV | 16,027,800 PCM16 mono samples at 24 kHz; 667.825 seconds; SHA-256 `6ee6c534329b01d42a40e1f72248d70701a3bd26159c44ba561d69ae5daa5483` |
| TTS wall time | 1,608.12 seconds; 2.41 seconds synthesis per audio second; first chunk 9.33 seconds |
| Native CKE resampling for Whisper | `audio_resample_windowed_sinc_f32`, 16 kHz, same duration; SHA-256 `69170ace0476624ee33e4085b6d10dee66ebdbb98b13a891b722e7d1de1fd6b7` |
| Generated Whisper, fixed windows | 23 windows; 766 transcript words; normalized WER 37.44% |
| Generated Whisper, timestamp seek | 25 contiguous windows covering all 667.825 seconds; 1,150 transcript words; normalized WER 6.33%; CER 2.41% |
| Exact numeric facts | FAIL: one occurrence of `1035` and two of `3.8` were absent from the timestamp-mode transcript; the transcript also introduced `-1035`, two `40`s, and other unmatched numbers |
| Kokoro connected waveform parity | Existing FAIL remains unresolved; same-input oracle comparison not run here |
| Human listening | NOT_TESTED |

The 30–60-second fixed window returned five words despite audible-energy
features throughout. The same isolated window produced five words with the
pinned PyTorch Whisper base reference. Moving the window to 40–70 seconds
recovered more speech; CKE timestamp seek recovered 45 words from 30–60 seconds.
This points to the long-audio window policy rather than a CKE-only decoder-state
fault. It does not prove that every word was spoken correctly.

The evaluator `version/v8/tts/evaluate_kokoro_whisper_long_reply.py` checks
source spans, WAV and runtime identities, resampling linkage, complete timestamp
window coverage, normalized WER, and exact numeric tokens. It exits nonzero
when WER or numeric fidelity fails. A good WER cannot override a critical-number
failure. Synthetic rejection controls run in standard nightly; the full
asset-backed experiment remains a retained diagnostic, not a nightly PASS.

On the P3 host, the source, WAV, both Whisper reports, and evaluator output are
retained under `/tmp/cke-kokoro-varied-1000-20261106*`,
`/tmp/cke-whisper-varied-1000-20261106*`, and
`/tmp/cke-kokoro-whisper-long-reply-evaluation-20261106.json`.
The WAV and source hashes above allow the pair to be checked after copying.

The next acceptance work is exact technical-fact pronunciation and a supervised
headset listening trial. Long-form timestamp decoding should be used for
Whisper-based diagnosis. Neither ASR accuracy nor the valid WAV certifies
Kokoro's independent numerical parity or comfortable speech.
