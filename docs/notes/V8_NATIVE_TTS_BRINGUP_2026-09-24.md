# CKE native TTS bring-up

Status: architecture and interface proposal. No CKE TTS model, speech session,
quality result, or CPU speed is certified by this document.

## Scope and ordering

The first implementation milestone is **phoneme IDs plus one fixed voice to a
generated FP32 waveform**. This isolates model arithmetic from text normalization
and G2P; it is phoneme-to-waveform, not complete native text-to-speech. The first
product milestone is one repeatable path from a local assistant response to audible
PCM. Keep Kokoro as the first generated-C model and fallback candidate;
evaluate Chatterbox Turbo for interaction and Qwen3-TTS 1.7B for capability only
after their exact upstream revisions, preprocessing, and numerical contracts are
recorded. The accessibility application may use an existing local speech runtime
until native Kokoro passes the complete gate. Kokoro implementation can proceed
in parallel with bounded generated-audio optimization and Gemma vision work;
reserve P3 or Ryzen for uncontended numerical and performance certification.

The accessibility design is in `docs/notes/ACCESSIBLE_AGENT_MEMORY_PALACE.md` in
the active accessibility checkout. Its application owns speech queuing, playback,
policy, emergency stop, and sentence navigation. Generated model code owns only
mathematical execution. The host owns text preparation, model state, scheduling,
PCM conversion, and cancellation.

## Current CKE boundary

`include/ck_session_v8.h` describes a text token session. The current
`version/v8/src/ck_cli_v8.c` implementation requires autoregressive decode,
text encode, and token decode capabilities at open. It uses process-global
generation variables and generated-model entry points without a session handle.
Its cancellation flag is checked between text generation steps. Reusing that ABI
unchanged would fail the requirements for independent speech requests, PCM
metadata, explicit model scratch, and bounded output. A speech ABI should be
separate and versioned; it can reuse descriptor/error conventions.

The v8 audio circuit and kernel maps primarily implement inbound audio and ASR:
PCM decode, resampling, STFT, Mel, Conv1D, Conformer, and Whisper encoder/decoder
work. This is useful for input conversion and selected forward primitives, but it
does not constitute a TTS decoder or a waveform oracle. Any shared kernel must
retain the target model's exact layout, padding, rounding, and operation order.
The current tree also has an `audio_lstm_step_f32` map; Kokoro's bidirectional
and stacked LSTM paths still need shape and numerical checks before reuse.
CKE's C implementations belong in `src/kernels`; v8 circuits and kernel maps
describe selection and lowering. TTS reference capture lives separately from
deployed arithmetic.

### DSL fit and compiler boundary

CKE already supports declarative `sequence`, `block_types` with
`header`/`body`/`footer`, component circuits with explicit `stitch` edges,
`graph_slots`, `weight_refs`, required numerical contracts, and circuit-owned
activation buffers. Kokoro's fixed arithmetic graph should be decomposed into
phoneme embedding/ALBERT, duration and alignment, prosody/text encoding,
decoder/generator, and inverse-STFT components. Repeated ALBERT and residual
blocks belong in circuit bodies; fixed voice selection and duration policy
belong in declared operations and the native host, not a `kokoro` branch in the
lowerer. New mathematical providers belong in `src/kernels` with kernel-map
contracts and oracle tests.

Current `build_ir_v8.py` still uses static `OP_DATAFLOW`,
`TEMPLATE_TO_KERNEL_OP`, and `TEMPLATE_OP_WEIGHTS` tables. Its circuit
activation-buffer expressions resolve from configuration at build time; they
cannot by themselves bind a live predicted duration sum. A fixed oracle
utterance can specialize those extents for numerical bring-up. General speech
requests need a generic checked runtime-extent/buffer-binding capability, or
bounded maximum buffers with live lengths passed through ordinary operation
parameters. The compiler must reject an unknown TTS operation or unresolved
extent; adding family-specific branches would hide missing DSL capability.
Ownership of shared codegen and registry changes must be agreed with agents
working in those files before implementation.

## Target architecture inventory

These are **candidate** upstream pins for an import audit, not compatible CKE
artifacts. Recheck licenses for every packaged dependency and voice asset
independently of the model repository's top-level license. The Kokoro oracle
environment still needs resolved wheel, spaCy model, and optional eSpeak data
versions and hashes; the selected voice ID and asset hash are also open.

| Target | Candidate model snapshot and license | Preparation, generation, output | State and missing CKE contracts |
| --- | --- | --- | --- |
| Kokoro v1.0 | [`hexgrad/Kokoro-82M@e8a90b41091c3c5b70375c47cc959799920fa4d6`](https://huggingface.co/hexgrad/Kokoro-82M/tree/e8a90b41091c3c5b70375c47cc959799920fa4d6), Apache 2.0 | Normalization and [Misaki G2P at `fba1236`](https://github.com/hexgrad/misaki/commit/fba1236595f2d2bf21d414ba6e57d25256afada3) produce phoneme IDs. [Kokoro code at `dfb907a`](https://github.com/hexgrad/kokoro/commit/dfb907a02bba8152ca444717ca5d78747ccb4bec) applies ALBERT, duration LSTM and length regulation, prosody and text encoding, then AdaIN/ISTFTNet waveform decoding. One exported voice table can supply fixed style conditioning. | Deterministic whole-utterance synthesis; cap symbols and predicted frames. Need exact LSTM, duration expansion, adaptive/instance norm, upsampling, inverse STFT, waveform and native G2P contracts. Lock resolved G2P packages and data before oracle capture. |
| Chatterbox Turbo | [`ResembleAI/chatterbox-turbo@749d1c1a46eb10492095d68fbcf55691ccf137cd`](https://huggingface.co/ResembleAI/chatterbox-turbo/commit/749d1c1a46eb10492095d68fbcf55691ccf137cd), MIT model/repo; review reused component notices | [Code at `5de7a54`](https://github.com/resemble-ai/chatterbox/commit/5de7a54aa4e5e2baadb0182dde554908b48b85c2) normalizes text, uses the tokenizer files in the model snapshot and stored conditioning, generates T3 speech tokens autoregressively, then S3Gen converts tokens to waveform via conditional flow and HiFTGAN/F0 stages. | Sampling seed, EOS/max-token caps, flow state and vocoder scratch must be explicit. Need model-specific token, flow, F0 and vocoder operations. Reference returns a whole waveform; audio streaming is unverified. |
| Qwen3-TTS 1.7B CustomVoice, 12 Hz | [`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice@0a272c2df2ee9be6b850d1df75bcf673541e523a`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice/tree/0a272c2df2ee9be6b850d1df75bcf673541e523a), Apache 2.0 | [Code at `022e286`](https://github.com/QwenLM/Qwen3-TTS/commit/022e286b98fbec7e1e916cb940cdf532cd9f488e) formats text using the model snapshot's tokenizer files and built-in speaker/language condition, generates codec tokens with talker and code predictor, then calls the separate neural speech tokenizer decoder for waveform. | Sampling, EOS, token and frame caps, code-predictor cache and codec state must be explicit. Need exact codec decoder operations and numerical contracts. Voice cloning applies to a different variant. Upstream input simulation does not establish waveform streaming. |

Existing CKE GEMM, embedding, attention, normalization, and selected audio Conv1D
maps are candidates for reuse after shape and numerical checks. The inbound
FFT-400 power map is not an inverse STFT. Model size does not determine porting
effort or CPU speed; each target must be measured as a full text-to-PCM path.

The first isolated C additions are `src/kernels/kokoro_shape_bounds.c` and
`src/kernels/audio_duration_expand.c`, declared in `include/ckernel_tts.h`.
The planner validates duration sums, alignment extent, decoder and generator
upsampling, inverse-STFT extent, and a conservative output-frame bound before
writing. The duration-expansion primitive accepts caller-owned output storage.
`tests/test_v8_kokoro_shape_bounds.py` compiles and exercises both. Neither
file is a generated Kokoro graph or waveform evidence.

## Proposed native speech session contract

Open a read-only model object once with pinned model artifact and manifest hashes.
Each session has its own generation state and a predeclared workspace. A model
descriptor reports supported voice IDs, language IDs, PCM sample rate, channels,
format, maximum UTF-8 bytes, maximum normalized symbols, maximum audio frames,
maximum synthesis step duration, and whether genuine incremental waveform output
is supported. No Python process participates after export.

The request includes UTF-8 text and byte length, voice ID, supported model-specific
settings, a reproducible seed if sampling exists, and explicit maximum input/output
limits. Unsupported settings fail before generation. Empty text has a defined
completed-with-zero-frames result. Oversized input fails before allocating a
request workspace. Speech rate belongs in the model request only if the pinned
model defines it; playback rate belongs to the application.

The output format for the first Kokoro path is interleaved signed PCM16, one
channel, with the sample rate read from the pinned artifact manifest. The model
math may produce FP32 waveform internally. Conversion to PCM16 is an explicit
native host operation with clipping and nonfinite-sample rejection. Each audio
callback receives a borrowed buffer, frame count, sample rate, channel count,
format, monotonic utterance/frame offsets, and a request ID. The buffer remains
valid only until the callback returns. The application copies or consumes it
before returning; ownership never crosses the ABI silently.

Events are `STARTED`, `AUDIO`, and exactly one terminal event: `COMPLETED`,
`CANCELLED`, or `ERROR`. `STARTED` occurs only after validation and workspace
reservation. `COMPLETED` reports the exact frame count and whether output was
sentence-batched or genuinely incremental. An error includes a stable code and
message. A callback that cannot accept audio returns a backpressure result;
generation blocks only while its bounded queue has capacity, or the request
returns a typed backpressure error. No unbounded buffering or silent dropping is
allowed. The first implementation may choose synchronous callbacks and no host
output queue; application queue limits still apply.

Cancellation is callable from another thread, sets a session-private atomic flag,
and wakes any backpressure wait. Generated execution checks it at bounded model
operation boundaries, including long acoustic and vocoder stages; a single large
noninterruptible forward call must be split before responsiveness is claimed.
The terminal event follows cancellation, and no further audio events follow it.
Playback stop and queued-text removal are separate application actions and must
occur immediately even while generation unwinds. Do not claim a cancellation
latency until it is measured on P3 and Ryzen.

Start with one session and one active request. Before enabling concurrent
sessions, verify generated-model globals and the global thread pool permit
concurrent calls; private scratch alone is insufficient. A later concurrent
implementation may share immutable weights but must never share mutable caches,
scratch, random state, or output buffers. Predeclare and cap every model
workspace, tokenizer buffer, vocoder workspace, and output chunk. No allocator
calls are allowed in generated hot paths. The host may allocate the full bounded
workspace at session creation.

## First Kokoro proof

1. Pin a complete upstream revision set: model weights, config, voices, model
   code, phonemizer and lexicon, tokenizer if any, and dependency licenses. Save
   checksums and a fixture manifest. Include a phoneme fixture so text-front-end
   drift cannot be confused with model arithmetic drift.
2. Export one voice and one short fixed utterance. Save oracle inputs and named
   intermediate tensors at every new numerical operation. Record waveform FP32
   and PCM16 reference hashes, tolerances, and listening sample.
3. Build the phoneme-to-waveform path through a circuit, operation contracts,
   kernel maps, generated C, a native phoneme host, and explicit scratch. Python
   is only an import and oracle tool. Native text preparation is a separate
   work package, and its uncertainty does not block this arithmetic gate.
4. Run the same fixed utterance twice in fresh and persistent sessions. Require
   deterministic outputs under a fixed seed or document model nondeterminism.
   Inspect silence, clipping, nonfinite samples, truncation, and repetitions.
5. After a native frontend is certified, connect `assistant text -> bounded
   sentence queue -> native speech session ->
   PCM player` with immediate queue clear and playback interruption on a new
   request. Buffer incomplete assistant fragments until a sentence boundary or
   explicit final flush; avoid speaking the same fragment twice.

Sentence-by-sentence synthesis is useful for application responsiveness but is
not model streaming. Publish time to first playable audio and chunk-gap results
before labeling any path streaming.

## Work packages and acceptance gates

1. **Artifact and oracle lock:** select one Kokoro voice and utterance, lock the
   model/code/G2P environment and licenses, publish phoneme IDs, style vector,
   named tensors, waveform, and manifest hashes. This gate is currently
   `missing evidence` until captures exist. The style row depends on the raw
   phoneme count including the reference's pad-wrap behavior; pin and test that
   selection, rather than treating the voice as one static vector.
2. **Primitive and circuit bring-up:** implement only the missing operations
   reached by the pinned Kokoro graph, each with shape, stride, padding,
   precision, scratch, and oracle tests. Lower the complete graph to generated
   C and verify every exported weight and operation map. This gate is
   `unsupported` until a generated waveform path runs.
3. **Native text frontend:** determine whether native preprocessing reproduces
   the pinned reference phoneme IDs. A native eSpeak-backed alternative needs
   its own license and pronunciation contract; differing phonemes cannot count
   as Misaki parity. This gate is `missing evidence` until tested.
4. **Speech session and host:** add a versioned native speech ABI with independent
   state, bounded PCM callbacks, and responsive cancellation. Test failed open,
   empty input, buffer limits, backpressure, and repeated requests in one
   session. Verify global model and thread-pool safety before adding concurrent
   sessions. Measure the longest noninterruptible operation; a descriptor cannot
   promise a cancellation bound across CPUs. This gate is `unsupported` until
   the ABI has an implementation and tests.
5. **Accessibility workflow:** connect bounded sentence buffering, native
   synthesis, playback, interruption, and the working fallback. Confirm the
   application can stop output immediately while model cancellation unwinds.
   This gate is `unsupported` until the full text-to-speaker path is exercised.
6. **Evidence and expansion:** run quality and speed certification on P3 and
   Ryzen, then apply the same ABI and reporting to Chatterbox Turbo and
   Qwen3-TTS. Neither model is claimed compatible today.

## Certification and reports

Keep numerical, intelligibility, and listening gates separate. Numerical gates
compare pinned preprocessing, intermediate tensors, FP32 waveform, and PCM16
conversion. Speech fixtures cover ordinary and long text, numbers, dates,
abbreviations, technical terms, CKE/kernel names, file paths, code explanations,
empty and punctuation-only input, unsupported symbols, oversized requests,
cancellation, repeated sessions, and interrupted playback. Listening review
checks pronunciation, omissions, repetitions, and naturalness; ASR transcripts
are diagnostic evidence only.

On both P3 and Ryzen, publish cold load, warm execution, first playable audio,
whole-utterance latency, audio seconds per wall second, chunk gaps, cancellation
latency, CPU use, and peak memory. Mark every result `passed`, `failed`,
`unsupported`, or `missing evidence`. PR CI should run cheap primitive oracle,
PCM conversion, and session state tests. Nightly should run bounded generated
speech; idle-host certification should run long text, listening fixtures, and
performance. Reuse existing report and visualizer conventions rather than
inferring support from a circuit file alone.

## Training lane

Inference does not wait for TTS backward coverage. V8 has a bounded FP32
generated-training certification and training -> checkpoint -> v8 inference
export workflow that selectively reuses v7 training IR/codegen. This does not
certify TTS-specific backward or composition-level training. Once the exact
TTS graph is pinned, inventory every new backward operation. Later work can
certify a small frozen-vocoder acoustic predictor, then generated training ->
checkpoint -> inference export; voice adaptation requires voice consent and
dataset provenance.
