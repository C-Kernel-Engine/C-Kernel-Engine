# V8 Local Inference Server

The v8 server loads one generated CKE runtime and exposes the supported subset
of the OpenAI Responses API. A Chat Completions compatibility route serves
clients such as Qwen Code through the same Responses implementation.

The HTTP layer remains a development server. Current boundaries are:

- stores are process-local and non-durable;
- one generation may use a loaded session at a time;
- there is no authentication or durable request queue;
- the Chat Completions route intentionally rejects options it cannot preserve.

FastAPI owns the current HTTP/session lifecycle. Native model execution remains
behind the C session ABI so a future C or Rust host can reuse the same boundary.

## Native runtime boundary

The stable host boundary is `include/ck_session_v8.h`, implemented by
`build/libck_session_v8.so`. Build it with:

```bash
make ck-session-v8
```

A Python prototype may load that library with `ctypes` or `cffi`; a Rust server
may bind the same C ABI. The host opens one generated model session, then calls
`ck_session_v8_generate`. CKE performs circuit-derived chat formatting, native
tokenization, model execution, generated stop/timestamp policy, and native
detokenization. The callback receives each token ID and its UTF-8 bytes, which
the HTTP layer can translate into response or SSE events.

Sampling values such as `temperature` and `top_p` belong to each request. They
do not select the tokenizer. The generated model declares tokenizer, chat,
stop-token, and modality capabilities at compile time. Session requests reset
KV/recurrent state by default; callers must explicitly set
`CK_SESSION_REQUEST_CONTINUE_STATE` to continue an existing sequence.

The current session ABI deliberately fails closed when a generated model lacks
the required tokenizer or chat capability. It does not infer a tokenizer or
chat template from the model name.

## Qwen Code pilot

The validated pilot uses Qwen Code 0.21.5 and the Qwen3.8 27B Q4_K_M artifact.
Generate at least a 16K runtime because the client's built-in system and tool
catalog can exceed small development contexts:

```bash
CK_NUM_THREADS=16 OMP_NUM_THREADS=1 \
version/v8/scripts/cks-v8-run serve \
  hf://ggml-org/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf \
  --run /path/to/qwen38-agent-runtime \
  --context-len 16384 \
  --force-compile \
  --model-name qwen38-27b-q4km
```

The server reuses cached model bytes and regenerates the candidate runtime. On
later `--no-build` starts it reads the compiled capacity from
`layout_decode.json`; an explicit context larger than that plan is rejected.

Point Qwen Code at the local server and begin with a read-only, bounded task:

```bash
OPENAI_API_KEY=cke-local-only \
OPENAI_BASE_URL=http://127.0.0.1:8080/v1 \
OPENAI_MODEL=qwen38-27b-q4km \
qwen --bare \
  --core-tools read_file \
  --allowed-tools read_file \
  --max-tool-calls 1 \
  --model qwen38-27b-q4km
```

The real-model pilot certifies a two-turn `read_file` workflow. Editing, shell
execution, concurrent sessions, and unattended operation require separate
permission and reliability validation.

Run the schema tests with:

```bash
python3 -m pip install -r server/requirements.txt
make test-server-schema
```

Use `cks-v8-run serve` as the model lifecycle entry point rather than adding
server flags to `ck_chat.py` or `ck_run_v8.py`.
