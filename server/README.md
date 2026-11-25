# V8 Local Inference Server

The v8 server loads one generated CKE runtime and exposes the supported subset
of the OpenAI Responses API. A Chat Completions compatibility route serves
clients such as Qwen Code through the same Responses implementation.

The HTTP layer remains a development server. Current boundaries are:

- stores are process-local and non-durable;
- ordinary serving admits one generation at a time; an experimental opt-in
  two-slot path is available only for generated KV-only batch-capable models;
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
`ck_session_v8_generate`. For the live server, the selected Jinja sidecar
renders the conversation and the session consumes that raw prompt. CKE then
performs native tokenization, model execution, generated stop policy, and
native detokenization. The callback receives each token ID and its UTF-8 bytes, which
the HTTP layer can translate into response or SSE events.

Sampling values such as `temperature` and `top_p` belong to each request. They
do not select the tokenizer. The generated model declares tokenizer, chat,
stop-token, and modality capabilities at compile time. Session requests reset
KV/recurrent state by default; callers must explicitly set
`CK_SESSION_REQUEST_CONTINUE_STATE` to continue an existing sequence.

The current session ABI deliberately fails closed when a generated model lacks
the required tokenizer or chat capability. It does not infer a tokenizer or
chat template from the model name.

## Experimental two-request HTTP decode

The opt-in flags, delivery limits, failure behavior and certification boundary
are documented in the
[v8 serving runbook](https://c-kernel-engine.github.io/C-Kernel-Engine/v8-runbook.html#experimental-batch-http).

## Qwen Code profiles

The historical validated pilot used Qwen Code 0.21.5 and the Qwen3.8 27B Q4_K_M artifact.
CKE provides separate settings profiles for short interactive work and large,
unattended artifact generation. They are loaded as Qwen Code system settings
and do not overwrite `~/.qwen/settings.json`.

Generate the runtime capacity required by the selected profile. This example is
the 16K interactive runtime:

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
Normal serving requires the converted `chat_template.jinja` before opening the
native session. Intentional untemplated serving requires both
`--no-chat-template` and `--allow-raw-prompt`; explicit serving overrides use
`--chat-template-file` or `--chat-template-inline`.
The native template's `qwen_xml` protocol recognizes
`<tool_call><function=...><parameter=...>` output. The Qwen Code compatibility
variant declares `qwen_code_xml` for the two XML envelopes observed in actual
Qwen Code runs. Other bundles must declare their actual tool protocol; a
template mentioning tools does not enable tool calls automatically. Supported
choices are `tagged_json`, `qwen_xml`, `qwen_code_xml`, and `bare_json`.
A `tool_protocol.json` sidecar records the choice with schema
`cke.v8.tool_protocol.v1`, `protocol`, and the SHA-256 of the selected
`chat_template.jinja` (or selected `additional_chat_templates/tool_use.jinja`)
as `template_sha256`. A mismatched sidecar fails startup. Serving a converted
bundle needs its BUMP, generated libraries, tokenizer, and template sidecars;
the source GGUF is not needed for `--no-build`.

For the pinned Qwen3.8 tool workload, see the
[v8 serving runbook](https://c-kernel-engine.github.io/C-Kernel-Engine/v8-runbook.html#converted-jinja-serving)
for the dedicated bundle setup, hash-bound Jinja variant, commands, nightly
gates, and remaining native-tokenizer mismatch.

Tool output is parsed only under the selected protocol. A template render
error returns an explicit failure instead of silently switching prompt format.
Tool-bearing streamed responses buffer model text until it can be validated;
the connection receives keep-alives while generation runs. This favors tool
correctness over live token display for these requests. Thinking deltas still
stream live while tools are attached.

User-facing serving instructions and modality limits are maintained in the
[v8 HTML runbook](../docs/site/_pages/v8-runbook.html#serve).

Point Qwen Code at the local server and begin with a read-only, bounded task:

```bash
CKE_ROOT=$(pwd)
OPENAI_API_KEY=cke-local-only \
OPENAI_BASE_URL=http://127.0.0.1:8080/v1 \
OPENAI_MODEL=qwen38-27b-q4km \
QWEN_CODE_MAX_OUTPUT_TOKENS=2048 \
QWEN_CODE_SYSTEM_SETTINGS_PATH="$CKE_ROOT/server/qwen-code/interactive.settings.json" \
qwen --bare \
  --auth-type openai-responses \
  --system-prompt 'Use read_file exactly once when asked, then answer without another tool call.' \
  --allowed-tools read_file \
  --exclude-tools edit,notebook_edit,run_shell_command,get_goal,update_goal \
  --max-tool-calls 1 \
  --model qwen38-27b-q4km
```

Qwen Code bare mode ignores `--core-tools`, so the command explicitly
removes the other bare-mode tools. The interactive profile declares 16,384
context tokens, a 2,048-token output allowance, and a 30-minute wall deadline.
The cached real-model pilot completed a server-issued `read_file` call and a
separate bounded `read_file` → `edit` task. A `write_file` request failed because
that tool was absent from Qwen Code's advertised tool set; the server rejected
the unknown call. Shell execution, file creation, and concurrent sessions
require separate permission and reliability validation.

For a generated runtime with 262,144-token capacity, select the overnight
profile instead:

```bash
CKE_ROOT=$(pwd)
OPENAI_API_KEY=cke-local-only \
OPENAI_BASE_URL=http://127.0.0.1:8080/v1 \
OPENAI_MODEL=qwen38-27b-q4km \
QWEN_CODE_MAX_OUTPUT_TOKENS=32768 \
QWEN_CODE_SYSTEM_SETTINGS_PATH="$CKE_ROOT/server/qwen-code/overnight.settings.json" \
qwen --bare \
  --allowed-tools read_file \
  --exclude-tools edit,notebook_edit,run_shell_command,get_goal,update_goal \
  --max-wall-time 18h \
  --model qwen38-27b-q4km \
  --output-format stream-json \
  --prompt "$(cat /path/to/reviewed-task.txt)" \
  > /path/to/task-events.jsonl
```

The overnight profile declares 262,144 context tokens and reserves up to 32,768
tokens for output. It does not provide mid-generation resume, a durable server
queue, or permission to publish, commit, or deploy results. Use an external
task ledger to retry whole tasks after interruption and retain outputs for
review.
On Qwen Code 0.24.6, the explicit output-cap environment variable is needed
for this local endpoint: the older settings profile alone did not constrain
the request in the bounded probe. Confirm the effective prompt and output
allowance against the generated runtime before an overnight task.

`GET /v1/models/{model}` reports `cke_context_length` and
`cke_default_max_output_tokens`. Before native execution, the server tokenizes
the fully rendered request and rejects prompt plus output reservations that
exceed the loaded runtime capacity. The Qwen Code profile must not advertise a
larger context than the generated runtime.

Chat Completions responses include a CKE extension named `cke_performance`.
For streaming requests it appears on the terminal chunk. The extension retains
native prompt/output token counts and prefill/decode timings, then adds
`request_total_ms` and `non_native_ms` for the complete server request. Client
tool execution occurs between requests and is not included, so measure that
interval separately when profiling an agent task.

Run the schema tests with:

```bash
python3 -m pip install -r server/requirements.txt
make test-server-schema
```

`make test-v8-serve-native-jinja` checks the pinned Jinja request and
continuation contract. Its scope and the model-backed nightly gate are
explained in the v8 serving runbook linked above.

Use `cks-v8-run serve` as the model lifecycle entry point rather than adding
server flags to `ck_chat.py` or `ck_run_v8.py`.
