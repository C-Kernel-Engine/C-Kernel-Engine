# Registered LoRA training jobs

The optional `decoder_lora_fp32_v1` recipe connects the existing v2 job lifecycle
to the public generated LoRA session. It does not open a network endpoint. A
trusted application adapter supplies the authenticated principal; request JSON
cannot choose a principal, filesystem path, command or Python code. Remote
transport authentication remains unimplemented in this repository interface.

The supported base remains a two-layer dense qwen3-template FP32 BUMP with
`rms_eps=1e-5`, 2–64 fixed tokens, embedding width at most 256 and vocabulary at
most 4096. This is not arbitrary downloaded-Qwen fine-tuning, BF16 training or
QLoRA. Registering a package does not numerically certify its model.

## Package and registration

An administrator supplies a package directory containing `package.json` and a
relative base directory with `weights.bump` and `weights_manifest.json`.
The JSON schema is:

```json
{
  "schema": "cke.lora.training-package.v1",
  "base": {
    "directory": "base",
    "manifest_sha256": "<64 lowercase hex digits>",
    "weights_sha256": "<64 lowercase hex digits>",
    "provenance": "pretrained"
  },
  "tokenizer": {"id": "<pinned tokenizer ID>", "sha256": "<tokenizer hash>"},
  "tokens": 4,
  "records": [
    {"id": "train-1", "split": "train", "tokens": [0,1,2,3], "targets": [1,2,3,4]},
    {"id": "validation-1", "split": "validation", "tokens": [0,1,2,5], "targets": [1,2,5,4]}
  ]
}
```

The token IDs above illustrate the format, not meaningful training data. Use
`provenance: synthetic` for generated test weights. The provenance label is an
administrator declaration, not proof of pretraining. Clients must tokenize with
the pinned tokenizer; this recipe does not perform text tokenization. The
package is limited to 8 MiB and 4,096 records. Both splits must be nonempty;
duplicate record IDs and identical token inputs across splits are rejected.
This overlap check is not a semantic data-leakage audit.

The administrator registry maps an opaque package ID to its local path and the
SHA-256 of the package JSON:

```json
{"demo": {"path": "/absolute/package/package.json", "sha256": "<package hash>"}}
```

Start the opt-in local protocol from a clean CKE checkout with a built engine:

```sh
python -m server.training.jobs --lora-registry /absolute/registry.json \
  --store /absolute/private-job-store --principal studio
```

Keep that process behind a trusted adapter; do not expose this stdin/stdout
protocol directly to unauthenticated callers. No running service is changed by
registration or the tests.

## Application sequence

Use `cke.training.jobs.v2` through `server.training.jobs.exchange` or the local
JSON-lines protocol. The existing independent client in
`server/training/bridge_consumer.py` validates replies.

1. **Discover.** Check `executable_recipes` for `decoder_lora_fp32_v1`. Without an
   administrator registration, LoRA remains unsupported. The nested validation
   capability reports `validation_only`; only the job wrapper advertises execution.
   Read registered package identities, split counts and projection targets/ranks.
2. **Prepare.** Pin the returned worker boot and capability hash. Supply an opaque
   32-hex client revision, experiment ID and this configuration:

   ```json
   {
     "recipe": "decoder_lora_fp32_v1", "package_id": "demo",
     "package_sha256": "<package hash>", "dtype": "fp32",
     "optimizer": "adamw", "method": "lora", "steps": 40,
     "learning_rate": 0.0003,
     "adapters": [
       {"target":"layer.0.wq","rank":2,"alpha":4,"name":"query","seed":41},
       {"target":"layer.1.wv","rank":3,"alpha":6,"name":"value","seed":42}
     ]
   }
   ```

   Preparation validates only: it does not initialize weights, compile, reserve
   the execution slot or train. Review the exact returned plan and its hash.
3. **Submit** that hash with an owner-scoped idempotency key and unchanged client
   revision. Changed packages, stale capabilities/boot IDs and expired plans fail.
4. **Observe** bounded progress and terminal status. A restart leaves interrupted
   work `recovery_required`; it never silently resubmits. One job runs at a time.
   Cancellation is explicitly unsupported; a configured wall deadline bounds work.
5. **Retrieve** report, adapter ZIP and complete checkpoint ZIP by opaque artifact
   ID in checked 8-KiB chunks. Verify byte counts and SHA-256. The base is not copied
   into the adapter export. Retain the original package/base identities.

The runner uses `prepare_lora` and the existing native session for all forward,
loss, backward and optimizer arithmetic. The host handles validated records,
job control, reporting and packaging. It checks unchanged frozen weights and
exact held-out loss/logit hashes after both adapter reload and checkpoint reload.
For subsequent local use, extract only the known archive members into a private
directory and call `LoRASession.load_adapters` or `load_checkpoint` with the matching
compiled runtime. Automated checkpoint continuation as a new service job is not
yet advertised.

## Verdicts and tests

Job success means native execution and reload checks passed. It does not require
held-out loss to decrease, and does not mean useful application quality. The report
retains per-record held-out losses/logit hashes before and after training, final
training loss, package identity and adapter plan. Numerical oracle certification
is `NOT_TESTED` for this job, independently of the existing gradient oracle suites.

```sh
CK_NUM_THREADS=4 OMP_NUM_THREADS=1 python -m pytest -q \
  server/tests/test_lora_jobs.py server/tests/test_training_jobs.py \
  server/tests/test_training_bridge.py server/tests/test_training_contract.py \
  tests/test_v8_lora_session.py tests/test_v8_decoder_lora_training.py
python scripts/nightly_runner.py --tests training_job_lifecycle --require-executed
```

The LoRA job test uses a clearly labelled synthetic base and launches the real
child runner. The development test replaces only the worker's clean-tree launch
guard so uncommitted changes can be tested; it does not replace generated training.
A final clean-tree job should additionally exercise the production launch guard.
Missing engine assets produce an explicit skip, never a native PASS.

Remaining completion gates are authenticated remote deployment, a compatible
pretrained base with meaningful held-out data, and broader model/precision
certification. The tested job interface does not remove those limitations.
