"""Bounded application-facing LoRA sessions using the shared generated backend.

No neural arithmetic, gradient calculation or optimizer update runs in Python.
Base weights and adapter initialization are artifact preparation, before codegen.
"""
from dataclasses import dataclass, asdict
import ctypes as C
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import threading

import numpy as np
from .lora import attach_lora, attach_lora_adapters
from .lora_weights import prepare_lora_weights, resolve_lora_parameters

ROOT = Path(__file__).resolve().parents[3]
for folder in (ROOT / 'version/v7/scripts', ROOT / 'version/v8/scripts'):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
import build_ir_train_v7 as author
import lower_ir2_backward_v7 as backward
import generate_train_layout_v7 as planner
import codegen_train_runtime_v7 as codegen
import run_training_workflow_v8 as workflow
from version.v8.scripts.lower_circuit_training_v8 import training_registry


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class LoRAConfig:
    target: str
    rank: int
    alpha: float
    name: str
    seed: int


def lora_capabilities():
    return dict(schema='cke.v8.lora_capabilities.v1', status='bounded',
                base_artifact='FP32 BUMP plus weights_manifest.json',
                topology='two-layer dense qwen3 template', tokens={'min': 2, 'max': 64, 'binding': 'compile_time'},
                precision={'base': 'fp32', 'adapters': 'fp32', 'gradients': 'fp32'},
                optimizer='generated_native_adamw', gradient_accumulation=1,
                normalization={'rms_epsilon': 1e-5, 'qk_epsilon': 1e-5},
                target_selection='explicit unique projection names', target_discovery=True,
                max_adapter_parameters=1000000,
                python_neural_execution=False, arbitrary_pretrained_import=False,
                execution='local_compiled_session', remote_jobs='unsupported',
                bf16=False, qlora=False)


def _base_ir(base, tokens):
    base = Path(base).resolve()
    manifest = read(base / 'weights_manifest.json')
    config = dict(manifest['config'])
    require(type(tokens) is int and 2 <= tokens <= 64, 'tokens must be in [2, 64]')
    require(config.get('model') == 'qwen3' and config.get('num_layers') == 2,
            'supported base: two-layer dense qwen3 template')
    require(config.get('dtype') == 'fp32' and config.get('num_heads') == config.get('num_kv_heads'),
            'supported base: FP32 dense attention; BF16 and QLoRA are not supported')
    require(config.get('rms_eps') == 1e-5,
            'current LoRA session certification requires explicit rms_eps=1e-5')
    require(config.get('context_len', 0) >= tokens, 'tokens exceed base context capacity')
    require(type(config.get('embed_dim')) is int and 0 < config['embed_dim'] <= 256
            and type(config.get('vocab_size')) is int and 0 < config['vocab_size'] <= 4096,
            'base exceeds certified small-decoder geometry')
    require((base / 'weights.bump').stat().st_size <= 256 * 1024**2, 'base weight file exceeds 256 MiB limit')
    config['train_tokens'] = tokens
    manifest = dict(manifest, config=config)
    template, _ = author._choose_template_with_source(manifest, explicit_template=None)
    require(template == read(ROOT / 'version/v7/templates/qwen3.json'),
            'custom base templates require separate certification')
    registry = training_registry()
    bindings = read(ROOT / 'version/v7/kernel_maps/kernel_bindings.json')
    rules = read(ROOT / 'version/v8/scripts/grad_rules_v8.json')
    ir = author.build_ir1_train(manifest, registry, bindings, rules, max_layers=2,
                               strict=True, bridge_lowering='explicit')
    annotations = [op for op in ir['ops'] if op.get('kernel_id') is None and op.get('phase') != 'bridge']
    for op in annotations:
        require(op['op'] in {'bpe_tokenizer', 'lm_head'} and not op['weights']
                and op['dataflow'] == {'inputs': {}, 'outputs': {}} and not op['requires_grad'],
                'unsupported executable operation')
    ir['ops'] = [op for op in ir['ops'] if op not in annotations]
    return base, manifest, ir, registry, bindings, rules


def lora_targets(base, *, tokens):
    """Read-only target discovery. Does not initialize, compile or train."""
    base, manifest, ir, registry, bindings, rules = _base_ir(base, tokens)
    targets = []
    for tid, meta in ir['tensors'].items():
        if meta.get('kind') != 'weight':
            continue
        name = tid.removeprefix('weight.')
        try:
            _, spec = attach_lora(ir, target=name, rank=1, alpha=1, name='discovery')
        except ValueError:
            continue
        matches = [op for op in ir['ops'] if any(ref.get('tensor') == tid for ref in op.get('weights', {}).values())]
        require(len(matches) == 1, 'ambiguous target ownership')
        max_rank = min(spec['inputs'], spec['outputs'], 1000000 // (spec['inputs'] + spec['outputs']))
        if max_rank < 1:
            continue
        targets.append(dict(target=name, layer=matches[0]['layer'], role=matches[0]['op'],
                            inputs=spec['inputs'], outputs=spec['outputs'],
                            rank_min=1, rank_max=max_rank,
                            parameters_per_rank=spec['inputs'] + spec['outputs'],
                            precision='fp32'))
    return dict(schema='cke.v8.lora_targets.v1',
                base_manifest_sha256=workflow._sha256(base / 'weights_manifest.json'),
                targets=targets, execution='NOT_RUN')


def prepare_lora(base, *, adapters, tokens, max_parameters=1000000):
    """Prepare a checked plan from an existing bounded FP32 decoder BUMP artifact.

    This first public session supports two-layer dense qwen3-template artifacts;
    it does not import arbitrary pretrained/quantized Qwen models.
    """
    require(type(max_parameters) is int and 0 < max_parameters <= 1000000,
            'adapter parameter budget must be in [1, 1000000]')
    require(adapters and all(isinstance(a, LoRAConfig) for a in adapters), 'explicit LoRAConfig declarations required')
    require(all(type(a.seed) is int and 0 <= a.seed < 2**32 for a in adapters), 'invalid initialization seed')
    base, manifest, ir, registry, bindings, rules = _base_ir(base, tokens)
    declarations = [{k: v for k, v in asdict(a).items() if k != 'seed'} for a in adapters]
    ir, specs = attach_lora_adapters(ir, adapters=declarations,
                                    max_parameters=max_parameters, freeze_base=True)
    schema = resolve_lora_parameters(ir)
    blob = (base / 'weights.bump').read_bytes()
    values = {}
    for row in manifest['entries']:
        name = row['name']
        if name not in schema:
            continue
        require(name not in values, 'duplicate base parameter')
        shape = schema[name]['shape']
        offset, size = row['offset'], row['size']
        require(row['dtype'] == 'fp32' and row['shape'] == shape
                and type(offset) is int and offset >= 0 and offset % 4 == 0
                and size == math.prod(shape) * 4 and offset + size <= len(blob),
                'invalid base parameter extent: ' + name)
        values[name] = np.frombuffer(blob, dtype='<f4', count=size // 4, offset=offset).reshape(shape).copy()
    values, evidence = prepare_lora_weights(ir, values, seeds={a.name: a.seed for a in adapters})
    identity = dict(schema='cke.v8.lora_plan.v1', base_manifest_sha256=workflow._sha256(base / 'weights_manifest.json'),
                    base_weights_sha256=hashlib.sha256(blob).hexdigest(),
                    template_sha256=workflow._sha256(ROOT / 'version/v7/templates/qwen3.json'),
                    tokens=tokens, adapters=specs, parameters=schema,
                    initialization=evidence, ir_sha256=digest(ir), registry_sha256=digest(registry),
                    bindings_sha256=digest(bindings), gradient_rules_sha256=digest(rules))
    return LoRAPlan(ir, values, registry, identity)


class LoRAPlan:
    def __init__(self, ir, values, registry, identity):
        self._ir, self._values, self._registry = ir, values, registry
        self._identity = identity

    def describe(self):
        return json.loads(json.dumps(self._identity))

    def compile(self, directory, *, learning_rate=3e-4, max_steps=1000,
                dataset_id, workspace_limit_bytes=512 * 1024**2):
        require(type(max_steps) is int and 0 < max_steps <= 100000, 'invalid step budget')
        require(isinstance(dataset_id, str) and dataset_id.strip(), 'explicit dataset identity required')
        require(type(learning_rate) in (int, float) and math.isfinite(learning_rate) and learning_rate > 0,
                'invalid learning rate')
        require(type(workspace_limit_bytes) is int and 0 < workspace_limit_bytes <= 2 * 1024**3,
                'invalid workspace budget')
        require(digest(self._ir) == self._identity['ir_sha256'], 'plan IR changed after preparation')
        # Revalidate values and scales before compilation, without reinitializing.
        values, evidence = prepare_lora_weights(self._ir, self._values, mode='resume')
        require(evidence['weights_sha256'] == self._identity['initialization']['weights_sha256'],
                'prepared weights changed')
        bindings = read(ROOT / 'version/v7/kernel_maps/kernel_bindings.json')
        rules = read(ROOT / 'version/v8/scripts/grad_rules_v8.json')
        require(digest(self._registry) == self._identity['registry_sha256']
                and digest(bindings) == self._identity['bindings_sha256']
                and digest(rules) == self._identity['gradient_rules_sha256'], 'compiler contracts changed since preparation')
        lowered = backward.synthesize_ir2_backward(self._ir, self._registry, bindings, rules,
                    strict=True, allow_partial=False, checkpoint_policy='none')
        layout = planner.build_layout(lowered, {'config': lowered['config']}, 64, strict=True)
        require(layout['total_bytes'] <= workspace_limit_bytes, 'training workspace exceeds budget')
        source, summary = codegen.generate_c(lowered, self._registry,
                         manifest={'config': lowered['config']}, layout=layout)
        require(not summary['skipped_ops'], 'incomplete generated computation')
        require(set(summary['parameter_gradient_order']) == set(self._ir['adapter_training_policy']['trainable_parameters']),
                'adapter-only gradient ownership violated')
        folder = Path(directory).resolve()
        folder.mkdir(parents=True, exist_ok=False)
        (folder / 'train.c').write_text(source)
        builder = workflow._load_module('cke_lora_shared_builder', ROOT / 'version/v7/scripts/ck_run_v7.py')
        engine = ROOT / 'build/libckernel_engine.so'
        require(engine.is_file(), 'build/libckernel_engine.so must be built first')
        builder._compile_train_runtime_with_identity(folder / 'train.c', folder / 'train.so',
            lib_ck=engine, cflags=(), defines={'CK_NUM_TOKENS': self._identity['tokens']},
            extra_sources=builder._resolve_training_provider_sources(summary, self._registry, engine),
            relocatable=True, optimization='-O2', compiler='cc', identity_out=folder / 'build-identity.json')
        packed = np.concatenate([values[n].ravel() for n in summary['init_weight_order']]).astype('<f4')
        packed.tofile(folder / 'initial.f32')
        policy = dict(plan_sha256=digest(self._identity), learning_rate=float(learning_rate),
                      optimizer='AdamW', beta1=.9, beta2=.999, epsilon=1e-8, weight_decay=.01,
                      max_steps=max_steps)
        document = dict(schema='cke.v8.lora_session.v1', plan=self._identity, policy=policy,
                        dataset_sha256=hashlib.sha256(dataset_id.encode()).hexdigest(), summary=summary,
                        files={name: workflow._sha256(folder / name) for name in
                               ('train.so', 'train.c', 'libckernel_engine.so', 'build-identity.json', 'initial.f32')})
        document['identity_sha256'] = digest(document)
        workflow._write_json(folder / 'session.json', document)
        return LoRASession.open(folder)


class LoRASession:
    """One isolated native runtime; calls are serialized, and close is explicit."""
    @classmethod
    def open(cls, directory):
        self = cls.__new__(cls)
        self._lock = threading.RLock()
        self._closed = True
        self._temp = None
        try:
            self._load(directory)
        except Exception:
            if hasattr(self, '_lib'):
                self._lib.ck_train_free()
            if self._temp is not None:
                self._temp.cleanup()
            raise
        return self

    def _load(self, directory):
        self._folder = Path(directory).resolve()
        self._doc = read(self._folder / 'session.json')
        require(self._doc.get('schema') == 'cke.v8.lora_session.v1', 'unsupported session artifact')
        require(self._doc.get('identity_sha256') == digest({k: v for k, v in self._doc.items() if k != 'identity_sha256'}),
                'session metadata identity mismatch')
        require(set(self._doc['files']) == {'train.so', 'train.c', 'libckernel_engine.so', 'build-identity.json', 'initial.f32'},
                'incomplete session inventory')
        for name, expected in self._doc['files'].items():
            require(workflow._sha256(self._folder / name) == expected, 'artifact identity mismatch: ' + name)
        require(self._doc['policy']['plan_sha256'] == digest(self._doc['plan']), 'plan identity mismatch')
        self._summary = self._doc['summary']
        self._tokens = self._doc['plan']['tokens']
        self._temp = tempfile.TemporaryDirectory(prefix='cke-lora-session-')
        local = Path(self._temp.name)
        # Distinct inode avoids dlopen sharing mutable generated globals between sessions.
        shutil.copy2(self._folder / 'train.so', local / 'train.so')
        require(workflow._sha256(local / 'train.so') == self._doc['files']['train.so'],
                'copied runtime identity mismatch')
        (local / 'libckernel_engine.so').symlink_to(self._folder / 'libckernel_engine.so')
        self._lib = C.CDLL(str(local / 'train.so'))
        identity = read(self._folder / 'build-identity.json')
        require(workflow.CERT._build_identity_check(self._lib, identity['build_identity_sha256'])['passed'],
                'loaded library build identity mismatch')
        workflow._configure(self._lib)
        self._lib.ck_train_get_runtime_contract_sha256.restype = C.c_char_p
        require(self._lib.ck_train_get_runtime_contract_sha256().decode() == self._summary['runtime_contract_sha256'],
                'loaded runtime contract mismatch')
        self._fp, self._ip = C.POINTER(C.c_float), C.POINTER(C.c_int32)
        self._lib.ck_train_init.argtypes = [self._fp, self._ip, C.c_int]
        self._lib.ck_train_step_ex.argtypes = [self._ip, self._ip, C.c_int, self._fp, C.c_float]
        self._lib.ck_train_set_batch_ex.argtypes = [self._ip, self._ip, C.c_int]
        self._lib.ck_train_eval_step.argtypes = [self._fp]
        initial = np.fromfile(self._folder / 'initial.f32', dtype='<f4')
        require(hashlib.sha256(initial.tobytes()).hexdigest() == self._doc['files']['initial.f32'],
                'loaded initial weight identity mismatch')
        sizes = np.asarray(self._summary['init_weight_numel'], dtype='i4')
        require(initial.size == int(sizes.sum()), 'initial weight extent mismatch')
        require(self._lib.ck_train_init(initial.ctypes.data_as(self._fp), sizes.ctypes.data_as(self._ip), len(sizes)) == len(sizes),
                'native initialization failed')
        self._lib.ck_set_strict_parity(1)
        self._closed = False
        return self

    def _batch(self, tokens, targets):
        require(not self._closed, 'session closed')
        arrays = []
        for value in (tokens, targets):
            a = np.asarray(value)
            require(a.dtype.kind in 'iu' and a.shape == (self._tokens,), 'fixed-length integer token batch required')
            require(np.all(a >= 0) and np.all(a < self._doc['plan']['parameters']['token_emb']['shape'][0]), 'token ID out of range')
            arrays.append(np.ascontiguousarray(a, dtype='i4'))
        return arrays

    def train_step(self, tokens, targets):
        with self._lock:
            ids, labels = self._batch(tokens, targets)
            require(self._lib.ck_train_get_opt_step() < self._doc['policy']['max_steps'], 'training step budget exhausted')
            loss = C.c_float()
            rc = self._lib.ck_train_step_ex(ids.ctypes.data_as(self._ip), labels.ctypes.data_as(self._ip),
                    self._tokens, C.byref(loss), self._doc['policy']['learning_rate'])
            require(rc > 0 and math.isfinite(loss.value), 'native training failed')
            return {'loss': loss.value, 'step': self._lib.ck_train_get_opt_step()}

    def evaluate(self, tokens, targets):
        with self._lock:
            ids, labels = self._batch(tokens, targets)
            require(self._lib.ck_train_set_batch_ex(ids.ctypes.data_as(self._ip), labels.ctypes.data_as(self._ip), self._tokens) > 0,
                    'native batch rejected')
            loss = C.c_float()
            require(self._lib.ck_train_eval_step(C.byref(loss)) > 0 and math.isfinite(loss.value), 'native evaluation failed')
            outputs = workflow.CERT._activation_snapshot(self._lib, self._summary)
            return {'loss': loss.value, 'logits': outputs['act.Sfooter.logits.0.y'].reshape(self._tokens, -1).copy(),
                    'step': self._lib.ck_train_get_opt_step()}

    def _checkpoint_metadata(self):
        policy = self._doc['policy']
        return dict(runtime_contract_sha256=self._summary['runtime_contract_sha256'],
                    training_config=policy, training_config_sha256=workflow._json_sha256(policy),
                    dataset_sha256=self._doc['dataset_sha256'], total_microsteps=policy['max_steps'],
                    next_microstep=int(self._lib.ck_train_get_opt_step()), grad_accum=1)

    def save_checkpoint(self, directory):
        with self._lock:
            require(not self._closed, 'session closed')
            path = Path(directory).resolve()
            return workflow._checkpoint_write(path.parent, path.name, self._lib, self._checkpoint_metadata())

    def load_checkpoint(self, directory):
        with self._lock:
            require(not self._closed, 'session closed')
            path = Path(directory).resolve()
            doc = read(path / 'checkpoint.json')
            expected = self._checkpoint_metadata()
            workflow._validate_checkpoint_document(doc, expected)
            require(all(type(doc[key]) is int for key in ('optimizer_step', 'next_microstep', 'accum_counter', 'accum_tokens'))
                    and doc['optimizer_step'] == doc['next_microstep'] <= expected['total_microsteps']
                    and doc['accum_counter'] == 0 and doc['accum_tokens'] == 0,
                    'checkpoint step ownership mismatch')
            # Validate every file before the shared loader can mutate native state.
            payloads = {}
            for name, row in doc['files'].items():
                require(row['file'] == name + '.f32', 'invalid checkpoint filename')
                payload = (path / row['file']).read_bytes()
                payloads[name] = payload
                current = workflow._array_export(self._lib, name)
                require(len(payload) == current.size * 4 and row['numel'] == current.size
                        and hashlib.sha256(payload).hexdigest() == row['sha256']
                        and np.isfinite(np.frombuffer(payload, dtype='<f4')).all(), 'invalid checkpoint payload')
            checkpoint_weights = np.frombuffer(payloads['weight'], dtype='<f4')
            offset = 0
            for name, count in zip(self._summary['init_weight_order'], self._summary['init_weight_numel']):
                if not self._doc['plan']['parameters'][name]['trainable']:
                    require(hashlib.sha256(checkpoint_weights[offset:offset + count].tobytes()).hexdigest()
                            == self._doc['plan']['initialization']['parameters'][name]['sha256'],
                            'checkpoint changed frozen parameter: ' + name)
                offset += count
            # Restore only the bytes just validated, even if the source changes.
            with tempfile.TemporaryDirectory(prefix='cke-lora-checkpoint-') as snapshot:
                snapshot = Path(snapshot)
                (snapshot / 'checkpoint.json').write_text(json.dumps(doc))
                for name, payload in payloads.items():
                    (snapshot / (name + '.f32')).write_bytes(payload)
                workflow._checkpoint_load(snapshot, self._lib, expected)

    def _adapter_identity(self):
        plan = self._doc['plan']
        identity = {key: plan[key] for key in ('base_manifest_sha256', 'base_weights_sha256',
                'template_sha256', 'parameters')}
        # Weight compatibility is independent of the compiled batch extent.
        identity['adapters'] = [{key: value for key, value in spec.items() if key != 'rows'}
                                for spec in plan['adapters']]
        return identity

    def export_adapters(self, directory):
        with self._lock:
            require(not self._closed, 'session closed')
            packed = workflow._weight_export(self._lib)
            names = {name for spec in self._doc['plan']['adapters'] for name in spec['parameters'].values()}
            arrays, offset = {}, 0
            for name, count in zip(self._summary['init_weight_order'], self._summary['init_weight_numel']):
                value = packed[offset:offset + count]
                if name in names:
                    arrays[name] = value.reshape(self._doc['plan']['parameters'][name]['shape']).copy()
                else:
                    require(hashlib.sha256(value.astype('<f4', copy=False).tobytes()).hexdigest()
                            == self._doc['plan']['initialization']['parameters'][name]['sha256'], 'frozen base changed')
                offset += count
            require(all(np.isfinite(a).all() for a in arrays.values()), 'nonfinite adapter output')
            folder = Path(directory).resolve()
            folder.mkdir(parents=True, exist_ok=False)
            np.savez(folder / 'adapters.npz', **arrays)
            workflow._write_json(folder / 'adapters.json', dict(schema='cke.v8.lora_adapter.v1',
                identity=self._adapter_identity(), sha256=workflow._sha256(folder / 'adapters.npz')))
            return folder

    def load_adapters(self, directory):
        with self._lock:
            require(not self._closed, 'session closed')
            require(self._lib.ck_train_get_opt_step() == 0, 'load adapters only into a fresh optimizer; use checkpoint resume otherwise')
            folder = Path(directory).resolve()
            doc = read(folder / 'adapters.json')
            require(doc.get('schema') == 'cke.v8.lora_adapter.v1' and doc.get('identity') == self._adapter_identity(),
                    'incompatible base or adapter declarations')
            require(workflow._sha256(folder / 'adapters.npz') == doc['sha256'], 'adapter payload identity mismatch')
            names = {name for spec in self._doc['plan']['adapters'] for name in spec['parameters'].values()}
            with np.load(folder / 'adapters.npz', allow_pickle=False) as payload:
                require(set(payload.files) == names, 'adapter tensor inventory mismatch')
                arrays = {name: payload[name].copy() for name in names}
            for name, array in arrays.items():
                require(array.dtype == np.float32 and list(array.shape) == self._doc['plan']['parameters'][name]['shape']
                        and np.isfinite(array).all(), 'invalid adapter tensor: ' + name)
            for spec in self._doc['plan']['adapters']:
                require(float(arrays[spec['parameters']['scale']][0]) == float(np.float32(spec['scale'])), 'adapter scale changed')
            packed = workflow._weight_export(self._lib)
            offset = 0
            for name, count in zip(self._summary['init_weight_order'], self._summary['init_weight_numel']):
                if name in arrays:
                    packed[offset:offset + count] = arrays[name].ravel()
                offset += count
            workflow._array_import(self._lib, 'weight', packed)

    def close(self):
        with self._lock:
            if not self._closed:
                self._lib.ck_train_free()
                self._closed = True
                self._temp.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()
