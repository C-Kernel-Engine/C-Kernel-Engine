"""Application-facing generated LoRA: independent consumer, no private trainer API."""
import json
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'version/v7'))
sys.path.insert(0, str(ROOT / 'version/v7/scripts'))
import ckernel_engine as cke
import init_tiny_train_model_v7 as initializer


def base_artifact(folder):
    initializer.build_tiny_model(folder, seed=41, init='xavier_uniform', template_name='qwen3',
        template_doc=json.loads((ROOT / 'version/v7/templates/qwen3.json').read_text()),
        n_layers=2, vocab_size=32, embed_dim=32, hidden_dim=64, num_heads=4,
        num_kv_heads=4, context_len=8, rope_theta=10000., kernel_policy='fp32',
        adamw_beta1=.9, adamw_beta2=.999, adamw_eps=1e-8, adamw_weight_decay=.01,
        omit_linear_biases=True)
    path = folder / 'weights_manifest.json'
    doc = json.loads(path.read_text())
    doc['config']['rms_eps'] = 1e-5
    path.write_text(json.dumps(doc))


class LoRASessionTests(unittest.TestCase):
    def test_rejected_plans_and_capabilities(self):
        self.assertFalse(cke.v8.lora_capabilities()['qlora'])
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / 'base'
            base_artifact(base)
            discovery = cke.v8.lora_targets(base, tokens=4)
            self.assertEqual(discovery['execution'], 'NOT_RUN')
            targets = {row['target']: row for row in discovery['targets']}
            self.assertIn('layer.0.wq', targets)
            self.assertIn('layer.1.wv', targets)
            self.assertNotIn('layer.0.ln1_gamma', targets)
            self.assertEqual(targets['layer.0.wq']['rank_max'], 32)
            self.assertEqual(targets['layer.0.wq']['parameters_per_rank'], 64)
            adapter = cke.v8.LoRAConfig('layer.0.wq', 2, 4, 'query', 42)
            for tokens in (0, 9, True):
                with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                    cke.v8.prepare_lora(base, adapters=[adapter], tokens=tokens)
            for target in ('missing', 'layer.0.ln1_gamma'):
                with self.subTest(target=target), self.assertRaises(ValueError):
                    cke.v8.prepare_lora(base, tokens=4, adapters=[cke.v8.LoRAConfig(target, 2, 4, 'x', 42)])
            with self.assertRaises(ValueError):
                cke.v8.prepare_lora(base, tokens=4, adapters=[adapter, adapter])
            path = base / 'weights_manifest.json'
            doc = json.loads(path.read_text())
            doc['config']['dtype'] = 'bf16'
            path.write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError, 'FP32'):
                cke.v8.prepare_lora(base, tokens=4, adapters=[adapter])

    def test_notebook_cells(self):
        import os
        from unittest.mock import patch
        if not (ROOT / 'build/libckernel_engine.so').exists():
            self.skipTest('native engine must be built')
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            base_artifact(folder / 'base')
            records = folder / 'records.jsonl'
            records.write_text(json.dumps(dict(tokens=[0,1,2,3], targets=[1,2,3,4])) + '\n')
            environment = dict(CKE_REPO=str(ROOT), CKE_LORA_BASE=str(folder / 'base'),
                CKE_LORA_RUN=str(folder / 'runtime'), CKE_LORA_RECORDS=str(records),
                CKE_LORA_TOKENS='4', CKE_LORA_STEPS='3')
            notebook = json.loads((ROOT / 'version/v8/notebooks/03_generated_lora_session.ipynb').read_text())
            scope = {}
            with patch.dict(os.environ, environment):
                for cell in notebook['cells']:
                    if cell['cell_type'] == 'code':
                        exec(compile(''.join(cell['source']), '<lora-notebook>', 'exec'), scope)
            self.assertEqual(scope['result']['step'], 3)
            self.assertEqual(scope['replay']['step'], 0)

    def test_notebook_session(self):
        import numpy as np
        if not (ROOT / 'build/libckernel_engine.so').exists():
            self.skipTest('native engine must be built')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_artifact(root / 'base')
            adapters = [cke.v8.LoRAConfig('layer.0.wq', 2, 4, 'query', 42),
                        cke.v8.LoRAConfig('layer.1.wv', 3, 6, 'value', 43)]
            plan = cke.v8.prepare_lora(root / 'base', adapters=adapters, tokens=4)
            self.assertEqual(sum(np.prod(row['shape']) for row in plan.describe()['parameters'].values() if row['trainable']), 320)
            ids, labels = [0, 1, 2, 3], [1, 2, 3, 4]
            with plan.compile(root / 'runtime', learning_rate=.03, max_steps=40, dataset_id='toy-next-token-v1') as a:
                initial = a.evaluate(ids, labels)
                self.assertEqual(initial['step'], 0)
                with cke.v8.LoRASession.open(root / 'runtime') as b:
                    for _ in range(10):
                        a.train_step(ids, labels)
                    np.testing.assert_array_equal(b.evaluate(ids, labels)['logits'], initial['logits'])
                    checkpoint = a.save_checkpoint(root / 'checkpoint')
                    b.load_checkpoint(checkpoint)
                    np.testing.assert_array_equal(a.evaluate(ids, labels)['logits'], b.evaluate(ids, labels)['logits'])
                    self.assertEqual(a.train_step(ids, labels), b.train_step(ids, labels))
                before = a.evaluate(ids, labels)
                for invalid in ([0, 1], [0, 1, 2, 32], [-1, 1, 2, 3], [0., 1., 2., 3.]):
                    with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                        a.train_step(invalid, labels)
                np.testing.assert_array_equal(a.evaluate(ids, labels)['logits'], before['logits'])
                # Frozen base integrity is checked even for internally consistent files.
                checkpoint_json = checkpoint / 'checkpoint.json'
                saved_document = checkpoint_json.read_text()
                weight_path = checkpoint / 'weight.f32'
                saved_weights = weight_path.read_bytes()
                summary = json.loads((root / 'runtime/session.json').read_text())['summary']
                offset = 0
                for name, count in zip(summary['init_weight_order'], summary['init_weight_numel']):
                    if name == 'token_emb':
                        break
                    offset += count
                corrupt = np.frombuffer(saved_weights, dtype='<f4').copy()
                corrupt[offset] += 1
                weight_path.write_bytes(corrupt.tobytes())
                doc = json.loads(saved_document)
                doc['files']['weight']['sha256'] = hashlib.sha256(weight_path.read_bytes()).hexdigest()
                from version.v8.training.lora_session import workflow
                doc['checkpoint_identity_sha256'] = workflow._checkpoint_identity(doc)
                checkpoint_json.write_text(json.dumps(doc))
                with self.assertRaisesRegex(ValueError, 'frozen parameter'):
                    a.load_checkpoint(checkpoint)
                np.testing.assert_array_equal(a.evaluate(ids, labels)['logits'], before['logits'])
                weight_path.write_bytes(saved_weights)
                checkpoint_json.write_text(saved_document)
                # A corrupt later checkpoint file cannot partially import weights.
                (checkpoint / 'optimizer_state.f32').write_bytes(b'broken')
                with self.assertRaises(ValueError):
                    a.load_checkpoint(checkpoint)
                np.testing.assert_array_equal(a.evaluate(ids, labels)['logits'], before['logits'])
                for _ in range(29):
                    a.train_step(ids, labels)
                final = a.evaluate(ids, labels)
                self.assertLess(final['loss'], initial['loss'])
                with self.assertRaisesRegex(ValueError, 'budget'):
                    a.train_step(ids, labels)
                a.save_checkpoint(root / 'final')
                a.export_adapters(root / 'adapters')
                with cke.v8.LoRASession.open(root / 'runtime') as inference:
                    inference.load_adapters(root / 'adapters')
                    replay = inference.evaluate(ids, labels)
                    self.assertEqual(replay['step'], 0)
                    adapter_doc = root / 'adapters/adapters.json'
                    original_doc = adapter_doc.read_text()
                    bad = json.loads(original_doc)
                    bad['identity']['base_weights_sha256'] = 'wrong-base'
                    adapter_doc.write_text(json.dumps(bad))
                    with self.assertRaisesRegex(ValueError, 'incompatible base'):
                        inference.load_adapters(root / 'adapters')
                    adapter_doc.write_text(original_doc)
                    archive = root / 'adapters/adapters.npz'
                    original_payload = archive.read_bytes()
                    with np.load(archive, allow_pickle=False) as data:
                        valid_arrays = {name: data[name].copy() for name in data.files}
                    for variant in ('missing', 'nonfinite', 'scale'):
                        arrays = {name: value.copy() for name, value in valid_arrays.items()}
                        if variant == 'missing':
                            arrays.pop('lora.query.a')
                        elif variant == 'nonfinite':
                            arrays['lora.query.a'][0, 0] = np.nan
                        else:
                            arrays['lora.query.scale'][0] = 99
                        np.savez(archive, **arrays)
                        invalid_doc = json.loads(original_doc)
                        invalid_doc['sha256'] = hashlib.sha256(archive.read_bytes()).hexdigest()
                        adapter_doc.write_text(json.dumps(invalid_doc))
                        with self.subTest(adapter=variant), self.assertRaises(ValueError):
                            inference.load_adapters(root / 'adapters')
                        np.testing.assert_array_equal(inference.evaluate(ids, labels)['logits'], replay['logits'])
                    archive.write_bytes(original_payload)
                    adapter_doc.write_text(original_doc)
                    np.testing.assert_array_equal(inference.evaluate(ids, labels)['logits'], replay['logits'])
                    np.testing.assert_array_equal(replay['logits'], final['logits'])
                    inference.train_step(ids, labels)
                    with self.assertRaisesRegex(ValueError, 'fresh optimizer'):
                        inference.load_adapters(root / 'adapters')
                expected = final['logits'].tolist()
            longer = cke.v8.prepare_lora(root / 'base', adapters=adapters, tokens=7)
            with longer.compile(root / 'longer', dataset_id='same-base-longer', max_steps=1) as session:
                session.load_adapters(root / 'adapters')
                long_result = session.evaluate(list(range(7)), list(range(1, 8)))
                np.testing.assert_allclose(long_result['logits'][:4], expected, atol=3e-5, rtol=3e-4)
            with self.assertRaisesRegex(ValueError, 'closed'):
                a.evaluate(ids, labels)
            # Fresh interpreter imports only the public application interface.
            script = '''import sys,json
sys.path.insert(0,sys.argv[1])
import ckernel_engine as cke
with cke.v8.LoRASession.open(sys.argv[2]) as s:
 s.load_checkpoint(sys.argv[3])
 r=s.evaluate([0,1,2,3],[1,2,3,4])
 open(sys.argv[4],'w').write(json.dumps(r['logits'].tolist()))
assert 'torch' not in sys.modules and 'transformers' not in sys.modules
'''
            subprocess.run([sys.executable, '-c', script, str(ROOT / 'version/v7'), str(root / 'runtime'),
                            str(root / 'final'), str(root / 'fresh.json')], check=True, cwd=root)
            self.assertEqual(json.loads((root / 'fresh.json').read_text()), expected)


if __name__ == '__main__':
    unittest.main()
