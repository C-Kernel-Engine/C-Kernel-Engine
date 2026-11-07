"""Evidence-to-option-summary composition through the ordinary generated entry."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from tests.test_v8_clef_flash_head_ingress import compile_case
from tests.test_v8_clef_flash_evidence0 import prepare, torch_layer
from tests.test_v8_clef_flash_evidence_complete import both_synthetic_weights
from tests.test_v8_grouped_weighted_rows import ordered_reference

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'version/v8/decision'))
from build_clef_flash_option_summary_circuit import OUTPUT, build_circuit


class ClefFlashOptionSummaryTest(unittest.TestCase):
    def test_canonical_circuit_and_explicit_edges(self):
        graph = build_circuit()
        self.assertEqual(OUTPUT.read_text(), json.dumps(graph, indent=2)+'\n')
        block = graph['block_types']['option_summary']
        self.assertEqual(block['body']['ops'][0]['graph_slots']['inputs'], {
            'options':'evidence1_output', 'fields':'question_projected',
            'group_counts':'external:option_counts'})
        self.assertEqual(block['footer'][0]['params']['epsilon'], 1e-5)

    def test_generated_summary_unequal_groups_recovery_and_reordering(self):
        graph = build_circuit(channels=8, width=8, heads=2, feedforward=16)
        weights = both_synthetic_weights(8,8,16,16)
        weights['option_summary_norm.weight'] = np.linspace(.7,1.3,8,dtype='f4')
        weights['option_summary_norm.bias'] = np.linspace(-.1,.1,8,dtype='f4')
        with tempfile.TemporaryDirectory() as temp:
            arena, execute, view, calls, _loaded = compile_case(Path(temp), graph, weights,
                                                               enclosing_epsilon=.02)
            self.assertEqual(calls['errors'], [])
            self.assertEqual(len(calls['operations']), 43)
            hidden = np.random.default_rng(230).normal(0,.5,(6,8)).astype('f4')
            prepare(view, hidden)
            # Rejection precedes the parity run and cannot publish the tail.
            view('option_counts')[:]=[1,1]
            view('option_summary_normalized')[:]=-73
            self.assertNotEqual(execute(arena,len(arena)),0)
            np.testing.assert_array_equal(view('option_summary_normalized'),-73)
            view('option_counts')[:]=[2,1]
            self.assertEqual(execute(arena,len(arena)),0)
            expected=ordered_reference(view('evidence1_output').copy(),
                                       view('question_projected').copy(),[2,1])
            np.testing.assert_array_equal(view('option_summary_raw'),expected)
            baseline=view('option_summary_normalized').copy()
            spans=view('option_spans').copy()
            view('option_spans')[:]=spans[[1,0,2]]
            self.assertEqual(execute(arena,len(arena)),0)
            np.testing.assert_allclose(view('option_summary_normalized'),baseline,atol=2e-6,rtol=0)
            prepare(view, hidden+.1)
            self.assertEqual(execute(arena,len(arena)),0)
            self.assertFalse(np.array_equal(view('option_summary_normalized'),baseline))
            prepare(view, hidden)
            self.assertEqual(execute(arena,len(arena)),0)
            np.testing.assert_array_equal(view('option_summary_normalized'),baseline)
            self.assertNotEqual(execute(arena,len(arena)-1),0)
            np.testing.assert_array_equal(view('option_summary_normalized'),baseline)
            self.assertEqual(execute(arena,len(arena)),0)

    def test_connected_independent_torch_two_token_geometries(self):
        try:
            import torch
        except ImportError:
            self.skipTest('independent PyTorch comparison unavailable')
        for tokens,counts in ((6,[2,1]),(8,[1,2])):
            with self.subTest(tokens=tokens,counts=counts), tempfile.TemporaryDirectory() as temp:
                graph=build_circuit(tokens=tokens,channels=8,width=8,heads=2,feedforward=16)
                weights=both_synthetic_weights(8,8,16,16)
                weights['option_summary_norm.weight']=np.linspace(.7,1.3,8,dtype='f4')
                weights['option_summary_norm.bias']=np.linspace(-.1,.1,8,dtype='f4')
                arena,execute,view,_calls,_loaded=compile_case(Path(temp),graph,weights,
                                                              enclosing_epsilon=.02)
                prepare(view,np.random.default_rng(901+tokens).normal(0,.5,(tokens,8)).astype('f4'))
                view('option_counts')[:]=counts
                self.assertEqual(execute(arena,len(arena)),0)
                first=torch_layer(view('option_queries').copy(),view('memory').copy(),weights,2)
                second=torch_layer(first['evidence0_output'].numpy(),view('memory').copy(),weights,2,
                                   prefix='evidence_layers.1')['evidence0_output']
                fields=torch.from_numpy(view('question_projected').copy())
                raw=[]; offset=0
                for q,count in enumerate(counts):
                    options=second[offset:offset+count]
                    p=torch.softmax(options @ fields[q] / (8**.5),dim=0)
                    raw.append((p[:,None]*options).sum(0)); offset+=count
                raw=torch.stack(raw)
                norm=torch.nn.functional.layer_norm(raw,(8,),
                    torch.from_numpy(weights['option_summary_norm.weight']),
                    torch.from_numpy(weights['option_summary_norm.bias']),eps=1e-5)
                np.testing.assert_allclose(view('option_summary_raw'),raw.numpy(),atol=3e-6,rtol=0)
                np.testing.assert_allclose(view('option_summary_normalized'),norm.numpy(),atol=6e-6,rtol=0)

    def test_pinned_head_reference_prefix(self):
        import hashlib
        import importlib.util
        import os
        from types import SimpleNamespace
        asset = os.environ.get('CKE_CLEF_FLASH_HEAD_PATH')
        source = os.environ.get('CKE_CLEF_FLASH_REFERENCE_SOURCE')
        if not asset or not source:
            self.skipTest('pinned head archive/reference source unavailable')
        try:
            import torch
            from safetensors.torch import load_file
        except ImportError:
            self.skipTest('PyTorch/safetensors comparison unavailable')
        from tests.test_v8_clef_flash_evidence_complete import both_pinned_weights
        identity=json.loads((ROOT/'tests/fixtures/decision/clef_flash/identity.json').read_text())
        self.assertEqual(hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                         identity['joint_schema_model_sha256'])
        weights=both_pinned_weights(asset)  # also checks the pinned archive hash
        state=load_file(asset)
        for suffix in ('weight','bias'):
            name='option_summary_norm.'+suffix
            weights[name]=state[name].float().numpy()
        spec=importlib.util.spec_from_file_location('clef_option_summary_reference',source)
        module=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=module
        spec.loader.exec_module(module)
        head=module.JointSchemaHead(hidden_size=4096,width=1024,routing_layers=2,
                                   layers=4,heads=16,feedforward=4096,dropout=0)
        head.load_state_dict(state,strict=True)
        head.eval()
        captures={}
        class PrefixCaptured(Exception):
            pass
        def capture(_module,args,output):
            captures['raw']=args[0].detach().numpy().copy()
            captures['normalized']=output.detach().numpy().copy()
            raise PrefixCaptured()
        handle=head.option_summary_norm.register_forward_hook(capture)
        try:
            for tokens,counts in ((6,[2,1]),(8,[1,2])):
                with self.subTest(tokens=tokens), tempfile.TemporaryDirectory() as temp:
                    graph=build_circuit(tokens=tokens)
                    arena,execute,view,_calls,_loaded=compile_case(Path(temp),graph,weights)
                    hidden=np.random.default_rng(104+tokens).normal(0,.2,(tokens,4096)).astype('f4')
                    prepare(view,hidden)
                    view('option_counts')[:]=counts
                    self.assertEqual(execute(arena,len(arena)),0)
                    spans=[tuple(map(int,x)) for x in view('option_spans')]
                    questions=[]; offset=0
                    for q,n in enumerate(counts):
                        questions.append(SimpleNamespace(question_span=tuple(map(int,view('question_spans')[q])),
                            question_type=0,option_spans=spans[offset:offset+n]))
                        offset+=n
                    record=SimpleNamespace(questions=questions)
                    with torch.no_grad(), self.assertRaises(PrefixCaptured):
                        head(torch.from_numpy(hidden)[None],
                             torch.from_numpy(view('input_ids').copy().astype('int64'))[None],
                             torch.ones((1,tokens),dtype=torch.int64),[record],
                             torch.from_numpy(weights['output_embedding.weight']))
                    for name in ('raw','normalized'):
                        actual=view('option_summary_'+name)
                        expected=captures[name]
                        error=np.abs(actual.astype('f8')-expected)
                        print(f'pinned option summary tokens={tokens} {name}: '
                              f'max_abs={error.max():.9g} rmse={np.sqrt(np.mean(error**2)):.9g}')
                        np.testing.assert_allclose(actual,expected,atol=1e-5,rtol=0)
        finally:
            handle.remove()


if __name__ == '__main__':
    unittest.main()
