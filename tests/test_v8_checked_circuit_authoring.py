"""Shared declaration/build adapters used by multiple model definitions."""
import copy,tempfile,unittest
from pathlib import Path
import numpy as np
from version.v8.scripts.checked_circuit_authoring_v8 import declared_f32_weight_shapes,linear
from version.v8.scripts.compile_checked_bundle_v8 import compile_declared_f32_graph
from version.v8.tts.build_qwen3_tts_text_projection_circuit import build_circuit

class AuthoringTests(unittest.TestCase):
    def test_reject_inconsistent_shared_weight_and_unknown_port(self):
        a=linear('a','x','a',4,6,2,'shared');b=linear('b','a','b',6,6,2,'shared')
        graph={'block_types':{'block':{'body':{'ops':[a,b]}}}}
        with self.assertRaisesRegex(ValueError,'conflicting shapes'):declared_f32_weight_shapes(graph)
        a['weight_refs']={'unknown':'x'};graph['block_types']['block']['body']['ops']=[a]
        with self.assertRaisesRegex(ValueError,'unsupported declared weight port'):declared_f32_weight_shapes(graph)
    def test_qwen_projection_uses_shared_compile_adapter(self):
        g=build_circuit(tokens=3,vocab=8,text_width=4,hidden_width=6)
        rng=np.random.default_rng(52);weights={k:rng.normal(0,.2,s).astype('f4') for k,s in declared_f32_weight_shapes(g).items()}
        config={'model':g['name'],'arch':g['name'],'num_layers':1,'embed_dim':6,'num_heads':1,'num_kv_heads':1,
            'head_dim':6,'intermediate_size':12,'context_length':3,'max_seq_len':3,'vocab_size':8}
        with tempfile.TemporaryDirectory() as folder:
            fn,a,v,l=compile_declared_f32_graph(Path(folder),g,weights,source_config=config)
            v('text_ids',np.int32)[:]=[1,3,6];self.assertEqual(fn(a,len(a)),0)
            w=weights;x=w['talker.model.text_embedding.weight'][[1,3,6]].astype('f8')
            x=x@w['talker.text_projection.linear_fc1.weight'].T+w['talker.text_projection.linear_fc1.bias']
            x=x/(1+np.exp(-x));expected=x@w['talker.text_projection.linear_fc2.weight'].T+w['talker.text_projection.linear_fc2.bias']
            np.testing.assert_allclose(v('projection_output'),expected,atol=1e-6,rtol=1e-6)
if __name__=='__main__':unittest.main()
