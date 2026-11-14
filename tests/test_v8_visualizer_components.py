import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from version.v8.tools.visualizer_components import build_component_inventory


def put(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


class ComponentInventoryTests(unittest.TestCase):
    def test_components_are_artifacts_not_inferred_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for component in ('frontend', 'encoder', 'decoder', 'talker', 'predictor', 'codec'):
                put(root, component+'/call.json', {'format':'lowered-ir-v3','mode':'prefill',
                    'config':{'artifact_scope':component},'operations':[{'function':'real_provider',
                    'args':[{'buffer_ref':'sequence_owned_state'}]}]})
            report = put(root, 'report.json', {'schema':'producer.report.v1','status':'PASS',
                'transcript':'private words must not be copied','not_certified':['long audio']})
            before = hashlib.sha256(report.read_bytes()).hexdigest()
            with patch('subprocess.run', side_effect=AssertionError('reader executed a producer')):
                result = build_component_inventory(root)
            self.assertEqual(len(result['components']),7)
            self.assertEqual(result['execution_status'],'NOT_VERIFIED')
            self.assertEqual(result['connections_status'],'NOT_INFERRED')
            self.assertNotIn('private words',json.dumps(result))
            self.assertEqual(hashlib.sha256(report.read_bytes()).hexdigest(), before)
            self.assertEqual(result['components'][0]['artifacts'][0]['reported_status'],'PASS')

    def test_does_not_follow_external_artifact_symlinks(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root=Path(directory);other=Path(outside)
            put(other,'call.json',{'schema':'secret'})
            (root/'foreign').symlink_to(other,target_is_directory=True)
            (root/'call.json').symlink_to(other/'call.json')
            result=build_component_inventory(root)
            self.assertEqual(result['components'],[])
            self.assertEqual(len(result['issues']),2)
            self.assertNotIn('secret',json.dumps(result))

    def test_invalid_oversized_and_missing_artifacts_stay_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'bad.json').write_text('{"status":"FAIL","status":"PASS"}')
            (root/'large.json').write_bytes(b' '*100)
            with patch('version.v8.tools.visualizer_components.MAX_FILE_BYTES',64):
                result=build_component_inventory(root)
            self.assertEqual(result['components'],[])
            self.assertEqual(len(result['issues']),2)
            missing=build_component_inventory(root/'missing')
            self.assertTrue(missing['issues'])

    def test_catalog_cannot_be_confused_with_loaded_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);artifacts=root/'bundle';artifacts.mkdir();catalog=root/'circuits'
            for model in ('instella_moe','nemotron_h','cohere2','parakeet','kokoro','qwen_tts'):
                put(catalog,model+'.json',{'name':model,'family':'declared_family',
                    'contract':{'runtime_invariants':{'complete_synthesis':False,'reference_scope':'oracle-fed'}}})
            result=build_component_inventory(artifacts,catalog)
            self.assertEqual(result['components'],[])
            self.assertEqual(len(result['catalog']),6)
            self.assertIs(result['catalog'][0]['runtime_invariants']['complete_synthesis'],False)

    def test_byte_budget_charges_malformed_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'a.json').write_bytes(b'bad-json')
            put(root,'b.json',{'schema':'not-readable-with-remaining-budget'})
            with patch('version.v8.tools.visualizer_components.MAX_TOTAL_BYTES',12):
                result=build_component_inventory(root)
            self.assertEqual(result['bytes_read'],8)
            self.assertFalse(result['components'])

    def test_logical_lifetimes_and_model_independent_providers(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            put(root,'layout.json',{'config':{'model':'arbitrary-new-family'},'memory':{'activations':{'buffers':[
                {'name':'history','lifetime':'sequence','shape':'[2,8]','size':64,'dtype':'fp32'}]}}})
            result=build_component_inventory(root)
            artifact=result['components'][0]['artifacts'][0]
            self.assertEqual(artifact['model'],'arbitrary-new-family')
            self.assertEqual(artifact['declared_lifetimes'][0]['lifetime'],'sequence')

class OperationGraphTests(unittest.TestCase):
    def test_explicit_edges_and_missing_output(self):
        from version.v8.tools.visualizer_components import _operation_graph
        ops = [{'op_id': 0, 'op': 'state_read', 'dataflow': {'outputs': {'state': {}}}},
               {'op_id': 1, 'op': 'recurrent_update', 'dataflow': {'inputs': {
                   'history': {'from_op': 0, 'from_output': 'state'},
                   'bad': {'from_op': 0, 'from_output': 'missing'},
                   'external': {'from': 'external:audio'},
                   'unbound': {'slot': 'state'}}}}]
        graph = _operation_graph({'ops': ops}, 10)
        self.assertEqual([r['status'] for r in graph['nodes'][1]['inputs']],
                         ['resolved', 'output_not_declared', 'external_or_symbolic', 'not_recorded'])
        self.assertFalse(graph['truncated'])

    def test_duplicate_ids_and_truncation_do_not_invent_edges(self):
        from version.v8.tools.visualizer_components import _operation_graph
        ops = [{'op_id': 0}, {'op_id': 0}, {'op_id': 2, 'inputs': {'x': {'from_op': 0}}}]
        graph = _operation_graph({'operations': ops}, 10)
        self.assertEqual(graph['nodes'][2]['inputs'][0]['status'], 'ambiguous')
        outside = [ops[0], ops[2], ops[1]]
        self.assertEqual(_operation_graph({'ops': outside}, 2)['nodes'][1]['inputs'][0]['status'], 'ambiguous')
        self.assertTrue(_operation_graph({'operations': ops}, 1)['truncated'])
        self.assertEqual(_operation_graph({'operations': ops}, 0)['nodes'], [])

    def test_global_capture_budget_and_buffer_bindings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ops = [{'op_id': i, 'args': [{'buffer_ref': 'recurrent_history'}]} for i in range(300)]
            for i in range(10):
                put(root, f'{i}.json', {'operations': ops})
            result = build_component_inventory(root)
            graphs = [a['operation_graph'] for c in result['components'] for a in c['artifacts']]
            self.assertEqual(sum(len(g['nodes']) for g in graphs), 2048)
            self.assertTrue(all(g['truncated'] for g in graphs))
            self.assertEqual(graphs[0]['nodes'][0]['buffer_refs'], ['recurrent_history'])

    def test_no_family_or_order_based_dependencies(self):
        from version.v8.tools.visualizer_components import _operation_graph
        for name in ('rwkv7', 'parakeet', 'whisper', 'kokoro', 'instella', 'cohere2'):
            with self.subTest(name=name):
                graph = _operation_graph({'config': {'model': name}, 'ops': [
                    {'op_id': 1, 'op': 'read'}, {'op_id': 2, 'op': 'update'}]}, 10)
                self.assertEqual([n['inputs'] for n in graph['nodes']], [[], []])

class TrainingTensorEdges(unittest.TestCase):
    def test_existing_training_fixture_projection_edges(self):
        from tests.test_v8_selective_training import fixture
        from version.v8.tools.visualizer_components import _operation_graph
        graph = _operation_graph(fixture(), 256)
        refs = [r for node in graph['nodes'] for r in node['inputs'] if r['tensor']]
        self.assertTrue(refs)
        self.assertTrue(any(r['status'] == 'resolved_tensor' for r in refs))
        self.assertTrue(any(r['status'] == 'external_tensor' for r in refs))
        self.assertNotIn('not_recorded', [r['status'] for r in refs])
        self.assertFalse(any(r['status'] == 'unresolved_tensor' for r in refs))

    def test_tensor_producers_ignore_order_and_preserve_ambiguity(self):
        from version.v8.tools.visualizer_components import _operation_graph
        consumer = {'op_id': 5, 'inputs': {'x': {'tensor': 'act.x'}}}
        producer = {'op_id': 2, 'outputs': {'y': {'tensor': 'act.x'}}}
        graph = _operation_graph({'ops': [consumer, producer]}, 10)
        ref = graph['nodes'][0]['inputs'][0]
        self.assertEqual((ref['status'], ref['producer_id']), ('resolved_tensor', '2'))
        short = _operation_graph({'ops': [consumer, producer]}, 1)
        self.assertEqual(short['nodes'][0]['inputs'][0]['status'], 'producer_outside_capture')
        duplicate = _operation_graph({'ops': [consumer, producer, producer]}, 1)
        self.assertEqual(duplicate['nodes'][0]['inputs'][0]['status'], 'ambiguous')

    def test_external_and_unknown_tensors_are_not_assumed_producers(self):
        from version.v8.tools.visualizer_components import _operation_graph
        data = {'tensors': {'input.x': {'kind': 'input'}}, 'ops': [{'inputs': {
            'x': {'tensor': 'input.x'}, 'missing': {'tensor': 'act.missing'},
            'explicit': {'tensor': 'act.missing', 'from': 'external:supplied'}}}]}
        refs = _operation_graph(data, 10)['nodes'][0]['inputs']
        self.assertEqual([r['status'] for r in refs],
                         ['external_tensor', 'unresolved_tensor', 'external_or_symbolic'])

if __name__=='__main__':unittest.main()
