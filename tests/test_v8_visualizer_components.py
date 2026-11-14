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

if __name__=='__main__':unittest.main()
