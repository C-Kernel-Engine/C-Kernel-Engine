import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts.certify_vision_http import assess,run


def good():
    return {'id':'one','object':'chat.completion','choices':[{'finish_reason':'stop',
        'message':{'role':'assistant','content':'{"left":"red","right":"blue"}'}}],
        'usage':{'prompt_tokens':20,'completion_tokens':10,'total_tokens':30}}


class VisionAcceptanceTests(unittest.TestCase):
    def test_correct_complete_answer(self):
        self.assertEqual(json.loads(assess(good(),{'left':'red','right':'blue'})),{'left':'red','right':'blue'})

    def test_wrong_swapped_truncated_duplicate_and_contradictory_answers_fail(self):
        for text in ('{"left":"blue","right":"red"}',
                     '{"left":"red","right":"blue","left":"blue"}',
                     '{"left":"red","right":"blue"} Actually the left is green.',
                     '```json\n{"left":"red","right":"blue"}\n```'):
            response=good();response['choices'][0]['message']['content']=text
            with self.subTest(text=text),self.assertRaises(ValueError):assess(response,{'left':'red','right':'blue'})
        response=good();response['choices'][0]['finish_reason']='length'
        with self.assertRaises(ValueError):assess(response,{'left':'red','right':'blue'})

    def test_bad_usage_and_missing_identity_fail(self):
        for change in ({'id':''},{'usage':{'prompt_tokens':20,'completion_tokens':True,'total_tokens':21}}):
            with self.subTest(change=change),self.assertRaises(ValueError):assess(dict(good(),**change),{'left':'red','right':'blue'})

    def test_current_failure_retained(self):
        with tempfile.TemporaryDirectory() as root:
            output=Path(root)/'new-attempt'
            with patch('scripts.certify_vision_http.identity',side_effect=ValueError('missing boot identity')):
                report=run('unused','test',Path(root),output,'expected')
            self.assertEqual(report['status'],'FAIL')
            self.assertEqual(json.loads((output/'report.json').read_text())['error'],'missing boot identity')

    def test_restart_and_reused_request_id_fail(self):
        initial={'server_instance_id':'boot-one','bundle_identity':'expected','loaded_libraries':{'engine':'hash'}}
        for restarted in (True,False):
            with self.subTest(restarted=restarted),tempfile.TemporaryDirectory() as root:
                directory=Path(root);(directory/'red-blue.png').write_bytes(b'fixture')
                response=good();response['cke_evidence']={'bundle_identity':'expected',
                    'image_sha256':hashlib.sha256(b'fixture').hexdigest()}
                identities=[initial,dict(initial,server_instance_id='new-boot')] if restarted else [initial]*3
                with patch('scripts.certify_vision_http.identity',side_effect=identities),patch('scripts.certify_vision_http.request',return_value=response):
                    report=run('unused','test',directory,directory/'run','expected')
                self.assertEqual(report['status'],'FAIL')
                self.assertIn('service changed' if restarted else 'reused request ID',report['error'])
                self.assertIn('response',report['cases'][-1])

    def test_required_reference_cannot_be_omitted(self):
        with tempfile.TemporaryDirectory() as root:
            directory=Path(root);(directory/'red-blue.png').write_bytes(b'fixture')
            initial={'server_instance_id':'boot','bundle_identity':'expected','loaded_libraries':{'engine':'hash'}}
            response=good();response['cke_evidence']={'bundle_identity':'expected',
                'image_sha256':hashlib.sha256(b'fixture').hexdigest()}
            with patch('scripts.certify_vision_http.identity',return_value=initial),patch('scripts.certify_vision_http.request',return_value=response):
                report=run('unused','test',directory,directory/'run','expected',require_reference=True)
            self.assertEqual(report['status'],'FAIL')
            self.assertIn('missing direct',report['error'])

    def test_timeout_not_pass(self):
        with tempfile.TemporaryDirectory() as root:
            with patch('scripts.certify_vision_http.identity',side_effect=TimeoutError()):
                report=run('unused','test',Path(root),Path(root)/'attempt','expected')
            self.assertEqual(report['status'],'INCOMPLETE')

if __name__=='__main__':unittest.main()
