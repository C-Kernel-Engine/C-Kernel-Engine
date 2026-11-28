from decimal import Decimal
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts.pi_vision_once import (
    MAX_IMAGE_BYTES, assistant_result, main, make_validator, read_image, run, structured_result,
)


PNG = b'\x89PNG\r\n\x1a\nfixture'
OCR_FIXTURES = Path(__file__).resolve().parents[2] / 'tests' / 'fixtures' / 'nano_vl_ocr'


class PiVisionTests(unittest.TestCase):
    def test_approval_bounds_and_file_types(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            outside = parent / 'outside.png'
            outside.write_bytes(PNG)
            root = parent / 'approved'
            root.mkdir()
            (root / 'file-link.png').symlink_to(outside)
            (root / 'dir-link').symlink_to(parent, target_is_directory=True)
            (root / 'wrong.png').write_bytes(b'invalid')
            os.mkfifo(root / 'pipe.png')
            with (root / 'large.png').open('wb') as large:
                large.truncate(MAX_IMAGE_BYTES + 1)
            for path in (str(outside), '../outside.png', 'dir-link/outside.png', 'file-link.png',
                         'wrong.png', 'pipe.png', 'large.png'):
                with self.subTest(path=path), self.assertRaises((ValueError, OSError)):
                    read_image(root, path)
            (root / 'sub').mkdir()
            (root / 'sub' / 'safe.png').write_bytes(PNG)
            self.assertEqual(read_image(root, 'sub/safe.png'), ('.png', PNG))

    def test_stage_only_approved_bytes_and_disable_compaction(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'private.png').write_bytes(PNG)
            scratch = None

            def fake(args, *, cwd, env, **kwargs):
                nonlocal scratch
                scratch = cwd
                self.assertEqual(args[-2:], ['@image.png', 'Read it'])
                self.assertIn('--no-tools', args)
                self.assertIn('--no-context-files', args)
                self.assertEqual((cwd / 'image.png').read_bytes(), PNG)
                agent = Path(env['PI_CODING_AGENT_DIR'])
                settings = json.loads((agent / 'settings.json').read_text())
                self.assertFalse(settings['compaction']['enabled'])
                self.assertFalse(settings['retry']['enabled'])
                model = json.loads((agent / 'models.json').read_text())['providers']['local-cke']
                self.assertEqual(model['baseUrl'], 'http://127.0.0.1:8120/v1')
                self.assertEqual(model['models'][0]['maxTokens'], 160)
                output = json.dumps({'type': 'message_end', 'message': {'role': 'assistant',
                    'content': [{'type': 'text', 'text': 'FORM 42\n'}], 'stopReason': 'stop'}})
                return subprocess.CompletedProcess(args, 0, stdout=output)

            with patch('scripts.pi_vision_once.shutil.which', return_value='/usr/bin/pi'), \
                 patch('scripts.pi_vision_once.subprocess.run', side_effect=fake):
                result = run('pi', 'http://127.0.0.1:8120', 'nemotron-nano-vl', tmp, 'private.png', 'Read it',
                             200, max_tokens=160)
            self.assertEqual(result, {'text': 'FORM 42\n', 'stop_reason': 'stop'})
            self.assertFalse(scratch.exists())

    def test_reject_extra_attachment_and_remote_service_before_launch(self):
        for endpoint, prompt in (('http://127.0.0.1:8120', '@unapproved.png'),
                                 ('http://127.0.0.1:8120', '--tools=read'),
                                 ('http://example.com:8120', 'Read it')):
            with self.subTest(endpoint=endpoint, prompt=prompt), self.assertRaises(ValueError):
                run('pi', endpoint, 'nemotron-nano-vl', '/', 'notice.png', prompt, 30)

    def test_only_assistant_text_keeps_raw_format_and_media_private(self):
        events = '\n'.join(map(json.dumps, [
            {'type': 'message_end', 'message': {'role': 'user', 'content': [{'data': 'private-bytes'}]}},
            {'type': 'message_end', 'message': {'role': 'assistant',
                'content': [{'type': 'text', 'text': 'FORM 42\n'}], 'stopReason': 'length'}},
        ]))
        self.assertEqual(assistant_result(events), {'text': 'FORM 42\n', 'stop_reason': 'length'})
        with self.assertRaises(ValueError):
            assistant_result('{"type":"agent_end"}')

    def test_invalid_output_budgets_fail_before_launch(self):
        for value in (0, -1, 1024, True, 4.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                run('pi', 'http://127.0.0.1:8120', 'test', '/', 'a.png', 'Read it', 1, max_tokens=value)

    def test_single_object_or_fence_only_and_no_truncated_answers(self):
        validator = make_validator({'type': 'object', 'required': ['marked'], 'additionalProperties': False,
                                    'properties': {'marked': {'type': 'boolean'}}})
        for text in ('{"marked":true}', ' \r\n```json\r\n{"marked":true}\r\n```\r\n'):
            result = structured_result({'text': text, 'stop_reason': 'stop'}, validator)
            self.assertEqual(result['text'], text)
            self.assertEqual(result['data'], {'marked': True})
        for text, stop in (
            ('{"marked":true}', 'length'), ('{"marked":true', 'stop'),
            ('```json\n{"marked":true}\n', 'stop'), ('Here it is: {"marked":true}', 'stop'),
            ('{"marked":true}\n{"marked":false}', 'stop'),
            ('```json\n{"marked":true,"marked":false}\n```', 'stop'),
            ('```json\n{"marked":true}\n```\nExtra explanation', 'stop'),
            ('{"marked":1}', 'stop'), ('{"marked":true,"extra":true}', 'stop'),
            ('{"marked":NaN}', 'stop'), ('{"marked":1e999}', 'stop'), ('{}', 'stop'),
        ):
            with self.subTest(text=text, stop=stop), self.assertRaises(ValueError):
                structured_result({'text': text, 'stop_reason': stop}, validator)

    def test_cli_preserves_raw_rejection_without_publishing_data(self):
        argv = ['pi_vision_once.py', '--approved-root', '.', '--image', 'checkbox.png', '--prompt', 'OCR',
                '--max-tokens', '112', '--json-schema', str(OCR_FIXTURES / 'checkbox.schema.json')]
        raw = {'text': '{"email":true,"sms":false,"paper":true}', 'stop_reason': 'length'}
        with patch('sys.argv', argv), patch('scripts.pi_vision_once.run', return_value=raw) as launch, \
             redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(), 1)
        result = json.loads(output.getvalue())
        self.assertEqual(result['text'], raw['text'])
        self.assertNotIn('data', result)
        self.assertIn('validation_error', result)
        self.assertEqual(launch.call_args.kwargs, {'max_tokens': 112})

    def test_schema_references_are_local(self):
        validator = make_validator({'type': 'object', '$defs': {'doc': {'required': ['n']}}, '$ref': '#/$defs/doc'})
        raw = {'text': '{"n":1}', 'stop_reason': 'stop'}
        self.assertEqual(structured_result(raw, validator)['data'], {'n': 1})
        remote = make_validator({'type': 'object', '$ref': 'https://example.invalid/schema'})
        with self.assertRaisesRegex(ValueError, 'local OCR JSON schema'):
            structured_result(raw, remote)

    def test_replay_of_three_public_ocr_captures(self):
        def canonical_amount(value):
            return Decimal(str(value).removeprefix('$'))

        for case in json.loads((OCR_FIXTURES / 'cases.json').read_text()):
            name = case['name']
            with self.subTest(name=name):
                image = OCR_FIXTURES / (name + '.png')
                self.assertEqual(hashlib.sha256(image.read_bytes()).hexdigest(), case['image_sha256'])
                validator = make_validator(json.loads((OCR_FIXTURES / (name + '.schema.json')).read_text()))
                data = structured_result({'text': case['text'], 'stop_reason': case['stop_reason']}, validator)
                self.assertEqual(data['json_source'], 'fenced')
                self.assertEqual(data['text'], case['text'])
                actual, expected = data['data'], case['expected']
                for key in expected:
                    if key == 'items':
                        self.assertEqual({item: canonical_amount(price) for item, price in actual[key].items()},
                                         {item: canonical_amount(price) for item, price in expected[key].items()})
                    elif key in ('amount', 'subtotal', 'tax', 'total'):
                        self.assertEqual(canonical_amount(actual[key]), canonical_amount(expected[key]))
                    else:
                        self.assertEqual(actual[key], expected[key])


if __name__ == '__main__':
    unittest.main()
