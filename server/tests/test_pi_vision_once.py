import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts.pi_vision_once import MAX_IMAGE_BYTES, assistant_result, read_image, run


PNG = b'\x89PNG\r\n\x1a\nfixture'


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
                output = json.dumps({'type': 'message_end', 'message': {'role': 'assistant',
                    'content': [{'type': 'text', 'text': 'FORM 42\n'}], 'stopReason': 'stop'}})
                return subprocess.CompletedProcess(args, 0, stdout=output)

            with patch('scripts.pi_vision_once.shutil.which', return_value='/usr/bin/pi'), \
                 patch('scripts.pi_vision_once.subprocess.run', side_effect=fake):
                result = run('pi', 'http://127.0.0.1:8120', 'nemotron-nano-vl', tmp, 'private.png', 'Read it', 200)
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


if __name__ == '__main__':
    unittest.main()
