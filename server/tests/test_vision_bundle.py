import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from server.vision_bundle import pin,load

class VisionBundleTests(unittest.TestCase):
    def test_changed_artifact_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'profile.json'
            with patch('server.vision_bundle.sha',return_value='original'):
                path.write_text(json.dumps(pin(directory,directory,'tokens','model','marker')))
                self.assertTrue(load(path)['identity'])
            with patch('server.vision_bundle.sha',return_value='changed'),self.assertRaisesRegex(ValueError,'changed'):
                load(path)
    def test_unrecognized_fields_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'profile.json'
            path.write_text(json.dumps({'schema':'cke.vision.http.v1','extra':1}))
            with self.assertRaises(ValueError):load(path)
