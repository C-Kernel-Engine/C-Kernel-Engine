import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('speech_bench', Path(__file__).resolve().parents[2] / 'scripts/whisper_service_benchmark.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.loaded = {s: {'libraries': {'libmodel.so': s}, 'manifest_sha256': 'm',
                          'weights_sha256': 'w', 'worker_pid': i + 100}
                       for i, s in enumerate(('encoder', 'decoder'))}
        self.row = dict(loaded_artifact=self.loaded, text=' Hello.', expected_text='hello.',
                        stop='eos', wall_seconds=2., inference_seconds=1.9, queue_seconds=.01,
                        worker_generation=1, case='hello')

    def assess(self, rows):
        return b.assess(rows, b.identity(self.loaded), {'hello': 5.}, .1)

    def test_valid(self):
        self.assertEqual(self.assess([self.row])['hello']['median_wall_seconds'], 2.)

    def test_slow_and_wrong_transcript(self):
        for change in ({'wall_seconds': 30}, {'text': 'wrong'}, {'stop': 'max_tokens'}, {'inference_seconds': float('nan')}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.assess([dict(self.row, **change)])

    def test_old_library(self):
        row = copy.deepcopy(self.row)
        row['loaded_artifact']['decoder']['libraries']['libmodel.so'] = 'old'
        with self.assertRaises(ValueError):
            self.assess([row])

    def test_worker_restart(self):
        row = copy.deepcopy(self.row)
        row['loaded_artifact']['decoder']['worker_pid'] += 1
        with self.assertRaises(ValueError):
            self.assess([self.row, row])

    def test_busy_not_timing_failure(self):
        with self.assertRaises(RuntimeError):
            self.assess([dict(self.row, queue_seconds=5)])

    def test_config_failure_replaces_stale_pass(self):
        with tempfile.TemporaryDirectory() as td:
            out, cfg = Path(td) / 'latest.json', Path(td) / 'config.json'
            out.write_text('{"status":"PASS"}')
            cfg.write_text('{broken')
            report = b.run(cfg, out)
            self.assertEqual(report['status'], 'FAIL')
            self.assertEqual(json.loads(out.read_text())['status'], 'FAIL')

    def test_pause_replaces_stale_pass_before_configuration_or_requests(self):
        with tempfile.TemporaryDirectory() as td:
            out, pause = Path(td) / 'latest.json', Path(td) / 'PAUSE'
            out.write_text('{"status":"PASS"}')
            pause.touch()
            report = b.run(Path(td) / 'absent-config.json', out, pause_path=pause)
            self.assertEqual(report['status'], 'INCOMPLETE')
            self.assertIn('paused before startup', report['reason'])
            self.assertEqual(report['attempts'], [])
            self.assertEqual(json.loads(out.read_text())['status'], 'INCOMPLETE')
