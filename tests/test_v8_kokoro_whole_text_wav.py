"""CKE whole-text command contract with a synthetic persistent native worker."""

import importlib.util
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch
import wave


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "version/v8/tts/speak_kokoro_text.py"
HOST = ROOT / "version/v8/tts/native_kokoro_wav_host.c"


class WholeTextWavTest(unittest.TestCase):
    def test_native_chunk_limit_matches_text_planner(self):
        match = re.search(r"#define CK_TTS_MAX_SEQUENCE_CHUNKS (\d+)u",
                          HOST.read_text())
        self.assertIsNotNone(match)
        import sys
        sys.path.insert(0, str(SOURCE.parent))
        try:
            from prepare_kokoro_accessibility_text import MAX_CHUNKS
        finally:
            sys.path.pop(0)
        self.assertEqual(int(match.group(1)), MAX_CHUNKS)

    def test_many_chunks_and_worker_failure(self):
        import sys
        sys.path.insert(0, str(SOURCE.parent))
        try:
            spec = importlib.util.spec_from_file_location("speak_kokoro_text", SOURCE)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        finally:
            sys.path.pop(0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle"
            bundle.mkdir()
            worker = bundle / "native-kokoro-dispatch"
            worker.write_text('''#!/usr/bin/env python3
import sys, wave
print('READY\\tloads=1', flush=True)
output = None
samples = 0
for line in sys.stdin:
    parts = line.strip().split('\\t')
    if parts[0] == 'BEGIN':
        output = parts[1]
        print('BEGUN', flush=True)
    elif parts[0] == 'APPEND':
        if parts[1] == '99':
            print('ERROR\\t2', flush=True)
            continue
        samples += 20
        print(f'CHUNK_READY\\t20\\t{samples}', flush=True)
    elif parts[0] == 'END':
        with wave.open(output, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(b'\\0\\0' * samples)
        print(f'WAV_READY\\t{samples}\\t{output}', flush=True)
    elif parts[0] == 'QUIT':
        break
''')
            worker.chmod(0o755)
            plan = {"chunks": [
                {"index": index, "source_start": index,
                 "source_end": index + 1, "source_text": "abc"[index],
                 "phoneme_ids": [0, 50, 0], "valid_token_count": 3,
                 "voice_row": 0}
                for index in range(3)],
                "source_text": "abc",
                "bundle_identity_sha256": "synthetic",
                "bundle_manifest_sha256": "synthetic"}
            with patch.object(module, "prepare", return_value=plan):
                output = root / "whole.wav"
                result = module.synthesize("abc", root / "config", bundle,
                                           output, 7)
                self.assertEqual(result["sample_count"], 60)
                self.assertEqual(len(result["chunks"]), 3)
                with wave.open(str(output), "rb") as audio:
                    self.assertEqual(audio.getnframes(), 60)
                self.assertEqual(list(root.glob(".kokoro-speech-*")), [])
                long_text = "x" * 250
                long_plan = {**plan, "source_text": long_text,
                    "chunks": [{"index": index, "source_start": index,
                                "source_end": index + 1, "source_text": "x",
                                "phoneme_ids": [0, 50, 0],
                                "valid_token_count": 3, "voice_row": 0}
                               for index in range(250)]}
                with patch.object(module, "prepare", return_value=long_plan):
                    long_output = root / "long.wav"
                    long_result = module.synthesize(long_text, root / "config",
                                                    bundle, long_output, 1000)
                self.assertEqual(len(long_result["chunks"]), 250)
                self.assertEqual(long_result["sample_count"], 5000)
                self.assertEqual(long_result["chunks"][-1]["source_end"], 250)
                with wave.open(str(long_output), "rb") as audio:
                    self.assertEqual(audio.getnframes(), 5000)
                with patch.object(module, "prepare", return_value={
                        **plan, "chunks": plan["chunks"] * 171}):
                    with self.assertRaisesRegex(module.SpeechError, "at most 512 chunks"):
                        module.synthesize("abc", root / "config", bundle,
                                          root / "too-long.wav", 7)
                self.assertFalse((root / "too-long.wav").exists())
                with self.assertRaisesRegex(module.SpeechError, "native worker rejected APPEND"):
                    module.synthesize("abc", root / "config", bundle,
                                      root / "failed.wav", 97)
                self.assertFalse((root / "failed.wav").exists())
                self.assertEqual(list(root.glob(".kokoro-speech-*")), [])


if __name__ == "__main__":
    unittest.main()
