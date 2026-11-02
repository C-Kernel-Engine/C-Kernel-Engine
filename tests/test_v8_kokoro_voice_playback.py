"""Asset-free Kokoro speech adapter contracts for the accessible voice client."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version/v8/tts/ck_kokoro_play.py"
SPEC = importlib.util.spec_from_file_location("ck_kokoro_play", SCRIPT)
assert SPEC and SPEC.loader
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


class KokoroVoicePlaybackTest(unittest.TestCase):
    def test_private_report_rejects_existing_path(self):
        with tempfile.TemporaryDirectory() as folder:
            report = Path(folder) / "report.json"
            adapter._write_private_report(report, {"source_text": "private"})
            self.assertEqual(os.stat(report).st_mode & 0o777, 0o600)
            self.assertEqual(json.loads(report.read_text()),
                             {"source_text": "private"})
            with self.assertRaises(FileExistsError):
                adapter._write_private_report(report, {"source_text": "changed"})
            self.assertEqual(json.loads(report.read_text()),
                             {"source_text": "private"})

    def test_incremental_order_one_worker_and_middle_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            bundle = root / "bundle"
            bundle.mkdir()
            worker = bundle / "native-kokoro-dispatch"
            launches = root / "launches.txt"
            played = root / "played.txt"
            fail = root / "fail.txt"
            worker.write_text(
                "#!/usr/bin/env python3\nimport sys,wave\nfrom pathlib import Path\n"
                f"with Path({str(launches)!r}).open('a') as f:f.write('load\\n')\n"
                "print('READY\\tloads=1',flush=True)\n"
                "for line in sys.stdin:\n"
                " parts=line.strip().split('\\t')\n"
                " if parts[0]=='QUIT':break\n"
                " if parts[0]!='RUN':print('ERROR\\tcommand',flush=True);continue\n"
                " output=Path(parts[1]);index=int(output.stem.split('-')[1])\n"
                " with wave.open(str(output),'wb') as w:\n"
                "  w.setnchannels(1);w.setsampwidth(2);w.setframerate(24000)\n"
                "  w.writeframes(bytes([index%256,0])*8)\n"
                f" bad=Path({str(fail)!r}).exists() and index==2\n"
                " print(f'WAV_READY\\t{9 if bad else 8}\\t{output}',flush=True)\n")
            worker.chmod(0o700)
            player = root / "player"
            player.write_text(
                "#!/usr/bin/env python3\nimport sys,time,wave\nfrom pathlib import Path\n"
                "p=Path(sys.argv[1])\n"
                "with wave.open(str(p),'rb') as w:assert w.getnframes()==8\n"
                f"with Path({str(played)!r}).open('a') as f:f.write(p.stem+'\\n')\n"
                "time.sleep(0.002)\n")
            player.chmod(0o700)
            def plan(text, _config, _bundle):
                return {"source_text": text, "bundle_identity_sha256": "synthetic",
                        "chunks": [{"index": i, "source_start": i,
                                    "source_end": i + 1, "source_text": "a",
                                    "phoneme_ids": [0, 1, 0]}
                                   for i in range(len(text))]}
            result = adapter.play_incremental("a" * 80, root, bundle, player,
                                              prepare_plan=plan)
            self.assertEqual(result["sample_count"], 640)
            self.assertEqual(len(result["chunks"]), 80)
            self.assertEqual(launches.read_text().splitlines(), ["load"])
            self.assertEqual(played.read_text().splitlines(),
                             [f"chunk-{i}" for i in range(80)])
            fail.touch()
            played.unlink()
            with self.assertRaisesRegex(adapter.PlaybackError, "sample count"):
                adapter.play_incremental("aaaaa", root, bundle, player,
                                         prepare_plan=plan)
            self.assertEqual(launches.read_text().splitlines(), ["load", "load"])
            self.assertNotIn("chunk-3", played.read_text())

    def test_checked_wav_plays_and_invalid_output_never_reaches_player(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            synth = root / "synth.py"
            player = root / "player.py"
            marker = root / "played.txt"
            synth.write_text(
                "import argparse,hashlib,json,wave\n"
                "from pathlib import Path\n"
                "p=argparse.ArgumentParser()\n"
                "for name in ('text-file','config','bundle-dir','output','report'):\n"
                " p.add_argument('--'+name)\n"
                "a=p.parse_args()\n"
                "text=Path(a.text_file).read_text()\n"
                "with wave.open(a.output,'wb') as w:\n"
                " w.setnchannels(1);w.setsampwidth(2);w.setframerate(24000)\n"
                " w.writeframes(b'\\0\\0'*8)\n"
                "Path(a.report).write_text(json.dumps({'status':'execution_pass',"
                "'sample_count': 9 if text == 'bad' else 8,"
                "'source_sha256': hashlib.sha256(text.strip().encode()).hexdigest()}))\n",
                encoding="utf-8")
            player.write_text(
                "#!/usr/bin/env python3\nimport sys\nfrom pathlib import Path\n"
                f"Path({str(marker)!r}).write_text(sys.argv[1])\n",
                encoding="utf-8")
            player.chmod(0o700)
            result = adapter.play("Hello.", root, root, player, synthesizer=synth)
            self.assertEqual(result["sample_count"], 8)
            self.assertFalse(Path(marker.read_text()).exists())
            marker.unlink()
            adapter.play(" Hello. ", root, root, player, synthesizer=synth)
            self.assertTrue(marker.exists())
            marker.unlink()
            with self.assertRaises(adapter.PlaybackError):
                adapter.play("bad", root, root, player, synthesizer=synth)
            self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
