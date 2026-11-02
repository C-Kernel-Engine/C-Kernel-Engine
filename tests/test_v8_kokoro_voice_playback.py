"""Asset-free Kokoro speech adapter contracts for the accessible voice client."""

from __future__ import annotations

import importlib.util
import json
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
