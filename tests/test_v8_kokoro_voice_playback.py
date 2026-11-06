"""Asset-free Kokoro speech adapter contracts for the accessible voice client."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "version/v8/tts/ck_kokoro_play.py"
SPEC = importlib.util.spec_from_file_location("ck_kokoro_play", SCRIPT)
assert SPEC and SPEC.loader
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


class KokoroVoicePlaybackTest(unittest.TestCase):
    def test_cli_accepts_private_text_file_and_rejects_duplicate_source(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "reply.txt"
            source.write_text("Read /data/cke/report.txt.", encoding="utf-8")
            base = ["ck_kokoro_play.py", "--config", "config", "--bundle-dir",
                    "bundle", "--player", "player", "--incremental",
                    "--text-file", str(source)]
            original_handler = signal.getsignal(signal.SIGTERM)
            try:
                with mock.patch.object(adapter, "play_incremental",
                                       return_value={"sample_count": 5}) as play, \
                     mock.patch.object(sys, "argv", base):
                    self.assertEqual(adapter.main(), 0)
                self.assertEqual(play.call_args.args[0], source.read_text())
                with mock.patch.object(adapter, "play_incremental") as play, \
                     mock.patch.object(sys, "argv", base + ["duplicate"]):
                    self.assertEqual(adapter.main(), 2)
                play.assert_not_called()
            finally:
                signal.signal(signal.SIGTERM, original_handler)

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
                "#!/usr/bin/env python3\nimport os,sys,time,wave\nfrom pathlib import Path\n"
                f"with Path({str(launches)!r}).open('a') as f:f.write('load:'+str(os.getpid())+'\\n')\n"
                "print('READY\\tloads=1',flush=True)\n"
                "for line in sys.stdin:\n"
                " parts=line.strip().split('\\t')\n"
                " if parts[0]=='QUIT':break\n"
                " if parts[0]!='RUN':print('ERROR\\tcommand',flush=True);continue\n"
                " output=Path(parts[1]);index=int(output.stem.split('-')[1])\n"
                " if index%3==0:time.sleep(0.01)\n"
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
            self.assertEqual(result["first_audible_audio"], "not_measured")
            self.assertGreaterEqual(result["plan_seconds"], 0)
            self.assertGreaterEqual(result["worker_startup_seconds"], 0)
            self.assertTrue(all(row["player_process_seconds"] >= 0
                                for row in result["chunks"]))
            self.assertAlmostEqual(
                result["playback_gap_total_seconds"],
                result["synthesis_underrun_total_seconds"] +
                result["handoff_delay_total_seconds"], places=5)
            self.assertEqual(len(launches.read_text().splitlines()), 1)
            self.assertEqual(played.read_text().splitlines(),
                             [f"chunk-{i}" for i in range(80)])
            self.assertEqual(result["synthesis_workers"], 1)
            self.assertEqual(result["native_model_loads"], 1)
            played.unlink()
            result = adapter.play_incremental("a" * 80, root, bundle, player,
                                              synthesis_workers=3, prepare_plan=plan)
            self.assertEqual(result["sample_count"], 640)
            self.assertEqual(result["synthesis_workers"], 3)
            self.assertEqual(result["native_model_loads"], 3)
            self.assertEqual(len({item["wav_sha256"] for item in result["chunks"]}), 80)
            self.assertEqual(len(launches.read_text().splitlines()), 4)
            self.assertEqual(played.read_text().splitlines(),
                             [f"chunk-{i}" for i in range(80)])
            fail.touch()
            played.unlink()
            with self.assertRaisesRegex(adapter.PlaybackError, "sample count"):
                adapter.play_incremental("aaaaa", root, bundle, player,
                                         synthesis_workers=3, prepare_plan=plan)
            self.assertEqual(len(launches.read_text().splitlines()), 7)
            self.assertNotIn("chunk-3", played.read_text())
            with self.assertRaisesRegex(adapter.PlaybackError, "consecutive indices"):
                adapter.play_incremental("aa", root, bundle, player,
                    synthesis_workers=3, prepare_plan=lambda *_: {
                        "source_text": "aa", "chunks": [
                            {"index": 0, "phoneme_ids": [0, 1, 0]},
                            {"index": 0, "phoneme_ids": [0, 1, 0]}]})
            self.assertEqual(len(launches.read_text().splitlines()), 7)
            played.unlink()
            with mock.patch.object(adapter, "_exchange", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    adapter.play_incremental("aaa", root, bundle, player,
                                             synthesis_workers=3, prepare_plan=plan)
            self.assertFalse(played.exists())
            launched = launches.read_text().splitlines()
            self.assertEqual(len(launched), 10)
            for entry in launched[-3:]:
                with self.assertRaises(ProcessLookupError):
                    os.kill(int(entry.split(":")[1]), 0)
            fail.unlink()
            one = adapter.play_incremental("a", root, bundle, player,
                                           synthesis_workers=3, prepare_plan=plan)
            self.assertEqual(one["synthesis_workers"], 1)
            self.assertEqual(one["native_model_loads"], 1)
            self.assertEqual(len(launches.read_text().splitlines()), 11)
            exact_plan = mock.Mock(side_effect=lambda text, config, bundle,
                                   *, technical_mode:
                                   {**plan(text, config, bundle),
                                    "pronunciation_expansions":
                                        [{"mode": technical_mode}]})
            exact = adapter.play_incremental("a", root, bundle, player,
                technical_mode="exact", prepare_plan=exact_plan)
            self.assertEqual(exact_plan.call_args.kwargs, {"technical_mode": "exact"})
            self.assertEqual(exact["technical_mode"], "exact")
            self.assertEqual(exact["pronunciation_expansions"], [{"mode": "exact"}])
            document = adapter.play_incremental("a", root, bundle, player,
                document=True, prepare_plan=plan)
            self.assertEqual(document["sample_count"], 8)
            self.assertEqual(document["native_model_loads"], 1)
            progress_path = root / "progress.json"
            def durable_plan(text, config, bundle):
                return {**plan(text, config, bundle),
                        "bundle_manifest_sha256": "manifest",
                        "frontend_sha256": {"pinned": "frontend"},
                        "voice": "af_heart"}
            durable = adapter.play_incremental("a", root, bundle, player,
                document=True, prepare_plan=durable_plan,
                progress_file=progress_path, session_id="document-1")
            self.assertEqual(durable["completed_samples"], 8)
            self.assertEqual(json.loads(progress_path.read_text())[
                "last_completed_chunk"], 0)
            launches_before_resume = len(launches.read_text().splitlines())
            done = adapter.play_incremental("a", root, bundle, player,
                document=True, prepare_plan=durable_plan,
                progress_file=progress_path, session_id="document-1", resume=True)
            self.assertEqual(done["status"], "already_complete")
            self.assertEqual(len(launches.read_text().splitlines()),
                             launches_before_resume)
            malformed_path = root / "malformed-progress.json"
            malformed_path.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(adapter.PlaybackError, "progress file is invalid"):
                adapter.play_incremental("a", root, bundle, player,
                    document=True, prepare_plan=durable_plan,
                    progress_file=malformed_path, resume=True)
            with self.assertRaisesRegex(adapter.PlaybackError, "identity differs"):
                adapter.play_incremental("a", root, bundle, player,
                    document=True, prepare_plan=durable_plan,
                    progress_file=progress_path, session_id="document-1",
                    seed=8, resume=True)
            with self.assertRaisesRegex(adapter.PlaybackError, "identity differs"):
                adapter.play_incremental("b", root, bundle, player,
                    document=True, prepare_plan=durable_plan,
                    progress_file=progress_path, session_id="document-1",
                    resume=True)
            interrupted_path = root / "interrupted-progress.json"
            fail.touch()
            with self.assertRaisesRegex(adapter.PlaybackError, "sample count"):
                adapter.play_incremental("aaaa", root, bundle, player,
                    document=True, synthesis_workers=3,
                    prepare_plan=durable_plan, progress_file=interrupted_path,
                    session_id="document-2")
            interrupted = json.loads(interrupted_path.read_text())
            self.assertEqual(interrupted["last_completed_chunk"], 0)
            fail.unlink()
            played.unlink()
            resumed = adapter.play_incremental("aaaa", root, bundle, player,
                document=True, synthesis_workers=3,
                prepare_plan=durable_plan, progress_file=interrupted_path,
                session_id="document-2", resume=True)
            self.assertEqual(resumed["resume_from_chunk"], 1)
            self.assertEqual([row["index"] for row in resumed["chunks"]],
                             [1, 2, 3])
            self.assertEqual(json.loads(interrupted_path.read_text())[
                "last_completed_chunk"], 3)
            with self.assertRaisesRegex(adapter.PlaybackError, "integer in 1..3"):
                adapter.play_incremental("a", root, bundle, player,
                                         synthesis_workers=4, prepare_plan=plan)
            self.assertEqual(len(launches.read_text().splitlines()), 20)

    def test_parallel_sigterm_reaps_workers_and_active_player(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            bundle = root / "bundle"
            bundle.mkdir()
            worker_pids = root / "worker-pids"
            worker_child_pids = root / "worker-child-pids"
            player_pid = root / "player-pid"
            finished = root / "finished"
            worker = bundle / "native-kokoro-dispatch"
            worker.write_text(
                "#!/usr/bin/env python3\nimport os,subprocess,sys,time,wave\n"
                "from pathlib import Path\n"
                f"with Path({str(worker_pids)!r}).open('a') as f:f.write(str(os.getpid())+'\\n')\n"
                "child=subprocess.Popen([sys.executable,'-c',"
                "'import time;time.sleep(30)'])\n"
                f"with Path({str(worker_child_pids)!r}).open('a') as f:"
                "f.write(str(child.pid)+'\\n')\n"
                "print('READY\\tloads=1',flush=True)\n"
                "for line in sys.stdin:\n"
                " parts=line.strip().split('\\t')\n"
                " if parts[0]=='QUIT':break\n"
                " if parts[0]!='RUN':raise SystemExit(2)\n"
                " time.sleep(0.05)\n"
                " with wave.open(parts[1],'wb') as w:\n"
                "  w.setnchannels(1);w.setsampwidth(2);w.setframerate(24000)\n"
                "  w.writeframes(b'\\0\\0'*8)\n"
                " print('WAV_READY\\t8\\t'+parts[1],flush=True)\n")
            worker.chmod(0o700)
            player = root / "player"
            player.write_text(
                "#!/usr/bin/env python3\nimport os,time\nfrom pathlib import Path\n"
                f"Path({str(player_pid)!r}).write_text(str(os.getpid()))\n"
                "time.sleep(30)\n")
            player.chmod(0o700)
            runner = root / "runner.py"
            runner.write_text(
                "import signal,sys\nfrom pathlib import Path\n"
                f"sys.path.insert(0,{str(SCRIPT.parent)!r})\n"
                "from ck_kokoro_play import play_incremental\n"
                "def stop(*_):raise KeyboardInterrupt\n"
                "signal.signal(signal.SIGTERM,stop)\n"
                "def plan(text,*_):\n"
                " return {'source_text':text,'bundle_identity_sha256':'synthetic',"
                "'chunks':[{'index':i,'source_start':i,'source_end':i+1,"
                "'source_text':'a','phoneme_ids':[0,1,0]} for i in range(5)]}\n"
                "try:\n"
                f" play_incremental('aaaaa',Path({str(root)!r}),"
                f"Path({str(bundle)!r}),Path({str(player)!r}),"
                "synthesis_workers=3,prepare_plan=plan)\n"
                "except KeyboardInterrupt:raise SystemExit(130)\n"
                f"Path({str(finished)!r}).write_text('done')\n")
            child = subprocess.Popen([sys.executable, str(runner)])
            try:
                deadline = time.monotonic() + 5
                while not player_pid.exists() and time.monotonic() < deadline:
                    self.assertIsNone(child.poll(), "playback runner exited early")
                    time.sleep(0.01)
                self.assertTrue(player_pid.exists(), "player did not start")
                child.send_signal(signal.SIGTERM)
                self.assertEqual(child.wait(timeout=5), 130)
                self.assertFalse(finished.exists())
                for pid in [*map(int, worker_pids.read_text().splitlines()),
                            int(player_pid.read_text())]:
                    with self.assertRaises(ProcessLookupError):
                        os.kill(pid, 0)
                for pid in map(int, worker_child_pids.read_text().splitlines()):
                    deadline = time.monotonic() + 2
                    while time.monotonic() < deadline:
                        state_file = Path(f"/proc/{pid}/stat")
                        if (not state_file.exists() or
                                state_file.read_text().split()[2] == "Z"):
                            break
                        time.sleep(0.01)
                    else:
                        self.fail("native worker child remained active after stop")
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait()

    @unittest.skipUnless(sys.platform.startswith("linux"),
                         "player process-group contract uses Linux /proc")
    def test_stop_kills_player_wrapper_and_its_audio_child(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            child_pid_file = root / "audio-child-pid"
            player = root / "player-wrapper"
            player.write_text(
                "#!/usr/bin/env python3\nimport subprocess,sys,time\n"
                "from pathlib import Path\n"
                "child=subprocess.Popen([sys.executable,'-c',"
                "'import time;time.sleep(30)'])\n"
                f"Path({str(child_pid_file)!r}).write_text(str(child.pid))\n"
                "time.sleep(30)\n")
            player.chmod(0o700)
            playing = adapter._Playing(player, root / "ignored.wav")
            try:
                deadline = time.monotonic() + 5
                while not child_pid_file.exists() and time.monotonic() < deadline:
                    self.assertIsNone(playing.child.poll(), "player exited early")
                    time.sleep(0.01)
                self.assertTrue(child_pid_file.exists(), "audio child did not start")
                child_pid = int(child_pid_file.read_text())
                playing.stop()
                self.assertIsNotNone(playing.child.poll())
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    state_file = Path(f"/proc/{child_pid}/stat")
                    if not state_file.exists() or state_file.read_text().split()[2] == "Z":
                        break
                    time.sleep(0.01)
                else:
                    self.fail("player child remained active after stop")
                playing.stop()
            finally:
                playing.stop()

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
                "p.add_argument('--technical-mode',default='natural')\n"
                "a=p.parse_args()\n"
                "text=Path(a.text_file).read_text()\n"
                "with wave.open(a.output,'wb') as w:\n"
                " w.setnchannels(1);w.setsampwidth(2);w.setframerate(24000)\n"
                " w.writeframes(b'\\0\\0'*8)\n"
                "Path(a.report).write_text(json.dumps({'status':'execution_pass',"
                "'sample_count': 9 if text == 'bad' else 8,"
                "'source_sha256': hashlib.sha256(text.strip().encode()).hexdigest(),"
                "'technical_mode':a.technical_mode}))\n",
                encoding="utf-8")
            player.write_text(
                "#!/usr/bin/env python3\nimport sys\nfrom pathlib import Path\n"
                f"Path({str(marker)!r}).write_text(sys.argv[1])\n",
                encoding="utf-8")
            player.chmod(0o700)
            result = adapter.play("Hello.", root, root, player, synthesizer=synth)
            self.assertEqual(result["sample_count"], 8)
            self.assertEqual(result["technical_mode"], "natural")
            self.assertFalse(Path(marker.read_text()).exists())
            marker.unlink()
            adapter.play(" Hello. ", root, root, player, synthesizer=synth)
            self.assertTrue(marker.exists())
            marker.unlink()
            exact = adapter.play("/data/file", root, root, player,
                                 synthesizer=synth, technical_mode="exact")
            self.assertEqual(exact["technical_mode"], "exact")
            marker.unlink()
            with self.assertRaises(adapter.PlaybackError):
                adapter.play("bad", root, root, player, synthesizer=synth)
            self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
