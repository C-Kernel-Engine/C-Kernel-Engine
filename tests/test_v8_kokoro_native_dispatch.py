"""Cheap native exact-length selector and artifact-identity controls."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "version/v8/tts/native_kokoro_dispatch.c"
sys.path.insert(0, str(ROOT / "version/v8/tts"))
import build_kokoro_dispatch_bundle as bundle_builder
from build_kokoro_request_runtime import select_conv_provider
from build_kokoro_waveform_circuit import build_circuit


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class KokoroNativeDispatchTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        voice = self.root / "af_heart.ckvoice"
        voice.write_bytes(b"synthetic voice bank")
        self.voice_hash = _hash(voice)
        records = []
        for count in (8, 36):
            directory = self.root / str(count)
            directory.mkdir()
            host = directory / "native-kokoro-wav"
            host.write_text(
                "#!/bin/sh\n"
                f"printf 'bucket={count}:%s:%s\\n' \"$3\" \"$4\" > \"$2\"\n"
            )
            host.chmod(0o755)
            library = directory / "generated.so"
            library.write_bytes(f"synthetic library {count}".encode())
            package = directory / "request.cktts"
            package.write_bytes(f"synthetic package {count}".encode())
            records.append((count, _hash(host), _hash(library), _hash(package)))
        self.assertNotEqual(records[0][1], records[1][1])
        header = [
            "#ifndef CKE_KOKORO_BUNDLE_IDENTITY_H",
            "#define CKE_KOKORO_BUNDLE_IDENTITY_H",
            "typedef struct { unsigned count; const char *host_sha256; "
            "const char *library_sha256; const char *package_sha256; } ck_tts_bucket;",
            f'static const char ck_tts_voice_sha256[] = "{self.voice_hash}";',
            "static const ck_tts_bucket ck_tts_buckets[] = {",
        ]
        header.extend(f'{{{count}, "{host}", "{library}", "{package}"}},'
                      for count, host, library, package in records)
        header.extend(["};", "#endif"])
        (self.root / "bundle_identity.h").write_text("\n".join(header))
        self.binary = self.root / "dispatch"
        subprocess.run(["cc", "-std=c11", "-O2", "-Wall", "-Wextra",
            "-Werror", "-pedantic", "-I", str(self.root), str(SOURCE),
            "-lcrypto", "-o", str(self.binary)], check=True,
            capture_output=True, text=True)

    def invoke(self, ids, name="out.wav"):
        output = self.root / name
        result = subprocess.run([str(self.binary), str(self.root), str(output),
            "7", ids], capture_output=True, text=True)
        return result, output

    def test_bundle_builder_rejects_bad_voice_and_occupied_output(self):
        output = self.root / "new-bundle"
        with self.assertRaisesRegex(ValueError, "voice bank differs"):
            bundle_builder.build_bundle(self.root / "missing-bump",
                                        self.root / "af_heart.ckvoice", output)
        self.assertFalse(output.exists())
        output.mkdir()
        sentinel = output / "keep.txt"
        sentinel.write_text("keep")
        with self.assertRaisesRegex(ValueError, "must be empty"):
            bundle_builder.build_bundle(self.root / "missing-bump",
                                        self.root / "af_heart.ckvoice", output)
        self.assertEqual(sentinel.read_text(), "keep")

    def test_experimental_counts_validate_before_export(self):
        self.assertEqual(bundle_builder.selected_counts(()), (8, 36))
        self.assertEqual(bundle_builder.selected_counts((12,)), (8, 12, 36))
        self.assertEqual(bundle_builder.selected_runtime_counts((12,), False),
                         (8, 12, 13, 36))
        self.assertEqual(bundle_builder.selected_runtime_counts((), True),
                         tuple(range(3, 41)))
        for counts in ((8,), (36,), (12, 12), (2,), (41,), (True,),
                       (12.0,), (3, 4, 5, 6, 7)):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                bundle_builder.build_bundle(self.root / "missing-bump",
                    self.root / "af_heart.ckvoice", self.root / "not-published",
                    experimental_counts=counts)
            self.assertFalse((self.root / "not-published").exists())
        for counts in ((8,), (13,), (36,), (12, 12), (2,), (41,),
                       (True,), (12.0,), (3, 4, 5, 6, 7)):
            with self.subTest(runtime_counts=counts), self.assertRaises(ValueError):
                bundle_builder.build_bundle(self.root / "missing-bump",
                    self.root / "af_heart.ckvoice", self.root / "not-published",
                    runtime_lengths=True, experimental_counts=counts)
        with self.assertRaisesRegex(ValueError, "range requires"):
            bundle_builder.build_bundle(self.root / "missing-bump",
                self.root / "af_heart.ckvoice", self.root / "not-published",
                runtime_experimental_range=True)
        with self.assertRaisesRegex(ValueError, "range or explicit"):
            bundle_builder.build_bundle(self.root / "missing-bump",
                self.root / "af_heart.ckvoice", self.root / "not-published",
                runtime_lengths=True, runtime_experimental_range=True,
                experimental_counts=(12,))
        self.assertFalse((self.root / "not-published").exists())

    def test_opt_in_third_bucket_routes_distinguishable_native_host(self):
        voice = self.root / "valid-voice.ckvoice"
        voice.write_bytes(b"CKVOICE1" + (34).to_bytes(4, "little") +
                          (256).to_bytes(4, "little") +
                          b"\0" * (34 * 256 * 4))
        bindings = self.root / "bindings.json"
        bindings.write_text(json.dumps({
            "native_voice_sha256": _hash(voice),
            "source_bump_sha256": "synthetic",
        }))
        built = []

        def fake_runtime(_bump, directory, count, *, conv_provider):
            self.assertEqual(conv_provider, "scalar")
            built.append(count)
            directory.mkdir()
            host = directory / "native-kokoro-wav"
            host.write_text("#!/bin/sh\n" +
                            f"printf 'bucket={count}:%s\\n' \"$4\" > \"$2\"\n")
            host.chmod(0o755)
            library = directory / "generated.so"
            library.write_bytes(f"library {count}".encode())
            package = directory / "request.cktts"
            package.write_bytes(f"package {count}".encode())
            return {
                "host_sha256": _hash(host),
                "library_sha256": _hash(library),
                "package_sha256": _hash(package),
                "compiler_manifest_sha256": "synthetic",
                "generated_c_sha256": "synthetic",
                "generated_operations": 1,
                "selected_conv_calls": 0,
            }

        bundle = self.root / "three-bucket-bundle"
        with mock.patch.object(bundle_builder, "DEFAULT_BINDINGS", bindings), \
             mock.patch.object(bundle_builder, "build_runtime",
                               side_effect=fake_runtime):
            report = bundle_builder.build_bundle(self.root / "synthetic-bump",
                voice, bundle, experimental_counts=(12,))
        self.assertEqual(built, [8, 12, 36])
        self.assertEqual(report["supported_phoneme_counts"], [8, 12, 36])
        self.assertEqual(report["baseline_phoneme_counts"], [8, 36])
        self.assertEqual(report["experimental_phoneme_counts"], [12])
        ids = ",".join(map(str, [0] + [4] * 10 + [0]))
        output = self.root / "third-bucket.wav"
        result = subprocess.run([str(bundle / "native-kokoro-dispatch"),
            str(bundle), str(output), "7", ids], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(output.read_text().strip(), f"bucket=12:{ids}")
        unsupported = self.root / "unsupported-third.wav"
        result = subprocess.run([str(bundle / "native-kokoro-dispatch"),
            str(bundle), str(unsupported), "7", ids + ",0"],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertFalse(unsupported.exists())
        session = subprocess.run([str(bundle / "native-kokoro-dispatch"),
            str(bundle), '--session'], capture_output=True, text=True)
        self.assertEqual(session.returncode, 2)
        self.assertIn('runtime-length bundle', session.stderr)

    def test_runtime_bundle_routes_three_lengths_to_one_generated_artifact(self):
        voice = self.root / "runtime-voice.ckvoice"
        voice.write_bytes(b"CKVOICE1" + (38).to_bytes(4, "little") +
                          (256).to_bytes(4, "little") +
                          b"\0" * (38 * 256 * 4))
        bindings = self.root / "runtime-bindings.json"
        bindings.write_text(json.dumps({
            "native_voice_sha256": _hash(voice),
            "source_bump_sha256": "synthetic",
        }))
        built = []

        def fake_runtime(_bump, directory, count, *, conv_provider,
                         runtime_lengths):
            self.assertEqual((count, conv_provider, runtime_lengths),
                             (64, "scalar", True))
            built.append(count)
            directory.mkdir()
            host = directory / "native-kokoro-wav"
            host.write_text("#!/bin/sh\n"
                            "if [ \"$2\" = --session ]; then "
                            "printf 'session:%s:%s\\n' \"$1\" \"$3\"; "
                            "else printf 'runtime:%s\\n' \"$4\" > \"$2\"; fi\n")
            host.chmod(0o755)
            library = directory / "generated.so"
            library.write_bytes(b"one synthetic runtime library")
            package = directory / "request.cktts"
            package.write_bytes(b"one synthetic runtime package")
            return {"host_sha256": _hash(host),
                    "library_sha256": _hash(library),
                    "package_sha256": _hash(package),
                    "compiler_manifest_sha256": "synthetic",
                    "generated_c_sha256": "synthetic",
                    "generated_operations": 1, "selected_conv_calls": 0}

        output = self.root / "runtime-bundle"
        with mock.patch.object(bundle_builder, "DEFAULT_BINDINGS", bindings), \
             mock.patch.object(bundle_builder, "build_runtime",
                               side_effect=fake_runtime):
            report = bundle_builder.build_bundle(self.root / "synthetic-bump",
                voice, output, runtime_lengths=True)
        self.assertEqual(built, [64])
        self.assertEqual(report["supported_phoneme_counts"], [8, 13, 36])
        self.assertTrue(report["runtime_lengths"])
        self.assertEqual(len({item["library_sha256"] for item in
                              report["buckets"].values()}), 1)
        session = subprocess.run([str(output / 'native-kokoro-dispatch'),
            str(output), '--session'], capture_output=True, text=True)
        self.assertEqual(session.returncode, 0, session.stderr)
        self.assertEqual(session.stdout.strip(),
            f"session:{output / '64/request.cktts'}:{output / 'af_heart.ckvoice'}")
        for count in (8, 13, 36, 8):
            ids = ",".join(map(str, [0] + [4] * (count - 2) + [0]))
            wav = self.root / f"runtime-{count}.wav"
            wav.unlink(missing_ok=True)
            run = subprocess.run([str(output / "native-kokoro-dispatch"),
                str(output), str(wav), "7", ids], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(wav.read_text().strip(), f"runtime:{ids}")
        ids = ",".join(map(str, [0] + [4] * 10 + [0]))
        rejected = self.root / "unsupported-runtime.wav"
        run = subprocess.run([str(output / "native-kokoro-dispatch"),
            str(output), str(rejected), "7", ids], capture_output=True, text=True)
        self.assertEqual(run.returncode, 2)
        self.assertFalse(rejected.exists())
        library = output / "64" / "generated.so"
        library.write_bytes(library.read_bytes() + b"tamper")
        allowed = ",".join(map(str, [0] + [4] * 6 + [0]))
        run = subprocess.run([str(output / "native-kokoro-dispatch"),
            str(output), str(rejected), "7", allowed],
            capture_output=True, text=True)
        self.assertEqual(run.returncode, 4)
        self.assertIn("artifact identity mismatch", run.stderr)
        self.assertFalse(rejected.exists())
        session = subprocess.run([str(output / 'native-kokoro-dispatch'),
            str(output), '--session'], capture_output=True, text=True)
        self.assertEqual(session.returncode, 4)
        range_output = self.root / "runtime-range-bundle"
        with mock.patch.object(bundle_builder, "DEFAULT_BINDINGS", bindings), \
             mock.patch.object(bundle_builder, "build_runtime",
                               side_effect=fake_runtime):
            range_report = bundle_builder.build_bundle(
                self.root / "synthetic-bump", voice, range_output,
                runtime_lengths=True, runtime_experimental_range=True)
        self.assertEqual(built, [64, 64])
        self.assertEqual(range_report["captured_phoneme_counts"], [8, 13, 36])
        self.assertEqual(range_report["execution_only_phoneme_counts"],
                         [count for count in range(3, 41)
                          if count not in (8, 13, 36)])
        short_voice = self.root / "short-range-voice.ckvoice"
        short_voice.write_bytes(b"CKVOICE1" + (37).to_bytes(4, "little") +
                                (256).to_bytes(4, "little") +
                                b"\0" * (37 * 256 * 4))
        short_bindings = self.root / "short-range-bindings.json"
        short_bindings.write_text(json.dumps({
            "native_voice_sha256": _hash(short_voice),
            "source_bump_sha256": "synthetic",
        }))
        with mock.patch.object(bundle_builder, "DEFAULT_BINDINGS", short_bindings), \
             self.assertRaisesRegex(ValueError, "voice bank geometry"):
            bundle_builder.build_bundle(self.root / "synthetic-bump",
                short_voice, self.root / "rejected-range-bundle",
                runtime_lengths=True, runtime_experimental_range=True)
        self.assertFalse((self.root / "rejected-range-bundle").exists())
        self.assertEqual(len({item["library_sha256"] for item in
                              range_report["buckets"].values()}), 1)
        for count in (3, 12, 40, 8, 12):
            ids = ",".join(map(str, [0] + [4] * (count - 2) + [0]))
            wav = self.root / f"range-{count}.wav"
            wav.unlink(missing_ok=True)
            run = subprocess.run([str(range_output / "native-kokoro-dispatch"),
                str(range_output), str(wav), "7", ids],
                capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(wav.read_text().strip(), f"runtime:{ids}")
        for count in (2, 41):
            ids = ",".join(map(str, [0] + [4] * (count - 2) + [0]))
            wav = self.root / f"rejected-{count}.wav"
            run = subprocess.run([str(range_output / "native-kokoro-dispatch"),
                str(range_output), str(wav), "7", ids],
                capture_output=True, text=True)
            self.assertEqual(run.returncode, 2)
            self.assertFalse(wav.exists())

    def test_explicit_avx2_selection_preserves_graph_ownership(self):
        scalar = build_circuit(8)
        candidate = build_circuit(8)
        self.assertEqual(select_conv_provider(scalar, "scalar"), 0)
        self.assertEqual(select_conv_provider(candidate, "avx2_fma"), 88)
        self.assertEqual(candidate["required_numerical_contracts"][
            "audio_conv1d_checked"]["phases"]["prefill"]["contract_id"],
            "audio_conv1d_checked_x86_avx2_fma_fp32")
        selected = [op["kernel"] for block in candidate["block_types"].values()
                    for op in block["header"] + block["body"]["ops"] +
                    block["footer"] if op.get("kernel", "").endswith("_avx2_fma")]
        self.assertEqual(len(selected), 88)
        self.assertIn("audio_conv1d_checked_channel_major_f32_avx2_fma", selected)
        self.assertIn("audio_conv1d_dilated_checked_channel_major_f32_avx2_fma", selected)
        self.assertEqual(select_conv_provider(build_circuit(36), "avx2_fma"), 88)
        parallel = build_circuit(8)
        self.assertEqual(select_conv_provider(parallel, "avx2_parallel", workers=4), 88)
        self.assertEqual(parallel["required_numerical_contracts"]["audio_conv1d_checked"]
                         ["phases"]["prefill"]["contract_id"],
                         "audio_conv1d_checked_x86_avx2_parallel_fp32")
        self.assertEqual(parallel["native_entry"]["params"][-1],
                         {"c_type": "void *", "name": "worker_pool"})
        parallel_ops = [op for block in parallel["block_types"].values()
                        for op in block["header"] + block["body"]["ops"] + block["footer"]
                        if op.get("kernel", "").endswith("_avx2_parallel")]
        self.assertEqual(len(parallel_ops), 88)
        self.assertTrue(all(op["params"]["conv_worker_pool"] == "worker_pool" and
                            op["params"]["conv_worker_count"] == 4 for op in parallel_ops))
        for workers in (0, 17):
            with self.assertRaisesRegex(ValueError, "worker"):
                select_conv_provider(build_circuit(8), "avx2_parallel", workers=workers)
            with self.assertRaisesRegex(ValueError, "worker"):
                bundle_builder.build_bundle(self.root / "missing", self.root /
                    "af_heart.ckvoice", self.root / "not-published",
                    conv_provider="avx2_parallel", conv_workers=workers)
        with self.assertRaisesRegex(ValueError, "unsupported Kokoro Conv1D"):
            select_conv_provider(build_circuit(8), "avx512_vnni")
        with self.assertRaisesRegex(ValueError, "unsupported Kokoro Conv1D"):
            bundle_builder.build_bundle(self.root / "missing", self.root /
                "af_heart.ckvoice", self.root / "not-published",
                conv_provider="avx512_vnni")
        self.assertFalse((self.root / "not-published").exists())

    def test_failed_build_does_not_publish_partial_bundle(self):
        voice = self.root / "valid.ckvoice"
        voice.write_bytes(b"CKVOICE1" + (34).to_bytes(4, "little") +
                          (256).to_bytes(4, "little") +
                          b"\0" * (34 * 256 * 4))
        bindings = self.root / "bindings.json"
        bindings.write_text(json.dumps({
            "native_voice_sha256": _hash(voice),
            "source_bump_sha256": "synthetic",
        }))
        output = self.root / "failed-build"
        output.mkdir()

        def fail_after_first_bucket(_bump, directory, count, *, conv_provider):
            self.assertEqual(conv_provider, "scalar")
            directory.mkdir()
            (directory / "partial").write_text("not deployable")
            if count == 36:
                raise RuntimeError("second bucket build failed")
            return {key: "synthetic" for key in (
                "library_sha256", "host_sha256", "package_sha256",
                "compiler_manifest_sha256", "generated_c_sha256",
                "generated_operations")} | {"selected_conv_calls": 0}

        with mock.patch.object(bundle_builder, "DEFAULT_BINDINGS", bindings), \
             mock.patch.object(bundle_builder, "build_runtime",
                               side_effect=fail_after_first_bucket):
            with self.assertRaisesRegex(RuntimeError,
                                        "second bucket build failed"):
                bundle_builder.build_bundle(self.root / "synthetic-bump",
                                            voice, output)
        self.assertEqual(list(output.iterdir()), [])
        self.assertEqual(list(self.root.glob(".failed-build.staging-*")), [])

    def test_selects_exact_lengths_and_rejects_others(self):
        for count in (8, 36):
            ids = ",".join(map(str, [0] + [4] * (count - 2) + [0]))
            result, output = self.invoke(ids, f"{count}.wav")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(output.read_text().strip(), f"bucket={count}:7:{ids}")
        for count in (7, 9, 35, 37, 40):
            ids = ",".join(map(str, [0] + [4] * (count - 2) + [0]))
            result, output = self.invoke(ids, f"unsupported-{count}.wav")
            self.assertEqual(result.returncode, 2)
            self.assertIn("unsupported phoneme count", result.stderr)
            self.assertFalse(output.exists())
        for ids in ("", "0,4,,0", "0,178,0", "-1,4,0", "0," * 41):
            result, output = self.invoke(ids, "invalid.wav")
            self.assertEqual(result.returncode, 2)
            self.assertFalse(output.exists())

    def test_wrong_host_route_is_rejected_before_execution(self):
        original_call = 'joined(host, sizeof(host), argv[1], directory_count, "native-kokoro-wav")'
        wrong_call = 'joined(host, sizeof(host), argv[1], 8u, "native-kokoro-wav")'
        source = SOURCE.read_text()
        self.assertEqual(source.count(original_call), 1)
        mutant = self.root / "wrong-route.c"
        mutant.write_text(source.replace(original_call, wrong_call, 1))
        wrong_dispatch = self.root / "wrong-route-dispatch"
        subprocess.run(["cc", "-std=c11", "-O2", "-Wall", "-Wextra",
            "-Werror", "-pedantic", "-I", str(self.root), str(mutant),
            "-lcrypto", "-o", str(wrong_dispatch)], check=True,
            capture_output=True, text=True)
        ids = ",".join(map(str, [0] + [4] * 34 + [0]))
        output = self.root / "wrong-route.wav"
        result = subprocess.run([str(wrong_dispatch), str(self.root), str(output),
            "7", ids], capture_output=True, text=True)
        self.assertEqual(result.returncode, 4, result.stderr)
        self.assertIn("artifact identity mismatch", result.stderr)
        self.assertFalse(output.exists())

    def test_existing_output_and_asset_aliases_are_rejected(self):
        ids = ",".join(map(str, [0] + [4] * 6 + [0]))
        package = self.root / "8/request.cktts"
        original = package.read_bytes()
        for label, target in (
                ("package", package),
                ("voice", self.root / "af_heart.ckvoice"),
                ("host", self.root / "8/native-kokoro-wav")):
            with self.subTest(target=label):
                result, _ = self.invoke(ids, str(target))
                self.assertEqual(result.returncode, 6, result.stderr)
                self.assertIn("output path already exists", result.stderr)
        for kind in ("symlink", "hardlink"):
            alias = self.root / f"{kind}-to-package.wav"
            if kind == "symlink":
                alias.symlink_to(package)
            else:
                os.link(package, alias)
            with self.subTest(alias=kind):
                result, _ = self.invoke(ids, str(alias))
                self.assertEqual(result.returncode, 6, result.stderr)
                self.assertEqual(package.read_bytes(), original)
                self.assertEqual(alias.read_bytes(), original)
        dangling = self.root / "dangling.wav"
        dangling.symlink_to(self.root / "missing.wav")
        result, _ = self.invoke(ids, str(dangling))
        self.assertEqual(result.returncode, 6, result.stderr)
        self.assertTrue(dangling.is_symlink())
        result, output = self.invoke(ids, "fresh-after-alias.wav")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(output.read_text().strip(), f"bucket=8:7:{ids}")

    def test_tampered_package_library_host_or_voice_fails_before_execution(self):
        ids = ",".join(map(str, [0] + [4] * 6 + [0]))
        for target in (self.root / "8/request.cktts",
                       self.root / "8/generated.so",
                       self.root / "8/native-kokoro-wav",
                       self.root / "af_heart.ckvoice"):
            with self.subTest(target=target.name):
                original = target.read_bytes()
                target.write_bytes(original + b"x")
                result, output = self.invoke(ids, f"tampered-{target.name}.wav")
                self.assertEqual(result.returncode, 4)
                self.assertIn("artifact identity mismatch", result.stderr)
                self.assertFalse(output.exists())
                target.write_bytes(original)
        print("CKE_NUMERICAL_CASE " + json.dumps({
            "case_id": "kokoro.native-dispatch.exact-length-identity-v1",
            "name": "Kokoro native 8/36 selector and artifact identity",
            "provider": "native_kokoro_dispatch", "dtype": "fp32",
            "direction": "inference", "oracle": "synthetic distinguishable hosts",
            "status": "pass", "gate": "session", "blocking": True,
            "max_diff": 0, "tolerance": 0,
            "configuration": "8/36 accepted; unsupported counts and four tampered artifacts rejected",
            "reproduction_command": "python3 -m unittest tests.test_v8_kokoro_native_dispatch",
        }, sort_keys=True))


class KokoroNativeDispatchFullTest(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("CKE_TTS_RUN_DISPATCH_FULL"),
                         "asset-backed dispatch bundle replay not requested")
    def test_pinned_bundle_routes_both_native_waveforms(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("CKE_TTS_BUILD_TMP_DIR")) as tmp:
            bundle = Path(tmp) / "bundle"
            provider = os.environ.get("CKE_TTS_CONV_PROVIDER", "scalar")
            report = bundle_builder.build_bundle(
                Path(os.environ["CKE_TTS_NATIVE_BUMP_DIR"]),
                Path(os.environ["CKE_TTS_NATIVE_VOICE_BANK"]), bundle,
                conv_provider=provider)
            self.assertEqual(report["supported_phoneme_counts"], [8, 36])
            self.assertEqual(report["conv_provider"], provider)
            if provider == "avx2_fma":
                for count in (8, 36):
                    self.assertEqual(report["buckets"][count][
                        "selected_conv_calls"], 88)
            wav_hashes = {}
            for count, fixture, expected in (
                (8, "kokoro_waveform_hello_8_pinned.npz",
                 "8f88a076382944cfd26826d964cf12f378ebc8c16f7ce98ea5ed24e98667eda1"),
                (36, "kokoro_encoder_pinned.npz",
                 "187e777636f7071776968bb7a7c4ced52f3d0438bedc281b84416562c844015e"),
            ):
                with np.load(ROOT / "tests/fixtures/tts" / fixture,
                             allow_pickle=False) as values:
                    ids = ",".join(str(int(value)) for value in
                                   values["word_ids"].ravel())
                wav = bundle / f"{count}.wav"
                run = subprocess.run([str(bundle / "native-kokoro-dispatch"),
                    str(bundle), str(wav), "7", ids], capture_output=True,
                    text=True)
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertIn("resolved generated symbol in", run.stderr)
                wav_hashes[count] = hashlib.sha256(wav.read_bytes()).hexdigest()
                self.assertEqual(wav_hashes[count], expected)
            unsupported = bundle / "unsupported.wav"
            run = subprocess.run([str(bundle / "native-kokoro-dispatch"),
                str(bundle), str(unsupported), "7", "0,4,4,4,4,4,0"],
                capture_output=True, text=True)
            self.assertEqual(run.returncode, 2)
            self.assertFalse(unsupported.exists())
            print("CKE_NUMERICAL_CASE " + json.dumps({
                "case_id": "kokoro.native-dispatch.pinned-bump-two-wavs-v1",
                "name": "Pinned BUMP native exact-length dispatch to two WAVs",
                "provider": "native_kokoro_dispatch+generated_kokoro_waveform",
                "dtype": "fp32", "direction": "inference",
                "oracle": "retained seeded native WAV SHA-256 baselines",
                "status": "pass", "gate": "session", "blocking": True,
                "max_diff": 0, "tolerance": 0,
                "wav_sha256": wav_hashes,
                "conv_provider": provider,
                "configuration": "8 and 36 IDs, af_heart, seed 7; unsupported 7 IDs rejected",
                "reproduction_command": "CKE_TTS_RUN_DISPATCH_FULL=1 "
                    "CKE_TTS_NATIVE_BUMP_DIR=<pinned-bump> "
                    "CKE_TTS_NATIVE_VOICE_BANK=<pinned-voice> "
                    "CKE_TTS_CONV_PROVIDER=<scalar-or-avx2_fma> "
                    "python3 -m unittest tests.test_v8_kokoro_native_dispatch",
            }, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
