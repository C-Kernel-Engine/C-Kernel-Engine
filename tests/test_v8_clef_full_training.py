"""Complete topology, small and released geometry: pinned independent autograd.

The source asset is hash-verified. Missing source/PyTorch is explicitly SKIP.
Released weights require a separate asset-backed lane; backbone features remain synthetic.
"""

import unittest


class FullHeadTraining(unittest.TestCase):
    def test_pinned_complete_head_training(self):
        self.certify(released=False)

    def test_released_head_training(self):
        self.certify(released=True)

    def certify(self, *, released, mixed=False, diagnostic=False, artifacts=None):
        import sys, os, json, ctypes, tempfile, subprocess, importlib.util, hashlib
        from pathlib import Path
        from types import SimpleNamespace as NS

        try:
            import numpy as np, torch
        except ImportError as exc:
            self.skipTest("independent PyTorch comparison unavailable: " + str(exc))
        failures = []
        step_reports = []
        first_divergence = None

        def bound(actual, limit, label):
            if diagnostic and not actual < limit:
                failures.append(
                    dict(
                        step=step, comparison=label, max_abs=float(actual), limit=limit
                    )
                )
            else:
                self.assertLess(actual, limit, msg=f"step {step}: {label}")

        def close(actual, expected, label, **tolerances):
            try:
                np.testing.assert_allclose(actual, expected, **tolerances)
            except AssertionError:
                if not diagnostic:
                    raise
                failures.append(
                    dict(
                        step=step,
                        comparison=label,
                        max_abs=float(np.max(np.abs(actual - expected))),
                        **tolerances,
                    )
                )

        torch.set_num_threads(1)
        torch.manual_seed(611)
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root))
        sys.path.insert(0, str(root / "version/v7/scripts"))
        sys.path.insert(0, str(root / "version/v8/scripts"))
        import lower_ir2_backward_v7 as b, generate_train_layout_v7 as l, codegen_train_runtime_v7 as c, run_training_workflow_v8 as wf
        from version.v8.decision.build_clef_training_ir import (
            build_training_ir,
            training_registry,
        )
        from version.v8.decision.package_clef_head import effective_weights

        source = os.environ.get("CKE_CLEF_FLASH_REFERENCE_SOURCE")
        if not source:
            self.skipTest("pinned independent Clef reference source unavailable")
        assert (
            hashlib.sha256(Path(source).read_bytes()).hexdigest()
            == json.loads(
                (root / "tests/fixtures/decision/clef_flash/identity.json").read_text()
            )["joint_schema_model_sha256"]
        )
        spec = importlib.util.spec_from_file_location("pinned_clef", source)
        m = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = m
        spec.loader.exec_module(m)
        channels, width, heads, feedforward = (
            (4096, 1024, 16, 4096) if released else (8, 8, 2, 16)
        )
        steps, split = (4, 2) if released else (40, 20)
        head_path = os.environ.get("CKE_CLEF_FLASH_HEAD_PATH")
        if released and not head_path:
            self.skipTest("pinned released head weights unavailable")
        head = m.JointSchemaHead(
            hidden_size=channels,
            width=width,
            routing_layers=2,
            layers=4,
            heads=heads,
            feedforward=feedforward,
            dropout=0,
        )
        if released:
            try:
                from safetensors.torch import load_file
            except ImportError as exc:
                self.skipTest("pinned weight reader unavailable: " + str(exc))

            identity = json.loads(
                (root / "tests/fixtures/decision/clef_flash/identity.json").read_text()
            )
            self.assertEqual(
                hashlib.sha256(Path(head_path).read_bytes()).hexdigest(),
                identity["joint_head_weights_sha256"],
            )
            head.load_state_dict(load_file(head_path), strict=True)
        head.eval()
        raw = {k: v.detach().numpy().copy() for k, v in head.state_dict().items()}
        raw["output_embedding.weight"] = (
            np.random.default_rng(9).normal(0, 0.2, (16, channels)).astype("f")
        )
        hidden = np.random.default_rng(42).normal(0, 0.2, (6, channels)).astype("f")
        ir, weights, graph = build_training_ir(
            raw, hidden, questions=2, options=3, heads=heads, feedforward=feedforward
        )
        if mixed:
            from version.v8.training.bf16_projection import use_bf16_projection
            from tests.clef_mixed_oracle import install

            for op in list(ir["ops"]):
                if op["kernel_id"] in (
                    "linear_rows_checked_f32",
                    "linear_rows_checked_no_bias_f32",
                ):
                    ir = use_bf16_projection(ir, op["op_id"])
            projection_captures = install(self, torch, head)
        reg = training_registry()
        o = b.synthesize_ir2_backward(
            ir,
            reg,
            json.loads(
                (root / "version/v7/kernel_maps/kernel_bindings.json").read_text()
            ),
            json.loads((root / "version/v8/scripts/grad_rules_v8.json").read_text()),
            strict=True,
            allow_partial=False,
            checkpoint_policy="none",
        )
        layout = l.build_layout(o, {"config": o["config"]}, 64, strict=True)
        src, summary = c.generate_c(
            o, reg, manifest={"config": o["config"]}, layout=layout
        )

        md = {
            "input_ids": [1, 2, 3, 4, 5, 6],
            "question_spans": [0, 2, 2, 4],
            "option_spans": [1, 3, 2, 5, 4, 6],
            "global_span": [5, 6],
            "option_counts": [2, 1],
            "question_type_ids": [0, 2],
            "labels": [1, 0],
            "broadcast_zeros": [0, 0],
        }

        metadata = np.array(
            [
                v
                for name in ir["config"]["training_integer_inputs"]
                for v in md[name.removeprefix("input.")]
            ],
            dtype="i4",
        )
        record = NS(
            questions=[
                NS(
                    question_span=(0, 2), question_type=0, option_spans=[(1, 3), (2, 5)]
                ),
                NS(question_span=(2, 4), question_type=2, option_spans=[(4, 6)]),
            ]
        )
        opt = torch.optim.AdamW(
            head.parameters(), lr=0.001, eps=1e-4, weight_decay=0.01, foreach=False
        )
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "train.c").write_text(src)
            engine = root / "build/libckernel_engine.so"
            if not engine.is_file():
                self.skipTest(
                    "native training library unavailable; build/libckernel_engine.so required"
                )
            subprocess.run(
                [
                    os.environ.get("CC", "cc"),
                    "-DCK_NUM_TOKENS=6",
                    "-DCK_ADAMW_EPS=1e-4f",
                    "-O2",
                    "-shared",
                    "-fPIC",
                    "-I" + str(root / "include"),
                    str(p / "train.c"),
                    str(root / "src/kernels/linear_rows_checked.c"),
                    str(engine),
                    "-Wl,-rpath," + str(engine.parent),
                    "-lm",
                    "-o",
                    str(p / "train.so"),
                ],
                check=True,
            )
            if artifacts is not None:
                import shutil

                artifact_dir = Path(artifacts)
                artifact_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p / "train.c", artifact_dir / "train.c")
                shutil.copy2(p / "train.so", artifact_dir / "train.so")
                (artifact_dir / "layout.json").write_text(
                    json.dumps(layout, indent=2) + "\n"
                )
                (artifact_dir / "backward_ir.json").write_text(
                    json.dumps(o, indent=2) + "\n"
                )
            lib = ctypes.CDLL(str(p / "train.so"))
            wf._configure(lib)
            self.addCleanup(lib.ck_train_free)
            fp = ctypes.POINTER(ctypes.c_float)
            ip = ctypes.POINTER(ctypes.c_int32)
            ptr = lambda a: a.ctypes.data_as(fp)
            lib.ck_train_init.argtypes = [fp, ip, ctypes.c_int]
            lib.ck_train_step_with_i32_ex.argtypes = [
                ip,
                ip,
                ctypes.c_int,
                ip,
                ctypes.c_size_t,
                fp,
                ctypes.c_float,
            ]
            packed = np.concatenate(
                [weights[n].ravel() for n in summary["init_weight_order"]]
            )
            sizes = np.array(summary["init_weight_numel"], dtype="i4")
            self.assertEqual(
                lib.ck_train_init(ptr(packed), sizes.ctypes.data_as(ip), len(sizes)),
                len(sizes),
            )
            lib.ck_set_strict_parity(1)
            ids = np.zeros(6, dtype="i4")

            def snapshot(kind, count):
                a = np.empty(count, dtype="f")
                fn = getattr(lib, "ck_train_export_" + kind + "_snapshot")
                fn.argtypes = [fp, ctypes.c_int]
                assert fn(ptr(a), count) == count
                return a

            losses = []
            errors = dict(gradient=0.0, weight=0.0, moment=0.0)
            # Rejections after partial forward work must not publish an optimizer update.
            for key, bad in (
                ("input_ids", [-1, 2, 3, 4, 5, 6]),
                ("global_span", [0, 1]),
                ("option_counts", [1, 1]),
                ("question_type_ids", [3, 2]),
                ("labels", [2, 0]),
            ):
                invalid = dict(md)
                invalid[key] = bad
                rejected = np.array(
                    [
                        v
                        for name in ir["config"]["training_integer_inputs"]
                        for v in invalid[name.removeprefix("input.")]
                    ],
                    dtype="i4",
                )
                before = snapshot("weight", summary["weight_snapshot_floats"])
                rejected_loss = ctypes.c_float()
                self.assertLess(
                    lib.ck_train_step_with_i32_ex(
                        ids.ctypes.data_as(ip),
                        ids.ctypes.data_as(ip),
                        6,
                        rejected.ctypes.data_as(ip),
                        len(rejected),
                        ctypes.byref(rejected_loss),
                        0.001,
                    ),
                    0,
                )
                np.testing.assert_array_equal(snapshot("weight", len(before)), before)
                self.assertEqual(lib.ck_train_get_opt_step(), 0)
            for step in range(steps):
                opt.zero_grad()
                if mixed:
                    projection_captures.clear()
                out = head(
                    torch.tensor(hidden)[None],
                    torch.tensor(md["input_ids"])[None],
                    torch.ones(1, 6, dtype=torch.long),
                    [record],
                    torch.tensor(raw["output_embedding.weight"]),
                )[0]
                loss = (
                    sum(
                        torch.nn.functional.cross_entropy(x[None], torch.tensor([y]))
                        for x, y in zip(out, [1, 0])
                    )
                    / 2
                )
                loss.backward()
                if mixed:
                    pre_flat = snapshot("weight", summary["weight_snapshot_floats"])
                    pre_weights = {}
                    off = 0
                    for name, count in zip(
                        summary["init_weight_order"], summary["init_weight_numel"]
                    ):
                        pre_weights[name] = (
                            pre_flat[off : off + count]
                            .reshape(weights[name].shape)
                            .copy()
                        )
                        off += count
                nl = ctypes.c_float()
                rc = lib.ck_train_step_with_i32_ex(
                    ids.ctypes.data_as(ip),
                    ids.ctypes.data_as(ip),
                    6,
                    metadata.ctypes.data_as(ip),
                    len(metadata),
                    ctypes.byref(nl),
                    0.001,
                )
                assert rc > 0
                bound(abs(nl.value - loss.item()), 3e-5, "loss")
                native_logits = wf.CERT._activation_snapshot(lib, summary)[
                    "act.option_logits"
                ].ravel()
                reference_logits = torch.cat(out).detach().numpy()
                close(
                    native_logits,
                    reference_logits,
                    "option_logits",
                    atol=3e-5,
                    rtol=3e-4,
                )
                losses.append(nl.value)
                grads = {k: v.grad.numpy().copy() for k, v in head.named_parameters()}
                grads["output_embedding.weight"] = np.zeros_like(
                    raw["output_embedding.weight"]
                )
                eg = effective_weights(graph, grads)
                actual = snapshot(
                    "parameter_gradient", summary["parameter_gradient_snapshot_floats"]
                )
                offset = 0
                errs = []
                gradient_details = []
                for n in summary["parameter_gradient_order"]:
                    size = eg[n].size
                    a = actual[offset : offset + size]
                    e = eg[n].ravel()
                    errs.append(
                        (float(np.max(np.abs(a - e))), n, float(np.max(np.abs(e))))
                    )
                    delta = a.astype("float64") - e.astype("float64")
                    gradient_details.append(
                        dict(
                            parameter=n,
                            max_abs=float(np.max(np.abs(delta))),
                            rmse=float(np.sqrt(np.mean(delta * delta))),
                            reference_l2=float(np.linalg.norm(e.astype("float64"))),
                            actual_l2=float(np.linalg.norm(a.astype("float64"))),
                        )
                    )
                    offset += size
                errors["gradient"] = max(errors["gradient"], max(x[0] for x in errs))
                worst = max(errs)
                if released:
                    print(
                        f"released step={step} loss={nl.value:.9g} worst_gradient={worst}",
                        flush=True,
                    )
                if mixed and worst[0] >= 3e-5 and first_divergence is None:
                    from tests.clef_mixed_oracle import diagnose

                    gradient_values = {}
                    off = 0
                    for name in summary["parameter_gradient_order"]:
                        gradient_values[name] = actual[off : off + eg[name].size].copy()
                        off += eg[name].size
                    first, packet = diagnose(
                        ir,
                        projection_captures,
                        wf.CERT._activation_snapshot(lib, summary),
                        pre_weights,
                        gradient_values,
                        layout,
                        snapshot("accum", lib.ck_train_get_accum_snapshot_numel()),
                    )
                    first_divergence = dict(step=step, projections=first)
                    if artifacts is not None and packet is not None:
                        packet_path = (
                            Path(artifacts) / "first_projection_divergence.npz"
                        )
                        packet_path.parent.mkdir(parents=True, exist_ok=True)
                        np.savez(packet_path, **packet)
                        first_divergence["packet"] = dict(
                            path=str(packet_path.resolve()),
                            sha256=hashlib.sha256(packet_path.read_bytes()).hexdigest(),
                        )
                    print("FIRST_FORWARD_DIVERGENCE", first[:1], flush=True)
                bound(worst[0], 3e-5, "parameter_gradient: " + worst[1])
                step_reports.append(
                    dict(
                        step=step,
                        native_loss=nl.value,
                        reference_loss=loss.item(),
                        worst_gradient=dict(
                            max_abs=worst[0],
                            parameter=worst[1],
                            reference_magnitude=worst[2],
                        ),
                        gradients=gradient_details,
                        native_logits=native_logits.tolist(),
                        reference_logits=reference_logits.tolist(),
                    )
                )
                opt.step()
                new = {
                    k: v.detach().numpy().copy() for k, v in head.state_dict().items()
                }
                new["output_embedding.weight"] = raw["output_embedding.weight"]
                ew = effective_weights(graph, new)
                ew["__features.hidden"] = hidden
                native_moments = snapshot(
                    "optimizer_state", summary["optimizer_state_snapshot_floats"]
                )
                expected_moments = []
                for kind in ("exp_avg", "exp_avg_sq"):
                    raw_moments = {
                        k: opt.state[v][kind].numpy().copy()
                        for k, v in head.named_parameters()
                    }
                    raw_moments["output_embedding.weight"] = np.zeros_like(
                        raw["output_embedding.weight"]
                    )
                    effective_moments = effective_weights(graph, raw_moments)
                    expected_moments.extend(
                        effective_moments[n].ravel()
                        for n in summary["parameter_gradient_order"]
                    )
                errors["moment"] = max(
                    errors["moment"],
                    float(
                        np.max(
                            np.abs(native_moments - np.concatenate(expected_moments))
                        )
                    ),
                )
                close(
                    native_moments,
                    np.concatenate(expected_moments),
                    "optimizer_moments",
                    atol=3e-6,
                    rtol=3e-4,
                )
                a = snapshot("weight", summary["weight_snapshot_floats"])
                e = np.concatenate(
                    [ew[n].ravel() for n in summary["init_weight_order"]]
                )
                errors["weight"] = max(errors["weight"], float(np.max(np.abs(a - e))))
                close(a, e, "master_weights", atol=3e-5, rtol=3e-4)
                offset = 0
                for name, count in zip(
                    summary["init_weight_order"], summary["init_weight_numel"]
                ):
                    if name in ("__features.hidden", "output_embedding.weight"):
                        self.assertEqual(
                            a[offset : offset + count].tobytes(),
                            weights[name].tobytes(),
                        )
                    offset += count
                if step == split - 1:
                    checkpoint_meta = dict(
                        runtime_contract_sha256=summary["runtime_contract_sha256"],
                        dataset_sha256=hashlib.sha256(
                            hidden.tobytes() + metadata.tobytes()
                        ).hexdigest(),
                        training_config=dict(
                            dtype=(
                                "mixed_projection_bf16_fp32_master" if mixed else "fp32"
                            ),
                            optimizer="AdamW",
                            lr=0.001,
                            eps=1e-4,
                            weight_decay=0.01,
                            beta1=0.9,
                            beta2=0.999,
                            grad_accum=1,
                        ),
                        training_config_sha256=wf._json_sha256(
                            dict(
                                dtype=(
                                    "mixed_projection_bf16_fp32_master"
                                    if mixed
                                    else "fp32"
                                ),
                                optimizer="AdamW",
                                lr=0.001,
                                eps=1e-4,
                                weight_decay=0.01,
                                beta1=0.9,
                                beta2=0.999,
                                grad_accum=1,
                            )
                        ),
                        next_microstep=split,
                        total_microsteps=steps,
                        grad_accum=1,
                    )
                    checkpoint = wf._checkpoint_write(p, "step20", lib, checkpoint_meta)
            print("loss trajectory", losses[0], losses[-1])
            assert losses[-1] < losses[0] * (1.0 if released else 0.7)
            final = snapshot("weight", summary["weight_snapshot_floats"])
            moments = snapshot(
                "optimizer_state", summary["optimizer_state_snapshot_floats"]
            )
            lib.ck_train_free()
            assert lib.ck_train_init(
                ptr(packed), sizes.ctypes.data_as(ip), len(sizes)
            ) == len(sizes)
            wf._checkpoint_load(checkpoint, lib, checkpoint_meta)
            for i in range(split, steps):
                nl = ctypes.c_float()
                assert (
                    lib.ck_train_step_with_i32_ex(
                        ids.ctypes.data_as(ip),
                        ids.ctypes.data_as(ip),
                        6,
                        metadata.ctypes.data_as(ip),
                        len(metadata),
                        ctypes.byref(nl),
                        0.001,
                    )
                    > 0
                )
                assert nl.value == losses[i]
            np.testing.assert_array_equal(snapshot("weight", len(final)), final)
            np.testing.assert_array_equal(
                snapshot("optimizer_state", len(moments)), moments
            )
            lib.ck_train_free()
            assert lib.ck_train_init(
                ptr(final), sizes.ctypes.data_as(ip), len(sizes)
            ) == len(sizes)
            lib.ck_train_set_batch_with_i32_ex.argtypes = [
                ip,
                ip,
                ctypes.c_int,
                ip,
                ctypes.c_size_t,
            ]
            assert (
                lib.ck_train_set_batch_with_i32_ex(
                    ids.ctypes.data_as(ip),
                    ids.ctypes.data_as(ip),
                    6,
                    metadata.ctypes.data_as(ip),
                    len(metadata),
                )
                > 0
            )
            assert lib.ck_train_forward_step() > 0
            activations = wf.CERT._activation_snapshot(lib, summary)
            expected = head(
                torch.tensor(hidden)[None],
                torch.tensor(md["input_ids"])[None],
                torch.ones(1, 6, dtype=torch.long),
                [record],
                torch.tensor(raw["output_embedding.weight"]),
            )[0]
            close(
                activations["act.option_logits"].ravel(),
                torch.cat(expected).detach().numpy(),
                "reloaded_logits",
                atol=3e-5,
                rtol=3e-4,
            )
            assert lib.ck_train_get_opt_step() == 0
            lib.ck_train_free()
            # A new process must restore the checkpoint without inherited native state.
            resume_request = dict(
                library=str(p / "train.so"),
                library_sha256=hashlib.sha256(
                    (p / "train.so").read_bytes()
                ).hexdigest(),
                sizes=sizes.tolist(),
                tokens=6,
                integer_inputs=metadata.tolist(),
                checkpoint=str(checkpoint),
                metadata=checkpoint_meta,
                remaining_steps=steps - split,
                learning_rate=0.001,
                result=str(p / "resumed.json"),
            )
            (p / "resume_request.json").write_text(json.dumps(resume_request))
            if not released:
                rejected_request = dict(
                    resume_request,
                    metadata=dict(checkpoint_meta, dataset_sha256="wrong-dataset"),
                )
                (p / "rejected_resume.json").write_text(json.dumps(rejected_request))
                rejected = subprocess.run(
                    [
                        sys.executable,
                        str(root / "tests/clef_training_resume_worker.py"),
                        str(p / "rejected_resume.json"),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                self.assertNotEqual(rejected.returncode, 0)
                self.assertIn("dataset_sha256 mismatch", rejected.stderr)
                self.assertFalse((p / "resumed.json").exists())
            subprocess.run(
                [
                    sys.executable,
                    str(root / "tests/clef_training_resume_worker.py"),
                    str(p / "resume_request.json"),
                ],
                check=True,
                timeout=300,
            )
            resumed = json.loads((p / "resumed.json").read_text())
            self.assertEqual(resumed["losses"], losses[split:])
            self.assertEqual(resumed["optimizer_step"], steps)
            self.assertEqual(
                resumed["weight_sha256"],
                hashlib.sha256(final.astype("<f4", copy=False).tobytes()).hexdigest(),
            )
            self.assertEqual(
                resumed["optimizer_state_sha256"],
                hashlib.sha256(moments.astype("<f4", copy=False).tobytes()).hexdigest(),
            )
            if not mixed:
                # A separately generated inference entry consumes the exported trained weights.
                from tests.test_v8_clef_flash_head_ingress import compile_case

                inference_weights = {}
                offset = 0
                for name, count in zip(
                    summary["init_weight_order"], summary["init_weight_numel"]
                ):
                    inference_weights[name] = (
                        final[offset : offset + count]
                        .reshape(weights[name].shape)
                        .copy()
                    )
                    offset += count
                inference_weights.pop("__features.hidden")
                with tempfile.TemporaryDirectory() as inf:
                    arena, execute, view, _, _ = compile_case(
                        Path(inf), graph, inference_weights
                    )
                    view("hidden")[:] = hidden
                    for name in (
                        "input_ids",
                        "question_spans",
                        "option_spans",
                        "global_span",
                        "option_counts",
                        "question_type_ids",
                    ):
                        view(name)[:] = np.array(md[name], dtype="i4").reshape(
                            view(name).shape
                        )
                    self.assertEqual(execute(arena, len(arena)), 0)
                    np.testing.assert_allclose(
                        view("option_logits").ravel(),
                        torch.cat(expected).detach().numpy(),
                        atol=3e-5,
                        rtol=3e-4,
                    )
            report = dict(
                schema_version=1,
                seed=611,
                policy=dict(
                    operand_rounding="rne_bf16" if mixed else "none",
                    projection_accumulation="ordered_fp32" if mixed else "fp64",
                    masters="fp32",
                    gradients="fp32",
                    optimizer="AdamW",
                    learning_rate=0.001,
                    epsilon=1e-4,
                    weight_decay=0.01,
                    beta1=0.9,
                    beta2=0.999,
                    other_operations="fp32",
                    rounding_vjp="identity" if mixed else "not_applicable",
                ),
                retained_artifacts=(
                    str(Path(artifacts).resolve()) if artifacts is not None else None
                ),
                dtype="mixed_projection_bf16_fp32_master" if mixed else "fp32",
                geometry=(
                    "released_complete_head" if released else "small_complete_head"
                ),
                numerical_parity="FAIL" if failures else "PASS",
                trajectory_comparisons=step_reports,
                numerical_failures=failures,
                first_divergence=first_divergence,
                max_abs_errors=errors,
                loss_first=losses[0],
                loss_last=losses[-1],
                torch_version=torch.__version__,
                reference_source_sha256=hashlib.sha256(
                    Path(source).read_bytes()
                ).hexdigest(),
                generated_c_sha256=hashlib.sha256(src.encode()).hexdigest(),
                engine_sha256=hashlib.sha256(engine.read_bytes()).hexdigest(),
                runtime_contract_sha256=summary["runtime_contract_sha256"],
                initial_weights_sha256=hashlib.sha256(packed.tobytes()).hexdigest(),
                dataset_sha256=hashlib.sha256(
                    hidden.tobytes() + metadata.tobytes()
                ).hexdigest(),
                released_head_training=(
                    ("FAIL" if failures else "PASS") if released else "NOT_TESTED"
                ),
                frozen_feature_provenance="synthetic; not generated backbone",
                head_weights_sha256=(
                    hashlib.sha256(Path(head_path).read_bytes()).hexdigest()
                    if released
                    else None
                ),
                same_process_resume="PASS",
                fresh_process_resume="PASS",
                generated_inference_export="NOT_TESTED" if mixed else "PASS",
                trainable_parameters=sum(
                    summary["init_weight_numel"][summary["init_weight_order"].index(n)]
                    for n in summary["parameter_gradient_order"]
                ),
            )
            print(
                json.dumps(
                    {
                        k: v
                        for k, v in report.items()
                        if k
                        not in (
                            "trajectory_comparisons",
                            "numerical_failures",
                            "first_divergence",
                        )
                    },
                    sort_keys=True,
                )
            )
            return report


class FullHeadTrainingContracts(unittest.TestCase):
    def test_resume_rejects_wrong_library_before_loading(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from tests import clef_training_resume_worker as worker

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "wrong.so").write_bytes(b"not the expected library")
            (path / "request.json").write_text(
                json.dumps(
                    dict(library=str(path / "wrong.so"), library_sha256="0" * 64)
                )
            )
            with patch.object(worker.ctypes, "CDLL") as loader:
                with self.assertRaisesRegex(ValueError, "library identity mismatch"):
                    worker.run(path / "request.json")
                loader.assert_not_called()

    def graph(self, mixed=False):
        try:
            import numpy as np
            import torch
        except ImportError as exc:
            self.skipTest("synthetic PyTorch parameters unavailable: " + str(exc))
        import json
        import sys
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "version/v7/scripts"))
        from tests.test_v8_clef_flash_complete_head import synthetic_weights
        from version.v8.decision.build_clef_training_ir import (
            build_training_ir,
            training_registry,
        )
        import lower_ir2_backward_v7 as backward

        ir, weights, _ = build_training_ir(
            synthetic_weights(),
            np.ones((6, 8), dtype="f4"),
            questions=2,
            options=3,
            heads=2,
            feedforward=16,
        )
        if mixed:
            from version.v8.training.bf16_projection import use_bf16_projection

            for op in list(ir["ops"]):
                if op["kernel_id"] in (
                    "linear_rows_checked_f32",
                    "linear_rows_checked_no_bias_f32",
                ):
                    ir = use_bf16_projection(ir, op["op_id"])
        registry = training_registry()
        ir2 = backward.synthesize_ir2_backward(
            ir,
            registry,
            json.loads(
                (root / "version/v7/kernel_maps/kernel_bindings.json").read_text()
            ),
            json.loads((root / "version/v8/scripts/grad_rules_v8.json").read_text()),
            strict=True,
            allow_partial=False,
            checkpoint_policy="none",
        )
        return ir2, registry

    def emit(self, ir, registry):
        import generate_train_layout_v7 as planner
        import codegen_train_runtime_v7 as codegen

        manifest = {"config": ir["config"]}
        layout = planner.build_layout(ir, manifest, 64, strict=True)
        return codegen.generate_c(ir, registry, manifest=manifest, layout=layout)

    def test_complete_topology_and_explicit_row_extents(self):
        ir, registry = self.graph()
        source, report = self.emit(ir, registry)
        self.assertEqual(report["skipped_ops"], [])
        self.assertEqual(len(report["parameter_gradient_order"]), 162)
        self.assertIn(
            "ck_residual_add_backward(grad_act_act_option_logits, tmp_grad_act_op143_left, tmp_grad_act_op143_right, 1, 3)",
            source,
        )
        self.assertIn("ck_expand_group_row_indices_checked_i32", source)
        self.assertIn("ck_indexed_rows_backward_f32", source)
        self.assertNotIn("weight.__features.hidden", report["parameter_gradient_order"])

    def test_generated_integer_must_be_defined_before_use(self):
        ir, registry = self.graph()
        ir["forward"] = [
            op
            for op in ir["forward"]
            if op["kernel_id"] != "expand_group_row_indices_checked_i32"
        ]
        with self.assertRaisesRegex(ValueError, "integer|initialized|metadata"):
            self.emit(ir, registry)

    def test_flatten_is_explicit_and_boolean(self):
        ir, registry = self.graph()
        mapping = next(
            k for k in registry["kernels"] if k["id"] == "add_rows_backward_train_f32"
        )
        mapping["training_contract"]["buffer_spans"][0]["flatten"] = "yes"
        with self.assertRaisesRegex(ValueError, "flatten must be boolean"):
            self.emit(ir, registry)

    def test_insufficient_declared_projection_extent(self):
        ir, registry = self.graph()
        op = next(
            op
            for op in ir["forward"]
            if op["kernel_id"] == "linear_rows_checked_no_bias_f32"
        )
        op["params"]["call_constants"]["linear_input_elements"] = 1
        with self.assertRaisesRegex(ValueError, "capacity"):
            self.emit(ir, registry)


if __name__ == "__main__":
    unittest.main()
