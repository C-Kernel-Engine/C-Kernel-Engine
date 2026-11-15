#!/usr/bin/env python3
"""Registered LoRA job: generated native training, held-out evaluation and reload."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import shutil
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from version.v8.training.lora_package import load_package
from version.v8.training.lora_session import LoRAConfig, LoRASession, prepare_lora, require


def write(path, doc):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(doc, allow_nan=False, indent=2))
    temporary.replace(path)


def run(package, package_sha256, config, folder):
    doc, base, discovery = load_package(package, package_sha256)
    from server.training.lora import packages, validate
    from server.training.contract import digest as config_digest
    require(not validate(config, packages({config['package_id']: dict(path=str(package), sha256=package_sha256)})), 'invalid job configuration')
    # Compile from a private snapshot, not mutable registry files.
    with tempfile.TemporaryDirectory(prefix='cke-lora-base-') as snapshot:
        snapshot = Path(snapshot)
        for filename, field in [('weights_manifest.json','manifest_sha256'),('weights.bump','weights_sha256')]:
            shutil.copyfile(base/filename, snapshot/filename)
            require(hashlib.sha256((snapshot/filename).read_bytes()).hexdigest() == doc['base'][field], 'base changed before initialization')
        plan = prepare_lora(snapshot, tokens=doc['tokens'], adapters=[LoRAConfig(**x) for x in config['adapters']])
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    train = [r for r in doc['records'] if r['split'] == 'train']
    validation = [r for r in doc['records'] if r['split'] == 'validation']
    def evaluate(session):
        rows = []
        for r in validation:
            result = session.evaluate(r['tokens'], r['targets'])
            rows.append(dict(id=r['id'], loss=result['loss'], logits_sha256=hashlib.sha256(result['logits'].tobytes()).hexdigest()))
        return dict(loss=sum(r['loss'] for r in rows) / len(rows), records=rows)
    with plan.compile(folder / 'runtime', dataset_id=package_sha256,
                      max_steps=config['steps'], learning_rate=config['learning_rate']) as session:
        before = evaluate(session)
        losses = []
        for step in range(config['steps']):
            row = train[step % len(train)]
            losses.append(session.train_step(row['tokens'], row['targets'])['loss'])
            write(folder / 'progress.json', dict(stage='training', completed_steps=step+1, total_steps=config['steps']))
        after = evaluate(session)
        session.save_checkpoint(folder / 'checkpoint')
        session.export_adapters(folder / 'adapters')  # Also verifies every frozen base tensor.
    with LoRASession.open(folder / 'runtime') as session:
        session.load_adapters(folder / 'adapters')
        exported = evaluate(session)
    with LoRASession.open(folder / 'runtime') as session:
        session.load_checkpoint(folder / 'checkpoint')
        resumed = evaluate(session)
    require(after == exported == resumed, 'held-out reload mismatch')
    require(all(math.isfinite(x) for x in losses), 'nonfinite loss')
    # Fixed archive members, never source/client paths. The base is supplied separately.
    with zipfile.ZipFile(folder / 'adapters.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((folder / 'adapters').iterdir()):
            archive.write(path, path.name)
    with zipfile.ZipFile(folder / 'checkpoint.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((folder / 'checkpoint').iterdir()):
            archive.write(path, path.name)
    write(folder / 'progress.json', dict(stage='complete', completed_steps=config['steps'], total_steps=config['steps']))
    return dict(schema='cke.lora.training-result.v1', status='PASS', passed=True,
                package_sha256=package_sha256, config_sha256=config_digest(config), completed_steps=config['steps'], plan=plan.describe(),
                checks={k: dict(passed=True) for k in ('native_execution', 'frozen_base', 'checkpoint_reload', 'adapter_reload')},
                metrics=dict(validation_loss_before=before['loss'], validation_loss_after=after['loss'],
                             training_loss_last=losses[-1]),
                validation=dict(before=before, after=after), training_losses=losses,
                verdicts=dict(execution='PASS', reload='PASS', numerical_oracle='NOT_TESTED',
                              held_out_quality='MEASURED_NOT_CERTIFIED', base_provenance=doc['base']['provenance']),
                artifacts=dict(checkpoint=str(folder / 'checkpoint')))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--package-sha256', required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--json-out', type=Path, required=True)
    a = p.parse_args()
    write(a.json_out, run(a.package, a.package_sha256, json.loads(a.config.read_text()), a.run_dir))
