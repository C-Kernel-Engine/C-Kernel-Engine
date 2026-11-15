#!/usr/bin/env python3
"""Notebook-compatible LoRA entry: load an existing supported FP32 base artifact.

No pretrained download or synthetic-base substitution is performed. Token IDs
must already use the base artifact's tokenizer. Use the JSON-lines input format
{"tokens": [0,1,2,3], "targets": [1,2,3,4]} for each fixed-length training record.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'version/v7'))
import ckernel_engine as cke


def train(base, run_dir, records, *, tokens, steps=40, learning_rate=3e-4):
    records = list(records)
    if not records:
        raise ValueError('training records are required')
    plan = cke.v8.prepare_lora(base, tokens=tokens, adapters=[
        cke.v8.LoRAConfig(target='layer.0.wq', rank=2, alpha=4, name='query', seed=42),
        cke.v8.LoRAConfig(target='layer.1.wv', rank=3, alpha=6, name='value', seed=43),
    ])
    # The full record content binds checkpoint identity, not a mutable file name.
    dataset_id = json.dumps(records, sort_keys=True)
    run_dir = Path(run_dir)
    with plan.compile(run_dir, dataset_id=dataset_id, max_steps=steps,
                      learning_rate=learning_rate) as session:
        losses = []
        for step in range(steps):
            row = records[step % len(records)]
            losses.append(session.train_step(row['tokens'], row['targets'])['loss'])
        session.save_checkpoint(run_dir / 'checkpoint')
        session.export_adapters(run_dir / 'adapters')
    return {'plan': plan.describe(), 'training_losses': losses,
            'checkpoint': str(run_dir / 'checkpoint'), 'adapters': str(run_dir / 'adapters'),
            'held_out_quality': 'NOT_TESTED'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--tokens', type=int, required=True)
    parser.add_argument('--steps', type=int, default=40)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.records.read_text().splitlines() if line.strip()]
    print(json.dumps(train(args.base, args.run_dir, records, tokens=args.tokens, steps=args.steps), indent=2))
