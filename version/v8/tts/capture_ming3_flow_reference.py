#!/usr/bin/env python3
"""Capture a pinned publisher flow sampler using a synthetic, explicitly non-DiT predictor.

Offline regeneration requires the audited fm/CFM.py; no checkpoint is loaded.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from version.v8.tts.build_ming3_flow_step_circuit import SOURCE_REVISION, CHECKPOINT_REVISION


SOURCE_SHA256 = '817146cde55d58ca42dc73524685e94e6dc8c77c915cf166ef436e769b82769c'

def capture(path):
    import torch
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = SOURCE_SHA256
    if digest != expected: raise ValueError('publisher CFM.py hash mismatch')
    spec = importlib.util.spec_from_file_location('pinned_ming_cfm', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    class Predictor(torch.nn.Module):
        def forward_with_cfg(self, *, x, t, c, latent_history, patch_size):
            pred = x * .25 + .75
            null = x * -.125 - .25
            calls.append({'latent': x.clone(), 'prediction': pred.clone(),
                          'null_prediction': null.clone(), 't': t.clone()})
            return torch.cat((pred, null), dim=0)
    torch.manual_seed(731)
    noise = torch.randn(1, 1, 7)
    sde = torch.randn(5, 1, 1, 7)
    final = module.CFM(Predictor()).sample(noise, None, None, sde,
        cfg_scale=2., sigma=.25, temperature=.04)
    times = module.get_epss_timesteps(5, noise.device, noise.dtype)
    times = times - (torch.cos(torch.pi / 2 * times) - 1 + times)
    rows = []
    for index, call in enumerate(calls):
        dt = times[index+1] - times[index]
        amplitude = .25 * (.04 ** .5) * dt.abs().sqrt()
        output = calls[index+1]['latent'] if index+1 < len(calls) else final
        rows.append({**{key: value.flatten().tolist() for key,value in call.items() if key != 't'},
            'noise': sde[index].flatten().tolist(), 'dt': dt.item(),
            'noise_amplitude': amplitude.item(), 'cfg': 2., 'expected': output.flatten().tolist()})
    return {'schema': 'cke.ming3.flow_reference.v1', 'source_revision': SOURCE_REVISION,
        'checkpoint_revision': CHECKPOINT_REVISION, 'source_sha256': digest,
        'torch_version': torch.__version__, 'seed': 731,
        'scope': 'Publisher CFM sampler, FP32 synthetic affine predictor; no DiT or model weights',
        'elements': 7, 'steps': rows, 'complete_synthesis': False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publisher-cfm', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args=parser.parse_args()
    args.output.write_text(json.dumps(capture(args.publisher_cfm), indent=2)+'\n')

if __name__ == '__main__': main()
