#!/usr/bin/env python3
"""Opt-in, same-host Whisper HTTP latency and loaded-artifact regression probe."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time
import urllib.error
import urllib.request
import wave


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    tmp.replace(path)


def normalize(text):
    return ' '.join(text.casefold().split())


def identity(loaded):
    return {side: {key: loaded[side][key] for key in
                  ('libraries', 'manifest_sha256', 'weights_sha256')}
            for side in ('encoder', 'decoder')}


def assess(rows, expected_identity, max_median_seconds, max_queue_seconds):
    if not rows:
        raise ValueError('no measured requests')
    generations = set()
    pids = set()
    for row in rows:
        if identity(row['loaded_artifact']) != expected_identity:
            raise ValueError('loaded artifact identity changed')
        if normalize(row['text']) != normalize(row['expected_text']) or row['stop'] != 'eos':
            raise ValueError('transcript or termination mismatch')
        for key in ('wall_seconds', 'inference_seconds', 'queue_seconds'):
            value = row[key]
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                raise ValueError('invalid timing: ' + key)
        if row['queue_seconds'] > max_queue_seconds:
            raise RuntimeError('busy endpoint: queue delay invalidates latency comparison')
        generations.add(row['worker_generation'])
        pids.add(tuple(row['loaded_artifact'][s]['worker_pid'] for s in ('encoder', 'decoder')))
    if len(generations) != 1 or len(pids) != 1:
        raise ValueError('resident workers changed during benchmark')
    by_case = {}
    for case in sorted({r['case'] for r in rows}):
        selected = [r for r in rows if r['case'] == case]
        wall = statistics.median(r['wall_seconds'] for r in selected)
        by_case[case] = {'median_wall_seconds': wall,
                         'median_inference_seconds': statistics.median(r['inference_seconds'] for r in selected),
                         'max_wall_seconds': max(r['wall_seconds'] for r in selected)}
        if wall > max_median_seconds[case]:
            raise ValueError(f'latency regression: {case}: {wall:.3f}s > {max_median_seconds[case]:.3f}s')
    return by_case


def request(url, data=None):
    req = urllib.request.Request(url, data=data,
                                 headers={'Content-Type': 'audio/wav'} if data else {})
    with urllib.request.urlopen(req, timeout=300) as response:
        return json.load(response)


def run(config_path, output):
    started = datetime.now(timezone.utc).isoformat()
    report = {'schema': 'cke.whisper_service_benchmark.v1', 'status': 'INCOMPLETE',
              'started_at': started, 'attempts': [], 'host_load': os.getloadavg()}
    write(output, report)  # Never leave a prior PASS as the current attempt.
    try:
        config = json.loads(Path(config_path).read_text())
        report['config_sha256'] = hashlib.sha256(Path(config_path).read_bytes()).hexdigest()
        report['source_commit'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[1], text=True).strip()
        cpu = next(line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                   if line.startswith('model name'))
        report['cpu'] = cpu
        if cpu != config['cpu']:
            raise RuntimeError('different CPU: timing baseline not applicable')
        repeats = config['repeats']
        if type(repeats) is not int or repeats < 3 or repeats > 20:
            raise ValueError('repeats must be 3..20')
        fixtures = []
        limits = {}
        for case in config['cases']:
            path = Path(case['wav'])
            if path.stat().st_size > 1048576:
                raise ValueError('fixture exceeds 1 MiB')
            audio = path.read_bytes()
            if hashlib.sha256(audio).hexdigest() != case['sha256']:
                raise ValueError('audio identity mismatch')
            with wave.open(io.BytesIO(audio)) as wav:
                if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 16000):
                    raise ValueError('fixture must be mono 16 kHz PCM16')
                duration = wav.getnframes() / 16000
            limit = case['max_median_seconds']
            if not isinstance(limit, (int, float)) or not math.isfinite(limit) or limit <= 0:
                raise ValueError('invalid latency budget')
            if case['name'] in limits:
                raise ValueError('duplicate case name')
            limits[case['name']] = limit
            fixtures.append((case, audio, duration))
        if not fixtures:
            raise ValueError('no fixtures configured')
        endpoint = config['endpoint'].rstrip('/')
        health = request(endpoint + '/health')
        if not health.get('ready') or health.get('active') or health.get('queued'):
            raise RuntimeError('endpoint not ready or busy')
        # First repetition is retained as warm-up, excluded only from timing assessment.
        for iteration in range(repeats + 1):
            for case, audio, duration in fixtures:
                begin = time.monotonic()
                row = request(endpoint + '/audio/transcriptions', audio)
                row.update(wall_seconds=time.monotonic() - begin, case=case['name'],
                           expected_text=case['expected_text'], audio_seconds=duration,
                           audio_sha256=case['sha256'], warmup=iteration == 0)
                report['attempts'].append(row)
                write(output, report)
                # Verify actual native worker configuration on this host.
                for side in ('encoder', 'decoder'):
                    pid = row['loaded_artifact'][side]['worker_pid']
                    env = dict(item.split(b'=', 1) for item in Path(f'/proc/{pid}/environ').read_bytes().split(b'\0') if b'=' in item)
                    row.setdefault('worker_environment', {})[side] = {
                        k: env.get(k.encode(), b'').decode() for k in config['worker_environment']}
                    if row['worker_environment'][side] != config['worker_environment']:
                        raise ValueError('native worker execution settings mismatch')
                assess([row], config['identity'], {case['name']: float('inf')}, config['max_queue_seconds'])
        report['cases'] = assess([r for r in report['attempts'] if not r['warmup']],
                                 config['identity'], limits, config['max_queue_seconds'])
        report['status'] = 'PASS'
    except urllib.error.HTTPError as exc:
        report.update(status='FAIL', reason=f'HTTP rejection: {exc.code}')
    except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
        report.update(status='INCOMPLETE', reason=str(exc))
    except Exception as exc:
        report.update(status='FAIL', reason=f'{type(exc).__name__}: {exc}')
    report['finished_at'] = datetime.now(timezone.utc).isoformat()
    write(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config, args.output)
    print(json.dumps({'status': result['status'], 'reason': result.get('reason'), 'cases': result.get('cases')}))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
