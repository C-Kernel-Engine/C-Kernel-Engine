"""Run a live A/follow-up/B/cancel/A/reset check through one CKE image REPL."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

from scripts.vision_repl import VisionHTTP


LEFT = 'What color is on the left? Answer in one word.'
FOLLOWUP = 'What color is on the other side? Answer in one word.'


def run(args, report):
    # Reuse loopback and bounds validation before opening a socket from this client.
    http = VisionHTTP(args.base_url, args.model, args.timeout, 12)
    base = http.base_url

    def request(route):
        with http.opener.open(base + route, timeout=5) as response:
            return json.load(response)

    boot = request('/v1/capabilities')
    if not (boot.get('capabilities', {}).get('vision') is True and boot.get('server_instance_id')
            and boot.get('bundle_identity') and boot.get('loaded_libraries')):
        raise ValueError('missing preflight image capability or artifact identities')
    report['preflight'] = boot
    request_ids = set()
    proc = subprocess.Popen([sys.executable, '-m', 'scripts.vision_repl', '--approved-root', str(args.fixtures),
        '--model', args.model, '--base-url', base, '--json-events', '--max-tokens', '12', '--timeout', str(args.timeout)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

    def receive(kind):
        while line := proc.stdout.readline():
            event = json.loads(line)
            if event['type'] == kind:
                return event
            if event['type'] != 'stage':
                raise ValueError(f'expected {kind}, got {event}')
        raise RuntimeError(f'REPL exited waiting for {kind}')

    def send(line, kind):
        proc.stdin.write(line + '\n')
        proc.stdin.flush()
        return receive(kind)

    def unique_request(request_id):
        if not request_id or request_id in request_ids:
            raise ValueError('missing or reused request ID')
        request_ids.add(request_id)

    def ask(label, image, prompt, expected):
        event = send(prompt, 'answer')
        case = dict(event, case=label, expected=expected)
        report['answers'].append(case)
        unique_request(event['request_id'])
        if event['stop_reason'] != 'stop' or event['text'].strip().casefold() != expected:
            raise ValueError(f'{label}: wrong answer or incomplete generation')
        evidence = event['evidence']
        if not isinstance(evidence, dict) or evidence.get('image_sha256') != hashlib.sha256(
                (args.fixtures / image).read_bytes()).hexdigest():
            raise ValueError(f'{label}: wrong uploaded image identity')
        if evidence.get('bundle_identity') != boot['bundle_identity']:
            raise ValueError(f'{label}: artifact differs from preflight')
        print(f'{label}: {expected}', flush=True)

    try:
        receive('ready')
        send('/image red-blue.png', 'attached')
        ask('A', 'red-blue.png', LEFT, 'red')
        ask('follow-up', 'red-blue.png', FOLLOWUP, 'blue')
        send('/image blue-red.png', 'attached')
        ask('B', 'blue-red.png', LEFT, 'blue')
        stage = send('Describe the full image in detail.', 'stage')
        unique_request(stage['request_id'])
        proc.send_signal(signal.SIGINT)
        receive('cancelled')
        for _ in range(50):
            if request('/health').get('busy') is False:
                break
            time.sleep(.1)
        else:
            raise ValueError('cancelled worker did not release admission')
        report['cancelled_request'] = stage['request_id']
        send('/image red-blue.png', 'attached')
        ask('A after cancel', 'red-blue.png', LEFT, 'red')
        evidence = [case['evidence'] for case in report['answers']]
        prefixes = [event.get('prefix_sha256') for event in evidence]
        if not prefixes[0] or not prefixes[2] or not (prefixes[0] == prefixes[1] == prefixes[3] != prefixes[2]):
            raise ValueError('prefix identity did not follow the selected image')
        current = request('/v1/capabilities')
        if any(current.get(key) != boot[key] for key in ('server_instance_id', 'bundle_identity', 'loaded_libraries')):
            raise ValueError('service changed during the session')
        send('/reset', 'reset')
        event = send(LEFT, 'error')
        if event['message'] != 'attach an image with /image first':
            raise ValueError('reset did not clear the active image')
        proc.stdin.write('/quit\n')
        proc.stdin.flush()
        if proc.wait(timeout=5):
            raise ValueError('REPL exited unsuccessfully')
        report['reset'] = 'PASS'
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:8120')
    parser.add_argument('--fixtures', type=Path, default=Path('tests/fixtures/vision_http_acceptance'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=420)
    args = parser.parse_args()
    report = {'model': args.model, 'status': 'INCOMPLETE', 'answers': []}
    with args.output.open('x') as output:
        try:
            run(args, report)
            report['status'] = 'PASS'
        except KeyboardInterrupt:
            report['error'] = 'interrupted'
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
            report.update(status='FAIL', error=str(error))
        finally:
            json.dump(report, output, indent=2)
            output.write('\n')
    print(report['status'])
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
