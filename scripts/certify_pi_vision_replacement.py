"""Test single-image session replacement in one Pi 0.73.1 RPC process."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from scripts.certify_vision_repl import LEFT
from scripts.pi_vision_once import read_image, write_pi_profile
from scripts.vision_repl import VisionHTTP


def call(pi, command):
    pi.stdin.write(json.dumps(command) + '\n')
    pi.stdin.flush()
    acknowledged = False
    for line in pi.stdout:
        event = json.loads(line)
        if event.get('type') == 'response' and event.get('id') == command['id']:
            if event.get('success') is not True or event.get('command') != command['type']:
                raise ValueError(f"Pi rejected {command['type']}")
            if command['type'] == 'new_session':
                if event.get('data', {}).get('cancelled') is not False:
                    raise ValueError('Pi did not replace the session')
                return None
            acknowledged = True
        elif command['type'] == 'prompt' and event.get('type') == 'agent_end':
            messages = event.get('messages', [])
            if (not acknowledged or [m.get('role') for m in messages] != ['user', 'assistant']
                    or [part for part in messages[0].get('content', []) if part.get('type') == 'image']
                    != command['images']):
                raise ValueError('Pi did not keep exactly the replacement image and turn')
            assistant = messages[-1]
            text = ''.join(b['text'] for b in assistant.get('content', []) if b.get('type') == 'text')
            return {'text': text, 'stop_reason': assistant.get('stopReason'),
                    'request_id': assistant.get('responseId'), 'usage': assistant.get('usage')}
    raise ValueError('Pi RPC stream ended before completing the command')


def run(args, report):
    client = VisionHTTP(args.base_url, args.model, 10, 12)

    def identity():
        with client.opener.open(client.base_url + '/v1/capabilities', timeout=10) as response:
            caps = json.load(response)
        if not (caps.get('capabilities', {}).get('vision') is True and caps.get('server_instance_id')
                and caps.get('bundle_identity')):
            raise ValueError('missing pinned image service identity')
        return caps

    boot = identity()
    context = boot['capabilities'].get('context_length')
    if type(context) is not int or context <= 12:
        raise ValueError('invalid served context capacity')
    report['bundle_identity'] = boot['bundle_identity']
    executable = shutil.which(args.pi)
    if not executable:
        raise ValueError('Pi executable not found')
    with tempfile.TemporaryDirectory(prefix='cke-pi-replacement-') as tmp:
        work = Path(tmp)
        write_pi_profile(work, client.base_url, args.model, context, 12)
        pi = subprocess.Popen([str(Path(executable).resolve()), '--offline', '--no-session', '--no-extensions',
            '--no-skills', '--no-context-files', '--no-tools', '--mode', 'rpc', '--model', 'local-cke/' + args.model],
            cwd=work, env=dict(os.environ, PI_CODING_AGENT_DIR=tmp), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, text=True)
        ids = set()
        try:
            for step, (name, expected) in enumerate((('red-blue.png', 'red'), ('blue-red.png', 'blue'),
                                                    ('red-blue.png', 'red'))):
                if step:
                    call(pi, {'id': f'reset-{step}', 'type': 'new_session'})
                suffix, data = read_image(args.fixtures, name)
                if suffix != '.png':
                    raise ValueError('expected a public PNG fixture')
                result = call(pi, {'id': f'turn-{step}', 'type': 'prompt', 'message': LEFT,
                    'images': [{'type': 'image', 'mimeType': 'image/png',
                                'data': base64.b64encode(data).decode('ascii')}]})
                report['answers'].append(dict(result, image=name, expected=expected,
                                              image_sha256=hashlib.sha256(data).hexdigest()))
                if result['stop_reason'] != 'stop' or result['text'].strip().casefold() != expected:
                    raise ValueError(f'{name}: wrong answer or incomplete generation')
                if not result['request_id'] or result['request_id'] in ids:
                    raise ValueError('missing or reused completion ID')
                ids.add(result['request_id'])
                print(f'{name}: {expected}', flush=True)
        finally:
            if pi.poll() is None:
                pi.terminate()
                try:
                    pi.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pi.kill()
                    pi.wait()
    current = identity()
    if any(current.get(key) != boot.get(key) for key in ('server_instance_id', 'bundle_identity', 'loaded_libraries')):
        raise ValueError('image service changed during the Pi run')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:8120')
    parser.add_argument('--pi', default='pi')
    parser.add_argument('--fixtures', type=Path, default=Path('tests/fixtures/vision_http_acceptance'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = {'status': 'INCOMPLETE', 'model': args.model, 'answers': []}
    with args.output.open('x') as output:
        try:
            run(args, report)
            report['status'] = 'PASS'
        except KeyboardInterrupt:
            report['error'] = 'interrupted'
        except (OSError, ValueError) as error:
            report.update(status='FAIL', error=str(error))
        finally:
            json.dump(report, output, indent=2)
            output.write('\n')
    print(report['status'])
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
