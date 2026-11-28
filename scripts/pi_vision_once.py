"""Attach one approved local image to Pi's one-shot CKE OCR session (Linux)."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile


MAX_IMAGE_BYTES = 8 << 20


def read_image(root, relative):
    parts = Path(relative).parts
    if not parts or Path(relative).is_absolute() or '..' in parts:
        raise ValueError('image must be a relative path beneath the approved root')
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory = os.open(Path(root).resolve(strict=True), directory_flags)
    try:
        # Anchor traversal to open directory descriptors, without following links.
        for part in parts[:-1]:
            next_directory = os.open(part, directory_flags, dir_fd=directory)
            os.close(directory)
            directory = next_directory
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        with os.fdopen(fd, 'rb') as image:
            info = os.fstat(image.fileno())
            if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= MAX_IMAGE_BYTES:
                raise ValueError('expected a regular nonempty image of at most 8 MiB')
            data = image.read(MAX_IMAGE_BYTES + 1)
    finally:
        os.close(directory)
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError('image exceeds 8 MiB')
    if data.startswith(b'\x89PNG\r\n\x1a\n'):
        return '.png', data
    if data.startswith(b'\xff\xd8\xff'):
        return '.jpg', data
    raise ValueError('expected PNG or JPEG bytes')


def assistant_result(output):
    message = None
    for line in output.splitlines():
        event = json.loads(line)
        if event.get('type') == 'message_end' and event.get('message', {}).get('role') == 'assistant':
            message = event['message']
    if not message or message.get('stopReason') not in ('stop', 'length'):
        raise ValueError('Pi did not produce a completed assistant response')
    text = ''.join(block['text'] for block in message['content'] if block.get('type') == 'text')
    return {'text': text, 'stop_reason': message['stopReason']}


def run(pi, endpoint, model, root, image, prompt, timeout):
    match = re.fullmatch(r'http://127\.0\.0\.1:([0-9]{1,5})/?', endpoint)
    if not match or not 1 <= int(match[1]) <= 65535:
        raise ValueError('use a loopback HTTP port (SSH forwarding is supported)')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', model):
        raise ValueError('invalid model ID')
    if not prompt.strip() or prompt.startswith(('-', '@')):
        raise ValueError('prompt must not be empty or start with a Pi option/attachment marker')
    executable = shutil.which(pi)
    if executable is None:
        raise ValueError('Pi executable not found; install the tested Pi version separately')
    suffix, data = read_image(root, image)
    with tempfile.TemporaryDirectory(prefix='cke-pi-vision-') as directory:
        scratch = Path(directory)
        agent = scratch / 'agent'
        agent.mkdir(mode=0o700)
        attachment = scratch / ('image' + suffix)
        attachment.write_bytes(data)
        config = {'providers': {'local-cke': {
            'baseUrl': endpoint.rstrip('/') + '/v1', 'api': 'openai-completions', 'apiKey': 'local-only',
            'compat': {'supportsStore': False, 'supportsDeveloperRole': False,
                       'supportsReasoningEffort': False, 'maxTokensField': 'max_tokens'},
            'models': [{'id': model, 'name': 'Pinned Nano VL', 'input': ['text', 'image'],
                        'contextWindow': 1024, 'maxTokens': 32,
                        'cost': dict.fromkeys(('input', 'output', 'cacheRead', 'cacheWrite'), 0)}],
        }}}
        (agent / 'models.json').write_text(json.dumps(config))
        (agent / 'settings.json').write_text(json.dumps({'compaction': {'enabled': False},
                                                         'retry': {'enabled': False}}))
        env = dict(os.environ, PI_CODING_AGENT_DIR=str(agent))
        result = subprocess.run([
            str(Path(executable).resolve()), '--offline', '--no-session', '--no-extensions',
            '--no-skills', '--no-context-files', '--no-tools', '--mode', 'json', '--print',
            '--model', 'local-cke/' + model, '@' + attachment.name, prompt,
        ], cwd=scratch, env=env, timeout=timeout, check=False, stdout=subprocess.PIPE, text=True)
        if result.returncode:
            raise ValueError('Pi exited unsuccessfully')
        # Pi JSON events include the full user image as base64. Never echo them.
        return assistant_result(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--approved-root', type=Path, required=True)
    parser.add_argument('--image', required=True, help='relative to the approved root')
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:8120')
    parser.add_argument('--model', default='nemotron-nano-vl')
    parser.add_argument('--pi', default='pi', help='trusted Pi executable (tested with 0.73.1)')
    parser.add_argument('--timeout', type=int, default=420)
    args = parser.parse_args()
    if not 1 <= args.timeout <= 1800:
        parser.error('timeout must be between 1 and 1800 seconds')
    try:
        result = run(args.pi, args.base_url, args.model, args.approved_root, args.image, args.prompt, args.timeout)
    except subprocess.TimeoutExpired:
        parser.exit(1, 'Pi vision: timed out before a completed assistant response\n')
    except (OSError, ValueError) as error:
        parser.exit(1, f'Pi vision: {error}\n')
    print(json.dumps(result))
    return 0 if result['stop_reason'] == 'stop' else 1


if __name__ == '__main__':
    raise SystemExit(main())
