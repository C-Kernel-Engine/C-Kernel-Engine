"""Attach one approved local image to Pi's one-shot CKE OCR session (Linux)."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile


MAX_IMAGE_BYTES = 8 << 20
CONTEXT_TOKENS = 1024
JSON_FENCE = re.compile(r'```(?:json)?[ \t]*\r?\n(.*?)\r?\n```', re.DOTALL | re.IGNORECASE)


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


def strict_json(text):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result

    def finite_float(raw):
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError('non-finite JSON number')
        return value

    return json.loads(text, object_pairs_hook=object_pairs, parse_float=finite_float,
                      parse_constant=finite_float)


def make_validator(schema):
    from jsonschema import Draft202012Validator, SchemaError
    from referencing import Registry

    if not isinstance(schema, dict) or schema.get('type') != 'object':
        raise ValueError('OCR JSON schema must declare an object at the root')
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as error:
        raise ValueError('invalid Draft 2020-12 OCR JSON schema') from error
    # The default registry for this explicit instance will not retrieve remote references.
    return Draft202012Validator(schema, registry=Registry())


def structured_result(result, validator):
    from referencing.exceptions import Unresolvable

    if result['stop_reason'] != 'stop':
        raise ValueError('structured answer did not stop normally')
    text = result['text'].strip(' \t\r\n')
    fence = JSON_FENCE.fullmatch(text)
    try:
        value = strict_json(fence[1].strip(' \t\r\n') if fence else text)
    except (ValueError, RecursionError) as error:
        raise ValueError('expected one JSON object or one complete JSON fence') from error
    if not isinstance(value, dict):
        raise ValueError('structured answer must be a JSON object')
    try:
        valid = validator.is_valid(value)
    except (Unresolvable, RecursionError) as error:
        raise ValueError('could not resolve or evaluate the local OCR JSON schema') from error
    if not valid:
        raise ValueError('structured answer did not match the OCR JSON schema')
    return dict(result, data=value, json_source='fenced' if fence else 'plain')


def run(pi, endpoint, model, root, image, prompt, timeout, *, max_tokens=32):
    if type(max_tokens) is not int or not 1 <= max_tokens < CONTEXT_TOKENS:
        raise ValueError('output budget must be between 1 and 1023 tokens')
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
                        'contextWindow': CONTEXT_TOKENS, 'maxTokens': max_tokens,
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
    parser.add_argument('--max-tokens', type=int, default=32, help='requested output budget; server enforces total context')
    parser.add_argument('--json-schema', type=Path, help='local Draft 2020-12 schema for an OCR JSON object')
    args = parser.parse_args()
    if not 1 <= args.timeout <= 1800:
        parser.error('timeout must be between 1 and 1800 seconds')
    try:
        validator = make_validator(strict_json(args.json_schema.read_text())) if args.json_schema else None
        result = run(args.pi, args.base_url, args.model, args.approved_root, args.image, args.prompt,
                     args.timeout, max_tokens=args.max_tokens)
    except subprocess.TimeoutExpired:
        parser.exit(1, 'Pi vision: timed out before a completed assistant response\n')
    except (OSError, ValueError, ImportError) as error:
        parser.exit(1, f'Pi vision: {error}\n')
    if validator is not None:
        try:
            result = structured_result(result, validator)
        except ValueError as error:
            print(json.dumps(dict(result, validation_error=str(error))))
            return 1
    print(json.dumps(result))
    return 0 if result['stop_reason'] == 'stop' else 1


if __name__ == '__main__':
    raise SystemExit(main())
