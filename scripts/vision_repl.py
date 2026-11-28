"""Interactive client-owned history for one uploaded image at a time."""
import argparse
import base64
import json
from pathlib import Path
import re
import urllib.error
import urllib.request

from scripts.pi_vision_once import read_image


MAX_LINE = 128 << 10
MAX_STREAM = 8 << 20
REQUEST_ID = re.compile(r'vision-[0-9a-f]{32}')


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class VisionHTTP:
    def __init__(self, base_url, model, timeout, max_tokens):
        endpoint = re.fullmatch(r'http://127\.0\.0\.1:([0-9]{1,5})/?', base_url)
        if not endpoint or not 1 <= int(endpoint[1]) <= 65535:
            raise ValueError('use a loopback HTTP port (SSH forwarding is supported)')
        if not 1 <= max_tokens <= 1024 or not 1 <= timeout <= 1800:
            raise ValueError('invalid output budget or timeout')
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def cancel(self, request_id):
        request = urllib.request.Request(self.base_url + '/v1/requests/' + request_id, method='DELETE')
        try:
            with self.opener.open(request, timeout=5):
                pass
        except OSError:
            # The request might already have finished; close the stream either way.
            pass

    def reply(self, messages, on_stage):
        body = {'model': self.model, 'messages': messages, 'max_tokens': self.max_tokens, 'stream': True}
        request = urllib.request.Request(self.base_url + '/v1/chat/completions',
                                         data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        with self.opener.open(request, timeout=self.timeout) as response:
            request_id = response.headers.get('X-Request-ID', '')
            if not REQUEST_ID.fullmatch(request_id) or response.headers.get_content_type() != 'text/event-stream':
                raise ValueError('invalid vision stream headers')
            try:
                return read_stream(response, request_id, on_stage)
            except KeyboardInterrupt:
                self.cancel(request_id)
                raise


def read_stream(response, request_id, on_stage):
    chunks = []
    finish = None
    size = 0
    while raw := response.readline(MAX_LINE + 1):
        size += len(raw)
        if len(raw) > MAX_LINE or size > MAX_STREAM:
            raise ValueError('vision stream exceeds byte budget')
        line = raw.strip()
        if not line:
            continue
        if line == b': native stage' and finish is None:
            on_stage(request_id)
            continue
        if not line.startswith(b'data: '):
            raise ValueError('invalid vision stream event')
        payload = line[6:]
        if payload == b'[DONE]':
            if finish is not None:
                return dict(finish, text=''.join(chunks))
            break
        event = json.loads(payload)
        if not isinstance(event, dict) or 'error' in event:
            raise ValueError('vision worker did not complete the request')
        choices = event.get('choices')
        if (finish is not None or event.get('id') != request_id or not isinstance(choices, list)
                or len(choices) != 1 or not isinstance(choices[0], dict)):
            raise ValueError('invalid vision stream choice or request identity')
        choice = choices[0]
        delta = choice.get('delta')
        content = delta.get('content', '') if isinstance(delta, dict) else None
        if not isinstance(content, str):
            raise ValueError('invalid vision stream delta')
        chunks.append(content)
        stop = choice.get('finish_reason')
        if stop is not None:
            if stop not in ('stop', 'length'):
                raise ValueError('invalid vision stream finish reason')
            finish = {'request_id': request_id, 'stop_reason': stop, 'evidence': event.get('cke_evidence'),
                      'usage': event.get('usage')}
    raise ValueError('vision stream ended without completion')


class VisionConversation:
    def __init__(self, approved_root, transport):
        self.root = approved_root
        self.transport = transport
        self.reset()

    def reset(self):
        self.image = None
        self.history = []

    def attach(self, path):
        suffix, data = read_image(self.root, path)
        mime = 'image/png' if suffix == '.png' else 'image/jpeg'
        url = f'data:{mime};base64,' + base64.b64encode(data).decode('ascii')
        self.image = {'type': 'image_url', 'image_url': {'url': url}}
        self.history = []

    def ask(self, prompt, on_stage):
        if self.image is None:
            raise ValueError('attach an image with /image first')
        if len(self.history) >= 30:
            raise ValueError('conversation is full; replace the image to start again')
        content = [{'type': 'text', 'text': prompt}]
        if not self.history:
            content.insert(0, self.image)
        messages = [*self.history, {'role': 'user', 'content': content}]
        result = self.transport.reply(messages, on_stage)
        if result['stop_reason'] == 'stop':
            self.history = [*messages, {'role': 'assistant', 'content': result['text']}]
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--approved-root', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:8120')
    parser.add_argument('--max-tokens', type=int, default=32)
    parser.add_argument('--timeout', type=int, default=420)
    parser.add_argument('--json-events', action='store_true')
    args = parser.parse_args()
    try:
        http = VisionHTTP(args.base_url, args.model, args.timeout, args.max_tokens)
    except ValueError as error:
        parser.error(str(error))
    conversation = VisionConversation(args.approved_root, http)

    def emit(kind, **fields):
        if args.json_events:
            print(json.dumps(dict(type=kind, **fields)), flush=True)
        elif kind == 'answer':
            print(fields['text'].rstrip())
            if fields['stop_reason'] == 'length':
                print('Output budget reached; turn not saved.')
        else:
            print(fields.get('message', kind), flush=True)

    emit('ready', message='/image PATH   /reset   /quit   Ctrl-C cancels an active request')
    while True:
        try:
            line = input('' if args.json_events else '> ').strip()
            if line == '/quit':
                break
            if line == '/reset':
                conversation.reset()
                emit('reset')
            elif line.startswith('/image '):
                conversation.attach(line[len('/image '):].strip())
                emit('attached')
            elif line.startswith('/'):
                raise ValueError('unknown command')
            elif line:
                result = conversation.ask(line, lambda rid: emit('stage', request_id=rid))
                emit('answer', **result)
        except EOFError:
            break
        except KeyboardInterrupt:
            emit('cancelled', message='Interrupted; turn not saved.')
        except urllib.error.HTTPError as error:
            error.close()
            emit('error', message=f'HTTP {error.code}; turn not saved.')
        except (OSError, ValueError) as error:
            emit('error', message=str(error))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
