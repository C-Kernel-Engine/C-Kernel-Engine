#!/usr/bin/env python3
"""Standalone Python-standard-library Whisper client. No CKE imports required."""
import argparse
import json
import os
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request
import uuid


def transcribe(base, audio, hint='', timeout=660):
    base = base.rstrip('/')
    with urllib.request.urlopen(base + '/health', timeout=10) as response:
        health = json.load(response)
    cap = health.get('capabilities', {})
    if cap.get('schema') != 'cke.whisper.capabilities.v1':
        raise RuntimeError('Service lacks capability discovery; upgrade it before using this client.')
    if not isinstance(health.get('service_instance_id'), str) or not health['service_instance_id']:
        raise RuntimeError('Service identity is missing; cannot correlate this attempt.')
    if not health.get('ready'):
        raise RuntimeError('Whisper worker is not ready.')
    if hint and not cap.get('vocabulary_hint', {}).get('supported'):
        raise RuntimeError('This service does not support vocabulary hints.')
    if hint and len(hint.encode('utf-8')) > cap['vocabulary_hint']['max_utf8_bytes']:
        raise RuntimeError('Vocabulary hint exceeds service byte limit; tokenizer limit is checked by the server.')
    with Path(audio).open('rb') as handle:
        body = handle.read(cap['audio']['max_upload_bytes'] + 1)
    if len(body) > cap['audio']['max_upload_bytes']:
        raise RuntimeError('Audio exceeds upload limit; this interface does not support long recordings.')
    query = '?' + urllib.parse.urlencode({'vocabulary_hint': hint}) if hint else ''
    request_id = uuid.uuid4().hex
    request = urllib.request.Request(base + '/audio/transcriptions' + query, data=body,
                                     headers={'Content-Type': 'audio/wav', 'X-CKE-Request-ID': request_id})
    # No automatic retry: a lost reply does not prove native execution stopped.
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    if result.get('id') != request_id or result.get('service_instance_id') != health.get('service_instance_id'):
        raise RuntimeError('Request correlation or service identity changed; retain the attempt for review.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8110')
    parser.add_argument('--audio', required=True, type=Path)
    parser.add_argument('--vocabulary-hint', default='')
    parser.add_argument('--timeout', type=float, default=660)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        result = transcribe(args.url, args.audio, args.vocabulary_hint, args.timeout)
    except urllib.error.HTTPError as error:
        # Avoid printing the URL: its query may contain private vocabulary.
        raise SystemExit(f'Whisper HTTP {error.code}; no automatic retry. Consult service capabilities.') from None
    except (RuntimeError, urllib.error.URLError, TimeoutError, OSError) as error:
        raise SystemExit(f'Whisper request failed ({type(error).__name__}); no automatic retry. Check service readiness and limits.') from None
    fd = os.open(args.output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, 'w') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print('Transcription saved; inspect vocabulary_hint_applied and vocabulary_hint_token_count in the private result.')


if __name__ == '__main__':
    main()
