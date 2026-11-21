"""Pinned local LoRA datasets. Registration is administrator-owned, not client paths."""
import hashlib
import json
from pathlib import Path

from .lora_session import lora_targets, require

MAX_RECORDS = 4096
MAX_BYTES = 8 * 1024**2


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_package(path, expected):
    path = Path(path).resolve()
    require(path.stat().st_size <= MAX_BYTES, 'package capacity exceeded')
    payload = path.read_bytes()
    require(len(payload) <= MAX_BYTES and hashlib.sha256(payload).hexdigest() == expected, 'package identity mismatch')
    doc = json.loads(payload)
    require(set(doc) == {'schema', 'base', 'tokenizer', 'tokens', 'records'}, 'invalid package fields')
    require(doc['schema'] == 'cke.lora.training-package.v1', 'invalid package schema')
    base = doc['base']
    require(type(base) is dict and set(base) == {'directory', 'manifest_sha256', 'weights_sha256', 'provenance'}, 'invalid base')
    require(base['provenance'] in ('synthetic', 'pretrained'), 'explicit base provenance required')
    require(type(base['directory']) is str and not Path(base['directory']).is_absolute(), 'relative base directory required')
    directory = (path.parent / base['directory']).resolve()
    require(directory.is_relative_to(path.parent), 'base escapes package')
    require((directory/'weights_manifest.json').stat().st_size <= MAX_BYTES and (directory/'weights.bump').stat().st_size <= 256*1024**2, 'base file capacity exceeded')
    require(sha(directory / 'weights_manifest.json') == base['manifest_sha256']
            and sha(directory / 'weights.bump') == base['weights_sha256'], 'base identity mismatch')
    tokenizer = doc['tokenizer']
    require(type(tokenizer) is dict and set(tokenizer) == {'id', 'sha256'}
            and type(tokenizer['id']) is str and 0 < len(tokenizer['id']) <= 128
            and type(tokenizer['sha256']) is str and len(tokenizer['sha256']) == 64
            and all(c in '0123456789abcdef' for c in tokenizer['sha256']), 'invalid tokenizer identity')
    discovery = lora_targets(directory, tokens=doc['tokens'])
    vocab = json.loads((directory / 'weights_manifest.json').read_text())['config']['vocab_size']
    records = doc['records']
    require(type(records) is list and 2 <= len(records) <= MAX_RECORDS, 'record count out of range')
    ids, split_inputs = set(), {'train': set(), 'validation': set()}
    for row in records:
        require(type(row) is dict and set(row) == {'id', 'split', 'tokens', 'targets'}, 'invalid record')
        require(type(row['id']) is str and 0 < len(row['id']) <= 64 and row['id'] not in ids, 'duplicate or invalid record ID')
        ids.add(row['id'])
        require(row['split'] in split_inputs, 'unsupported split')
        for key in ('tokens', 'targets'):
            require(type(row[key]) is list and len(row[key]) == doc['tokens']
                    and all(type(x) is int and 0 <= x < vocab for x in row[key]), 'invalid token record')
        split_inputs[row['split']].add(tuple(row['tokens']))
    require(all(split_inputs.values()), 'train and validation required')
    require(not split_inputs['train'] & split_inputs['validation'], 'train/validation input overlap')
    return doc, directory, discovery
