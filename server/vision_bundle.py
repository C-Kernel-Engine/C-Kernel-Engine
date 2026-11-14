"""Operator-pinned, immutable inputs for the experimental vision adapter."""
from pathlib import Path
import hashlib
import json

ENCODER_FILES = ('layout.json', 'weights.bump', 'weights_manifest.map',
                 'libencoder_v8.so', 'libckernel_engine.so')
DECODER_FILES = ('layout_decode.json', 'layout_prefill.json', 'config.json',
                 'weights.bump', 'weights_manifest.json', 'weights_manifest.map',
                 'libmodel.so', 'libckernel_engine.so', 'libckernel_tokenizer.so',
                 'chat_template.jinja')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def pin(encoder, decoder, tokenizer_gguf, model, image_placeholder):
    roots = {'encoder': Path(encoder).resolve(), 'decoder': Path(decoder).resolve()}
    files = {}
    for role, names in [('encoder', ENCODER_FILES), ('decoder', DECODER_FILES)]:
        for name in names:
            path = roots[role] / name
            files[f'{role}/{name}'] = sha(path)
    files['tokenizer_gguf'] = sha(tokenizer_gguf)
    files['session_library'] = sha(Path(__file__).resolve().parents[1]/'build/libck_session_v8.so')
    return dict(schema='cke.vision.http.v1', model=model,
                encoder=str(roots['encoder']), decoder=str(roots['decoder']),
                tokenizer_gguf=str(Path(tokenizer_gguf).resolve()),
                image_placeholder=image_placeholder, files=files)


def load(path):
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or data.get('schema') != 'cke.vision.http.v1':
        raise ValueError('invalid vision profile')
    if set(data) != {'schema','model','encoder','decoder','tokenizer_gguf','image_placeholder','files'}:
        raise ValueError('unknown or missing vision profile fields')
    if not all(isinstance(data.get(k), str) and data[k] for k in
               ('model','encoder','decoder','tokenizer_gguf','image_placeholder')):
        raise ValueError('invalid vision profile strings')
    if len(data['model']) > 128 or len(data['image_placeholder']) > 128:
        raise ValueError('oversized vision profile identifier')
    expected = pin(data['encoder'],data['decoder'],data['tokenizer_gguf'],data['model'],data['image_placeholder'])
    if data != expected:
        raise ValueError('vision bundle changed; rebuild and re-pin explicitly')
    data['identity'] = hashlib.sha256(json.dumps(expected,sort_keys=True).encode()).hexdigest()
    return data


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('encoder','decoder','tokenizer-gguf','model','image-placeholder','output'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    result=pin(args.encoder,args.decoder,args.tokenizer_gguf,args.model,args.image_placeholder)
    target=Path(args.output)
    with target.open('x') as stream:json.dump(result,stream,indent=2)

if __name__=='__main__':main()
