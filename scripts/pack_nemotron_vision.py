"""Copy unchanged Nano VL vision tensors out of the pinned publisher shard."""
import argparse
import json
from pathlib import Path

from server.vision_nemotron import PROJECTOR_SHAPES, VisionWeights


def pack(shard, output):
    source = VisionWeights(shard)
    norm = 'vision_model.radio_model.input_conditioner.norm_'
    names = {norm + 'mean': 'F32', norm + 'std': 'F32'}
    names.update({name: 'BF16' for name in PROJECTOR_SHAPES})
    from version.v8.scripts.nemotron_radio_checked_v8 import LAYERS, RADIO
    for name in ('cls_token.token', 'embedder.weight', 'pos_embed'):
        names[RADIO + 'patch_generator.' + name] = 'BF16'
    for layer in range(LAYERS):
        for name in ('attn.proj', 'attn.qkv', 'norm1', 'norm2', 'mlp.fc1', 'mlp.fc2'):
            for suffix in ('weight', 'bias'):
                names[f'{RADIO}blocks.{layer}.{name}.{suffix}'] = 'BF16'
    tensors = {name: source.read(name, dtype=dtype) for name, dtype in names.items()}
    index = {}
    offset = 0
    for name, tensor in tensors.items():
        index[name] = {'shape': tensor.shape, 'dtype': names[name],
                       'data_offsets': [offset, offset + tensor.nbytes]}
        offset += tensor.nbytes
    head = json.dumps(index).encode()
    head += b' ' * (-len(head) % 8)
    with Path(output).open('xb') as target:
        target.write(len(head).to_bytes(8, 'little'))
        target.write(head)
        for tensor in tensors.values():
            tensor.tofile(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('publisher_shard', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    pack(args.publisher_shard, args.output)


if __name__ == '__main__':
    main()
