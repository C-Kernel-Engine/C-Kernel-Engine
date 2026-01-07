#!/usr/bin/env python3
import json
import struct
import sys
import os
from pathlib import Path

def convert_tokenizer(tokenizer_path, output_dir):
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    vocab = data['model']['vocab']
    # Sort vocab by ID
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    
    # Pack strings into one blob and record offsets
    offsets = []
    strings_blob = b''
    current_offset = 0
    
    for token, _ in sorted_vocab:
        token_bytes = token.encode('utf-8')
        offsets.append(current_offset)
        strings_blob += token_bytes + b'\0' # Null terminated
        current_offset += len(token_bytes) + 1
        
    # Pack merges
    merges = data['model'].get('merges', [])
    merges_data = []
    for merge_str in merges:
        parts = merge_str.split(' ')
        if len(parts) == 2:
            # We need IDs for these parts. 
            # In some tokenizers merges are already IDs, in others they are strings.
            # HuggingFace usually has strings.
            id1 = vocab.get(parts[0], -1)
            id2 = vocab.get(parts[1], -1)
            # Resulting merged string
            res_str = parts[0] + parts[1]
            id_res = vocab.get(res_str, -1)
            if id1 != -1 and id2 != -1 and id_res != -1:
                merges_data.extend([id1, id2, id_res])

    # Write binary files
    out_path = Path(output_dir)
    with open(out_path / "vocab_offsets.bin", "wb") as f:
        f.write(struct.pack(f"{len(offsets)}i", *offsets))
    
    with open(out_path / "vocab_strings.bin", "wb") as f:
        f.write(strings_blob)
        
    with open(out_path / "vocab_merges.bin", "wb") as f:
        f.write(struct.pack(f"{len(merges_data)}i", *merges_data))
        
    print(f"Exported binary vocab: {len(sorted_vocab)} tokens, {len(merges_data)//3} merges")
    print(f"Total bytes: {len(strings_blob)}")
    
    # Update config.json with stats
    config_path = out_path / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        config['num_merges'] = len(merges_data) // 3
        config['total_vocab_bytes'] = len(strings_blob)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print("Updated config.json with vocab stats")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/tokenizer_to_bump.py <tokenizer.json> <output_dir>")
        sys.exit(1)
    convert_tokenizer(sys.argv[1], sys.argv[2])
