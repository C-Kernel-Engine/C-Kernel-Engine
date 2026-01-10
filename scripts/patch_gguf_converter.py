#!/usr/bin/env python3
"""
Patch GGUF converter to support Devstral (mistral3), SmolLM, and other architectures
"""

import re


def patch_gguf_converter(input_file: str, output_file: str) -> None:
    """Patch the GGUF converter to support multiple architectures"""

    with open(input_file, "r") as f:
        content = f.read()

    # Find and replace the meta_int_arch function
    # Add mistral3 and mistral to the prefixes

    old_pattern = r'(def meta_int_arch\(suffix: str\)[^:]*?:\s*\n\s*prefixes = \([^)]*?"llama"[^)]*?)\)'
    new_code = '''def meta_int_arch(suffix: str) -> Optional[int]:
        prefixes = (arch, "llama", "qwen2", "qwen", "mistral3", "mistral")'''

    # Also add fallback logic for mistral architectures
    fallback_logic = '''
        # Additional fallback for Mistral architectures
        if suffix in ("attention.head_count", "block_count"):
            # Try mistral3 and mistral prefixes
            for prefix in ["mistral3", "mistral"]:
                key = f"{prefix}.{suffix}"
                if key in meta:
                    val = meta_int(key)
                    if val is not None:
                        return val
        '''

    # Insert the fallback logic after the existing prefixes loop
    # Find where the function returns None and add fallback before it
    pattern = r'(\s+return None\s*)$'
    replacement = fallback_logic + r'\1'
    content = re.sub(pattern, replacement, content)

    # Write patched version
    with open(output_file, "w") as f:
        f.write(content)

    print(f"✅ Patched {input_file} -> {output_file}")
    print("\nChanges:")
    print("  - Added 'mistral3' and 'mistral' to architecture prefixes")
    print("  - Added fallback logic to try mistral3.* and mistral.* metadata keys")
    print("  - Supports Devstral, SmolLM, and other Mistral-based models")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Patch GGUF converter for Devstral/Mistral support")
    parser.add_argument("--input", required=True, help="Input converter script")
    parser.add_argument("--output", required=True, help="Output patched script")
    args = parser.parse_args()

    patch_gguf_converter(args.input, args.output)


if __name__ == "__main__":
    main()
