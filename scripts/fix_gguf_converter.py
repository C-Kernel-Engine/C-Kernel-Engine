#!/usr/bin/env python3
"""
Quick fix for GGUF converter to support Devstral/Mistral architecture
"""

import re


def patch_gguf_converter(input_file: str, output_file: str) -> None:
    """Patch the GGUF converter to support Mistral architecture"""

    with open(input_file, "r") as f:
        content = f.read()

    # Patch 1: Add Mistral architecture support
    old_arch_check = '''def meta_int_arch(suffix: str) -> Optional[int]:
            prefixes = (arch, "llama", "qwen2", "qwen")'''

    new_arch_check = '''def meta_int_arch(suffix: str) -> Optional[int]:
            prefixes = (arch, "llama", "qwen2", "qwen", "mistral3", "mistral")'''

    content = content.replace(old_arch_check, new_arch_check)

    # Patch 2: Handle Mistral's different head count metadata
    # Devstral might use different keys for attention parameters
    # Let's add more flexible metadata lookup

    # Find the meta_int_arch function and improve it
    pattern = r'(def meta_int_arch\(suffix: str\) -> Optional\[int\]:.*?seen\.add\(prefix\).*?return None)'
    replacement = r'''\1
            # Additional fallback for Mistral
            if suffix == "attention.head_count":
                # Try direct key lookup
                direct_key = f"mistral3.{suffix}"
                if direct_key in meta:
                    return meta_int(direct_key)
                # Try without prefix
                if suffix in meta:
                    return meta_int(suffix)'''

    content = re.sub(pattern, replacement, content, flags=re.DOTALL)

    with open(output_file, "w") as f:
        f.write(content)

    print(f"✅ Patched {input_file} -> {output_file}")
    print("\nChanges:")
    print("  - Added 'mistral3' and 'mistral' to architecture prefixes")
    print("  - Added fallback metadata lookup for Mistral attention params")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Patch GGUF converter for Devstral/Mistral")
    parser.add_argument("--input", required=True, help="Input converter script")
    parser.add_argument("--output", required=True, help="Output patched script")
    args = parser.parse_args()

    patch_gguf_converter(args.input, args.output)
