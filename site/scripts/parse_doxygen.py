#!/usr/bin/env python3
"""
Parse Doxygen XML output and generate API documentation HTML.

This script extracts function declarations from Doxygen XML and
generates an HTML partial that can be included in the docs site.
"""

import xml.etree.ElementTree as ET
import os
import sys
from pathlib import Path
import html

def parse_compound_xml(xml_file):
    """Parse a Doxygen compound XML file and extract functions."""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
    except ET.ParseError:
        return []

    functions = []

    for memberdef in root.iter('memberdef'):
        if memberdef.get('kind') != 'function':
            continue

        func = {}

        # Get function name
        name_elem = memberdef.find('name')
        func['name'] = name_elem.text if name_elem is not None else 'unknown'

        # Get return type
        type_elem = memberdef.find('type')
        func['return_type'] = ''.join(type_elem.itertext()).strip() if type_elem is not None else 'void'

        # Get brief description
        brief = memberdef.find('.//briefdescription/para')
        func['brief'] = ''.join(brief.itertext()).strip() if brief is not None else ''

        # Get detailed description
        detail = memberdef.find('.//detaileddescription/para')
        func['detail'] = ''.join(detail.itertext()).strip() if detail is not None else ''

        # Get parameters
        func['params'] = []
        for param in memberdef.findall('param'):
            param_info = {}
            param_type = param.find('type')
            param_name = param.find('declname')
            param_info['type'] = ''.join(param_type.itertext()).strip() if param_type is not None else ''
            param_info['name'] = param_name.text if param_name is not None else ''
            func['params'].append(param_info)

        # Get source location
        location = memberdef.find('location')
        if location is not None:
            func['file'] = location.get('file', '')
            func['line'] = location.get('line', '')

        functions.append(func)

    return functions

def categorize_functions(functions):
    """Categorize functions by their prefix/module."""
    categories = {
        'gemm': {'name': 'GEMM (Matrix Multiplication)', 'funcs': []},
        'layernorm': {'name': 'Layer Normalization', 'funcs': []},
        'rmsnorm': {'name': 'RMS Normalization', 'funcs': []},
        'gelu': {'name': 'GELU Activation', 'funcs': []},
        'softmax': {'name': 'Softmax', 'funcs': []},
        'attention': {'name': 'Attention', 'funcs': []},
        'mlp': {'name': 'MLP / Feed-Forward', 'funcs': []},
        'sigmoid': {'name': 'Sigmoid Activation', 'funcs': []},
        'swiglu': {'name': 'SwiGLU Activation', 'funcs': []},
        'rope': {'name': 'RoPE (Rotary Position Embedding)', 'funcs': []},
        'fc': {'name': 'Fully Connected Layers', 'funcs': []},
        'other': {'name': 'Other Functions', 'funcs': []},
    }

    for func in functions:
        name = func['name'].lower()
        categorized = False

        for prefix in ['gemm', 'layernorm', 'rmsnorm', 'gelu', 'softmax',
                       'attention', 'mlp', 'sigmoid', 'swiglu', 'rope']:
            if name.startswith(prefix) or prefix in name:
                categories[prefix]['funcs'].append(func)
                categorized = True
                break

        if not categorized:
            if name.startswith('fc') or '_fc' in name:
                categories['fc']['funcs'].append(func)
            else:
                categories['other']['funcs'].append(func)

    return categories

def generate_html(categories):
    """Generate HTML content for the API page."""

    html_parts = []

    for key, cat in categories.items():
        if not cat['funcs']:
            continue

        html_parts.append(f'''
<details open>
    <summary>{html.escape(cat['name'])} ({len(cat['funcs'])})</summary>
    <div class="accordion-content">
''')

        for func in sorted(cat['funcs'], key=lambda x: x['name']):
            # Build signature
            params_str = ', '.join([
                f"{p['type']} {p['name']}" if p['name'] else p['type']
                for p in func['params']
            ])
            signature = f"{func['return_type']} {func['name']}({params_str})"

            # Determine if forward or backward
            name_lower = func['name'].lower()
            badge = ''
            if 'backward' in name_lower:
                badge = '<span class="badge badge-blue" style="margin-left: 0.5rem;">Backward</span>'
            elif 'forward' in name_lower or not any(x in name_lower for x in ['backward', 'cache', 'backend']):
                badge = '<span class="badge badge-green" style="margin-left: 0.5rem;">Forward</span>'

            brief = func.get('brief', '') or func.get('detail', '')
            if not brief:
                # Generate description from function name
                if 'backward' in name_lower:
                    brief = 'Backward pass / gradient computation'
                elif 'forward' in name_lower:
                    brief = 'Forward pass computation'

            html_parts.append(f'''
        <div class="card" style="margin-bottom: 1rem;">
            <div style="display: flex; align-items: center; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 0.75rem;">
                <code style="font-size: 1rem; background: none; color: var(--orange); padding: 0;">{html.escape(func['name'])}</code>
                {badge}
            </div>
            <pre style="margin: 0.5rem 0; font-size: 0.8rem; overflow-x: auto;">{html.escape(signature)}</pre>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">{html.escape(brief)}</p>
        </div>
''')

        html_parts.append('    </div>\n</details>\n')

    return '\n'.join(html_parts)

def main():
    # Find the XML directory
    script_dir = Path(__file__).parent
    docs_dir = script_dir.parent.parent
    xml_dir = docs_dir / 'doxygen_output' / 'xml'

    if not xml_dir.exists():
        print(f"Error: XML directory not found at {xml_dir}")
        print("Run 'doxygen Doxyfile' in the docs directory first.")
        sys.exit(1)

    # Parse all XML files
    all_functions = []
    for xml_file in xml_dir.glob('*.xml'):
        if xml_file.name in ('index.xml', 'Doxyfile.xml'):
            continue
        functions = parse_compound_xml(xml_file)
        all_functions.extend(functions)

    # Remove duplicates
    seen = set()
    unique_functions = []
    for func in all_functions:
        if func['name'] not in seen:
            seen.add(func['name'])
            unique_functions.append(func)

    print(f"Found {len(unique_functions)} unique functions")

    # Categorize and generate HTML
    categories = categorize_functions(unique_functions)
    api_html = generate_html(categories)

    # Write output
    output_file = script_dir.parent / '_partials' / 'api_content.html'
    with open(output_file, 'w') as f:
        f.write(api_html)

    print(f"Generated API content: {output_file}")

if __name__ == '__main__':
    main()
