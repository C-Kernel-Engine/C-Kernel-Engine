#!/usr/bin/env python3
"""
visualize_kernel_flow.py - Kernel Pipeline Visualization Tool for CK-Engine v6.6

Reads lowered_prefill.json/lowered_decode.json and generates an interactive HTML
visualization showing:
1. Kernel connections as a flow graph
2. Data dimensions and dtypes at each stage
3. Memory offsets for inputs/outputs
4. Layer-by-layer navigation

Usage:
    python visualize_kernel_flow.py lowered_prefill.json --output kernel_flow.html
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any


def load_ir(path: str) -> Dict:
    """Load lowered IR JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def format_bytes(size: int) -> str:
    """Format byte size to human readable."""
    if size >= 1024**3:
        return f"{size / 1024**3:.2f} GB"
    elif size >= 1024**2:
        return f"{size / 1024**2:.2f} MB"
    elif size >= 1024:
        return f"{size / 1024:.2f} KB"
    return f"{size} B"


def get_buffer_color(name: str) -> str:
    """Get color for buffer type."""
    colors = {
        "token": "#4ecdc4",
        "embedded": "#45b7d1",
        "layer": "#96ceb4",
        "residual": "#ffeaa7",
        "q_scratch": "#dfe6e9",
        "k_scratch": "#b2bec3",
        "v_scratch": "#636e72",
        "attn": "#fd79a8",
        "mlp": "#a29bfe",
        "logits": "#00b894",
        "kv_cache": "#e17055",
        "rope": "#74b9ff",
    }
    for key, color in colors.items():
        if key in name.lower():
            return color
    return "#b2bec3"


def generate_html(ir: Dict, output_path: str):
    """Generate interactive HTML visualization."""
    config = ir.get("config", {})
    operations = ir.get("operations", [])
    memory = ir.get("memory", {})

    # Build nodes and edges for visualization
    nodes = []
    edges = []
    buffer_nodes = {}  # Track buffer nodes

    for i, op in enumerate(operations):
        op_id = f"op_{i}"
        kernel = op.get("kernel", "unknown")
        op_type = op.get("op", "unknown")
        layer = op.get("layer", -1)
        section = op.get("section", "body")

        # Create operation node
        nodes.append({
            "id": op_id,
            "type": "operation",
            "label": kernel,
            "op": op_type,
            "layer": layer,
            "section": section,
            "params": op.get("params", {}),
            "idx": i
        })

        # Add input buffers and edges
        for input_name, input_data in op.get("activations", {}).items():
            buf_name = input_data.get("buffer", input_name)
            buf_id = f"buf_{buf_name}"

            if buf_id not in buffer_nodes:
                buffer_nodes[buf_id] = {
                    "id": buf_id,
                    "type": "buffer",
                    "label": buf_name,
                    "dtype": input_data.get("dtype", "fp32"),
                    "offset": input_data.get("activation_offset", 0),
                    "color": get_buffer_color(buf_name)
                }
                nodes.append(buffer_nodes[buf_id])

            edges.append({
                "from": buf_id,
                "to": op_id,
                "label": f"{input_name}",
                "offset": input_data.get("activation_offset", 0)
            })

        # Add weight inputs
        for weight_name, weight_data in op.get("weights", {}).items():
            wt_id = f"wt_{weight_data.get('name', weight_name)}"
            if wt_id not in buffer_nodes:
                buffer_nodes[wt_id] = {
                    "id": wt_id,
                    "type": "weight",
                    "label": weight_data.get("name", weight_name),
                    "dtype": weight_data.get("dtype", "?"),
                    "offset": weight_data.get("bump_offset", 0),
                    "color": "#00ff88"
                }
                nodes.append(buffer_nodes[wt_id])

            edges.append({
                "from": wt_id,
                "to": op_id,
                "label": weight_name,
                "weight": True
            })

        # Add output buffers and edges
        for output_name, output_data in op.get("outputs", {}).items():
            buf_name = output_data.get("buffer", output_name)
            buf_id = f"buf_{buf_name}"

            if buf_id not in buffer_nodes:
                buffer_nodes[buf_id] = {
                    "id": buf_id,
                    "type": "buffer",
                    "label": buf_name,
                    "dtype": output_data.get("dtype", "fp32"),
                    "offset": output_data.get("activation_offset", 0),
                    "color": get_buffer_color(buf_name)
                }
                nodes.append(buffer_nodes[buf_id])

            edges.append({
                "from": op_id,
                "to": buf_id,
                "label": output_name,
                "offset": output_data.get("activation_offset", 0)
            })

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>CK-Engine Kernel Flow Visualization</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.28.1/cytoscape.min.js"></script>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }}
        .header {{
            background: #0f3460;
            padding: 15px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid #e94560;
        }}
        .header h1 {{ color: #e94560; font-size: 20px; }}
        .controls {{
            display: flex;
            gap: 15px;
            align-items: center;
        }}
        .controls select, .controls button {{
            padding: 8px 15px;
            border: none;
            border-radius: 5px;
            background: #16213e;
            color: #eee;
            cursor: pointer;
        }}
        .controls button:hover {{ background: #e94560; }}
        .main {{ display: flex; flex: 1; overflow: hidden; }}
        #cy {{
            flex: 1;
            background: #1a1a2e;
        }}
        .sidebar {{
            width: 400px;
            background: #16213e;
            padding: 20px;
            overflow-y: auto;
            border-left: 2px solid #0f3460;
        }}
        .sidebar h2 {{
            color: #e94560;
            margin-bottom: 15px;
            font-size: 16px;
        }}
        .info-card {{
            background: #1a1a2e;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
        }}
        .info-card h3 {{
            color: #4ecdc4;
            margin-bottom: 10px;
            font-size: 14px;
        }}
        .info-row {{
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid #0f3460;
            font-size: 12px;
        }}
        .info-row:last-child {{ border: none; }}
        .info-label {{ color: #888; }}
        .info-value {{ color: #fff; font-family: monospace; }}
        .buffer-list {{ margin-top: 10px; }}
        .buffer-item {{
            background: #0f3460;
            padding: 8px 12px;
            margin: 5px 0;
            border-radius: 5px;
            font-size: 11px;
        }}
        .buffer-item .name {{ color: #4ecdc4; font-weight: bold; }}
        .buffer-item .details {{ color: #888; margin-top: 3px; }}
        .stats {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-bottom: 20px;
        }}
        .stat {{
            background: #1a1a2e;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-value {{ font-size: 24px; color: #e94560; font-weight: bold; }}
        .stat-label {{ font-size: 10px; color: #888; margin-top: 5px; }}
        .legend {{
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            margin-top: 10px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 5px;
            font-size: 11px;
        }}
        .legend-color {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>CK-Engine v6.6 Kernel Flow</h1>
        <div class="controls">
            <select id="layerFilter">
                <option value="all">All Layers</option>
                {"".join(f'<option value="{i}">Layer {i}</option>' for i in range(-1, config.get("num_layers", 24)))}
            </select>
            <select id="sectionFilter">
                <option value="all">All Sections</option>
                <option value="header">Header</option>
                <option value="body">Body</option>
                <option value="footer">Footer</option>
            </select>
            <button onclick="resetView()">Reset View</button>
            <button onclick="fitView()">Fit</button>
        </div>
    </div>

    <div class="main">
        <div id="cy"></div>
        <div class="sidebar">
            <div class="stats">
                <div class="stat">
                    <div class="stat-value">{len(operations)}</div>
                    <div class="stat-label">Operations</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{config.get("num_layers", "?")}</div>
                    <div class="stat-label">Layers</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{config.get("embed_dim", "?")}</div>
                    <div class="stat-label">Embed Dim</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{config.get("num_heads", "?")}</div>
                    <div class="stat-label">Heads</div>
                </div>
            </div>

            <div class="legend">
                <div class="legend-item"><div class="legend-color" style="background: #e94560;"></div>Kernel</div>
                <div class="legend-item"><div class="legend-color" style="background: #00ff88;"></div>Weight</div>
                <div class="legend-item"><div class="legend-color" style="background: #45b7d1;"></div>Buffer</div>
            </div>

            <h2>Selected Element</h2>
            <div id="selection-info">
                <div class="info-card">
                    <h3>Click a node to see details</h3>
                    <p style="color: #888; font-size: 12px;">Use mouse wheel to zoom, drag to pan</p>
                </div>
            </div>

            <h2>Model Config</h2>
            <div class="info-card">
                {"".join(f'<div class="info-row"><span class="info-label">{k}</span><span class="info-value">{v}</span></div>' for k, v in sorted(config.items()))}
            </div>
        </div>
    </div>

    <script>
        const nodes = {json.dumps(nodes)};
        const edges = {json.dumps(edges)};
        const operations = {json.dumps(operations)};

        // Convert to Cytoscape format
        const elements = [];

        nodes.forEach(n => {{
            let color = '#e94560';
            let shape = 'roundrectangle';

            if (n.type === 'buffer') {{
                color = n.color || '#45b7d1';
                shape = 'ellipse';
            }} else if (n.type === 'weight') {{
                color = '#00ff88';
                shape = 'diamond';
            }}

            elements.push({{
                data: {{
                    id: n.id,
                    label: n.label,
                    ...n
                }},
                style: {{
                    'background-color': color,
                    'shape': shape
                }}
            }});
        }});

        edges.forEach((e, i) => {{
            elements.push({{
                data: {{
                    id: 'edge_' + i,
                    source: e.from,
                    target: e.to,
                    label: e.label,
                    ...e
                }}
            }});
        }});

        const cy = cytoscape({{
            container: document.getElementById('cy'),
            elements: elements,
            style: [
                {{
                    selector: 'node',
                    style: {{
                        'label': 'data(label)',
                        'text-valign': 'center',
                        'color': '#fff',
                        'font-size': '10px',
                        'text-outline-color': '#000',
                        'text-outline-width': 1,
                        'width': 'label',
                        'height': 30,
                        'padding': '10px',
                        'border-width': 2,
                        'border-color': '#0f3460'
                    }}
                }},
                {{
                    selector: 'node[type="operation"]',
                    style: {{
                        'background-color': '#e94560',
                        'shape': 'roundrectangle'
                    }}
                }},
                {{
                    selector: 'node[type="buffer"]',
                    style: {{
                        'background-color': 'data(color)',
                        'shape': 'ellipse'
                    }}
                }},
                {{
                    selector: 'node[type="weight"]',
                    style: {{
                        'background-color': '#00ff88',
                        'shape': 'diamond'
                    }}
                }},
                {{
                    selector: 'edge',
                    style: {{
                        'width': 2,
                        'line-color': '#555',
                        'target-arrow-color': '#555',
                        'target-arrow-shape': 'triangle',
                        'curve-style': 'bezier',
                        'label': 'data(label)',
                        'font-size': '8px',
                        'color': '#888',
                        'text-rotation': 'autorotate'
                    }}
                }},
                {{
                    selector: 'edge[weight]',
                    style: {{
                        'line-color': '#00ff88',
                        'target-arrow-color': '#00ff88'
                    }}
                }},
                {{
                    selector: ':selected',
                    style: {{
                        'border-width': 4,
                        'border-color': '#fff'
                    }}
                }}
            ],
            layout: {{
                name: 'breadthfirst',
                directed: true,
                padding: 50,
                spacingFactor: 1.5
            }}
        }});

        // Handle node selection
        cy.on('tap', 'node', function(evt) {{
            const node = evt.target;
            const data = node.data();
            updateSelectionInfo(data);
        }});

        function updateSelectionInfo(data) {{
            const container = document.getElementById('selection-info');

            if (data.type === 'operation') {{
                const op = operations[data.idx];
                let html = `<div class="info-card">
                    <h3>${{data.label}}</h3>
                    <div class="info-row"><span class="info-label">Op Type</span><span class="info-value">${{data.op}}</span></div>
                    <div class="info-row"><span class="info-label">Layer</span><span class="info-value">${{data.layer}}</span></div>
                    <div class="info-row"><span class="info-label">Section</span><span class="info-value">${{data.section}}</span></div>
                </div>`;

                if (op) {{
                    // Inputs
                    const inputs = Object.entries(op.activations || {{}});
                    if (inputs.length > 0) {{
                        html += `<div class="info-card"><h3>Inputs</h3><div class="buffer-list">`;
                        inputs.forEach(([name, info]) => {{
                            html += `<div class="buffer-item">
                                <div class="name">${{name}}: ${{info.buffer}}</div>
                                <div class="details">Offset: 0x${{info.activation_offset?.toString(16).toUpperCase()}} | Dtype: ${{info.dtype}}</div>
                            </div>`;
                        }});
                        html += `</div></div>`;
                    }}

                    // Weights
                    const weights = Object.entries(op.weights || {{}});
                    if (weights.length > 0) {{
                        html += `<div class="info-card"><h3>Weights</h3><div class="buffer-list">`;
                        weights.forEach(([name, info]) => {{
                            html += `<div class="buffer-item">
                                <div class="name">${{name}}: ${{info.name}}</div>
                                <div class="details">Offset: 0x${{info.bump_offset?.toString(16).toUpperCase()}} | Dtype: ${{info.dtype}} | Size: ${{info.size}}</div>
                            </div>`;
                        }});
                        html += `</div></div>`;
                    }}

                    // Outputs
                    const outputs = Object.entries(op.outputs || {{}});
                    if (outputs.length > 0) {{
                        html += `<div class="info-card"><h3>Outputs</h3><div class="buffer-list">`;
                        outputs.forEach(([name, info]) => {{
                            html += `<div class="buffer-item">
                                <div class="name">${{name}}: ${{info.buffer}}</div>
                                <div class="details">Offset: 0x${{info.activation_offset?.toString(16).toUpperCase()}} | Dtype: ${{info.dtype}}</div>
                            </div>`;
                        }});
                        html += `</div></div>`;
                    }}

                    // Params
                    const params = Object.entries(op.params || {{}});
                    if (params.length > 0) {{
                        html += `<div class="info-card"><h3>Parameters</h3>`;
                        params.forEach(([k, v]) => {{
                            html += `<div class="info-row"><span class="info-label">${{k}}</span><span class="info-value">${{v}}</span></div>`;
                        }});
                        html += `</div>`;
                    }}
                }}

                container.innerHTML = html;
            }} else {{
                container.innerHTML = `<div class="info-card">
                    <h3>${{data.label}}</h3>
                    <div class="info-row"><span class="info-label">Type</span><span class="info-value">${{data.type}}</span></div>
                    <div class="info-row"><span class="info-label">Dtype</span><span class="info-value">${{data.dtype}}</span></div>
                    <div class="info-row"><span class="info-label">Offset</span><span class="info-value">0x${{data.offset?.toString(16).toUpperCase()}}</span></div>
                </div>`;
            }}
        }}

        // Filtering
        document.getElementById('layerFilter').addEventListener('change', applyFilters);
        document.getElementById('sectionFilter').addEventListener('change', applyFilters);

        function applyFilters() {{
            const layer = document.getElementById('layerFilter').value;
            const section = document.getElementById('sectionFilter').value;

            cy.nodes('[type="operation"]').forEach(node => {{
                const data = node.data();
                let show = true;

                if (layer !== 'all' && data.layer !== parseInt(layer)) show = false;
                if (section !== 'all' && data.section !== section) show = false;

                if (show) {{
                    node.style('display', 'element');
                }} else {{
                    node.style('display', 'none');
                }}
            }});

            cy.layout({{ name: 'breadthfirst', directed: true }}).run();
        }}

        function resetView() {{
            cy.nodes().style('display', 'element');
            document.getElementById('layerFilter').value = 'all';
            document.getElementById('sectionFilter').value = 'all';
            cy.layout({{ name: 'breadthfirst', directed: true }}).run();
        }}

        function fitView() {{
            cy.fit();
        }}
    </script>
</body>
</html>
"""

    with open(output_path, 'w') as f:
        f.write(html)
    print(f"Kernel flow visualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize CK-Engine kernel flow")
    parser.add_argument("ir_file", help="Path to lowered IR JSON file")
    parser.add_argument("--output", "-o", default="kernel_flow.html", help="Output HTML file")

    args = parser.parse_args()

    try:
        ir = load_ir(args.ir_file)
    except Exception as e:
        print(f"Error loading IR: {e}", file=sys.stderr)
        return 1

    generate_html(ir, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
