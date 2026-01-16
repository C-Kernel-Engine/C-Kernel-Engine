#!/usr/bin/env python3
"""Generate HTML test report from nightly_report.json"""

import json
import sys
from datetime import datetime

def generate_html(report_path: str, output_path: str):
    with open(report_path) as f:
        data = json.load(f)

    summary = data.get('summary', {})
    results = data.get('results', [])

    passed = summary.get('passed', 0)
    failed = summary.get('failed', 0)
    total = summary.get('total', 0)

    # Group by category
    categories = {}
    for r in results:
        cat = r.get('category', 'other')
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M UTC')

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CK-Engine Test Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1000px; margin: 0 auto; }}
        h1 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
        .summary {{ display: flex; gap: 20px; margin: 20px 0; }}
        .stat {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; flex: 1; }}
        .stat-value {{ font-size: 36px; font-weight: bold; }}
        .stat-label {{ color: #666; margin-top: 5px; }}
        .passed {{ color: #28a745; }}
        .failed {{ color: #dc3545; }}
        .category {{ background: white; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); overflow: hidden; }}
        .category-header {{ background: #007bff; color: white; padding: 15px 20px; font-weight: bold; }}
        .test-list {{ padding: 0; margin: 0; list-style: none; }}
        .test-item {{ padding: 12px 20px; border-bottom: 1px solid #eee; display: flex; justify-content: space-between; align-items: center; }}
        .test-item:last-child {{ border-bottom: none; }}
        .test-name {{ font-weight: 500; }}
        .test-status {{ padding: 4px 12px; border-radius: 4px; font-size: 12px; font-weight: bold; }}
        .status-pass {{ background: #d4edda; color: #155724; }}
        .status-fail {{ background: #f8d7da; color: #721c24; }}
        .status-skip {{ background: #fff3cd; color: #856404; }}
        .timestamp {{ color: #666; font-size: 14px; margin-top: 20px; text-align: center; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔧 CK-Engine Nightly Test Report</h1>

        <div class="summary">
            <div class="stat">
                <div class="stat-value passed">{passed}</div>
                <div class="stat-label">Passed</div>
            </div>
            <div class="stat">
                <div class="stat-value failed">{failed}</div>
                <div class="stat-label">Failed</div>
            </div>
            <div class="stat">
                <div class="stat-value">{total}</div>
                <div class="stat-label">Total</div>
            </div>
        </div>
'''

    for cat, tests in sorted(categories.items()):
        html += f'''
        <div class="category">
            <div class="category-header">{cat.replace('_', ' ').title()} ({len(tests)} tests)</div>
            <ul class="test-list">
'''
        for t in tests:
            status = t.get('status', 'unknown')
            status_class = 'status-pass' if status == 'pass' else 'status-fail' if status in ('fail', 'timeout') else 'status-skip'
            status_text = '✓ PASS' if status == 'pass' else '✗ FAIL' if status == 'fail' else '⏱ TIMEOUT' if status == 'timeout' else '○ SKIP'

            html += f'''                <li class="test-item">
                    <span class="test-name">{t.get('name', 'Unknown')}</span>
                    <span class="test-status {status_class}">{status_text}</span>
                </li>
'''
        html += '''            </ul>
        </div>
'''

    html += f'''
        <p class="timestamp">Generated: {timestamp}</p>
    </div>
</body>
</html>
'''

    with open(output_path, 'w') as f:
        f.write(html)

    print(f"Generated: {output_path}")
    print(f"Summary: {passed}/{total} passed")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <nightly_report.json> <output.html>")
        sys.exit(1)

    generate_html(sys.argv[1], sys.argv[2])
