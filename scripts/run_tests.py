#!/usr/bin/env python3
"""
Run all C-Kernel-Engine tests and generate a results summary for documentation.

Executes pytest on all unittest files and captures pass/fail status,
max differences, and timing for inclusion in the PyTorch Parity page.
"""

import subprocess
import json
import re
import os
import sys
from pathlib import Path
from datetime import datetime

def run_tests():
    """Run all tests and collect results."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent
    unittest_dir = project_root / 'unittest'
    build_dir = project_root / 'build'

    # Ensure library is built
    if not (build_dir / 'libckernel_engine.so').exists():
        print("Warning: libckernel_engine.so not found. Run 'make' first.")

    results = {
        'timestamp': datetime.now().isoformat(),
        'tests': [],
        'summary': {
            'total': 0,
            'passed': 0,
            'failed': 0,
            'skipped': 0
        }
    }

    test_files = sorted(unittest_dir.glob('test_*.py'))

    for test_file in test_files:
        test_name = test_file.stem.replace('test_', '')
        print(f"Running {test_name}...")

        try:
            # Run pytest with verbose output
            result = subprocess.run(
                ['python3', '-m', 'pytest', str(test_file), '-v', '--tb=short'],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(project_root)
            )

            output = result.stdout + result.stderr

            # Parse results
            test_result = {
                'name': test_name,
                'file': test_file.name,
                'status': 'passed' if result.returncode == 0 else 'failed',
                'tests': [],
                'max_diff': None,
                'duration': None
            }

            # Extract individual test results
            for line in output.split('\n'):
                # Match test results like "test_forward PASSED" or "test_backward FAILED"
                match = re.search(r'(test_\w+)\s+(PASSED|FAILED|SKIPPED)', line)
                if match:
                    test_result['tests'].append({
                        'name': match.group(1),
                        'status': match.group(2).lower()
                    })

                # Extract max diff from output
                diff_match = re.search(r'max diff[:\s]+([0-9.e+-]+)', line, re.IGNORECASE)
                if diff_match:
                    try:
                        diff = float(diff_match.group(1))
                        if test_result['max_diff'] is None or diff > test_result['max_diff']:
                            test_result['max_diff'] = diff
                    except ValueError:
                        pass

            # Extract duration
            duration_match = re.search(r'(\d+\.\d+)s', output)
            if duration_match:
                test_result['duration'] = float(duration_match.group(1))

            results['tests'].append(test_result)
            results['summary']['total'] += 1
            if test_result['status'] == 'passed':
                results['summary']['passed'] += 1
            else:
                results['summary']['failed'] += 1

        except subprocess.TimeoutExpired:
            results['tests'].append({
                'name': test_name,
                'file': test_file.name,
                'status': 'timeout',
                'tests': [],
                'max_diff': None,
                'duration': None
            })
            results['summary']['total'] += 1
            results['summary']['failed'] += 1

        except Exception as e:
            results['tests'].append({
                'name': test_name,
                'file': test_file.name,
                'status': 'error',
                'error': str(e),
                'tests': [],
                'max_diff': None,
                'duration': None
            })
            results['summary']['total'] += 1
            results['summary']['failed'] += 1

    return results

def generate_html(results):
    """Generate HTML content for the PyTorch Parity page."""
    html_parts = []

    # Summary stats
    total = results['summary']['total']
    passed = results['summary']['passed']
    failed = results['summary']['failed']
    pass_rate = (passed / total * 100) if total > 0 else 0

    html_parts.append(f'''
<div class="grid grid-4">
    <div class="card card-accent">
        <div class="stat-number">{total}</div>
        <div class="stat-label">Total Test Suites</div>
    </div>
    <div class="card card-green">
        <div class="stat-number" style="color: var(--green);">{passed}</div>
        <div class="stat-label">Passed</div>
    </div>
    <div class="card" style="border-left-color: {'var(--green)' if failed == 0 else '#e74c3c'};">
        <div class="stat-number" style="color: {'var(--green)' if failed == 0 else '#e74c3c'};">{failed}</div>
        <div class="stat-label">Failed</div>
    </div>
    <div class="card">
        <div class="stat-number">{pass_rate:.0f}%</div>
        <div class="stat-label">Pass Rate</div>
    </div>
</div>

<p style="color: var(--text-muted); font-size: 0.85rem; margin-top: 1rem;">
    Last run: {results['timestamp'][:19].replace('T', ' ')}
</p>
''')

    # Test results table
    html_parts.append('''
<h2>Test Results</h2>

<table>
    <tr>
        <th>Kernel</th>
        <th>Status</th>
        <th>Max Diff</th>
        <th>Tests</th>
        <th>Duration</th>
    </tr>
''')

    for test in results['tests']:
        status_badge = 'badge-green' if test['status'] == 'passed' else 'badge-grey'
        status_text = test['status'].upper()

        max_diff = test.get('max_diff')
        if max_diff is not None:
            if max_diff < 1e-5:
                diff_str = f'<span style="color: var(--green);">{max_diff:.2e}</span>'
            elif max_diff < 1e-3:
                diff_str = f'<span style="color: var(--orange);">{max_diff:.2e}</span>'
            else:
                diff_str = f'<span style="color: #e74c3c;">{max_diff:.2e}</span>'
        else:
            diff_str = '-'

        test_count = len(test.get('tests', []))
        passed_count = sum(1 for t in test.get('tests', []) if t['status'] == 'passed')
        test_str = f'{passed_count}/{test_count}' if test_count > 0 else '-'

        duration = test.get('duration')
        duration_str = f'{duration:.2f}s' if duration else '-'

        html_parts.append(f'''
    <tr>
        <td><code>{test['name']}</code></td>
        <td><span class="badge {status_badge}">{status_text}</span></td>
        <td>{diff_str}</td>
        <td>{test_str}</td>
        <td>{duration_str}</td>
    </tr>
''')

    html_parts.append('</table>\n')

    # Individual test details
    html_parts.append('<h2>Detailed Results</h2>\n')

    for test in results['tests']:
        if not test.get('tests'):
            continue

        status_class = 'card-green' if test['status'] == 'passed' else ''
        html_parts.append(f'''
<details>
    <summary>{test['name']} - {len([t for t in test['tests'] if t['status'] == 'passed'])}/{len(test['tests'])} passed</summary>
    <div class="accordion-content">
        <ul>
''')
        for t in test['tests']:
            icon = '✓' if t['status'] == 'passed' else '✗'
            color = 'var(--green)' if t['status'] == 'passed' else '#e74c3c'
            html_parts.append(f'            <li><span style="color: {color};">{icon}</span> {t["name"]}</li>\n')

        html_parts.append('        </ul>\n    </div>\n</details>\n')

    return '\n'.join(html_parts)

def main():
    script_dir = Path(__file__).parent
    output_file = script_dir.parent / '_partials' / 'test_results.html'
    json_file = script_dir.parent / '_partials' / 'test_results.json'

    print("Running C-Kernel-Engine test suite...")
    results = run_tests()

    # Save JSON for debugging
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Generate HTML
    html_content = generate_html(results)
    with open(output_file, 'w') as f:
        f.write(html_content)

    print(f"\nSummary: {results['summary']['passed']}/{results['summary']['total']} passed")
    print(f"Results saved to: {output_file}")

if __name__ == '__main__':
    main()
