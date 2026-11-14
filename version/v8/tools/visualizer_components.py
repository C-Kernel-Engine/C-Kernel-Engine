"""Bounded, read-only artifact inventory for the v8 component viewer.

Directory membership is not execution order. This reader never loads native
libraries, runs producers, follows child symlinks or promotes reported PASS to
independently verified certification.
"""
from __future__ import annotations

from collections import deque
import hashlib
from itertools import islice
import json
import os
from pathlib import Path
import stat

MAX_FILE_BYTES = 16 << 20
MAX_TOTAL_BYTES = 64 << 20
MAX_CATALOG_BYTES = 128 << 20
MAX_FILES = 256
MAX_DIRECTORIES = 128
MAX_ENTRIES = 512
MAX_DEPTH = 2


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def _read(path, budget):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as source:
        info = os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError('not a regular file')
        limit = min(MAX_FILE_BYTES, budget)
        if info.st_size > limit:
            raise ValueError('artifact exceeds remaining byte budget')
        raw = source.read(limit + 1)
        if len(raw) > limit:
            raise ValueError('artifact grew beyond byte budget')
    # Charge malformed files as well as valid ones against the shared budget.
    digest = hashlib.sha256(raw).hexdigest()
    try:
        value = json.loads(raw, object_pairs_hook=_object)
        if not isinstance(value, dict):
            raise ValueError('expected a JSON object')
        return value, digest, len(raw), None
    except (ValueError, UnicodeError, RecursionError) as error:
        return None, digest, len(raw), str(error)


def _paths(root, issues):
    queue = deque([(root, 0)])
    directories = files = 0
    while queue:
        directory, depth = queue.popleft()
        directories += 1
        if directories > MAX_DIRECTORIES:
            issues.append('Directory limit reached; inventory is incomplete.')
            return
        try:
            with os.scandir(directory) as entries:
                selected = list(islice(entries, MAX_ENTRIES + 1))
        except OSError:
            issues.append(f'{directory.relative_to(root)}: directory unreadable')
            continue
        if len(selected) > MAX_ENTRIES:
            issues.append(f'{directory.relative_to(root)}: entry limit reached')
        for entry in sorted(selected[:MAX_ENTRIES], key=lambda item: item.name):
            path = Path(entry.path)
            if entry.is_symlink():
                issues.append(f'{path.relative_to(root)}: symlink not followed')
                continue
            if entry.is_dir(follow_symlinks=False):
                if depth < MAX_DEPTH:
                    queue.append((path, depth + 1))
                else:
                    issues.append(f'{path.relative_to(root)}: outside scan depth')
            elif entry.name.endswith('.json'):
                files += 1
                if files > MAX_FILES:
                    issues.append('JSON file limit reached; inventory is incomplete.')
                    return
                yield path


def _ops(data):
    for key in ('operations', 'ops'):
        value = data.get(key)
        if isinstance(value, list):
            return [op for op in value if isinstance(op, dict)]
    return []


def _label(value):
    if isinstance(value, (str, int, float, bool)):
        return str(value)[:2000]
    return None


def _invariants(value):
    if not isinstance(value, dict):
        return {}
    return {str(k)[:128]: v if isinstance(v, (int, float, bool)) else _label(v)
            for k, v in list(value.items())[:64] if isinstance(v, (str, int, float, bool))}


def _operation_graph(data, budget):
    """Export only explicit IR references; never infer edges from list order or slots."""
    operations = _ops(data)
    limit = min(256, budget)
    nodes = []
    for index, op in enumerate(operations[:limit]):
        flow = op.get('dataflow') if isinstance(op.get('dataflow'), dict) else op
        inputs = flow.get('inputs') if isinstance(flow.get('inputs'), dict) else {}
        outputs = flow.get('outputs') if isinstance(flow.get('outputs'), dict) else {}
        refs = []
        for name, ref in list(inputs.items())[:64]:
            if not isinstance(ref, dict):
                continue
            refs.append({'input': _label(name), 'from_op': _label(ref.get('from_op')),
                         'from_output': _label(ref.get('from_output')), 'from': _label(ref.get('from')),
                         'dtype': _label(ref.get('dtype')), 'slot': _label(ref.get('slot'))})
        nodes.append({'index': index, 'id': _label(op.get('op_id')),
                      'operation': _label(op.get('op')), 'layer': _label(op.get('layer')),
                      'provider': _label(op.get('function') or op.get('kernel') or op.get('kernel_id')),
                      'inputs': refs, 'outputs': [_label(key) for key in list(outputs)[:64]],
                      'outputs_truncated': len(outputs) > 64,
                      'bindings_truncated': len(inputs) > 64 or len(outputs) > 64 or
                          (isinstance(op.get('args'), list) and len(op['args']) > 64),
                      'buffer_refs': [_label(arg.get('buffer_ref')) for arg in
                          (op.get('args') if isinstance(op.get('args'), list) else [])[:64]
                          if isinstance(arg, dict) and arg.get('buffer_ref') is not None]})
    id_counts = {}
    for op in operations:
        identity = _label(op.get('op_id'))
        if identity is not None:
            id_counts[identity] = id_counts.get(identity, 0) + 1
    ids = {}
    for node in nodes:
        if node['id'] is not None:
            ids.setdefault(node['id'], []).append(node)
    for node in nodes:
        for ref in node['inputs']:
            source = ids.get(ref['from_op'], [])
            if ref['from_op'] is None:
                ref['status'] = 'external_or_symbolic' if ref['from'] else 'not_recorded'
            elif id_counts.get(ref['from_op'], 0) > 1:
                ref['status'] = 'ambiguous'
            elif len(source) != 1:
                ref['status'] = 'unresolved_or_outside_capture'
            elif ref['from_output'] not in source[0]['outputs']:
                ref['status'] = 'output_outside_capture' if source[0]['outputs_truncated'] else 'output_not_declared'
            else:
                ref['status'] = 'resolved'
    return {'nodes': nodes, 'total_operations': len(operations),
            'truncated': len(nodes) < len(operations), 'edge_policy': 'explicit_from_op_only'}


def _description(path, data, digest):
    config = data.get('config') if isinstance(data.get('config'), dict) else {}
    operations = _ops(data)
    providers = sorted({str(op.get('function') or op.get('kernel') or op.get('kernel_id'))
                        for op in operations if op.get('function') or op.get('kernel') or op.get('kernel_id')})
    bindings = sorted({arg['buffer_ref'] for op in operations
                       for arg in (op.get('args') if isinstance(op.get('args'), list) else [])
                       if isinstance(arg, dict) and isinstance(arg.get('buffer_ref'), str)})
    memory = data.get('memory') if isinstance(data.get('memory'), dict) else {}
    activations = memory.get('activations') if isinstance(memory.get('activations'), dict) else {}
    buffers = activations.get('buffers') if isinstance(activations.get('buffers'), list) else []
    lifetimes = [{key: _label(buffer.get(key)) for key in ('name', 'dtype', 'size', 'shape', 'lifetime')}
                 for buffer in buffers if isinstance(buffer, dict) and buffer.get('lifetime')][:128]
    schema = data.get('schema') or data.get('format') or data.get('kind')
    # Preserve the producer's labels; directory or model names do not infer modality.
    return {
        'path': path, 'sha256': digest, 'schema': _label(schema),
        'phase': _label(data.get('mode')), 'model': _label(config.get('model') or config.get('model_type')),
        'scope': _label(config.get('artifact_scope') or data.get('scope')),
        'reported_status': _label(data.get('status')),
        'operation_count': len(operations) if operations else None,
        'providers': providers[:128], 'providers_truncated': len(providers) > 128,
        'declared_lifetimes': lifetimes,
        'buffer_bindings': bindings[:128], 'bindings_truncated': len(bindings) > 128,
        'not_certified': [_label(v) for v in data.get('not_certified', [])[:64]] if isinstance(data.get('not_certified'), list) else [],
        'runtime_invariants': _invariants(config.get('runtime_invariants')),
        'has_layout': isinstance(data.get('memory'), dict) or isinstance(data.get('bump_layout'), dict),
    }


def build_component_inventory(root: Path, circuits: Path | None = None):
    root = root.resolve()
    issues = []
    artifacts = []
    budget = MAX_TOTAL_BYTES
    graph_budget = 2048
    for path in _paths(root, issues):
        relative = str(path.relative_to(root))
        try:
            value, digest, size, error = _read(path, budget)
            budget -= size
            if error:
                issues.append(f'{relative}: {error}')
                continue
            # Plain configs/manifests remain useful, but arbitrary transcripts and
            # tensor values are never embedded in this inventory.
            artifact = _description(relative, value, digest)
            artifact['operation_graph'] = _operation_graph(value, graph_budget)
            graph_budget -= len(artifact['operation_graph']['nodes'])
            artifacts.append(artifact)
        except (OSError, ValueError) as error:
            issues.append(f'{relative}: {type(error).__name__}: artifact unavailable or oversized')
        if budget <= 0:
            issues.append('Total byte budget exhausted; inventory is incomplete.')
            break
    components = {}
    for artifact in artifacts:
        component = str(Path(artifact['path']).parent)
        components.setdefault(component, []).append(artifact)
    catalog = []
    if circuits is not None:
        # This separate declaration catalog is never counted as a loaded model.
        catalog_budget = MAX_CATALOG_BYTES
        try:
            with os.scandir(circuits) as entries:
                selected = list(islice(entries, MAX_ENTRIES + 1))
            if len(selected) > MAX_ENTRIES:
                issues.append('Circuit catalog entry limit reached.')
            for entry in sorted(selected[:MAX_ENTRIES], key=lambda item: item.name):
                if entry.is_symlink() or not entry.name.endswith('.json'):
                    continue
                try:
                    value, digest, size, error = _read(Path(entry.path), catalog_budget)
                    catalog_budget -= size
                    if error:
                        raise ValueError(error)
                    contract = value.get('contract') if isinstance(value.get('contract'), dict) else {}
                    serving = value.get('serving') if isinstance(value.get('serving'), dict) else {}
                    catalog.append({'name': _label(value.get('name', entry.name)), 'family': _label(value.get('family')),
                        'path': entry.name, 'sha256': digest, 'declared_status': _label(value.get('status')),
                        'contract_sections': sorted(contract), 'serving_profile': _label(serving.get('profile_ref')),
                        'runtime_invariants': _invariants(contract.get('runtime_invariants')),
                        'checked_native_entry': _label(value.get('checked_native_entry'))})
                except (OSError, ValueError) as error:
                    reason = str(error) if isinstance(error, ValueError) else 'unreadable'
                    issues.append(f'Circuit {entry.name}: {reason}')
        except OSError:
            issues.append('Circuit declaration catalog unavailable.')
    return {'schema': 'cke.visualizer.components.v1', 'read_only': True,
            'scan_depth': MAX_DEPTH, 'components': [
                {'id': key, 'artifacts': value} for key, value in sorted(components.items())],
            'catalog': catalog, 'issues': issues,
            'connections_status': 'NOT_INFERRED', 'execution_status': 'NOT_VERIFIED',
            'bytes_read': MAX_TOTAL_BYTES - budget}
