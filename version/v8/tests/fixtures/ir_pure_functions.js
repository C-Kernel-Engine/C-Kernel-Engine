// IR Visualizer — pure utility functions exported for testing.
// These are canonical copies of functions defined in ir_visualizer.html.
// Keep in sync: if you change the source, update this fixture and bump
// the hash in ir_visualizer_contract.json.
//
// Usage: node version/v8/tests/fixtures/ir_pure_functions.js
// (self-test when run directly)

function formatBytes(bytes) {
    if (bytes >= 1024*1024*1024) return (bytes / (1024*1024*1024)).toFixed(2) + ' GB';
    if (bytes >= 1024*1024) return (bytes / (1024*1024)).toFixed(2) + ' MB';
    if (bytes >= 1024) return (bytes / 1024).toFixed(2) + ' KB';
    return bytes + ' B';
}

function normalizeShapeInput(shape) {
    if (shape === null || shape === undefined) return [];
    if (Array.isArray(shape)) {
        return shape
            .filter(v => v !== null && v !== undefined && String(v).trim() !== '')
            .map(v => String(v));
    }
    if (typeof shape === 'number' && Number.isFinite(shape)) {
        return [String(shape)];
    }
    if (typeof shape === 'string') {
        const s = shape.trim();
        if (!s) return [];
        const wrapped = (s.startsWith('[') && s.endsWith(']')) || (s.startsWith('(') && s.endsWith(')'));
        if (wrapped) {
            const inner = s.slice(1, -1).trim();
            if (!inner) return [];
            try {
                const parsed = JSON.parse(s.replace(/\(/g, '[').replace(/\)/g, ']'));
                if (Array.isArray(parsed)) return parsed.map(v => String(v));
            } catch (_) {}
            return inner.split(/[x×,\s]+/).filter(Boolean);
        }
        const parts = s.split(/[x×,\s]+/).filter(Boolean);
        if (parts.length > 1) return parts;
        return [s];
    }
    if (typeof shape === 'object') {
        if (Array.isArray(shape.shape)) return normalizeShapeInput(shape.shape);
        if (Array.isArray(shape.dims)) return normalizeShapeInput(shape.dims);
        if (Array.isArray(shape.dimensions)) return normalizeShapeInput(shape.dimensions);
        const numericKeys = Object.keys(shape)
            .filter(k => /^\d+$/.test(k))
            .sort((a, b) => Number(a) - Number(b));
        if (numericKeys.length > 0) return numericKeys.map(k => String(shape[k]));
        const scalarValues = Object.values(shape).filter(v => typeof v === 'number' || typeof v === 'string');
        if (scalarValues.length > 0 && scalarValues.length <= 4) return scalarValues.map(v => String(v));
        try { return [JSON.stringify(shape)]; } catch (_) { return ['[object]']; }
    }
    return [String(shape)];
}

function formatShapeDisplay(shape, separator = ' × ') {
    const dims = normalizeShapeInput(shape);
    if (dims.length === 0) return '-';
    if (dims.length === 1) return `[${dims[0]}]`;
    return dims.join(separator);
}

function normalizeMode(mode) {
    return mode === 'prefill' ? 'prefill' : 'decode';
}

function escapeHtml(value) {
    return String(value || '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function quoteShell(value) {
    const s = String(value || '');
    if (!s) return '';
    if (/^[A-Za-z0-9_./:-]+$/.test(s)) return s;
    return `"${s.replace(/"/g, '\\"')}"`;
}

function normalizePathString(value) {
    return String(value || '').replace(/\\/g, '/').replace(/\/+$/, '');
}

function pathDirname(value) {
    const p = normalizePathString(value);
    if (!p) return '';
    const idx = p.lastIndexOf('/');
    if (idx <= 0) return idx === 0 ? '/' : '';
    return p.slice(0, idx);
}

function extractGgufStem(modelInput) {
    const raw = String(modelInput || '');
    if (!raw) return '';
    const clean = raw.split('?')[0].replace(/\\/g, '/');
    const idx = clean.lastIndexOf('/');
    const base = idx >= 0 ? clean.slice(idx + 1) : clean;
    if (!base.toLowerCase().endsWith('.gguf')) return '';
    return base.slice(0, -5);
}

function relativePathFromTo(fromDir, toPath) {
    const from = normalizePathString(fromDir);
    const to = normalizePathString(toPath);
    if (!from || !to) return null;
    if (!from.startsWith('/') || !to.startsWith('/')) return null;
    const a = from.split('/').filter(Boolean);
    const b = to.split('/').filter(Boolean);
    let i = 0;
    while (i < a.length && i < b.length && a[i] === b[i]) i += 1;
    const up = new Array(Math.max(0, a.length - i)).fill('..');
    const down = b.slice(i);
    const parts = up.concat(down);
    return parts.length ? parts.join('/') : '.';
}

// Export for test harness (CommonJS for Node.js compatibility)
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        formatBytes, normalizeShapeInput, formatShapeDisplay,
        normalizeMode, escapeHtml, quoteShell, normalizePathString,
        pathDirname, extractGgufStem, relativePathFromTo
    };
}
