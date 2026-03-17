// Dataset Viewer — pure utility functions exported for testing.
// These are canonical copies of functions defined in
// build_svg_dataset_visualizer_v7.py (embedded JS).
// Keep in sync: if you change the source, update this fixture.
//
// Usage: node version/v7/tests/fixtures/ds_pure_functions.js
// (self-test when run directly)

function attnColor(v, cmap) {
    const t = Math.max(0, Math.min(1, v));
    if (cmap === 'heatmap') {
        const blue   = [7, 100, 248];
        const mid    = [240, 240, 240];
        const orange = [255, 160, 0];
        if (t < 0.5) {
            const s = t * 2;
            return [Math.round(blue[0] + (mid[0] - blue[0]) * s),
                    Math.round(blue[1] + (mid[1] - blue[1]) * s),
                    Math.round(blue[2] + (mid[2] - blue[2]) * s)];
        }
        const s = (t - 0.5) * 2;
        return [Math.round(mid[0] + (orange[0] - mid[0]) * s),
                Math.round(mid[1] + (orange[1] - mid[1]) * s),
                Math.round(mid[2] + (orange[2] - mid[2]) * s)];
    }
    const targets = {
        orange: [255, 180, 0],
        blue:   [7, 173, 248],
        green:  [71, 180, 117],
    };
    const col = targets[cmap] || targets.orange;
    return [Math.round(col[0] * t), Math.round(col[1] * t), Math.round(col[2] * t)];
}

function embColor(t) {
    const blue   = [7, 173, 248];
    const mid    = [195, 200, 208];
    const orange = [255, 180, 0];
    let r, g, b;
    if (t < 0.5) {
        const s = t * 2;
        r = blue[0] + (mid[0] - blue[0]) * s;
        g = blue[1] + (mid[1] - blue[1]) * s;
        b = blue[2] + (mid[2] - blue[2]) * s;
    } else {
        const s = (t - 0.5) * 2;
        r = mid[0] + (orange[0] - mid[0]) * s;
        g = mid[1] + (orange[1] - mid[1]) * s;
        b = mid[2] + (orange[2] - mid[2]) * s;
    }
    return [Math.round(r), Math.round(g), Math.round(b)];
}

function embNormalise(matrix, mode) {
    if (!matrix || !matrix.length || !matrix[0]) return { norm: [], vmin: 0, vmax: 0, note: '' };
    const V = matrix.length, D = matrix[0].length;
    const out = matrix.map(r => Float32Array.from(r));
    if (mode === 'global') {
        let mn = Infinity, mx = -Infinity;
        for (const r of matrix) for (const v of r) { if (v < mn) mn = v; if (v > mx) mx = v; }
        const rng = (mx - mn) || 1;
        for (let i = 0; i < V; i++) for (let j = 0; j < D; j++) out[i][j] = (matrix[i][j] - mn) / rng;
        return { norm: out, vmin: mn, vmax: mx, note: '' };
    } else if (mode === 'col') {
        for (let j = 0; j < D; j++) {
            let s = 0, s2 = 0;
            for (let i = 0; i < V; i++) { s += matrix[i][j]; s2 += matrix[i][j] ** 2; }
            const mean = s / V, std = Math.sqrt(s2 / V - mean * mean) || 1;
            for (let i = 0; i < V; i++) out[i][j] = Math.max(0, Math.min(1, (matrix[i][j] - mean) / (3 * std) * 0.5 + 0.5));
        }
        return { norm: out, vmin: -3, vmax: 3, note: 'per-column z (±3σ)' };
    } else {
        for (let i = 0; i < V; i++) {
            let s = 0, s2 = 0;
            for (const v of matrix[i]) { s += v; s2 += v ** 2; }
            const mean = s / D, std = Math.sqrt(s2 / D - mean * mean) || 1;
            for (let j = 0; j < D; j++) out[i][j] = Math.max(0, Math.min(1, (matrix[i][j] - mean) / (3 * std) * 0.5 + 0.5));
        }
        return { norm: out, vmin: -3, vmax: 3, note: 'per-row z (±3σ)' };
    }
}

function cosineSim(a, b) {
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] ** 2; nb += b[i] ** 2; }
    return dot / (Math.sqrt(na * nb) || 1);
}

function attnEntropy(row) {
    let h = 0;
    for (const v of row) { if (v > 1e-9) h -= v * Math.log2(v); }
    return h;
}

function avgMatrices(matrices) {
    if (!matrices || !matrices.length || !matrices[0]) return [[]];
    const L = matrices[0].length;
    const out = Array.from({length:L}, () => new Float32Array(L));
    for (const m of matrices)
        for (let i = 0; i < L; i++)
            for (let j = 0; j < L; j++)
                out[i][j] += m[i][j] / matrices.length;
    return out.map(r => Array.from(r));
}

// ── Training chart utilities ─────────────────────────────────────
function fmtAxisVal(v) {
    if (Math.abs(v) >= 1e6) return (v / 1e6).toFixed(1) + 'M';
    if (Math.abs(v) >= 1e3) return (v / 1e3).toFixed(1) + 'K';
    if (Math.abs(v) < 0.001 && v !== 0) return v.toExponential(1);
    if (Number.isInteger(v)) return v.toString();
    return v.toPrecision(3);
}

// Export for test harness (CommonJS for Node.js compatibility)
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        attnColor, embColor, embNormalise, cosineSim, attnEntropy, avgMatrices, fmtAxisVal
    };
}
