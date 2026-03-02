import { clear, fmt, fmtExp, toNum } from './utils.js';

function getCheckpoints(files) {
    const cps = files && files.analysis_checkpoints && Array.isArray(files.analysis_checkpoints.checkpoints)
        ? files.analysis_checkpoints.checkpoints
        : [];
    return [...cps].sort((a, b) => toNum(a.step, 0) - toNum(b.step, 0));
}

function flattenGrid(grid) {
    if (!Array.isArray(grid)) return [];
    const out = [];
    grid.forEach((row) => {
        if (Array.isArray(row)) row.forEach((v) => out.push(toNum(v, 0)));
    });
    return out;
}

function renderWeightGrid(root, grid) {
    if (!root) return;
    root.innerHTML = '';
    if (!Array.isArray(grid) || grid.length === 0) {
        root.innerHTML = '<div style="color:var(--text-muted);">No sampled_grid_32x32 available for this weight.</div>';
        return;
    }
    const values = flattenGrid(grid);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const span = Math.max(Math.abs(min), Math.abs(max), 1e-9);

    const matrix = document.createElement('div');
    matrix.className = 'weight-matrix';
    matrix.style.gridTemplateColumns = `repeat(${grid[0].length || 32}, 1fr)`;

    grid.forEach((row) => {
        row.forEach((valueRaw) => {
            const value = toNum(valueRaw, 0);
            const n = Math.max(-1, Math.min(1, value / span));
            const hue = n >= 0 ? 220 : 10;
            const sat = 72;
            const light = 50 - Math.abs(n) * 28;

            const cell = document.createElement('div');
            cell.className = 'weight-cell';
            cell.style.background = `hsl(${hue} ${sat}% ${light}%)`;
            cell.title = value.toExponential(3);
            matrix.appendChild(cell);
        });
    });

    root.appendChild(matrix);
}

function renderHistogram(root, hist, label) {
    if (!root) return;
    root.innerHTML = '';
    const counts = Array.isArray(hist && hist.counts) ? hist.counts.map((x) => toNum(x, 0)) : [];
    const bins = Array.isArray(hist && hist.bins) ? hist.bins : [];
    if (counts.length === 0) {
        root.innerHTML = `<div style="color:var(--text-muted);">No histogram for ${label}.</div>`;
        return;
    }

    const max = Math.max(...counts, 1);
    const histEl = document.createElement('div');
    histEl.className = 'act-histogram';

    counts.forEach((c, idx) => {
        const bar = document.createElement('div');
        bar.className = 'hist-bar';
        bar.style.height = `${Math.max(2, (c / max) * 100)}%`;
        const binMid = bins[idx] !== undefined ? bins[idx] : idx;
        const nearZero = Math.abs(toNum(binMid, 1)) < 1e-8;
        bar.style.background = nearZero ? '#ef4444' : '#f59e0b';
        bar.title = `bin=${binMid} count=${c}`;
        histEl.appendChild(bar);
    });

    root.appendChild(histEl);
}

function paramMovement(checkpoints) {
    const first = checkpoints[0] || {};
    const last = checkpoints[checkpoints.length - 1] || {};
    const firstW = first.weights && typeof first.weights === 'object' ? first.weights : {};
    const lastW = last.weights && typeof last.weights === 'object' ? last.weights : {};
    const deltas = last.weight_deltas && typeof last.weight_deltas === 'object' ? last.weight_deltas : {};

    const names = Array.from(new Set([...Object.keys(firstW), ...Object.keys(lastW), ...Object.keys(deltas)]));
    const rows = names.map((name) => {
        const fw = firstW[name] || {};
        const lw = lastW[name] || {};
        const d = deltas[name] || {};
        const firstNorm = toNum(fw.frobenius_norm, NaN);
        const lastNorm = toNum(lw.frobenius_norm, NaN);
        const deltaNorm = toNum(d.norm, Number.isFinite(firstNorm) && Number.isFinite(lastNorm) ? Math.abs(lastNorm - firstNorm) : NaN);
        const rel = toNum(d.relative, Number.isFinite(firstNorm) && firstNorm > 0 ? deltaNorm / firstNorm : NaN);
        return { name, rel, deltaNorm, lastNorm };
    }).filter((r) => Number.isFinite(r.rel) || Number.isFinite(r.deltaNorm));

    return rows.sort((a, b) => toNum(b.rel, -Infinity) - toNum(a.rel, -Infinity));
}

export function renderWeightActivation(files) {
    const root = document.getElementById('trainWeightsRoot');
    if (!root) return;
    clear(root);

    const checkpoints = getCheckpoints(files);
    if (checkpoints.length === 0) {
        const runCtx = getRunContext();
        const runRef = runCtx.runDir || '$RUN';
        root.innerHTML = `
            <div class="parity-section">
                <h3><span class="badge badge-orange">No Checkpoints</span> Weights/Activations Needs Analysis Snapshots</h3>
                <div style="color:var(--text-muted);margin-bottom:0.45rem;">
                    Missing artifact: <code>analysis_checkpoint_step_*.json</code>. This tab requires sampled weights/activations/gradients saved during training.
                </div>
                <pre style="font-size:0.76rem;white-space:pre-wrap;">python3 version/v7/scripts/ck_run_v7.py train --run ${runRef} --analysis-checkpoints log
python3 version/v7/tools/open_ir_visualizer.py --generate --run ${runRef} --html-only</pre>
            </div>
        `;
        return;
    }

    const last = checkpoints[checkpoints.length - 1];
    const weights = last.weights && typeof last.weights === 'object' ? last.weights : {};
    const activations = last.activations && typeof last.activations === 'object' ? last.activations : {};
    const gradients = last.gradients && typeof last.gradients === 'object' ? last.gradients : {};

    const weightNames = Object.keys(weights).sort();
    const activationNames = Object.keys(activations).sort();

    root.innerHTML = `
        <div class="parity-section">
            <h3><span class="badge badge-orange">Weight Movement</span> What Is Changing?</h3>
            <div id="weightMovementTable"></div>
        </div>
        <div class="parity-section" style="margin-top:0.8rem;">
            <h3><span class="badge badge-blue">Weight Heatmap</span> sampled 32×32</h3>
            <div style="display:flex;gap:0.6rem;align-items:center;flex-wrap:wrap;margin-bottom:0.6rem;">
                <label class="step-label">Weight</label>
                <select id="weightParamSelect" class="param-select"></select>
            </div>
            <div id="weightGridRoot"></div>
        </div>
        <div class="parity-section" style="margin-top:0.8rem;display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:0.8rem;">
            <div>
                <h3><span class="badge badge-green">Activations</span> Histogram</h3>
                <div style="display:flex;gap:0.6rem;align-items:center;flex-wrap:wrap;margin-bottom:0.5rem;">
                    <label class="step-label">Tensor</label>
                    <select id="activationSelect" class="param-select"></select>
                </div>
                <div id="activationHistRoot"></div>
            </div>
            <div>
                <h3><span class="badge badge-orange">Gradients</span> Histogram</h3>
                <div style="display:flex;gap:0.6rem;align-items:center;flex-wrap:wrap;margin-bottom:0.5rem;">
                    <label class="step-label">Tensor</label>
                    <select id="gradientSelect" class="param-select"></select>
                </div>
                <div id="gradientHistRoot"></div>
            </div>
        </div>
    `;

    const movementRows = paramMovement(checkpoints).slice(0, 120);
    const movementEl = document.getElementById('weightMovementTable');
    if (movementEl) {
        movementEl.innerHTML = movementRows.length === 0
            ? '<div style="color:var(--text-muted);">No weight movement stats available yet.</div>'
            : `<table>
                <thead><tr><th>Parameter</th><th>Relative Movement</th><th>Delta Norm</th><th>Latest Norm</th></tr></thead>
                <tbody>
                    ${movementRows.map((r) => `<tr><td>${r.name}</td><td>${fmtExp(r.rel, 2)}</td><td>${fmtExp(r.deltaNorm, 2)}</td><td>${fmtExp(r.lastNorm, 2)}</td></tr>`).join('')}
                </tbody>
            </table>`;
    }

    const weightSel = document.getElementById('weightParamSelect');
    const gridRoot = document.getElementById('weightGridRoot');
    if (weightSel) {
        weightSel.innerHTML = weightNames.map((n) => `<option value="${n}">${n}</option>`).join('');
        const defaultWeight = weightNames.find((n) => n.includes('.wq') || n.includes('.w1')) || weightNames[0] || '';
        if (defaultWeight) weightSel.value = defaultWeight;
        const draw = () => {
            const payload = weights[weightSel.value] || {};
            renderWeightGrid(gridRoot, payload.sampled_grid_32x32);
        };
        weightSel.addEventListener('change', draw);
        draw();
    }

    const actSel = document.getElementById('activationSelect');
    const actRoot = document.getElementById('activationHistRoot');
    if (actSel) {
        actSel.innerHTML = activationNames.map((n) => `<option value="${n}">${n}</option>`).join('');
        if (activationNames[0]) actSel.value = activationNames[0];
        const draw = () => renderHistogram(actRoot, (activations[actSel.value] || {}).histogram, actSel.value || 'activation');
        actSel.addEventListener('change', draw);
        draw();
    }

    const gradSel = document.getElementById('gradientSelect');
    const gradRoot = document.getElementById('gradientHistRoot');
    const gradientNames = Object.keys(gradients).sort();
    if (gradSel) {
        gradSel.innerHTML = gradientNames.map((n) => `<option value="${n}">${n}</option>`).join('');
        if (gradientNames[0]) gradSel.value = gradientNames[0];
        const draw = () => renderHistogram(gradRoot, (gradients[gradSel.value] || {}).histogram, gradSel.value || 'gradient');
        gradSel.addEventListener('change', draw);
        draw();
    }
}
