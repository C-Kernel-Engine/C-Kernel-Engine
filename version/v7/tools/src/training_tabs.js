import { clear, fmtExp, parseParityRows, toNum } from './utils.js';

function renderGradientPanel(files) {
    const root = document.getElementById('trainGradientRoot');
    if (!root) return;
    clear(root);

    const grad = files.training_grad_norms || {};
    const steps = Array.isArray(grad.steps) ? grad.steps : [];
    const params = grad.params && typeof grad.params === 'object' ? grad.params : {};
    const names = Object.keys(params);

    if (names.length === 0) {
        root.innerHTML = '<p style="color:var(--text-muted);">No training_grad_norms.json loaded.</p>';
        return;
    }

    const summary = names.map((name) => {
        const arr = Array.isArray(params[name]) ? params[name] : [];
        const last = arr.length > 0 ? toNum(arr[arr.length - 1], NaN) : NaN;
        return { name, last };
    }).sort((a, b) => Math.abs(b.last) - Math.abs(a.last));

    const rows = summary.slice(0, 40).map((s) => {
        let badge = '<span class="grad-healthy">healthy</span>';
        if (!Number.isFinite(s.last) || s.last < 1e-7) badge = '<span class="grad-dead">dead</span>';
        else if (s.last < 1e-5 || s.last > 0.1) badge = '<span class="grad-danger">danger</span>';
        else if (s.last < 1e-4 || s.last > 0.05) badge = '<span class="grad-warning">warning</span>';
        return `<tr><td>${s.name}</td><td>${fmtExp(s.last, 2)}</td><td>${badge}</td></tr>`;
    }).join('');

    root.innerHTML = `
        <div class="parity-section">
            <h3><span class="badge badge-orange">Gradient Health</span> Top Parameters by Latest Norm</h3>
            <div style="color:var(--text-muted);margin-bottom:0.5rem;">steps: ${steps.length} | params: ${names.length}</div>
            <table>
                <thead><tr><th>Parameter</th><th>Latest Norm</th><th>Status</th></tr></thead>
                <tbody>${rows}</tbody>
            </table>
        </div>
    `;
}

function renderParityPanel(files) {
    const root = document.getElementById('trainParityRoot');
    if (!root) return;
    clear(root);

    const rows = parseParityRows(files.training_parity);
    if (rows.length === 0) {
        root.innerHTML = '<p style="color:var(--text-muted);">No training_parity.json loaded.</p>';
        return;
    }

    const top = rows.slice(-60);
    const tableRows = top.map((r) => `
        <tr>
            <td>${r.step}</td>
            <td>${fmtExp(r.loss_diff, 2)}</td>
            <td>${fmtExp(r.max_param_diff, 2)}</td>
            <td>${r.worst_param}</td>
        </tr>
    `).join('');

    const maxDiff = rows.reduce((m, r) => Math.max(m, toNum(r.max_param_diff, 0)), 0);
    const status = maxDiff > 1e-3 ? 'critical' : (maxDiff > 1e-5 ? 'warning' : 'info');
    const msg = maxDiff > 1e-3
        ? `Parity divergence detected: ${fmtExp(maxDiff, 2)}`
        : (maxDiff > 1e-5 ? `Parity drift elevated: ${fmtExp(maxDiff, 2)}` : `Parity stable: ${fmtExp(maxDiff, 2)}`);

    root.innerHTML = `
        <div class="parity-section">
            <h3><span class="badge badge-blue">Parity Tracker</span> CK vs PyTorch</h3>
            <div class="alert-item ${status}" style="margin-bottom:0.8rem;">${msg}</div>
            <table>
                <thead><tr><th>Step</th><th>Loss Diff</th><th>Max Param Diff</th><th>Worst Param</th></tr></thead>
                <tbody>${tableRows}</tbody>
            </table>
        </div>
    `;
}

export function renderTrainingExtensionTab(tabId, files) {
    if (tabId === 'train-gradient') {
        renderGradientPanel(files);
    } else if (tabId === 'train-parity') {
        renderParityPanel(files);
    }
}
