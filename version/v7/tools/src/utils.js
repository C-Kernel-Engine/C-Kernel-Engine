// Shared helpers for v7 IR visualizer training extensions.

export function getEmbeddedFiles() {
    const embedded = window.EMBEDDED_IR_DATA;
    if (!embedded || typeof embedded !== 'object') return {};
    return embedded.files && typeof embedded.files === 'object' ? embedded.files : {};
}

export function toNum(value, fallback = 0) {
    const n = Number(value);
    return Number.isFinite(n) ? n : fallback;
}

export function fmt(value, digits = 3) {
    const n = Number(value);
    if (!Number.isFinite(n)) return '-';
    return n.toFixed(digits);
}

export function fmtExp(value, digits = 3) {
    const n = Number(value);
    if (!Number.isFinite(n)) return '-';
    return n.toExponential(digits);
}

export function clear(el) {
    if (!el) return;
    while (el.firstChild) {
        el.removeChild(el.firstChild);
    }
}

export function ensurePanel(panelId, title, subtitle) {
    let panel = document.getElementById(panelId);
    if (panel) return panel;

    const main = document.querySelector('main');
    if (!main) return null;

    panel = document.createElement('div');
    panel.id = panelId;
    panel.className = 'panel';
    panel.innerHTML = `
        <h2 style="margin-bottom: 0.6rem;">${title}</h2>
        <p style="color: var(--text-muted); margin-bottom: 1rem;">${subtitle}</p>
    `;
    main.insertBefore(panel, document.getElementById('profile') || null);
    return panel;
}

export function tabIdForElement(tabEl) {
    if (!tabEl) return '';
    if (tabEl.dataset && tabEl.dataset.tab) return tabEl.dataset.tab;
    const onclick = tabEl.getAttribute('onclick') || '';
    const m = onclick.match(/showTab\('([^']+)'\)/);
    return m ? m[1] : '';
}

export function parseLossSteps(lossCurve) {
    const steps = Array.isArray(lossCurve && lossCurve.steps) ? lossCurve.steps : [];
    return steps
        .map((s) => ({
            step: toNum(s.step, NaN),
            loss_ck: toNum(s.loss_ck, NaN),
            loss_pt: toNum(s.loss_pt, NaN),
            lr: toNum(s.lr, NaN),
            grad_norm: toNum(s.grad_norm, NaN),
            epoch: toNum(s.epoch, NaN),
            source_stage: typeof s.source_stage === 'string' ? s.source_stage : null,
        }))
        .filter((s) => Number.isFinite(s.step));
}

export function parseParityRows(parity) {
    const steps = Array.isArray(parity && parity.steps) ? parity.steps : [];
    return steps
        .map((row) => ({
            step: toNum(row.step, NaN),
            loss_diff: toNum(row.loss_diff, NaN),
            max_param_diff: toNum(row.max_param_diff, NaN),
            worst_param: String(row.worst_param || '-'),
        }))
        .filter((row) => Number.isFinite(row.step));
}
