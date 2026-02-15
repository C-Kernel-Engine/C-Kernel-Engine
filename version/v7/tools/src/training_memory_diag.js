import { clear, fmt, toNum } from "./utils.js";

function boolBadge(ok) {
    if (ok === true) return '<span class="badge badge-green">PASS</span>';
    if (ok === false) return '<span class="badge badge-orange">FAIL</span>';
    return '<span class="badge badge-blue">N/A</span>';
}

function shortText(v, fallback = "-") {
    if (typeof v === "string" && v.trim()) return v.trim();
    if (Number.isFinite(Number(v))) return String(v);
    return fallback;
}

function safeList(v) {
    return Array.isArray(v) ? v : [];
}

function phaseLabel(diag) {
    const phase = shortText(diag && diag.phase, "unknown");
    const idx = diag && Number.isFinite(Number(diag.index)) ? Number(diag.index) : null;
    if (idx === null) return phase;
    return `${phase} #${idx}`;
}

function slotByIndex(runtimeSummary, slotIdx) {
    const idx = Number(slotIdx);
    if (!Number.isFinite(idx) || idx < 0) return null;
    const slots = safeList(runtimeSummary && runtimeSummary.tensor_slots);
    const row = slots.find((s) => Number(s && s.index) === idx);
    return row || null;
}

function resolveFailedOp(diag, runtimeSummary, meta) {
    const opId = Number(diag && diag.failed_op_id);
    const fallbackId = Number(meta && meta.failed_op_id);
    const id = Number.isFinite(opId) && opId >= 0
        ? opId
        : (Number.isFinite(fallbackId) && fallbackId >= 0 ? fallbackId : null);
    if (id === null) return null;
    const trace = safeList(runtimeSummary && runtimeSummary.backward_op_trace);
    const row = trace.find((r) => Number(r && r.op_id) === id);
    if (row) return { op_id: id, ...row };
    return { op_id: id };
}

function buildCorruptionLocator(diag, meta, runtimeSummary) {
    const out = {
        ok: !!(diag && diag.ok === true),
        phase: shortText(diag && diag.phase, "unknown"),
        rc: Number(diag && diag.rc),
        canary_index: null,
        canary_range: null,
        left_slot: null,
        right_slot: null,
        failed_op: null,
    };

    const diagIdx = Number(diag && diag.index);
    const metaIdx = Number(meta && meta.failed_canary_idx);
    const canaryIdx = Number.isFinite(diagIdx) && diagIdx >= 0
        ? diagIdx
        : (Number.isFinite(metaIdx) && metaIdx >= 0 ? metaIdx : null);

    if (canaryIdx !== null) {
        out.canary_index = canaryIdx;
        const ranges = safeList(runtimeSummary && runtimeSummary.canary_ranges);
        if (canaryIdx < ranges.length) {
            const range = ranges[canaryIdx];
            out.canary_range = range;
            out.left_slot = slotByIndex(runtimeSummary, range && range.left_slot_idx);
            out.right_slot = slotByIndex(runtimeSummary, range && range.right_slot_idx);
        }
    }

    out.failed_op = resolveFailedOp(diag, runtimeSummary, meta);
    return out;
}

function renderLocator(locator) {
    if (locator.ok) {
        return '<div class="alert-item info">No corruption detected in latest strict diagnostic run.</div>';
    }

    const hasAny = locator.canary_index !== null || locator.failed_op !== null;
    if (!hasAny) {
        return '<div class="alert-item warning">Diagnostic failed but no index/op metadata was captured.</div>';
    }

    const range = locator.canary_range || {};
    const left = locator.left_slot || {};
    const right = locator.right_slot || {};
    const op = locator.failed_op || {};

    return `
        <table>
            <tbody>
                <tr><th>Phase</th><td>${shortText(locator.phase)}</td></tr>
                <tr><th>RC</th><td>${Number.isFinite(locator.rc) ? fmt(locator.rc, 0) : '-'}</td></tr>
                <tr><th>Canary Index</th><td>${locator.canary_index !== null ? locator.canary_index : '-'}</td></tr>
                <tr><th>Range</th><td>start=${shortText(range.start)} len=${shortText(range.length)}</td></tr>
                <tr><th>Left Slot</th><td>${shortText(left.name)} (idx=${shortText(range.left_slot_idx)})</td></tr>
                <tr><th>Right Slot</th><td>${shortText(right.name)} (idx=${shortText(range.right_slot_idx)})</td></tr>
                <tr><th>Failed Backward Op</th><td>${shortText(op.op)} (id=${shortText(op.op_id)}, kernel=${shortText(op.kernel_id)})</td></tr>
            </tbody>
        </table>`;
}

function renderChecksTable(audit) {
    const checks = safeList(audit && audit.checks);
    if (checks.length === 0) {
        return '<div style="color:var(--text-muted);">No layout audit checks loaded.</div>';
    }
    const rows = checks.map((c) => {
        const passed = !!(c && c.passed === true);
        return `
            <tr>
                <td>${shortText(c && c.name)}</td>
                <td>${passed ? '<span class="badge badge-green">pass</span>' : '<span class="badge badge-orange">fail</span>'}</td>
                <td style="font-family:JetBrains Mono,monospace;font-size:0.78rem;">${shortText(c && c.detail)}</td>
            </tr>`;
    }).join("");
    return `
        <table>
            <thead><tr><th>Check</th><th>Status</th><th>Detail</th></tr></thead>
            <tbody>${rows}</tbody>
        </table>`;
}

function renderCanaryTable(runtimeSummary, focusIndex) {
    const ranges = safeList(runtimeSummary && runtimeSummary.canary_ranges);
    if (ranges.length === 0) {
        return '<div style="color:var(--text-muted);">No canary ranges in runtime summary.</div>';
    }

    const maxRows = 24;
    let rows = ranges.slice(0, maxRows);
    if (Number.isFinite(focusIndex) && focusIndex >= 0 && focusIndex < ranges.length) {
        const start = Math.max(0, focusIndex - 4);
        rows = ranges.slice(start, Math.min(ranges.length, start + maxRows));
    }

    const htmlRows = rows.map((r) => {
        const idx = Number(r && r.index);
        const mark = Number.isFinite(focusIndex) && idx === focusIndex
            ? '<span class="badge badge-orange">focus</span>'
            : '';
        return `
            <tr>
                <td>${shortText(r && r.index)}</td>
                <td>${fmt(toNum(r && r.start, NaN), 0)}</td>
                <td>${fmt(toNum(r && r.length, NaN), 0)}</td>
                <td>${shortText(r && r.left_slot)}</td>
                <td>${shortText(r && r.right_slot)}</td>
                <td>${mark}</td>
            </tr>`;
    }).join("");

    return `
        <div style="color:var(--text-muted);margin-bottom:0.4rem;">showing ${rows.length}/${ranges.length} ranges</div>
        <table>
            <thead><tr><th>Idx</th><th>Start</th><th>Length</th><th>Left Slot</th><th>Right Slot</th><th>Mark</th></tr></thead>
            <tbody>${htmlRows}</tbody>
        </table>`;
}

function renderTensorSlots(runtimeSummary, focusSlotIdx) {
    const slots = safeList(runtimeSummary && runtimeSummary.tensor_slots);
    if (slots.length === 0) {
        return '<div style="color:var(--text-muted);">No tensor slot map in runtime summary.</div>';
    }

    const writable = slots.filter((s) => Number(s && s.writable_fwd) === 1 || Number(s && s.writable_bwd) === 1);
    const source = writable.length > 0 ? writable : slots;

    const maxRows = 30;
    let rows = source.slice(0, maxRows);
    if (Number.isFinite(focusSlotIdx) && focusSlotIdx >= 0) {
        const directIdx = source.findIndex((s) => Number(s && s.index) === focusSlotIdx);
        if (directIdx >= 0) {
            const start = Math.max(0, directIdx - 6);
            rows = source.slice(start, Math.min(source.length, start + maxRows));
        }
    }

    const htmlRows = rows.map((s) => {
        const idx = Number(s && s.index);
        const isFocus = Number.isFinite(focusSlotIdx) && idx === focusSlotIdx;
        return `
            <tr${isFocus ? ' style="background:rgba(245,158,11,0.12);"' : ""}>
                <td>${shortText(s && s.index)}</td>
                <td>${shortText(s && s.name)}</td>
                <td>${shortText(s && s.section)}</td>
                <td>${fmt(toNum(s && s.offset, NaN), 0)}</td>
                <td>${fmt(toNum(s && s.numel, NaN), 0)}</td>
                <td>${Number(s && s.writable_fwd) === 1 ? 'Y' : 'N'}</td>
                <td>${Number(s && s.writable_bwd) === 1 ? 'Y' : 'N'}</td>
            </tr>`;
    }).join("");

    return `
        <div style="color:var(--text-muted);margin-bottom:0.4rem;">showing ${rows.length}/${source.length} slots (${writable.length > 0 ? 'writable-only view' : 'all slots'})</div>
        <table>
            <thead><tr><th>Idx</th><th>Name</th><th>Section</th><th>Offset</th><th>Numel</th><th>W Fwd</th><th>W Bwd</th></tr></thead>
            <tbody>${htmlRows}</tbody>
        </table>`;
}

export function renderTrainingMemoryDiag(files) {
    const panel = document.getElementById("train-memory-canary");
    if (!panel) return;
    const root = document.getElementById("trainMemoryCanaryRoot");
    if (!root) return;

    clear(root);

    const memoryDiag = (files && files.memory_diagnostic) || {};
    const diag = (memoryDiag && memoryDiag.diagnostic) || {};
    const diagMeta = (memoryDiag && memoryDiag.meta) || {};
    const runtimeSummary = (files && files.generated_train_runtime_summary) || {};
    const layoutAudit = (files && files.layout_train_audit) || {};
    const layoutTrain = (files && files.layout_train) || {};
    const embedded = window.EMBEDDED_IR_DATA || {};
    const runDir = embedded && embedded.meta && typeof embedded.meta.run_dir === "string" && embedded.meta.run_dir.trim()
        ? embedded.meta.run_dir.trim()
        : "/tmp/v7_exp1";

    const hasAny = Object.keys(memoryDiag).length > 0 || Object.keys(runtimeSummary).length > 0 || Object.keys(layoutAudit).length > 0;
    if (!hasAny) {
        root.innerHTML = `
            <div class="parity-section">
                <h3><span class="badge badge-orange">No Memory Diagnostics</span> Missing training memory artifacts</h3>
                <p style="color:var(--text-muted);">Run strict CK training once to emit canary diagnostics and runtime memory summary.</p>
                <pre style="font-size:0.8rem;white-space:pre-wrap;">version/v7/scripts/cks-v7-run train --run ${runDir} --backend ck --train-strict --prompt "hello" --train-epochs 1 --train-seq-len 8 --train-total-tokens 64 --train-grad-accum 4</pre>
            </div>`;
        return;
    }

    const diagOk = diag && diag.ok === true;
    const auditOk = layoutAudit && layoutAudit.passed === true;
    const locator = buildCorruptionLocator(diag, diagMeta, runtimeSummary);

    const canaryFocus = locator.canary_index;
    const slotFocus = locator.left_slot && Number.isFinite(Number(locator.left_slot.index))
        ? Number(locator.left_slot.index)
        : null;

    const regionCount = safeList(layoutTrain && layoutTrain.regions).length;
    const totalBytes = toNum(layoutTrain && layoutTrain.total_bytes, NaN);
    const tensorSlots = safeList(runtimeSummary && runtimeSummary.tensor_slots).length;
    const canaryRanges = safeList(runtimeSummary && runtimeSummary.canary_ranges).length;
    const tailCanary = toNum(runtimeSummary && runtimeSummary.canary_tail_floats, NaN);

    root.innerHTML = `
        <div class="stats-grid">
            <div class="stat-card"><div class="stat-value">${boolBadge(diagOk)}</div><div class="stat-label">Memory Diagnostic</div></div>
            <div class="stat-card"><div class="stat-value">${boolBadge(auditOk)}</div><div class="stat-label">Layout Audit</div></div>
            <div class="stat-card"><div class="stat-value">${shortText(diag && diag.phase, "-")}</div><div class="stat-label">Last Phase</div></div>
            <div class="stat-card"><div class="stat-value">${Number.isFinite(toNum(diag && diag.rc, NaN)) ? fmt(toNum(diag.rc, NaN), 0) : '-'}</div><div class="stat-label">Diag RC</div></div>
            <div class="stat-card"><div class="stat-value">${Number.isFinite(totalBytes) ? fmt(totalBytes / (1024 * 1024), 1) + ' MB' : '-'}</div><div class="stat-label">Train Arena</div></div>
            <div class="stat-card"><div class="stat-value">${regionCount || '-'}</div><div class="stat-label">Layout Regions</div></div>
            <div class="stat-card"><div class="stat-value">${tensorSlots || '-'}</div><div class="stat-label">Tensor Slots</div></div>
            <div class="stat-card"><div class="stat-value">${canaryRanges || '-'}</div><div class="stat-label">Canary Ranges (+tail ${Number.isFinite(tailCanary) ? fmt(tailCanary, 0) : '-'})</div></div>
        </div>

        <div class="parity-section" style="margin-top:0.8rem;">
            <h3><span class="badge badge-orange">Locator</span> First Corruption Locator</h3>
            ${renderLocator(locator)}
        </div>

        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:0.8rem;">
            <div class="parity-section">
                <h3><span class="badge badge-orange">Diagnostic</span> Latest Memory Check</h3>
                <div style="color:var(--text-muted);margin-bottom:0.5rem;">phase/index: <code>${phaseLabel(diag)}</code></div>
                <pre style="font-size:0.78rem;white-space:pre-wrap;">${JSON.stringify(diag, null, 2)}</pre>
            </div>
            <div class="parity-section">
                <h3><span class="badge badge-blue">Audit</span> layout_train_audit.json</h3>
                ${renderChecksTable(layoutAudit)}
            </div>
        </div>

        <div class="parity-section" style="margin-top:0.8rem;">
            <h3><span class="badge badge-orange">Canary</span> Range Map</h3>
            ${renderCanaryTable(runtimeSummary, canaryFocus)}
        </div>

        <div class="parity-section" style="margin-top:0.8rem;">
            <h3><span class="badge badge-blue">Slots</span> Tensor Slot Map</h3>
            ${renderTensorSlots(runtimeSummary, slotFocus)}
        </div>

        <div class="parity-section" style="margin-top:0.8rem;">
            <h3><span class="badge badge-green">Runbook</span> Re-run Strict Memory Checks</h3>
            <pre style="font-size:0.8rem;white-space:pre-wrap;">version/v7/scripts/cks-v7-run train --run ${runDir} --backend ck --train-strict --prompt "hello" --train-epochs 1 --train-seq-len 8 --train-total-tokens 64 --train-grad-accum 4</pre>
            <pre style="font-size:0.8rem;white-space:pre-wrap;">python3 version/v7/tools/open_ir_visualizer.py --generate --run ${runDir} --html-only</pre>
        </div>
    `;
}
