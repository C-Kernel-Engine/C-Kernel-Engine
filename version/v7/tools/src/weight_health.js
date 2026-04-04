import { clear, fmt, fmtExp, toNum } from './utils.js';

// ---------------------------------------------------------------------------
// Data access helpers
// ---------------------------------------------------------------------------

function getWeightHealth(files) {
    const wh = files && files.weight_health;
    return wh && typeof wh === 'object' ? wh : null;
}

function getRunContext() {
    const embedded = window.EMBEDDED_IR_DATA;
    if (!embedded || typeof embedded !== 'object') return {};
    const meta = embedded.meta || {};
    return { runDir: meta.run_dir || meta.model_dir || '' };
}

// ---------------------------------------------------------------------------
// Checkpoint delta health table
// ---------------------------------------------------------------------------

function severityBadge(flags) {
    if (!Array.isArray(flags) || flags.length === 0) {
        return '<span class="grad-healthy">healthy</span>';
    }
    const text = flags.join(', ');
    const hasCritical = flags.some((f) =>
        f.includes('NaN') || f.includes('Inf')
    );
    const hasWarning = flags.some((f) =>
        f.includes('no meaningful update') || f.includes('persistently near-zero')
    );
    if (hasCritical) return `<span class="grad-danger">${text}</span>`;
    if (hasWarning) return `<span class="grad-warning">${text}</span>`;
    return `<span class="grad-warning">${text}</span>`;
}

function renderDeltaTable(root, tensors) {
    if (!root) return;
    const names = Object.keys(tensors).sort();
    if (names.length === 0) {
        root.innerHTML = '<div style="color:var(--text-muted);">No tensor delta data available.</div>';
        return;
    }

    const rows = names.map((name) => {
        const t = tensors[name];
        const delta = t.delta || {};
        const init = t.init_stats || {};
        const latest = t.latest_stats || {};
        const flags = t.flags || [];
        const initNorm = toNum(init.frobenius_norm, NaN);
        const normDelta = toNum(delta.norm_delta, NaN);
        const relMovement = Number.isFinite(initNorm) && initNorm > 0 && Number.isFinite(normDelta)
            ? normDelta / initNorm
            : NaN;
        return { name, delta, init, latest, flags, normDelta, relMovement };
    });

    // Sort: flagged first, then by relative movement ascending (least movement = most suspicious).
    rows.sort((a, b) => {
        const af = a.flags.length > 0 ? 0 : 1;
        const bf = b.flags.length > 0 ? 0 : 1;
        if (af !== bf) return af - bf;
        const am = Number.isFinite(a.relMovement) ? a.relMovement : Infinity;
        const bm = Number.isFinite(b.relMovement) ? b.relMovement : Infinity;
        return am - bm;
    });

    const html = `
        <div style="max-height:500px;overflow:auto;">
        <table>
            <thead>
                <tr>
                    <th>Tensor</th>
                    <th>Shape</th>
                    <th>Norm Δ</th>
                    <th>Rel. Movement</th>
                    <th>Max Δ</th>
                    <th>Unchanged %</th>
                    <th>Zeros</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                ${rows.map((r) => `
                    <tr>
                        <td style="font-family:'JetBrains Mono',monospace;font-size:0.78rem;">${r.name}</td>
                        <td style="font-size:0.78rem;">${(r.init.count || r.latest.count || '-')}</td>
                        <td>${fmtExp(r.normDelta, 2)}</td>
                        <td>${Number.isFinite(r.relMovement) ? fmtExp(r.relMovement, 2) : '-'}</td>
                        <td>${fmtExp(r.delta.max_delta, 2)}</td>
                        <td>${Number.isFinite(r.delta.unchanged_frac) ? fmt(r.delta.unchanged_frac * 100, 1) + '%' : '-'}</td>
                        <td>${r.latest.zero !== undefined ? r.latest.zero : '-'}</td>
                        <td>${severityBadge(r.flags)}</td>
                    </tr>
                `).join('')}
            </tbody>
        </table>
        </div>
    `;
    root.innerHTML = html;
}

// ---------------------------------------------------------------------------
// Gradient reachability heatmap
// ---------------------------------------------------------------------------

function renderGradReachability(svgEl, tensors) {
    if (!svgEl || !window.d3) return;
    const d3 = window.d3;

    // Collect params that have grad_reachability data.
    const params = [];
    Object.entries(tensors).forEach(([name, t]) => {
        const gr = t.grad_reachability;
        if (!gr) return;
        const layer = (() => {
            const m = String(name).match(/layer\.(\d+)\./);
            return m ? Number(m[1]) : null;
        })();
        params.push({
            name,
            layer,
            meanNorm: toNum(gr.mean_norm, 0),
            latestNorm: toNum(gr.latest_norm, 0),
            nearZeroRatio: toNum(gr.near_zero_ratio, 0),
        });
    });

    if (params.length === 0) {
        svgEl.parentElement.innerHTML = '<div style="color:var(--text-muted);">No gradient reachability data. Run training with <code>--analysis-checkpoints log</code> to collect per-parameter gradient norms.</div>';
        return;
    }

    params.sort((a, b) => {
        if (a.layer !== null && b.layer !== null) return a.layer - b.layer;
        if (a.layer !== null) return -1;
        if (b.layer !== null) return 1;
        return a.name.localeCompare(b.name);
    });

    const width = Math.max(svgEl.clientWidth || 680, 680);
    const barHeight = 18;
    const maxLabelWidth = 200;
    const height = Math.max(params.length * barHeight + 60, 100);
    const margin = { top: 20, right: 20, bottom: 30, left: maxLabelWidth };

    const maxNorm = d3.max(params, (p) => p.meanNorm) || 1;
    const x = d3.scaleLog()
        .domain([Math.max(1e-12, d3.min(params, (p) => p.meanNorm > 0 ? p.meanNorm : maxNorm) * 0.5), maxNorm * 1.5])
        .range([margin.left, width - margin.right]);
    const y = d3.scaleBand()
        .domain(params.map((_, i) => String(i)))
        .range([margin.top, height - margin.bottom])
        .padding(0.12);

    const color = (nzr) => {
        if (nzr >= 0.8) return '#ef4444';
        if (nzr >= 0.5) return '#f59e0b';
        return '#10b981';
    };

    const svg = d3.select(svgEl);
    svg.selectAll('*').remove();
    svg.attr('viewBox', `0 0 ${width} ${height}`);

    // Bars for mean grad norm, colored by near-zero ratio.
    svg.append('g')
        .selectAll('rect')
        .data(params)
        .enter()
        .append('rect')
        .attr('x', margin.left)
        .attr('y', (_, i) => y(String(i)))
        .attr('width', (d) => Math.max(2, x(Math.max(d.meanNorm, 1e-12)) - margin.left))
        .attr('height', y.bandwidth())
        .attr('fill', (d) => color(d.nearZeroRatio))
        .attr('rx', 2)
        .append('title')
        .text((d) => `${d.name}\nmean grad norm: ${d.meanNorm.toExponential(2)}\nnear-zero ratio: ${(d.nearZeroRatio * 100).toFixed(1)}%`);

    // Parameter name labels.
    svg.append('g')
        .selectAll('text')
        .data(params)
        .enter()
        .append('text')
        .attr('x', margin.left - 6)
        .attr('y', (_, i) => y(String(i)) + y.bandwidth() / 2)
        .attr('dy', '0.35em')
        .attr('text-anchor', 'end')
        .attr('fill', '#9ca3af')
        .attr('font-size', Math.min(11, y.bandwidth() - 2))
        .attr('font-family', "'JetBrains Mono', monospace")
        .text((d) => {
            const short = d.name.length > 28 ? '…' + d.name.slice(-27) : d.name;
            return short;
        });

    // X axis.
    svg.append('g')
        .attr('transform', `translate(0,${height - margin.bottom})`)
        .call(d3.axisBottom(x).ticks(4, '.0e'))
        .call((g) => g.selectAll('text').attr('fill', '#9ca3af').attr('font-size', 10))
        .call((g) => g.selectAll('line,path').attr('stroke', '#4b5563'));

    // Legend.
    const legend = svg.append('g').attr('transform', `translate(${margin.left + 8},${margin.top - 8})`);
    const items = [
        { label: 'healthy (<50% near-zero)', color: '#10b981' },
        { label: 'warning (50-80%)', color: '#f59e0b' },
        { label: 'stale (≥80% near-zero)', color: '#ef4444' },
    ];
    items.forEach((item, idx) => {
        const g = legend.append('g').attr('transform', `translate(${idx * 180},0)`);
        g.append('rect').attr('width', 10).attr('height', 10).attr('rx', 2).attr('fill', item.color);
        g.append('text').attr('x', 14).attr('y', 9).attr('fill', '#9ca3af').attr('font-size', 10).text(item.label);
    });
}

// ---------------------------------------------------------------------------
// Summary cards
// ---------------------------------------------------------------------------

function renderSummary(root, summary) {
    if (!root || !summary) return;

    const total = toNum(summary.total_tensors, 0);
    const analysed = toNum(summary.tensors_with_values, 0);
    const flagged = toNum(summary.flagged_tensors, 0);
    const params = toNum(summary.total_parameters, 0);
    const gradTensors = toNum(summary.grad_reachability_tensors, 0);
    const flags = summary.flag_counts || {};

    const flagList = Object.entries(flags)
        .sort((a, b) => b[1] - a[1])
        .map(([k, v]) => `<span style="margin-right:0.8rem;">${k}: <strong>${v}</strong></span>`)
        .join('');

    root.innerHTML = `
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:0.6rem;margin-bottom:1rem;">
            <div class="alert-item info">
                <div style="font-size:1.4rem;font-weight:700;">${total}</div>
                <div style="font-size:0.78rem;color:var(--text-muted);">tensors</div>
            </div>
            <div class="alert-item info">
                <div style="font-size:1.4rem;font-weight:700;">${params.toLocaleString()}</div>
                <div style="font-size:0.78rem;color:var(--text-muted);">parameters</div>
            </div>
            <div class="alert-item info">
                <div style="font-size:1.4rem;font-weight:700;">${analysed}</div>
                <div style="font-size:0.78rem;color:var(--text-muted);">analysed</div>
            </div>
            <div class="alert-item ${flagged > 0 ? 'warning' : 'info'}">
                <div style="font-size:1.4rem;font-weight:700;">${flagged}</div>
                <div style="font-size:0.78rem;color:var(--text-muted);">flagged</div>
            </div>
            <div class="alert-item info">
                <div style="font-size:1.4rem;font-weight:700;">${gradTensors}</div>
                <div style="font-size:0.78rem;color:var(--text-muted);">grad tracked</div>
            </div>
        </div>
        ${flagList ? `<div style="font-size:0.82rem;color:var(--text-secondary);margin-bottom:0.6rem;">${flagList}</div>` : ''}
    `;
}

// ---------------------------------------------------------------------------
// Stale-parameter flag list
// ---------------------------------------------------------------------------

function renderFlaggedList(root, tensors) {
    if (!root) return;

    const flagged = Object.entries(tensors)
        .filter(([, t]) => Array.isArray(t.flags) && t.flags.length > 0)
        .map(([name, t]) => ({ name, flags: t.flags, delta: t.delta || {}, grad: t.grad_reachability }))
        .sort((a, b) => b.flags.length - a.flags.length);

    if (flagged.length === 0) {
        root.innerHTML = '<div style="color:var(--text-muted);">No parameters flagged. All tensors show meaningful updates.</div>';
        return;
    }

    root.innerHTML = flagged.map((item) => {
        const gradNote = item.grad
            ? ` | mean grad norm: ${fmtExp(item.grad.mean_norm, 2)}, near-zero: ${fmt(item.grad.near_zero_ratio * 100, 1)}%`
            : '';
        const deltaNote = item.delta.unchanged_frac !== undefined
            ? ` | unchanged: ${fmt(item.delta.unchanged_frac * 100, 1)}%`
            : '';
        return `
            <div class="alert-item ${item.flags.some((f) => f.includes('NaN') || f.includes('Inf')) ? 'critical' : 'warning'}" style="margin-bottom:0.4rem;">
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.82rem;font-weight:600;">${item.name}</div>
                <div style="font-size:0.78rem;color:var(--text-secondary);margin-top:2px;">
                    ${item.flags.map((f) => `<span class="${f.includes('NaN') || f.includes('Inf') ? 'grad-danger' : 'grad-warning'}" style="margin-right:6px;">${f}</span>`).join('')}
                </div>
                <div style="font-size:0.74rem;color:var(--text-muted);margin-top:2px;">
                    norm Δ: ${fmtExp(item.delta.norm_delta, 2)}${deltaNote}${gradNote}
                </div>
            </div>
        `;
    }).join('');
}

// ---------------------------------------------------------------------------
// Main render entrypoint
// ---------------------------------------------------------------------------

export function renderWeightHealth(files) {
    const root = document.getElementById('trainWeightHealthRoot');
    if (!root) return;
    clear(root);

    const wh = getWeightHealth(files);
    if (!wh) {
        const runCtx = getRunContext();
        const runRef = runCtx.runDir || '$RUN';
        root.innerHTML = `
            <div class="parity-section">
                <h3><span class="badge badge-orange">No Data</span> Weight Health Needs a Probe Run</h3>
                <div style="color:var(--text-muted);margin-bottom:0.45rem;">
                    Missing artifact: <code>weight_health_latest.json</code>. Run the weight health probe to generate it.
                </div>
                <pre style="font-size:0.76rem;white-space:pre-wrap;">python3 version/v7/scripts/weight_health_probe_v7.py --run ${runRef}
python3 version/v7/tools/open_ir_visualizer.py --generate --run ${runRef} --html-only</pre>
            </div>
        `;
        return;
    }

    const tensors = wh.tensors || {};
    const summary = wh.summary || {};

    root.innerHTML = `
        <div class="parity-section">
            <h3><span class="badge badge-orange">Overview</span> Is Every Parameter Learning?</h3>
            <div id="whSummaryRoot"></div>
        </div>
        <div class="parity-section" style="margin-top:0.8rem;">
            <h3><span class="badge badge-blue">Checkpoint Delta</span> Init → Latest Per-Tensor Movement</h3>
            <div style="font-size:0.78rem;color:var(--text-muted);margin-bottom:0.5rem;">
                Comparing <code>${wh.init_checkpoint || '?'}</code> → <code>${wh.latest_checkpoint || '?'}</code>
            </div>
            <div id="whDeltaTableRoot"></div>
        </div>
        <div class="parity-section" style="margin-top:0.8rem;">
            <h3><span class="badge badge-green">Gradient Reachability</span> Mean Grad Norm per Parameter</h3>
            <svg id="whGradReachSvg" style="width:100%;min-height:200px;"></svg>
        </div>
        <div class="parity-section" style="margin-top:0.8rem;">
            <h3><span class="badge badge-orange">Flagged Parameters</span> Tensors Needing Attention</h3>
            <div id="whFlaggedRoot"></div>
        </div>
    `;

    renderSummary(document.getElementById('whSummaryRoot'), summary);
    renderDeltaTable(document.getElementById('whDeltaTableRoot'), tensors);
    renderGradReachability(document.getElementById('whGradReachSvg'), tensors);
    renderFlaggedList(document.getElementById('whFlaggedRoot'), tensors);
}
