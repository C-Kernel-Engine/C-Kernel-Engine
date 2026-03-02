import { clear, fmtExp, toNum } from './utils.js';

function getCheckpoints(files) {
    const cps = files && files.analysis_checkpoints && Array.isArray(files.analysis_checkpoints.checkpoints)
        ? files.analysis_checkpoints.checkpoints
        : [];
    return [...cps].sort((a, b) => toNum(a.step, 0) - toNum(b.step, 0));
}

function layerFromParam(name) {
    const m = String(name || '').match(/layer\.(\d+)\./);
    return m ? Number(m[1]) : null;
}

function kindFromParam(name) {
    const n = String(name || '').toLowerCase();
    if (n.includes('.wq') || n.includes('.wk') || n.includes('.wv') || n.includes('.wo') || n.includes('attn')) {
        return 'attention';
    }
    if (n.includes('.w1') || n.includes('.w2') || n.includes('.w3') || n.includes('mlp')) {
        return 'mlp';
    }
    if (n.includes('norm') || n.includes('gamma') || n.includes('ln')) {
        return 'norm';
    }
    return 'other';
}

function aggregateLayerGradients(checkpoint) {
    const gradients = checkpoint && checkpoint.gradients && typeof checkpoint.gradients === 'object'
        ? checkpoint.gradients
        : {};
    const rows = new Map();

    Object.entries(gradients).forEach(([param, payload]) => {
        const layer = layerFromParam(param);
        if (layer === null) return;
        const norm = toNum(payload && payload.norm, NaN);
        if (!Number.isFinite(norm)) return;
        if (!rows.has(layer)) {
            rows.set(layer, { layer, attention: 0, mlp: 0, norm: 0, other: 0, total: 0 });
        }
        const r = rows.get(layer);
        const kind = kindFromParam(param);
        r[kind] += norm;
        r.total += norm;
    });

    return Array.from(rows.values()).sort((a, b) => a.layer - b.layer);
}

function renderWaterfall(svgEl, rows) {
    if (!svgEl || !window.d3 || !Array.isArray(rows) || rows.length === 0) return;
    const d3 = window.d3;

    const width = Math.max(svgEl.clientWidth || 680, rows.length * 36 + 140);
    const height = 220;
    const margin = { top: 12, right: 16, bottom: 34, left: 58 };
    const kinds = ['attention', 'mlp', 'norm'];
    const colors = { attention: '#3b82f6', mlp: '#10b981', norm: '#8b5cf6' };

    const minPositive = d3.min(rows.flatMap((r) => kinds.map((k) => r[k]).filter((v) => v > 0))) || 1e-9;
    const maxValue = d3.max(rows.flatMap((r) => kinds.map((k) => r[k]))) || 1;

    const x = d3.scaleBand().domain(rows.map((r) => String(r.layer))).range([margin.left, width - margin.right]).padding(0.16);
    const xInner = d3.scaleBand().domain(kinds).range([0, x.bandwidth()]).padding(0.08);
    const y = d3.scaleLog().domain([Math.max(minPositive * 0.9, 1e-12), maxValue * 1.2]).range([height - margin.bottom, margin.top]);

    const svg = d3.select(svgEl);
    svg.selectAll('*').remove();
    svg.attr('viewBox', `0 0 ${width} ${height}`);

    svg.append('g')
        .attr('transform', `translate(0,${height - margin.bottom})`)
        .call(d3.axisBottom(x).tickValues(rows.map((r) => String(r.layer)).filter((_, i) => i % Math.ceil(rows.length / 14) === 0)))
        .call((g) => g.selectAll('text').attr('fill', '#9ca3af').attr('font-size', 10))
        .call((g) => g.selectAll('line,path').attr('stroke', '#4b5563'));

    svg.append('g')
        .attr('transform', `translate(${margin.left},0)`)
        .call(d3.axisLeft(y).ticks(4, '.1e'))
        .call((g) => g.selectAll('text').attr('fill', '#9ca3af').attr('font-size', 10))
        .call((g) => g.selectAll('line,path').attr('stroke', '#4b5563'));

    const layer = svg.append('g');
    rows.forEach((r) => {
        kinds.forEach((k) => {
            const v = toNum(r[k], 0);
            if (!Number.isFinite(v) || v <= 0) return;
            layer.append('rect')
                .attr('x', x(String(r.layer)) + xInner(k))
                .attr('y', y(v))
                .attr('width', xInner.bandwidth())
                .attr('height', Math.max(1, y(Math.max(minPositive, 1e-12)) - y(v)))
                .attr('rx', 1.5)
                .attr('fill', colors[k])
                .append('title')
                .text(`L${r.layer} ${k}: ${fmtExp(v, 2)}`);
        });
    });
}

function renderRatio(svgEl, rows) {
    if (!svgEl || !window.d3 || !Array.isArray(rows) || rows.length < 2) return;
    const d3 = window.d3;

    const ratioRows = rows.slice(0, -1).map((row, idx) => {
        const next = rows[idx + 1];
        const ratio = next.total > 0 ? row.total / next.total : NaN;
        return { layer: row.layer, ratio };
    }).filter((r) => Number.isFinite(r.ratio));

    if (ratioRows.length === 0) return;

    const width = Math.max(svgEl.clientWidth || 680, ratioRows.length * 26 + 120);
    const height = 180;
    const margin = { top: 12, right: 16, bottom: 30, left: 50 };

    const x = d3.scaleBand().domain(ratioRows.map((r) => String(r.layer))).range([margin.left, width - margin.right]).padding(0.15);
    const yMax = d3.max(ratioRows, (r) => r.ratio) || 1;
    const y = d3.scaleLinear().domain([0, Math.max(2.2, yMax * 1.1)]).range([height - margin.bottom, margin.top]);

    const color = (v) => (v > 2 ? '#ef4444' : (v < 0.5 ? '#fbbf24' : '#22c55e'));

    const svg = d3.select(svgEl);
    svg.selectAll('*').remove();
    svg.attr('viewBox', `0 0 ${width} ${height}`);

    svg.append('line')
        .attr('x1', margin.left)
        .attr('x2', width - margin.right)
        .attr('y1', y(1))
        .attr('y2', y(1))
        .attr('stroke', '#9ca3af')
        .attr('stroke-dasharray', '4,4');

    svg.append('g')
        .attr('transform', `translate(0,${height - margin.bottom})`)
        .call(d3.axisBottom(x).tickValues(ratioRows.map((r) => String(r.layer)).filter((_, i) => i % Math.ceil(ratioRows.length / 16) === 0)))
        .call((g) => g.selectAll('text').attr('fill', '#9ca3af').attr('font-size', 10))
        .call((g) => g.selectAll('line,path').attr('stroke', '#4b5563'));

    svg.append('g')
        .attr('transform', `translate(${margin.left},0)`)
        .call(d3.axisLeft(y).ticks(4))
        .call((g) => g.selectAll('text').attr('fill', '#9ca3af').attr('font-size', 10))
        .call((g) => g.selectAll('line,path').attr('stroke', '#4b5563'));

    svg.append('g')
        .selectAll('rect')
        .data(ratioRows)
        .enter()
        .append('rect')
        .attr('x', (d) => x(String(d.layer)))
        .attr('y', (d) => y(d.ratio))
        .attr('width', x.bandwidth())
        .attr('height', (d) => height - margin.bottom - y(d.ratio))
        .attr('fill', (d) => color(d.ratio))
        .attr('rx', 2)
        .append('title')
        .text((d) => `L${d.layer} ratio=${d.ratio.toFixed(3)}`);
}

function renderHeatmap(svgEl, checkpoints) {
    if (!svgEl || !window.d3 || !Array.isArray(checkpoints) || checkpoints.length === 0) return;
    const d3 = window.d3;

    const matrix = [];
    checkpoints.forEach((cp) => {
        const rows = aggregateLayerGradients(cp);
        for (let i = 0; i < rows.length - 1; i += 1) {
            const denom = rows[i + 1].total;
            if (!(denom > 0) || !(rows[i].total > 0)) continue;
            const ratio = rows[i].total / denom;
            const logRatio = Math.log10(ratio);
            matrix.push({
                step: toNum(cp.step, 0),
                layer: rows[i].layer,
                value: Math.max(-1, Math.min(1, logRatio)),
                ratio,
            });
        }
    });

    if (matrix.length === 0) return;

    const steps = Array.from(new Set(matrix.map((m) => m.step))).sort((a, b) => a - b);
    const layers = Array.from(new Set(matrix.map((m) => m.layer))).sort((a, b) => a - b);

    const width = Math.max(svgEl.clientWidth || 680, steps.length * 28 + 120);
    const height = Math.max(220, layers.length * 18 + 54);
    const margin = { top: 14, right: 16, bottom: 34, left: 54 };

    const x = d3.scaleBand().domain(steps.map(String)).range([margin.left, width - margin.right]).padding(0.05);
    const y = d3.scaleBand().domain(layers.map(String)).range([margin.top, height - margin.bottom]).padding(0.05);
    const c = d3.scaleSequential(d3.interpolateRdYlGn).domain([-1, 1]);

    const svg = d3.select(svgEl);
    svg.selectAll('*').remove();
    svg.attr('viewBox', `0 0 ${width} ${height}`);

    svg.append('g')
        .selectAll('rect')
        .data(matrix)
        .enter()
        .append('rect')
        .attr('x', (d) => x(String(d.step)))
        .attr('y', (d) => y(String(d.layer)))
        .attr('width', x.bandwidth())
        .attr('height', y.bandwidth())
        .attr('fill', (d) => c(d.value))
        .append('title')
        .text((d) => `step=${d.step}, L${d.layer}, ratio=${d.ratio.toFixed(3)}`);

    svg.append('g')
        .attr('transform', `translate(0,${height - margin.bottom})`)
        .call(d3.axisBottom(x).tickValues(steps.filter((_, i) => i % Math.ceil(steps.length / 10) === 0).map(String)))
        .call((g) => g.selectAll('text').attr('fill', '#9ca3af').attr('font-size', 10))
        .call((g) => g.selectAll('line,path').attr('stroke', '#4b5563'));

    svg.append('g')
        .attr('transform', `translate(${margin.left},0)`)
        .call(d3.axisLeft(y).tickValues(layers.filter((_, i) => i % Math.ceil(layers.length / 14) === 0).map(String)))
        .call((g) => g.selectAll('text').attr('fill', '#9ca3af').attr('font-size', 10))
        .call((g) => g.selectAll('line,path').attr('stroke', '#4b5563'));
}

export function renderGradientFlow(files) {
    const root = document.getElementById('trainGradFlowRoot');
    if (!root) return;
    clear(root);

    const checkpoints = getCheckpoints(files);
    if (checkpoints.length === 0) {
        const runCtx = getRunContext();
        const runRef = runCtx.runDir || '$RUN';
        root.innerHTML = `
            <div class="parity-section">
                <h3><span class="badge badge-orange">No Checkpoints</span> Gradient Flow Needs Analysis Snapshots</h3>
                <div style="color:var(--text-muted);margin-bottom:0.45rem;">
                    Missing artifact: <code>analysis_checkpoint_step_*.json</code>. These files drive Gradient Flow, Weights &amp; Activations, and Attention Inspector tabs.
                </div>
                <pre style="font-size:0.76rem;white-space:pre-wrap;">python3 version/v7/scripts/ck_run_v7.py train --run ${runRef} --analysis-checkpoints log
python3 version/v7/tools/open_ir_visualizer.py --generate --run ${runRef} --html-only</pre>
            </div>
        `;
        return;
    }

    root.innerHTML = `
        <div class="parity-section">
            <h3><span class="badge badge-orange">Gradient Flow</span> Are Gradients Reaching Every Layer?</h3>
            <div style="display:flex;gap:0.6rem;align-items:center;flex-wrap:wrap;margin:0.5rem 0 0.8rem;">
                <label class="step-label">Checkpoint</label>
                <select id="gradFlowStepSelect" class="param-select"></select>
                <span class="step-label">components: attention / mlp / norm</span>
            </div>
            <svg id="gradFlowWaterfall" style="width:100%;height:230px;"></svg>
        </div>
        <div class="parity-section" style="margin-top:0.8rem;">
            <h3><span class="badge badge-blue">Flow Ratio</span> grad[L] / grad[L+1]</h3>
            <svg id="gradFlowRatio" style="width:100%;height:190px;"></svg>
        </div>
        <div class="parity-section" style="margin-top:0.8rem;">
            <h3><span class="badge badge-green">Layer × Step</span> Gradient Propagation Heatmap</h3>
            <svg id="gradFlowHeatmap" style="width:100%;height:280px;"></svg>
        </div>
    `;

    const select = document.getElementById('gradFlowStepSelect');
    const steps = checkpoints.map((c) => toNum(c.step, 0));
    if (select) {
        select.innerHTML = checkpoints
            .map((cp, idx) => `<option value="${idx}">step ${toNum(cp.step, 0)}</option>`)
            .join('');
        select.value = String(checkpoints.length - 1);

        const draw = () => {
            const idx = Math.max(0, Math.min(checkpoints.length - 1, toNum(select.value, checkpoints.length - 1)));
            const rows = aggregateLayerGradients(checkpoints[idx]);
            renderWaterfall(document.getElementById('gradFlowWaterfall'), rows);
            renderRatio(document.getElementById('gradFlowRatio'), rows);
        };
        select.addEventListener('change', draw);
        draw();
    }

    renderHeatmap(document.getElementById('gradFlowHeatmap'), checkpoints);
}
