import { clear, toNum } from './utils.js';

function getCheckpoints(files) {
    const cps = files && files.analysis_checkpoints && Array.isArray(files.analysis_checkpoints.checkpoints)
        ? files.analysis_checkpoints.checkpoints
        : [];
    return [...cps].sort((a, b) => toNum(a.step, 0) - toNum(b.step, 0));
}

function flattenGrid(grid) {
    const out = [];
    if (!Array.isArray(grid)) return out;
    grid.forEach((row) => {
        if (Array.isArray(row)) row.forEach((v) => out.push(toNum(v, 0)));
    });
    return out;
}

function cosineSimilarity(a, b) {
    if (!Array.isArray(a) || !Array.isArray(b) || a.length === 0 || b.length === 0 || a.length !== b.length) return NaN;
    let dot = 0;
    let na = 0;
    let nb = 0;
    for (let i = 0; i < a.length; i += 1) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if (na <= 0 || nb <= 0) return NaN;
    return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

function drawHeatmap(svgEl, grid) {
    if (!svgEl || !window.d3 || !Array.isArray(grid) || grid.length === 0) return;
    const d3 = window.d3;
    const rows = grid.length;
    const cols = Array.isArray(grid[0]) ? grid[0].length : 0;
    if (cols === 0) return;

    const width = Math.max(svgEl.clientWidth || 420, 380);
    const height = Math.max(svgEl.clientHeight || 320, 280);
    const margin = { top: 12, right: 12, bottom: 28, left: 34 };

    const vals = flattenGrid(grid);
    const min = d3.min(vals) ?? 0;
    const max = d3.max(vals) ?? 1;

    const x = d3.scaleBand().domain(d3.range(cols).map(String)).range([margin.left, width - margin.right]).padding(0);
    const y = d3.scaleBand().domain(d3.range(rows).map(String)).range([margin.top, height - margin.bottom]).padding(0);
    const c = d3.scaleSequential(d3.interpolateViridis).domain([min, max]);

    const svg = d3.select(svgEl);
    svg.selectAll('*').remove();
    svg.attr('viewBox', `0 0 ${width} ${height}`);

    const cells = [];
    for (let r = 0; r < rows; r += 1) {
        for (let k = 0; k < cols; k += 1) {
            cells.push({ r, k, v: toNum(grid[r][k], 0) });
        }
    }

    svg.append('g')
        .selectAll('rect')
        .data(cells)
        .enter()
        .append('rect')
        .attr('x', (d) => x(String(d.k)))
        .attr('y', (d) => y(String(d.r)))
        .attr('width', x.bandwidth())
        .attr('height', y.bandwidth())
        .attr('fill', (d) => c(d.v))
        .append('title')
        .text((d) => `q=${d.r} k=${d.k} v=${d.v.toExponential(3)}`);
}

function drawEntropyTimeline(svgEl, checkpoints, layerName, headIndex) {
    if (!svgEl || !window.d3 || !Array.isArray(checkpoints) || checkpoints.length === 0) return;
    const d3 = window.d3;

    const rows = checkpoints.map((cp) => {
        const attnLayer = cp.attention && cp.attention[layerName];
        const entropy = Array.isArray(attnLayer && attnLayer.entropy_per_head) ? attnLayer.entropy_per_head : [];
        return {
            step: toNum(cp.step, 0),
            selected: toNum(entropy[headIndex], NaN),
            min: entropy.length ? Math.min(...entropy.map((v) => toNum(v, 0))) : NaN,
            max: entropy.length ? Math.max(...entropy.map((v) => toNum(v, 0))) : NaN,
        };
    }).filter((r) => Number.isFinite(r.step));

    if (rows.length === 0) return;

    const width = Math.max(svgEl.clientWidth || 640, 460);
    const height = 220;
    const margin = { top: 12, right: 12, bottom: 30, left: 44 };

    const yVals = rows.flatMap((r) => [r.min, r.max, r.selected]).filter(Number.isFinite);
    if (yVals.length === 0) return;

    const x = d3.scaleLinear().domain(d3.extent(rows, (d) => d.step)).range([margin.left, width - margin.right]);
    const y = d3.scaleLinear().domain([Math.min(...yVals), Math.max(...yVals)]).nice().range([height - margin.bottom, margin.top]);

    const area = d3.area()
        .x((d) => x(d.step))
        .y0((d) => y(d.min))
        .y1((d) => y(d.max))
        .curve(d3.curveMonotoneX);

    const line = d3.line()
        .x((d) => x(d.step))
        .y((d) => y(d.selected))
        .curve(d3.curveMonotoneX);

    const svg = d3.select(svgEl);
    svg.selectAll('*').remove();
    svg.attr('viewBox', `0 0 ${width} ${height}`);

    svg.append('path').datum(rows).attr('fill', 'rgba(59,130,246,0.22)').attr('d', area);
    svg.append('path').datum(rows.filter((r) => Number.isFinite(r.selected))).attr('fill', 'none').attr('stroke', '#f59e0b').attr('stroke-width', 2).attr('d', line);

    svg.append('g')
        .attr('transform', `translate(0,${height - margin.bottom})`)
        .call(d3.axisBottom(x).ticks(6))
        .call((g) => g.selectAll('text').attr('fill', '#9ca3af').attr('font-size', 10))
        .call((g) => g.selectAll('line,path').attr('stroke', '#4b5563'));

    svg.append('g')
        .attr('transform', `translate(${margin.left},0)`)
        .call(d3.axisLeft(y).ticks(5))
        .call((g) => g.selectAll('text').attr('fill', '#9ca3af').attr('font-size', 10))
        .call((g) => g.selectAll('line,path').attr('stroke', '#4b5563'));
}

function renderRedundancy(root, headGrids) {
    if (!root || !window.d3 || !Array.isArray(headGrids) || headGrids.length === 0) {
        if (root) root.innerHTML = '<div style="color:var(--text-muted);">No head grids available.</div>';
        return;
    }
    const d3 = window.d3;
    root.innerHTML = '<svg id="attentionRedundancySvg" style="width:100%;height:280px;"></svg>';
    const svgEl = document.getElementById('attentionRedundancySvg');
    if (!svgEl) return;

    const vectors = headGrids.map((g) => flattenGrid(g));
    const n = vectors.length;
    const cells = [];
    for (let i = 0; i < n; i += 1) {
        for (let j = 0; j < n; j += 1) {
            cells.push({ i, j, v: cosineSimilarity(vectors[i], vectors[j]) });
        }
    }

    const width = Math.max(svgEl.clientWidth || 420, 320);
    const height = 280;
    const margin = { top: 14, right: 16, bottom: 32, left: 34 };
    const x = d3.scaleBand().domain(d3.range(n).map(String)).range([margin.left, width - margin.right]).padding(0.03);
    const y = d3.scaleBand().domain(d3.range(n).map(String)).range([margin.top, height - margin.bottom]).padding(0.03);
    const c = d3.scaleSequential(d3.interpolateRdYlGn).domain([0.5, 1]);

    const svg = d3.select(svgEl);
    svg.selectAll('*').remove();
    svg.attr('viewBox', `0 0 ${width} ${height}`);

    svg.append('g')
        .selectAll('rect')
        .data(cells)
        .enter()
        .append('rect')
        .attr('x', (d) => x(String(d.j)))
        .attr('y', (d) => y(String(d.i)))
        .attr('width', x.bandwidth())
        .attr('height', y.bandwidth())
        .attr('fill', (d) => c(Number.isFinite(d.v) ? d.v : 0.5))
        .append('title')
        .text((d) => `head ${d.i} vs ${d.j}: ${Number.isFinite(d.v) ? d.v.toFixed(4) : 'NA'}`);
}

export function renderAttentionInspector(files) {
    const root = document.getElementById('trainAttentionRoot');
    if (!root) return;
    clear(root);

    const checkpoints = getCheckpoints(files);
    if (checkpoints.length === 0) {
        const runCtx = getRunContext();
        const runRef = runCtx.runDir || '$RUN';
        root.innerHTML = `
            <div class="parity-section">
                <h3><span class="badge badge-orange">No Checkpoints</span> Attention Inspector Needs Analysis Snapshots</h3>
                <div style="color:var(--text-muted);margin-bottom:0.45rem;">
                    Missing artifact: <code>analysis_checkpoint_step_*.json</code> with attention blocks per step/head.
                </div>
                <pre style="font-size:0.76rem;white-space:pre-wrap;">python3 version/v7/scripts/ck_run_v7.py train --run ${runRef} --analysis-checkpoints log
python3 version/v7/tools/open_ir_visualizer.py --generate --run ${runRef} --html-only</pre>
            </div>
        `;
        return;
    }

    const layers = Array.from(new Set(checkpoints.flatMap((cp) => Object.keys((cp && cp.attention) || {})))).sort();
    if (layers.length === 0) {
        root.innerHTML = '<p style="color:var(--text-muted);">No attention section found in analysis checkpoints.</p>';
        return;
    }

    root.innerHTML = `
        <div class="parity-section">
            <h3><span class="badge badge-orange">Attention</span> What Is It Attending To?</h3>
            <div style="display:flex;gap:0.6rem;align-items:center;flex-wrap:wrap;margin-bottom:0.7rem;">
                <label class="step-label">Step</label>
                <select id="attnStepSelect" class="param-select"></select>
                <label class="step-label">Layer</label>
                <select id="attnLayerSelect" class="param-select"></select>
                <label class="step-label">Head</label>
                <select id="attnHeadSelect" class="param-select"></select>
            </div>
            <svg id="attnMainHeatmap" style="width:100%;height:330px;"></svg>
            <div id="attnMiniGrid" class="mini-head-grid" style="margin-top:0.8rem;"></div>
        </div>
        <div class="parity-section" style="margin-top:0.8rem;">
            <h3><span class="badge badge-blue">Entropy</span> Timeline</h3>
            <svg id="attnEntropyTimeline" style="width:100%;height:230px;"></svg>
        </div>
        <div class="parity-section" style="margin-top:0.8rem;">
            <h3><span class="badge badge-green">Head Redundancy</span> Cosine Similarity</h3>
            <div id="attnRedundancyRoot"></div>
        </div>
    `;

    const stepSel = document.getElementById('attnStepSelect');
    const layerSel = document.getElementById('attnLayerSelect');
    const headSel = document.getElementById('attnHeadSelect');
    const mainSvg = document.getElementById('attnMainHeatmap');
    const miniRoot = document.getElementById('attnMiniGrid');

    if (!stepSel || !layerSel || !headSel || !mainSvg || !miniRoot) return;

    stepSel.innerHTML = checkpoints.map((cp, idx) => `<option value="${idx}">step ${toNum(cp.step, 0)}</option>`).join('');
    layerSel.innerHTML = layers.map((l) => `<option value="${l}">${l}</option>`).join('');
    stepSel.value = String(checkpoints.length - 1);
    layerSel.value = layers[0];

    function drawMiniGrid(headGrids) {
        miniRoot.innerHTML = '';
        headGrids.forEach((grid, idx) => {
            const host = document.createElement('div');
            host.className = `mini-heatmap${toNum(headSel.value, 0) === idx ? ' selected' : ''}`;
            host.style.padding = '4px';
            host.style.borderRadius = '6px';
            host.style.background = 'rgba(255,255,255,0.02)';
            host.innerHTML = `<div style="font-size:11px;color:var(--text-muted);margin-bottom:4px;">H${idx}</div><svg style="width:100%;height:90px;"></svg>`;
            const svg = host.querySelector('svg');
            drawHeatmap(svg, grid);
            host.addEventListener('click', () => {
                headSel.value = String(idx);
                draw();
            });
            miniRoot.appendChild(host);
        });
    }

    function syncHeads() {
        const cp = checkpoints[toNum(stepSel.value, checkpoints.length - 1)] || {};
        const layer = layerSel.value;
        const entry = cp.attention && cp.attention[layer] ? cp.attention[layer] : {};
        const grids = Array.isArray(entry.sampled_qk_grid_32x32) ? entry.sampled_qk_grid_32x32 : [];
        const headCount = grids.length;
        const prev = toNum(headSel.value, 0);
        headSel.innerHTML = Array.from({ length: headCount }, (_, i) => `<option value="${i}">head ${i}</option>`).join('');
        headSel.value = String(Math.max(0, Math.min(headCount - 1, prev)));
    }

    function draw() {
        const cp = checkpoints[toNum(stepSel.value, checkpoints.length - 1)] || {};
        const layer = layerSel.value;
        const entry = cp.attention && cp.attention[layer] ? cp.attention[layer] : {};
        const grids = Array.isArray(entry.sampled_qk_grid_32x32) ? entry.sampled_qk_grid_32x32 : [];
        const headIndex = toNum(headSel.value, 0);

        if (grids[headIndex]) {
            drawHeatmap(mainSvg, grids[headIndex]);
        }
        drawMiniGrid(grids);
        drawEntropyTimeline(document.getElementById('attnEntropyTimeline'), checkpoints, layer, headIndex);
        renderRedundancy(document.getElementById('attnRedundancyRoot'), grids);
    }

    stepSel.addEventListener('change', () => { syncHeads(); draw(); });
    layerSel.addEventListener('change', () => { syncHeads(); draw(); });
    headSel.addEventListener('change', draw);

    syncHeads();
    draw();
}
