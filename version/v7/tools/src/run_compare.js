import { clear, fmt, fmtExp, parseLossSteps, toNum } from './utils.js';

function parseBaselineJson(obj) {
    if (!obj || typeof obj !== 'object') return { steps: [] };
    if (Array.isArray(obj.steps)) return { steps: obj.steps };
    if (obj.training_loss_curve && Array.isArray(obj.training_loss_curve.steps)) {
        return { steps: obj.training_loss_curve.steps };
    }
    return { steps: [] };
}

function drawOverlay(svgEl, primary, baseline, key, colorPrimary, colorBaseline) {
    if (!svgEl || !window.d3) return;
    const d3 = window.d3;
    const pRows = primary.filter((r) => Number.isFinite(r.step) && Number.isFinite(r[key]));
    const bRows = baseline.filter((r) => Number.isFinite(r.step) && Number.isFinite(r[key]));
    if (pRows.length === 0 && bRows.length === 0) return;

    const width = svgEl.clientWidth || 520;
    const height = 210;
    const margin = { top: 12, right: 12, bottom: 28, left: 44 };

    const all = [...pRows, ...bRows];
    const x = d3.scaleLinear().domain(d3.extent(all, (d) => d.step)).range([margin.left, width - margin.right]);
    const y = d3.scaleLinear().domain(d3.extent(all, (d) => d[key])).nice().range([height - margin.bottom, margin.top]);

    const line = d3.line().x((d) => x(d.step)).y((d) => y(d[key])).curve(d3.curveMonotoneX);

    const svg = d3.select(svgEl);
    svg.selectAll('*').remove();
    svg.attr('viewBox', `0 0 ${width} ${height}`);

    if (pRows.length) {
        svg.append('path').datum(pRows).attr('fill', 'none').attr('stroke', colorPrimary).attr('stroke-width', 2).attr('d', line);
    }
    if (bRows.length) {
        svg.append('path').datum(bRows).attr('fill', 'none').attr('stroke', colorBaseline).attr('stroke-width', 2).attr('stroke-dasharray', '6,4').attr('d', line);
    }

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

export function renderRunCompare(files) {
    const root = document.getElementById('trainCompareRoot');
    if (!root) return;
    clear(root);

    const primary = parseLossSteps(files.training_loss_curve);

    root.innerHTML = `
        <div class="parity-section">
            <h3><span class="badge badge-orange">Run Compare</span> How Does This Compare?</h3>
            <div id="trainCompareDrop" class="drop-zone">
                Drop baseline JSON here (training_loss_curve.json or train_e2e_latest.json),
                or <input id="trainCompareFile" type="file" accept=".json" style="margin-left:6px;" />
            </div>
            <div id="trainCompareMeta" style="margin-top:0.55rem;color:var(--text-muted);"></div>
        </div>
        <div class="parity-section" style="margin-top:0.8rem;">
            <h3><span class="badge badge-blue">Loss Overlay</span> current vs baseline</h3>
            <svg id="trainCompareLoss" style="width:100%;height:220px;"></svg>
        </div>
        <div class="parity-section" style="margin-top:0.8rem;">
            <h3><span class="badge badge-green">Grad Norm Overlay</span> current vs baseline</h3>
            <svg id="trainCompareGrad" style="width:100%;height:220px;"></svg>
        </div>
        <div class="parity-section" style="margin-top:0.8rem;">
            <h3><span class="badge badge-orange">Pointwise Delta</span> nearest-step comparison</h3>
            <div id="trainCompareDelta"></div>
        </div>
    `;

    const drop = document.getElementById('trainCompareDrop');
    const fileInput = document.getElementById('trainCompareFile');
    const meta = document.getElementById('trainCompareMeta');

    let baselineSteps = [];

    function render() {
        drawOverlay(document.getElementById('trainCompareLoss'), primary, baselineSteps, 'loss_ck', '#f59e0b', '#60a5fa');
        drawOverlay(document.getElementById('trainCompareGrad'), primary, baselineSteps, 'grad_norm', '#10b981', '#a78bfa');

        const deltaEl = document.getElementById('trainCompareDelta');
        if (!deltaEl) return;
        if (baselineSteps.length === 0 || primary.length === 0) {
            deltaEl.innerHTML = '<div style="color:var(--text-muted);">Load a baseline run to see deltas.</div>';
            return;
        }

        const paired = primary.slice(0, Math.min(primary.length, baselineSteps.length)).map((p, i) => {
            const b = baselineSteps[i];
            return {
                step: p.step,
                loss_delta: toNum(p.loss_ck, NaN) - toNum(b.loss_ck, NaN),
                grad_delta: toNum(p.grad_norm, NaN) - toNum(b.grad_norm, NaN),
            };
        }).filter((r) => Number.isFinite(r.loss_delta) || Number.isFinite(r.grad_delta));

        const latest = paired[paired.length - 1] || null;
        const maxAbsLoss = paired.reduce((m, r) => Math.max(m, Math.abs(toNum(r.loss_delta, 0))), 0);
        const maxAbsGrad = paired.reduce((m, r) => Math.max(m, Math.abs(toNum(r.grad_delta, 0))), 0);

        deltaEl.innerHTML = `
            <div class="stats-grid">
                <div class="stat-card"><div class="stat-value">${latest ? fmtExp(latest.loss_delta, 2) : '-'}</div><div class="stat-label">Latest Loss Delta</div></div>
                <div class="stat-card"><div class="stat-value">${latest ? fmtExp(latest.grad_delta, 2) : '-'}</div><div class="stat-label">Latest Grad Delta</div></div>
                <div class="stat-card"><div class="stat-value">${fmtExp(maxAbsLoss, 2)}</div><div class="stat-label">Max |Loss Delta|</div></div>
                <div class="stat-card"><div class="stat-value">${fmtExp(maxAbsGrad, 2)}</div><div class="stat-label">Max |Grad Delta|</div></div>
            </div>
        `;
    }

    function loadJsonText(text, sourceLabel) {
        try {
            const parsed = JSON.parse(text);
            baselineSteps = parseLossSteps(parseBaselineJson(parsed));
            if (meta) {
                meta.textContent = baselineSteps.length
                    ? `Loaded baseline (${sourceLabel}) with ${baselineSteps.length} points.`
                    : `Loaded ${sourceLabel}, but no step series found.`;
            }
            render();
        } catch (err) {
            if (meta) meta.textContent = `Failed to parse baseline JSON: ${err}`;
        }
    }

    if (fileInput) {
        fileInput.addEventListener('change', (ev) => {
            const file = ev.target && ev.target.files && ev.target.files[0];
            if (!file) return;
            file.text().then((txt) => loadJsonText(txt, file.name));
        });
    }

    if (drop) {
        drop.addEventListener('dragover', (ev) => {
            ev.preventDefault();
            drop.classList.add('drag-over');
        });
        drop.addEventListener('dragleave', () => drop.classList.remove('drag-over'));
        drop.addEventListener('drop', (ev) => {
            ev.preventDefault();
            drop.classList.remove('drag-over');
            const file = ev.dataTransfer && ev.dataTransfer.files && ev.dataTransfer.files[0];
            if (!file) return;
            file.text().then((txt) => loadJsonText(txt, file.name));
        });
    }

    render();
}
