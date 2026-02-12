import { clear, fmt, fmtExp, parseLossSteps, parseParityRows, toNum } from './utils.js';

function drawLineChart(svgEl, points, key, color, secondaryKey) {
    if (!svgEl || !window.d3 || !Array.isArray(points) || points.length === 0) return;
    const d3 = window.d3;
    const width = svgEl.clientWidth || 360;
    const height = svgEl.clientHeight || 150;
    const margin = { top: 12, right: 12, bottom: 22, left: 36 };

    const seriesA = points.filter((p) => Number.isFinite(p[key]));
    const seriesB = secondaryKey ? points.filter((p) => Number.isFinite(p[secondaryKey])) : [];
    if (seriesA.length === 0 && seriesB.length === 0) return;

    const x = d3.scaleLinear()
        .domain(d3.extent(points, (p) => p.step))
        .range([margin.left, width - margin.right]);

    const allYValues = [];
    seriesA.forEach((p) => allYValues.push(p[key]));
    seriesB.forEach((p) => allYValues.push(p[secondaryKey]));
    const yMin = d3.min(allYValues);
    const yMax = d3.max(allYValues);
    const y = d3.scaleLinear()
        .domain([Math.min(0, yMin), yMax === yMin ? yMax + 1 : yMax])
        .nice()
        .range([height - margin.bottom, margin.top]);

    const line = d3.line()
        .x((p) => x(p.step))
        .y((p) => y(p[key]))
        .curve(d3.curveMonotoneX);

    const lineSecondary = secondaryKey ? d3.line()
        .x((p) => x(p.step))
        .y((p) => y(p[secondaryKey]))
        .curve(d3.curveMonotoneX) : null;

    const svg = d3.select(svgEl);
    svg.selectAll('*').remove();
    svg.attr('viewBox', `0 0 ${width} ${height}`);

    svg.append('rect')
        .attr('x', 0)
        .attr('y', 0)
        .attr('width', width)
        .attr('height', height)
        .attr('fill', '#232323');

    svg.append('g')
        .attr('transform', `translate(0,${height - margin.bottom})`)
        .call(d3.axisBottom(x).ticks(5).tickSizeOuter(0))
        .call((g) => g.selectAll('text').attr('fill', '#9ca3af').attr('font-size', 10))
        .call((g) => g.selectAll('line,path').attr('stroke', '#4b5563'));

    svg.append('g')
        .attr('transform', `translate(${margin.left},0)`)
        .call(d3.axisLeft(y).ticks(4).tickSizeOuter(0))
        .call((g) => g.selectAll('text').attr('fill', '#9ca3af').attr('font-size', 10))
        .call((g) => g.selectAll('line,path').attr('stroke', '#4b5563'));

    if (seriesA.length > 0) {
        svg.append('path')
            .datum(seriesA)
            .attr('fill', 'none')
            .attr('stroke', color)
            .attr('stroke-width', 2)
            .attr('d', line);
    }

    if (lineSecondary && seriesB.length > 0) {
        svg.append('path')
            .datum(seriesB)
            .attr('fill', 'none')
            .attr('stroke', '#60a5fa')
            .attr('stroke-width', 2)
            .attr('stroke-dasharray', '5,4')
            .attr('d', lineSecondary);
    }
}

function healthAlerts(lossSteps, parityRows) {
    const alerts = [];
    const last = lossSteps[lossSteps.length - 1];
    if (last && Number.isFinite(last.grad_norm)) {
        if (last.grad_norm < 1e-5) {
            alerts.push({ level: 'warning', text: `Gradient norm low (${fmtExp(last.grad_norm, 2)}): possible vanishing trend.` });
        } else if (last.grad_norm > 0.1) {
            alerts.push({ level: 'critical', text: `Gradient norm high (${fmtExp(last.grad_norm, 2)}): possible explosion risk.` });
        }
    }
    const worstParity = parityRows.reduce((m, row) => Math.max(m, toNum(row.max_param_diff, 0)), 0);
    if (worstParity > 1e-3) {
        alerts.push({ level: 'critical', text: `Parity divergence above threshold: max param diff ${fmtExp(worstParity, 2)}.` });
    } else if (worstParity > 1e-5) {
        alerts.push({ level: 'warning', text: `Parity drift elevated: max param diff ${fmtExp(worstParity, 2)}.` });
    } else if (worstParity > 0) {
        alerts.push({ level: 'info', text: `Parity stable: max param diff ${fmtExp(worstParity, 2)}.` });
    }
    return alerts;
}

export function renderTrainingDashboard(files) {
    const panel = document.getElementById('train-dashboard');
    if (!panel) return;
    const root = document.getElementById('trainDashboardRoot');
    if (!root) return;

    clear(root);
    const lossSteps = parseLossSteps(files.training_loss_curve);
    const parityRows = parseParityRows(files.training_parity);
    const profile = files.training_step_profile || {};
    const finalStep = lossSteps[lossSteps.length - 1] || null;
    const worstParity = parityRows.reduce((m, row) => Math.max(m, toNum(row.max_param_diff, 0)), 0);
    const decodeTokS = toNum(profile.decode_tok_s, NaN);

    if (lossSteps.length === 0) {
        root.innerHTML = `
            <div class="parity-section">
                <h3><span class="badge badge-orange">No Training Telemetry</span> Missing training_loss_curve.json</h3>
                <p style="color: var(--text-muted);">
                    Generate training artifacts first, then reload this report.
                </p>
                <pre style="font-size:0.8rem;">make v7-gate-train
python version/v7/tools/open_ir_visualizer.py --generate MODEL --mode train</pre>
            </div>
        `;
        return;
    }

    root.innerHTML = `
        <div class="stats-grid">
            <div class="stat-card"><div class="stat-value">${finalStep ? fmt(finalStep.loss_ck, 4) : '-'}</div><div class="stat-label">Final CK Loss</div></div>
            <div class="stat-card"><div class="stat-value">${finalStep && Number.isFinite(finalStep.grad_norm) ? fmtExp(finalStep.grad_norm, 2) : '-'}</div><div class="stat-label">Grad Norm</div></div>
            <div class="stat-card"><div class="stat-value">${Number.isFinite(worstParity) && worstParity > 0 ? fmtExp(worstParity, 2) : '-'}</div><div class="stat-label">Max Param Diff</div></div>
            <div class="stat-card"><div class="stat-value">${Number.isFinite(decodeTokS) ? fmt(decodeTokS, 2) : '-'}</div><div class="stat-label">Decode tok/s</div></div>
        </div>
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:0.8rem;">
            <div class="parity-section">
                <h3><span class="badge badge-blue">Loss</span> CK vs PyTorch</h3>
                <svg id="trainLossChart" style="width:100%;height:160px;"></svg>
            </div>
            <div class="parity-section">
                <h3><span class="badge badge-orange">Gradient</span> Grad Norm vs Step</h3>
                <svg id="trainGradChart" style="width:100%;height:160px;"></svg>
            </div>
            <div class="parity-section">
                <h3><span class="badge badge-green">Schedule</span> LR vs Step</h3>
                <svg id="trainLrChart" style="width:100%;height:160px;"></svg>
            </div>
        </div>
        <div class="parity-section" style="margin-top:0.8rem;">
            <h3><span class="badge badge-orange">Health</span> Training Alerts</h3>
            <div id="trainAlertList"></div>
        </div>
    `;

    drawLineChart(document.getElementById('trainLossChart'), lossSteps, 'loss_ck', '#f87171', 'loss_pt');
    drawLineChart(document.getElementById('trainGradChart'), lossSteps, 'grad_norm', '#fbbf24');
    drawLineChart(document.getElementById('trainLrChart'), lossSteps, 'lr', '#60a5fa');

    const alertsEl = document.getElementById('trainAlertList');
    const alerts = healthAlerts(lossSteps, parityRows);
    if (!alertsEl) return;
    if (alerts.length === 0) {
        alertsEl.innerHTML = '<div class="alert-item info">No health alerts. Training telemetry looks stable.</div>';
        return;
    }
    alertsEl.innerHTML = alerts.map((a) =>
        `<div class="alert-item ${a.level}">${a.text}</div>`
    ).join('');
}
