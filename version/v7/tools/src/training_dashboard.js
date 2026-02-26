import { clear, fmt, fmtExp, parseLossSteps, parseParityRows, toNum } from './utils.js';

function formatBytesHuman(bytes) {
    const n = Number(bytes || 0);
    if (!Number.isFinite(n) || n <= 0) return '-';
    const gb = 1024 * 1024 * 1024;
    const mb = 1024 * 1024;
    const kb = 1024;
    if (n >= gb) return `${(n / gb).toFixed(2)} GB`;
    if (n >= mb) return `${(n / mb).toFixed(2)} MB`;
    if (n >= kb) return `${(n / kb).toFixed(2)} KB`;
    return `${Math.round(n)} B`;
}

function drawLineChart(svgEl, points, key, color, secondaryKey, opts = {}) {
    if (!svgEl || !window.d3 || !Array.isArray(points) || points.length === 0) return;
    const d3 = window.d3;
    const width = svgEl.clientWidth || 360;
    const height = svgEl.clientHeight || 150;
    const margin = { top: 12, right: 12, bottom: 22, left: 54 };

    const seriesA = points.filter((p) => Number.isFinite(p[key]));
    const seriesB = secondaryKey ? points.filter((p) => Number.isFinite(p[secondaryKey])) : [];
    if (seriesA.length === 0 && seriesB.length === 0) return;

    const x = d3.scaleLinear()
        .domain(d3.extent(points, (p) => p.step))
        .range([margin.left, width - margin.right]);

    const allYValues = [];
    seriesA.forEach((p) => allYValues.push(p[key]));
    seriesB.forEach((p) => allYValues.push(p[secondaryKey]));
    const yMinRaw = d3.min(allYValues);
    const yMaxRaw = d3.max(allYValues);
    const yRangeRaw = Number.isFinite(yMinRaw) && Number.isFinite(yMaxRaw) ? (yMaxRaw - yMinRaw) : NaN;
    const useZeroBaseline = /^loss/.test(String(key || '')) || /^loss/.test(String(secondaryKey || ''));
    const maxAbsRaw = d3.max(allYValues, (v) => Math.abs(Number(v)));
    let axisScalePow10 = 0;
    if (!useZeroBaseline && Number.isFinite(maxAbsRaw) && maxAbsRaw > 0 && (maxAbsRaw < 1e-2 || maxAbsRaw >= 1e4)) {
        axisScalePow10 = Math.max(-8, Math.min(8, -Math.floor(Math.log10(maxAbsRaw))));
        if (Math.abs(axisScalePow10) <= 1) axisScalePow10 = 0;
    }
    const axisScale = Math.pow(10, axisScalePow10);
    const yMin = Number.isFinite(yMinRaw) ? yMinRaw * axisScale : yMinRaw;
    const yMax = Number.isFinite(yMaxRaw) ? yMaxRaw * axisScale : yMaxRaw;
    const yRange = Number.isFinite(yRangeRaw) ? yRangeRaw * axisScale : yRangeRaw;
    let yDomainMin = 0;
    let yDomainMax = 1;
    if (Number.isFinite(yRange) && Number.isFinite(yMin) && Number.isFinite(yMax)) {
        if (yRange <= 0) {
            const pad = Math.max(Math.abs(yMax) * 0.08, 1e-8);
            yDomainMin = useZeroBaseline ? Math.min(0, yMin - pad) : (yMin - pad);
            yDomainMax = yMax + pad;
            if (yDomainMin === yDomainMax) yDomainMax = yDomainMin + 1e-6;
        } else {
            const pad = Math.max(yRange * 0.08, 1e-8);
            yDomainMin = useZeroBaseline ? Math.min(0, yMin - pad) : (yMin - pad);
            yDomainMax = yMax + pad;
            if (yDomainMin === yDomainMax) yDomainMax = yDomainMin + 1e-6;
        }
    }
    const y = d3.scaleLinear()
        .domain([yDomainMin, yDomainMax])
        .nice()
        .range([height - margin.bottom, margin.top]);

    const line = d3.line()
        .x((p) => x(p.step))
        .y((p) => y(p[key] * axisScale))
        .curve(d3.curveMonotoneX);

    const lineSecondary = secondaryKey ? d3.line()
        .x((p) => x(p.step))
        .y((p) => y(p[secondaryKey] * axisScale))
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

    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;

    svg.append('g')
        .attr('transform', `translate(0,${height - margin.bottom})`)
        .call(d3.axisBottom(x).ticks(5).tickSize(-innerH).tickFormat(''))
        .call((g) => g.selectAll('line').attr('stroke', '#374151').attr('stroke-opacity', 0.45))
        .call((g) => g.select('path').attr('stroke', 'none'));

    svg.append('g')
        .attr('transform', `translate(${margin.left},0)`)
        .call(d3.axisLeft(y).ticks(4).tickSize(-innerW).tickFormat(''))
        .call((g) => g.selectAll('line').attr('stroke', '#374151').attr('stroke-opacity', 0.45))
        .call((g) => g.select('path').attr('stroke', 'none'));

    svg.append('g')
        .attr('transform', `translate(0,${height - margin.bottom})`)
        .call(d3.axisBottom(x).ticks(5).tickSizeOuter(0))
        .call((g) => g.selectAll('text').attr('fill', '#9ca3af').attr('font-size', 10))
        .call((g) => g.selectAll('line,path').attr('stroke', '#4b5563'));

    const yTickFmt = (v) => {
        if (!Number.isFinite(v)) return '';
        const abs = Math.abs(v);
        if ((abs > 0 && abs < 1e-3) || abs >= 1e4) return d3.format('.2e')(v);
        if (abs < 1) return d3.format('.3f')(v);
        return d3.format('.3~g')(v);
    };

    svg.append('g')
        .attr('transform', `translate(${margin.left},0)`)
        .call(d3.axisLeft(y).ticks(4).tickSizeOuter(0))
        .call((g) => g.selectAll('.tick text').text((d) => yTickFmt(d)))
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

    const primaryLast = seriesA.length > 0 ? seriesA[seriesA.length - 1] : null;
    if (primaryLast) {
        const stopX = x(primaryLast.step);
        svg.append('line')
            .attr('x1', stopX)
            .attr('x2', stopX)
            .attr('y1', margin.top)
            .attr('y2', height - margin.bottom)
            .attr('stroke', '#9ca3af')
            .attr('stroke-width', 1)
            .attr('stroke-dasharray', '3,3')
            .attr('opacity', 0.75);
        const stopNearRight = stopX > (width - margin.right - 56);
        svg.append('text')
            .attr('x', stopNearRight ? (stopX - 4) : Math.min(width - margin.right - 2, stopX + 4))
            .attr('y', margin.top + 10)
            .attr('fill', '#9ca3af')
            .attr('font-size', 10)
            .attr('text-anchor', stopNearRight ? 'end' : 'start')
            .text(`stop ${Math.round(primaryLast.step)}`);
    }

    if (axisScalePow10 !== 0) {
        const invPow = -axisScalePow10;
        const scaleLabel = `axis ×1e${axisScalePow10}  (actual = tick × 1e${invPow})`;
        svg.append('text')
            .attr('x', margin.left)
            .attr('y', margin.top - 2)
            .attr('fill', '#9ca3af')
            .attr('font-size', 9)
            .text(scaleLabel);
    }

    const mergedByStep = new Map();
    seriesA.forEach((p) => {
        mergedByStep.set(p.step, { step: p.step, a: p[key], b: NaN });
    });
    seriesB.forEach((p) => {
        const existing = mergedByStep.get(p.step) || { step: p.step, a: NaN, b: NaN };
        existing.b = p[secondaryKey];
        mergedByStep.set(p.step, existing);
    });
    const hoverSeries = Array.from(mergedByStep.values()).sort((a, b) => a.step - b.step);
    if (hoverSeries.length === 0) return;

    const bisect = d3.bisector((d) => d.step).left;
    const hoverRoot = svg.append('g').style('display', 'none');
    const hoverLine = hoverRoot.append('line')
        .attr('stroke', '#cbd5e1')
        .attr('stroke-width', 1)
        .attr('stroke-dasharray', '3,3')
        .attr('y1', margin.top)
        .attr('y2', height - margin.bottom);
    const hoverDotA = hoverRoot.append('circle').attr('r', 3.2).attr('fill', color).attr('stroke', '#111827').attr('stroke-width', 1);
    const hoverDotB = hoverRoot.append('circle').attr('r', 3.2).attr('fill', '#60a5fa').attr('stroke', '#111827').attr('stroke-width', 1).style('display', 'none');
    const hoverBox = hoverRoot.append('g');
    const hoverBoxBg = hoverBox.append('rect')
        .attr('rx', 6)
        .attr('ry', 6)
        .attr('fill', '#0f172a')
        .attr('stroke', '#334155')
        .attr('opacity', 0.95);
    const hoverLines = [
        hoverBox.append('text').attr('fill', '#e5e7eb').attr('font-size', 10).attr('font-family', 'JetBrains Mono, monospace'),
        hoverBox.append('text').attr('fill', color).attr('font-size', 10).attr('font-family', 'JetBrains Mono, monospace'),
        hoverBox.append('text').attr('fill', '#60a5fa').attr('font-size', 10).attr('font-family', 'JetBrains Mono, monospace'),
    ];

    const fmtMetric = (v) => {
        if (!Number.isFinite(v)) return '-';
        const abs = Math.abs(v);
        if ((abs > 0 && abs < 1e-3) || abs >= 1e4) return fmtExp(v, 2);
        return fmt(v, 4);
    };

    const overlay = svg.append('rect')
        .attr('x', margin.left)
        .attr('y', margin.top)
        .attr('width', innerW)
        .attr('height', innerH)
        .attr('fill', 'transparent')
        .style('cursor', 'crosshair');

    overlay
        .on('mouseenter', () => hoverRoot.style('display', null))
        .on('mouseleave', () => hoverRoot.style('display', 'none'))
        .on('mousemove', function onMove(event) {
            const [mx] = d3.pointer(event, this);
            const xStep = x.invert(Math.max(margin.left, Math.min(width - margin.right, mx)));
            let idx = bisect(hoverSeries, xStep, 1);
            if (idx >= hoverSeries.length) idx = hoverSeries.length - 1;
            const prev = hoverSeries[Math.max(0, idx - 1)];
            const next = hoverSeries[idx];
            const d = !next ? prev : (Math.abs(next.step - xStep) < Math.abs(xStep - prev.step) ? next : prev);
            if (!d) return;

            const px = x(d.step);
            hoverLine.attr('x1', px).attr('x2', px);

            if (Number.isFinite(d.a)) {
                hoverDotA.style('display', null).attr('cx', px).attr('cy', y(d.a * axisScale));
            } else {
                hoverDotA.style('display', 'none');
            }
            if (secondaryKey && Number.isFinite(d.b)) {
                hoverDotB.style('display', null).attr('cx', px).attr('cy', y(d.b * axisScale));
            } else {
                hoverDotB.style('display', 'none');
            }

            const lines = [`step ${Math.round(d.step)}`, `${key}: ${fmtMetric(d.a)}`];
            if (secondaryKey) lines.push(`${secondaryKey}: ${fmtMetric(d.b)}`);
            else lines.push('');

            hoverLines[0].text(lines[0]);
            hoverLines[1].text(lines[1]);
            hoverLines[2].text(lines[2]);

            const maxChars = lines.reduce((m, s) => Math.max(m, String(s).length), 0);
            const tipW = Math.max(104, maxChars * 6.2 + 12);
            const tipH = secondaryKey ? 46 : 32;
            const tipX = px + 8 + tipW <= (width - margin.right) ? px + 8 : px - tipW - 8;
            const tipYBase = margin.top + 6;
            const tipY = Math.min(height - margin.bottom - tipH, Math.max(margin.top + 2, tipYBase));
            hoverBox.attr('transform', `translate(${tipX},${tipY})`);
            hoverBoxBg.attr('width', tipW).attr('height', tipH);
            hoverLines[0].attr('x', 6).attr('y', 13);
            hoverLines[1].attr('x', 6).attr('y', 26);
            hoverLines[2].attr('x', 6).attr('y', 39).style('display', secondaryKey ? null : 'none');
        });

    if (opts && typeof opts.title === 'string' && opts.title) {
        svgEl.setAttribute('data-ck-svg-expand', opts.title);
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

function getRunContext() {
    const embedded = window.EMBEDDED_IR_DATA || {};
    const meta = embedded.meta || {};
    const runDir = typeof meta.run_dir === 'string' && meta.run_dir.trim()
        ? meta.run_dir
        : null;
    const modelPath = typeof meta.path === 'string' && meta.path.trim()
        ? meta.path
        : null;
    return { runDir, modelPath };
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
    const sweep = files.training_epoch_sweep || {};
    const sweepRows = Array.isArray(sweep.runs) ? sweep.runs : [];
    const sweepProfile = sweep && typeof sweep === 'object' ? sweep.profile_run : null;
    const finalStep = lossSteps[lossSteps.length - 1] || null;
    const worstParity = parityRows.reduce((m, row) => Math.max(m, toNum(row.max_param_diff, 0)), 0);
    const trainTokS = toNum(profile.train_tok_s, NaN);
    const fallbackTokS = toNum(profile.decode_tok_s, NaN);
    const throughputTokS = Number.isFinite(trainTokS) ? trainTokS : fallbackTokS;
    const checkpoints = Array.isArray(files?.analysis_checkpoints?.checkpoints) ? files.analysis_checkpoints.checkpoints : [];
    const trainE2E = files.train_e2e || {};
    const layoutTrain = files.layout_train || {};
    const layoutBytes = toNum(layoutTrain.total_bytes, NaN);
    const layoutRegionCount = Array.isArray(layoutTrain.regions) ? layoutTrain.regions.length : 0;
    const runCtx = getRunContext();
    const memoryDiag = files.memory_diagnostic || {};
    const memoryDiagState = memoryDiag && memoryDiag.diagnostic && memoryDiag.diagnostic.ok === true
        ? 'PASS'
        : (Object.keys(memoryDiag).length ? 'FAIL' : '-');
    const layoutAudit = files.layout_train_audit || {};
    const layoutAuditState = layoutAudit && layoutAudit.passed === true
        ? 'PASS'
        : (Object.keys(layoutAudit).length ? 'FAIL' : '-');
    const pipeline = files.training_pipeline || {};
    const dataLab = pipeline && typeof pipeline === 'object' && pipeline.data_lab && typeof pipeline.data_lab === 'object'
        ? pipeline.data_lab
        : {};
    const postEval = dataLab.post_train_eval && typeof dataLab.post_train_eval === 'object'
        ? dataLab.post_train_eval
        : (files.post_train_eval || {});
    const postEvalStatus = String(postEval.status || '').toLowerCase();
    const postEvalSkipped = postEvalStatus === 'skipped';
    const validSvgRate = toNum(postEval.valid_svg_rate, NaN);
    const closureRate = toNum(postEval.closure_success_rate, NaN);
    const loopScore = toNum(postEval.repetition_loop_score, NaN);
    const finalLoss = finalStep && Number.isFinite(finalStep.loss_ck)
        ? finalStep.loss_ck
        : toNum(trainE2E.final_ck_loss, NaN);
    const maxParamDiff = Number.isFinite(worstParity) && worstParity > 0
        ? worstParity
        : toNum(trainE2E.final_param_max_abs_diff, NaN);
    const trainE2EStatus = (() => {
        if (trainE2E?.status === 'pass' || trainE2E?.pass === true || trainE2E?.passed === true || trainE2E?.pass_parity === true) {
            return 'PASS';
        }
        if (trainE2E?.status === 'fail' || trainE2E?.pass === false || trainE2E?.passed === false || trainE2E?.pass_parity === false) {
            return 'FAIL';
        }
        return Object.keys(trainE2E).length ? 'CHECK' : '-';
    })();
    const validSvgLabel = postEvalSkipped ? 'SKIP' : (Number.isFinite(validSvgRate) ? fmt(validSvgRate, 4) : '-');
    const closureLabel = postEvalSkipped ? 'SKIP' : (Number.isFinite(closureRate) ? fmt(closureRate, 4) : '-');
    const loopLabel = postEvalSkipped ? 'SKIP' : (Number.isFinite(loopScore) ? fmt(loopScore, 4) : '-');

    const reportCmd = runCtx.runDir
        ? `python3 version/v7/tools/open_ir_visualizer.py --generate --run ${runCtx.runDir} --html-only`
        : 'python3 version/v7/tools/open_ir_visualizer.py --generate <model> --html-only';
    const suiteCmd = runCtx.runDir
        ? `version/v7/scripts/cks-v7-run train-suite --run ${runCtx.runDir} --profile-train perf --profile-epoch 3`
        : 'version/v7/scripts/cks-v7-run train-suite --run ./version/v7/runs/exp1 --profile-train perf --profile-epoch 3';

    if (lossSteps.length === 0) {
        root.innerHTML = `
            <div class="parity-section">
                <h3><span class="badge badge-orange">No Training Telemetry</span> Missing training_loss_curve.json</h3>
                <p style="color: var(--text-muted);">
                    Generate training artifacts first, then reload this report.
                </p>
                <pre style="font-size:0.8rem;white-space:pre-wrap;">${reportCmd}</pre>
            </div>
        `;
        return;
    }

    root.innerHTML = `
        <div class="stats-grid">
            <div class="stat-card"><div class="stat-value">${Number.isFinite(finalLoss) ? fmt(finalLoss, 4) : '-'}</div><div class="stat-label">Final CK Loss</div></div>
            <div class="stat-card"><div class="stat-value">${finalStep && Number.isFinite(finalStep.grad_norm) ? fmtExp(finalStep.grad_norm, 2) : '-'}</div><div class="stat-label">Grad Norm</div></div>
            <div class="stat-card"><div class="stat-value">${Number.isFinite(maxParamDiff) ? fmtExp(maxParamDiff, 2) : '-'}</div><div class="stat-label">Max Param Diff</div></div>
            <div class="stat-card"><div class="stat-value">${Number.isFinite(throughputTokS) ? fmt(throughputTokS, 2) : '-'}</div><div class="stat-label">Train tok/s</div></div>
            <div class="stat-card"><div class="stat-value">${checkpoints.length}</div><div class="stat-label">Analysis Checkpoints</div></div>
            <div class="stat-card"><div class="stat-value">${Number.isFinite(layoutBytes) ? formatBytesHuman(layoutBytes) : '-'}</div><div class="stat-label">Train Memory Arena</div></div>
            <div class="stat-card"><div class="stat-value">${layoutRegionCount > 0 ? layoutRegionCount : '-'}</div><div class="stat-label">Memory Regions</div></div>
            <div class="stat-card"><div class="stat-value">${trainE2EStatus}</div><div class="stat-label">Train E2E Status</div></div>
            <div class="stat-card"><div class="stat-value">${memoryDiagState}</div><div class="stat-label">Memory Diagnostic</div></div>
            <div class="stat-card"><div class="stat-value">${layoutAuditState}</div><div class="stat-label">Layout Audit</div></div>
            <div class="stat-card"><div class="stat-value">${validSvgLabel}</div><div class="stat-label">Valid SVG Rate</div></div>
            <div class="stat-card"><div class="stat-value">${closureLabel}</div><div class="stat-label">Closure Success</div></div>
            <div class="stat-card"><div class="stat-value">${loopLabel}</div><div class="stat-label">Loop Score</div></div>
        </div>
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:0.8rem;">
            <div class="parity-section">
                <h3><span class="badge badge-blue">Loss</span> CK vs PyTorch</h3>
                <svg id="trainLossChart" style="width:100%;height:160px;"></svg>
                <button class="ck-svg-expand-btn" data-title="Training Loss: CK vs PyTorch">⛶ Expand</button>
            </div>
            <div class="parity-section">
                <h3><span class="badge badge-orange">Gradient</span> Grad Norm vs Step</h3>
                <svg id="trainGradChart" style="width:100%;height:160px;"></svg>
                <button class="ck-svg-expand-btn" data-title="Training Gradient Norm vs Step">⛶ Expand</button>
            </div>
            <div class="parity-section">
                <h3><span class="badge badge-green">Schedule</span> LR vs Step</h3>
                <svg id="trainLrChart" style="width:100%;height:160px;"></svg>
                <button class="ck-svg-expand-btn" data-title="Training Learning Rate vs Step">⛶ Expand</button>
            </div>
        </div>
        <div style="color:var(--text-muted);font-size:0.78rem;margin-top:0.35rem;">
            Hover any chart for exact step/value. Dashed vertical line marks the final training step. Click chart or ⛶ to open fullscreen zoom/pan.
        </div>
        <div style="color:var(--text-muted);font-size:0.76rem;margin-top:0.15rem;">
            LR chart shows base AdamW learning rate schedule; Adam's per-parameter adaptive scaling is internal and not plotted directly.
        </div>
        <div class="parity-section" style="margin-top:0.8rem;">
            <h3><span class="badge badge-orange">Health</span> Training Alerts</h3>
            <div id="trainAlertList"></div>
        </div>
        <div class="parity-section" style="margin-top:0.8rem;">
            <h3><span class="badge badge-blue">Sweep</span> Epoch Stability Summary</h3>
            <div id="trainSweepTable"></div>
        </div>
        <div class="parity-section" style="margin-top:0.8rem;">
            <h3><span class="badge badge-orange">Runbook</span> Operator Commands (Copy/Paste)</h3>
            <p style="color:var(--text-muted);margin-bottom:0.6rem;">Producer = CLI writes run_dir artifacts, viewer consumes run_dir.</p>
            <pre style="font-size:0.8rem;white-space:pre-wrap;">${reportCmd}</pre>
            <pre style="font-size:0.8rem;white-space:pre-wrap;">${suiteCmd}</pre>
        </div>
    `;

    drawLineChart(document.getElementById('trainLossChart'), lossSteps, 'loss_ck', '#f87171', 'loss_pt', { title: 'Training Loss: CK vs PyTorch' });
    drawLineChart(document.getElementById('trainGradChart'), lossSteps, 'grad_norm', '#fbbf24', null, { title: 'Training Gradient Norm vs Step' });
    drawLineChart(document.getElementById('trainLrChart'), lossSteps, 'lr', '#60a5fa', null, { title: 'Training Learning Rate vs Step' });

    const alertsEl = document.getElementById('trainAlertList');
    const alerts = healthAlerts(lossSteps, parityRows);
    const svgThreshold = toNum(postEval.min_valid_svg_rate, 0.70);
    if (postEvalSkipped) {
        alerts.push({
            level: 'info',
            text: `Output quality gate skipped (${postEval.reason || 'disabled_by_flag'}). This does not indicate CK-vs-PyTorch parity failure.`,
        });
    } else if (Number.isFinite(validSvgRate) && validSvgRate < svgThreshold) {
        alerts.push({
            level: 'warning',
            text: `Output quality gate is below target (valid_svg_rate=${fmt(validSvgRate, 4)} < ${fmt(svgThreshold, 4)}). This is data/task fit, not a CK-vs-PyTorch parity failure. Improve corpus coverage and add instruction-to-SVG SFT pairs.`,
        });
    }
    if (!alertsEl) return;
    if (alerts.length === 0) {
        alertsEl.innerHTML = '<div class="alert-item info">No health alerts. Training telemetry looks stable.</div>';
    } else {
        alertsEl.innerHTML = alerts.map((a) =>
            `<div class="alert-item ${a.level}">${a.text}</div>`
        ).join('');
    }

    const sweepEl = document.getElementById('trainSweepTable');
    if (!sweepEl) return;
    if (!Array.isArray(sweepRows) || sweepRows.length === 0) {
        sweepEl.innerHTML = '<div style="color:var(--text-muted);">No training_epoch_sweep_latest.json loaded yet. Run <code>ck_run_v7.py train-suite</code>.</div>';
        return;
    }

    const rows = [...sweepRows]
        .sort((a, b) => toNum(a.epoch, 0) - toNum(b.epoch, 0))
        .map((r) => {
            const ok = r.pass_parity === true;
            const badge = ok
                ? '<span class="badge badge-green">PASS</span>'
                : '<span class="badge badge-orange">FAIL</span>';
            return `
                <tr>
                    <td>${toNum(r.epoch, 0)}</td>
                    <td>${badge}</td>
                    <td>${fmt(toNum(r.final_ck_loss, NaN), 4)}</td>
                    <td>${fmtExp(toNum(r.max_loss_abs_diff, NaN), 2)}</td>
                    <td>${fmtExp(toNum(r.final_param_max_abs_diff, NaN), 2)}</td>
                    <td>${Number.isFinite(toNum(r.train_tok_s, NaN)) ? fmt(toNum(r.train_tok_s, NaN), 1) : '-'}</td>
                </tr>`;
        })
        .join('');

    const profileMeta = sweepProfile && typeof sweepProfile === 'object'
        ? `<div style="color:var(--text-muted);margin-bottom:0.5rem;">Spot profile: epoch ${toNum(sweepProfile.epoch, 0)} (${String(sweepProfile.mode || 'none')})</div>`
        : '';

    sweepEl.innerHTML = `
        ${profileMeta}
        <table>
            <thead><tr><th>Epoch</th><th>Parity</th><th>Final Loss</th><th>Max Loss Diff</th><th>Max Param Diff</th><th>Train tok/s</th></tr></thead>
            <tbody>${rows}</tbody>
        </table>
    `;
}
