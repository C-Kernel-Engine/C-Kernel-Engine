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

    const boundaries = Array.isArray(opts.boundaries) ? opts.boundaries : [];
    if (boundaries.length > 0) {
        boundaries.slice(0, 12).forEach((b, idx) => {
            const stepVal = Number(b && b.step);
            if (!Number.isFinite(stepVal)) return;
            const bx = x(stepVal);
            if (!Number.isFinite(bx) || bx < margin.left || bx > (width - margin.right)) return;
            const bColor = typeof b.color === 'string' && b.color.trim() ? b.color : '#6b7280';
            svg.append('line')
                .attr('x1', bx)
                .attr('x2', bx)
                .attr('y1', margin.top)
                .attr('y2', height - margin.bottom)
                .attr('stroke', bColor)
                .attr('stroke-width', 1)
                .attr('stroke-dasharray', '2,3')
                .attr('opacity', 0.6);
            const label = String((b && b.label) || '').trim();
            if (label && idx < 8) {
                const nearRight = bx > (width - margin.right - 70);
                svg.append('text')
                    .attr('x', nearRight ? (bx - 3) : (bx + 3))
                    .attr('y', margin.top + 20 + (idx % 2) * 10)
                    .attr('fill', bColor)
                    .attr('font-size', 9)
                    .attr('text-anchor', nearRight ? 'end' : 'start')
                    .text(label);
            }
        });
    }

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
    const gradSeries = Array.isArray(lossSteps)
        ? lossSteps
            .map((s) => toNum(s.grad_norm, NaN))
            .filter((v) => Number.isFinite(v))
        : [];
    if (gradSeries.length > 0) {
        const allZero = gradSeries.every((v) => Math.abs(v) <= 1e-12);
        if (allZero) {
            alerts.push({
                level: 'warning',
                text: 'Grad norm is 0.00e+0 at every logged step. This can indicate missing grad telemetry in the current runtime path; treat as "unknown" unless loss and parity also degrade.',
            });
        } else if (last && Number.isFinite(last.grad_norm)) {
            if (last.grad_norm < 1e-5) {
                alerts.push({ level: 'warning', text: `Gradient norm low (${fmtExp(last.grad_norm, 2)}): possible vanishing trend.` });
            } else if (last.grad_norm > 0.1) {
                alerts.push({ level: 'critical', text: `Gradient norm high (${fmtExp(last.grad_norm, 2)}): possible explosion risk.` });
            }
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

const STAGE_COLOR = {
    pretrain: '#60a5fa',
    midtrain: '#22d3ee',
    sft: '#f59e0b',
    dpo: '#a78bfa',
    grpo: '#8b5cf6',
    ppo: '#c084fc',
    unassigned: '#6b7280',
};

function normalizeStageName(raw) {
    const s = String(raw || '').trim().toLowerCase();
    if (!s) return '';
    if (s === 'stage_a') return 'pretrain';
    if (s === 'stage_b') return 'midtrain';
    return s;
}

function stageRangesFromTimeline(timeline) {
    const ranges = {};
    if (!Array.isArray(timeline)) return ranges;
    timeline.forEach((row) => {
        if (!row || typeof row !== 'object') return;
        const stage = normalizeStageName(row.stage);
        if (!stage) return;
        const start = Number(row.step_start);
        const end = Number(row.step_end);
        const hasStart = Number.isFinite(start);
        const hasEnd = Number.isFinite(end);
        if (!hasStart && !hasEnd) return;
        ranges[stage] = {
            start: hasStart ? start : null,
            end: hasEnd ? end : null,
        };
    });
    return ranges;
}

function resolvePointStage(point, stageRanges, activeStage) {
    const direct = normalizeStageName(point && point.source_stage);
    if (direct) return direct;
    const step = Number(point && point.step);
    if (Number.isFinite(step)) {
        for (const [stage, range] of Object.entries(stageRanges || {})) {
            const s = Number(range && range.start);
            const e = Number(range && range.end);
            const geStart = !Number.isFinite(s) || step >= s;
            const leEnd = !Number.isFinite(e) || step <= e;
            if (geStart && leEnd) return stage;
        }
    }
    return normalizeStageName(activeStage) || 'unassigned';
}

function buildStageBoundaries(points) {
    const out = [];
    for (let i = 1; i < points.length; i++) {
        const prev = normalizeStageName(points[i - 1] && points[i - 1]._stage);
        const cur = normalizeStageName(points[i] && points[i]._stage);
        if (!prev || !cur || prev === cur) continue;
        const step = Number(points[i].step);
        if (!Number.isFinite(step)) continue;
        out.push({
            step,
            label: `${prev}→${cur}`,
            color: STAGE_COLOR[cur] || STAGE_COLOR.unassigned,
        });
    }
    return out;
}

function computeSaturation(points, activeStage, windowSize = 20) {
    const stage = normalizeStageName(activeStage) || 'unassigned';
    const stagePoints = points
        .filter((p) => normalizeStageName(p && p._stage) === stage && Number.isFinite(Number(p && p.loss_ck)))
        .map((p) => Number(p.loss_ck));
    if (stagePoints.length < 4) return null;
    const tail = stagePoints.slice(-Math.max(4, windowSize));
    const n = tail.length;
    const xs = Array.from({ length: n }, (_, i) => i);
    const meanX = xs.reduce((a, b) => a + b, 0) / n;
    const meanY = tail.reduce((a, b) => a + b, 0) / n;
    let num = 0;
    let den = 0;
    for (let i = 0; i < n; i++) {
        const dx = xs[i] - meanX;
        num += dx * (tail[i] - meanY);
        den += dx * dx;
    }
    const slope = den > 0 ? (num / den) : 0;
    const variance = tail.reduce((acc, v) => acc + ((v - meanY) * (v - meanY)), 0) / n;
    const std = Math.sqrt(Math.max(0, variance));
    if (slope > 1e-3) {
        return { verdict: 'EXPLODING', color: '#ef4444', text: 'Loss is rising; reduce LR or increase grad clipping.', slope, std, n, stage };
    }
    if (Math.abs(slope) < 1e-4 && std < 1e-2) {
        return { verdict: 'PLATEAUED', color: '#f59e0b', text: 'Stage looks saturated; move to next curriculum stage.', slope, std, n, stage };
    }
    if (slope < -1e-4) {
        return { verdict: 'DECLINING', color: '#47b475', text: 'Still learning in this stage; continue training.', slope, std, n, stage };
    }
    return { verdict: 'NOISY', color: '#9ca3af', text: 'Signal is noisy; monitor another 10-20 steps before switching.', slope, std, n, stage };
}

function renderTrainingDataPipelineSection(container, pipeline, dataLab, postEval, lossSteps) {
    if (!container || !pipeline || typeof pipeline !== 'object') return;

    const activeStage = String(pipeline.active_stage || 'pretrain').toLowerCase();
    const curriculumStage = String(pipeline.curriculum_stage || 'stage_a').toLowerCase();
    const stageTimeline = Array.isArray(pipeline.stage_timeline) ? pipeline.stage_timeline : [];
    const catalog = Array.isArray(pipeline.dataset_catalog) ? pipeline.dataset_catalog : [];
    const provenance = Array.isArray(pipeline.data_provenance) ? pipeline.data_provenance : [];
    const tokLineage = (pipeline.tokenizer_lineage && typeof pipeline.tokenizer_lineage === 'object')
        ? pipeline.tokenizer_lineage : {};
    const trainDims = (pipeline.train_dims && typeof pipeline.train_dims === 'object')
        ? pipeline.train_dims : {};
    const tokenRoundtrip = (dataLab && dataLab.tokenizer_roundtrip && typeof dataLab.tokenizer_roundtrip === 'object')
        ? dataLab.tokenizer_roundtrip : {};
    const dsQc = (dataLab && dataLab.dataset_qc && typeof dataLab.dataset_qc === 'object')
        ? dataLab.dataset_qc : {};
    const activeProv = provenance.length > 0 ? provenance[0] : null;

    const STAGE_DEFS = [
        { id: 'pretrain_a', stage: 'pretrain', curriculum: 'stage_a', label: 'Pretrain — Stage A', teaches: 'SVG syntax, shapes, paths, colors, attribute patterns' },
        { id: 'pretrain_b', stage: 'pretrain', curriculum: 'stage_b', label: 'Pretrain — Stage B', teaches: 'Complex compositions, groups, transforms, responsive layouts' },
        { id: 'midtrain', stage: 'midtrain', curriculum: null, label: 'Mid-train', teaches: 'Structured prompt routing, SVG completion from partial instructions' },
        { id: 'sft', stage: 'sft', curriculum: null, label: 'SFT', teaches: '"draw X" → SVG generation, instruction following' },
        { id: 'dpo', stage: 'dpo', curriculum: null, label: 'Preference (DPO/GRPO/PPO)', teaches: 'Output quality ranking, alignment to preference' },
    ];

    const timelineMap = {};
    stageTimeline.forEach((s) => { timelineMap[s.stage] = s; });

    function stageStatus(def) {
        if (def.stage === activeStage) {
            if (def.curriculum === null || def.curriculum === curriculumStage) return 'active';
            return 'ready';
        }
        const t = timelineMap[def.stage];
        if (t && t.status === 'completed') return 'completed';
        if (t && t.status === 'active') return 'active';
        return 'planned';
    }

    function getRows(def) {
        if (def.id === 'pretrain_a') {
            const e = catalog.find((c) => c.kind === 'active_dataset' && c.stage === 'pretrain');
            return e ? e.rows : (toNum(dsQc.non_empty_lines, null) || toNum(dsQc.total_lines, null));
        }
        if (def.id === 'pretrain_b') {
            const e = catalog.find((c) => c.stage === 'pretrain' && String(c.name || '').includes('stage_b_syn'));
            return e ? e.rows : 63013;
        }
        if (def.id === 'midtrain') {
            const entries = catalog.filter((c) => c.stage === 'midtrain' && toNum(c.rows, 0) > 0);
            const total = entries.reduce((s, c) => s + toNum(c.rows, 0), 0);
            return total > 0 ? total : null;
        }
        if (def.id === 'sft') {
            const e = catalog.find((c) => c.stage === 'sft' && toNum(c.rows, 0) > 0);
            return e ? e.rows : 533000;
        }
        return null;
    }

    // dsQc may reflect a different dataset than what was trained (e.g. Stage B QC run after Stage A training).
    // activeProv is the ground truth for what was actually trained — always prefer it.
    const dsQcMatchesProv = activeProv && dsQc.dataset_name &&
        (dsQc.dataset_name === activeProv.dataset_name ||
         String(dsQc.dataset_name).includes('stage_a') === String(activeProv.dataset_name || '').includes('stage_a'));

    function getTokens(def) {
        if (def.id === 'pretrain_a') {
            // Prefer provenance token count; fall back to roundtrip only if dsQc matches the trained dataset
            return (activeProv ? toNum(activeProv.token_count, null) : null)
                || (dsQcMatchesProv ? toNum(tokenRoundtrip.token_count, null) : null);
        }
        return null;
    }

    function getBytes(def) {
        if (def.id === 'pretrain_a') {
            // Prefer provenance byte_size — dsQc.bytes could point to a different stage
            return (activeProv ? toNum(activeProv.byte_size, null) : null)
                || (dsQcMatchesProv ? toNum(dsQc.bytes, null) : null);
        }
        if (def.id === 'pretrain_b') return 15703061;
        return null;
    }

    function fmtRows(n) {
        if (n == null || !Number.isFinite(Number(n))) return '—';
        const v = Number(n);
        if (v >= 1000000) return `${(v / 1000000).toFixed(1)}M`;
        if (v >= 1000) return `${(v / 1000).toFixed(0)}k`;
        return String(v);
    }

    function fmtTok(n) {
        if (n == null || !Number.isFinite(Number(n))) return '—';
        const v = Number(n);
        if (v >= 1000000) return `${(v / 1000000).toFixed(2)}M`;
        if (v >= 1000) return `${(v / 1000).toFixed(0)}k`;
        return String(v);
    }

    const STATUS_BADGE = {
        active: '<span class="badge badge-green">ACTIVE</span>',
        ready: '<span class="badge badge-orange">READY</span>',
        planned: '<span style="color:#6b7280;font-size:0.78rem;">planned</span>',
        completed: '<span class="badge badge-blue">DONE</span>',
    };

    const NEXT_ACTIONS = {
        pretrain_a: 'Check loss convergence → include Stage B in next run',
        pretrain_b: 'Add to run config; re-train or continue from checkpoint',
        midtrain: 'Await Stage B saturation → activate mid-train pack',
        sft: 'After mid-train gate passes → launch SFT run',
        dpo: 'After SFT convergence → curate preference pairs',
    };

    const stageRows = STAGE_DEFS.map((def) => {
        const status = stageStatus(def);
        const rows = getRows(def);
        const tokens = getTokens(def);
        const bytes = getBytes(def);
        const action = (status === 'active' || status === 'ready') ? NEXT_ACTIONS[def.id] : '—';
        return `<tr>
            <td><strong>${def.label}</strong></td>
            <td style="color:var(--text-muted);font-size:0.78rem;">${def.teaches}</td>
            <td style="text-align:right;">${fmtRows(rows)}</td>
            <td style="text-align:right;">${fmtTok(tokens)}</td>
            <td style="text-align:right;">${bytes != null ? formatBytesHuman(bytes) : '—'}</td>
            <td>${STATUS_BADGE[status] || ''}</td>
            <td style="color:var(--text-muted);font-size:0.76rem;">${action}</td>
        </tr>`;
    }).join('');

    const vocabSize = toNum(tokLineage.vocab_size, null);
    const bpeMode = tokLineage.bpe_mode || tokLineage.type || '—';
    const tokStatus = String(tokenRoundtrip.status || '—').toUpperCase();
    const tokExact = tokenRoundtrip.exact_match === true ? 'yes' : (tokenRoundtrip.exact_match === false ? 'no' : '—');
    const tokCount = toNum(tokenRoundtrip.token_count, null);
    const tokInputBytes = toNum(tokenRoundtrip.input_bytes, null);
    const tokRatio = (tokCount && tokInputBytes) ? (tokCount / tokInputBytes).toFixed(3) : '—';

    const validRate = toNum(postEval.valid_svg_rate, null);
    const closureRate = toNum(postEval.closure_success_rate, null);
    const loopScore = toNum(postEval.repetition_loop_score, null);
    const qualityGateColor = (validRate != null && validRate < 0.7) ? '#ef4444' : '#ffb400';

    const firstStep = lossSteps[0];
    const lastStep = lossSteps[lossSteps.length - 1];
    const lossRange = (firstStep && lastStep && Number.isFinite(firstStep.loss_ck) && Number.isFinite(lastStep.loss_ck))
        ? `${fmt(firstStep.loss_ck, 2)} → ${fmt(lastStep.loss_ck, 2)} over ${Math.round(lastStep.step - firstStep.step)} steps`
        : '—';

    // activeProv is ground truth for what was actually trained.
    // dsQc may point to a different (e.g. newer) dataset — only use it as fallback if dataset names match.
    const activeDatasetName = (activeProv && activeProv.dataset_name)
        || (dsQcMatchesProv ? dsQc.dataset_name : null) || '—';
    const catalogActiveEntry = catalog.find((c) => c.kind === 'active_dataset' && c.stage === 'pretrain');
    const activeRows = (catalogActiveEntry ? toNum(catalogActiveEntry.rows, null) : null)
        || (dsQcMatchesProv ? (toNum(dsQc.non_empty_lines, null) || toNum(dsQc.total_lines, null)) : null);
    const activeBytes = (activeProv ? toNum(activeProv.byte_size, null) : null)
        || (dsQcMatchesProv ? toNum(dsQc.bytes, null) : null);
    const activeTokCount = (activeProv ? toNum(activeProv.token_count, null) : null)
        || (dsQcMatchesProv ? tokCount : null);

    const promptLines = [
        '# CK Engine Training Analysis Request',
        '',
        '## Model Architecture',
        `- Layers: ${toNum(trainDims.num_layers, '?')}`,
        `- Embed dim: ${toNum(trainDims.embed_dim, '?')}`,
        `- Heads: ${toNum(trainDims.num_heads, '?')} (KV heads: ${toNum(trainDims.num_kv_heads, '?')})`,
        `- Vocab: ${toNum(trainDims.vocab_size, '?')} tokens (ascii_bpe, SVG-domain)`,
        '',
        '## Training State',
        `- Active stage: ${activeStage} / curriculum: ${curriculumStage}`,
        `- Loss trajectory: ${lossRange}`,
        `- Data trained on: Stage A only (${fmtRows(activeRows)} rows, ${fmtTok(activeTokCount)} tokens)`,
        `- Stage B (${fmtRows(63013)} rows, 15MB) exists but NOT yet trained`,
        `- Tokenizer roundtrip: ${tokStatus} (exact_match=${tokExact})`,
        '',
        '## Output Quality Gate',
        `- valid_svg_rate: ${validRate != null ? fmt(validRate, 4) : 'n/a'} (target ≥0.70)`,
        `- closure_success_rate: ${closureRate != null ? fmt(closureRate, 4) : 'n/a'}`,
        `- repetition_loop_score: ${loopScore != null ? fmt(loopScore, 4) : 'n/a'}`,
        '',
        '## Untrained Datasets Available',
        '- Stage B pretrain: 63k rows (~15MB SVG compositions)',
        '- Mid-train instruction pack: ~28k rows',
        '- SFT instruction pairs: ~533k pairs (holdout: ~59k)',
        '',
        '## Analysis Questions',
        '1. Is Stage A pretrain sufficient to advance to Stage B, or should we continue?',
        '2. What does the loss curve shape tell us about data saturation?',
        '3. Is valid_svg_rate=0% a kernel bug, data coverage gap, or sampling collapse?',
        '4. What minimum data prep is needed before SFT instruction fine-tuning?',
        '5. Can instruction following work with 1024-token ascii_bpe vocab (English as char-by-char)?',
        '6. What Stage B → mid-train mix ratio do you recommend?',
        '7. How do we measure Stage B saturation before advancing?',
    ];
    const promptText = promptLines.join('\n');
    const promptHtml = promptText.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

    const timelinePills = STAGE_DEFS.map((def) => {
        const s = stageStatus(def);
        const clr = s === 'active' ? '#47b475' : s === 'ready' ? '#ffb400' : s === 'completed' ? '#07adf8' : '#4b5563';
        const bdr = s === 'active' ? `2px solid ${clr}` : `1px solid ${clr}`;
        return `<div style="padding:0.22rem 0.55rem;border:${bdr};border-radius:4px;font-size:0.74rem;color:${clr};white-space:nowrap;">${def.label}</div>`;
    }).join('<div style="color:#4b5563;font-size:0.74rem;padding:0 0.1rem;">→</div>');

    container.innerHTML = `
        <div class="parity-section" style="margin-top:0.8rem;">
            <h3><span class="badge badge-blue">Data Pipeline</span> Training Stage Awareness</h3>

            <div style="display:flex;gap:0.4rem;align-items:center;flex-wrap:wrap;margin-bottom:0.85rem;">
                ${timelinePills}
            </div>

            <div style="background:#2a2a2a;border-left:3px solid #47b475;padding:0.5rem 0.75rem;border-radius:0 4px 4px 0;margin-bottom:0.75rem;font-size:0.8rem;">
                <strong style="color:#47b475;">Active dataset:</strong>
                <span style="color:var(--text-muted);margin-left:0.35rem;">${activeDatasetName}</span>
                <span style="color:#4b5563;margin:0 0.4rem;">·</span>
                <span style="color:var(--text-muted);">${fmtRows(activeRows)} rows</span>
                <span style="color:#4b5563;margin:0 0.4rem;">·</span>
                <span style="color:var(--text-muted);">${formatBytesHuman(activeBytes)}</span>
                <span style="color:#4b5563;margin:0 0.4rem;">·</span>
                <span style="color:var(--text-muted);">${fmtTok(activeTokCount)} tokens</span>
                <span style="color:#4b5563;margin:0 0.4rem;">·</span>
                <span style="color:var(--text-muted);">vocab ${vocabSize != null ? vocabSize : '?'} (${bpeMode})</span>
            </div>

            <table style="width:100%;font-size:0.8rem;">
                <thead>
                    <tr>
                        <th>Stage</th>
                        <th>Model learns</th>
                        <th style="text-align:right;">Rows</th>
                        <th style="text-align:right;">Tokens</th>
                        <th style="text-align:right;">Size</th>
                        <th>Status</th>
                        <th>Operator action</th>
                    </tr>
                </thead>
                <tbody>${stageRows}</tbody>
            </table>

            <div style="background:#2a2a2a;padding:0.6rem 0.75rem;border-radius:4px;margin-top:0.75rem;font-size:0.78rem;">
                <div style="color:#ffb400;font-weight:600;margin-bottom:0.3rem;">Tokenizer Reality</div>
                <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:0.35rem;color:var(--text-muted);">
                    <span><strong>Vocab:</strong> ${vocabSize != null ? vocabSize : '—'} tokens (${bpeMode})</span>
                    <span><strong>Roundtrip:</strong> ${tokStatus} · exact=${tokExact}</span>
                    <span><strong>Density:</strong> ${tokRatio} tok/byte</span>
                    <span><strong>SVG structural:</strong> ~462 tokens</span>
                    <span><strong>English words:</strong> char-by-char (no word tokens)</span>
                </div>
                <div style="color:#6b7280;font-size:0.74rem;margin-top:0.35rem;">
                    English instruction prompts tokenize 3–5× more tokens per word than SVG primitives. SFT instruction following is feasible but context budget is expensive per English word.
                </div>
            </div>

            <div style="background:#2a2a2a;border-left:3px solid ${qualityGateColor};padding:0.45rem 0.75rem;border-radius:0 4px 4px 0;margin-top:0.55rem;font-size:0.78rem;">
                <strong style="color:${qualityGateColor};">Output Quality Gate</strong>
                <span style="color:var(--text-muted);margin-left:0.5rem;">
                    valid_svg=${validRate != null ? fmt(validRate, 3) : '—'} &nbsp;·&nbsp;
                    closure=${closureRate != null ? fmt(closureRate, 3) : '—'} &nbsp;·&nbsp;
                    loop=${loopScore != null ? fmt(loopScore, 3) : '—'}
                </span>
                <div style="color:#6b7280;font-size:0.74rem;margin-top:0.22rem;">
                    valid_svg=0 indicates repetition collapse or data coverage gap — not a kernel parity failure. Add Stage B + SFT pairs and re-evaluate.
                </div>
            </div>

            <div style="margin-top:0.75rem;">
                <button id="copyAnalysisPromptBtn" style="background:#323232;border:1px solid #ffb400;color:#ffb400;padding:0.3rem 0.75rem;border-radius:4px;cursor:pointer;font-size:0.78rem;font-family:inherit;">
                    📋 Copy Analysis Prompt
                </button>
                <span id="copyAnalysisPromptFeedback" style="color:#47b475;font-size:0.76rem;margin-left:0.5rem;display:none;">Copied!</span>
                <pre id="analysisPromptText" style="font-size:0.7rem;max-height:200px;overflow-y:auto;margin-top:0.45rem;padding:0.55rem 0.7rem;background:#1a1a1a;border-radius:4px;white-space:pre-wrap;border:1px solid #333;line-height:1.4;">${promptHtml}</pre>
            </div>
        </div>
    `;

    const btn = container.querySelector('#copyAnalysisPromptBtn');
    const fb = container.querySelector('#copyAnalysisPromptFeedback');
    if (btn) {
        btn.addEventListener('click', () => {
            navigator.clipboard.writeText(promptText).then(() => {
                if (fb) { fb.style.display = 'inline'; setTimeout(() => { fb.style.display = 'none'; }, 2000); }
            }).catch(() => {
                const pre = container.querySelector('#analysisPromptText');
                if (pre) {
                    const r = document.createRange();
                    r.selectNodeContents(pre);
                    const sel = window.getSelection();
                    sel.removeAllRanges();
                    sel.addRange(r);
                }
            });
        });
    }
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
    const stageTimeline = Array.isArray(pipeline.stage_timeline) ? pipeline.stage_timeline : [];
    const stageRanges = stageRangesFromTimeline(stageTimeline);
    const activeStage = normalizeStageName(pipeline.active_stage) || 'unassigned';
    const lossWithStage = lossSteps.map((p) => ({ ...p, _stage: resolvePointStage(p, stageRanges, activeStage) }));
    const stagesPresent = Array.from(new Set(lossWithStage.map((p) => normalizeStageName(p._stage)).filter(Boolean)));
    const stageFilterKey = `trainStageFilters:${runCtx.runDir || runCtx.modelPath || 'default'}`;
    const savedFilters = window._ckTrainStageFilters && typeof window._ckTrainStageFilters === 'object'
        ? window._ckTrainStageFilters[stageFilterKey]
        : null;
    const activeFilters = (savedFilters instanceof Set && savedFilters.size > 0)
        ? new Set(Array.from(savedFilters).filter((s) => stagesPresent.includes(s)))
        : new Set(stagesPresent);
    if (activeFilters.size === 0) stagesPresent.forEach((s) => activeFilters.add(s));
    const filteredLossSteps = lossWithStage.filter((p) => activeFilters.has(normalizeStageName(p._stage)));
    const chartLossSteps = filteredLossSteps.length > 0 ? filteredLossSteps : lossWithStage;
    const stageBoundaries = buildStageBoundaries(chartLossSteps);
    const stageFilterHtml = stagesPresent.map((stage) => {
        const selected = activeFilters.has(stage);
        const clr = STAGE_COLOR[stage] || STAGE_COLOR.unassigned;
        const bg = selected ? `${clr}22` : 'rgba(255,255,255,0.04)';
        const border = selected ? clr : 'rgba(255,255,255,0.14)';
        const txt = selected ? '#e5e7eb' : '#9ca3af';
        return `<button type="button" data-train-stage-filter="${stage}" style="background:${bg};border:1px solid ${border};color:${txt};padding:0.12rem 0.5rem;border-radius:999px;font-size:0.72rem;cursor:pointer;">${stage}</button>`;
    }).join('');
    const saturation = computeSaturation(lossWithStage, activeStage, 20);
    const saturationHtml = saturation
        ? `<div class="parity-section" style="margin-top:0.8rem;">
            <h3><span class="badge badge-blue">Saturation</span> Active Stage Trend</h3>
            <div style="display:flex;gap:0.6rem;align-items:center;flex-wrap:wrap;">
                <span style="background:${saturation.color}22;border:1px solid ${saturation.color};color:${saturation.color};padding:0.08rem 0.45rem;border-radius:4px;font-size:0.74rem;font-weight:700;">${saturation.verdict}</span>
                <span style="color:var(--text-muted);font-size:0.78rem;">${saturation.text}</span>
            </div>
            <div style="color:#9ca3af;font-size:0.74rem;margin-top:0.35rem;">
                stage=${saturation.stage} · slope=${fmt(saturation.slope, 6)} · std=${fmt(saturation.std, 4)} · window=${saturation.n} steps
            </div>
        </div>`
        : '';

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
        <div style="margin-top:0.6rem;margin-bottom:0.4rem;display:flex;gap:0.35rem;align-items:center;flex-wrap:wrap;">
            <span style="color:#9ca3af;font-size:0.72rem;">Stage filters:</span>
            <button type="button" data-train-stage-filter="__all__" style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.18);color:#d1d5db;padding:0.12rem 0.45rem;border-radius:999px;font-size:0.72rem;cursor:pointer;">all</button>
            ${stageFilterHtml}
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
        <div style="color:var(--text-muted);font-size:0.76rem;margin-top:0.15rem;">
            Grad norm = global L2 norm of parameter gradients before optimizer update. If this line is flat at 0.00e+0 for all steps, gradients may be truly collapsed or grad telemetry may be missing in this runtime path; confirm using loss trend and parity/replay gates.
        </div>
        ${saturationHtml}
        <div class="parity-section" style="margin-top:0.8rem;">
            <h3><span class="badge badge-orange">Health</span> Training Alerts</h3>
            <div id="trainAlertList"></div>
        </div>
        <div class="parity-section" style="margin-top:0.8rem;">
            <h3><span class="badge badge-blue">Sweep</span> Epoch Stability Summary</h3>
            <div id="trainSweepTable"></div>
        </div>
        <div id="trainDataPipelineSection"></div>
        <div class="parity-section" style="margin-top:0.8rem;">
            <h3><span class="badge badge-orange">Runbook</span> Operator Commands (Copy/Paste)</h3>
            <p style="color:var(--text-muted);margin-bottom:0.6rem;">Producer = CLI writes run_dir artifacts, viewer consumes run_dir.</p>
            <pre style="font-size:0.8rem;white-space:pre-wrap;">${reportCmd}</pre>
            <pre style="font-size:0.8rem;white-space:pre-wrap;">${suiteCmd}</pre>
        </div>
    `;

    drawLineChart(document.getElementById('trainLossChart'), chartLossSteps, 'loss_ck', '#f87171', 'loss_pt', {
        title: 'Training Loss: CK vs PyTorch',
        boundaries: stageBoundaries,
    });
    drawLineChart(document.getElementById('trainGradChart'), chartLossSteps, 'grad_norm', '#fbbf24', null, {
        title: 'Training Gradient Norm vs Step',
        boundaries: stageBoundaries,
    });
    drawLineChart(document.getElementById('trainLrChart'), chartLossSteps, 'lr', '#60a5fa', null, {
        title: 'Training Learning Rate vs Step',
        boundaries: stageBoundaries,
    });

    const filterButtons = root.querySelectorAll('button[data-train-stage-filter]');
    filterButtons.forEach((btn) => {
        btn.addEventListener('click', () => {
            const stage = String(btn.getAttribute('data-train-stage-filter') || '');
            const next = new Set(activeFilters);
            if (stage === '__all__') {
                stagesPresent.forEach((s) => next.add(s));
            } else if (next.has(stage)) {
                next.delete(stage);
            } else {
                next.add(stage);
            }
            if (next.size === 0) stagesPresent.forEach((s) => next.add(s));
            if (!window._ckTrainStageFilters || typeof window._ckTrainStageFilters !== 'object') {
                window._ckTrainStageFilters = {};
            }
            window._ckTrainStageFilters[stageFilterKey] = next;
            renderTrainingDashboard(files);
        });
    });

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
        renderTrainingDataPipelineSection(document.getElementById('trainDataPipelineSection'), pipeline, dataLab, postEval, lossSteps);
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

    renderTrainingDataPipelineSection(document.getElementById('trainDataPipelineSection'), pipeline, dataLab, postEval, lossSteps);
}
