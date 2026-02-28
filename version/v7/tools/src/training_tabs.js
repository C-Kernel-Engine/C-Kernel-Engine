import { clear, fmtExp, parseParityRows, toNum } from './utils.js';

function quoteShell(value) {
    const s = String(value || '');
    if (!s) return '';
    if (/^[A-Za-z0-9_./:-]+$/.test(s)) return s;
    return `"${s.replace(/"/g, '\\"')}"`;
}

function getRunContext() {
    const embedded = window.EMBEDDED_IR_DATA || {};
    const meta = embedded.meta || {};
    const runDir = typeof meta.run_dir === 'string' && meta.run_dir.trim()
        ? meta.run_dir
        : null;
    return { runDir };
}

function renderGradientPanel(files) {
    const root = document.getElementById('trainGradientRoot');
    if (!root) return;
    clear(root);

    const grad = files.training_grad_norms || {};
    const steps = Array.isArray(grad.steps) ? grad.steps : [];
    const globalNorms = Array.isArray(grad.global) ? grad.global : [];
    const params = grad.params && typeof grad.params === 'object' ? grad.params : {};
    const names = Object.keys(params);

    // No data at all — file missing or genuinely empty
    if (steps.length === 0 && names.length === 0 && globalNorms.length === 0) {
        root.innerHTML = '<div class="parity-section"><h3><span class="badge badge-orange">No Gradient Data</span></h3><p style="color:var(--text-muted);">training_grad_norms_latest.json not found or empty. This file is written automatically during training.</p></div>';
        return;
    }

    // Build point series for chart
    const globalSeries = steps.map((s, i) => ({
        step: toNum(s, i + 1),
        grad_norm: toNum(globalNorms[i], NaN),
    })).filter((p) => Number.isFinite(p.grad_norm));

    // Health stats
    const validNorms = globalSeries.map((p) => p.grad_norm);
    const lastNorm = validNorms.length > 0 ? validNorms[validNorms.length - 1] : NaN;
    const minNorm = validNorms.length > 0 ? Math.min(...validNorms) : NaN;
    const maxNorm = validNorms.length > 0 ? Math.max(...validNorms) : NaN;
    const meanNorm = validNorms.length > 0 ? validNorms.reduce((s, v) => s + v, 0) / validNorms.length : NaN;
    const allZero = validNorms.length > 0 && validNorms.every((v) => Math.abs(v) < 1e-12);

    let statusBadge, statusClass;
    if (allZero) {
        statusBadge = 'dead / no telemetry'; statusClass = 'grad-dead';
    } else if (!Number.isFinite(lastNorm)) {
        statusBadge = 'unknown'; statusClass = 'grad-warning';
    } else if (lastNorm > 2.0) {
        statusBadge = 'elevated'; statusClass = 'grad-warning';
    } else if (lastNorm < 1e-5) {
        statusBadge = 'vanishing'; statusClass = 'grad-warning';
    } else {
        statusBadge = 'healthy'; statusClass = 'grad-healthy';
    }

    // Per-param section (when available)
    let paramSection = '';
    if (names.length > 0) {
        const summary = names.map((name) => {
            const arr = Array.isArray(params[name]) ? params[name] : [];
            const last = arr.length > 0 ? toNum(arr[arr.length - 1], NaN) : NaN;
            return { name, last };
        }).sort((a, b) => Math.abs(b.last) - Math.abs(a.last));

        const paramRows = summary.slice(0, 40).map((s) => {
            let badge = '<span class="grad-healthy">healthy</span>';
            if (!Number.isFinite(s.last) || s.last < 1e-7) badge = '<span class="grad-dead">dead</span>';
            else if (s.last < 1e-5 || s.last > 0.1) badge = '<span class="grad-danger">danger</span>';
            else if (s.last < 1e-4 || s.last > 0.05) badge = '<span class="grad-warning">warning</span>';
            return `<tr><td>${s.name}</td><td>${fmtExp(s.last, 2)}</td><td>${badge}</td></tr>`;
        }).join('');

        paramSection = `
        <div class="parity-section" style="margin-top:0.8rem;">
            <h3><span class="badge badge-orange">Per-Parameter</span> Top ${Math.min(names.length, 40)} by Latest Norm</h3>
            <div style="color:var(--text-muted);margin-bottom:0.5rem;">params tracked: ${names.length}</div>
            <table>
                <thead><tr><th>Parameter</th><th>Latest Norm</th><th>Status</th></tr></thead>
                <tbody>${paramRows}</tbody>
            </table>
        </div>`;
    } else {
        paramSection = `
        <div style="background:#2a2a2a;border-left:3px solid #4b5563;padding:0.45rem 0.75rem;border-radius:0 4px 4px 0;margin-top:0.6rem;font-size:0.78rem;color:#6b7280;">
            Per-parameter breakdown not available — only global norm logged. Add <code>--grad-norms-per-param</code> to the training run to see per-layer gradient health.
        </div>`;
    }

    root.innerHTML = `
        <div class="stats-grid" style="grid-template-columns:repeat(auto-fit,minmax(130px,1fr));margin-bottom:0.8rem;">
            <div class="stat-card"><div class="stat-value"><span class="${statusClass}">${statusBadge}</span></div><div class="stat-label">Global Status</div></div>
            <div class="stat-card"><div class="stat-value">${Number.isFinite(lastNorm) ? fmtExp(lastNorm, 2) : '—'}</div><div class="stat-label">Latest Norm</div></div>
            <div class="stat-card"><div class="stat-value">${Number.isFinite(minNorm) ? fmtExp(minNorm, 2) : '—'}</div><div class="stat-label">Min</div></div>
            <div class="stat-card"><div class="stat-value">${Number.isFinite(maxNorm) ? fmtExp(maxNorm, 2) : '—'}</div><div class="stat-label">Max</div></div>
            <div class="stat-card"><div class="stat-value">${Number.isFinite(meanNorm) ? fmtExp(meanNorm, 2) : '—'}</div><div class="stat-label">Mean</div></div>
            <div class="stat-card"><div class="stat-value">${globalSeries.length}</div><div class="stat-label">Steps Logged</div></div>
        </div>
        <div class="parity-section">
            <h3><span class="badge badge-orange">Global Norm</span> L2 Gradient Norm vs Step</h3>
            <svg id="gradGlobalNormChart" style="width:100%;height:160px;"></svg>
            <button class="ck-svg-expand-btn" data-title="Global Gradient Norm vs Step">⛶ Expand</button>
            <div style="color:var(--text-muted);font-size:0.76rem;margin-top:0.3rem;">
                Global L2 norm of all parameter gradients before the optimizer step. Values 0.5–3.0 are typical for this model size. Spikes above 5 risk instability; sustained values near zero indicate vanishing gradients or missing telemetry.
            </div>
        </div>
        ${paramSection}
    `;

    // drawLineChart is defined in training_dashboard.js — accessible in the same bundle scope
    if (typeof drawLineChart === 'function' && globalSeries.length > 0) {
        drawLineChart(
            document.getElementById('gradGlobalNormChart'),
            globalSeries, 'grad_norm', '#fbbf24', null,
            { title: 'Global Gradient Norm vs Step' }
        );
    }
}

function renderParityPanel(files) {
    const root = document.getElementById('trainParityRoot');
    if (!root) return;
    clear(root);

    const rows = parseParityRows(files.training_parity);

    // Detect CK-only mode: parity file exists but worst_param = 'ck_only' means no PyTorch reference ran
    const allCkOnly = rows.length > 0 && rows.every((r) => String(r.worst_param || '').toLowerCase() === 'ck_only');
    if (allCkOnly) {
        const runCtx = getRunContext();
        const runDir = runCtx.runDir ? quoteShell(runCtx.runDir) : '<run_dir>';
        const parityCmd = runCtx.runDir
            ? `python3 version/v7/scripts/ck_run_v7.py parity --run ${runDir} --backend ck --parity-on --train-epochs 1 --train-seq-len 8 --train-total-tokens 512 --train-grad-accum 8 --no-train-save-final`
            : 'python3 version/v7/scripts/ck_run_v7.py parity --run <run_dir> --backend ck --parity-on --train-epochs 1 --train-seq-len 8 --train-total-tokens 512 --train-grad-accum 8 --no-train-save-final';
        root.innerHTML = `
            <div class="parity-section">
                <h3><span class="badge badge-blue">CK-Only Mode</span> No PyTorch Reference Available</h3>
                <p style="color:var(--text-muted);margin-bottom:0.6rem;">
                    This run was trained without a PyTorch reference process. The zeros in the parity file indicate
                    <strong style="color:var(--text-primary);">no comparison was performed</strong>, not that parity passed.
                    Run with <code>--parity-on</code> to enable CK vs PyTorch numerical validation.
                </p>
                <pre style="font-size:0.8rem;white-space:pre-wrap;">${parityCmd}</pre>
            </div>
        `;
        return;
    }

    if (rows.length === 0) {
        const runCtx = getRunContext();
        const runDir = runCtx.runDir ? quoteShell(runCtx.runDir) : '<run_dir>';
        const parityCmd = runCtx.runDir
            ? `python3 version/v7/scripts/ck_run_v7.py parity --run ${runDir} --backend ck --parity-on --train-epochs 1 --train-seq-len 8 --train-total-tokens 512 --train-grad-accum 8 --no-train-save-final`
            : 'python3 version/v7/scripts/ck_run_v7.py parity --run <run_dir> --backend ck --parity-on --train-epochs 1 --train-seq-len 8 --train-total-tokens 512 --train-grad-accum 8 --no-train-save-final';
        const reportCmd = runCtx.runDir
            ? `python3 version/v7/tools/open_ir_visualizer.py --generate --run ${runDir} --html-only --strict-run-artifacts --output ${runDir}/ir_report.html`
            : 'python3 version/v7/tools/open_ir_visualizer.py --generate --run <run_dir> --html-only --strict-run-artifacts --output <run_dir>/ir_report.html';
        root.innerHTML = `
            <div class="parity-section">
                <h3><span class="badge badge-orange">No Parity Data</span> Missing training_parity.json</h3>
                <p style="color:var(--text-muted);margin-bottom:0.6rem;">Generate parity telemetry in the same run directory, then regenerate this report.</p>
                <pre style="font-size:0.8rem;white-space:pre-wrap;">${parityCmd}
${reportCmd}</pre>
            </div>
        `;
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

function resolveDataLab(files) {
    const pipeline = files.training_pipeline && typeof files.training_pipeline === 'object'
        ? files.training_pipeline
        : {};
    const dataLab = pipeline.data_lab && typeof pipeline.data_lab === 'object'
        ? pipeline.data_lab
        : {};
    const provenance = Array.isArray(pipeline.data_provenance)
        ? pipeline.data_provenance.filter((row) => row && typeof row === 'object')
        : [];
    const firstProv = provenance.length > 0 ? provenance[0] : {};
    const tokLineage = pipeline.tokenizer_lineage && typeof pipeline.tokenizer_lineage === 'object'
        ? pipeline.tokenizer_lineage
        : {};
    if (!dataLab.dataset_path && typeof firstProv.source_path === 'string') dataLab.dataset_path = firstProv.source_path;
    if (!dataLab.dataset_dir && typeof dataLab.dataset_path === 'string' && dataLab.dataset_path.includes('/')) {
        dataLab.dataset_dir = dataLab.dataset_path.slice(0, dataLab.dataset_path.lastIndexOf('/'));
    }
    if (!dataLab.tokenizer_json_path && typeof tokLineage.tokenizer_path === 'string') dataLab.tokenizer_json_path = tokLineage.tokenizer_path;
    if (!dataLab.dataset_qc && files.dataset_qc) dataLab.dataset_qc = files.dataset_qc;
    if (!dataLab.dataset_profile && files.dataset_profile) dataLab.dataset_profile = files.dataset_profile;
    if (!dataLab.tokenizer_roundtrip && files.tokenizer_roundtrip) dataLab.tokenizer_roundtrip = files.tokenizer_roundtrip;
    if (!dataLab.post_train_eval && files.post_train_eval) dataLab.post_train_eval = files.post_train_eval;
    if (!dataLab.tokenizer_preview && files.tokenizer_preview) dataLab.tokenizer_preview = files.tokenizer_preview;
    return dataLab;
}

function htmlEscape(value) {
    return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
}

function stageTone(status) {
    const s = String(status || '').toLowerCase();
    if (s === 'active') return 'green';
    if (s === 'completed' || s === 'pass') return 'blue';
    if (s === 'missing' || s === 'fail') return 'red';
    return 'blue';
}

function renderDataLabPanel(files) {
    const root = document.getElementById('trainDataLabRoot');
    if (!root) return;
    clear(root);

    const pipeline = files.training_pipeline && typeof files.training_pipeline === 'object'
        ? files.training_pipeline
        : {};
    const dataLab = resolveDataLab(files);
    const qc = dataLab.dataset_qc && typeof dataLab.dataset_qc === 'object' ? dataLab.dataset_qc : {};
    const profile = dataLab.dataset_profile && typeof dataLab.dataset_profile === 'object' ? dataLab.dataset_profile : {};
    const roundtrip = dataLab.tokenizer_roundtrip && typeof dataLab.tokenizer_roundtrip === 'object' ? dataLab.tokenizer_roundtrip : {};
    const roundtripLines = roundtrip.line_eval && typeof roundtrip.line_eval === 'object' ? roundtrip.line_eval : {};
    const postEval = dataLab.post_train_eval && typeof dataLab.post_train_eval === 'object' ? dataLab.post_train_eval : {};
    const artifacts = dataLab.artifacts && typeof dataLab.artifacts === 'object' ? dataLab.artifacts : {};
    const tokenizerPreview = dataLab.tokenizer_preview && typeof dataLab.tokenizer_preview === 'object'
        ? dataLab.tokenizer_preview
        : (files.tokenizer_preview || {});
    const activeStage = typeof pipeline.active_stage === 'string' && pipeline.active_stage.trim()
        ? pipeline.active_stage
        : 'pretrain';
    const timeline = Array.isArray(pipeline.stage_timeline)
        ? pipeline.stage_timeline.filter((row) => row && typeof row === 'object')
        : [];
    const stageArtifacts = Array.isArray(pipeline.stage_artifacts)
        ? pipeline.stage_artifacts.filter((row) => row && typeof row === 'object')
        : [];
    const artifactByStage = {};
    for (const row of stageArtifacts) {
        const raw = String(row && row.stage ? row.stage : '').trim().toLowerCase();
        const stage = raw === 'stage_a' ? 'pretrain' : (raw === 'stage_b' ? 'midtrain' : raw);
        if (stage) artifactByStage[stage] = row;
    }
    const timelineByStage = {};
    for (const row of timeline) {
        const raw = String(row && row.stage ? row.stage : '').trim().toLowerCase();
        const stage = raw === 'stage_a' ? 'pretrain' : (raw === 'stage_b' ? 'midtrain' : raw);
        if (stage) timelineByStage[stage] = row;
    }
    const fallbackStageOrder = ['pretrain', 'midtrain', 'sft', 'dpo', 'grpo', 'ppo'];
    const runCtx = getRunContext();

    function pathHref(pathValue) {
        const raw = String(pathValue || '').trim();
        if (!raw || raw === '-') return '';
        if (/^(https?:\/\/|file:\/\/|\.{0,2}\/)/i.test(raw)) return raw;
        if (runCtx.runDir && raw.startsWith(`${runCtx.runDir}/`)) {
            const rel = raw.slice(runCtx.runDir.length + 1)
                .split('/')
                .map((seg) => encodeURIComponent(seg))
                .join('/');
            return `./${rel}`;
        }
        if (raw.startsWith('/')) {
            const fsUrl = window.CK_LIVE_MODE && typeof window.CK_LIVE_MODE === 'object'
                ? String(window.CK_LIVE_MODE.fsUrl || '').trim()
                : '';
            if (fsUrl) {
                const sep = fsUrl.includes('?') ? '&' : '?';
                return `${fsUrl}${sep}path=${encodeURIComponent(raw)}`;
            }
            // Absolute paths outside run_dir are not web-served in typical http mode.
            // Keep file:// only in local file mode; otherwise render as non-link.
            if (window.location && window.location.protocol === 'file:') return `file://${raw}`;
            return '';
        }
        return raw;
    }

    function pathCell(pathValue) {
        const raw = String(pathValue || '').trim();
        if (!raw || raw === '-') return '<code>-</code>';
        const href = pathHref(raw);
        if (!href) return `<span style="display:block;"><code>${htmlEscape(raw)}</code><span style="color:#6b7280;font-size:0.68rem;margin-left:0.3rem;">(not served by this report root)</span></span>`;
        const title = href.startsWith('file://')
            ? 'Local file path (may be blocked by browser when served over http)'
            : raw;
        return `<a href="${htmlEscape(href)}" target="_blank" rel="noopener" title="${htmlEscape(title)}" style="color:#7dd3fc;text-decoration:none;"><code>${htmlEscape(raw)}</code></a>`;
    }

    function pathCellCompact(pathValue, label = '') {
        const raw = String(pathValue || '').trim();
        if (!raw || raw === '-') return `<span style="display:block;color:#6b7280;"><span style="color:#9ca3af;">${htmlEscape(label)}:</span> -</span>`;
        const href = pathHref(raw);
        if (!href) {
            return `<span style="display:block;max-width:340px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;"><span style="color:#9ca3af;">${htmlEscape(label)}:</span> <code>${htmlEscape(raw)}</code> <span style="color:#6b7280;font-size:0.66rem;">(not served)</span></span>`;
        }
        const title = href.startsWith('file://')
            ? 'Local file path (may be blocked by browser when served over http)'
            : raw;
        return `<span style="display:block;max-width:340px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;" title="${htmlEscape(title)}"><span style="color:#9ca3af;">${htmlEscape(label)}:</span> <a href="${htmlEscape(href)}" target="_blank" rel="noopener" style="color:#7dd3fc;text-decoration:none;"><code>${htmlEscape(raw)}</code></a></span>`;
    }

    function stageOrderKey(row, idx) {
        if (!row || typeof row !== 'object') return idx + 1;
        const seq = Number(row.seq);
        if (Number.isFinite(seq)) return seq;
        const order = Number(row.order);
        if (Number.isFinite(order)) return order + 1;
        const index = Number(row.index);
        if (Number.isFinite(index)) return index + 1;
        return idx + 1;
    }

    // Per-stage lookup maps for clickable detail
    const provByStage = {};
    if (Array.isArray(pipeline.data_provenance)) {
        pipeline.data_provenance.forEach((p) => {
            const raw = String(p && p.stage ? p.stage : '').trim().toLowerCase();
            const stage = raw === 'stage_a' ? 'pretrain' : (raw === 'stage_b' ? 'midtrain' : raw);
            if (p && stage) provByStage[stage] = p;
        });
    }
    const catalogActiveByStage = {};
    if (Array.isArray(pipeline.dataset_catalog)) {
        pipeline.dataset_catalog.forEach((c) => {
            const raw = String(c && c.stage ? c.stage : '').trim().toLowerCase();
            const stage = raw === 'stage_a' ? 'pretrain' : (raw === 'stage_b' ? 'midtrain' : raw);
            if (c && c.kind === 'active_dataset' && stage) catalogActiveByStage[stage] = c;
        });
    }
    const stageBindingByStage = {};
    if (Array.isArray(pipeline.stage_dataset_bindings)) {
        pipeline.stage_dataset_bindings.forEach((row) => {
            const stage = normStageName(row && row.stage);
            if (stage) stageBindingByStage[stage] = row;
        });
    }
    function primaryBindingDataset(stage) {
        const row = stageBindingByStage[stage];
        if (!row || !Array.isArray(row.datasets) || row.datasets.length === 0) return null;
        const datasets = row.datasets.filter((d) => d && typeof d === 'object');
        if (datasets.length === 0) return null;
        return datasets.find((d) => d.status === 'ready') || datasets[0];
    }
    const tokLineage = pipeline.tokenizer_lineage && typeof pipeline.tokenizer_lineage === 'object'
        ? pipeline.tokenizer_lineage : {};
    const activeStageKey = normStageName(activeStage) || String(activeStage || 'pretrain');
    const pipelineContract = pipeline.pipeline && typeof pipeline.pipeline === 'object'
        ? pipeline.pipeline
        : {};
    const pipelineSchema = String(pipeline.schema || '').trim();
    const pipelineContractRows = Array.isArray(pipelineContract.stages)
        ? pipelineContract.stages
        : [];
    const manifestV2Rows = (pipelineSchema === 'ck-pipeline-manifest-v2' && Array.isArray(pipeline.pipeline))
        ? pipeline.pipeline
        : [];
    const strictStageContractRows = pipelineContractRows.length > 0 ? pipelineContractRows : manifestV2Rows;
    const strictManifestMode = strictStageContractRows.length > 0;
    const strictStageRowByStage = {};
    const tokCorpusEntry = catalogActiveByStage['pretrain'] || null;
    const tokCorpusName = strictManifestMode
        ? ((Array.isArray(tokLineage.tokenizer_corpora) && tokLineage.tokenizer_corpora.length > 0)
            ? tokLineage.tokenizer_corpora[0].name
            : null)
        : ((Array.isArray(tokLineage.corpus_datasets) && tokLineage.corpus_datasets.length > 0)
            ? tokLineage.corpus_datasets[0].name
            : (tokCorpusEntry ? tokCorpusEntry.name : null));
    const tokRtStatus = String((dataLab.tokenizer_roundtrip && dataLab.tokenizer_roundtrip.status) || '—').toUpperCase();

    function stageTokCoverage(stage) {
        if (strictManifestMode) {
            const strictRow = strictStageRowByStage[stage];
            const strictDatasets = strictRow && Array.isArray(strictRow.datasets)
                ? strictRow.datasets.filter((d) => d && typeof d === 'object')
                : [];
            const strictDs = strictDatasets.find((d) => d.status === 'ready') || strictDatasets[0] || null;
            if (strictDs && typeof strictDs.in_tokenizer_corpus === 'boolean') return strictDs.in_tokenizer_corpus;
            return null;
        }
        const bindDs = primaryBindingDataset(stage);
        if (bindDs && typeof bindDs.in_tokenizer_corpus === 'boolean') return bindDs.in_tokenizer_corpus;
        const bind = stageBindingByStage[stage];
        if (bind && bind.tokenizer_coverage && typeof bind.tokenizer_coverage === 'object') {
            const notIn = Number(bind.tokenizer_coverage.not_in_corpus);
            const inC = Number(bind.tokenizer_coverage.in_corpus);
            if (Number.isFinite(notIn) && Number.isFinite(inC)) {
                if (notIn > 0 && inC === 0) return false;
                if (inC > 0 && notIn === 0) return true;
            }
        }
        const cat = catalogActiveByStage[stage];
        if (cat && typeof cat.tokenizer_coverage === 'boolean') return cat.tokenizer_coverage;
        if (stage === 'pretrain') return true;
        const nm = String((cat && cat.name) || (artifactByStage[stage] && artifactByStage[stage].dataset_name) || '');
        if (!nm) return null;
        return (nm.includes('stage_a') && !nm.includes('stage_b')) ? true : false;
    }

    // Per-manifest tokenizer coverage: explicit field > corpus_datasets lookup > stage heuristic
    function tokManifestCoverage(row) {
        if (typeof row.tokenizer_coverage === 'boolean') return row.tokenizer_coverage ? 'yes' : 'no';
        if (Array.isArray(tokLineage.corpus_datasets) && tokLineage.corpus_datasets.length > 0) {
            const rn = String(row.name || '').toLowerCase();
            const found = tokLineage.corpus_datasets.some((c) => {
                const cn = String(c.name || c || '').toLowerCase();
                return cn && rn && cn.length > 4 && rn.includes(cn.slice(0, Math.min(cn.length, 20)));
            });
            return found ? 'yes' : 'no';
        }
        // Heuristic: pretrain manifests fed the tokenizer corpus
        const stage = normStageName(row && row.stage);
        if (stage === 'pretrain') return 'inferred';
        return 'unknown';
    }
    function tokManifestBadge(cov) {
        if (cov === 'yes') return '<span style="color:#6ee7b7;font-size:0.75rem;" title="Confirmed in tokenizer corpus">✓</span>';
        if (cov === 'no') return '<span style="color:#fb923c;font-size:0.75rem;" title="NOT in tokenizer corpus — tok gap risk if trained">✗</span>';
        if (cov === 'inferred') return '<span style="color:#9ca3af;font-size:0.75rem;" title="Inferred: pretrain manifests typically feed the tokenizer corpus. Add corpus_datasets to tokenizer_lineage to confirm.">~✓</span>';
        return '<span style="color:#4b5563;font-size:0.75rem;" title="Coverage unknown">?</span>';
    }

    function normStageName(value) {
        const s = String(value || '').trim().toLowerCase();
        if (!s) return '';
        if (s === 'stage_a') return 'pretrain';
        if (s === 'stage_b') return 'midtrain';
        return s;
    }

    const stageSequence = pipeline.stage_sequence && typeof pipeline.stage_sequence === 'object'
        ? pipeline.stage_sequence
        : {};
    const stageSequenceEntries = Array.isArray(stageSequence.entries)
        ? stageSequence.entries.filter((row) => row && typeof row === 'object')
        : [];
    const stageSeqMetaByStage = {};
    const orderedStages = [];
    const addOrderedStage = (rawStage, meta = null) => {
        const stage = normStageName(rawStage);
        if (!stage || orderedStages.includes(stage)) return;
        orderedStages.push(stage);
        if (meta && typeof meta === 'object') {
            stageSeqMetaByStage[stage] = meta;
        } else if (!stageSeqMetaByStage[stage]) {
            stageSeqMetaByStage[stage] = {};
        }
    };

    if (strictManifestMode) {
        const strictStages = strictStageContractRows
            .filter((row) => row && typeof row === 'object')
            .map((row, idx) => {
                const stage = normStageName(row.stage || row.name);
                if (!stage) return null;
                const type = String(row.type || '').toLowerCase();
                const isTrainingLike = type === 'training_stage' || type === 'training';
                if (!isTrainingLike) return null;
                const stageId = Number(row.stage_id);
                const seq = Number(row.seq);
                const key = Number.isFinite(stageId)
                    ? (stageId + 1)
                    : (Number.isFinite(seq) ? seq : (idx + 1));
                return { row, key, idx, stage };
            })
            .filter((x) => !!x)
            .sort((a, b) => (a.key - b.key) || (a.idx - b.idx));
        for (const item of strictStages) {
            strictStageRowByStage[item.stage] = item.row;
            addOrderedStage(item.stage, {
                seq: item.key,
                status: item.row.status,
                active: item.row.active === true || item.stage === activeStageKey,
                strict: true,
            });
        }
    } else if (stageSequenceEntries.length > 0) {
        const sorted = stageSequenceEntries
            .map((row, idx) => {
                const seq = Number(row.seq);
                const order = Number(row.order);
                const index = Number(row.index);
                const key = Number.isFinite(seq)
                    ? seq
                    : (Number.isFinite(order) ? order + 1 : (Number.isFinite(index) ? index + 1 : idx + 1));
                return { row, key, idx };
            })
            .sort((a, b) => (a.key - b.key) || (a.idx - b.idx));
        for (const item of sorted) {
            const row = item.row;
            addOrderedStage(row.stage, {
                seq: item.key,
                status: row.status,
                active: row.active === true,
            });
        }
    } else if (timeline.length > 0) {
        const timelineSorted = timeline
            .map((row, idx) => ({ row, key: stageOrderKey(row, idx), idx }))
            .sort((a, b) => (a.key - b.key) || (a.idx - b.idx));
        for (const item of timelineSorted) {
            const row = item.row;
            addOrderedStage(row.stage, {
                seq: item.key,
                status: row.status,
                active: row.active === true,
            });
        }
    }
    if (!strictManifestMode && !orderedStages.includes(activeStageKey)) addOrderedStage(activeStageKey);

    const stageLossHistory = pipeline.stage_loss_history && typeof pipeline.stage_loss_history === 'object'
        ? pipeline.stage_loss_history
        : {};
    const stageLossEntries = Array.isArray(stageLossHistory.entries)
        ? stageLossHistory.entries.filter((row) => row && typeof row === 'object')
        : [];
    const stageLossByStage = {};
    for (const entry of stageLossEntries) {
        const stage = normStageName(entry.stage);
        if (!stage) continue;
        if (!stageLossByStage[stage]) stageLossByStage[stage] = [];
        stageLossByStage[stage].push(entry);
    }
    for (const stage of Object.keys(stageLossByStage)) {
        stageLossByStage[stage].sort((a, b) => {
            const aa = String(a.ended_at || a.run_id || '');
            const bb = String(b.ended_at || b.run_id || '');
            return aa.localeCompare(bb);
        });
    }
    if (!strictManifestMode) {
        for (const s of fallbackStageOrder) {
            if (stageLossByStage[s] && !orderedStages.includes(s)) addOrderedStage(s);
        }
        for (const s of Object.keys(stageLossByStage)) {
            if (!orderedStages.includes(s)) addOrderedStage(s);
        }
    }
    if (!strictManifestMode && orderedStages.length === 0) {
        addOrderedStage('pretrain');
        addOrderedStage('midtrain');
        addOrderedStage('sft');
        addOrderedStage('dpo');
        addOrderedStage('grpo');
        addOrderedStage('ppo');
    }

    function fmtLoss(value) {
        const n = Number(value);
        if (!Number.isFinite(n)) return '—';
        if (Math.abs(n) >= 10) return n.toFixed(2);
        if (Math.abs(n) >= 1) return n.toFixed(3);
        return n.toFixed(4);
    }

    function stageLossSparkline(entry) {
        if (!entry || !Array.isArray(entry.points) || entry.points.length < 2) return '';
        const losses = entry.points
            .map((p) => Number(p && p.loss))
            .filter((v) => Number.isFinite(v));
        if (losses.length < 2) return '';
        const minV = Math.min(...losses);
        const maxV = Math.max(...losses);
        const span = Math.max(maxV - minV, 1e-9);
        const width = 120;
        const height = 24;
        const pts = losses.map((v, i) => {
            const x = losses.length === 1 ? 0 : (i / (losses.length - 1)) * width;
            const y = ((maxV - v) / span) * (height - 2) + 1;
            return `${x.toFixed(1)},${y.toFixed(1)}`;
        }).join(' ');
        return `<svg viewBox="0 0 ${width} ${height}" style="width:100%;height:24px;margin-top:0.18rem;display:block;">
            <polyline points="${pts}" fill="none" stroke="#fbbf24" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"></polyline>
        </svg>`;
    }

    const stageCardMeta = {};
    const flowCards = orderedStages.map((stage, idx) => {
        const seqMeta = stageSeqMetaByStage[stage] && typeof stageSeqMetaByStage[stage] === 'object'
            ? stageSeqMetaByStage[stage]
            : {};
        const strictRow = strictManifestMode && strictStageRowByStage[stage] && typeof strictStageRowByStage[stage] === 'object'
            ? strictStageRowByStage[stage]
            : null;
        const t = timelineByStage[stage] || {};
        const status = strictManifestMode
            ? String((strictRow && strictRow.status) || (stage === activeStageKey ? 'active' : 'planned'))
            : String(t.status || seqMeta.status || (stage === activeStageKey ? 'active' : 'planned'));
        const isActive = strictManifestMode
            ? Boolean((strictRow && strictRow.active === true) || stage === activeStageKey || status === 'active')
            : Boolean(t.active === true || seqMeta.active === true || stage === activeStageKey);
        const tone = stageTone(status);
        const chip = statusPill(status, tone);
        const border = isActive ? '2px solid rgba(71,180,117,0.6)' : '1px solid rgba(255,255,255,0.10)';
        const bg = isActive ? 'rgba(71,180,117,0.09)' : 'rgba(255,255,255,0.03)';
        const data = artifactByStage[stage] && typeof artifactByStage[stage] === 'object' ? artifactByStage[stage] : {};
        const arts = Array.isArray(data.artifacts) ? data.artifacts : [];
        const cat = catalogActiveByStage[stage] || {};
        const prov = provByStage[stage] || {};
        const bindDs = strictManifestMode
            ? (() => {
                const rows = strictRow && Array.isArray(strictRow.datasets)
                    ? strictRow.datasets.filter((d) => d && typeof d === 'object')
                    : [];
                return rows.find((d) => d.status === 'ready') || rows[0] || {};
            })()
            : (primaryBindingDataset(stage) || {});
        const lossRuns = stageLossByStage[normStageName(stage)] || [];
        const latestLossRun = lossRuns.length > 0 ? lossRuns[lossRuns.length - 1] : null;
        const dsName = strictManifestMode
            ? (bindDs.name || null)
            : (bindDs.name || data.dataset_name || cat.name || prov.dataset_name || (latestLossRun && latestLossRun.dataset_name) || null);
        const covered = stageTokCoverage(stage);
        const hasLoss = latestLossRun && Number.isFinite(Number(latestLossRun.first_loss)) && Number.isFinite(Number(latestLossRun.final_loss));
        const dropPct = hasLoss ? Number(latestLossRun.drop_pct) : NaN;

        stageCardMeta[stage] = {
            status, isActive, dsName,
            dsRows: strictManifestMode ? (bindDs.rows || null) : (bindDs.rows || cat.rows || null),
            dsTok: strictManifestMode ? (bindDs.tokens || null) : (bindDs.tokens || data.token_count || prov.token_count || (latestLossRun && latestLossRun.total_tokens) || null),
            byteSize: strictManifestMode ? (bindDs.bytes || null) : (bindDs.bytes || prov.byte_size || null),
            sourcePath: strictManifestMode ? (bindDs.path || null) : (bindDs.path || data.source_path || prov.source_path || cat.path || (latestLossRun && latestLossRun.source_path) || null),
            rawSourcePath: strictManifestMode ? (bindDs.path || null) : (bindDs.path || data.source_path || prov.source_path || cat.path || (latestLossRun && latestLossRun.raw_source_path) || null),
            tokenStreamPath: (latestLossRun && latestLossRun.token_stream_path) || null,
            covered, artifacts: arts,
            catNote: strictManifestMode ? null : (cat.note || null),
            lossRuns,
            latestLossRun,
            seq: Number.isFinite(Number(seqMeta.seq)) ? Number(seqMeta.seq) : (idx + 1),
        };

        const short = dsName ? (dsName.length > 24 ? dsName.slice(0, 22) + '…' : dsName) : null;
        const dsPill = short
            ? `<div title="${htmlEscape(dsName)}" style="margin-top:0.28rem;background:${isActive ? 'rgba(71,180,117,0.12)' : 'rgba(255,255,255,0.05)'};border:1px solid ${isActive ? 'rgba(71,180,117,0.28)' : 'rgba(255,255,255,0.10)'};border-radius:4px;padding:0.12rem 0.38rem;font-size:0.65rem;color:var(--text-muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">${htmlEscape(short)}</div>`
            : `<div style="margin-top:0.28rem;font-size:0.65rem;color:#4b5563;font-style:italic;">${status === 'planned' ? 'not assigned' : '—'}</div>`;

        const covBadge = covered === false
            ? `<div style="margin-top:0.2rem;display:inline-block;background:rgba(234,88,12,0.12);border:1px solid rgba(234,88,12,0.3);border-radius:3px;padding:0.07rem 0.28rem;font-size:0.62rem;color:#fb923c;">⚠ tok gap</div>`
            : (covered === true && dsName
                ? `<div style="margin-top:0.2rem;display:inline-block;background:rgba(71,180,117,0.08);border:1px solid rgba(71,180,117,0.2);border-radius:3px;padding:0.07rem 0.28rem;font-size:0.62rem;color:#6ee7b7;">✓ tok ok</div>`
                : '');
        const lossSummary = hasLoss
            ? `<div style="margin-top:0.22rem;font-size:0.64rem;color:#fcd34d;">loss ${fmtLoss(latestLossRun.first_loss)} → ${fmtLoss(latestLossRun.final_loss)}${Number.isFinite(dropPct) ? ` · Δ ${dropPct.toFixed(1)}%` : ''}</div>${stageLossSparkline(latestLossRun)}`
            : `<div style="margin-top:0.22rem;font-size:0.62rem;color:#4b5563;">loss curve not recorded</div>`;

        return `
            <div data-dp-stage="${htmlEscape(stage)}" style="min-width:145px;max-width:190px;flex:0 0 auto;padding:0.5rem 0.6rem;border-radius:10px;border:${border};background:${bg};cursor:pointer;user-select:none;" title="Click to inspect ${htmlEscape(stage)} datasets &amp; artifacts">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:0.2rem;">
                    <div style="font-weight:700;font-size:0.88rem;">${htmlEscape(stage)}</div>
                    <div style="font-size:0.62rem;color:#6b7280;white-space:nowrap;">seq ${Number.isFinite(Number(seqMeta.seq)) ? Number(seqMeta.seq) : (idx + 1)}/${orderedStages.length}</div>
                </div>
                <div style="margin-top:0.22rem;">${chip}</div>
                ${dsPill}
                ${covBadge}
                ${lossSummary}
                <div style="margin-top:0.25rem;color:#4b5563;font-size:0.61rem;">${arts.length} artifact${arts.length !== 1 ? 's' : ''} · tap ↓</div>
            </div>
        `;
    }).join('<div style="align-self:center;color:var(--text-muted);font-size:0.75rem;padding:0 0.05rem;">→</div>');

    const trainDims = pipeline.train_dims && typeof pipeline.train_dims === 'object' ? pipeline.train_dims : {};
    const dimsSummary = [
        `layers=${trainDims.num_layers ?? '-'}`,
        `embed_dim=${trainDims.embed_dim ?? '-'}`,
        `hidden_dim=${trainDims.hidden_dim ?? '-'}`,
        `vocab=${trainDims.vocab_size ?? '-'}`,
        `heads=${trainDims.num_heads ?? '-'}`,
        `ctx=${trainDims.context_length ?? '-'}`,
    ].join(' | ');

    const tokSourcePanel = `
        <div style="background:#192019;border:1px solid rgba(71,180,117,0.22);border-radius:8px;padding:0.55rem 0.75rem;margin-bottom:0.7rem;">
            <div style="display:flex;align-items:center;gap:0.4rem;flex-wrap:wrap;margin-bottom:0.38rem;">
                <span style="font-size:0.66rem;color:#6b7280;font-weight:700;letter-spacing:0.07em;">TOKENIZER</span>
                <span class="badge badge-green">${htmlEscape(tokLineage.type || 'unknown')}</span>
                <span style="background:rgba(255,255,255,0.07);border-radius:4px;padding:0.08rem 0.38rem;font-size:0.72rem;color:var(--text-muted);">vocab: ${tokLineage.vocab_size ?? '?'}</span>
                ${tokLineage.reused_run_tokenizer ? '<span style="background:rgba(7,173,248,0.08);border:1px solid rgba(7,173,248,0.22);border-radius:4px;padding:0.08rem 0.38rem;font-size:0.68rem;color:#7dd3fc;">↗ reused</span>' : ''}
                <span style="background:rgba(255,255,255,0.06);border-radius:4px;padding:0.08rem 0.38rem;font-size:0.7rem;color:var(--text-muted);">roundtrip: ${htmlEscape(tokRtStatus)}</span>
            </div>
            <div style="font-size:0.75rem;margin-bottom:0.28rem;display:flex;align-items:center;gap:0.4rem;flex-wrap:wrap;">
                <span style="color:#6b7280;">Trained on:</span>
                ${tokCorpusName
                    ? `<span style="background:rgba(71,180,117,0.12);border:1px solid rgba(71,180,117,0.25);border-radius:4px;padding:0.1rem 0.45rem;font-size:0.7rem;color:#6ee7b7;" title="${htmlEscape(tokCorpusName)}">${htmlEscape(tokCorpusName.length > 44 ? tokCorpusName.slice(0, 42) + '…' : tokCorpusName)}</span>`
                    : '<span style="color:#4b5563;font-size:0.71rem;">corpus not recorded · add <code>corpus_datasets</code> to tokenizer_lineage</span>'
                }
            </div>
            <div style="font-size:0.7rem;color:#6b7280;">⚠ Stages using datasets outside this corpus may encounter novel token boundaries — check ⚠ tok gap badges above.</div>
            ${strictManifestMode ? '<div style="font-size:0.7rem;color:#6ee7b7;margin-top:0.2rem;">strict contract mode: stage/data mapping from pipeline contract only (fallbacks disabled)</div>' : ''}
        </div>
    `;

    const stageFlowSection = `
        <div class="parity-section">
            <h3><span class="badge badge-orange">Stage Flow</span> ${htmlEscape(orderedStages.join(' -> '))}</h3>
            ${tokSourcePanel}
            <div style="display:flex;gap:0.4rem;flex-wrap:wrap;align-items:stretch;">${flowCards}</div>
            <div id="dpStageDetail" style="display:none;margin-top:0.55rem;"></div>
            <div style="margin-top:0.45rem;color:var(--text-muted);font-size:0.8rem;">
                active_stage=${htmlEscape(activeStageKey)} | model_dims: <code>${htmlEscape(dimsSummary)}</code>
            </div>
        </div>
    `;
    window._ckDpStageCardMeta = stageCardMeta;

    function fmtPlanInt(value) {
        const n = Number(value);
        return Number.isFinite(n) ? Math.round(n).toLocaleString() : '—';
    }
    function fmtPlanPct(value) {
        const n = Number(value);
        return Number.isFinite(n) ? `${n >= 0 ? '↓' : '↑'} ${Math.abs(n).toFixed(1)}%` : '—';
    }

    const exec = pipeline.execution && typeof pipeline.execution === 'object' ? pipeline.execution : {};
    const execSeqLen = Number(exec.seq_len);
    const stagePlanRows = orderedStages.map((stage) => {
        const meta = stageCardMeta[stage] || {};
        const latest = meta.latestLossRun && typeof meta.latestLossRun === 'object' ? meta.latestLossRun : null;
        const runs = Array.isArray(meta.lossRuns) ? meta.lossRuns.length : 0;
        const seqLenRaw = latest && Number.isFinite(Number(latest.seq_len))
            ? Number(latest.seq_len)
            : (meta.isActive && Number.isFinite(execSeqLen) ? execSeqLen : NaN);
        const tokenRaw = Number.isFinite(Number(meta.dsTok))
            ? Number(meta.dsTok)
            : (latest && Number.isFinite(Number(latest.total_tokens)) ? Number(latest.total_tokens) : NaN);
        const estSteps = Number.isFinite(tokenRaw) && Number.isFinite(seqLenRaw) && seqLenRaw > 0
            ? Math.ceil(tokenRaw / seqLenRaw)
            : NaN;
        const latestSteps = latest && Number.isFinite(Number(latest.steps)) ? Number(latest.steps) : NaN;
        const stageLooksComplete = Number.isFinite(latestSteps) && Number.isFinite(estSteps) && estSteps > 0 && latestSteps >= estSteps;

        let trainState = 'planned';
        let stateColor = '#60a5fa';
        if (stage === 'unassigned') {
            trainState = 'legacy-unlabeled';
            stateColor = '#fbbf24';
        } else if (meta.isActive || String(meta.status || '').toLowerCase() === 'active') {
            if (stageLooksComplete) {
                trainState = 'completed(last run)';
                stateColor = '#60a5fa';
            } else {
            trainState = 'active';
            stateColor = '#6ee7b7';
            }
        } else if (runs > 0 || String(meta.status || '').toLowerCase() === 'completed') {
            trainState = 'trained';
            stateColor = '#60a5fa';
        } else if (meta.dsName) {
            trainState = 'ready';
            stateColor = '#c084fc';
        }

        const coverage = meta.covered === true ? 'tok ok' : (meta.covered === false ? 'tok gap' : 'unknown');
        const coverageColor = meta.covered === true ? '#6ee7b7' : (meta.covered === false ? '#fb923c' : '#9ca3af');
        const lossCell = latest
            ? `${fmtLoss(latest.first_loss)} → ${fmtLoss(latest.final_loss)} <span style="color:#9ca3af;">(${fmtPlanPct(latest.drop_pct)})</span>`
            : '—';
        const rawPath = meta.rawSourcePath || (latest && latest.raw_source_path) || (latest && latest.source_path) || meta.sourcePath || '-';
        const tokenPath = meta.tokenStreamPath || (latest && latest.token_stream_path) || '-';
        const pathsCell = (tokenPath !== '-' && tokenPath !== rawPath)
            ? `${pathCellCompact(rawPath, 'raw')}${pathCellCompact(tokenPath, 'tok')}`
            : pathCellCompact(rawPath, 'path');

        return `
            <tr>
                <td>${htmlEscape(stage)}</td>
                <td>${htmlEscape(meta.seq ?? '-')}</td>
                <td><span style="color:${stateColor};font-weight:700;">${htmlEscape(trainState)}</span></td>
                <td>${htmlEscape(meta.dsName || 'not assigned')}</td>
                <td style="text-align:right;">${fmtPlanInt(meta.dsRows)}</td>
                <td style="text-align:right;">${fmtPlanInt(tokenRaw)}</td>
                <td style="text-align:right;">${fmtPlanInt(seqLenRaw)}</td>
                <td style="text-align:right;">${fmtPlanInt(estSteps)}</td>
                <td style="text-align:right;">${fmtPlanInt(runs)}</td>
                <td>${lossCell}</td>
                <td><span style="color:${coverageColor};font-weight:700;">${htmlEscape(coverage)}</span></td>
                <td style="max-width:360px;overflow-wrap:anywhere;">${pathsCell}</td>
            </tr>
        `;
    }).join('');

    const stageLedgerRows = stageLossEntries
        .slice()
        .sort((a, b) => {
            const aa = String(a.ended_at || a.run_id || '');
            const bb = String(b.ended_at || b.run_id || '');
            return aa.localeCompare(bb);
        })
        .slice(-120)
        .map((entry) => {
            const stage = normStageName(entry.stage) || 'unassigned';
            const dsName = String(entry.dataset_name || (entry.path ? String(entry.path).split('/').pop() : '') || '-');
            const srcPath = String(entry.source_path || entry.path || '-');
            const rawPath = String(entry.raw_source_path || (entry.token_stream_path ? '-' : srcPath) || '-');
            const tokPath = String(entry.token_stream_path || '-');
            const steps = Number.isFinite(Number(entry.steps)) ? Math.round(Number(entry.steps)).toLocaleString() : '—';
            const tokens = Number.isFinite(Number(entry.total_tokens)) ? Math.round(Number(entry.total_tokens)).toLocaleString() : '—';
            const seqLen = Number.isFinite(Number(entry.seq_len)) ? Math.round(Number(entry.seq_len)).toLocaleString() : '—';
            const firstLoss = Number.isFinite(Number(entry.first_loss)) ? fmtLoss(entry.first_loss) : '—';
            const finalLoss = Number.isFinite(Number(entry.final_loss)) ? fmtLoss(entry.final_loss) : '—';
            const when = entry.ended_at ? String(entry.ended_at).replace('T', ' ').replace('+00:00', 'Z') : '-';
            const prov = String(entry.dataset_provenance || '-');
            return `
                <tr>
                    <td>${htmlEscape(stage)}</td>
                    <td>${htmlEscape(String(entry.run_id || '-'))}</td>
                    <td>${htmlEscape(when)}</td>
                    <td>${htmlEscape(dsName)}</td>
                    <td style="text-align:right;">${seqLen}</td>
                    <td style="text-align:right;">${tokens}</td>
                    <td style="text-align:right;">${steps}</td>
                    <td>${firstLoss} → ${finalLoss}</td>
                    <td>${htmlEscape(prov)}</td>
                    <td style="max-width:360px;overflow-wrap:anywhere;">${pathCellCompact(rawPath, 'raw')}${tokPath && tokPath !== '-' ? pathCellCompact(tokPath, 'tok') : ''}</td>
                </tr>
            `;
        })
        .join('');

    const stagePlanSection = `
        <div class="parity-section" style="margin-top:0.8rem;">
            <h3><span class="badge badge-blue">Stage Plan</span> Dataset Training Plan & Progress</h3>
            <div style="font-size:0.74rem;color:#6b7280;margin-bottom:0.5rem;">
                Per-stage status from sequence + artifacts + historical loss runs. Estimated steps/epoch = ceil(tokens / seq_len).
            </div>
            <div style="overflow-x:auto;">
                <table style="font-size:0.78rem;">
                    <thead>
                        <tr>
                            <th>Stage</th>
                            <th>Seq</th>
                            <th>Status</th>
                            <th>Dataset</th>
                            <th style="text-align:right;">Rows</th>
                            <th style="text-align:right;">Tokens</th>
                            <th style="text-align:right;">Seq Len</th>
                            <th style="text-align:right;">Est Steps/Epoch</th>
                            <th style="text-align:right;">Runs</th>
                            <th>Latest Loss</th>
                            <th>Tokenizer</th>
                            <th>Data Paths</th>
                        </tr>
                    </thead>
                    <tbody>${stagePlanRows}</tbody>
                </table>
            </div>
            <details style="margin-top:0.6rem;">
                <summary style="cursor:pointer;font-size:0.76rem;color:#9ca3af;">Run-level ledger (latest ${Math.min(stageLossEntries.length, 120)} stage runs)</summary>
                ${stageLedgerRows
                    ? `<div style="margin-top:0.45rem;overflow-x:auto;">
                        <table style="font-size:0.75rem;">
                            <thead>
                                <tr>
                                    <th>Stage</th>
                                    <th>Run ID</th>
                                    <th>When</th>
                                    <th>Dataset</th>
                                    <th style="text-align:right;">Seq</th>
                                    <th style="text-align:right;">Tokens</th>
                                    <th style="text-align:right;">Steps</th>
                                    <th>Loss</th>
                                    <th>Provenance</th>
                                    <th>Data Paths</th>
                                </tr>
                            </thead>
                            <tbody>${stageLedgerRows}</tbody>
                        </table>
                    </div>`
                    : '<div style="margin-top:0.35rem;color:var(--text-muted);">No stage loss runs recorded yet.</div>'
                }
            </details>
        </div>
    `;

    const datasetCatalog = Array.isArray(pipeline.dataset_catalog)
        ? pipeline.dataset_catalog.filter((row) => row && typeof row === 'object')
        : [];
    const corpusSampling = files.corpus_sampling_log && typeof files.corpus_sampling_log === 'object'
        ? files.corpus_sampling_log
        : {};
    const samplingEpochs = Array.isArray(corpusSampling.epochs)
        ? corpusSampling.epochs.filter((row) => row && typeof row === 'object')
        : [];

    function normPath(value) {
        const raw = String(value || '').trim();
        if (!raw) return '';
        return raw.replace(/\\/g, '/').replace(/\/+/g, '/').toLowerCase();
    }

    function normDatasetId(value) {
        const raw = String(value || '').trim().toLowerCase();
        return raw || '';
    }

    const samplingByPath = {};
    const samplingByDatasetId = {};
    for (const epochRow of samplingEpochs) {
        const epochNum = Number(epochRow.epoch);
        const datasets = Array.isArray(epochRow.datasets) ? epochRow.datasets : [];
        for (const ds of datasets) {
            if (!ds || typeof ds !== 'object') continue;
            const rowsSampled = Number(ds.rows_sampled);
            const rowsTotal = Number(ds.rows_total);
            const pathKey = normPath(ds.path);
            const idKey = normDatasetId(ds.dataset_id);
            const slot = {
                epoch: Number.isFinite(epochNum) ? epochNum : null,
                rows_sampled: Number.isFinite(rowsSampled) ? rowsSampled : null,
                rows_total: Number.isFinite(rowsTotal) ? rowsTotal : null,
                coverage_pct: Number(ds.coverage_pct),
            };
            if (pathKey) samplingByPath[pathKey] = slot;
            if (idKey) samplingByDatasetId[idKey] = slot;
        }
    }

    function coverageCell(row) {
        const pathKey = normPath(row && row.path);
        const idKey = normDatasetId(row && row.name ? String(row.name).split('.')[0] : '');
        const hit = (pathKey && samplingByPath[pathKey]) || (idKey && samplingByDatasetId[idKey]) || null;
        if (!hit) {
            return '<span style="color:#6b7280;">✗ Not sampled</span>';
        }
        const ep = hit.epoch != null ? `Ep ${Math.round(hit.epoch)}` : 'ep ?';
        const rs = Number.isFinite(hit.rows_sampled) ? Math.round(hit.rows_sampled).toLocaleString() : '?';
        const rt = Number.isFinite(hit.rows_total) ? Math.round(hit.rows_total).toLocaleString() : '?';
        const cp = Number.isFinite(hit.coverage_pct)
            ? `${Math.max(0, Math.min(100, hit.coverage_pct)).toFixed(1)}%`
            : (Number.isFinite(hit.rows_sampled) && Number.isFinite(hit.rows_total) && hit.rows_total > 0
                ? `${((hit.rows_sampled / hit.rows_total) * 100).toFixed(1)}%`
                : null);
        if (cp && Number(cp.replace('%', '')) >= 99.9) {
            return `<span style="color:#6ee7b7;">✓ ${ep} · ${rs}/${rt} (${cp})</span>`;
        }
        if (cp) {
            return `<span style="color:#fbbf24;">~ ${ep} · ${rs}/${rt} (${cp})</span>`;
        }
        return `<span style="color:#93c5fd;">~ ${ep} · sampled ${rs}</span>`;
    }

    // Split catalog into three tiers by kind
    const trainedEntries   = datasetCatalog.filter((r) => r.kind === 'active_dataset');
    const generatedEntries = datasetCatalog.filter((r) => r.kind === 'generated_dataset');
    const manifestEntries  = datasetCatalog.filter((r) => r.kind !== 'active_dataset' && r.kind !== 'generated_dataset');

    function catalogRow(row, highlight) {
        const rowStyle = highlight
            ? 'border-left:3px solid #47b475;background:rgba(71,180,117,0.06);'
            : '';
        return `<tr style="${rowStyle}">
            <td>${htmlEscape(row.stage ?? '-')}</td>
            <td>${htmlEscape(row.kind ?? '-')}</td>
            <td>${htmlEscape(row.name ?? '-')}</td>
            <td style="text-align:right;">${htmlEscape(row.rows ?? '-')}</td>
            <td>${coverageCell(row)}</td>
            <td style="max-width:380px;overflow-wrap:anywhere;font-size:0.74rem;">${pathCell(row.path ?? '-')}</td>
            <td style="max-width:240px;overflow-wrap:anywhere;">${htmlEscape(row.note ?? '-')}</td>
        </tr>`;
    }

    const trainedSection = trainedEntries.length > 0 ? `
        <div style="margin-bottom:0.75rem;">
            <div style="font-size:0.78rem;color:#47b475;font-weight:600;margin-bottom:0.35rem;letter-spacing:0.04em;">
                ▶ TRAINED ON (${trainedEntries.length} dataset${trainedEntries.length > 1 ? 's' : ''})
            </div>
            <table style="font-size:0.8rem;">
                <thead><tr><th>Stage</th><th>Kind</th><th>Name</th><th style="text-align:right;">Rows</th><th>Coverage</th><th>Path</th><th>Note</th></tr></thead>
                <tbody>${trainedEntries.map((r) => catalogRow(r, true)).join('')}</tbody>
            </table>
        </div>
    ` : `
        <div style="background:#2a2a2a;border-left:3px solid #4b5563;padding:0.45rem 0.75rem;border-radius:0 4px 4px 0;margin-bottom:0.75rem;font-size:0.78rem;color:#6b7280;">
            No <code>active_dataset</code> entry found — cannot determine what this run trained on.
        </div>
    `;

    const generatedSection = generatedEntries.length > 0 ? `
        <div style="margin-bottom:0.75rem;">
            <div style="font-size:0.78rem;color:#ffb400;font-weight:600;margin-bottom:0.35rem;letter-spacing:0.04em;">
                ◈ GENERATED (${generatedEntries.length} file${generatedEntries.length > 1 ? 's' : ''} — available, not trained)
            </div>
            <table style="font-size:0.8rem;">
                <thead><tr><th>Stage</th><th>Kind</th><th>Name</th><th style="text-align:right;">Rows</th><th>Coverage</th><th>Path</th><th>Note</th></tr></thead>
                <tbody>${generatedEntries.map((r) => catalogRow(r, false)).join('')}</tbody>
            </table>
        </div>
    ` : '';

    const manifestSection = manifestEntries.length > 0 ? `
        <details style="margin-bottom:0.5rem;">
            <summary style="cursor:pointer;font-size:0.78rem;color:#6b7280;font-weight:600;letter-spacing:0.04em;user-select:none;padding:0.3rem 0;">
                ◦ CORPUS MANIFESTS (${manifestEntries.length} entries — the dataset library, not what was trained)
            </summary>
            <div style="background:rgba(255,255,255,0.02);border-left:2px solid rgba(255,255,255,0.07);border-radius:0 4px 4px 0;padding:0.45rem 0.65rem;margin:0.4rem 0 0.55rem 0;font-size:0.74rem;color:#9ca3af;">
                <strong style="color:#d1d5db;">What are these?</strong> Each manifest describes a raw corpus dataset that exists in your data library — source files, row counts, symbol coverage.
                They were assembled into the active training packs shown above (green rows).<br>
                <strong style="color:#d1d5db;">Tok?</strong> column: was this dataset in the tokenizer training corpus?
                ✓ = confirmed · ~✓ = inferred (pretrain-stage) · ✗ = not in corpus (tok gap risk if this data is trained on later) · ? = not tracked (add <code>corpus_datasets</code> to <code>tokenizer_lineage</code> in pipeline JSON to get exact coverage).<br>
                <strong style="color:#d1d5db;">Operator use:</strong> verify all expected corpus files are registered, plan which to include in future stages, flag any ✗ entries before using them in midtrain/SFT.
            </div>
            <div style="margin-top:0.2rem;overflow-x:auto;">
                <table style="font-size:0.76rem;">
                    <thead><tr><th>Stage</th><th title="Was this dataset in the tokenizer training corpus?">Tok?</th><th>Name</th><th style="text-align:right;">Rows</th><th>Note</th></tr></thead>
                    <tbody>
                        ${manifestEntries.slice(0, 300).map((row) => {
                            const cov = tokManifestCoverage(row);
                            const rowStyle = cov === 'no' ? 'background:rgba(234,88,12,0.05);' : '';
                            return `<tr style="${rowStyle}">
                                <td>${htmlEscape(row.stage ?? '-')}</td>
                                <td style="text-align:center;">${tokManifestBadge(cov)}</td>
                                <td>${htmlEscape(row.name ?? '-')}</td>
                                <td style="text-align:right;">${htmlEscape(row.rows ?? '-')}</td>
                                <td style="max-width:260px;overflow-wrap:anywhere;">${htmlEscape(row.note ?? '-')}</td>
                            </tr>`;
                        }).join('')}
                    </tbody>
                </table>
            </div>
        </details>
    ` : '';

    const datasetCatalogSection = `
        <div class="parity-section" style="margin-top:0.8rem;">
            <h3><span class="badge badge-blue">Dataset Catalog</span> What Was Trained vs Available</h3>
            <div style="font-size:0.74rem;color:#6b7280;margin-bottom:0.5rem;">Green rows = actively trained on this run · Amber = generated but not trained · Grey (collapsed) = manifests and registry entries.</div>
            ${datasetCatalog.length > 0
                ? `<div style="color:var(--text-muted);font-size:0.78rem;margin-bottom:0.65rem;">${datasetCatalog.length} total entries — ${trainedEntries.length} trained · ${generatedEntries.length} generated · ${manifestEntries.length} manifests</div>
                   ${trainedSection}${generatedSection}${manifestSection}`
                : '<div style="color:var(--text-muted);">No dataset_catalog embedded yet. Regenerate report to derive from manifests.</div>'
            }
        </div>
    `;

    const pathRows = [
        ['dataset_dir', dataLab.dataset_dir || qc.dataset_dir || profile.dataset_dir || '-'],
        ['dataset_path', dataLab.dataset_path || qc.path || profile.path || '-'],
        ['tokenizer_json_path', dataLab.tokenizer_json_path || roundtrip.tokenizer_json_path || tokenizerPreview.path || '-'],
        ['dataset_qc_json', artifacts.dataset_qc_json || '-'],
        ['dataset_profile_json', artifacts.dataset_profile_json || '-'],
        ['tokenizer_roundtrip_json', artifacts.tokenizer_roundtrip_json || '-'],
        ['post_train_eval_json', artifacts.post_train_eval_json || '-'],
    ];
    const pathTable = `
        <table>
            <thead><tr><th>Path</th><th>Value</th></tr></thead>
            <tbody>
                ${pathRows.map((r) => `
                    <tr>
                        <td>${htmlEscape(r[0])}</td>
                        <td style="max-width:620px;overflow-wrap:anywhere;">${pathCell(r[1])}</td>
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;

    const profileTopChars = Array.isArray(profile.top_chars) ? profile.top_chars.slice(0, 8) : [];
    const topCharsLine = profileTopChars.length > 0
        ? profileTopChars.map((c) => `${c.char}:${c.count}`).join(', ')
        : '-';
    const profileDuplicates = profile.duplicates && typeof profile.duplicates === 'object' ? profile.duplicates : {};
    const lineLength = profile.line_length && typeof profile.line_length === 'object' ? profile.line_length : {};
    const lineAvg = Number.isFinite(Number(lineLength.avg)) ? Number(lineLength.avg).toFixed(2) : '-';
    const lineMin = lineLength.min !== undefined ? lineLength.min : '-';
    const lineMax = lineLength.max !== undefined ? lineLength.max : '-';

    const sampleRows = Array.isArray(roundtrip.sample_rows) ? roundtrip.sample_rows.slice(0, 16) : [];
    const sampleTable = sampleRows.length > 0
        ? `
            <table>
                <thead><tr><th>Line</th><th>Exact</th><th>Tokens</th><th>Token IDs</th><th>Decoded Preview</th></tr></thead>
                <tbody>
                    ${sampleRows.map((row) => `
                        <tr>
                            <td>${htmlEscape(row.line_no ?? '-')}</td>
                            <td>${row.exact_match ? '<span class="badge badge-green">yes</span>' : '<span class="badge badge-orange">no</span>'}</td>
                            <td>${htmlEscape(row.token_count ?? '-')}</td>
                            <td style="max-width:280px;overflow-wrap:anywhere;"><code>${htmlEscape(Array.isArray(row.token_ids) ? row.token_ids.join(' ') : '-')}</code></td>
                            <td style="max-width:380px;overflow-wrap:anywhere;"><code>${htmlEscape(row.decoded ?? '-')}</code></td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        `
        : '<div style="color:var(--text-muted);">No row-level tokenizer roundtrip samples.</div>';

    const tokenizerStatsRows = [
        ['status', tokenizerPreview.status || '-'],
        ['model_type', tokenizerPreview.model_type || '-'],
        ['vocab_size', tokenizerPreview.vocab_size ?? '-'],
        ['merge_count', tokenizerPreview.merge_count ?? '-'],
        ['pre_tokenizer', tokenizerPreview.pre_tokenizer_type || '-'],
        ['decoder', tokenizerPreview.decoder_type || '-'],
        ['bytelevel_mode', tokenizerPreview.bytelevel_mode === true ? 'true' : (tokenizerPreview.bytelevel_mode === false ? 'false' : '-')],
        ['ascii_piece_count', tokenizerPreview.ascii_piece_count ?? '-'],
        ['non_ascii_piece_count', tokenizerPreview.non_ascii_piece_count ?? '-'],
    ];
    const mergeRows = Array.isArray(tokenizerPreview.merge_samples) ? tokenizerPreview.merge_samples.slice(0, 24) : [];
    const vocabRows = Array.isArray(tokenizerPreview.vocab_samples) ? tokenizerPreview.vocab_samples.slice(0, 24) : [];
    const flowRows = tokenizerPreview.encode_decode_example && Array.isArray(tokenizerPreview.encode_decode_example.token_flow)
        ? tokenizerPreview.encode_decode_example.token_flow.slice(0, 32)
        : [];
    const tokVocabMismatch = tokenizerPreview.vocab_size != null && trainDims.vocab_size != null
        && Number(tokenizerPreview.vocab_size) !== Number(trainDims.vocab_size);
    const tokenizerSection = tokenizerPreview && Object.keys(tokenizerPreview).length > 0
        ? `
            <div class="parity-section" style="margin-top:0.8rem;">
                <h3><span class="badge badge-blue">Tokenizer Internals</span> Merges + Piece Map + Decode Behavior</h3>
                <div style="background:rgba(255,255,255,0.02);border-left:2px solid rgba(7,173,248,0.2);border-radius:0 4px 4px 0;padding:0.4rem 0.65rem;margin-bottom:0.55rem;font-size:0.73rem;color:#9ca3af;">
                    <strong style="color:#d1d5db;">This section vs the Data &amp; Tokenizer tab:</strong>
                    Here you see a static audit of the <code>tokenizer.json</code> artifact — what BPE merges were learned, what vocab pieces exist, and how a sample string encodes.
                    The <em>Data &amp; Tokenizer</em> tab shows live runtime tokenizer behavior used during inference.
                    Both reference the same file; this view focuses on training-data fidelity.<br>
                    <strong style="color:#d1d5db;">Key checks:</strong>
                    vocab_size here should equal your model config (${trainDims.vocab_size ?? '?'} declared).
                    ${tokVocabMismatch ? '<span style="color:#fb923c;font-weight:600;">MISMATCH: tokenizer vocab_size (' + (tokenizerPreview.vocab_size ?? '?') + ') ≠ model config (' + (trainDims.vocab_size ?? '?') + ') — embedding matrix size will be wrong.</span>' : (tokenizerPreview.vocab_size != null ? '<span style="color:#6ee7b7;">vocab_size matches model config.</span>' : '')}
                    ascii_piece_count = vocab_size means 100% ASCII vocabulary coverage — ideal for SVG data.
                </div>
                <table>
                    <thead><tr><th>Field</th><th>Value</th></tr></thead>
                    <tbody>
                        ${tokenizerStatsRows.map((r) => `
                            <tr>
                                <td>${htmlEscape(r[0])}</td>
                                <td style="max-width:620px;overflow-wrap:anywhere;"><code>${htmlEscape(r[1])}</code></td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
                <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:0.7rem;margin-top:0.7rem;">
                    <div>
                        <strong>Merges (top)</strong>
                        ${mergeRows.length > 0 ? `
                            <table>
                                <thead><tr><th>Rank</th><th>left</th><th>right</th><th>merged</th></tr></thead>
                                <tbody>
                                    ${mergeRows.map((m) => `
                                        <tr>
                                            <td>${htmlEscape(m.rank ?? '-')}</td>
                                            <td><code>${htmlEscape(m.left ?? '-')}</code></td>
                                            <td><code>${htmlEscape(m.right ?? '-')}</code></td>
                                            <td><code>${htmlEscape(m.merged_hint ?? '-')}</code></td>
                                        </tr>
                                    `).join('')}
                                </tbody>
                            </table>
                        ` : '<div style="color:var(--text-muted);">No merge samples (likely non-BPE tokenizer JSON).</div>'}
                    </div>
                    <div>
                        <strong>Vocab Pieces (by ID)</strong>
                        ${vocabRows.length > 0 ? `
                            <table>
                                <thead><tr><th>id</th><th>piece</th><th>bytes</th></tr></thead>
                                <tbody>
                                    ${vocabRows.map((v) => `
                                        <tr>
                                            <td>${htmlEscape(v.id ?? '-')}</td>
                                            <td><code>${htmlEscape(v.piece ?? '-')}</code></td>
                                            <td><code>${htmlEscape(v.bytes_hex ?? '-')}</code></td>
                                        </tr>
                                    `).join('')}
                                </tbody>
                            </table>
                        ` : '<div style="color:var(--text-muted);">No vocab samples available.</div>'}
                    </div>
                </div>
                <div style="margin-top:0.7rem;">
                    <strong>Encode/Decode Walkthrough</strong>
                    ${flowRows.length > 0 ? `
                        <div style="margin-top:0.3rem;"><code>source:</code> <code>${htmlEscape(tokenizerPreview.encode_decode_example.source ?? '-')}</code></div>
                        <div style="margin-top:0.2rem;"><code>decoded:</code> <code>${htmlEscape(tokenizerPreview.encode_decode_example.decoded ?? '-')}</code></div>
                        <div style="margin-top:0.2rem;">exact_match: ${tokenizerPreview.encode_decode_example.exact_match ? '<span class="badge badge-green">yes</span>' : '<span class="badge badge-orange">no</span>'}</div>
                        <table style="margin-top:0.4rem;">
                            <thead><tr><th>token_id</th><th>piece</th></tr></thead>
                            <tbody>
                                ${flowRows.map((f) => `
                                    <tr>
                                        <td>${htmlEscape(f.id ?? '-')}</td>
                                        <td><code>${htmlEscape(f.piece ?? '-')}</code></td>
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    ` : '<div style="color:var(--text-muted);">No token-flow example yet. Generate roundtrip artifacts to populate this automatically.</div>'}
                </div>
            </div>
        `
        : `
            <div class="parity-section" style="margin-top:0.8rem;">
                <h3><span class="badge badge-blue">Tokenizer Internals</span> Merges + Piece Map + Decode Behavior</h3>
                <div style="color:var(--text-muted);">Tokenizer preview unavailable. Regenerate report after ensuring <code>tokenizer.json</code> is present in the run/model directory.</div>
            </div>
        `;

    root.innerHTML = `
        <div style="background:rgba(71,180,117,0.05);border:1px solid rgba(71,180,117,0.17);border-radius:8px;padding:0.55rem 0.85rem;margin-bottom:0.75rem;">
            <div style="display:flex;align-items:center;gap:0.5rem;flex-wrap:wrap;margin-bottom:0.35rem;">
                <span style="font-size:0.82rem;font-weight:700;color:#6ee7b7;">Data Lab</span>
                <span style="font-size:0.74rem;color:var(--text-muted);">Operator view — what the model trained on, tokenizer health, and output quality gates</span>
            </div>
            <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:0.28rem 0.65rem;">
                <div style="font-size:0.71rem;color:#9ca3af;"><span style="color:#6ee7b7;font-weight:600;">Stage Flow</span> — click a card to inspect datasets &amp; artifacts per stage</div>
                <div style="font-size:0.71rem;color:#9ca3af;"><span style="color:#fb923c;font-weight:600;">⚠ tok gap</span> — stage dataset was outside the tokenizer training corpus</div>
                <div style="font-size:0.71rem;color:#9ca3af;"><span style="color:#6ee7b7;font-weight:600;">Dataset Catalog</span> — trained (green) · generated (amber) · manifests (collapsed)</div>
                <div style="font-size:0.71rem;color:#9ca3af;"><span style="color:#6ee7b7;font-weight:600;">Sample Browser</span> — row-level tokenizer roundtrip and exact-match rate</div>
            </div>
        </div>
        ${stageFlowSection}
        ${stagePlanSection}
        <div class="parity-section">
            <h3><span class="badge badge-blue">Paths</span> Dataset + Tokenizer Sources</h3>
            <div style="font-size:0.74rem;color:#6b7280;margin-bottom:0.5rem;">Verify correct dataset and tokenizer files were loaded for this run.</div>
            ${pathTable}
        </div>
        ${datasetCatalogSection}
        <div class="stats-grid" style="margin-top:0.8rem;">
            <div class="stat-card"><div class="stat-value">${qc.status || '-'}</div><div class="stat-label">QC Status</div></div>
            <div class="stat-card"><div class="stat-value">${qc.non_empty_lines ?? profile.non_empty_lines ?? '-'}</div><div class="stat-label">Non-empty Rows</div></div>
            <div class="stat-card"><div class="stat-value">${profileDuplicates.duplicate_unique_rows ?? '-'}</div><div class="stat-label">Duplicate Unique Rows</div></div>
            <div class="stat-card"><div class="stat-value">${roundtrip.exact_match === true ? 'yes' : (roundtrip.exact_match === false ? 'no' : '-')}</div><div class="stat-label">Roundtrip Exact</div></div>
            <div class="stat-card"><div class="stat-value">${roundtripLines.exact_match_rate !== undefined ? Number(roundtripLines.exact_match_rate).toFixed(4) : '-'}</div><div class="stat-label">Line Match Rate</div></div>
            <div class="stat-card"><div class="stat-value">${postEval.valid_svg_rate !== undefined ? Number(postEval.valid_svg_rate).toFixed(4) : '-'}</div><div class="stat-label">Valid SVG Rate</div></div>
            <div class="stat-card"><div class="stat-value">${postEval.closure_success_rate !== undefined ? Number(postEval.closure_success_rate).toFixed(4) : '-'}</div><div class="stat-label">Closure Rate</div></div>
            <div class="stat-card"><div class="stat-value">${postEval.repetition_loop_score !== undefined ? Number(postEval.repetition_loop_score).toFixed(4) : '-'}</div><div class="stat-label">Loop Score</div></div>
        </div>
        <div class="parity-section" style="margin-top:0.8rem;">
            <h3><span class="badge badge-orange">Profile</span> Dataset Shape Snapshot</h3>
            <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:0.5rem;">
                <div><strong>Line avg/min/max:</strong> ${lineAvg} / ${lineMin} / ${lineMax}</div>
                <div><strong>Total chars:</strong> ${profile.total_chars ?? '-'}</div>
                <div><strong>Top chars:</strong> ${htmlEscape(topCharsLine)}</div>
            </div>
        </div>
        <div class="parity-section" style="margin-top:0.8rem;">
            <h3><span class="badge badge-green">Sample Browser</span> Row → Token IDs → Decoded</h3>
            <div style="background:rgba(255,255,255,0.02);border-left:2px solid rgba(71,180,117,0.2);border-radius:0 4px 4px 0;padding:0.4rem 0.65rem;margin-bottom:0.55rem;font-size:0.73rem;color:#9ca3af;">
                Each row: training line → token IDs the model sees → decoded back to text.
                <strong style="color:#d1d5db;">All Exact=yes</strong> → tokenizer is healthy, training data round-trips cleanly.
                <strong style="color:#d1d5db;">Any Exact=no</strong> → that row's decoded text differs from the original; the model trains on a subtly wrong signal — investigate which characters fail to round-trip (likely non-ASCII, BOMs, or unusual whitespace).
                <strong style="color:#d1d5db;">Token count</strong> → high token-per-character ratio (e.g. 242 tokens for a multi-element SVG) is expected; watch for single-line rows that exceed your context length (${trainDims.context_length ?? '?'} tokens).
            </div>
            <div style="overflow:auto;">${sampleTable}</div>
        </div>
        ${tokenizerSection}
    `;

    // --- Stage card click handlers (post-render) ---
    {
        const cards = root.querySelectorAll('[data-dp-stage]');
        const detailBox = root.querySelector('#dpStageDetail');
        let openStage = null;

        function fmtNum(n) {
            return n != null && Number.isFinite(Number(n)) ? Number(n).toLocaleString() : '—';
        }
        function fmtBytes(b) {
            if (b == null) return '—';
            const n = Number(b);
            if (!Number.isFinite(n)) return String(b);
            if (n >= 1e9) return (n / 1e9).toFixed(2) + ' GB';
            if (n >= 1e6) return (n / 1e6).toFixed(2) + ' MB';
            if (n >= 1e3) return (n / 1e3).toFixed(1) + ' KB';
            return n + ' B';
        }

        function closeStageDetail() {
            if (!detailBox) return;
            detailBox.style.display = 'none';
            detailBox.innerHTML = '';
            openStage = null;
            window._ckDpStageOpen = null;
            cards.forEach((c) => { c.style.outline = ''; });
        }

        function buildStageChartSeries(meta) {
            const runs = Array.isArray(meta?.lossRuns) ? meta.lossRuns : [];
            const latestRun = runs.length > 0 ? runs[runs.length - 1] : null;
            const previousRun = runs.length > 1 ? runs[runs.length - 2] : null;
            const latestPointsRaw = Array.isArray(latestRun?.points) ? latestRun.points : [];
            const previousPointsRaw = Array.isArray(previousRun?.points) ? previousRun.points : [];

            const previousByStep = new Map();
            previousPointsRaw.forEach((p, i) => {
                const step = toNum(p && p.step, i + 1);
                const loss = toNum(p && p.loss, NaN);
                if (Number.isFinite(step) && Number.isFinite(loss)) previousByStep.set(step, loss);
            });

            const points = latestPointsRaw.map((p, i) => {
                const step = toNum(p && p.step, i + 1);
                const loss = toNum(p && p.loss, NaN);
                return {
                    step,
                    loss,
                    loss_prev: previousByStep.has(step) ? previousByStep.get(step) : NaN,
                };
            }).filter((p) => Number.isFinite(p.step) && Number.isFinite(p.loss));
            points.sort((a, b) => a.step - b.step);

            return {
                points,
                hasPrevious: points.some((p) => Number.isFinite(p.loss_prev)),
                latestRunId: latestRun ? String(latestRun.run_id || latestRun.source || 'run') : '',
                previousRunId: previousRun ? String(previousRun.run_id || previousRun.source || 'run') : '',
            };
        }

        function renderStageDetail(stage, shouldScroll = true) {
            const meta = (window._ckDpStageCardMeta || {})[stage];
            if (!meta || !detailBox) return;

            openStage = stage;
            window._ckDpStageOpen = stage;
            cards.forEach((c) => {
                c.style.outline = c.dataset.dpStage === stage ? '2px solid rgba(71,180,117,0.75)' : '';
            });

            const covColor = meta.covered === true ? '#6ee7b7' : (meta.covered === false ? '#fb923c' : '#9ca3af');
            const covMark = meta.covered === true ? '✓' : (meta.covered === false ? '⚠' : '?');
            const covLabel = meta.covered === true
                ? 'Tokenizer corpus covers this dataset — no novel boundaries expected'
                : (meta.covered === false
                    ? 'Tok gap — dataset outside tokenizer corpus; watch for boundary artifacts'
                    : 'Coverage not recorded — add corpus_datasets to tokenizer_lineage');
            const latestRun = meta.latestLossRun && typeof meta.latestLossRun === 'object' ? meta.latestLossRun : null;
            const hasLatestLoss = latestRun && Number.isFinite(Number(latestRun.first_loss)) && Number.isFinite(Number(latestRun.final_loss));
            const runCount = Array.isArray(meta.lossRuns) ? meta.lossRuns.length : 0;
            const latestLossLabel = hasLatestLoss
                ? `${fmtLoss(latestRun.first_loss)} → ${fmtLoss(latestRun.final_loss)}`
                : 'not recorded';

            const artRows = meta.artifacts.length > 0
                ? meta.artifacts.map((a) => {
                    const rawPath = typeof a === 'string' ? a : (a.path || a.name || '');
                    if (rawPath) {
                        return `<div style="font-family:monospace;font-size:0.73rem;color:var(--text-muted);word-break:break-all;padding:0.1rem 0;">${pathCell(rawPath)}</div>`;
                    }
                    return `<div style="font-family:monospace;font-size:0.73rem;color:var(--text-muted);word-break:break-all;padding:0.1rem 0;">${htmlEscape(JSON.stringify(a))}</div>`;
                }).join('')
                : '<div style="font-size:0.76rem;color:#4b5563;font-style:italic;">No artifacts recorded for this stage yet.</div>';
            const lossRunsRows = runCount > 0
                ? meta.lossRuns.map((run) => {
                    const runId = String(run.run_id || run.source || 'run');
                    const start = fmtLoss(run.first_loss);
                    const end = fmtLoss(run.final_loss);
                    const steps = Number.isFinite(Number(run.steps)) ? Number(run.steps).toLocaleString() : '—';
                    const lr = Number.isFinite(Number(run.lr)) ? Number(run.lr).toExponential(1) : '—';
                    const seq = Number.isFinite(Number(run.seq_len)) ? Number(run.seq_len) : '—';
                    return `<tr>
                        <td style="max-width:190px;overflow-wrap:anywhere;"><code>${htmlEscape(runId)}</code></td>
                        <td>${start} → ${end}</td>
                        <td>${steps}</td>
                        <td>${seq}</td>
                        <td>${lr}</td>
                    </tr>`;
                }).join('')
                : '';

            const chartSeries = buildStageChartSeries(meta);
            const chartId = `dpStageLossChart_${String(stage).replace(/[^a-zA-Z0-9_-]/g, '_')}`;
            const chartTitle = `Stage ${stage} Loss vs Step`;
            const chartLegend = chartSeries.hasPrevious
                ? `latest run: <code>${htmlEscape(chartSeries.latestRunId)}</code> · baseline: <code>${htmlEscape(chartSeries.previousRunId)}</code> (dashed)`
                : `latest run: <code>${htmlEscape(chartSeries.latestRunId || 'run')}</code>`;
            const chartPanel = chartSeries.points.length > 1
                ? `
                    <div style="margin:0.1rem 0 0.5rem 0;border:1px solid rgba(255,255,255,0.10);border-radius:7px;padding:0.45rem 0.5rem;background:rgba(255,255,255,0.02);">
                        <div style="display:flex;justify-content:space-between;align-items:center;gap:0.5rem;margin-bottom:0.2rem;">
                            <div style="font-size:0.66rem;color:#6b7280;font-weight:700;letter-spacing:0.07em;">STAGE LOSS CURVE</div>
                            <button class="ck-svg-expand-btn" data-title="${htmlEscape(chartTitle)}">⛶ Expand</button>
                        </div>
                        <svg id="${chartId}" style="width:100%;height:210px;"></svg>
                        <div style="font-size:0.72rem;color:#9ca3af;margin-top:0.25rem;">${chartLegend}</div>
                    </div>
                `
                : '';

            detailBox.innerHTML = `
                <div style="border:1px solid rgba(71,180,117,0.25);border-radius:8px;background:rgba(8,18,10,0.88);padding:0.65rem 0.85rem;">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem;">
                        <div style="font-weight:700;font-size:0.9rem;color:#e5e7eb;">${htmlEscape(stage)}</div>
                        <button id="dpStageDetailClose" style="background:transparent;border:none;color:#6b7280;font-size:1.2rem;cursor:pointer;line-height:1;padding:0 0.2rem;" title="Close">×</button>
                    </div>
                    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(185px,1fr));gap:0.5rem;margin-bottom:0.5rem;">
                        <div style="background:rgba(255,255,255,0.04);border-radius:6px;padding:0.45rem 0.6rem;">
                            <div style="font-size:0.65rem;color:#6b7280;font-weight:700;letter-spacing:0.07em;margin-bottom:0.2rem;">DATASET</div>
                            <div style="font-size:0.8rem;word-break:break-all;">${meta.dsName ? htmlEscape(meta.dsName) : '<span style="color:#4b5563;font-style:italic;">not assigned</span>'}</div>
                        </div>
                        <div style="background:rgba(255,255,255,0.04);border-radius:6px;padding:0.45rem 0.6rem;">
                            <div style="font-size:0.65rem;color:#6b7280;font-weight:700;letter-spacing:0.07em;margin-bottom:0.2rem;">DATA PATHS</div>
                            <div style="font-size:0.74rem;word-break:break-all;">
                                ${pathCellCompact(meta.rawSourcePath || (latestRun && latestRun.raw_source_path) || meta.sourcePath || (latestRun && latestRun.source_path) || '-', 'raw')}
                                ${(meta.tokenStreamPath || (latestRun && latestRun.token_stream_path))
                                    ? pathCellCompact(meta.tokenStreamPath || latestRun.token_stream_path, 'tok')
                                    : ''
                                }
                            </div>
                            <div style="font-size:0.68rem;color:#6b7280;margin-top:0.15rem;">${htmlEscape((latestRun && latestRun.dataset_provenance) ? String(latestRun.dataset_provenance) : 'stage_artifacts/data_provenance')}</div>
                        </div>
                        <div style="background:rgba(255,255,255,0.04);border-radius:6px;padding:0.45rem 0.6rem;">
                            <div style="font-size:0.65rem;color:#6b7280;font-weight:700;letter-spacing:0.07em;margin-bottom:0.2rem;">SHAPE</div>
                            <div style="font-size:0.78rem;">rows: ${fmtNum(meta.dsRows)}</div>
                            <div style="font-size:0.78rem;">tokens: ${fmtNum(meta.dsTok)}</div>
                            <div style="font-size:0.78rem;">bytes: ${fmtBytes(meta.byteSize)}</div>
                        </div>
                        <div style="background:rgba(255,255,255,0.04);border-radius:6px;padding:0.45rem 0.6rem;">
                            <div style="font-size:0.65rem;color:#6b7280;font-weight:700;letter-spacing:0.07em;margin-bottom:0.2rem;">TOKENIZER COVERAGE</div>
                            <div style="font-size:0.78rem;color:${covColor};">${covMark} ${covLabel}</div>
                        </div>
                        <div style="background:rgba(255,255,255,0.04);border-radius:6px;padding:0.45rem 0.6rem;">
                            <div style="font-size:0.65rem;color:#6b7280;font-weight:700;letter-spacing:0.07em;margin-bottom:0.2rem;">TRAINING LOSS</div>
                            <div style="font-size:0.78rem;color:#fcd34d;">${latestLossLabel}</div>
                            <div style="font-size:0.72rem;color:#9ca3af;">runs: ${runCount}</div>
                        </div>
                    </div>
                    ${latestRun ? `<div style="margin:-0.05rem 0 0.45rem 0;">${stageLossSparkline(latestRun)}</div>` : ''}
                    ${chartPanel}
                    ${runCount > 0 ? `
                        <div style="font-size:0.65rem;color:#6b7280;font-weight:700;letter-spacing:0.07em;margin-bottom:0.22rem;">LOSS RUNS (${runCount})</div>
                        <div style="overflow:auto;max-height:180px;margin-bottom:0.45rem;">
                            <table style="font-size:0.74rem;">
                                <thead><tr><th>Run</th><th>Loss (start → end)</th><th>Steps</th><th>Seq</th><th>LR</th></tr></thead>
                                <tbody>${lossRunsRows}</tbody>
                            </table>
                        </div>
                    ` : ''}
                    ${meta.catNote ? `<div style="font-size:0.74rem;color:#9ca3af;background:rgba(255,255,255,0.03);border-radius:4px;padding:0.3rem 0.5rem;margin-bottom:0.45rem;font-style:italic;">${htmlEscape(meta.catNote)}</div>` : ''}
                    <div style="font-size:0.65rem;color:#6b7280;font-weight:700;letter-spacing:0.07em;margin-bottom:0.22rem;">ARTIFACTS (${meta.artifacts.length})</div>
                    ${artRows}
                </div>
            `;

            detailBox.style.display = 'block';
            const closeBtn = detailBox.querySelector('#dpStageDetailClose');
            if (closeBtn) closeBtn.addEventListener('click', closeStageDetail);

            if (typeof drawLineChart === 'function' && chartSeries.points.length > 1) {
                drawLineChart(
                    detailBox.querySelector(`#${chartId}`),
                    chartSeries.points,
                    'loss',
                    '#fbbf24',
                    chartSeries.hasPrevious ? 'loss_prev' : null,
                    { title: chartTitle }
                );
            }

            if (shouldScroll) detailBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }

        cards.forEach((card) => {
            card.addEventListener('click', () => {
                const stage = card.dataset.dpStage;
                if (!stage || !(window._ckDpStageCardMeta || {})[stage] || !detailBox) return;
                if (openStage === stage) {
                    closeStageDetail();
                    return;
                }
                renderStageDetail(stage, true);
            });
        });

        const persistedStage = String(window._ckDpStageOpen || '').trim();
        if (persistedStage && (window._ckDpStageCardMeta || {})[persistedStage]) {
            renderStageDetail(persistedStage, false);
        }
    }
}

function statusPill(label, tone) {
    const klass = tone === 'green'
        ? 'badge badge-green'
        : (tone === 'red' ? 'badge badge-orange' : 'badge badge-blue');
    return `<span class="${klass}">${htmlEscape(label)}</span>`;
}

function extractPostEval(files) {
    const pipeline = files.training_pipeline && typeof files.training_pipeline === 'object'
        ? files.training_pipeline
        : {};
    const dataLab = pipeline.data_lab && typeof pipeline.data_lab === 'object'
        ? pipeline.data_lab
        : {};
    const postEval = dataLab.post_train_eval && typeof dataLab.post_train_eval === 'object'
        ? dataLab.post_train_eval
        : (files.post_train_eval || {});
    return postEval && typeof postEval === 'object' ? postEval : {};
}

function extractRoundtrip(files) {
    const pipeline = files.training_pipeline && typeof files.training_pipeline === 'object'
        ? files.training_pipeline
        : {};
    const dataLab = pipeline.data_lab && typeof pipeline.data_lab === 'object'
        ? pipeline.data_lab
        : {};
    const roundtrip = dataLab.tokenizer_roundtrip && typeof dataLab.tokenizer_roundtrip === 'object'
        ? dataLab.tokenizer_roundtrip
        : (files.tokenizer_roundtrip || {});
    return roundtrip && typeof roundtrip === 'object' ? roundtrip : {};
}

function trainE2EState(trainE2E) {
    if (!trainE2E || typeof trainE2E !== 'object' || Object.keys(trainE2E).length === 0) {
        return { label: 'missing', pass: null };
    }
    if (trainE2E.status === 'pass' || trainE2E.pass === true || trainE2E.passed === true || trainE2E.pass_parity === true) {
        return { label: 'pass', pass: true };
    }
    if (trainE2E.status === 'fail' || trainE2E.pass === false || trainE2E.passed === false || trainE2E.pass_parity === false) {
        return { label: 'fail', pass: false };
    }
    return { label: 'check', pass: null };
}

function latestLoss(files) {
    const curve = files.training_loss_curve && Array.isArray(files.training_loss_curve.steps)
        ? files.training_loss_curve.steps
        : [];
    if (curve.length > 0) {
        const last = curve[curve.length - 1];
        const n = toNum(last && last.loss_ck, NaN);
        if (Number.isFinite(n)) return n;
    }
    const trainE2E = files.train_e2e && typeof files.train_e2e === 'object' ? files.train_e2e : {};
    const fallback = toNum(trainE2E.final_ck_loss, NaN);
    return Number.isFinite(fallback) ? fallback : NaN;
}

function renderTrainingLogbookPanel(files) {
    const root = document.getElementById('trainLogbookRoot');
    if (!root) return;
    clear(root);

    const pipeline = files.training_pipeline && typeof files.training_pipeline === 'object'
        ? files.training_pipeline
        : {};
    const dataLab = resolveDataLab(files);
    const qc = dataLab.dataset_qc && typeof dataLab.dataset_qc === 'object' ? dataLab.dataset_qc : {};
    const profile = dataLab.dataset_profile && typeof dataLab.dataset_profile === 'object' ? dataLab.dataset_profile : {};
    const tokenizerPreview = dataLab.tokenizer_preview && typeof dataLab.tokenizer_preview === 'object'
        ? dataLab.tokenizer_preview
        : (files.tokenizer_preview || {});
    const activeStage = typeof pipeline.active_stage === 'string' && pipeline.active_stage.trim()
        ? pipeline.active_stage
        : 'pretrain';
    const trainE2E = files.train_e2e && typeof files.train_e2e === 'object' ? files.train_e2e : {};
    const e2e = trainE2EState(trainE2E);
    const parityRows = parseParityRows(files.training_parity);
    const worstParity = parityRows.reduce((m, row) => Math.max(m, toNum(row.max_param_diff, 0)), 0);
    const postEval = extractPostEval(files);
    const roundtrip = extractRoundtrip(files);
    const roundtripEval = roundtrip.line_eval && typeof roundtrip.line_eval === 'object' ? roundtrip.line_eval : {};
    const regimen = files.training_parity_regimen && typeof files.training_parity_regimen === 'object'
        ? files.training_parity_regimen
        : {};
    const logbook = files.training_logbook && typeof files.training_logbook === 'object'
        ? files.training_logbook
        : {};
    const runCtx = getRunContext();
    const latest = latestLoss(files);
    const lossCurve = files.training_loss_curve && Array.isArray(files.training_loss_curve.steps)
        ? files.training_loss_curve.steps
        : [];
    const firstLoss = lossCurve.length > 0 ? toNum(lossCurve[0]?.loss_ck, NaN) : NaN;
    const reducedLoss = Number.isFinite(firstLoss) && Number.isFinite(latest) ? (latest < firstLoss) : false;

    const failures = [];
    const nextSteps = [];

    const lineRate = toNum(roundtripEval.exact_match_rate, NaN);
    const roundtripExact = roundtrip.exact_match === true;
    if (roundtrip.exact_match === false || (Number.isFinite(lineRate) && lineRate < 1.0)) {
        failures.push({
            title: 'Tokenizer roundtrip mismatch',
            why: `exact_match=${roundtrip.exact_match === true ? 'true' : 'false'}, line_rate=${Number.isFinite(lineRate) ? lineRate.toFixed(4) : '-'}`,
            severity: 'warning',
        });
        nextSteps.push('Fix tokenizer encode/decode fidelity first (line_rate should be 1.0000 before long runs).');
    }

    if (e2e.pass === false) {
        failures.push({
            title: 'Train E2E parity failed',
            why: `status=${e2e.label}`,
            severity: 'error',
        });
        nextSteps.push('Run parity canary rows and inspect first divergence before scaling training.');
    }

    if (Number.isFinite(worstParity) && worstParity > 1e-3) {
        failures.push({
            title: 'High CK vs PyTorch drift',
            why: `worst max_param_diff=${fmtExp(worstParity, 2)} (>1e-3)`,
            severity: 'error',
        });
        nextSteps.push('Use training parity regimen + xray to isolate the first mismatching layer/op.');
    } else if (Number.isFinite(worstParity) && worstParity > 1e-5) {
        failures.push({
            title: 'Elevated parity drift',
            why: `worst max_param_diff=${fmtExp(worstParity, 2)} (>1e-5)`,
            severity: 'warning',
        });
        nextSteps.push('Track drift trend; rerun parity regimen if this grows across commits.');
    }

    const regimenStatus = String(regimen.status || '').toLowerCase();
    if (regimenStatus === 'fail') {
        failures.push({
            title: 'Parity regimen failed',
            why: 'training_parity_regimen_latest.json status=fail',
            severity: 'error',
        });
        nextSteps.push('Open training_parity_regimen_latest.md and fix failing gate before full retrain.');
    }

    const postStatus = String(postEval.status || '').toLowerCase();
    const validSvgRate = toNum(postEval.valid_svg_rate, NaN);
    const minValidSvgRate = toNum(postEval.min_valid_svg_rate, 0.70);
    if (postStatus === 'fail' || (Number.isFinite(validSvgRate) && validSvgRate < minValidSvgRate)) {
        failures.push({
            title: 'Output quality/data-fit gate below target',
            why: `valid_svg_rate=${Number.isFinite(validSvgRate) ? validSvgRate.toFixed(4) : '-'} threshold=${minValidSvgRate.toFixed(4)}`,
            severity: 'warning',
        });
        nextSteps.push('Improve corpus coverage/SFT pairs; this is usually data-fit, not a numeric parity bug.');
    }

    if (failures.length === 0) {
        nextSteps.push('Continue with the next curriculum stage and keep parity gates enabled on regenerated code.');
    }

    const datasetRows = qc.non_empty_lines ?? profile.non_empty_lines ?? '-';
    const tokenizerVocab = tokenizerPreview.vocab_size ?? '-';
    const tokenCount = roundtrip.token_count ?? tokenizerPreview.encode_decode_example?.token_count ?? '-';
    const lineRateLabel = Number.isFinite(lineRate) ? lineRate.toFixed(4) : '-';
    const dataPass = String(qc.status || '').toLowerCase() === 'pass';
    const tokenizePass = tokenizerPreview && typeof tokenizerPreview === 'object' && (
        String(tokenizerPreview.status || '').toLowerCase() === 'ok'
        || Number.isFinite(Number(tokenizerPreview.vocab_size))
    );
    const roundtripPass = roundtripExact && (!Number.isFinite(lineRate) || lineRate >= 0.9999);
    const trainingPass = Number.isFinite(firstLoss) && Number.isFinite(latest) && reducedLoss;
    const stageRows = [
        {
            stage: '1) Data Intake (SVG)',
            status: dataPass ? 'pass' : 'check',
            evidence: `dataset_rows=${datasetRows}, qc_status=${qc.status || '-'}, dataset_path=${dataLab.dataset_path || qc.path || '-'}`,
        },
        {
            stage: '2) Tokenize (ASCII BPE)',
            status: tokenizePass ? 'pass' : 'check',
            evidence: `tokenizer=${tokenizerPreview.model_type || 'bpe'} vocab=${tokenizerVocab} token_count=${tokenCount}`,
        },
        {
            stage: '3) Encode -> Decode Fidelity',
            status: roundtripPass ? 'pass' : 'fail',
            evidence: `exact_match=${roundtrip.exact_match === true ? 'true' : 'false'} line_rate=${lineRateLabel} evaluated_lines=${roundtripEval.evaluated_lines ?? '-'}`,
        },
        {
            stage: '4) Train (1+ epochs)',
            status: trainingPass ? 'pass' : 'check',
            evidence: `loss_first=${Number.isFinite(firstLoss) ? firstLoss.toFixed(6) : '-'} loss_final=${Number.isFinite(latest) ? latest.toFixed(6) : '-'} reduced=${trainingPass ? 'yes' : 'no'}`,
        },
    ];

    const sampleRows = Array.isArray(roundtrip.sample_rows) ? roundtrip.sample_rows.slice(0, 10) : [];
    const sampleTable = sampleRows.length > 0
        ? `
            <table>
                <thead><tr><th>row</th><th>tokens</th><th>exact</th><th>decoded/sample</th></tr></thead>
                <tbody>
                    ${sampleRows.map((row) => {
                        const decoded = String(row.decoded || row.source || '-');
                        const shortDecoded = decoded.length > 220 ? `${decoded.slice(0, 220)}...` : decoded;
                        return `
                            <tr>
                                <td>${htmlEscape(row.line_no ?? '-')}</td>
                                <td>${htmlEscape(row.token_count ?? '-')}</td>
                                <td>${row.exact_match ? '<span class="badge badge-green">yes</span>' : '<span class="badge badge-orange">no</span>'}</td>
                                <td style="max-width:560px;overflow-wrap:anywhere;"><code>${htmlEscape(shortDecoded)}</code></td>
                            </tr>
                        `;
                    }).join('')}
                </tbody>
            </table>
        `
        : '<div style="color:var(--text-muted);">No sample rows embedded. Regenerate with tokenizer roundtrip artifacts.</div>';

    const currentStateRows = [
        ['run_dir', runCtx.runDir || '-'],
        ['stage', activeStage],
        ['train_e2e', e2e.label],
        ['latest_ck_loss', Number.isFinite(latest) ? latest.toFixed(6) : '-'],
        ['parity_worst_max_param_diff', Number.isFinite(worstParity) && worstParity > 0 ? fmtExp(worstParity, 2) : '-'],
        ['valid_svg_rate', Number.isFinite(validSvgRate) ? validSvgRate.toFixed(4) : '-'],
    ];

    const logMarkdown = typeof logbook.markdown === 'string' ? logbook.markdown.trim() : '';
    const logPath = typeof logbook.path === 'string' && logbook.path.trim() ? logbook.path : '-';
    const logSource = typeof logbook.source === 'string' && logbook.source.trim() ? logbook.source : '-';

    root.innerHTML = `
        <div class="parity-section">
            <h3><span class="badge badge-blue">State</span> Where We Are</h3>
            <table>
                <thead><tr><th>Signal</th><th>Value</th></tr></thead>
                <tbody>
                    ${currentStateRows.map((r) => `
                        <tr>
                            <td>${htmlEscape(r[0])}</td>
                            <td style="max-width:620px;overflow-wrap:anywhere;"><code>${htmlEscape(r[1])}</code></td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
            <div style="margin-top:0.6rem;display:flex;gap:0.5rem;flex-wrap:wrap;">
                ${statusPill(`train_e2e: ${e2e.label}`, e2e.pass === false ? 'red' : (e2e.pass === true ? 'green' : 'blue'))}
                ${statusPill(`stage: ${activeStage}`, 'blue')}
                ${statusPill(`failures: ${failures.length}`, failures.length > 0 ? 'red' : 'green')}
            </div>
        </div>
        <div class="parity-section" style="margin-top:0.8rem;">
            <h3><span class="badge badge-orange">Failures</span> What Failed + Why</h3>
            ${failures.length === 0
                ? '<div style="color:var(--text-muted);">No active failure signals in loaded artifacts.</div>'
                : failures.map((f) => `
                    <div class="alert-item ${f.severity === 'error' ? 'critical' : 'warning'}" style="margin-bottom:0.6rem;">
                        <div><strong>${htmlEscape(f.title)}</strong></div>
                        <div style="margin-top:0.2rem;color:var(--text-muted);">${htmlEscape(f.why)}</div>
                    </div>
                `).join('')}
        </div>
        <div class="parity-section" style="margin-top:0.8rem;">
            <h3><span class="badge badge-green">Next</span> Suggested Next Steps</h3>
            <ul style="margin:0.2rem 0 0 1.1rem;padding:0;">
                ${nextSteps.map((s) => `<li style="margin:0.28rem 0;">${htmlEscape(s)}</li>`).join('')}
            </ul>
        </div>
        <div class="parity-section" style="margin-top:0.8rem;">
            <h3><span class="badge badge-blue">Pipeline</span> Data -> Tokens -> Fidelity -> Training</h3>
            <table>
                <thead><tr><th>Stage</th><th>Status</th><th>Evidence</th></tr></thead>
                <tbody>
                    ${stageRows.map((s) => `
                        <tr>
                            <td>${htmlEscape(s.stage)}</td>
                            <td>${statusPill(String(s.status).toUpperCase(), s.status === 'pass' ? 'green' : (s.status === 'fail' ? 'red' : 'blue'))}</td>
                            <td style="max-width:640px;overflow-wrap:anywhere;"><code>${htmlEscape(s.evidence)}</code></td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        </div>
        <div class="parity-section" style="margin-top:0.8rem;">
            <h3><span class="badge badge-blue">Data Viewer</span> SVG Rows (HF-style sample)</h3>
            <div style="overflow:auto;">${sampleTable}</div>
        </div>
        <div class="parity-section" style="margin-top:0.8rem;">
            <h3><span class="badge badge-blue">Logbook</span> Operator Notes</h3>
            <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:0.5rem;margin-bottom:0.6rem;">
                <div><strong>source:</strong> <code>${htmlEscape(logSource)}</code></div>
                <div><strong>path:</strong> <code style="overflow-wrap:anywhere;">${htmlEscape(logPath)}</code></div>
            </div>
            ${logMarkdown
                ? `<pre style="font-size:0.8rem;white-space:pre-wrap;max-height:460px;overflow:auto;">${htmlEscape(logMarkdown)}</pre>`
                : '<div style="color:var(--text-muted);">No embedded training logbook markdown found for this run yet.</div>'}
        </div>
    `;
}

if (typeof window !== 'undefined') {
    // Global bridges for legacy (non-module) sections in ir_visualizer.html.
    window.getRunContext = getRunContext;
    window.resolveDataLab = resolveDataLab;
    window.htmlEscape = htmlEscape;
}

export function renderTrainingExtensionTab(tabId, files) {
    if (tabId === 'train-gradient') {
        renderGradientPanel(files);
    } else if (tabId === 'train-logbook') {
        renderTrainingLogbookPanel(files);
    } else if (tabId === 'train-data-lab') {
        renderDataLabPanel(files);
    } else if (tabId === 'train-parity') {
        renderParityPanel(files);
    }
}
