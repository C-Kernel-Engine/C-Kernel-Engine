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
        if (typeof row.stage === 'string' && row.stage.trim()) artifactByStage[row.stage] = row;
    }
    const timelineByStage = {};
    for (const row of timeline) {
        if (typeof row.stage === 'string' && row.stage.trim()) timelineByStage[row.stage] = row;
    }
    const defaultStageOrder = ['pretrain', 'midtrain', 'sft', 'dpo', 'grpo', 'ppo'];
    const orderedStages = [];
    for (const s of defaultStageOrder) {
        if (timelineByStage[s] || s === activeStage) orderedStages.push(s);
    }
    for (const row of timeline) {
        const s = String(row.stage || '');
        if (s && !orderedStages.includes(s)) orderedStages.push(s);
    }
    if (!orderedStages.includes(activeStage)) orderedStages.push(activeStage);

    const flowCards = orderedStages.map((stage) => {
        const t = timelineByStage[stage] || {};
        const status = String(t.status || (stage === activeStage ? 'active' : 'planned'));
        const isActive = Boolean(t.active === true || stage === activeStage);
        const tone = stageTone(status);
        const chip = statusPill(status, tone);
        const border = isActive ? '1px solid rgba(71,180,117,0.55)' : '1px solid rgba(255,255,255,0.10)';
        const bg = isActive ? 'rgba(71,180,117,0.12)' : 'rgba(255,255,255,0.03)';
        const data = artifactByStage[stage] && typeof artifactByStage[stage] === 'object' ? artifactByStage[stage] : {};
        const artifactsForStage = Array.isArray(data.artifacts) ? data.artifacts : [];
        const artifactCount = artifactsForStage.length;
        return `
            <div style="min-width:130px;padding:0.45rem 0.55rem;border-radius:10px;${`border:${border};background:${bg};`}">
                <div style="font-weight:700;">${htmlEscape(stage)}</div>
                <div style="margin-top:0.2rem;">${chip}</div>
                <div style="margin-top:0.2rem;color:var(--text-muted);font-size:0.78rem;">artifacts: ${artifactCount}</div>
            </div>
        `;
    }).join('<div style="align-self:center;color:var(--text-muted);">-></div>');

    const activeStageMeta = artifactByStage[activeStage] && typeof artifactByStage[activeStage] === 'object'
        ? artifactByStage[activeStage]
        : {};
    const activeStageArtifacts = Array.isArray(activeStageMeta.artifacts) ? activeStageMeta.artifacts : [];
    const activeArtifactTable = activeStageArtifacts.length > 0
        ? `
            <table>
                <thead><tr><th>Label</th><th>Path</th><th>Required</th><th>Exists</th></tr></thead>
                <tbody>
                    ${activeStageArtifacts.map((a) => `
                        <tr>
                            <td>${htmlEscape(a.label ?? '-')}</td>
                            <td style="max-width:620px;overflow-wrap:anywhere;"><code>${htmlEscape(a.path ?? '-')}</code></td>
                            <td>${a.required ? statusPill('yes', 'blue') : statusPill('no', 'blue')}</td>
                            <td>${a.exists ? statusPill('yes', 'green') : statusPill('no', 'red')}</td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        `
        : '<div style="color:var(--text-muted);">No active-stage artifact map found. Regenerate training_pipeline_latest.json.</div>';

    const trainDims = pipeline.train_dims && typeof pipeline.train_dims === 'object' ? pipeline.train_dims : {};
    const dimsSummary = [
        `layers=${trainDims.num_layers ?? '-'}`,
        `embed_dim=${trainDims.embed_dim ?? '-'}`,
        `hidden_dim=${trainDims.hidden_dim ?? '-'}`,
        `vocab=${trainDims.vocab_size ?? '-'}`,
        `heads=${trainDims.num_heads ?? '-'}`,
        `ctx=${trainDims.context_length ?? '-'}`,
    ].join(' | ');
    const stageFlowSection = `
        <div class="parity-section">
            <h3><span class="badge badge-orange">Stage Flow</span> Pretrain -> Midtrain -> SFT/DPO/RL Readiness</h3>
            <div style="display:flex;gap:0.45rem;flex-wrap:wrap;align-items:center;">${flowCards}</div>
            <div style="margin-top:0.6rem;color:var(--text-muted);font-size:0.85rem;">
                active_stage=${htmlEscape(activeStage)} | model_dims: <code>${htmlEscape(dimsSummary)}</code>
            </div>
            <div style="margin-top:0.6rem;">
                <strong>Active Stage Evidence</strong>
                <div style="margin-top:0.35rem;overflow:auto;">${activeArtifactTable}</div>
            </div>
        </div>
    `;

    const datasetCatalog = Array.isArray(pipeline.dataset_catalog)
        ? pipeline.dataset_catalog.filter((row) => row && typeof row === 'object')
        : [];
    const datasetCatalogSection = `
        <div class="parity-section" style="margin-top:0.8rem;">
            <h3><span class="badge badge-blue">Dataset Catalog</span> Generated Data + Manifests</h3>
            ${datasetCatalog.length > 0
                ? `
                    <div style="margin-bottom:0.4rem;color:var(--text-muted);">
                        ${datasetCatalog.length} entries discovered for this run scope.
                    </div>
                    <table>
                        <thead><tr><th>Stage</th><th>Kind</th><th>Name</th><th>Rows</th><th>Path</th><th>Note</th></tr></thead>
                        <tbody>
                            ${datasetCatalog.slice(0, 200).map((row) => `
                                <tr>
                                    <td>${htmlEscape(row.stage ?? '-')}</td>
                                    <td>${htmlEscape(row.kind ?? '-')}</td>
                                    <td>${htmlEscape(row.name ?? '-')}</td>
                                    <td>${htmlEscape(row.rows ?? '-')}</td>
                                    <td style="max-width:500px;overflow-wrap:anywhere;"><code>${htmlEscape(row.path ?? '-')}</code></td>
                                    <td style="max-width:300px;overflow-wrap:anywhere;">${htmlEscape(row.note ?? '-')}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                `
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
                        <td style="max-width:620px;overflow-wrap:anywhere;"><code>${htmlEscape(r[1])}</code></td>
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
    const tokenizerSection = tokenizerPreview && Object.keys(tokenizerPreview).length > 0
        ? `
            <div class="parity-section" style="margin-top:0.8rem;">
                <h3><span class="badge badge-blue">Tokenizer Internals</span> Merges + Piece Map + Decode Behavior</h3>
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
        ${stageFlowSection}
        <div class="parity-section">
            <h3><span class="badge badge-blue">Paths</span> Dataset + Tokenizer Sources</h3>
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
            <div style="overflow:auto;">${sampleTable}</div>
        </div>
        ${tokenizerSection}
    `;
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
