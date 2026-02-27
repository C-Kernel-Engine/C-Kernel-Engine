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

    // Per-stage lookup maps for clickable detail
    const provByStage = {};
    if (Array.isArray(pipeline.data_provenance)) {
        pipeline.data_provenance.forEach((p) => { if (p && p.stage) provByStage[p.stage] = p; });
    }
    const catalogActiveByStage = {};
    if (Array.isArray(pipeline.dataset_catalog)) {
        pipeline.dataset_catalog.forEach((c) => { if (c && c.kind === 'active_dataset' && c.stage) catalogActiveByStage[c.stage] = c; });
    }
    const tokLineage = pipeline.tokenizer_lineage && typeof pipeline.tokenizer_lineage === 'object'
        ? pipeline.tokenizer_lineage : {};
    const tokCorpusEntry = catalogActiveByStage['pretrain'] || null;
    const tokCorpusName = (Array.isArray(tokLineage.corpus_datasets) && tokLineage.corpus_datasets.length > 0)
        ? tokLineage.corpus_datasets[0].name
        : (tokCorpusEntry ? tokCorpusEntry.name : null);
    const tokRtStatus = String((dataLab.tokenizer_roundtrip && dataLab.tokenizer_roundtrip.status) || '—').toUpperCase();

    function stageTokCoverage(stage) {
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
        if (row.stage === 'pretrain') return 'inferred';
        return 'unknown';
    }
    function tokManifestBadge(cov) {
        if (cov === 'yes') return '<span style="color:#6ee7b7;font-size:0.75rem;" title="Confirmed in tokenizer corpus">✓</span>';
        if (cov === 'no') return '<span style="color:#fb923c;font-size:0.75rem;" title="NOT in tokenizer corpus — tok gap risk if trained">✗</span>';
        if (cov === 'inferred') return '<span style="color:#9ca3af;font-size:0.75rem;" title="Inferred: pretrain manifests typically feed the tokenizer corpus. Add corpus_datasets to tokenizer_lineage to confirm.">~✓</span>';
        return '<span style="color:#4b5563;font-size:0.75rem;" title="Coverage unknown">?</span>';
    }

    const stageCardMeta = {};
    const flowCards = orderedStages.map((stage, idx) => {
        const t = timelineByStage[stage] || {};
        const status = String(t.status || (stage === activeStage ? 'active' : 'planned'));
        const isActive = Boolean(t.active === true || stage === activeStage);
        const tone = stageTone(status);
        const chip = statusPill(status, tone);
        const border = isActive ? '2px solid rgba(71,180,117,0.6)' : '1px solid rgba(255,255,255,0.10)';
        const bg = isActive ? 'rgba(71,180,117,0.09)' : 'rgba(255,255,255,0.03)';
        const data = artifactByStage[stage] && typeof artifactByStage[stage] === 'object' ? artifactByStage[stage] : {};
        const arts = Array.isArray(data.artifacts) ? data.artifacts : [];
        const cat = catalogActiveByStage[stage] || {};
        const prov = provByStage[stage] || {};
        const dsName = data.dataset_name || cat.name || prov.dataset_name || null;
        const covered = stageTokCoverage(stage);

        stageCardMeta[stage] = {
            status, isActive, dsName,
            dsRows: cat.rows || null,
            dsTok: data.token_count || prov.token_count || null,
            byteSize: prov.byte_size || null,
            covered, artifacts: arts,
            catNote: cat.note || null,
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

        return `
            <div data-dp-stage="${htmlEscape(stage)}" style="min-width:145px;max-width:190px;flex:0 0 auto;padding:0.5rem 0.6rem;border-radius:10px;border:${border};background:${bg};cursor:pointer;user-select:none;" title="Click to inspect ${htmlEscape(stage)} datasets &amp; artifacts">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:0.2rem;">
                    <div style="font-weight:700;font-size:0.88rem;">${htmlEscape(stage)}</div>
                    <div style="font-size:0.62rem;color:#6b7280;white-space:nowrap;">seq ${idx + 1}/${orderedStages.length}</div>
                </div>
                <div style="margin-top:0.22rem;">${chip}</div>
                ${dsPill}
                ${covBadge}
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
        </div>
    `;

    const stageFlowSection = `
        <div class="parity-section">
            <h3><span class="badge badge-orange">Stage Flow</span> Pretrain -> Midtrain -> SFT/DPO/RL Readiness</h3>
            ${tokSourcePanel}
            <div style="display:flex;gap:0.4rem;flex-wrap:wrap;align-items:stretch;">${flowCards}</div>
            <div id="dpStageDetail" style="display:none;margin-top:0.55rem;"></div>
            <div style="margin-top:0.45rem;color:var(--text-muted);font-size:0.8rem;">
                active_stage=${htmlEscape(activeStage)} | model_dims: <code>${htmlEscape(dimsSummary)}</code>
            </div>
        </div>
    `;
    window._ckDpStageCardMeta = stageCardMeta;

    const datasetCatalog = Array.isArray(pipeline.dataset_catalog)
        ? pipeline.dataset_catalog.filter((row) => row && typeof row === 'object')
        : [];

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
            <td style="max-width:380px;overflow-wrap:anywhere;font-size:0.74rem;"><code>${htmlEscape(row.path ?? '-')}</code></td>
            <td style="max-width:240px;overflow-wrap:anywhere;">${htmlEscape(row.note ?? '-')}</td>
        </tr>`;
    }

    const trainedSection = trainedEntries.length > 0 ? `
        <div style="margin-bottom:0.75rem;">
            <div style="font-size:0.78rem;color:#47b475;font-weight:600;margin-bottom:0.35rem;letter-spacing:0.04em;">
                ▶ TRAINED ON (${trainedEntries.length} dataset${trainedEntries.length > 1 ? 's' : ''})
            </div>
            <table style="font-size:0.8rem;">
                <thead><tr><th>Stage</th><th>Kind</th><th>Name</th><th style="text-align:right;">Rows</th><th>Path</th><th>Note</th></tr></thead>
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
                <thead><tr><th>Stage</th><th>Kind</th><th>Name</th><th style="text-align:right;">Rows</th><th>Path</th><th>Note</th></tr></thead>
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

        cards.forEach((card) => {
            card.addEventListener('click', () => {
                const stage = card.dataset.dpStage;
                const meta = (window._ckDpStageCardMeta || {})[stage];
                if (!meta || !detailBox) return;

                // Toggle off on second click of same card
                if (openStage === stage) {
                    detailBox.style.display = 'none';
                    detailBox.innerHTML = '';
                    openStage = null;
                    cards.forEach((c) => { c.style.outline = ''; });
                    return;
                }

                openStage = stage;
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

                const artRows = meta.artifacts.length > 0
                    ? meta.artifacts.map((a) => {
                        const nm = typeof a === 'string' ? a : (a.name || a.path || JSON.stringify(a));
                        return `<div style="font-family:monospace;font-size:0.73rem;color:var(--text-muted);word-break:break-all;padding:0.1rem 0;">${htmlEscape(nm)}</div>`;
                    }).join('')
                    : '<div style="font-size:0.76rem;color:#4b5563;font-style:italic;">No artifacts recorded for this stage yet.</div>';

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
                                <div style="font-size:0.65rem;color:#6b7280;font-weight:700;letter-spacing:0.07em;margin-bottom:0.2rem;">SHAPE</div>
                                <div style="font-size:0.78rem;">rows: ${fmtNum(meta.dsRows)}</div>
                                <div style="font-size:0.78rem;">tokens: ${fmtNum(meta.dsTok)}</div>
                                <div style="font-size:0.78rem;">bytes: ${fmtBytes(meta.byteSize)}</div>
                            </div>
                            <div style="background:rgba(255,255,255,0.04);border-radius:6px;padding:0.45rem 0.6rem;">
                                <div style="font-size:0.65rem;color:#6b7280;font-weight:700;letter-spacing:0.07em;margin-bottom:0.2rem;">TOKENIZER COVERAGE</div>
                                <div style="font-size:0.78rem;color:${covColor};">${covMark} ${covLabel}</div>
                            </div>
                        </div>
                        ${meta.catNote ? `<div style="font-size:0.74rem;color:#9ca3af;background:rgba(255,255,255,0.03);border-radius:4px;padding:0.3rem 0.5rem;margin-bottom:0.45rem;font-style:italic;">${htmlEscape(meta.catNote)}</div>` : ''}
                        <div style="font-size:0.65rem;color:#6b7280;font-weight:700;letter-spacing:0.07em;margin-bottom:0.22rem;">ARTIFACTS (${meta.artifacts.length})</div>
                        ${artRows}
                    </div>
                `;

                detailBox.style.display = 'block';
                const closeBtn = detailBox.querySelector('#dpStageDetailClose');
                if (closeBtn) {
                    closeBtn.addEventListener('click', () => {
                        detailBox.style.display = 'none';
                        detailBox.innerHTML = '';
                        openStage = null;
                        cards.forEach((c) => { c.style.outline = ''; });
                    });
                }
                detailBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            });
        });
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
