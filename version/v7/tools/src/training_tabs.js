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
    const params = grad.params && typeof grad.params === 'object' ? grad.params : {};
    const names = Object.keys(params);

    if (names.length === 0) {
        root.innerHTML = '<p style="color:var(--text-muted);">No training_grad_norms.json loaded.</p>';
        return;
    }

    const summary = names.map((name) => {
        const arr = Array.isArray(params[name]) ? params[name] : [];
        const last = arr.length > 0 ? toNum(arr[arr.length - 1], NaN) : NaN;
        return { name, last };
    }).sort((a, b) => Math.abs(b.last) - Math.abs(a.last));

    const rows = summary.slice(0, 40).map((s) => {
        let badge = '<span class="grad-healthy">healthy</span>';
        if (!Number.isFinite(s.last) || s.last < 1e-7) badge = '<span class="grad-dead">dead</span>';
        else if (s.last < 1e-5 || s.last > 0.1) badge = '<span class="grad-danger">danger</span>';
        else if (s.last < 1e-4 || s.last > 0.05) badge = '<span class="grad-warning">warning</span>';
        return `<tr><td>${s.name}</td><td>${fmtExp(s.last, 2)}</td><td>${badge}</td></tr>`;
    }).join('');

    root.innerHTML = `
        <div class="parity-section">
            <h3><span class="badge badge-orange">Gradient Health</span> Top Parameters by Latest Norm</h3>
            <div style="color:var(--text-muted);margin-bottom:0.5rem;">steps: ${steps.length} | params: ${names.length}</div>
            <table>
                <thead><tr><th>Parameter</th><th>Latest Norm</th><th>Status</th></tr></thead>
                <tbody>${rows}</tbody>
            </table>
        </div>
    `;
}

function renderParityPanel(files) {
    const root = document.getElementById('trainParityRoot');
    if (!root) return;
    clear(root);

    const rows = parseParityRows(files.training_parity);
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

function renderDataLabPanel(files) {
    const root = document.getElementById('trainDataLabRoot');
    if (!root) return;
    clear(root);

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
        <div class="parity-section">
            <h3><span class="badge badge-blue">Paths</span> Dataset + Tokenizer Sources</h3>
            ${pathTable}
        </div>
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

export function renderTrainingExtensionTab(tabId, files) {
    if (tabId === 'train-gradient') {
        renderGradientPanel(files);
    } else if (tabId === 'train-data-lab') {
        renderDataLabPanel(files);
    } else if (tabId === 'train-parity') {
        renderParityPanel(files);
    }
}
