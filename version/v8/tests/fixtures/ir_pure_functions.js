// IR Visualizer — pure utility functions exported for testing.
// These are canonical copies of functions defined in ir_visualizer.html.
// Keep in sync: if you change the source, update this fixture and bump
// the hash in ir_visualizer_contract.json.
//
// Usage: node version/v8/tests/fixtures/ir_pure_functions.js
// (self-test when run directly)

function formatBytes(bytes) {
    if (bytes >= 1024*1024*1024) return (bytes / (1024*1024*1024)).toFixed(2) + ' GB';
    if (bytes >= 1024*1024) return (bytes / (1024*1024)).toFixed(2) + ' MB';
    if (bytes >= 1024) return (bytes / 1024).toFixed(2) + ' KB';
    return bytes + ' B';
}

function normalizeShapeInput(shape) {
    if (shape === null || shape === undefined) return [];
    if (Array.isArray(shape)) {
        return shape
            .filter(v => v !== null && v !== undefined && String(v).trim() !== '')
            .map(v => String(v));
    }
    if (typeof shape === 'number' && Number.isFinite(shape)) {
        return [String(shape)];
    }
    if (typeof shape === 'string') {
        const s = shape.trim();
        if (!s) return [];
        const wrapped = (s.startsWith('[') && s.endsWith(']')) || (s.startsWith('(') && s.endsWith(')'));
        if (wrapped) {
            const inner = s.slice(1, -1).trim();
            if (!inner) return [];
            try {
                const parsed = JSON.parse(s.replace(/\(/g, '[').replace(/\)/g, ']'));
                if (Array.isArray(parsed)) return parsed.map(v => String(v));
            } catch (_) {}
            return inner.split(/[x×,\s]+/).filter(Boolean);
        }
        const parts = s.split(/[x×,\s]+/).filter(Boolean);
        if (parts.length > 1) return parts;
        return [s];
    }
    if (typeof shape === 'object') {
        if (Array.isArray(shape.shape)) return normalizeShapeInput(shape.shape);
        if (Array.isArray(shape.dims)) return normalizeShapeInput(shape.dims);
        if (Array.isArray(shape.dimensions)) return normalizeShapeInput(shape.dimensions);
        const numericKeys = Object.keys(shape)
            .filter(k => /^\d+$/.test(k))
            .sort((a, b) => Number(a) - Number(b));
        if (numericKeys.length > 0) return numericKeys.map(k => String(shape[k]));
        const scalarValues = Object.values(shape).filter(v => typeof v === 'number' || typeof v === 'string');
        if (scalarValues.length > 0 && scalarValues.length <= 4) return scalarValues.map(v => String(v));
        try { return [JSON.stringify(shape)]; } catch (_) { return ['[object]']; }
    }
    return [String(shape)];
}

function formatShapeDisplay(shape, separator = ' × ') {
    const dims = normalizeShapeInput(shape);
    if (dims.length === 0) return '-';
    if (dims.length === 1) return `[${dims[0]}]`;
    return dims.join(separator);
}

function normalizeMode(mode) {
    return mode === 'prefill' ? 'prefill' : 'decode';
}

function escapeHtml(value) {
    return String(value || '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function quoteShell(value) {
    const s = String(value || '');
    if (!s) return '';
    if (/^[A-Za-z0-9_./:-]+$/.test(s)) return s;
    return `"${s.replace(/"/g, '\\"')}"`;
}

function normalizePathString(value) {
    return String(value || '').replace(/\\/g, '/').replace(/\/+$/, '');
}

function pathDirname(value) {
    const p = normalizePathString(value);
    if (!p) return '';
    const idx = p.lastIndexOf('/');
    if (idx <= 0) return idx === 0 ? '/' : '';
    return p.slice(0, idx);
}

function extractGgufStem(modelInput) {
    const raw = String(modelInput || '');
    if (!raw) return '';
    const clean = raw.split('?')[0].replace(/\\/g, '/');
    const idx = clean.lastIndexOf('/');
    const base = idx >= 0 ? clean.slice(idx + 1) : clean;
    if (!base.toLowerCase().endsWith('.gguf')) return '';
    return base.slice(0, -5);
}

function relativePathFromTo(fromDir, toPath) {
    const from = normalizePathString(fromDir);
    const to = normalizePathString(toPath);
    if (!from || !to) return null;
    if (!from.startsWith('/') || !to.startsWith('/')) return null;
    const a = from.split('/').filter(Boolean);
    const b = to.split('/').filter(Boolean);
    let i = 0;
    while (i < a.length && i < b.length && a[i] === b[i]) i += 1;
    const up = new Array(Math.max(0, a.length - i)).fill('..');
    const down = b.slice(i);
    const parts = up.concat(down);
    return parts.length ? parts.join('/') : '.';
}

// ── X-Ray tab pure helpers (mirrors of ir_visualizer.html) ──────────────────

function xrayReportKindLabel(report) {
    if (!report || typeof report !== 'object') return 'X-Ray Report';
    const schema = String(report.schema || '');
    if (schema === 'cke.whisper_encoder_pytorch_xray') return 'Whisper Encoder X-Ray';
    if (schema === 'cke.xray_ranking_report') return 'Ranking Report';
    if (schema === 'cke.xray_execution_trace') return 'Execution Trace';
    if (schema === 'cke.xray_execution_state_report') return 'Execution State Report';
    if (schema === 'cke.xray.decoder_pytorch') return 'Decoder X-Ray';
    if (schema === 'cke.xray_orchestration_report') return 'Vision X-Ray (orchestration)';
    if (schema === 'cke.xray_numerical_report') return 'Numerical Parity X-Ray';
    return schema || 'X-Ray Report';
}

function xrayBackendLabel(report) {
    if (!report || typeof report !== 'object') return 'unknown';
    const schema = String(report.schema || '');
    if (schema === 'cke.whisper_encoder_pytorch_xray') return 'PyTorch';
    if (schema === 'cke.xray.decoder_pytorch') return 'PyTorch';
    if (schema === 'cke.xray_ranking_report') return 'ggml-oracle';
    if (schema === 'cke.xray_execution_trace') return String(report.backend || 'unknown');
    const oracle = report.oracle_backend
        || (report.final_report && report.final_report.oracle_backend)
        || (report.provenance && report.provenance.pytorch ? 'pytorch' : null)
        || report.backend;
    const b = String(oracle || 'unknown').toLowerCase();
    if (b.indexOf('llama') >= 0) return 'llama.cpp';
    if (b.indexOf('torch') >= 0) return 'PyTorch';
    if (b.indexOf('ggml') >= 0) return 'ggml-oracle';
    return oracle ? String(oracle) : 'unknown';
}

function xrayReportStatus(report) {
    if (!report || typeof report !== 'object') return 'unknown';
    if (typeof report.status === 'string' && report.status) return report.status;
    if (Array.isArray(report.checks) && report.checks.length) {
        return report.checks.every(c => c && c.status === 'pass') ? 'pass' : 'fail';
    }
    return 'unknown';
}

// Canonical X-Ray checkpoint row:
//   sequence_index  - chart ordering only
//   checkpoint_id   - stable semantic identity
//   op_idx          - nullable explicit call-IR operation identity
//   resolved_execution: {phase, layer, function, kernel_id, resolved_contract_id}
//   metrics: {max_abs, mean_abs, rmse, relative_rmse, byte_exact, exact_ratio}
//   (rmse/relative_rmse are null unless the producer actually computed
//    squared-error statistics; mean_abs is never relabelled as rmse)
//
// Normalize any checkpoint-table X-ray report into chart rows.
function xrayDriftRows(report) {
    if (!report || typeof report !== 'object') return [];

    function normalizeRow(cp, i) {
        const exec = (cp && cp.resolved_execution) || {};
        const metrics = (cp && cp.metrics) || cp || {};
        const cid = (cp && cp.checkpoint_id != null) ? String(cp.checkpoint_id) : ('stop ' + i);
        const seq = (cp && cp.sequence_index != null) ? cp.sequence_index : i;
        return {
            stop: seq,
            sequenceIndex: seq,
            checkpointId: cid,
            opIdx: (cp && cp.op_idx != null) ? cp.op_idx : null,
            label: cid,
            layer: (exec.layer != null) ? exec.layer : ((cp && cp.layer != null) ? cp.layer : null),
            maxAbs: metrics.max_abs != null ? Number(metrics.max_abs) : null,
            meanAbs: metrics.mean_abs != null ? Number(metrics.mean_abs) : null,
            rmse: metrics.rmse != null ? Number(metrics.rmse) : null,
            relativeRmse: metrics.relative_rmse != null ? Number(metrics.relative_rmse) : null,
            byteExact: !!metrics.byte_exact,
            exactRatio: metrics.exact_ratio != null ? Number(metrics.exact_ratio) : null,
            raw: cp,
        };
    }

    // Legacy Whisper-style checkpoints.
    const cps = Array.isArray(report.checkpoints) ? report.checkpoints : null;
    if (cps && cps.length && cps[0] && cps[0].metrics) {
        return cps.map((cp, i) => {
            const row = normalizeRow(cp, i);
            row.stop = (cp.stop != null ? cp.stop : i);
            row.sequenceIndex = row.stop;
            row.label = String(cp.checkpoint != null ? cp.checkpoint : ('stop ' + i));
            row.checkpointId = row.label;
            return row;
        });
    }

    // Real producer output: final_report.comparisons (Qwen llama.cpp) or report.comparisons.
    const comparisons = (report.final_report && Array.isArray(report.final_report.comparisons))
        ? report.final_report.comparisons
        : (Array.isArray(report.comparisons) ? report.comparisons : null);
    if (comparisons && comparisons.length) {
        return comparisons.map((cp, i) => normalizeRow(cp, i));
    }

    // Legacy fixture-style drift_progression.
    const drift = (report.final_report && report.final_report.drift_progression) || report.drift_progression;
    if (drift && Array.isArray(drift.checkpoints)) {
        return drift.checkpoints.map((cp, i) => normalizeRow(cp, i));
    }

    return [];
}

function buildXrayRunbookHtml(runCtx, emptyCtx) {
    const hasRun = !!(runCtx && runCtx.runDir);
    const runDir = hasRun ? quoteShell(runCtx.runDir) : '<run_dir>';
    const out = (name) => hasRun ? quoteShell(runCtx.runDir + '/' + name) : ('<run_dir>/' + name);
    const ec = (typeof xrayEmptyStateContext === 'function') ? xrayEmptyStateContext(emptyCtx) : null;
    const ctxHeader = ec ? ''
        + '<div style="font-size:0.68rem; color:var(--text-secondary); margin-bottom:0.45rem; line-height:1.7;">'
        + 'context: model <code>' + escapeHtml(ec.modelPath || '<model_path>') + '</code>'
        + ' · xray-dir <code>' + escapeHtml(ec.xrayDir || '<xray_dir>') + '</code>'
        + ' · backend <code>' + escapeHtml(ec.backend || '<none loaded>') + '</code>'
        + ' · phase <code>' + escapeHtml(ec.phase || '<none loaded>') + '</code><br>'
        + 'expected artifacts: <code>xray_qwen3vl_llamacpp_summary.json</code>, <code>xray_qwen3vl_bf16_summary.json</code>, '
        + '<code>xray_summary.json</code> (generic, assigned by backend inspection), <code>whisper-encoder-xray.json</code>, <code>xray_decoder_summary.json</code>'
        + '</div>' : '';
    const whisperCmd = 'python3 version/v8/scripts/compare_whisper_encoder_pytorch_v8.py \\\n'
        + '  --run-dir ' + runDir + ' \\\n'
        + '  --checkpoint <hf_whisper_checkpoint> \\\n'
        + '  --stops key \\\n'
        + '  --output ' + out('whisper-encoder-xray.json');
    const visionLlamaCmd = 'make xray-vision-parity BACKEND=llamacpp GGUF=<mmproj.gguf> XRAY_OUTPUT_DIR=build/xray/qwen3vl_llamacpp';
    const visionTorchCmd = 'make xray-vision-parity BACKEND=pytorch CHECKPOINT=<hf_qwen3vl_ckpt> RUNTIME_DIR=<runtime_dir> WEIGHTS_BUMP=<weights.bump> CALL_IR=<call.json> XRAY_OUTPUT_DIR=build/xray/qwen3vl_bf16';
    const decoderCmd = 'python3 version/v8/scripts/xray_decoder_pytorch_v8.py \\\n'
        + '  --checkpoint <hf_decoder_ckpt> --runtime <libdecoder.so> --call-ir <call.json> \\\n'
        + '  --token-ids 1,2,3 --output-dir build/xray/decoder_pytorch';
    const rankingCmd = 'python3 version/v8/scripts/normalize_xray_ranking_report_v8.py \\\n'
        + '  --input <logits_trace.json> --kind teacher_forced \\\n'
        + '  --output ' + out('xray_ranking_report.json');
    const stateCmd = 'python3 version/v8/scripts/xray_execution_state_v8.py \\\n'
        + '  --subject-trace <ck_trace.json> --oracle-trace <oracle_trace.json> \\\n'
        + '  --output ' + out('xray_execution_state_report.json');
    const refreshCmd = hasRun
        ? 'python3 version/v8/tools/open_ir_visualizer_v8.py --generate --run ' + runDir + ' --html-only --strict-run-artifacts --output ' + out('ir_report.html')
        : 'python3 version/v8/tools/open_ir_visualizer_v8.py --generate --run <run_dir> --html-only --strict-run-artifacts --output <run_dir>/ir_report.html';
    const block = (title, cmd) => ''
        + '<div style="margin-top:0.45rem;">' + escapeHtml(title) + '</div>'
        + '<pre style="font-size:0.72rem;white-space:pre-wrap;margin-top:0.35rem;">' + escapeHtml(cmd) + '</pre>';
    return ''
        + '<div style="margin-top:0.5rem; padding:0.75rem 1rem; background:rgba(255,180,0,0.08); border:1px solid rgba(255,180,0,0.3); border-radius:8px; font-size:0.72rem; color:var(--text-secondary); line-height:1.6;">'
        + '<strong style="color:var(--orange);">No X-ray artifacts loaded.</strong>'
        + ctxHeader
        + '<div style="margin-top:0.35rem;">Produce backend parity X-ray reports with any of these, then refresh this report:</div>'
        + block('Whisper encoder vs PyTorch (per-checkpoint drift):', whisperCmd)
        + block('Qwen3-VL vision parity vs llama.cpp:', visionLlamaCmd)
        + block('Qwen3-VL vision parity vs PyTorch (bf16):', visionTorchCmd)
        + block('Decoder parity vs PyTorch:', decoderCmd)
        + block('Logits ranking report (ggml oracle):', rankingCmd)
        + block('Execution-state divergence stages:', stateCmd)
        + block('Refresh this report after artifacts land:', refreshCmd)
        + '</div>';
}

// Phase of a report's subject run: prefill / decode / teacher_forced / mixed_prefill.
function xrayPhaseOf(report) {
    if (!report || typeof report !== 'object') return null;
    const runPhase = report.run && typeof report.run.phase === 'string' ? report.run.phase : null;
    if (runPhase) return runPhase;
    const schema = String(report.schema || '');
    if (schema === 'cke.whisper_encoder_pytorch_xray') return 'prefill';
    if (schema === 'cke.xray.decoder_pytorch') return 'decode';
    const comps = (report.final_report && Array.isArray(report.final_report.comparisons)) ? report.final_report.comparisons
        : (Array.isArray(report.comparisons) ? report.comparisons : []);
    for (const c of comps) {
        const ph = c && c.resolved_execution && c.resolved_execution.phase;
        if (typeof ph === 'string' && ph) return ph;
    }
    const drift = (report.final_report && report.final_report.drift_progression) || report.drift_progression;
    if (drift && Array.isArray(drift.checkpoints)) {
        for (const cp of drift.checkpoints) {
            const ph = cp && cp.resolved_execution && cp.resolved_execution.phase;
            if (typeof ph === 'string' && ph) return ph;
        }
    }
    return null;
}

// Contract chip state for one call-IR op.
function xrayContractState(op) {
    if (!op || typeof op !== 'object') return 'none';
    const req = op.required_contract != null ? op.required_contract : null;
    const res = op.resolved_contract || op.resolved_codegen_capability || null;
    if (req && res) {
        const reqId = (typeof req === 'object')
            ? (req.contract_id || req['numerics.attention_reduction'] || null) : null;
        const resId = (typeof res === 'object')
            ? (res.resolved_contract_id || res.contract_id || null) : null;
        if (reqId && resId && reqId !== resId) return 'substitution';
        return 'match';
    }
    if (req) return 'unresolved';
    if (res) return 'resolved';
    return 'none';
}

// Join call-IR ops (execution order) with X-ray drift rows.
// Matching precedence:
//   1. Explicit op_idx on the checkpoint row.
//   2. Unambiguous stable execution identity (layer + function + kernel_id + resolved_contract_id).
//   3. Leave the checkpoint visibly unmapped.
function xrayCircuitRows(callIr, driftRows, registry) {
    const ops = callIr && Array.isArray(callIr.operations) ? callIr.operations : [];
    const names = {};
    const kernels = registry && Array.isArray(registry.kernels) ? registry.kernels : [];
    kernels.forEach(k => { if (k && k.id != null) names[String(k.id)] = k.name || String(k.id); });

    const byIdx = {};
    ops.forEach(op => {
        if (op && op.idx != null && !(op.idx in byIdx)) byIdx[op.idx] = op;
    });

    function execKey(layer, func, kernelId, contractId) {
        return JSON.stringify([
            layer,
            String(func || ''),
            String(kernelId || ''),
            String(contractId || '')
        ]);
    }
    function opExecKey(op) {
        const layer = op.layer != null ? op.layer : null;
        const func = op.function || null;
        const kernelId = (op.call_abi && op.call_abi.kernel_id) || func || null;
        const contractId = (op.resolved_contract && op.resolved_contract.resolved_contract_id) || null;
        return execKey(layer, func, kernelId, contractId);
    }
    const byExecution = {};
    ops.forEach(op => {
        const key = opExecKey(op);
        (byExecution[key] = byExecution[key] || []).push(op);
    });

    function findOpByExecution(row) {
        const raw = row.raw || {};
        const exec = raw.resolved_execution || {};
        const layer = row.layer != null ? row.layer : (exec.layer != null ? exec.layer : null);
        const func = exec.function || raw.function || null;
        const kernelId = exec.kernel_id || null;
        const contractId = exec.resolved_contract_id || null;

        // Never join on an empty identity: a checkpoint whose function,
        // kernel_id and contract are all unresolved stays unmapped.
        if (func == null && kernelId == null && contractId == null) return null;

        const candidates = byExecution[execKey(layer, func, kernelId, contractId)] || [];
        const available = candidates.filter(o => !usedOps.has(o));
        if (available.length === 1) return available[0];

        if (layer == null) {
            let fallback = null;
            let fallbackCount = 0;
            Object.values(byExecution).forEach(group => {
                group.forEach(op => {
                    if (usedOps.has(op)) return;
                    const opFunc = op.function || null;
                    const opKernelId = (op.call_abi && op.call_abi.kernel_id) || opFunc || null;
                    const opContractId = (op.resolved_contract && op.resolved_contract.resolved_contract_id) || null;
                    if (String(opFunc || '') === String(func || '') &&
                        String(opKernelId || '') === String(kernelId || '') &&
                        String(opContractId || '') === String(contractId || '')) {
                        fallback = op;
                        fallbackCount++;
                    }
                });
            });
            if (fallbackCount === 1) return fallback;
        }
        return null;
    }

    const opToRow = {};
    const usedOps = new Set();
    const unmapped = [];
    (Array.isArray(driftRows) ? driftRows : []).forEach(row => {
        if (!row) return;
        let op = null;

        if (row.opIdx != null && byIdx[row.opIdx] != null && !usedOps.has(byIdx[row.opIdx])) {
            op = byIdx[row.opIdx];
        }

        if (!op) op = findOpByExecution(row);

        if (op) {
            opToRow[op.idx] = row;
            usedOps.add(op);
        } else {
            unmapped.push(row);
        }
    });

    const rows = ops.map(op => {
        const idx = op.idx != null ? op.idx : null;
        const xr = idx != null ? opToRow[idx] : null;
        const kernelId = (op.call_abi && op.call_abi.kernel_id) || op.function || null;
        return {
            idx,
            op: op.op || '',
            function: op.function || '',
            provider: (kernelId && names[String(kernelId)]) || kernelId || '',
            layer: op.layer != null ? op.layer : null,
            contract: xrayContractState(op),
            phase: (op.required_contract && op.required_contract['execution.phase']) || op.mode || op.section || null,
            hasCheckpoint: !!xr,
            checkpointId: xr ? xr.checkpointId : null,
            maxAbs: xr ? xr.maxAbs : null,
            rmse: xr ? xr.rmse : null,
            exactRatio: xr ? xr.exactRatio : null,
            byteExact: xr ? xr.byteExact : false,
        };
    });

    return {
        rows: rows,
        mappedCount: rows.filter(r => r.hasCheckpoint).length,
        unmapped: unmapped,
        unmappedCount: unmapped.length,
    };
}

// Edges where max_abs grows faster than `threshold`× between consecutive stops.
function xrayGrowthEdges(rows, threshold) {
    const t = threshold != null ? threshold : 3;
    const edges = [];
    const rs = Array.isArray(rows) ? rows : [];
    for (let i = 1; i < rs.length; i++) {
        const prev = rs[i - 1] && rs[i - 1].maxAbs;
        const cur = rs[i] && rs[i].maxAbs;
        if (prev != null && cur != null && prev > 0 && cur > 0 && cur / prev > t) {
            edges.push({ index: i, ratio: cur / prev });
        }
    }
    return edges;
}

// Extract the run identity used for ranking correlation.
// Every field comes from the report itself; nothing is invented.
function xrayRunIdentityOf(rep) {
    if (!rep || typeof rep !== 'object') return null;
    const prov = (rep.provenance && typeof rep.provenance === 'object') ? rep.provenance : {};
    const subj = (rep.subject && typeof rep.subject === 'object') ? rep.subject
        : ((prov.subject && typeof prov.subject === 'object') ? prov.subject : null);
    const orcl = (rep.oracle && typeof rep.oracle === 'object') ? rep.oracle
        : ((prov.oracle && typeof prov.oracle === 'object') ? prov.oracle : null);
    return {
        runId: rep.run_id || (rep.run && rep.run.id) || prov.run_id || null,
        phase: rep.phase || (rep.run && rep.run.phase) || prov.phase || xrayPhaseOf(rep) || null,
        modelSha256: rep.model_sha256 || prov.model_sha256 || null,
        subjectBackend: (subj && subj.backend) || rep.subject_backend
            || (rep.final_report && rep.final_report.subject_backend) || null,
        subjectSha256: (subj && subj.runtime_sha256) || null,
        subjectGeneratedModelSha256: (subj && subj.generated_model_sha256) || null,
        oracleBackend: (orcl && orcl.backend) || rep.oracle_backend
            || (rep.final_report && rep.final_report.oracle_backend) || null,
        oracleSha256: (orcl && orcl.runtime_sha256) || null,
        oracleFingerprintSha256: (orcl && orcl.fingerprint_sha256) || null,
    };
}

// Determine whether a ranking report actually belongs to the given report.
// Fail-closed: correlation requires complete, exactly agreeing identity on
// run_id, phase, model, subject backend+fingerprint and oracle
// backend+fingerprint. Missing provenance means "unscoped ranking", and an
// unscoped ranking report stays an independent card — never a match.
function xrayRankingMatchesReport(report, rankingReport) {
    if (!report || !rankingReport || rankingReport.schema !== 'cke.xray_ranking_report') return false;
    const a = xrayRunIdentityOf(report);
    const b = xrayRunIdentityOf(rankingReport);
    if (!a || !b) return false;
    const fields = ['runId', 'phase', 'modelSha256', 'subjectBackend', 'subjectSha256', 'subjectGeneratedModelSha256', 'oracleBackend', 'oracleFingerprintSha256'];
    for (const f of fields) {
        if (a[f] == null || b[f] == null) return false;
        if (String(a[f]) !== String(b[f])) return false;
    }
    return true;
}

// Circuit scope of a report: which call-IR artifact its checkpoints belong to.
// An explicit circuit_scope field wins; legacy reports fall back to
// checkpoint-prefix / schema inference.
function xrayCircuitScope(report) {
    if (!report || typeof report !== 'object') return null;
    const explicit = report.circuit_scope
        || (report.final_report && report.final_report.circuit_scope) || null;
    if (explicit) return String(explicit);
    const schema = String(report.schema || '');
    if (schema === 'cke.whisper_encoder_pytorch_xray') return 'audio_encoder';
    if (schema === 'cke.xray.decoder_pytorch') return 'decoder';
    const rows = xrayDriftRows(report);
    for (const row of rows) {
        const cid = String(row.checkpointId || '');
        if (cid.indexOf('vision.') === 0) return 'vision_encoder';
        if (cid.indexOf('audio.') === 0 || cid.indexOf('encoder.') === 0) return 'audio_encoder';
    }
    const phase = xrayPhaseOf(report);
    if (phase === 'mixed_prefill') return 'mixed_prefill';
    if (phase === 'decode' || phase === 'teacher_forced') return 'decoder';
    return null;
}

// Select the call-IR artifact for a circuit scope.
// Returns { key, callIr } or null when the scoped IR is not loaded.
function xraySelectCircuitCallIr(files, scope) {
    if (!files || typeof files !== 'object') return null;
    const candidates = {
        vision_encoder: ['bridge_encoder_call', 'lowered_prefill_call'],
        audio_encoder: ['bridge_encoder_call', 'lowered_prefill_call'],
        mixed_prefill: ['lowered_prefill_call', 'lowered_decode_call'],
        decoder: ['lowered_decode_call', 'lowered_prefill_call'],
    };
    const keys = candidates[scope] || ['lowered_decode_call', 'lowered_prefill_call'];
    for (const key of keys) {
        const ir = files[key];
        if (ir && Array.isArray(ir.operations) && ir.operations.length) return { key, callIr: ir };
    }
    return null;
}

// Resolve the embedded-data key for a live-polled file. Canonical filenames
// map statically; the generic producer output xray_summary.json is assigned
// by inspecting its declared backend so a llama.cpp capture never lands on
// the PyTorch card (or vice versa).
function xrayLiveKeyForFile(fname, payload) {
    const staticMap = {
        'training_pipeline_latest.json': 'training_pipeline',
        'training_loss_curve_latest.json': 'training_loss_curve',
        'training_grad_norms_latest.json': 'training_grad_norms',
        'training_parity_latest.json': 'training_parity',
        'training_step_profile_latest.json': 'training_step_profile',
        'training_checkpoint_policy_latest.json': 'training_checkpoint_policy',
        'corpus_sampling_log_latest.json': 'corpus_sampling_log',
        'whisper-encoder-xray.json': 'xray_whisper_encoder',
        'xray_ranking_report.json': 'xray_ranking',
        'xray_execution_trace.json': 'xray_execution_trace',
        'xray_execution_state_report.json': 'xray_execution_state',
        'xray_decoder_summary.json': 'xray_decoder_pytorch',
        'xray_qwen3vl_bf16_summary.json': 'xray_qwen3vl_pytorch',
        'xray_qwen3vl_llamacpp_summary.json': 'xray_qwen3vl_llamacpp',
        'xray_monotonic.json': 'xray_monotonic',
        'xray_monotonic_provider_gate.json': 'xray_monotonic',
    };
    if (Object.prototype.hasOwnProperty.call(staticMap, fname)) return staticMap[fname];
    if (fname === 'xray_summary.json') {
        const backend = payload && typeof payload === 'object'
            ? String(payload.backend
                || (payload.final_report && payload.final_report.oracle_backend)
                || '').toLowerCase()
            : '';
        if (backend.indexOf('llama') >= 0 || backend.indexOf('ggml') >= 0) return 'xray_qwen3vl_llamacpp';
        if (backend.indexOf('torch') >= 0 || backend.indexOf('pytorch') >= 0) return 'xray_qwen3vl_pytorch';
        return null; // unknown backend: do not guess a card
    }
    return null;
}

// Board verdict: pass (green) / cosmetic drift (amber) / behavioral divergence (red).
// A top-1 flip in a correlated ranking report is always behavioral divergence,
// even when the tensor-level parity report is green.
function xrayBoardVerdict(report, rankingReport) {
    const status = xrayReportStatus(report);
    const checks = rankingReport && Array.isArray(rankingReport.checks) ? rankingReport.checks : [];
    const flips = checks.filter(c => c && c.ck_top1 !== c.oracle_top1).length;
    if (flips > 0) {
        return { level: 'behavioral', line: flips + ' top-1 flip(s) vs oracle — behavioral divergence' };
    }
    if (status === 'pass') return { level: 'pass', line: 'all within gate' };
    if (status === 'fail' || status === 'diverged' || status === 'error') {
        if (checks.length) {
            return { level: 'cosmetic', line: 'thresholds crossed, ranking top-1 agrees — drift, output unchanged' };
        }
        return { level: 'fail', line: 'thresholds crossed (no ranking report to classify)' };
    }
    return { level: 'unknown', line: 'no gate status' };
}

// ── X-Ray per-panel actionable empty states ───────────────────────────────
// One shared builder. Rules:
//  - Commands that produce evidence are strictly separated from commands
//    that only refresh the visualizer.
//  - Context values come only from loaded report provenance / run metadata;
//    anything unknown stays a visible placeholder. Nothing is inferred.
//  - Every flag below was verified against the producer's CLI parser.

function xrayEmptyStateContext(ctx) {
    const c = {
        runDir: null, modelPath: null, xrayDir: null,
        runId: null, phase: null, modelSha256: null,
        subjectRuntimeSha256: null, subjectGeneratedModelSha256: null,
        oracleBackend: null, oracleFingerprintSha256: null,
        oracleCommit: null, oracleMode: null, backend: null,
    };
    if (!ctx || typeof ctx !== 'object') return c;
    c.runDir = ctx.runDir || null;
    c.modelPath = ctx.modelPath || null;
    c.xrayDir = ctx.xrayDir || null;
    const rep = ctx.referenceReport;
    if (rep && typeof rep === 'object') {
        const id = (typeof xrayRunIdentityOf === 'function') ? xrayRunIdentityOf(rep) : null;
        if (id) {
            c.runId = id.runId;
            c.phase = id.phase;
            c.modelSha256 = id.modelSha256;
            c.subjectRuntimeSha256 = id.subjectSha256;
            c.subjectGeneratedModelSha256 = id.subjectGeneratedModelSha256;
            c.oracleBackend = id.oracleBackend;
            c.oracleFingerprintSha256 = id.oracleFingerprintSha256;
        }
        const prov = (rep.provenance && typeof rep.provenance === 'object') ? rep.provenance : {};
        const orcl = (prov.oracle && typeof prov.oracle === 'object') ? prov.oracle
            : ((rep.oracle && typeof rep.oracle === 'object') ? rep.oracle : {});
        c.oracleCommit = orcl.commit || null;
        c.oracleMode = orcl.mode || null;
        c.backend = rep.backend || c.oracleBackend || null;
    }
    return c;
}

function xrayEmptyStateCopyBlock(title, cmd, note) {
    return ''
        + '<div class="profile-howto-block">'
        + '<div class="profile-howto-block-head"><h4>' + escapeHtml(title) + '</h4>'
        + `<button type="button" class="profile-copy-btn" data-command="${encodeURIComponent(cmd)}" onclick="copyProfileRunbookCommand(this)">Copy</button></div>`
        + '<pre>' + escapeHtml(cmd) + '</pre>'
        + (note ? `<div style="font-size:0.66rem; color:var(--text-muted); margin-top:0.25rem;">${escapeHtml(note)}</div>` : '')
        + '</div>';
}

function xrayRefreshCommand(c) {
    const target = c.runDir ? ('--run ' + quoteShell(c.runDir))
        : (c.modelPath ? quoteShell(c.modelPath) : '<model_or_run_dir>');
    const xrayArg = c.xrayDir ? (' --xray-dir ' + quoteShell(c.xrayDir)) : '';
    const out = c.runDir ? quoteShell(c.runDir + '/ir_report.html') : '<output>/ir_report.html';
    return 'python3 version/v8/tools/open_ir_visualizer_v8.py --generate --html-only '
        + target + xrayArg + ' --output ' + out;
}

function xrayRankingProducerCommand(c) {
    const v = (val, ph) => (val != null ? String(val) : ph);
    const out = c.xrayDir
        ? quoteShell(c.xrayDir + '/xray_ranking_report.json')
        : '<xray_dir>/xray_ranking_report.json';
    return 'python3 version/v8/scripts/normalize_xray_ranking_report_v8.py \\\n'
        + '  --input <logits_trace.json> \\\n'
        + '  --kind teacher_forced \\\n'
        + '  --run-id ' + v(c.runId, '<run_id>') + ' \\\n'
        + '  --phase ' + v(c.phase, '<phase>') + ' \\\n'
        + '  --model-sha256 ' + v(c.modelSha256, '<model_sha256>') + ' \\\n'
        + '  --subject-backend ck \\\n'
        + '  --subject-runtime-sha256 ' + v(c.subjectRuntimeSha256, '<subject_runtime_sha256>') + ' \\\n'
        + '  --subject-generated-model-sha256 ' + v(c.subjectGeneratedModelSha256, '<subject_generated_model_sha256>') + ' \\\n'
        + '  --oracle-backend ' + v(c.oracleBackend, '<oracle_backend>') + ' \\\n'
        + '  --oracle-fingerprint-sha256 ' + v(c.oracleFingerprintSha256, '<oracle_fingerprint_sha256>') + ' \\\n'
        + '  --output ' + out;
}

function xrayExecutionStateProducerCommand(c) {
    const out = c.xrayDir
        ? quoteShell(c.xrayDir + '/xray_execution_state_report.json')
        : '<xray_dir>/xray_execution_state_report.json';
    return 'python3 version/v8/scripts/xray_execution_state_v8.py \\\n'
        + '  --subject-trace <ck_trace.json> --oracle-trace <oracle_trace.json> \\\n'
        + '  --output ' + out;
}

function xrayDriftProducerCommands(c) {
    const llamaOut = c.xrayDir ? quoteShell(c.xrayDir) : 'build/xray/qwen3vl_llamacpp';
    return [
        ['Qwen3-VL vision vs llama.cpp (orchestration)',
            'python3 version/v8/scripts/xray_qwen3vl_llamacpp_v8.py \\\n'
            + '  --gguf <mmproj.gguf> --image <image> --layer 0 --threads 1 \\\n'
            + '  --output-dir ' + llamaOut],
        ['Qwen3-VL vision vs PyTorch (bf16)',
            'python3 version/v8/scripts/xray_qwen3vl_bf16_v8.py \\\n'
            + '  --checkpoint <hf_qwen3vl_ckpt> --runtime-dir <runtime_dir> \\\n'
            + '  --weights-bump <weights.bump> --call-ir <call.json> --image <image> \\\n'
            + '  --output-dir build/xray/qwen3vl_bf16'],
        ['Whisper encoder vs PyTorch',
            'python3 version/v8/scripts/compare_whisper_encoder_pytorch_v8.py \\\n'
            + '  --run-dir ' + (c.runDir ? quoteShell(c.runDir) : '<run_dir>') + ' \\\n'
            + '  --checkpoint <hf_whisper_dir> --stops key \\\n'
            + '  --output ' + (c.runDir ? quoteShell(c.runDir + '/whisper-encoder-xray.json') : '<run_dir>/whisper-encoder-xray.json')],
        ['Decoder vs PyTorch (teacher-forced)',
            'python3 version/v8/scripts/xray_decoder_pytorch_v8.py \\\n'
            + '  --checkpoint <hf_decoder_ckpt> --runtime <libdecoder.so> --call-ir <call.json> \\\n'
            + '  --token-ids 1,2,3 --output-dir build/xray/decoder_pytorch'],
    ];
}

function buildXrayPanelEmptyState(panel, ctx) {
    const c = xrayEmptyStateContext(ctx);
    const hdr = (msg) => `<div style="font-size:0.72rem; color:var(--text-muted); margin-bottom:0.4rem;">${escapeHtml(msg)}</div>`;
    const section = (label) => `<div style="font-size:0.64rem; color:var(--orange); margin:0.6rem 0 0.3rem; text-transform:uppercase; letter-spacing:0.05em;">${escapeHtml(label)}</div>`;
    const expected = c.xrayDir ? (c.xrayDir + '/') : '<xray_dir>/';

    if (panel === 'ranking') {
        return hdr('No ranking report loaded (expected ' + expected + 'xray_ranking_report.json).')
            + section('Produces evidence')
            + xrayEmptyStateCopyBlock(
                'Logits ranking vs oracle',
                xrayRankingProducerCommand(c),
                'Ranking stays an independent, unscoped card unless complete matching provenance is supplied: run_id, phase, model_sha256, subject backend + runtime + generated-model hashes, oracle backend + fingerprint.')
            + section('Refresh only (produces no evidence)')
            + xrayEmptyStateCopyBlock('Regenerate this report', xrayRefreshCommand(c), null);
    }

    if (panel === 'trace') {
        return hdr('No execution trace or execution-state report loaded (expected '
                + expected + 'xray_execution_trace.json and ' + expected + 'xray_execution_state_report.json).')
            + section('Prerequisite missing')
            + hdr('xray_execution_trace.json has no in-tree producer yet: traces must be emitted by an '
                + 'instrumented runtime run following version/v8/schemas/xray_execution_trace.schema.json. '
                + 'The state-report command below is unusable until both trace files exist.')
            + section('Produces evidence (needs both traces first)')
            + xrayEmptyStateCopyBlock('Execution-state report from traces', xrayExecutionStateProducerCommand(c), null)
            + section('Refresh only (produces no evidence)')
            + xrayEmptyStateCopyBlock('Regenerate this report', xrayRefreshCommand(c), null);
    }

    if (panel === 'circuit') {
        const scope = (ctx && ctx.scope) || 'unknown';
        const artifact = scope === 'vision_encoder' || scope === 'audio_encoder'
            ? 'multimodal_bridge/encoder/call.json'
            : (scope === 'mixed_prefill' ? 'lowered_prefill_call.json' : 'lowered_decode_call.json');
        let html = hdr('No call IR loaded for circuit scope ' + scope + ' (expected artifact: ' + artifact + ').');
        if (scope === 'vision_encoder') {
            html += section('Produces the call IR')
                + xrayEmptyStateCopyBlock(
                    'Multimodal bridge build (materializes encoder call.json)',
                    'python3 version/v8/scripts/run_multimodal_bridge_v8.py \\\n'
                    + '  --decoder-gguf <decoder.gguf> --encoder-gguf <mmproj.gguf> \\\n'
                    + '  --workdir <workdir>',
                    'Writes multimodal_bridge/encoder/call.json under the model cache.');
        } else {
            html += section('Produces the call IR')
                + hdr('The scoped call IR is materialized by the model/bridge build pipeline for this run; '
                    + 'no single X-Ray producer command exists for it.');
        }
        html += section('Refresh only (produces no evidence)')
            + xrayEmptyStateCopyBlock('Regenerate this report', xrayRefreshCommand(c), null);
        return html;
    }

    if (panel === 'drift') {
        let html = hdr('No checkpoint-table X-ray report loaded (vision parity, whisper encoder, or decoder drift).')
            + section('Produces evidence');
        xrayDriftProducerCommands(c).forEach(([title, cmd]) => {
            html += xrayEmptyStateCopyBlock(title, cmd, null);
        });
        html += section('Refresh only (produces no evidence)')
            + xrayEmptyStateCopyBlock('Regenerate this report', xrayRefreshCommand(c), null);
        return html;
    }

    return hdr('No X-ray artifact loaded for this panel.');
}


// -- Explain-this-operation panel (canonical copies from ir_visualizer.html) --
// Keep in sync with the Explain tab implementation.

const EXPLAIN_DOC_LINKS = [
    { label: 'Kernel Maps - anatomy of a kernel map', href: 'kernel-maps.html#anatomy' },
    { label: 'Kernel Maps - the resolution algorithm', href: 'kernel-maps.html#resolution' },
    { label: 'Kernel Maps - a real resolver trace', href: 'kernel-maps.html#selection-trace' },
    { label: 'Codegen - v8 pipeline and failure table', href: 'codegen.html' },
];

function explainModePayloads(files, mode) {
    files = files || {};
    if (mode === 'prefill') {
        return {
            ir1: files.ir1_prefill || null,
            call: files.lowered_prefill_call || files.lowered_prefill || null,
        };
    }
    return {
        ir1: files.ir1_decode || null,
        call: files.lowered_decode_call || files.lowered_decode || null,
    };
}

function explainOpsOf(payload) {
    if (!payload || typeof payload !== 'object') return [];
    if (Array.isArray(payload.operations)) return payload.operations;
    if (Array.isArray(payload.ops)) return payload.ops;
    return [];
}

function explainIndexRegistry(registry) {
    const index = {};
    const kernels = registry && Array.isArray(registry.kernels) ? registry.kernels : [];
    kernels.forEach(k => { if (k && k.id != null) index[String(k.id)] = k; });
    return index;
}

function explainCollectOps(files, mode) {
    // Pair IR1 ops with call-IR ops by op name + layer, preferring an
    // identical op_id/idx, then first unmatched occurrence. Positional
    // pairing would silently join the wrong op when stages insert or
    // drop ops between IR levels.
    const payloads = explainModePayloads(files, mode);
    const ir1Ops = explainOpsOf(payloads.ir1);
    const callOps = explainOpsOf(payloads.call);
    const callUsed = new Array(callOps.length).fill(false);
    const rows = [];

    function keyOf(name, layer) {
        return String(name || '') + '|' + String(layer == null ? '' : layer);
    }

    ir1Ops.forEach((ir1Op) => {
        const key = keyOf(ir1Op.op, ir1Op.layer != null ? ir1Op.layer : null);
        const ir1Id = ir1Op.op_id != null ? ir1Op.op_id : null;
        let partner = -1;
        if (ir1Id != null) {
            for (let i = 0; i < callOps.length; i++) {
                if (callUsed[i]) continue;
                if (callOps[i].idx === ir1Id && keyOf(callOps[i].op, callOps[i].layer != null ? callOps[i].layer : null) === key) {
                    partner = i;
                    break;
                }
            }
        }
        if (partner < 0) {
            for (let i = 0; i < callOps.length; i++) {
                if (callUsed[i]) continue;
                if (keyOf(callOps[i].op, callOps[i].layer != null ? callOps[i].layer : null) === key) {
                    partner = i;
                    break;
                }
            }
        }
        if (partner >= 0) callUsed[partner] = true;
        const callOp = partner >= 0 ? callOps[partner] : null;
        rows.push({
            mode: mode,
            ref: callOp && callOp.idx != null ? ('call:' + callOp.idx) : ('ir1:' + (ir1Id != null ? ir1Id : rows.length)),
            op: ir1Op.op || (callOp && callOp.op) || '',
            layer: ir1Op.layer != null ? ir1Op.layer : (callOp && callOp.layer != null ? callOp.layer : null),
            section: ir1Op.section || (callOp && callOp.section) || null,
            kernel: ir1Op.kernel || null,
            hasIr1: true,
            hasCall: !!callOp,
            ir1Op: ir1Op,
            callOp: callOp,
        });
    });

    callOps.forEach((callOp, i) => {
        if (callUsed[i]) return;
        rows.push({
            mode: mode,
            ref: callOp.idx != null ? ('call:' + callOp.idx) : ('callpos:' + i),
            op: callOp.op || '',
            layer: callOp.layer != null ? callOp.layer : null,
            section: callOp.section || null,
            kernel: (callOp.call_abi && callOp.call_abi.kernel_id) || callOp.function || null,
            hasIr1: false,
            hasCall: true,
            ir1Op: null,
            callOp: callOp,
        });
    });
    return rows;
}

function explainDataflowSummary(dataflow) {
    const df = dataflow && typeof dataflow === 'object' ? dataflow : {};
    const summarize = (ports) => Object.keys(ports || {}).map(name => {
        const p = ports[name] || {};
        const bits = [name];
        if (p.dtype) bits.push(String(p.dtype));
        if (p.slot) bits.push('slot ' + String(p.slot));
        else if (p.from) bits.push('from ' + String(p.from));
        return bits.join(' / ');
    });
    return { inputs: summarize(df.inputs), outputs: summarize(df.outputs) };
}

function explainBuildChain(files, registry, mode, ref) {
    const payloads = explainModePayloads(files, mode);
    const rows = explainCollectOps(files, mode);
    const row = rows.find(r => r.ref === ref) || null;
    const regIndex = explainIndexRegistry(registry);
    const hasRegistry = !!(registry && Array.isArray(registry.kernels));
    if (!row) {
        return {
            found: false,
            mode: mode,
            ref: ref,
            reason: 'no explanation available for this op',
            stages: [],
            docs: EXPLAIN_DOC_LINKS,
        };
    }
    const ir1Op = row.ir1Op;
    const callOp = row.callOp;
    const stages = [];

    // 1. Circuit operation (ir1_*).
    if (ir1Op) {
        const df = explainDataflowSummary(ir1Op.dataflow);
        const iv = ir1Op.interface_validation || {};
        stages.push({
            id: 'circuit', label: 'Circuit operation', status: 'resolved',
            fields: [
                ['op', ir1Op.op],
                ['kernel', ir1Op.kernel],
                ['layer', ir1Op.layer != null ? ir1Op.layer : '-'],
                ['section', ir1Op.section || '-'],
                ['inputs', df.inputs.length ? df.inputs.join('; ') : '-'],
                ['outputs', df.outputs.length ? df.outputs.join('; ') : '-'],
                ['interface', iv.operation_interface || '-'],
            ],
            summary: String(ir1Op.op || '') + ' (layer ' + (ir1Op.layer != null ? ir1Op.layer : '-') + ')',
        });
    } else {
        stages.push({
            id: 'circuit', label: 'Circuit operation', status: 'not_generated',
            detail: 'ir1_' + mode + '.json was not loaded for this run; the circuit operation view is unavailable.',
        });
    }

    // 2. Required contract (call IR first, circuit interface join second).
    const rc = callOp && callOp.required_contract ? callOp.required_contract : null;
    const iv = ir1Op && ir1Op.interface_validation ? ir1Op.interface_validation : null;
    if (rc && rc.contract_id) {
        stages.push({
            id: 'contract', label: 'Required contract', status: 'resolved',
            fields: [
                ['contract_id', rc.contract_id],
                ['validation', rc.validation || '-'],
                ['evidence', rc.evidence || '-'],
            ],
            summary: String(rc.contract_id),
        });
    } else if (iv && iv.operation_interface) {
        stages.push({
            id: 'contract', label: 'Required contract', status: 'resolved',
            fields: [
                ['operation_interface', iv.operation_interface],
                ['join_status', iv.status || '-'],
                ['schema', iv.schema || '-'],
            ],
            detail: 'From the circuit interface join; the lowered call IR was not loaded for this op.',
            summary: String(iv.operation_interface),
        });
    } else {
        stages.push({
            id: 'contract', label: 'Required contract',
            status: callOp || ir1Op ? 'missing' : 'not_generated',
            detail: callOp || ir1Op
                ? 'The loaded artifacts carry no required-contract record for this op.'
                : 'lowered_' + mode + '_call.json was not loaded; nothing past the circuit stage was generated for this op.',
        });
    }

    // 3. Provider selection (call_abi + resolved contract).
    const abi = callOp && callOp.call_abi ? callOp.call_abi : null;
    const resolved = callOp && callOp.resolved_contract ? callOp.resolved_contract : null;
    if (abi || resolved) {
        stages.push({
            id: 'provider', label: 'Provider selection', status: 'resolved',
            fields: [
                ['provider', (abi && abi.kernel_id) || (resolved && resolved.kernel_id) || '-'],
                ['binding_owner', (abi && abi.owner) || '-'],
                ['map_file', (abi && abi.source_file) || '-'],
                ['function', (resolved && resolved.function) || (callOp && callOp.function) || '-'],
                ['resolved_contract', (resolved && (resolved.resolved_contract_id || resolved.contract_id)) || '-'],
                ['selector', resolved && resolved.selector != null ? resolved.selector : 'default ranking'],
            ],
            summary: String((abi && abi.kernel_id) || (resolved && resolved.kernel_id) || ''),
        });
    } else {
        stages.push({
            id: 'provider', label: 'Provider selection',
            status: callOp ? 'missing' : 'not_generated',
            detail: callOp
                ? 'The call IR carries no call_abi/resolved_contract for this op.'
                : 'lowered_' + mode + '_call.json was not loaded; provider selection output was not generated.',
        });
    }

    // 4. Kernel map (registry entry).
    const kernelId = (abi && abi.kernel_id) || (resolved && resolved.kernel_id)
        || (callOp && callOp.function) || (ir1Op && ir1Op.kernel) || null;
    const mapEntry = kernelId != null ? regIndex[String(kernelId)] || null : null;
    if (mapEntry) {
        const caps = Array.isArray(mapEntry.numerical_capabilities) ? mapEntry.numerical_capabilities : [];
        const cap0 = caps.length ? caps[0] : null;
        const quant = mapEntry.quant || {};
        stages.push({
            id: 'kernel_map', label: 'Kernel map', status: 'resolved',
            fields: [
                ['map_id', mapEntry.id],
                ['operation_interface', mapEntry.operation_interface || '-'],
                ['variant', mapEntry.variant || '-'],
                ['quant', ['weight=' + (quant.weight || '-'), 'activation=' + (quant.activation || '-'), 'output=' + (quant.output || '-')].join(' ')],
                ['numerical_contract', cap0 ? String(cap0.contract_id) + ' (' + String(cap0.status || '-') + ')' : '-'],
            ],
            summary: String(mapEntry.id || ''),
        });
    } else {
        stages.push({
            id: 'kernel_map', label: 'Kernel map',
            status: kernelId == null ? 'not_generated' : 'missing',
            detail: kernelId == null
                ? 'No provider was resolved, so no kernel map entry applies.'
                : (hasRegistry
                    ? 'No kernel map entry "' + kernelId + '" exists in the loaded registry.'
                    : 'The kernel registry was not loaded; the kernel map stage cannot be shown.'),
        });
    }

    // 5. Generated call.
    if (callOp && callOp.function) {
        const args = Array.isArray(callOp.args) ? callOp.args : [];
        stages.push({
            id: 'call', label: 'Generated call', status: 'resolved',
            fields: [
                ['function', callOp.function],
                ['errors', Array.isArray(callOp.errors) && callOp.errors.length ? callOp.errors.join('; ') : 'none'],
            ],
            args: args.map(a => ({
                name: a && a.name != null ? String(a.name) : '?',
                expr: a && a.expr != null ? String(a.expr) : '',
                source: a && a.source != null ? String(a.source) : '',
            })),
            summary: String(callOp.function) + '(' + args.length + ' args)',
        });
    } else {
        stages.push({
            id: 'call', label: 'Generated call',
            status: callOp ? 'missing' : 'not_generated',
            detail: callOp
                ? 'The call IR entry for this op carries no function/args.'
                : 'lowered_' + mode + '_call.json was not loaded; no generated call exists for this op.',
        });
    }

    // 6. C implementation / tests (kernel map impl + tests).
    const impl = mapEntry && mapEntry.impl ? mapEntry.impl : null;
    if (impl && impl.function) {
        const tests = mapEntry.tests || {};
        const unitTests = Array.isArray(tests.unit) ? tests.unit : [];
        const parityTests = Array.isArray(tests.parity) ? tests.parity : [];
        stages.push({
            id: 'c_impl', label: 'C implementation / tests', status: 'resolved',
            fields: [
                ['function', impl.function],
                ['sources', Array.isArray(impl.sources) && impl.sources.length ? impl.sources.join('; ') : '-'],
                ['unit_tests', unitTests.length ? unitTests.join('; ') : '-'],
                ['parity', parityTests.length ? parityTests.map(p => String((p && p.oracle) || p)).join('; ') : '-'],
            ],
            summary: String(impl.function),
        });
    } else {
        stages.push({
            id: 'c_impl', label: 'C implementation / tests',
            status: mapEntry ? 'missing' : 'not_generated',
            detail: mapEntry
                ? 'The kernel map entry carries no impl block.'
                : 'No kernel map entry was resolved, so the C implementation stage does not apply.',
        });
    }

    return {
        found: true,
        mode: mode,
        ref: ref,
        op: { op: row.op, layer: row.layer, section: row.section, kernel: row.kernel },
        kernelId: kernelId,
        stages: stages,
        docs: EXPLAIN_DOC_LINKS,
    };
}

function explainProvenanceRows(prov) {
    return prov && Array.isArray(prov.artifacts) ? prov.artifacts : [];
}

function explainStalenessSummary(prov) {
    const rows = explainProvenanceRows(prov);
    const counts = { match: 0, stale: 0, no_stamp: 0, not_loaded: 0 };
    const messages = [];
    rows.forEach(r => {
        const status = String(r.status || 'not_loaded');
        if (counts[status] != null) counts[status] += 1;
        if (status === 'stale') {
            messages.push(String(r.key) + ': artifact on disk differs from its ' + String(r.stamp_bundle || '.ck bundle') + ' stamp; this stage may come from a different build (stale or mixed-revision).');
        } else if (status === 'no_stamp') {
            messages.push(String(r.key) + ': no bundle stamp records this artifact; revision provenance is unavailable for it.');
        }
    });
    return { counts: counts, messages: messages, stale: counts.stale > 0 };
}

function explainStatusChip(status) {
    if (status === 'resolved' || status === 'available' || status === 'match') return '<span class="status-chip status-chip-good">' + escapeHtml(status) + '</span>';
    if (status === 'failed' || status === 'stale') return '<span class="status-chip status-chip-bad">' + escapeHtml(status) + '</span>';
    if (status === 'missing' || status === 'no_stamp') return '<span class="status-chip status-chip-warn">' + escapeHtml(String(status).replace(/_/g, ' ')) + '</span>';
    return '<span class="status-chip status-chip-muted">' + escapeHtml(String(status).replace(/_/g, ' ')) + '</span>';
}

function buildExplainEmptyStateHtml(reason) {
    return '<div style="margin-top:0.5rem; padding:0.75rem 1rem; background:rgba(255,180,0,0.08); border:1px solid rgba(255,180,0,0.3); border-radius:8px; font-size:0.74rem; color:var(--text-secondary); line-height:1.6;">'
        + '<strong style="color:var(--orange);">No explanation available for this op.</strong>'
        + '<div style="margin-top:0.3rem;">' + escapeHtml(reason || 'This run carries no IR or call artifacts the explanation chain can be derived from.') + '</div>'
        + '</div>';
}

function buildExplainChainHtml(chain) {
    if (!chain || !chain.found) {
        return buildExplainEmptyStateHtml(chain && chain.reason ? chain.reason : 'no explanation available for this op');
    }
    let html = '';
    chain.stages.forEach(stage => {
        html += '<div style="border:1px solid var(--grey); border-radius:8px; padding:0.6rem 0.9rem; margin-top:0.5rem;">';
        html += '<div style="display:flex; justify-content:space-between; align-items:center; gap:0.5rem;">'
            + '<strong>' + escapeHtml(stage.label) + '</strong>' + explainStatusChip(stage.status) + '</div>';
        if (Array.isArray(stage.fields)) {
            html += '<table style="margin-top:0.4rem; font-size:0.76rem; width:100%;">';
            stage.fields.forEach(([k, v]) => {
                html += '<tr><td style="color:var(--text-muted); padding-right:0.75rem; white-space:nowrap; vertical-align:top;">' + escapeHtml(String(k)) + '</td>'
                    + '<td style="word-break:break-word;"><code>' + escapeHtml(v == null ? '-' : String(v)) + '</code></td></tr>';
            });
            html += '</table>';
        }
        if (Array.isArray(stage.args) && stage.args.length) {
            html += '<pre style="font-size:0.72rem; white-space:pre-wrap; margin-top:0.4rem;">'
                + escapeHtml(stage.args.map(a => a.name.padEnd(18) + ' ' + a.expr + (a.source ? '    (' + a.source + ')' : '')).join('\n'))
                + '</pre>';
        }
        if (stage.detail) {
            html += '<div style="margin-top:0.35rem; font-size:0.74rem; color:var(--text-muted);">' + escapeHtml(stage.detail) + '</div>';
        }
        html += '</div>';
    });
    if (Array.isArray(chain.docs) && chain.docs.length) {
        html += '<div style="margin-top:0.6rem; font-size:0.74rem; color:var(--text-muted);">Docs: '
            + chain.docs.map(d => '<a href="' + escapeHtml(d.href) + '">' + escapeHtml(d.label) + '</a>').join(' | ')
            + '</div>';
    }
    return html;
}

function buildExplainProvenanceHtml(prov) {
    if (!prov || !Array.isArray(prov.artifacts) || !prov.artifacts.length) {
        return '<div style="font-size:0.74rem; color:var(--text-muted);">No provenance data embedded for this run.</div>';
    }
    const summary = explainStalenessSummary(prov);
    let html = '<div style="font-size:0.74rem; margin-bottom:0.4rem;">'
        + '<span class="status-chip ' + (prov.strict_run_artifacts ? 'status-chip-good' : 'status-chip-warn') + '">strict run artifacts: ' + (prov.strict_run_artifacts ? 'on' : 'off') + '</span>'
        + ' <span class="status-chip status-chip-good">match: ' + summary.counts.match + '</span>'
        + ' <span class="status-chip ' + (summary.counts.stale ? 'status-chip-bad' : 'status-chip-muted') + '">stale: ' + summary.counts.stale + '</span>'
        + ' <span class="status-chip ' + (summary.counts.no_stamp ? 'status-chip-warn' : 'status-chip-muted') + '">no stamp: ' + summary.counts.no_stamp + '</span>'
        + ' <span class="status-chip status-chip-muted">not loaded: ' + summary.counts.not_loaded + '</span>'
        + '</div>';
    if (summary.messages.length) {
        html += '<div style="margin-bottom:0.4rem; padding:0.5rem 0.75rem; background:rgba(231,76,60,0.08); border:1px solid rgba(231,76,60,0.3); border-radius:8px; font-size:0.74rem; color:var(--text-secondary);">'
            + summary.messages.map(m => '<div>' + escapeHtml(m) + '</div>').join('') + '</div>';
    }
    html += '<table style="font-size:0.74rem; width:100%;">';
    html += '<tr><th style="text-align:left;">artifact</th><th style="text-align:left;">status</th><th style="text-align:left;">loaded sha256</th><th style="text-align:left;">stamped sha256</th></tr>';
    prov.artifacts.forEach(r => {
        html += '<tr><td><code>' + escapeHtml(String(r.key)) + '</code></td><td>' + explainStatusChip(r.status) + '</td>'
            + '<td style="word-break:break-all; color:var(--text-muted);">' + escapeHtml(r.loaded_sha256 ? String(r.loaded_sha256).slice(0, 16) + '...' : '-') + '</td>'
            + '<td style="word-break:break-all; color:var(--text-muted);">' + escapeHtml(r.stamped_sha256 ? String(r.stamped_sha256).slice(0, 16) + '...' : '-') + '</td></tr>';
    });
    html += '</table>';
    const roots = Array.isArray(prov.planner_source_roots) ? prov.planner_source_roots : [];
    const commits = Array.isArray(prov.embedded_commits) ? prov.embedded_commits : [];
    const stamps = Array.isArray(prov.bundle_stamps) ? prov.bundle_stamps : [];
    html += '<div style="margin-top:0.4rem; font-size:0.72rem; color:var(--text-muted);">';
    if (stamps.length) html += '<div>bundle stamps: ' + stamps.map(s => '<code>' + escapeHtml(String(s)) + '</code>').join(' ') + '</div>';
    if (roots.length) html += '<div>producing checkout(s): ' + roots.map(s => '<code>' + escapeHtml(String(s)) + '</code>').join(' ') + '</div>';
    if (commits.length) {
        html += '<div>embedded commit stamps: ' + commits.map(c => '<code>' + escapeHtml(c.origin + '=' + c.value) + '</code>').join(' ') + '</div>';
    } else {
        html += '<div>embedded commit stamps: none recorded in the loaded artifacts</div>';
    }
    html += '</div>';
    return html;
}

function buildExplainReportHtml(name, report) {
    const reconstructed = report.reconstructed === true;
    let html = '<div style="border:1px solid var(--grey); border-radius:8px; padding:0.75rem 1rem; margin-top:0.6rem;">';
    html += '<div style="display:flex; justify-content:space-between; align-items:center; gap:0.5rem; flex-wrap:wrap;">'
        + '<strong>' + escapeHtml(String(report.title || name)) + '</strong>'
        + '<span>' + (reconstructed ? '<span class="status-chip status-chip-warn">reconstructed fixture</span> ' : '')
        + '<span class="status-chip status-chip-muted">' + escapeHtml(String(report.kind || 'report')) + '</span></span></div>';
    const req = report.requested || {};
    html += '<div style="margin-top:0.4rem; font-size:0.76rem;">requested: <code>' + escapeHtml(String(req.op || '-')) + '</code>'
        + ' phase <code>' + escapeHtml(String(req.phase || '-')) + '</code>'
        + (req.layer != null ? ' layer <code>' + escapeHtml(String(req.layer)) + '</code>' : '')
        + (req.circuit ? ' circuit <code>' + escapeHtml(String(req.circuit)) + '</code>' : '') + '</div>';
    if (Array.isArray(report.stages) && report.stages.length) {
        html += '<div style="margin-top:0.4rem;">';
        report.stages.forEach(s => {
            html += '<div style="font-size:0.74rem; margin-top:0.2rem;">' + explainStatusChip(s.status)
                + ' <strong>' + escapeHtml(String(s.stage).replace(/_/g, ' ')) + '</strong>'
                + (s.fault ? ' <code>' + escapeHtml(String(s.fault)) + '</code>' : '')
                + (s.detail ? '<div style="color:var(--text-muted); margin-left:0.4rem;">' + escapeHtml(String(s.detail)) + '</div>' : '')
                + '</div>';
        });
        html += '</div>';
    }
    if (Array.isArray(report.candidates) && report.candidates.length) {
        html += '<table style="margin-top:0.5rem; font-size:0.74rem; width:100%;">'
            + '<tr><th style="text-align:left;">provider</th><th style="text-align:left;">decision</th><th style="text-align:left;">stage</th><th style="text-align:left;">reason</th></tr>';
        report.candidates.forEach(c => {
            html += '<tr><td><code>' + escapeHtml(String(c.provider || '-')) + '</code></td>'
                + '<td>' + escapeHtml(String(c.decision || '-')) + '</td>'
                + '<td>' + escapeHtml(String(c.stage || '-')) + '</td>'
                + '<td><code>' + escapeHtml(String(c.reason || '-')) + '</code>'
                + (c.note ? '<div style="color:var(--text-muted);">' + escapeHtml(String(c.note)) + '</div>' : '')
                + '</td></tr>';
        });
        html += '</table>';
        html += '<div style="font-size:0.74rem; margin-top:0.25rem;">selected: <code>' + escapeHtml(report.selected == null ? 'none' : String(report.selected)) + '</code></div>';
    }
    const pm = report.port_mismatch;
    if (pm && typeof pm === 'object') {
        html += '<div style="margin-top:0.5rem; font-size:0.74rem;">'
            + '<div>declared ports: <code>' + escapeHtml((Array.isArray(pm.declared) ? pm.declared : []).map(p => String(p.port) + ' -> ' + String(p.slot || '')).join(', ')) + '</code></div>'
            + '<div>canonical ports: <code>' + escapeHtml((Array.isArray(pm.canonical) ? pm.canonical : []).map(p => String(p.port) + ' (' + String(p.role || '') + ')').join(', ')) + '</code></div>'
            + (pm.secondary ? '<div style="color:var(--text-muted);">' + escapeHtml(String(pm.secondary)) + '</div>' : '')
            + '</div>';
    }
    if (report.guidance) {
        html += '<div style="margin-top:0.5rem; padding:0.5rem 0.75rem; background:rgba(255,180,0,0.08); border:1px solid rgba(255,180,0,0.3); border-radius:8px; font-size:0.74rem; color:var(--text-secondary);">'
            + escapeHtml(String(report.guidance)) + '</div>';
    }
    if (Array.isArray(report.source_refs) && report.source_refs.length) {
        html += '<div style="margin-top:0.4rem; font-size:0.72rem; color:var(--text-muted);">source: ' + report.source_refs.map(r => '<code>' + escapeHtml(String(r)) + '</code>').join(' ') + '</div>';
    }
    if (Array.isArray(report.verifying_tests) && report.verifying_tests.length) {
        html += '<div style="font-size:0.72rem; color:var(--text-muted);">verified by: ' + report.verifying_tests.map(r => '<code>' + escapeHtml(String(r)) + '</code>').join(' ') + '</div>';
    }
    if (Array.isArray(report.docs) && report.docs.length) {
        html += '<div style="margin-top:0.3rem; font-size:0.72rem; color:var(--text-muted);">Docs: '
            + report.docs.map(d => '<a href="' + escapeHtml(String(d.href || '')) + '">' + escapeHtml(String(d.label || d.href || '')) + '</a>').join(' | ')
            + '</div>';
    }
    html += '</div>';
    return html;
}

function explainBuildDiagnostic(opts) {
    // Compact bundle for a human or a smaller local agent: what was
    // requested, how far the chain resolved, why selection/build
    // failed, where the relevant source lives, which test verifies a
    // repair. Repair guidance is "where to look", never "apply this".
    opts = opts || {};
    const meta = opts.meta || {};
    const chain = opts.chain || null;
    const reports = opts.reports && typeof opts.reports === 'object' ? opts.reports : {};
    const prov = opts.provenance || null;
    const sourceRefs = [];
    const verifyingTests = [];
    const failureReports = Object.keys(reports).sort().map(name => {
        const r = reports[name] || {};
        (Array.isArray(r.source_refs) ? r.source_refs : []).forEach(s => { if (!sourceRefs.includes(String(s))) sourceRefs.push(String(s)); });
        (Array.isArray(r.verifying_tests) ? r.verifying_tests : []).forEach(s => { if (!verifyingTests.includes(String(s))) verifyingTests.push(String(s)); });
        const failedStage = (Array.isArray(r.stages) ? r.stages : []).find(s => s.status === 'failed') || null;
        return {
            name: name,
            kind: r.kind || null,
            title: r.title || null,
            reconstructed: r.reconstructed === true,
            requested: r.requested || null,
            failed_stage: failedStage ? { stage: failedStage.stage, fault: failedStage.fault || null, detail: failedStage.detail || null } : null,
            rejected: (Array.isArray(r.candidates) ? r.candidates : [])
                .filter(c => c.decision === 'rejected')
                .map(c => ({ provider: c.provider || null, stage: c.stage || null, reason: c.reason || null })),
            selected: r.selected != null ? r.selected : null,
        };
    });
    if (chain && chain.found && chain.kernelId) {
        const mapRef = 'version/v8/kernel_maps/' + String(chain.kernelId) + '.json';
        if (!sourceRefs.includes(mapRef)) sourceRefs.push(mapRef);
    }
    return {
        schema: 'cke.v8.explain_diagnostic',
        schema_version: 1,
        run: {
            model: meta.model || null,
            run_dir: meta.run_dir || null,
            strict_run_artifacts: !!meta.strict_run_artifacts,
            generated_at: meta.generated_at || null,
        },
        requested: opts.requested || null,
        chain: chain && chain.found
            ? chain.stages.map(s => ({
                stage: s.id,
                status: s.status,
                summary: s.summary || null,
                detail: s.detail || null,
            }))
            : null,
        chain_reason: chain && !chain.found ? (chain.reason || 'no explanation available for this op') : null,
        failure_reports: failureReports,
        provenance: prov ? {
            strict_run_artifacts: !!prov.strict_run_artifacts,
            bundle_stamps: Array.isArray(prov.bundle_stamps) ? prov.bundle_stamps : [],
            planner_source_roots: Array.isArray(prov.planner_source_roots) ? prov.planner_source_roots : [],
            embedded_commits: Array.isArray(prov.embedded_commits) ? prov.embedded_commits : [],
            artifact_status: explainProvenanceRows(prov).map(r => ({ key: r.key, status: r.status })),
            staleness: explainStalenessSummary(prov).messages,
        } : null,
        source_refs: sourceRefs,
        verifying_tests: verifyingTests,
        docs: EXPLAIN_DOC_LINKS,
        guidance: 'Repair guidance is limited to where to look: the circuit JSON, the candidate kernel map JSONs, and the resolver in version/v8/scripts/build_ir_v8.py. Nothing in this bundle is an automatically safe fix; do not apply changes from it without reading the referenced source.',
    };
}

// Export for test harness (CommonJS for Node.js compatibility)
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        formatBytes, normalizeShapeInput, formatShapeDisplay,
        normalizeMode, escapeHtml, quoteShell, normalizePathString,
        pathDirname, extractGgufStem, relativePathFromTo,
        xrayReportKindLabel, xrayBackendLabel, xrayReportStatus,
        xrayDriftRows, buildXrayRunbookHtml,
        xrayPhaseOf, xrayContractState, xrayCircuitRows,
        xrayGrowthEdges, xrayRunIdentityOf, xrayRankingMatchesReport, xrayBoardVerdict,
        xrayCircuitScope, xraySelectCircuitCallIr, xrayLiveKeyForFile,
        xrayEmptyStateContext, buildXrayPanelEmptyState,
        explainModePayloads, explainOpsOf, explainIndexRegistry,
        explainCollectOps, explainDataflowSummary, explainBuildChain,
        explainProvenanceRows, explainStalenessSummary, explainStatusChip,
        buildExplainEmptyStateHtml, buildExplainChainHtml,
        buildExplainProvenanceHtml, buildExplainReportHtml,
        explainBuildDiagnostic
    };
}
