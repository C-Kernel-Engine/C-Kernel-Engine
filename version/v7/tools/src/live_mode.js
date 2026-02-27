// Live polling mode for the IR Visualizer.
//
// TWO ways to activate:
//
//  A) Simple HTTP server (works out-of-the-box):
//       python3 -m http.server 7700 -d /path/to/run/dir
//     Then open  http://localhost:7700/ir_report.html
//     The visualizer auto-detects it is served over HTTP and starts polling
//     the JSON artifact files relative to the page URL.
//
//  B) Custom live server via  python3 open_ir_visualizer.py --live --run <dir>
//     Injects  window.CK_LIVE_MODE = { pollUrl: '/api/snapshot', intervalMs: 5000 }
//     and serves a combined snapshot endpoint (fewer round-trips).
//
// initLiveMode() is called from main.js init(). No-ops if neither condition holds.

const _LIVE_FILES = [
    'training_loss_curve_latest.json',
    'training_grad_norms_latest.json',
    'training_parity_latest.json',
    'training_step_profile_latest.json',
    'training_checkpoint_policy_latest.json',
];

const _LIVE_KEY_MAP = {
    'training_loss_curve_latest.json':        'training_loss_curve',
    'training_grad_norms_latest.json':         'training_grad_norms',
    'training_parity_latest.json':             'training_parity',
    'training_step_profile_latest.json':       'training_step_profile',
    'training_checkpoint_policy_latest.json':  'training_checkpoint_policy',
};

let _liveTimer  = null;
let _tickTimer  = null;
let _lastSeen   = {};   // fname → JSON string (for direct polling change detection)
let _lastMtimes = {};   // fname → mtime (for snapshot-mode change detection)
let _lastUpdateMs = null;

function initLiveMode() {
    const cfg = window.CK_LIVE_MODE || {};
    const isHttp = window.location.protocol !== 'file:';

    // Not in any live-capable mode → nothing to do
    if (!cfg.pollUrl && !isHttp) return;

    // Compute base URL for direct file polling (Mode A)
    // Strip query string and ensure trailing slash
    const href = window.location.href.split('?')[0];
    cfg._baseUrl = cfg._baseUrl || href.slice(0, href.lastIndexOf('/') + 1);
    cfg.intervalMs = Math.max(1000, Number(cfg.intervalMs) || 5000);

    _injectLiveStyle();
    _showLiveBar(cfg);

    const poll = cfg.pollUrl ? () => _pollSnapshot(cfg) : () => _pollFiles(cfg);
    poll().then(() => {
        _liveTimer = setInterval(poll, cfg.intervalMs);
    });
    _tickTimer = setInterval(_tickAge, 1000);
}

// ─── Mode A: poll individual files relative to page URL ──────────────────────

async function _pollFiles(cfg) {
    const changed = [];
    const fetches = _LIVE_FILES.map(async (fname) => {
        try {
            const resp = await fetch(cfg._baseUrl + fname + '?t=' + Date.now(), { cache: 'no-store' });
            if (!resp.ok) return;
            const text = await resp.text();
            if (text === _lastSeen[fname]) return; // unchanged
            _lastSeen[fname] = text;
            const content = JSON.parse(text);
            const key = _LIVE_KEY_MAP[fname];
            if (key) {
                const embedded = window.EMBEDDED_IR_DATA || {};
                (embedded.files = embedded.files || {})[key] = content;
                window.EMBEDDED_IR_DATA = embedded;
            }
            changed.push(fname);
        } catch (_) { /* file not yet present — normal during early training */ }
    });
    await Promise.all(fetches);
    if (changed.length > 0) _rerender();
    _lastUpdateMs = Date.now();
    _setLiveStatus((changed.length > 0 ? '↺ ' : '') + _buildSummary(), 'ok');
}

// ─── Mode B: single snapshot endpoint (custom --live server) ─────────────────

async function _pollSnapshot(cfg) {
    try {
        const resp = await fetch(cfg.pollUrl + '?t=' + Date.now(), { cache: 'no-store' });
        if (!resp.ok) { _setLiveStatus('poll error ' + resp.status, 'error'); return; }
        const snapshot = await resp.json();
        let anyChanged = false;
        for (const [fname, entry] of Object.entries(snapshot)) {
            if (!entry || entry.mtime === _lastMtimes[fname]) continue;
            _lastMtimes[fname] = entry.mtime;
            const key = _LIVE_KEY_MAP[fname];
            if (key && entry.content !== undefined) {
                const embedded = window.EMBEDDED_IR_DATA || {};
                (embedded.files = embedded.files || {})[key] = entry.content;
                window.EMBEDDED_IR_DATA = embedded;
            }
            anyChanged = true;
        }
        if (anyChanged) _rerender();
        _lastUpdateMs = Date.now();
        _setLiveStatus((anyChanged ? '↺ ' : '') + _buildSummary(), 'ok');
    } catch (e) {
        _setLiveStatus('network error — is the live server running?', 'error');
    }
}

// ─── Re-render active training tab ───────────────────────────────────────────

function _rerender() {
    const active = document.querySelector('.tabs .tab.active');
    const tabId = active ? (active.dataset.tab || '') : '';
    if (typeof renderTrainingTab === 'function' && tabId) {
        try { renderTrainingTab(tabId); } catch (_) {}
    }
}

// ─── Status bar summary line ──────────────────────────────────────────────────

function _buildSummary() {
    try {
        const files = (window.EMBEDDED_IR_DATA || {}).files || {};
        const lc    = files.training_loss_curve || {};
        const cp    = files.training_checkpoint_policy || {};
        const steps = Array.isArray(lc.steps) ? lc.steps : [];
        const last  = steps.length > 0 ? steps[steps.length - 1] : null;
        const step  = cp.latest_step ?? (last ? last.step : null);
        const loss  = last && last.loss_ck != null ? Number(last.loss_ck).toFixed(4) : null;
        const prof  = files.training_step_profile || {};
        const tokS  = prof.train_tok_s != null ? Number(prof.train_tok_s).toFixed(1) + ' tok/s' : null;
        const parts = [];
        if (step != null) parts.push('step ' + step);
        if (loss != null) parts.push('loss ' + loss);
        if (tokS)         parts.push(tokS);
        return parts.length > 0 ? parts.join(' · ') : 'no data yet';
    } catch (_) { return ''; }
}

function _tickAge() {
    if (_lastUpdateMs === null) return;
    const ageS = Math.round((Date.now() - _lastUpdateMs) / 1000);
    const el = document.getElementById('ck-live-age');
    if (el) el.textContent = ageS < 5 ? 'just now' : ageS + 's ago';
}

// ─── UI helpers ───────────────────────────────────────────────────────────────

function _injectLiveStyle() {
    if (document.getElementById('ck-live-style')) return;
    const s = document.createElement('style');
    s.id = 'ck-live-style';
    s.textContent = `
        @keyframes ck-live-pulse { 0%,100%{opacity:1;box-shadow:0 0 5px #47b475;} 50%{opacity:.3;box-shadow:none;} }
        #ck-live-bar {
            position:fixed;bottom:0;left:0;right:0;z-index:9999;
            background:#07100a;border-top:1px solid rgba(71,180,117,.28);
            padding:.3rem 1rem;display:flex;align-items:center;gap:.55rem;
            font-size:.75rem;color:#9ca3af;font-family:inherit;
        }
        #ck-live-dot {
            width:8px;height:8px;border-radius:50%;background:#47b475;flex-shrink:0;
            animation:ck-live-pulse 1.8s ease-in-out infinite;
        }
        #ck-live-dot.stopped{background:#4b5563;animation:none;box-shadow:none;}
    `;
    document.head.appendChild(s);
}

function _showLiveBar(cfg) {
    if (document.getElementById('ck-live-bar')) return;
    const bar = document.createElement('div');
    bar.id = 'ck-live-bar';
    const sec = Math.round(cfg.intervalMs / 1000);
    const mode = cfg.pollUrl ? 'snapshot' : 'file poll';
    bar.innerHTML = `
        <div id="ck-live-dot"></div>
        <span style="color:#6ee7b7;font-weight:700;letter-spacing:.04em;">LIVE</span>
        <span id="ck-live-msg" style="color:#9ca3af;">connecting…</span>
        <span style="color:#4b5563;font-size:.68rem;">·</span>
        <span id="ck-live-age" style="color:#6b7280;font-size:.69rem;"></span>
        <span style="margin-left:auto;display:flex;gap:.5rem;align-items:center;">
            <span style="color:#374151;font-size:.68rem;">${mode} · ${sec}s</span>
            <button id="ck-live-stop-btn"
                style="background:none;border:1px solid rgba(255,255,255,.1);border-radius:3px;
                       color:#6b7280;cursor:pointer;font-size:.68rem;padding:.1rem .4rem;line-height:1.4;">
                stop
            </button>
        </span>
    `;
    document.body.appendChild(bar);
    const btn = document.getElementById('ck-live-stop-btn');
    if (btn) btn.addEventListener('click', _stopLive);
}

function _setLiveStatus(msg, state) {
    const el = document.getElementById('ck-live-msg');
    if (el) {
        el.textContent = msg;
        el.style.color = state === 'error' ? '#fb923c' : '#9ca3af';
    }
}

function _stopLive() {
    if (_liveTimer)  { clearInterval(_liveTimer);  _liveTimer  = null; }
    if (_tickTimer)  { clearInterval(_tickTimer);  _tickTimer  = null; }
    const dot = document.getElementById('ck-live-dot');
    if (dot) dot.classList.add('stopped');
    _setLiveStatus('polling stopped', 'ok');
    const btn = document.getElementById('ck-live-stop-btn');
    if (btn) btn.style.display = 'none';
}
