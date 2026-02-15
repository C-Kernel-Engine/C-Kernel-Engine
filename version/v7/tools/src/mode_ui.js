import { ensurePanel, tabIdForElement } from './utils.js';

const SHARED_TABS = ['memory', 'kernels', 'profile'];
const INFERENCE_ONLY = ['interpretability', 'quantization', 'parity', 'dataflow', 'tests', 'stats'];
const TRAINING_TABS = [
    'train-dashboard',
    'training',
    'train-gradient',
    'train-parity',
    'train-grad-flow',
    'train-weights',
    'train-attention',
    'train-compare',
    'train-memory-canary',
];

function injectStyles() {
    if (document.getElementById('ck-v7-train-style')) return;
    const style = document.createElement('style');
    style.id = 'ck-v7-train-style';
    style.textContent = `
        :root {
            --train-fwd: #3b82f6;
            --train-bwd: #f59e0b;
            --train-opt: #10b981;
            --train-zero: #6b7280;
        }
        .report-mode-toggle { display: inline-flex; border: 1px solid var(--grey); border-radius: 6px; overflow: hidden; }
        .report-mode-toggle button {
            background: var(--dark-card);
            border: none;
            color: var(--text-secondary);
            padding: 0.45rem 0.8rem;
            cursor: pointer;
            font-size: 0.82rem;
            font-weight: 600;
        }
        .report-mode-toggle button.active {
            background: var(--orange);
            color: var(--dark);
        }
        .alert-item {
            border-left: 3px solid;
            padding: 10px 12px;
            background: var(--dark-card);
            border-radius: 0 6px 6px 0;
            margin: 0.45rem 0;
        }
        .alert-item.critical { border-color: #ef4444; }
        .alert-item.warning { border-color: #f59e0b; }
        .alert-item.info { border-color: #3b82f6; }

        .grad-healthy, .grad-warning, .grad-danger, .grad-dead {
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            display: inline-block;
        }
        .grad-healthy { background: #065f46; color: #6ee7b7; }
        .grad-warning { background: #78350f; color: #fcd34d; }
        .grad-danger  { background: #7f1d1d; color: #fca5a5; }
        .grad-dead    { background: #374151; color: #9ca3af; }

        .weight-matrix { display: grid; grid-template-columns: repeat(32, 1fr); gap: 1px; }
        .weight-cell { aspect-ratio: 1; min-width: 3px; border-radius: 1px; }

        .act-histogram { display: flex; align-items: flex-end; height: 150px; gap: 2px; }
        .hist-bar { flex: 1; border-radius: 2px 2px 0 0; min-width: 4px; }

        .mini-head-grid { display: grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap: 8px; }
        .mini-heatmap { cursor: pointer; border: 2px solid transparent; border-radius: 4px; }
        .mini-heatmap.selected { border-color: var(--orange); }

        .step-slider { width: 100%; accent-color: var(--orange); }
        .step-label {
            font-size: 0.8rem;
            color: var(--text-muted);
            font-family: 'JetBrains Mono', monospace;
        }

        .param-select, .layer-select, .head-select {
            background: var(--dark-card);
            color: var(--text-secondary);
            border: 1px solid var(--grey);
            border-radius: 4px;
            padding: 0.4rem 0.6rem;
            font-size: 0.82rem;
            min-width: 180px;
        }

        .drop-zone {
            border: 2px dashed var(--grey);
            border-radius: 8px;
            padding: 1.2rem;
            text-align: center;
            color: var(--text-muted);
            transition: var(--transition);
        }
        .drop-zone.drag-over {
            border-color: var(--orange);
            background: rgba(255, 180, 0, 0.05);
        }
    `;
    document.head.appendChild(style);
}

function ensureTabDataAttrs() {
    document.querySelectorAll('.tabs .tab').forEach((tab) => {
        if (!tab.dataset.tab) {
            const id = tabIdForElement(tab);
            if (id) tab.dataset.tab = id;
        }
        if (!tab.dataset.mode) tab.dataset.mode = 'inference';
    });
}

function setTabMode(tabId, modeCsv, label) {
    const tab = document.querySelector(`.tabs .tab[data-tab="${tabId}"]`);
    if (!tab) return;
    tab.dataset.mode = modeCsv;
    if (label) tab.textContent = label;
}

function appendTrainingTabs() {
    const tabsRoot = document.querySelector('.tabs');
    if (!tabsRoot) return;
    if (tabsRoot.querySelector('.tab[data-tab="train-dashboard"]')) return;

    const insertBefore = tabsRoot.querySelector('.tab[data-tab="profile"]');
    const specs = [
        { id: 'train-dashboard', label: 'Train Dashboard' },
        { id: 'train-gradient', label: 'Gradient Health' },
        { id: 'train-parity', label: 'Parity Tracker' },
        { id: 'train-grad-flow', label: 'Gradient Flow' },
        { id: 'train-weights', label: 'Weights & Activations' },
        { id: 'train-attention', label: 'Attention' },
        { id: 'train-compare', label: 'Run Compare' },
        { id: 'train-memory-canary', label: 'Memory & Canary' },
    ];
    specs.forEach((spec) => {
        const tab = document.createElement('div');
        tab.className = 'tab';
        tab.dataset.tab = spec.id;
        tab.dataset.mode = 'training';
        tab.setAttribute('onclick', `showTab('${spec.id}')`);
        tab.textContent = spec.label;
        tabsRoot.insertBefore(tab, insertBefore || null);
    });
}

function ensurePanelRoot(panelId, rootId) {
    const panel = document.getElementById(panelId);
    if (!panel) return;
    if (panel.querySelector(`#${rootId}`)) return;
    const root = document.createElement('div');
    root.id = rootId;
    panel.appendChild(root);
}

function ensureTrainingPanels() {
    ensurePanel(
        'train-dashboard',
        'Training Dashboard',
        'Loss, gradients, LR and parity health from training telemetry.'
    );
    ensurePanel(
        'train-gradient',
        'Gradient Health',
        'Per-parameter gradient norms and health buckets.'
    );
    ensurePanel(
        'train-parity',
        'Training Parity Tracker',
        'Step-level CK vs PyTorch numerical drift summary.'
    );
    ensurePanel(
        'train-grad-flow',
        'Gradient Flow',
        'Layer-wise gradient propagation (waterfall, ratio, heatmap).'
    );
    ensurePanel(
        'train-weights',
        'Weights & Activations',
        'Weight movement and activation/gradient distribution snapshots.'
    );
    ensurePanel(
        'train-attention',
        'Attention Inspector',
        'Q×K patterns, entropy timeline, and head redundancy.'
    );
    ensurePanel(
        'train-compare',
        'Run Compare',
        'Overlay current run vs baseline artifacts.'
    );
    ensurePanel(
        'train-memory-canary',
        'Memory & Canary Diagnostics',
        'Canary corruption checks, tensor slot map, and layout audit for generated train runtime.'
    );

    ensurePanelRoot('train-dashboard', 'trainDashboardRoot');
    ensurePanelRoot('train-gradient', 'trainGradientRoot');
    ensurePanelRoot('train-parity', 'trainParityRoot');
    ensurePanelRoot('train-grad-flow', 'trainGradFlowRoot');
    ensurePanelRoot('train-weights', 'trainWeightsRoot');
    ensurePanelRoot('train-attention', 'trainAttentionRoot');
    ensurePanelRoot('train-compare', 'trainCompareRoot');
    ensurePanelRoot('train-memory-canary', 'trainMemoryCanaryRoot');
}

function installModeToggle() {
    if (document.getElementById('ckReportModeToggle')) return;
    const controls = document.querySelector('.controls');
    if (!controls) return;
    const host = document.createElement('div');
    host.className = 'report-mode-toggle';
    host.id = 'ckReportModeToggle';
    host.innerHTML = `
        <button type="button" data-mode="inference" class="active">Inference</button>
        <button type="button" data-mode="training">Training</button>
    `;
    controls.appendChild(host);
}

function visibleTabElements() {
    return Array.from(document.querySelectorAll('.tabs .tab')).filter((tab) => tab.style.display !== 'none');
}

function firstVisibleTabId() {
    const first = visibleTabElements()[0];
    return first ? tabIdForElement(first) : '';
}

export function applyReportMode(mode) {
    const normalized = mode === 'training' ? 'training' : 'inference';
    window.__ckReportMode = normalized;

    document.querySelectorAll('#ckReportModeToggle button').forEach((btn) => {
        btn.classList.toggle('active', btn.dataset.mode === normalized);
    });

    document.querySelectorAll('.tabs .tab').forEach((tab) => {
        const allow = (tab.dataset.mode || 'inference').split(',').map((s) => s.trim());
        tab.style.display = allow.includes(normalized) ? '' : 'none';
    });

    const activeTab = document.querySelector('.tabs .tab.active');
    const activeTabId = tabIdForElement(activeTab);
    if (!activeTab || activeTab.style.display === 'none') {
        const nextTabId = normalized === 'training' ? 'train-dashboard' : firstVisibleTabId();
        if (nextTabId) {
            window.showTab(nextTabId);
        }
    } else if (normalized === 'training' && INFERENCE_ONLY.includes(activeTabId)) {
        window.showTab('train-dashboard');
    }
}

export function initModeUI() {
    injectStyles();
    ensureTabDataAttrs();
    setTabMode('memory', 'inference,training');
    setTabMode('kernels', 'inference,training');
    setTabMode('profile', 'inference,training');
    setTabMode('training', 'training', 'Backward IR');
    INFERENCE_ONLY.forEach((id) => setTabMode(id, 'inference'));

    appendTrainingTabs();
    ensureTrainingPanels();
    installModeToggle();

    document.querySelectorAll('#ckReportModeToggle button').forEach((btn) => {
        btn.addEventListener('click', () => applyReportMode(btn.dataset.mode || 'inference'));
    });

    applyReportMode('inference');
}

export function trainingTabIds() {
    return [...TRAINING_TABS, ...SHARED_TABS];
}
