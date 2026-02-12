import { ensurePanel, tabIdForElement } from './utils.js';

const SHARED_TABS = ['memory', 'kernels', 'profile'];
const INFERENCE_ONLY = ['interpretability', 'quantization', 'parity', 'dataflow', 'tests', 'stats'];
const TRAINING_TABS = ['train-dashboard', 'training', 'train-gradient', 'train-parity'];

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

function ensureTrainingPanels() {
    const dashboard = ensurePanel(
        'train-dashboard',
        'Training Dashboard',
        'Loss, gradients, LR and parity health from training telemetry.'
    );
    if (dashboard && !dashboard.querySelector('#trainDashboardRoot')) {
        const root = document.createElement('div');
        root.id = 'trainDashboardRoot';
        dashboard.appendChild(root);
    }

    const gradient = ensurePanel(
        'train-gradient',
        'Gradient Health',
        'Per-parameter gradient norms and health buckets.'
    );
    if (gradient && !gradient.querySelector('#trainGradientRoot')) {
        const root = document.createElement('div');
        root.id = 'trainGradientRoot';
        gradient.appendChild(root);
    }

    const parity = ensurePanel(
        'train-parity',
        'Training Parity Tracker',
        'Step-level CK vs PyTorch numerical drift summary.'
    );
    if (parity && !parity.querySelector('#trainParityRoot')) {
        const root = document.createElement('div');
        root.id = 'trainParityRoot';
        parity.appendChild(root);
    }
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
