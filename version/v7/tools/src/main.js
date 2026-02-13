import { getEmbeddedFiles } from './utils.js';
import { initModeUI, trainingTabIds } from './mode_ui.js';
import { renderTrainingDashboard } from './training_dashboard.js';
import { renderTrainingExtensionTab } from './training_tabs.js';
import { renderGradientFlow } from './gradient_flow.js';
import { renderWeightActivation } from './weight_activation.js';
import { renderAttentionInspector } from './attention_inspector.js';
import { renderRunCompare } from './run_compare.js';
import { applyQuestionHeaders } from './question_headers.js';

function currentFiles() {
    return getEmbeddedFiles();
}

function renderTrainingTab(tabId) {
    const files = currentFiles();
    if (tabId === 'train-dashboard') {
        renderTrainingDashboard(files);
        return;
    }
    if (tabId === 'train-grad-flow') {
        renderGradientFlow(files);
        return;
    }
    if (tabId === 'train-weights') {
        renderWeightActivation(files);
        return;
    }
    if (tabId === 'train-attention') {
        renderAttentionInspector(files);
        return;
    }
    if (tabId === 'train-compare') {
        renderRunCompare(files);
        return;
    }
    renderTrainingExtensionTab(tabId, files);
}

function patchShowTab() {
    if (window.__ckV7ShowTabPatched) return;
    const original = window.showTab;
    if (typeof original !== 'function') return;

    window.showTab = function patchedShowTab(tabId) {
        const result = original(tabId);
        if (trainingTabIds().includes(tabId)) {
            renderTrainingTab(tabId);
        }
        return result;
    };
    window.__ckV7ShowTabPatched = true;
}

function installListeners() {
    window.addEventListener('ck:tab-shown', (event) => {
        const tabId = event && event.detail ? event.detail.tabId : '';
        if (!tabId) return;
        if (trainingTabIds().includes(tabId)) {
            renderTrainingTab(tabId);
        }
    });

    window.addEventListener('ckEmbeddedDataLoaded', () => {
        const active = document.querySelector('.tabs .tab.active');
        const tabId = active ? (active.dataset.tab || '') : '';
        if (trainingTabIds().includes(tabId)) {
            renderTrainingTab(tabId);
        }
    });
}

function init() {
    initModeUI();
    applyQuestionHeaders();
    patchShowTab();
    installListeners();

    const active = document.querySelector('.tabs .tab.active');
    const tabId = active ? (active.dataset.tab || '') : '';
    if (trainingTabIds().includes(tabId)) {
        renderTrainingTab(tabId);
    }
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
