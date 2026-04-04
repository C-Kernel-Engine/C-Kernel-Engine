const QUESTION_TABS = {
    'train-dashboard': {
        label: 'Is It Training?',
        title: 'Loss, gradients, learning rate and run health overview.',
    },
    'training': {
        label: 'How Does the Graph Flow?',
        title: 'Forward/backward IR topology, invariants and gate status.',
    },
    'train-gradient': {
        label: 'Are Gradients Healthy?',
        title: 'Per-parameter gradient norms and stability status.',
    },
    'train-parity': {
        label: 'Does It Match PyTorch?',
        title: 'Step-level numerical drift between CK and PyTorch.',
    },
    'train-grad-flow': {
        label: 'Are Gradients Reaching Every Layer?',
        title: 'Layer waterfall, ratios and propagation heatmap.',
    },
    'train-weights': {
        label: 'What Is Changing?',
        title: 'Weight movement ranking, heatmaps and activation distributions.',
    },
    'train-attention': {
        label: 'What Is It Attending To?',
        title: 'Q×K attention structure, entropy and head redundancy.',
    },
    'train-weight-health': {
        label: 'Is Every Parameter Learning?',
        title: 'Checkpoint delta health, gradient reachability, and stale-parameter flags.',
    },
    'train-compare': {
        label: 'How Does This Compare?',
        title: 'Overlay baseline vs current run and inspect deltas.',
    },
    'train-memory-canary': {
        label: 'Where Is Memory Safe?',
        title: 'Canary corruption checks, slot ownership, and layout audit status.',
    },
};

export function applyQuestionHeaders() {
    Object.entries(QUESTION_TABS).forEach(([tabId, spec]) => {
        const tab = document.querySelector(`.tabs .tab[data-tab="${tabId}"]`);
        if (!tab) return;
        tab.textContent = spec.label;
        tab.title = spec.title;
    });
}
