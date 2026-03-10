import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';

const repoRoot = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const toolsSrcDir = path.join(repoRoot, 'version', 'v7', 'tools', 'src');

function toDataUrl(source) {
    return `data:text/javascript;base64,${Buffer.from(source, 'utf8').toString('base64')}`;
}

async function importTrainingTabsModule() {
    const utilsPath = path.join(toolsSrcDir, 'utils.js');
    const trainingTabsPath = path.join(toolsSrcDir, 'training_tabs.js');
    const utilsUrl = toDataUrl(fs.readFileSync(utilsPath, 'utf8'));
    const trainingTabsSource = fs.readFileSync(trainingTabsPath, 'utf8')
        .replace("from './utils.js';", `from '${utilsUrl}';`);
    return import(toDataUrl(trainingTabsSource));
}

function fakeRoot() {
    return {
        innerHTML: '',
        firstChild: null,
        removeChild() {},
        querySelectorAll() { return []; },
        querySelector() { return null; },
    };
}

function fakeDocument(root) {
    return {
        getElementById(id) {
            return id === 'trainDataLabRoot' ? root : null;
        },
    };
}

function sampleFiles() {
    return {
        training_parity: {
            steps: Array.from({ length: 480 }, (_, idx) => ({
                step: idx + 1,
                max_param_diff: 0.0,
                worst_param: 'stable',
            })),
        },
        train_e2e: { pass: true },
        training_parity_regimen: {
            summary: {
                passed: true,
                passed_stages: 13,
                total_stages: 13,
            },
            stages: [
                { id: 'A1', status: 'PASS', metrics: { max_loss_abs_diff: 0.0 } },
                { id: 'A3', status: 'PASS' },
                { id: 'A4', status: 'PASS' },
                { id: 'B2', status: 'PASS' },
                { id: 'B4', status: 'PASS' },
                { id: 'B8', status: 'PASS' },
                { id: 'C1', status: 'PASS' },
                { id: 'C2', status: 'PASS' },
                { id: 'C3', status: 'PASS' },
                { id: 'D1', status: 'PASS' },
                { id: 'D2', status: 'PASS' },
                { id: 'E1', status: 'PASS' },
                { id: 'F1', status: 'PASS' },
            ],
        },
        training_pipeline: {
            active_stage: 'pretrain',
            train_dims: {
                num_layers: 2,
                embed_dim: 64,
                hidden_dim: 128,
                vocab_size: 256,
                num_heads: 4,
                context_length: 512,
            },
            dataset_catalog: [],
            execution: {},
        },
    };
}

function sampleFilesMissingRegimen() {
    const files = sampleFiles();
    delete files.training_parity_regimen;
    return files;
}

test('Data Lab keeps parity summary and A/B/C/D/E/F readiness visible', async () => {
    const root = fakeRoot();
    global.window = {
        EMBEDDED_IR_DATA: {
            meta: {
                run_dir: '/tmp/spec04',
            },
        },
        location: { protocol: 'file:' },
    };
    global.document = fakeDocument(root);

    const { renderTrainingExtensionTab } = await importTrainingTabsModule();
    renderTrainingExtensionTab('train-data-lab', sampleFiles());

    const html = root.innerHTML;
    assert.match(html, /PyTorch Parity Gate/);
    assert.match(html, /PARITY REGIMEN<\/div>\s*<div[^>]*>PASS<\/div>/);
    assert.match(html, /13\/13 stages/);
    assert.match(html, /Training Readiness/);
    assert.match(html, /Parity Gate - Stage-by-Stage/);
    for (const stageId of ['A1', 'B2', 'C1', 'D1', 'E1', 'F1']) {
        assert.match(html, new RegExp(`>${stageId}<`));
    }
});

test('Data Lab keeps A/B/C/D/E/F cards visible when regimen json is missing', async () => {
    const root = fakeRoot();
    global.window = {
        EMBEDDED_IR_DATA: {
            meta: {
                run_dir: '/tmp/legacy-run',
            },
        },
        location: { protocol: 'file:' },
    };
    global.document = fakeDocument(root);

    const { renderTrainingExtensionTab } = await importTrainingTabsModule();
    renderTrainingExtensionTab('train-data-lab', sampleFilesMissingRegimen());

    const html = root.innerHTML;
    assert.match(html, /Parity regimen not loaded/);
    assert.match(html, /eval_stage_v7\.py/);
    assert.match(html, /run_training_parity_regimen_v7\.py --run-dir \/tmp\/legacy-run/);
    assert.match(html, /training_parity_regimen_latest\.json/);
    for (const stageId of ['A1', 'A3', 'A4', 'B2', 'B4', 'B8', 'C1', 'C2', 'C3', 'D1', 'D2', 'E1', 'F1']) {
        assert.match(html, new RegExp(`>${stageId}<`));
    }
    assert.match(html, />MISSING</);
});
