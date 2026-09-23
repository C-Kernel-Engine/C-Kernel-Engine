import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import vm from 'node:vm';

const repoRoot = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const visualizerPath = path.join(repoRoot, 'version', 'v8', 'tools', 'ir_visualizer.html');
const fixturePath = path.join(repoRoot, 'tests', 'fixtures', 'v8', 'training_visualizer_identity.json');

function loadRenderers() {
    const source = fs.readFileSync(visualizerPath, 'utf8');
    const start = source.indexOf('function buildTrainingExperimentIdentitySection(files)');
    const end = source.indexOf('\nfunction renderDataLabPanel(files)', start);
    assert.ok(start >= 0 && end > start, 'identity renderer must exist in the shipped visualizer');
    const context = {};
    vm.createContext(context);
    vm.runInContext(`
        function htmlEscape(value) {
            return String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        }
        ${source.slice(start, end)}
    `, context);
    return context;
}

test('identity-bound training panels render populated evidence', () => {
    const renderers = loadRenderers();
    const files = JSON.parse(fs.readFileSync(fixturePath, 'utf8'));
    const identityHtml = renderers.buildTrainingExperimentIdentitySection(files);
    assert.match(identityHtml, /Experiment Identity/);
    assert.match(identityHtml, /fixture-run-42/);
    assert.match(identityHtml, /MATCHED/);
    assert.match(identityHtml, /PASS/);
    assert.match(identityHtml, /109 forward · 209 backward ops/);
    assert.match(identityHtml, /Experiment summary/);
    assert.match(identityHtml, /Dataset tokens \+ masks/);

    const batchHtml = renderers.buildTrainingBatchPreviewSection(files);
    assert.match(batchHtml, /Serialized/);
    assert.match(batchHtml, /causal_next_token/);
    assert.match(batchHtml, /\[101,102,103,0\]/);
    assert.match(batchHtml, /\[102,103,104,0\]/);
    assert.match(batchHtml, /\[1,1,1,0\]/);
});

test('identity panel keeps failed, missing, and mismatched evidence distinct', () => {
    const renderers = loadRenderers();
    const files = JSON.parse(fs.readFileSync(fixturePath, 'utf8'));

    const failed = structuredClone(files);
    failed.training_experiment_manifest.verdict = { status: 'FAIL', passed: false };
    assert.match(renderers.buildTrainingExperimentIdentitySection(failed), /FAIL/);

    const mismatched = structuredClone(files);
    mismatched.training_experiment_manifest.validation = {
        status: 'MISMATCH', passed: false, artifact_count: 14,
        failures: [{ reason: 'artifact_hash', role: 'pytorch_parity' }],
    };
    const mismatchHtml = renderers.buildTrainingExperimentIdentitySection(mismatched);
    assert.match(mismatchHtml, /MISMATCH/);
    assert.match(mismatchHtml, /artifact_hash/);
    assert.doesNotMatch(mismatchHtml, /badge badge-green">MISMATCH/);

    const missingHtml = renderers.buildTrainingExperimentIdentitySection({});
    assert.match(missingHtml, /MISSING/);
    assert.match(missingHtml, /cannot establish a current PASS/);
    assert.match(renderers.buildTrainingBatchPreviewSection({}), /Serialized Training Batches/);
});
