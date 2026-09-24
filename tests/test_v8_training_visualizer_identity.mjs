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

test('SVG source panel renders escaped documents only as matched run evidence', () => {
    const renderers = loadRenderers();
    const files = JSON.parse(fs.readFileSync(fixturePath, 'utf8'));
    const identity = files.training_experiment_manifest.identity;
    files.svg_fixture_evidence = {
        experiment_identity: structuredClone(identity),
        document_policy: 'one_complete_document_per_split',
        label_policy: 'cyclic_causal_next_token_within_document',
        documents: {
            train: { sha256: 'train-sha', source_complete_document: true, consumed_complete_document: true,
                tokenized_tokens: 32, consumed_tokens: 32, xml: '<svg><rect fill="red"/></svg>' },
            validation: { sha256: 'val-sha', source_complete_document: true, consumed_complete_document: true,
                tokenized_tokens: 32, consumed_tokens: 32, xml: '<svg><circle fill="blue"/></svg>' },
        },
        validation_relationship: 'near_duplicate_geometry_with_color_changes_only',
        generated_samples: {
            before: { text: '<svg unfinished', well_formed_svg: false, new_tokens_generated: 48,
                new_tokens_requested: 48, stop_reason: 'token_budget_exhausted' },
            after: { text: '<svg xmlns="http://www.w3.org/2000/svg"/>', well_formed_svg: true,
                new_tokens_generated: 48, new_tokens_requested: 48, stop_reason: 'token_budget_exhausted' },
        },
    };
    const matched = renderers.buildTrainingSvgFixtureSection(files);
    assert.match(matched, /data-training-svg-fixture="matched"/);
    assert.match(matched, /&lt;svg&gt;&lt;rect/);
    assert.doesNotMatch(matched, /<svg><rect/);
    assert.match(matched, /data:image\/svg\+xml/);
    assert.match(matched, /not well-formed SVG/);
    assert.match(matched, /well-formed SVG/);
    assert.match(matched, /diagnostic, not a certification gate/);
    assert.match(matched, /Consumed complete: yes/);
    assert.match(matched, /48 \/ 48 new tokens/);
    assert.match(matched, /near_duplicate_geometry/);
    const stale = structuredClone(files);
    stale.svg_fixture_evidence.experiment_identity.run_id = 'older-run';
    const staleHtml = renderers.buildTrainingSvgFixtureSection(stale);
    assert.match(staleHtml, /data-training-svg-fixture="unverified"/);
    assert.doesNotMatch(staleHtml, /data:image\/svg\+xml/);
    assert.doesNotMatch(staleHtml, /&lt;svg unfinished/);
    const missing = structuredClone(files);
    missing.training_experiment_manifest.validation.status = 'MISSING';
    assert.match(renderers.buildTrainingSvgFixtureSection(missing), /data-training-svg-fixture="unverified"/);
});
