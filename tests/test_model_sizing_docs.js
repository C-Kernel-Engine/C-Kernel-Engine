const assert = require('node:assert/strict');
const test = require('node:test');
const { estimate, parseHfUrl, parseModelConfig, circuitTopology, splitHybridLayers, HARDWARE_PRESETS, GIB } = require('../docs/site/assets/model-sizing.js');

const base = {
  weightBytes: 24 * GIB,
  capacityBytes: 96e9,
  bandwidthBytesPerSecond: 1597e9,
  tokens: 32768,
  requests: 5,
  fullOwners: 8,
  localOwners: 0,
  recurrentLayers: 24,
  kvHeads: 2,
  kDim: 256,
  vDim: 256,
  kvBytes: 2,
  window: 1024,
  stateBytes: 1024 ** 2,
  reserveFraction: 0.1,
  efficiency: 0.5,
  replicas: 1,
  localStorage: 'full',
};

test('full attention scales KV with context and concurrency', () => {
  const one = estimate({ ...base, requests: 1 });
  const two = estimate({ ...base, requests: 2, tokens: 2 * base.tokens });
  assert.equal(two.fullKvBytes, 2 * one.fullKvBytes);
  assert.equal(two.usedBytes - one.weightBytes - one.runtimeReserveBytes,
    2 * (two.fullKvBytes + two.stateTotalBytes));
});

test('sliding eviction reduces allocation but not modeled reads', () => {
  const input = { ...base, fullOwners: 0, localOwners: 8, recurrentLayers: 0 };
  const full = estimate({ ...input, localStorage: 'full' });
  const evict = estimate({ ...input, localStorage: 'evict' });
  assert.equal(evict.localKvBytes, 8 * input.window * input.kvHeads * (input.kDim + input.vDim) * input.kvBytes);
  assert.ok(full.localKvBytes > evict.localKvBytes);
  assert.equal(full.kvReadBytes, evict.kvReadBytes);
});

test('global and local attention may have different KV geometry', () => {
  const result = estimate({ ...base, fullOwners: 1, localOwners: 1,
    kvHeads: 4, kDim: 512, vDim: 512,
    localKvHeads: 16, localKDim: 256, localVDim: 256,
    localStorage: 'evict' });
  assert.equal(result.fullKvBytes, base.tokens * 4 * 1024 * 2);
  assert.equal(result.localKvBytes, base.window * 16 * 512 * 2);
});

test('recurrent state is fixed with context length', () => {
  const input = { ...base, fullOwners: 0, recurrentLayers: 24 };
  assert.equal(estimate({ ...input, tokens: 1024 }).perRequestBytes,
    estimate({ ...input, tokens: 262144 }).perRequestBytes);
});

test('weight and KV dtypes affect different terms', () => {
  const baseResult = estimate(base);
  const smallerWeights = estimate({ ...base, weightBytes: 12 * GIB });
  const smallerKv = estimate({ ...base, kvBytes: 1 });
  assert.equal(smallerWeights.fullKvBytes, baseResult.fullKvBytes);
  assert.equal(smallerKv.fullKvBytes, baseResult.fullKvBytes / 2);
  assert.equal(smallerKv.weightBytes, baseResult.weightBytes);
});

test('replicas do not pool memory and only multiply the optimistic fleet ceiling', () => {
  const one = estimate(base);
  const three = estimate({ ...base, replicas: 3 });
  assert.equal(one.usedBytes, three.usedBytes);
  assert.equal(three.replicaUpperAggregateTokensPerSecond, 3 * one.upperAggregateTokensPerSecond);
  assert.equal(estimate({ ...base, replicas: 0 }).replicaUpperAggregateTokensPerSecond, 0);
});

test('over-capacity and invalid inputs are explicit', () => {
  assert.equal(estimate({ ...base, weightBytes: 100e9 }).fits, false);
  assert.throws(() => estimate({ ...base, requests: -1 }), /Active requests/);
  assert.throws(() => estimate({ ...base, localOwners: 1, window: 0 }), /positive window/);
  assert.throws(() => estimate({ ...base, tokens: Infinity }), /Resident tokens/);
});

test('Hugging Face parser accepts model URLs only', () => {
  assert.deepEqual(parseHfUrl('https://huggingface.co/Qwen/Qwen3.5-35B-A3B/tree/main'), { repo: 'Qwen/Qwen3.5-35B-A3B' });
  assert.throws(() => parseHfUrl('https://example.com/Qwen/model'), /Only huggingface/);
  assert.throws(() => parseHfUrl('https://huggingface.co/Qwen'), /owner\/model/);
});

test('explicit hybrid config populates only evidenced fields', () => {
  const parsed = parseModelConfig({ text_config: {
    layer_types: ['linear_attention', 'linear_attention', 'linear_attention', 'full_attention'],
    num_key_value_heads: 2, head_dim: 256,
  } });
  assert.equal(parsed.fields.fullOwners, 1);
  assert.equal(parsed.fields.recurrentLayers, 3);
  assert.equal(parsed.fields.kvHeads, 2);
  assert.match(parsed.warnings.join(' '), /state bytes remain manual/);
});

test('shared KV is not double-counted from a layer list', () => {
  const parsed = parseModelConfig({ text_config: {
    layer_types: ['sliding_attention', 'full_attention'],
    num_kv_shared_layers: 1,
  } });
  assert.equal(parsed.fields.fullOwners, undefined);
  assert.match(parsed.warnings.join(' '), /owner map/);
});

test('hardware presets are positive and cover CPU and GPU', () => {
  const { HARDWARE_PRESETS: presets } = require('../docs/site/assets/model-sizing.js');
  const ids = new Set();
  for (const preset of presets) {
    assert.ok(preset.capacityGiB > 0 && preset.bandwidthGBs > 0, preset.id);
    assert.ok(preset.note.length > 0, preset.id);
    assert.ok(!ids.has(preset.id), 'duplicate preset id ' + preset.id);
    ids.add(preset.id);
  }
  assert.ok(presets.some((p) => p.kind === 'cpu'));
  assert.ok(presets.some((p) => p.kind === 'gpu'));
});

test('circuit topology comes only from declared circuit flags', () => {
  const { CIRCUIT_TOPOLOGY } = require('../docs/site/assets/model-sizing.js');
  assert.equal(circuitTopology('qwen38').pattern, '3x_recurrent_then_1x_full_attention');
  assert.equal(circuitTopology('nemotron_h').pattern, 'config.hybrid_override_pattern');
  assert.equal(circuitTopology('does_not_exist'), null);
  for (const [name, entry] of Object.entries(CIRCUIT_TOPOLOGY)) {
    assert.ok(entry.source.endsWith('circuits/' + name + '.json'), name);
  }
});

test('hybrid layer split follows the declared block pattern', () => {
  assert.deepEqual(splitHybridLayers(32, '3x_recurrent_then_1x_full_attention'), { fullOwners: 8, recurrentLayers: 24 });
  assert.deepEqual(splitHybridLayers(30, '3x_recurrent_then_1x_full_attention'), { fullOwners: 8, recurrentLayers: 22 });
  assert.equal(splitHybridLayers(32, 'config.layer_kinds'), null);
  assert.equal(splitHybridLayers(0, '3x_recurrent_then_1x_full_attention'), null);
});

test('model config exposes total layer count for pattern splits', () => {
  const parsed = parseModelConfig({ text_config: { num_hidden_layers: 32, num_key_value_heads: 2, head_dim: 128 } });
  assert.equal(parsed.fields.totalLayers, 32);
});
