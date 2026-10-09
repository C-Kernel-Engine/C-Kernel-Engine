(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.CkeModelSizing = api;
})(typeof globalThis === "object" ? globalThis : this, function () {
  "use strict";

  const GIB = 1024 ** 3;
  const limits = { layers: 1000, tokens: 2000000, bytes: 1e13, requests: 10000 };

  function number(value, label, max, integer) {
    const n = Number(value);
    if (!Number.isFinite(n) || n < 0 || n > max || (integer && !Number.isInteger(n))) {
      throw new Error(label + " must be " + (integer ? "an integer" : "a number") + " from 0 to " + max);
    }
    return n;
  }

  function estimate(raw) {
    const weightBytes = number(raw.weightBytes, "Weights", limits.bytes, false);
    const capacityBytes = number(raw.capacityBytes, "GPU capacity", limits.bytes, false);
    const bandwidthBytesPerSecond = number(raw.bandwidthBytesPerSecond, "Bandwidth", limits.bytes, false);
    const tokens = number(raw.tokens, "Resident tokens", limits.tokens, true);
    const requests = number(raw.requests, "Active requests", limits.requests, true);
    const fullOwners = number(raw.fullOwners, "Full KV owners", limits.layers, true);
    const localOwners = number(raw.localOwners, "Local KV owners", limits.layers, true);
    const recurrentLayers = number(raw.recurrentLayers, "Recurrent layers", limits.layers, true);
    const kvHeads = number(raw.kvHeads, "KV heads", 1024, true);
    const kDim = number(raw.kDim, "K dimension", 8192, true);
    const vDim = number(raw.vDim, "V dimension", 8192, true);
    const localKvHeads = number(raw.localKvHeads ?? raw.kvHeads, "Local KV heads", 1024, true);
    const localKDim = number(raw.localKDim ?? raw.kDim, "Local K dimension", 8192, true);
    const localVDim = number(raw.localVDim ?? raw.vDim, "Local V dimension", 8192, true);
    const kvBytes = number(raw.kvBytes, "KV bytes per element", 8, false);
    const window = number(raw.window, "Window", limits.tokens, true);
    const stateBytes = number(raw.stateBytes, "State bytes per recurrent layer", limits.bytes, false);
    const reserveFraction = number(raw.reserveFraction, "Reserve fraction", 0.9, false);
    const efficiency = number(raw.efficiency, "Bandwidth efficiency", 1, false);
    const replicas = number(raw.replicas, "Replicas", 6, true);
    const mode = raw.localStorage;
    if (!capacityBytes || !bandwidthBytesPerSecond || !kvHeads || !kDim || !vDim || !localKvHeads || !localKDim || !localVDim || !kvBytes || !requests || !efficiency) {
      throw new Error("Capacity, bandwidth, KV geometry, requests and efficiency must be positive");
    }
    if (localOwners && !window) throw new Error("Local layers need a positive window");
    if (mode !== "evict" && mode !== "full") throw new Error("Local storage must be evict or full");

    const bytesPerKvToken = kvHeads * (kDim + vDim) * kvBytes;
    const bytesPerLocalKvToken = localKvHeads * (localKDim + localVDim) * kvBytes;
    const localAllocatedTokens = mode === "evict" ? Math.min(tokens, window) : tokens;
    const fullKvBytes = fullOwners * tokens * bytesPerKvToken;
    const localKvBytes = localOwners * localAllocatedTokens * bytesPerLocalKvToken;
    const stateTotalBytes = recurrentLayers * stateBytes;
    const perRequestBytes = fullKvBytes + localKvBytes + stateTotalBytes;
    const runtimeReserveBytes = capacityBytes * reserveFraction;
    const usedBytes = weightBytes + requests * perRequestBytes + runtimeReserveBytes;
    const freeBytes = capacityBytes - usedBytes;
    const maxRequests = perRequestBytes > 0
      ? Math.max(0, Math.floor((capacityBytes - runtimeReserveBytes - weightBytes) / perRequestBytes))
      : (weightBytes + runtimeReserveBytes <= capacityBytes ? null : 0);

    // An optimistic decode traffic model: one weight read is shared by the active batch.
    // Local kernels only read the live window even when storage retains old rows.
    const kvReadBytes = fullOwners * tokens * bytesPerKvToken
      + localOwners * Math.min(tokens, window) * bytesPerLocalKvToken;
    const bytesPerBatchStep = weightBytes + requests * (kvReadBytes + stateTotalBytes);
    const upperAggregateTokensPerSecond = bytesPerBatchStep > 0
      ? requests * bandwidthBytesPerSecond * efficiency / bytesPerBatchStep : null;

    return {
      weightBytes, fullKvBytes, localKvBytes, stateTotalBytes, perRequestBytes,
      runtimeReserveBytes, usedBytes, freeBytes, fits: freeBytes >= 0, maxRequests,
      kvReadBytes, bytesPerBatchStep, upperAggregateTokensPerSecond,
      upperPerRequestTokensPerSecond: upperAggregateTokensPerSecond === null ? null : upperAggregateTokensPerSecond / requests,
      replicaUpperAggregateTokensPerSecond: upperAggregateTokensPerSecond === null ? null : replicas * upperAggregateTokensPerSecond,
    };
  }

  function formatBytes(value) {
    return (value / GIB).toFixed(value < GIB ? 2 : 1) + " GiB";
  }

  function parseHfUrl(value) {
    let url;
    try { url = new URL(value); } catch (_) { throw new Error("Enter a Hugging Face model URL"); }
    if (url.hostname !== "huggingface.co") throw new Error("Only huggingface.co model URLs are supported");
    const parts = url.pathname.split("/").filter(Boolean);
    if (parts.length < 2 || !/^[\w.-]+$/.test(parts[0]) || !/^[\w.-]+$/.test(parts[1])) {
      throw new Error("URL must identify an owner/model repository");
    }
    return { repo: parts[0] + "/" + parts[1] };
  }

  function parseModelConfig(config) {
    const text = config && (config.text_config || config.language_config || config);
    if (!text || typeof text !== "object") return { fields: {}, warnings: ["No text model configuration found"] };
    const fields = {};
    const warnings = [];
    const kinds = text.layer_types;
    if (Array.isArray(kinds) && kinds.length > 0) {
      const counts = { full_attention: 0, sliding_attention: 0, linear_attention: 0, mamba: 0 };
      for (const kind of kinds) {
        if (!(kind in counts)) { warnings.push("Unrecognized layer type: " + kind); break; }
        counts[kind]++;
      }
      if (!warnings.length && !Number(text.num_kv_shared_layers || 0)) {
        fields.fullOwners = counts.full_attention;
        fields.localOwners = counts.sliding_attention;
        fields.recurrentLayers = counts.linear_attention + counts.mamba;
      } else if (Number(text.num_kv_shared_layers || 0)) {
        warnings.push("Shared-KV layers require an explicit owner map; layer counts were not applied");
      }
    } else {
      warnings.push("No explicit per-layer type list; enter the layer mix manually");
    }
    const positive = (v) => Number.isSafeInteger(Number(v)) && Number(v) > 0;
    if (positive(text.num_global_key_value_heads)) fields.kvHeads = Number(text.num_global_key_value_heads);
    else if (positive(text.num_key_value_heads)) fields.kvHeads = Number(text.num_key_value_heads);
    if (positive(text.global_head_dim)) fields.kDim = fields.vDim = Number(text.global_head_dim);
    else if (positive(text.head_dim)) fields.kDim = fields.vDim = Number(text.head_dim);
    if (positive(text.num_key_value_heads)) fields.localKvHeads = Number(text.num_key_value_heads);
    if (positive(text.head_dim)) fields.localKDim = fields.localVDim = Number(text.head_dim);
    if (positive(text.sliding_window)) fields.window = Number(text.sliding_window);
    if (fields.recurrentLayers) warnings.push("Recurrent state bytes remain manual; this config does not prove the runtime state layout");
    return { fields, warnings };
  }

  return { estimate, formatBytes, parseHfUrl, parseModelConfig, GIB };
});
