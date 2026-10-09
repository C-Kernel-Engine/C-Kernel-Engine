(function () {
  "use strict";
  const calc = window.CkeModelSizing;
  const $ = (id) => document.getElementById(id);
  const ids = ["weight-gib", "full-owners", "local-owners", "recurrent-layers", "state-mib", "kv-heads", "k-dim", "v-dim", "local-kv-heads", "local-k-dim", "local-v-dim", "kv-bytes", "window", "local-storage", "tokens", "requests", "capacity-gib", "bandwidth-gbs", "reserve-pct", "efficiency-pct"];
  const NS = "http://www.w3.org/2000/svg";
  let source = { kind: "manual" };
  let last = null;

  function value(id) { return $(id).value; }
  function set(id, n) { $(id).value = n; }
  function status(message, error) {
    $("hf-status").textContent = message;
    $("hf-status").classList.toggle("error", Boolean(error));
  }
  function bytes(giB) { return Number(giB) * calc.GIB; }
  function prettyRate(n) { return n === null ? "N/A" : n.toLocaleString(undefined, { maximumFractionDigits: 1 }); }

  function replicaCount() {
    return Array.from($("fleet").querySelectorAll("select")).filter((s) => s.value === "replica").length;
  }

  function scenario() {
    return {
      weightBytes: bytes(value("weight-gib")), capacityBytes: bytes(value("capacity-gib")),
      bandwidthBytesPerSecond: Number(value("bandwidth-gbs")) * 1e9,
      fullOwners: value("full-owners"), localOwners: value("local-owners"),
      recurrentLayers: value("recurrent-layers"), stateBytes: Number(value("state-mib")) * 1024 ** 2,
      kvHeads: value("kv-heads"), kDim: value("k-dim"), vDim: value("v-dim"),
      localKvHeads: value("local-kv-heads"), localKDim: value("local-k-dim"), localVDim: value("local-v-dim"), kvBytes: value("kv-bytes"),
      window: value("window"), localStorage: value("local-storage"), tokens: value("tokens"),
      requests: value("requests"), reserveFraction: Number(value("reserve-pct")) / 100,
      efficiency: Number(value("efficiency-pct")) / 100, replicas: replicaCount(),
    };
  }

  function rect(svg, x, width, color) {
    const element = document.createElementNS(NS, "rect");
    element.setAttribute("x", x.toFixed(2));
    element.setAttribute("y", "21");
    element.setAttribute("width", Math.max(0, width).toFixed(2));
    element.setAttribute("height", "28");
    element.setAttribute("fill", color);
    svg.appendChild(element);
  }

  function drawMemory(result, input) {
    const svg = $("memory-svg");
    svg.replaceChildren();
    const parts = [
      [result.weightBytes, "#ffb400"],
      [input.requests * (result.fullKvBytes + result.localKvBytes), "#74c5e8"],
      [input.requests * result.stateTotalBytes, "#a894e6"],
      [result.runtimeReserveBytes, "#738c90"],
    ];
    const totalWidth = 620;
    rect(svg, 10, totalWidth, "#33454b");
    let x = 10;
    for (const [amount, color] of parts) {
      const width = Math.max(0, Math.min(630 - x, amount / input.capacityBytes * totalWidth));
      if (width > 0) rect(svg, x, width, color);
      x += width;
    }
    const outline = document.createElementNS(NS, "rect");
    for (const [key, val] of Object.entries({ x:10, y:21, width:620, height:28, rx:5, fill:"none", stroke: result.fits ? "#82d9a0" : "#ffa19a", "stroke-width":2 })) outline.setAttribute(key, val);
    svg.appendChild(outline);
    svg.setAttribute("aria-label", "GPU memory: " + calc.formatBytes(result.usedBytes) + " used of " + calc.formatBytes(input.capacityBytes) + (result.fits ? ", fits" : ", exceeds capacity"));
  }

  function render() {
    try {
      const input = scenario();
      const result = calc.estimate(input);
      last = { input, result, source };
      $("error").textContent = "";
      $("fit").textContent = result.fits ? "Fits" : "Exceeds";
      $("fit").parentElement.className = "metric " + (result.fits ? "ok" : "no");
      $("used").textContent = calc.formatBytes(result.usedBytes);
      $("free").textContent = (result.freeBytes < 0 ? "−" : "") + calc.formatBytes(Math.abs(result.freeBytes));
      $("free").parentElement.className = "metric " + (result.fits ? "ok" : "no");
      $("weight").textContent = calc.formatBytes(result.weightBytes);
      $("per-request").textContent = calc.formatBytes(result.perRequestBytes);
      $("max-requests").textContent = result.maxRequests === null ? "Not KV-limited" : result.maxRequests.toLocaleString();
      $("per-user-speed").textContent = result.fits ? prettyRate(result.upperPerRequestTokensPerSecond) : "N/A";
      $("aggregate-speed").textContent = result.fits ? prettyRate(result.upperAggregateTokensPerSecond) : "N/A";
      $("fleet-speed").textContent = result.fits ? prettyRate(result.replicaUpperAggregateTokensPerSecond) : "N/A";
      $("breakdown").textContent = "Per request: full KV " + calc.formatBytes(result.fullKvBytes) + "; local KV " + calc.formatBytes(result.localKvBytes) + "; recurrent state " + calc.formatBytes(result.stateTotalBytes) + ". Runtime reserve: " + calc.formatBytes(result.runtimeReserveBytes) + ".";
      $("fleet-summary").textContent = input.replicas + " of 6 GPUs hold independent replicas of this model. " + (result.fits ? "Estimated maximum active requests across those replicas: " + (result.maxRequests === null ? "not limited by modeled state" : (result.maxRequests * input.replicas).toLocaleString()) + "." : "The model does not fit the selected concurrency on a replica; no fleet speed is reported.");
      drawMemory(result, input);
    } catch (error) {
      last = null;
      $("error").textContent = error.message;
      for (const id of ["fit", "used", "free", "weight", "per-request", "max-requests", "per-user-speed", "aggregate-speed", "fleet-speed"]) $(id).textContent = "-";
      $("memory-svg").replaceChildren();
    }
  }

  function buildFleet() {
    for (let i = 0; i < 6; i++) {
      const card = document.createElement("div");
      card.className = "gpu";
      const title = document.createElement("strong");
      title.textContent = "GPU " + (i + 1);
      const label = document.createElement("label");
      label.textContent = "Assignment";
      const select = document.createElement("select");
      select.setAttribute("aria-label", "GPU " + (i + 1) + " assignment");
      for (const [key, text] of [["replica", "This model"], ["other", "Other service / reserved"], ["idle", "Idle / elastic"]]) {
        const option = document.createElement("option"); option.value = key; option.textContent = text; select.appendChild(option);
      }
      select.value = i === 0 ? "replica" : i < 4 ? "other" : "idle";
      select.addEventListener("change", render);
      label.appendChild(select); card.append(title, label); $("fleet").appendChild(card);
    }
  }

  function applyPattern(kind) {
    const patterns = {
      qwen: { "full-owners":8, "local-owners":0, "recurrent-layers":24, "state-mib":1, "kv-heads":2, "k-dim":256, "v-dim":256, "local-kv-heads":2, "local-k-dim":256, "local-v-dim":256 },
      gemma: { "full-owners":8, "local-owners":24, "recurrent-layers":0, "state-mib":0, "kv-heads":4, "k-dim":512, "v-dim":512, "local-kv-heads":16, "local-k-dim":256, "local-v-dim":256, "window":1024, "local-storage":"full" },
      nemotron: { "full-owners":8, "local-owners":0, "recurrent-layers":24, "state-mib":1, "kv-heads":8, "k-dim":128, "v-dim":128, "local-kv-heads":8, "local-k-dim":128, "local-v-dim":128 },
    };
    for (const [id, v] of Object.entries(patterns[kind])) set(id, v);
    source = { kind:"illustrative-pattern", pattern:kind };
    $("hf-source").textContent = "Illustrative " + kind + " layer pattern only; not a checkpoint-specific configuration. Replace with exact metadata before using this for planning.";
    render();
  }

  async function loadHf() {
    try {
      const { repo } = calc.parseHfUrl(value("hf-url"));
      status("Reading repository metadata; no weights will be downloaded.");
      const response = await fetch("https://huggingface.co/api/models/" + repo + "?blobs=true", { mode:"cors" });
      if (!response.ok) throw new Error("Hub metadata unavailable (HTTP " + response.status + "); use manual values.");
      const model = await response.json();
      const files = (model.siblings || []).filter((file) => /\.(gguf|safetensors)$/i.test(file.rfilename || "") && Number.isFinite(file.size) && file.size > 0);
      if (!files.length) throw new Error("No GGUF or safetensors file sizes found; use manual values.");
      const list = $("hf-files");
      list.replaceChildren(); list.hidden = false;
      const types = new Set(files.map((file) => file.rfilename.split(".").pop().toLowerCase()));
      const hint = document.createElement("p");
      hint.className = "fine";
      hint.textContent = types.size > 1 ? "Choose one format only. Select one GGUF, or all shards of one safetensors checkpoint." : "Choose one GGUF, or all shards of one safetensors checkpoint.";
      list.appendChild(hint);
      for (const file of files.slice(0, 200)) {
        const label = document.createElement("label");
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox"; checkbox.dataset.size = file.size; checkbox.dataset.format = file.rfilename.split(".").pop().toLowerCase();
        checkbox.addEventListener("change", () => {
          const selected = Array.from(list.querySelectorAll("input:checked"));
          const formats = new Set(selected.map((entry) => entry.dataset.format));
          if (formats.size > 1 || (formats.has("gguf") && selected.length > 1)) {
            checkbox.checked = false; status("Select one GGUF or safetensors shards from one checkpoint, not mixed formats.", true); return;
          }
          const total = selected.reduce((sum, entry) => sum + Number(entry.dataset.size), 0);
          if (total) set("weight-gib", (total / calc.GIB).toFixed(3));
          source = { ...source, kind:"huggingface", repo, revision:model.sha || "unknown", selectedFiles:selected.map((entry) => entry.parentElement.dataset.path) };
          status(selected.length + " file(s) selected; verify the complete checkpoint and loaded-engine size.");
          render();
        });
        label.dataset.path = file.rfilename;
        label.append(checkbox, document.createTextNode(file.rfilename + " · " + calc.formatBytes(file.size)));
        list.appendChild(label);
      }
      if (files.length > 200) status("Only the first 200 weight files are shown; this repository is too large to select reliably. Use manual values.", true);
      else status("Choose exact files. Architecture fields remain manual unless verified separately.");
      source = { kind:"huggingface", repo, revision:model.sha || "unknown", selectedFiles:[] };
      $("hf-source").textContent = repo + " @ " + (model.sha || "unknown revision") + ". File sizes come from Hub metadata; layer geometry is not inferred from file size.";
      if ((model.siblings || []).some((file) => file.rfilename === "config.json") && model.sha) {
        try {
          const configResponse = await fetch("https://huggingface.co/" + repo + "/resolve/" + model.sha + "/config.json", { mode:"cors" });
          const size = Number(configResponse.headers.get("content-length") || 0);
          if (!configResponse.ok || size > 1024 * 1024) throw new Error("config.json unavailable or too large");
          const config = await configResponse.json();
          const parsed = calc.parseModelConfig(config);
          const mapping = { fullOwners:"full-owners", localOwners:"local-owners", recurrentLayers:"recurrent-layers", kvHeads:"kv-heads", kDim:"k-dim", vDim:"v-dim", localKvHeads:"local-kv-heads", localKDim:"local-k-dim", localVDim:"local-v-dim", window:"window" };
          for (const [key, number] of Object.entries(parsed.fields)) set(mapping[key], number);
          $("hf-source").textContent += " Applied explicit config fields: " + Object.keys(parsed.fields).join(", ") + ". " + parsed.warnings.join(" ");
          source.configFields = parsed.fields;
          source.configWarnings = parsed.warnings;
          status("Configuration read. Verify state size, cache dtype, sharing and loaded-engine bytes before planning.");
        } catch (error) {
          $("hf-source").textContent += " config.json could not be read: " + error.message + ". Enter geometry manually.";
        }
      }
      render();
    } catch (error) { status(error.message, true); }
  }

  function share() {
    const params = new URLSearchParams();
    for (const id of ids) params.set(id, value(id));
    params.set("fleet", Array.from($("fleet").querySelectorAll("select")).map((s) => s.value[0]).join(""));
    const url = location.origin + location.pathname + "?" + params.toString();
    navigator.clipboard.writeText(url).then(() => { $("export-status").textContent = "Scenario link copied. Model URL and repository identity are not included."; }, () => { $("export-status").textContent = "Clipboard unavailable; use the page URL after changing inputs."; history.replaceState(null, "", "?" + params); });
  }

  function restore() {
    const params = new URLSearchParams(location.search);
    for (const id of ids) if (params.has(id)) set(id, params.get(id));
    const fleet = params.get("fleet");
    if (fleet && /^[roi]{6}$/.test(fleet)) Array.from($("fleet").querySelectorAll("select")).forEach((select, i) => { select.value = {r:"replica",o:"other",i:"idle"}[fleet[i]]; });
  }

  function exportReport() {
    if (!last) { $("export-status").textContent = "Fix invalid inputs before exporting."; return; }
    const blob = new Blob([JSON.stringify({ schema:"cke.docs.model_sizing.v1", warning:"Theoretical estimate, not measured throughput", ...last }, null, 2)], { type:"application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a"); a.href = url; a.download = "cke-model-sizing.json"; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 30000);
  }

  buildFleet(); restore();
  for (const id of ids) $(id).addEventListener("input", render);
  $("hf-load").addEventListener("click", loadHf);
  for (const name of ["qwen", "gemma", "nemotron"]) $("preset-" + name).addEventListener("click", () => applyPattern(name));
  $("share").addEventListener("click", share);
  $("export").addEventListener("click", exportReport);
  render();
})();
