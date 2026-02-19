/**
 * Memory Explorer — v7 BUMP allocator grid visualization
 *
 * Renders the BUMP allocator's arena as a grid of colored squares
 * (1 square = 1 cache line = 64 bytes by default).
 *
 * Handles both v7 data formats:
 *
 * 1) Training (layout-train-v7):
 *    regions[]:  { name, offset, bytes, count }
 *    tensors[]:  { id, dtype, numel, bytes, kind, persistent, requires_grad,
 *                  region, shape, producer, bump_offset, bump_size, offset, end }
 *
 * 2) Inference (memory-layout, from ck-cli):
 *    memory.weights.entries[]:      { name, dtype, size, offset, abs_offset }
 *    memory.activations.buffers[]:  { name, size, offset, usage }
 *
 * Features:
 *  - CSS Grid (no D3), one div = one cache line (or page/block at zoom)
 *  - Color by: tensor, region, kind/role, dtype, alignment audit
 *  - Hover → tooltip: offset, cache line #, tensor id, dtype, shape, region
 *  - Click → highlight tensor, detail panel with head-parallelism split
 *  - Alignment audit: green ✓ / red ✗ per tensor and per region
 *  - Zoom: cache line (64B) | page (4KB) | overview (64KB)
 *  - Hex offset ruler, virtual scrolling for large arenas
 *  - Region boundary markers
 *
 * Expects globals: layoutData, formatBytes (from ir_visualizer.html)
 */

/* ─── Constants ─────────────────────────────────────────── */
const ME_CACHE_LINE = 64;
const ME_PAGE_SIZE  = 4096;
const ME_ZOOM_LEVELS = [
    { label: 'Cache Line (64B)', bytesPerCell: 64,    cellPx: 10 },
    { label: 'Page (4KB)',        bytesPerCell: 4096,  cellPx: 8  },
    { label: 'Overview (64KB)',   bytesPerCell: 65536, cellPx: 6  },
];

/* Region colour palette — matches v7 region names */
const ME_REGION_COLORS = {
    params:            { bg: '#2a5599', border: '#4a90d9' },
    grads:             { bg: '#993333', border: '#e74c3c' },
    optimizer_m:       { bg: '#336666', border: '#1abc9c' },
    optimizer_v:       { bg: '#5a2a7a', border: '#9b59b6' },
    activations:       { bg: '#1a7a3a', border: '#2ecc71' },
    grad_activations:  { bg: '#885522', border: '#e67e22' },
    saved:             { bg: '#7a5a1a', border: '#f39c12' },
    scratch:           { bg: '#555555', border: '#888888' },
    /* v7 inference layout uses role names rather than region names */
    weight:            { bg: '#2a5599', border: '#4a90d9' },
    activation:        { bg: '#1a7a3a', border: '#2ecc71' },
    kv_cache:          { bg: '#5a2a7a', border: '#9b59b6' },
    gap:               { bg: '#111',    border: '#222'    },
};

/* Per-tensor hue rotation */
const ME_TENSOR_HUES = [
    '#4a90d9', '#47b475', '#e6a23c', '#e74c3c', '#9b59b6',
    '#1abc9c', '#e67e22', '#2ecc71', '#3498db', '#f39c12',
    '#8e44ad', '#16a085', '#c0392b', '#2980b9', '#27ae60',
    '#d35400', '#7f8c8d', '#f1c40f', '#e91e63', '#00bcd4',
];

/* ─── State ─────────────────────────────────────────────── */
const meState = {
    zoomLevel:      0,
    colorMode:      'region',    // 'tensor' | 'region' | 'dtype' | 'alignment'
    viewRegion:     'all',       // 'all' | region name
    selectedTensor: null,
    gridCols:       128,
    tensors:        [],          // processed [{id, offset, end, bytes, kind, region, dtype, shape, persistent, requires_grad, colorIdx}]
    regions:        [],          // [{name, offset, bytes, count}]
    totalBytes:     0,
    config:         {},
    lookup:         null,        // (byteOffset) => tensor or null
};

/* ─── Data Extraction ───────────────────────────────────── */

function meExtractData(ld) {
    if (!ld) return;

    const tensors = [];
    const regions = [];
    let totalBytes = 0;

    /* v7 layout-train format (primary) */
    if (ld.format && ld.format.startsWith('layout-train')) {
        if (Array.isArray(ld.regions)) {
            for (const r of ld.regions) {
                regions.push({
                    name:   r.name,
                    offset: Number(r.offset) || 0,
                    bytes:  Number(r.bytes)  || 0,
                    count:  Number(r.count)  || 0,
                });
            }
        }
        if (Array.isArray(ld.tensors)) {
            ld.tensors.forEach(function(t, idx) {
                if (!t) return;
                tensors.push({
                    id:            t.id || t.name || 'tensor_' + idx,
                    offset:        Number(t.offset) || 0,
                    end:           Number(t.end)    || (Number(t.offset || 0) + Number(t.bytes || 0)),
                    bytes:         Number(t.bytes)  || 0,
                    bump_offset:   Number(t.bump_offset) || 0,
                    bump_size:     Number(t.bump_size)   || 0,
                    kind:          t.kind || '',
                    region:        t.region || '',
                    dtype:         t.dtype || 'fp32',
                    shape:         t.shape || [],
                    persistent:    !!t.persistent,
                    requires_grad: !!t.requires_grad,
                    producer:      t.producer || null,
                    numel:         Number(t.numel) || 0,
                    colorIdx:      idx,
                });
            });
        }
        /* total bytes from regions */
        if (regions.length) {
            totalBytes = regions.reduce(function(mx, r) { return Math.max(mx, r.offset + r.bytes); }, 0);
        }
    }

    /* v7 inference format: memory.weights.entries + memory.activations.buffers */
    if (tensors.length === 0) {
        var memory = ld.memory || ld;
        var wEntries = ((memory.weights || {}).entries || []);
        var aBuffers = ((memory.activations || {}).buffers || []);

        wEntries.forEach(function(e, idx) {
            tensors.push(meParseFlatEntry(e, 'params', idx));
        });
        aBuffers.forEach(function(e, idx) {
            tensors.push(meParseFlatEntry(e, 'activations', wEntries.length + idx));
        });

        /* v7 tensors[] flat array (e.g. from ir1 or lowered IR) */
        if (tensors.length === 0 && Array.isArray(ld.tensors)) {
            ld.tensors.forEach(function(t, idx) {
                if (!t) return;
                var b = Number(t.bytes) || (Number(t.numel || 0) * 4);
                var off = Number(t.offset || t.bump_offset || 0);
                tensors.push({
                    id:       t.name || t.id || 'tensor_' + idx,
                    offset:   off,
                    end:      off + b,
                    bytes:    b,
                    bump_offset: Number(t.bump_offset) || 0,
                    bump_size:   Number(t.bump_size) || 0,
                    kind:     t.kind || '',
                    region:   t.region || meInferRegion(t.kind || ''),
                    dtype:    t.dtype || 'fp32',
                    shape:    t.shape || [],
                    persistent: !!t.persistent,
                    requires_grad: !!t.requires_grad,
                    producer: null,
                    numel:    Number(t.numel) || 0,
                    colorIdx: idx,
                });
            });
        }

        if (tensors.length) {
            var last = tensors.reduce(function(a, b) { return a.end > b.end ? a : b; });
            totalBytes = last.end;
        }
        if (memory.total_bytes) totalBytes = memory.total_bytes;
    }

    /* Sort by offset */
    tensors.sort(function(a, b) { return a.offset - b.offset; });
    tensors.forEach(function(t, i) { t.colorIdx = i; });

    meState.tensors   = tensors;
    meState.regions   = regions;
    meState.totalBytes = totalBytes;
    meState.config    = ld.config || {};
    meState.lookup    = meBuildLookup(tensors);
}

function meParseFlatEntry(e, region, idx) {
    var off = meParseHex(e.offset || e.bump_offset || 0);
    var sz  = Number(e.size || e.bytes || 0);
    return {
        id:       e.name || e.key || 'tensor_' + idx,
        offset:   off,
        end:      off + sz,
        bytes:    sz,
        bump_offset: Number(e.bump_offset) || off,
        bump_size:   sz,
        kind:     e.kind || e.role || '',
        region:   region,
        dtype:    e.dtype || 'fp32',
        shape:    e.shape || [],
        persistent: true,
        requires_grad: false,
        producer: null,
        numel:    0,
        colorIdx: idx,
    };
}

function meInferRegion(kind) {
    var k = String(kind).toLowerCase();
    if (k.includes('weight') || k === 'weight') return 'params';
    if (k.includes('grad_weight') || k.includes('grad_bias')) return 'grads';
    if (k.includes('adam_m')) return 'optimizer_m';
    if (k.includes('adam_v')) return 'optimizer_v';
    if (k.includes('grad_act')) return 'grad_activations';
    if (k.includes('activation') || k.includes('act_')) return 'activations';
    if (k.includes('scratch') || k.includes('temp')) return 'scratch';
    return 'params';
}

function meParseHex(val) {
    if (typeof val === 'number') return val;
    if (typeof val === 'string') {
        if (val.startsWith('0x') || val.startsWith('0X')) return parseInt(val, 16);
        return parseInt(val, 10) || 0;
    }
    return 0;
}

/* Binary-search lookup: byte offset → tensor */
function meBuildLookup(tensors) {
    var sorted = tensors.filter(function(t) { return t.bytes > 0; }).sort(function(a, b) { return a.offset - b.offset; });
    return function(byteOffset) {
        var lo = 0, hi = sorted.length - 1;
        while (lo <= hi) {
            var mid = (lo + hi) >>> 1;
            var t = sorted[mid];
            if (byteOffset < t.offset)     hi = mid - 1;
            else if (byteOffset >= t.end)  lo = mid + 1;
            else                            return t;
        }
        return null;
    };
}

/* ─── Alignment Analysis ────────────────────────────────── */

function meAnalyzeAlignment(tensors) {
    var aligned = 0, misaligned = 0;
    var issues = [];

    for (var i = 0; i < tensors.length; i++) {
        var t = tensors[i];
        if (!t.bytes || t.bytes === 0) continue;
        var mod = t.offset % ME_CACHE_LINE;
        if (mod === 0) { aligned++; }
        else {
            misaligned++;
            issues.push({ id: t.id, offset: t.offset, mod: mod, wasted: ME_CACHE_LINE - mod, region: t.region });
        }
    }
    return { aligned: aligned, misaligned: misaligned, issues: issues, total: aligned + misaligned };
}

/* ─── Head Parallelism Analysis ─────────────────────────── */

function meHeadParallelism(tensor, config) {
    if (!config || !tensor) return null;
    var n = tensor.id.toLowerCase();
    var numHeads   = config.num_heads || 8;
    var numKVHeads = config.num_kv_heads || numHeads;
    var headDim    = config.head_dim || (Math.floor((config.embed_dim || 128) / numHeads));

    var heads = 0;
    if (n.includes('wq') || n.includes('w_q') || n.includes('.q.') || n.includes('q_proj')) {
        heads = numHeads;
    } else if (n.includes('wk') || n.includes('w_k') || n.includes('.k.') || n.includes('k_proj') ||
               n.includes('wv') || n.includes('w_v') || n.includes('.v.') || n.includes('v_proj')) {
        heads = numKVHeads;
    } else if (n.includes('wo') || n.includes('w_o') || n.includes('o_proj')) {
        heads = numHeads;
    }

    if (heads === 0) return null;
    var headBytes = Math.floor(tensor.bytes / heads);
    var splits = [];
    for (var h = 0; h < Math.min(heads, 128); h++) {
        var off = tensor.offset + h * headBytes;
        splits.push({
            head: h,
            offset: off,
            bytes: headBytes,
            cacheAligned: off % ME_CACHE_LINE === 0,
        });
    }
    return { heads: heads, headDim: headDim, headBytes: headBytes, splits: splits };
}

/* ─── Cell Color ────────────────────────────────────────── */

function meCellColor(tensor) {
    if (!tensor) return ME_REGION_COLORS.gap.bg;

    switch (meState.colorMode) {
    case 'tensor':
        return ME_TENSOR_HUES[tensor.colorIdx % ME_TENSOR_HUES.length];

    case 'region':
        var rc = ME_REGION_COLORS[tensor.region] || ME_REGION_COLORS.params;
        return rc.bg;

    case 'dtype':
        var dt = (tensor.dtype || '').toLowerCase();
        if (dt.includes('q4'))   return '#c0392b';
        if (dt.includes('q5'))   return '#e74c3c';
        if (dt.includes('q6'))   return '#e67e22';
        if (dt.includes('q8'))   return '#f39c12';
        if (dt.includes('bf16')) return '#9b59b6';
        if (dt.includes('fp16')) return '#8e44ad';
        if (dt.includes('fp32')) return '#3498db';
        if (dt.includes('i32'))  return '#1abc9c';
        if (dt.includes('u8'))   return '#7f8c8d';
        return '#555';

    case 'alignment':
        return (tensor.offset % ME_CACHE_LINE === 0) ? '#47b475' : '#e74c3c';

    default:
        return ME_TENSOR_HUES[tensor.colorIdx % ME_TENSOR_HUES.length];
    }
}

/* ─── Formatting Helpers ────────────────────────────────── */

function meFmtBytes(n) {
    if (typeof formatBytes === 'function') return formatBytes(n);
    if (n >= 1024 * 1024 * 1024) return (n / (1024 * 1024 * 1024)).toFixed(2) + ' GB';
    if (n >= 1024 * 1024)        return (n / (1024 * 1024)).toFixed(1) + ' MB';
    if (n >= 1024)               return (n / 1024).toFixed(1) + ' KB';
    return n + ' B';
}

function meHex(n) {
    return '0x' + n.toString(16).toUpperCase().padStart(8, '0');
}

function meEsc(s) {
    var d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
}

/* ─── Main Render ───────────────────────────────────────── */

function renderMemoryExplorer(containerId) {
    var container = document.getElementById(containerId || 'memoryExplorerContainer');
    if (!container) return;

    /* Extract data from global layoutData */
    meExtractData(typeof layoutData !== 'undefined' ? layoutData : null);

    if (meState.tensors.length === 0) {
        container.innerHTML =
            '<div style="text-align:center; padding:3rem; color:var(--text-muted,#888);">' +
                '<p style="font-size:1.3em; margin-bottom:0.5rem;">No memory layout data loaded</p>' +
                '<p>Load a <code>layout_train.json</code> (v7 format) to explore the BUMP allocator arena.</p>' +
            '</div>';
        return;
    }

    var alignment = meAnalyzeAlignment(meState.tensors);
    container.innerHTML = '';

    /* ── Controls Bar ── */
    var ctrl = document.createElement('div');
    ctrl.className = 'me-controls';
    var regionOpts = meState.regions.length
        ? meState.regions.map(function(r) { return '<option value="' + r.name + '"' + (meState.viewRegion === r.name ? ' selected' : '') + '>' + r.name + ' (' + meFmtBytes(r.bytes) + ')</option>'; }).join('')
        : '';
    ctrl.innerHTML =
        '<div class="me-ctrl-group">' +
            '<label>Zoom</label>' +
            '<select onchange="meSetZoom(+this.value)">' +
                ME_ZOOM_LEVELS.map(function(z, i) { return '<option value="' + i + '"' + (i === meState.zoomLevel ? ' selected' : '') + '>' + z.label + '</option>'; }).join('') +
            '</select>' +
        '</div>' +
        '<div class="me-ctrl-group">' +
            '<label>Color</label>' +
            '<select onchange="meSetColor(this.value)">' +
                '<option value="region"' + (meState.colorMode === 'region' ? ' selected' : '') + '>By Region</option>' +
                '<option value="tensor"' + (meState.colorMode === 'tensor' ? ' selected' : '') + '>Per Tensor</option>' +
                '<option value="dtype"'  + (meState.colorMode === 'dtype'  ? ' selected' : '') + '>By Dtype</option>' +
                '<option value="alignment"' + (meState.colorMode === 'alignment' ? ' selected' : '') + '>Alignment Audit</option>' +
            '</select>' +
        '</div>' +
        '<div class="me-ctrl-group">' +
            '<label>Region</label>' +
            '<select onchange="meSetRegion(this.value)">' +
                '<option value="all"' + (meState.viewRegion === 'all' ? ' selected' : '') + '>Full Arena (' + meFmtBytes(meState.totalBytes) + ')</option>' +
                regionOpts +
            '</select>' +
        '</div>' +
        '<div class="me-ctrl-group">' +
            '<button class="me-btn" onclick="meClearSelection()">Clear</button>' +
        '</div>';
    container.appendChild(ctrl);

    /* ── Summary Cards ── */
    var sumDiv = document.createElement('div');
    sumDiv.className = 'me-summary';

    var sumHTML =
        '<div class="me-card"><div class="me-card-val">' + meFmtBytes(meState.totalBytes) + '</div><div class="me-card-lbl">Arena</div></div>' +
        '<div class="me-card"><div class="me-card-val">' + meState.tensors.length + '</div><div class="me-card-lbl">Tensors</div></div>' +
        '<div class="me-card"><div class="me-card-val">' + (meState.regions.length || Object.keys(ME_REGION_COLORS).length) + '</div><div class="me-card-lbl">Regions</div></div>';

    for (var ri = 0; ri < meState.regions.length; ri++) {
        var r = meState.regions[ri];
        var rci = ME_REGION_COLORS[r.name] || ME_REGION_COLORS.params;
        sumHTML += '<div class="me-card" style="border-left:3px solid ' + rci.border + '"><div class="me-card-val">' + meFmtBytes(r.bytes) + '</div><div class="me-card-lbl">' + meEsc(r.name) + '</div></div>';
    }
    var alignCls = alignment.misaligned > 0 ? 'me-card-warn' : 'me-card-ok';
    sumHTML += '<div class="me-card ' + alignCls + '"><div class="me-card-val">' + alignment.aligned + '/' + alignment.total + '</div><div class="me-card-lbl">Cache-Aligned' + (alignment.misaligned > 0 ? ' ⚠' : ' ✓') + '</div></div>';
    sumHTML += '<div class="me-card"><div class="me-card-val">' + Math.ceil(meState.totalBytes / ME_CACHE_LINE).toLocaleString() + '</div><div class="me-card-lbl">Cache Lines</div></div>';

    sumDiv.innerHTML = sumHTML;
    container.appendChild(sumDiv);

    /* ── Alignment Issues ── */
    if (alignment.misaligned > 0) {
        var issDiv = document.createElement('div');
        issDiv.className = 'me-align-issues';
        var issRows = alignment.issues.slice(0, 50).map(function(iss) {
            return '<tr>' +
                '<td>' + meEsc(iss.id) + '</td>' +
                '<td>' + iss.region + '</td>' +
                '<td style="font-family:var(--font-mono,monospace);">' + meHex(iss.offset) + '</td>' +
                '<td>' + iss.mod + 'B</td>' +
                '<td>' + iss.wasted + 'B</td>' +
            '</tr>';
        }).join('');
        if (alignment.issues.length > 50) {
            issRows += '<tr><td colspan="5" style="color:var(--text-muted,#888);">...and ' + (alignment.issues.length - 50) + ' more</td></tr>';
        }
        issDiv.innerHTML =
            '<details>' +
                '<summary style="cursor:pointer; color:#e74c3c; font-weight:600;">' +
                    '⚠ ' + alignment.misaligned + ' misaligned tensor' + (alignment.misaligned > 1 ? 's' : '') + ' — click to expand' +
                '</summary>' +
                '<table class="me-issue-tbl">' +
                    '<thead><tr><th>Tensor</th><th>Region</th><th>Offset</th><th>mod 64</th><th>Wasted/line</th></tr></thead>' +
                    '<tbody>' + issRows + '</tbody>' +
                '</table>' +
            '</details>';
        container.appendChild(issDiv);
    }

    /* ── Region Bar (objdump-like section overview) ── */
    if (meState.regions.length > 0) {
        var regionBar = document.createElement('div');
        regionBar.className = 'me-region-bar';
        regionBar.innerHTML = '<div class="me-region-bar-label">Arena Regions</div>';
        var barInner = document.createElement('div');
        barInner.className = 'me-region-bar-inner';
        for (var rbi = 0; rbi < meState.regions.length; rbi++) {
            var reg = meState.regions[rbi];
            var pct = (reg.bytes / meState.totalBytes * 100);
            var regC = ME_REGION_COLORS[reg.name] || ME_REGION_COLORS.params;
            var seg = document.createElement('div');
            seg.className = 'me-region-seg';
            seg.style.width = Math.max(pct, 0.5) + '%';
            seg.style.backgroundColor = regC.bg;
            seg.style.borderLeft = '1px solid ' + regC.border;
            seg.title = reg.name + ': ' + meHex(reg.offset) + ' — ' + meHex(reg.offset + reg.bytes) + ' (' + meFmtBytes(reg.bytes) + ', ' + reg.count + ' tensors)';
            (function(regName) {
                seg.onclick = function() { meSetRegion(regName); };
            })(reg.name);
            var lbl = document.createElement('span');
            lbl.className = 'me-region-seg-lbl';
            lbl.textContent = pct > 5 ? reg.name : '';
            seg.appendChild(lbl);
            barInner.appendChild(seg);
        }
        regionBar.appendChild(barInner);
        container.appendChild(regionBar);
    }

    /* ── Grid Wrapper ── */
    var wrapper = document.createElement('div');
    wrapper.className = 'me-grid-wrapper';
    wrapper.id = 'meGridWrapper';

    var ruler = document.createElement('div');
    ruler.className = 'me-ruler';
    ruler.id = 'meRuler';

    var grid = document.createElement('div');
    grid.className = 'me-grid';
    grid.id = 'meGrid';

    wrapper.appendChild(ruler);
    wrapper.appendChild(grid);
    container.appendChild(wrapper);

    /* ── Detail Panel ── */
    var detail = document.createElement('div');
    detail.className = 'me-detail';
    detail.id = 'meDetail';
    detail.innerHTML = '<div class="me-detail-empty">Click a cell to inspect a tensor</div>';
    container.appendChild(detail);

    /* ── Legend ── */
    var legend = document.createElement('div');
    legend.className = 'me-legend';
    legend.id = 'meLegend';
    container.appendChild(legend);
    meUpdateLegend();

    /* Render grid */
    meRenderGrid();
}

/* ─── Grid Rendering ────────────────────────────────────── */

function meRenderGrid() {
    var grid  = document.getElementById('meGrid');
    var ruler = document.getElementById('meRuler');
    if (!grid) return;

    var zoom = ME_ZOOM_LEVELS[meState.zoomLevel];
    var cellPx = zoom.cellPx;
    var bpc    = zoom.bytesPerCell;
    var cols   = meState.gridCols;

    /* Filter tensors by region view */
    var tensors = meState.tensors;
    var regionFilter = null;
    if (meState.viewRegion !== 'all') {
        regionFilter = meState.regions.find(function(r) { return r.name === meState.viewRegion; });
        tensors = tensors.filter(function(t) { return t.region === meState.viewRegion; });
    }

    if (tensors.length === 0) {
        grid.innerHTML = '<div style="padding:2rem; color:var(--text-muted,#888);">No tensors in this region</div>';
        if (ruler) ruler.innerHTML = '';
        return;
    }

    /* Byte range */
    var minOff = regionFilter ? regionFilter.offset : 0;
    var maxOff = regionFilter ? (regionFilter.offset + regionFilter.bytes) : meState.totalBytes;
    var totalCells = Math.ceil((maxOff - minOff) / bpc);
    var rows = Math.ceil(totalCells / cols);

    /* Lookup */
    var lookup = meBuildLookup(tensors);

    /* Region boundary set (for markers) */
    var regionBoundsArr = [];
    for (var rbi2 = 0; rbi2 < meState.regions.length; rbi2++) {
        var rb = meState.regions[rbi2];
        regionBoundsArr.push(rb.offset);
        regionBoundsArr.push(rb.offset + rb.bytes);
    }

    /* Virtual scroll sizing */
    var rowH = cellPx + 1;
    var viewH = 560;
    var maxVisRows = Math.ceil(viewH / rowH) + 4;

    grid.innerHTML = '';
    grid.style.setProperty('--me-cell', cellPx + 'px');
    grid.style.setProperty('--me-cols', cols);

    var scroller = document.createElement('div');
    scroller.className = 'me-scroller';
    scroller.style.height = Math.min(rows * rowH, viewH) + 'px';
    scroller.style.overflowY = rows * rowH > viewH ? 'auto' : 'hidden';

    var inner = document.createElement('div');
    inner.className = 'me-inner';
    inner.style.height = rows * rowH + 'px';
    inner.style.position = 'relative';

    scroller.appendChild(inner);
    grid.appendChild(scroller);

    if (ruler) {
        ruler.innerHTML = '';
        ruler.style.height = Math.min(rows * rowH, viewH) + 'px';
        ruler.style.overflowY = 'hidden';
    }

    var renderedRows = {};

    function paint() {
        var scrollTop = scroller.scrollTop;
        var startRow = Math.max(0, Math.floor(scrollTop / rowH) - 1);
        var endRow = Math.min(startRow + maxVisRows, rows);

        /* Garbage-collect off-screen rows */
        for (var rk in renderedRows) {
            var rn = parseInt(rk, 10);
            if (rn < startRow || rn >= endRow) {
                var el = inner.querySelector('[data-r="' + rn + '"]');
                if (el) el.remove();
                delete renderedRows[rk];
            }
        }

        /* Ruler labels */
        if (ruler) {
            ruler.innerHTML = '';
            for (var rl = startRow; rl < endRow; rl++) {
                var off = minOff + rl * cols * bpc;
                var lblEl = document.createElement('div');
                lblEl.className = 'me-ruler-lbl';
                lblEl.style.top  = (rl * rowH - scrollTop) + 'px';
                lblEl.style.height = rowH + 'px';
                lblEl.textContent = meHex(off);
                ruler.appendChild(lblEl);
            }
        }

        /* Cell rows */
        for (var rowIdx = startRow; rowIdx < endRow; rowIdx++) {
            if (renderedRows[rowIdx]) continue;
            renderedRows[rowIdx] = true;

            var rowEl = document.createElement('div');
            rowEl.className = 'me-row';
            rowEl.setAttribute('data-r', rowIdx);
            rowEl.style.position = 'absolute';
            rowEl.style.top = rowIdx * rowH + 'px';
            rowEl.style.display = 'grid';
            rowEl.style.gridTemplateColumns = 'repeat(' + cols + ', var(--me-cell))';
            rowEl.style.gap = '1px';

            for (var c = 0; c < cols; c++) {
                var cellIdx = rowIdx * cols + c;
                if (cellIdx >= totalCells) break;

                var byteOff = minOff + cellIdx * bpc;
                var tensor  = lookup(byteOff);

                var cell = document.createElement('div');
                cell.className = 'me-cell';
                cell.style.backgroundColor = meCellColor(tensor);
                cell.style.width  = cellPx + 'px';
                cell.style.height = cellPx + 'px';

                if (meState.selectedTensor && tensor && tensor.id === meState.selectedTensor) {
                    cell.classList.add('me-sel');
                }

                /* tensor start marker */
                if (tensor && byteOff <= tensor.offset && byteOff + bpc > tensor.offset) {
                    cell.classList.add('me-start');
                    cell.classList.add(tensor.offset % ME_CACHE_LINE === 0 ? 'me-ok' : 'me-bad');
                }

                /* region boundary */
                for (var rbi3 = 0; rbi3 < regionBoundsArr.length; rbi3++) {
                    if (byteOff <= regionBoundsArr[rbi3] && byteOff + bpc > regionBoundsArr[rbi3] && byteOff !== minOff) {
                        cell.classList.add('me-region-bound');
                        break;
                    }
                }

                cell.setAttribute('data-off', byteOff);
                cell.setAttribute('data-tid', tensor ? tensor.id : '');
                cell.setAttribute('data-cl', Math.floor(byteOff / ME_CACHE_LINE));

                cell.addEventListener('mouseenter', meCellHover);
                cell.addEventListener('click', meCellClick);

                rowEl.appendChild(cell);
            }
            inner.appendChild(rowEl);
        }
    }

    scroller.addEventListener('scroll', paint);
    paint();
}

/* ─── Interaction ───────────────────────────────────────── */

function meCellHover(e) {
    var c = e.target;
    var off = parseInt(c.getAttribute('data-off'), 10);
    var tid = c.getAttribute('data-tid');
    var cl  = parseInt(c.getAttribute('data-cl'), 10);

    if (!tid) {
        c.title = 'Gap — ' + meHex(off) + ' — cache line #' + cl.toLocaleString();
        return;
    }

    var t = null;
    for (var i = 0; i < meState.tensors.length; i++) {
        if (meState.tensors[i].id === tid) { t = meState.tensors[i]; break; }
    }
    if (!t) return;

    var bpc = ME_ZOOM_LEVELS[meState.zoomLevel].bytesPerCell;
    var blk = Math.floor((off - t.offset) / bpc);
    var blkTotal = Math.ceil(t.bytes / bpc);

    c.title = [
        t.id,
        'Region: ' + t.region,
        'Offset: ' + meHex(off) + '   Tensor start: ' + meHex(t.offset),
        'Cache line: #' + cl.toLocaleString(),
        'Size: ' + meFmtBytes(t.bytes) + ' (' + t.bytes.toLocaleString() + ' B)',
        'Dtype: ' + t.dtype + '   Shape: [' + t.shape.join(', ') + ']',
        'Block: ' + (blk + 1) + '/' + blkTotal,
        'Kind: ' + t.kind + '   ' + (t.persistent ? 'persistent' : 'ephemeral') + '   ' + (t.requires_grad ? 'grad ✓' : 'no grad'),
        t.offset % ME_CACHE_LINE === 0 ? '✓ Cache-aligned' : '✗ Misaligned (mod 64 = ' + (t.offset % ME_CACHE_LINE) + ')',
    ].join('\n');
}

function meCellClick(e) {
    var tid = e.target.getAttribute('data-tid');
    if (!tid) return;

    meState.selectedTensor = tid;

    /* Highlight */
    var allSel = document.querySelectorAll('.me-sel');
    for (var i = 0; i < allSel.length; i++) allSel[i].classList.remove('me-sel');

    var matching = document.querySelectorAll('.me-cell[data-tid="' + tid.replace(/"/g, '\\"') + '"]');
    for (var j = 0; j < matching.length; j++) matching[j].classList.add('me-sel');

    /* Detail panel */
    var t = null;
    for (var k = 0; k < meState.tensors.length; k++) {
        if (meState.tensors[k].id === tid) { t = meState.tensors[k]; break; }
    }
    if (t) meShowDetail(t);
}

function meShowDetail(t) {
    var panel = document.getElementById('meDetail');
    if (!panel) return;

    var rc = ME_REGION_COLORS[t.region] || ME_REGION_COLORS.params;
    var mod = t.offset % ME_CACHE_LINE;
    var pageMod = t.offset % ME_PAGE_SIZE;
    var cacheLines = Math.ceil(t.bytes / ME_CACHE_LINE);
    var pages = Math.ceil(t.bytes / ME_PAGE_SIZE);

    /* Head parallelism */
    var headHTML = '';
    var hp = meHeadParallelism(t, meState.config);
    if (hp) {
        var hrows = '';
        for (var h = 0; h < hp.splits.length; h++) {
            var s = hp.splits[h];
            hrows += '<tr>' +
                '<td>Head ' + s.head + '</td>' +
                '<td style="font-family:var(--font-mono,monospace);">' + meHex(s.offset) + '</td>' +
                '<td>' + meFmtBytes(s.bytes) + '</td>' +
                '<td>' + Math.ceil(s.bytes / ME_CACHE_LINE) + '</td>' +
                '<td>' + (s.cacheAligned ? '<span style="color:#47b475;">✓</span>' : '<span style="color:#e74c3c;">✗ +' + (s.offset % ME_CACHE_LINE) + 'B</span>') + '</td>' +
            '</tr>';
        }
        headHTML =
            '<div class="me-detail-section">' +
                '<h4 style="color:#e6a23c;">Head Parallelism Split</h4>' +
                '<p style="color:var(--text-muted,#888); font-size:0.85em;">' +
                    hp.heads + ' heads × ' + meFmtBytes(hp.headBytes) + '/head (head_dim=' + hp.headDim + ')' +
                '</p>' +
                '<table class="me-detail-tbl">' +
                    '<thead><tr><th>Head</th><th>Offset</th><th>Size</th><th>Cache Lines</th><th>Aligned</th></tr></thead>' +
                    '<tbody>' + hrows + '</tbody>' +
                '</table>' +
            '</div>';
    }

    panel.innerHTML =
        '<div class="me-detail-hdr">' +
            '<span class="me-detail-name">' + meEsc(t.id) + '</span>' +
            '<span class="me-detail-badge" style="color:' + rc.border + ';">⬤ ' + t.region + '</span>' +
            (t.requires_grad ? '<span class="me-detail-badge" style="color:#e74c3c;">∇ grad</span>' : '') +
            (t.persistent ? '<span class="me-detail-badge" style="color:#47b475;">persistent</span>' : '<span class="me-detail-badge" style="color:#888;">ephemeral</span>') +
        '</div>' +
        '<div class="me-detail-grid">' +
            '<div class="me-kv"><span class="me-k">Kind</span><span class="me-v">' + t.kind + '</span></div>' +
            '<div class="me-kv"><span class="me-k">Offset</span><span class="me-v" style="font-family:var(--font-mono,monospace);">' + meHex(t.offset) + '</span></div>' +
            '<div class="me-kv"><span class="me-k">End</span><span class="me-v" style="font-family:var(--font-mono,monospace);">' + meHex(t.end) + '</span></div>' +
            '<div class="me-kv"><span class="me-k">Size</span><span class="me-v">' + meFmtBytes(t.bytes) + ' (' + t.bytes.toLocaleString() + ' B)</span></div>' +
            '<div class="me-kv"><span class="me-k">BUMP offset</span><span class="me-v" style="font-family:var(--font-mono,monospace);">' + meHex(t.bump_offset) + '</span></div>' +
            '<div class="me-kv"><span class="me-k">BUMP size</span><span class="me-v">' + meFmtBytes(t.bump_size) + '</span></div>' +
            '<div class="me-kv"><span class="me-k">Dtype</span><span class="me-v">' + t.dtype + '</span></div>' +
            '<div class="me-kv"><span class="me-k">Shape</span><span class="me-v">[' + t.shape.join(', ') + ']</span></div>' +
            '<div class="me-kv"><span class="me-k">Elements</span><span class="me-v">' + t.numel.toLocaleString() + '</span></div>' +
            '<div class="me-kv"><span class="me-k">Cache Lines</span><span class="me-v">' + cacheLines.toLocaleString() + '</span></div>' +
            '<div class="me-kv"><span class="me-k">Pages (4KB)</span><span class="me-v">' + pages.toLocaleString() + '</span></div>' +
            '<div class="me-kv"><span class="me-k">Cache Align</span><span class="me-v">' +
                (mod === 0
                    ? '<span style="color:#47b475;">✓ aligned (mod 64 = 0)</span>'
                    : '<span style="color:#e74c3c;">✗ misaligned (mod 64 = ' + mod + ')</span>') +
            '</span></div>' +
            '<div class="me-kv"><span class="me-k">Page Align</span><span class="me-v">' +
                (pageMod === 0
                    ? '<span style="color:#47b475;">✓ page-aligned</span>'
                    : '<span style="color:#ffb400;">mod 4096 = ' + pageMod + '</span>') +
            '</span></div>' +
            (t.producer ? '<div class="me-kv"><span class="me-k">Producer</span><span class="me-v">' + meEsc(t.producer) + '</span></div>' : '') +
        '</div>' +
        headHTML;
}

/* ─── Global Controls ───────────────────────────────────── */

if (typeof window !== 'undefined') {
    window.meSetZoom = function(v) {
        meState.zoomLevel = v;
        meRenderGrid();
    };
    window.meSetColor = function(v) {
        meState.colorMode = v;
        meRenderGrid();
        meUpdateLegend();
    };
    window.meSetRegion = function(v) {
        meState.viewRegion = v;
        meRenderGrid();
    };
    window.meClearSelection = function() {
        meState.selectedTensor = null;
        var allSel = document.querySelectorAll('.me-sel');
        for (var i = 0; i < allSel.length; i++) allSel[i].classList.remove('me-sel');
        var p = document.getElementById('meDetail');
        if (p) p.innerHTML = '<div class="me-detail-empty">Click a cell to inspect a tensor</div>';
    };
    window.renderMemoryExplorer = renderMemoryExplorer;
}

/* ─── Legend ─────────────────────────────────────────────── */

function meUpdateLegend() {
    var el = document.getElementById('meLegend');
    if (!el) return;

    var html = '<div class="me-legend-title">Legend</div><div class="me-legend-items">';

    switch (meState.colorMode) {
    case 'region':
        for (var i = 0; i < meState.regions.length; i++) {
            var r = meState.regions[i];
            var rcc = ME_REGION_COLORS[r.name] || ME_REGION_COLORS.params;
            html += '<div class="me-legend-item"><div class="me-swatch" style="background:' + rcc.bg + '; border:1px solid ' + rcc.border + ';"></div>' + r.name + ' (' + r.count + ')</div>';
        }
        if (meState.regions.length === 0) {
            var seenRegions = {};
            for (var j = 0; j < meState.tensors.length; j++) {
                seenRegions[meState.tensors[j].region] = (seenRegions[meState.tensors[j].region] || 0) + 1;
            }
            for (var rName in seenRegions) {
                var colors = ME_REGION_COLORS[rName] || ME_REGION_COLORS.params;
                html += '<div class="me-legend-item"><div class="me-swatch" style="background:' + colors.bg + '; border:1px solid ' + colors.border + ';"></div>' + rName + ' (' + seenRegions[rName] + ')</div>';
            }
        }
        break;

    case 'dtype':
        var dtypes = [
            ['Q4_x', '#c0392b'], ['Q5_x', '#e74c3c'], ['Q6_x', '#e67e22'], ['Q8_x', '#f39c12'],
            ['BF16', '#9b59b6'], ['FP16', '#8e44ad'], ['FP32', '#3498db'], ['I32', '#1abc9c'], ['U8', '#7f8c8d'],
        ];
        for (var di = 0; di < dtypes.length; di++) {
            html += '<div class="me-legend-item"><div class="me-swatch" style="background:' + dtypes[di][1] + ';"></div>' + dtypes[di][0] + '</div>';
        }
        break;

    case 'alignment':
        html += '<div class="me-legend-item"><div class="me-swatch" style="background:#47b475;"></div>Cache-aligned (mod 64 = 0)</div>';
        html += '<div class="me-legend-item"><div class="me-swatch" style="background:#e74c3c;"></div>Misaligned</div>';
        break;

    case 'tensor':
    default:
        var shown = meState.tensors.slice(0, 20);
        for (var ti = 0; ti < shown.length; ti++) {
            var tc = ME_TENSOR_HUES[shown[ti].colorIdx % ME_TENSOR_HUES.length];
            var tlbl = shown[ti].id.length > 32 ? shown[ti].id.slice(0, 29) + '...' : shown[ti].id;
            html += '<div class="me-legend-item"><div class="me-swatch" style="background:' + tc + ';"></div>' + meEsc(tlbl) + '</div>';
        }
        if (meState.tensors.length > 20) {
            html += '<div class="me-legend-item" style="color:var(--text-muted,#888);">...and ' + (meState.tensors.length - 20) + ' more</div>';
        }
        break;
    }

    html += '</div>';
    el.innerHTML = html;
}
