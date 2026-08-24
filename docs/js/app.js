/* Page wiring: load the exported results and build every panel. */

import {
  barChart, cssVar, fmt, groupedBars, heatmap, hideTip, histogram, lineChart,
  ramp, rampCss, scatter, showTip,
} from "./charts.js";
import { SpotMap, spriteStyle } from "./spatial.js";

const state = {
  manifest: null,
  sections: new Map(),
  section: null,
  colorBy: "true",
  gene: null,
  mode: "single",
  selectedModel: null,
  hiddenLayers: new Set(),
  pinned: -1,
};

let mapMain, mapLeft, mapRight;

const $ = (id) => document.getElementById(id);

/* ---------------------------------------------------------------- boot */

async function boot() {
  initTheme();
  const response = await fetch("data/manifest.json");
  if (!response.ok) throw new Error("data/manifest.json is missing — run src.export_dashboard");
  state.manifest = await response.json();

  buildHero();
  buildSectionsTable();
  buildLayerBars();
  buildModelsTable();
  buildSslPanels();
  buildEfficiency();
  buildAblation();
  buildRegression();
  buildUmap();
  buildFooter();

  await initExplorer();
  await buildSchematic();
  initScrollSpy();
}

/* ---------------------------------------------------------------- hero */

function buildHero() {
  const m = state.manifest;
  const d = m.dataset;
  // The best model is read off the results rather than assumed, because on this
  // task the fused model does not turn out to be the winner.
  const ranked = [...m.models].sort((a, b) => (b.macro_f1 ?? -1) - (a.macro_f1 ?? -1));
  const headline = ranked[0];
  const expressionOnly = m.models.find((x) => x.tag === "expression_only");
  const imageOnly = m.models.find((x) => x.tag === "image_only");
  const bestFusion = ranked.find((x) => x.modality === "both");

  const cards = [
    { k: "Paired spots", v: fmt.compact(d.n_spots),
      s: `${fmt.compact(d.n_labelled)} with a layer call` },
    { k: "Sections / donors", v: `${d.n_sections} / ${d.n_donors}`,
      s: "results on a held-out donor" },
  ];
  if (headline) {
    cards.push({ k: "Best macro-F1", v: fmt.n(headline.macro_f1, 3),
      s: headline.label.toLowerCase() });
  }
  if (expressionOnly && imageOnly) {
    cards.push({ k: "Transcriptome alone", v: fmt.n(expressionOnly.macro_f1, 3),
      s: "macro-F1, no image" });
    cards.push({ k: "Histology alone", v: fmt.n(imageOnly.macro_f1, 3),
      s: "macro-F1, no expression" });
  }
  if (bestFusion && expressionOnly) {
    const gain = bestFusion.macro_f1 - expressionOnly.macro_f1;
    cards.push({
      k: "Fusion gain", v: (gain >= 0 ? "+" : "") + fmt.n(gain, 3),
      s: "best fused model over transcriptome alone",
    });
  }
  const eff = (m.efficiency || []);
  const low = eff.filter((p) => p.percent === 1);
  if (low.length === 2) {
    const s = low.find((p) => p.arm === "scratch");
    const ssl = low.find((p) => p.arm === "ssl");
    cards.push({
      k: "SSL gain at 1% labels",
      v: "+" + fmt.n(ssl.macro_f1 - s.macro_f1, 3),
      s: `macro-F1, ${fmt.int(s.n_train)} labelled spots`,
    });
  }
  $("hero-stats").innerHTML = cards.map(statCard).join("");
}

function statCard(c) {
  return `<div class="stat"><div class="k">${c.k}</div><div class="v">${c.v}</div>` +
    `<div class="s">${c.s}</div></div>`;
}

/* ---------------------------------------------------------------- data section */

function buildSectionsTable() {
  const m = state.manifest;
  const rows = m.dataset.sections.map((s) => `
    <tr title="${s.n_labelled} of ${s.n_spots} spots annotated">
      <td><code>${s.section}</code></td>
      <td>${s.donor}</td>
      <td><span class="tag ${s.split === "test" ? "ssl" : ""}">${s.split}</span></td>
      <td class="num">${fmt.int(s.n_spots)}</td>
      <td class="num">${fmt.int(s.n_labelled)}</td>
      <td class="num">${fmt.int(s.median_umi)}</td>
      <td class="num">${fmt.int(s.median_genes)}</td>
    </tr>`).join("");
  $("sections-table").innerHTML = `
    <thead><tr><th>Section</th><th>Donor</th><th>Split</th><th>Spots</th>
      <th>Annotated</th><th>Median UMI</th><th>Median genes</th></tr></thead>
    <tbody>${rows}</tbody>`;
}

function buildLayerBars() {
  const m = state.manifest;
  const totals = {};
  for (const layer of m.layers) totals[layer] = 0;
  for (const s of m.dataset.sections) {
    for (const [layer, count] of Object.entries(s.layer_counts)) totals[layer] += count;
  }
  const grand = Object.values(totals).reduce((a, b) => a + b, 0);
  barChart($("layer-bars"), {
    items: m.layers.map((layer) => ({
      label: layer, value: totals[layer], color: m.layer_colors[layer],
      tip: `<div class="t">${layer}</div><div class="r">spots<b>${fmt.int(totals[layer])}</b></div>` +
        `<div class="r">share<b>${fmt.pct(totals[layer] / grand)}</b></div>`,
    })),
    format: (v) => fmt.int(v),
    rowHeight: 26,
  });
}

async function buildSchematic() {
  const m = state.manifest;
  const example = m.example_spot;
  if (!example) return;

  const payload = await loadSection(example.section);
  const pick = example.index;
  const layer = m.layers[payload.true[pick]] || "unannotated";

  $("spot-hint").textContent =
    `a ${m.dataset.patch_size} px H&E crop, a ${fmt.int(m.dataset.n_genes_modelled)}-gene vector, one layer call`;

  const genes = m.marker_genes.slice(0, 22);
  const values = genes.map((g) => (payload.genes[g] ? payload.genes[g][pick] : 0));
  const lo = Math.min(...values, -1);
  const hi = Math.max(...values, 1);

  const strip = genes.map((g, i) => {
    const t = (values[i] - lo) / (hi - lo || 1);
    return `<div title="${g}  z=${fmt.n(values[i], 2)}" style="flex:1;height:34px;
      background:${ramp.magma(t)};border-radius:2px"></div>`;
  }).join("");

  $("spot-schematic").innerHTML = `
    <div style="display:flex;gap:26px;align-items:center;flex-wrap:wrap">
      <div style="text-align:center">
        <img src="assets/example_spot.jpg" alt="H&amp;E crop centred on one Visium spot"
          style="width:150px;height:150px;border-radius:9px;border:1px solid var(--border);display:block">
        <div style="font-size:.72rem;color:var(--text-faint);margin-top:6px">H&amp;E crop · 112 px</div>
      </div>
      <div style="flex:1;min-width:260px">
        <div style="display:flex;gap:2px;align-items:flex-end">${strip}</div>
        <div style="font-size:.72rem;color:var(--text-faint);margin-top:6px">
          standardised expression, ${genes.length} of ${fmt.compact(m.dataset.n_genes_modelled)} modelled genes
        </div>
      </div>
      <div style="text-align:center;min-width:110px">
        <div style="font-size:.68rem;text-transform:uppercase;letter-spacing:.07em;
          color:var(--text-faint);font-weight:600">Annotation</div>
        <div style="font-size:1.7rem;font-weight:660;color:${m.layer_colors[layer] || "var(--text)"}">${layer}</div>
        <div style="font-size:.72rem;color:var(--text-faint)">section ${example.section}</div>
      </div>
    </div>`;
}

/* ---------------------------------------------------------------- explorer */

async function loadSection(id) {
  if (state.sections.has(id)) return state.sections.get(id);
  const response = await fetch(`data/sections/${id}.json`);
  const payload = await response.json();
  state.sections.set(id, payload);
  return payload;
}

async function initExplorer() {
  const m = state.manifest;
  const select = $("sel-section");
  select.innerHTML = m.dataset.sections.map((s) =>
    `<option value="${s.section}">${s.section} — ${s.donor} (${s.split})</option>`).join("");

  const test = m.dataset.sections.find((s) => s.split === "test");
  state.section = test ? test.section : m.dataset.sections[0].section;
  select.value = state.section;

  $("sel-gene").innerHTML = m.marker_genes
    .map((g) => `<option value="${g}">${g}</option>`).join("");
  state.gene = m.marker_genes[0];

  mapMain = new SpotMap($("map-main"));
  mapLeft = new SpotMap($("map-left"));
  mapRight = new SpotMap($("map-right"));

  select.addEventListener("change", async () => {
    state.section = select.value;
    state.pinned = -1;
    await refreshExplorer();
  });
  $("sel-color").addEventListener("change", (e) => {
    state.colorBy = e.target.value;
    $("gene-field").style.display = state.colorBy === "gene" ? "flex" : "none";
    refreshExplorer();
  });
  $("sel-gene").addEventListener("change", (e) => {
    state.gene = e.target.value;
    refreshExplorer();
  });
  $("rng-size").addEventListener("input", (e) => {
    const r = +e.target.value;
    for (const map of [mapMain, mapLeft, mapRight]) { map.radius = r; map.draw(); }
  });
  $("chk-tissue").addEventListener("change", (e) => {
    for (const map of [mapMain, mapLeft, mapRight]) {
      map.showBackdrop = e.target.checked;
      map.draw();
    }
  });
  for (const button of $("view-mode").querySelectorAll("button")) {
    button.addEventListener("click", () => {
      state.mode = button.dataset.mode;
      for (const other of $("view-mode").querySelectorAll("button")) {
        other.classList.toggle("on", other === button);
      }
      $("single-view").classList.toggle("hidden", state.mode !== "single");
      $("compare-view").classList.toggle("hidden", state.mode !== "compare");
      refreshExplorer();
    });
  }

  attachHover($("map-main"), mapMain);
  attachHover($("map-left"), mapLeft);
  attachHover($("map-right"), mapRight);

  await refreshExplorer();
}

function attachHover(canvas, map) {
  canvas.addEventListener("mousemove", (event) => {
    const index = map.hitTest(event.clientX, event.clientY);
    if (index !== map.highlight) {
      for (const other of [mapMain, mapLeft, mapRight]) {
        other.highlight = index;
        if (other.section) other.draw();
      }
      if (index >= 0) showSpot(index);
    }
  });
  canvas.addEventListener("mouseleave", () => {
    for (const other of [mapMain, mapLeft, mapRight]) {
      other.highlight = -1;
      if (other.section) other.draw();
    }
    if (state.pinned >= 0) showSpot(state.pinned);
  });
  canvas.addEventListener("click", (event) => {
    const index = map.hitTest(event.clientX, event.clientY);
    state.pinned = index === state.pinned ? -1 : index;
    for (const other of [mapMain, mapLeft, mapRight]) {
      other.pinned = state.pinned;
      if (other.section) other.draw();
    }
  });
}

function colorFactory(payload, mode) {
  const m = state.manifest;
  const layerColor = (v) => (v >= 0 ? m.layer_colors[m.layers[v]] : "#c9c9c4");

  if (mode === "true" || mode === "pred") {
    const values = mode === "true" ? payload.true : payload.pred;
    if (!values) return null;
    return {
      color: (i) => layerColor(values[i]),
      dim: (i) => state.hiddenLayers.has(values[i]),
      legend: "layers",
    };
  }
  if (mode === "error") {
    if (!payload.pred) return null;
    return {
      color: (i) => {
        if (payload.true[i] < 0) return "#c9c9c4";
        if (payload.true[i] === payload.pred[i]) return cssVar("--accent");
        return Math.abs(payload.true[i] - payload.pred[i]) === 1 ? "#e0a458" : "#b8433a";
      },
      dim: () => false,
      legend: [
        ["matches annotation", cssVar("--accent")],
        ["off by one layer", "#e0a458"],
        ["further", "#b8433a"],
        ["unannotated", "#c9c9c4"],
      ],
    };
  }
  if (mode === "confidence") {
    if (!payload.confidence) return null;
    return continuous(payload.confidence, ramp.viridis, "model confidence", 1 / 7, 1);
  }
  if (mode === "gate") {
    if (!payload.gate_expression) return null;
    return continuous(payload.gate_expression, ramp.balance,
      "← histology    gate weight    expression →", 0, 1);
  }
  if (mode === "umi") {
    const logged = payload.total_counts.map((v) => Math.log10(Math.max(v, 1)));
    return continuous(logged, ramp.magma, "log₁₀ total UMI");
  }
  if (mode === "gene") {
    const values = payload.genes[state.gene];
    if (!values) return null;
    return continuous(values, ramp.magma, `${state.gene} (standardised)`);
  }
  return null;
}

function continuous(values, rampFn, label, forceLo, forceHi) {
  const clean = values.filter((v) => v !== null && Number.isFinite(v));
  const sorted = [...clean].sort((a, b) => a - b);
  const lo = forceLo !== undefined ? forceLo : sorted[Math.floor(sorted.length * 0.02)];
  const hi = forceHi !== undefined ? forceHi : sorted[Math.floor(sorted.length * 0.98)];
  return {
    color: (i) => (values[i] === null ? "#c9c9c4"
      : rampFn((values[i] - lo) / (hi - lo || 1))),
    dim: () => false,
    legend: { ramp: rampFn, lo, hi, label },
  };
}

async function refreshExplorer() {
  const m = state.manifest;
  const payload = await loadSection(state.section);
  const tissue = m.tissue[state.section];
  const url = tissue ? `assets/tissue/${state.section}.jpg` : null;

  const scheme = colorFactory(payload, state.colorBy)
    || colorFactory(payload, "true");

  if (state.mode === "single") {
    mapMain.setSection(payload, tissue, url).setColoring(scheme.color, scheme.dim);
    mapMain.pinned = state.pinned;
    mapMain.draw();
  } else {
    const left = colorFactory(payload, "true");
    const right = colorFactory(payload, "pred") || left;
    mapLeft.setSection(payload, tissue, url).setColoring(left.color, left.dim).draw();
    mapRight.setSection(payload, tissue, url).setColoring(right.color, right.dim).draw();
  }

  drawLegend(state.mode === "compare" ? colorFactory(payload, "true") : scheme);
  updateExplorerNote(payload);
  if (state.pinned >= 0) showSpot(state.pinned);
}

function drawLegend(scheme) {
  const m = state.manifest;
  const node = $("map-legend");
  if (!scheme) { node.innerHTML = ""; return; }

  if (scheme.legend === "layers") {
    node.innerHTML = m.layers.map((layer, i) =>
      `<span class="item ${state.hiddenLayers.has(i) ? "off" : ""}" data-layer="${i}">
        <span class="sw" style="background:${m.layer_colors[layer]}"></span>${layer}</span>`).join("") +
      `<span class="item" style="cursor:default"><span class="sw" style="background:#c9c9c4"></span>unannotated</span>`;
    for (const item of node.querySelectorAll("[data-layer]")) {
      item.addEventListener("click", () => {
        const i = +item.dataset.layer;
        if (state.hiddenLayers.has(i)) state.hiddenLayers.delete(i);
        else state.hiddenLayers.add(i);
        refreshExplorer();
      });
    }
  } else if (Array.isArray(scheme.legend)) {
    node.innerHTML = scheme.legend.map(([label, colour]) =>
      `<span class="item" style="cursor:default"><span class="sw" style="background:${colour}"></span>${label}</span>`).join("");
  } else {
    const l = scheme.legend;
    node.innerHTML = `<div style="width:100%;max-width:340px">
      <div class="rampbar" style="background:${rampCss(l.ramp)}"></div>
      <div class="ramplabels"><span>${fmt.n(l.lo, 2)}</span>
        <span style="color:var(--text-muted)">${l.label}</span>
        <span>${fmt.n(l.hi, 2)}</span></div></div>`;
  }
}

function updateExplorerNote(payload) {
  const m = state.manifest;
  const model = m.models.find((x) => x.tag === m.headline_model);
  const seen = payload.split === "test"
    ? "This donor was held out, so the model has never seen this tissue."
    : `This section is in the <strong>${payload.split}</strong> split, so the model was fitted on it (or on its donor). Predictions here look better than they would on new tissue.`;
  const accuracy = model && model.per_section && model.per_section[payload.section]
    ? ` Accuracy on this section is ${fmt.pct(model.per_section[payload.section].accuracy)}, macro-F1 ${fmt.n(model.per_section[payload.section].macro_f1, 3)}.` : "";
  $("explorer-note").innerHTML = `Predictions come from <strong>${model ? model.label : "the fused model"}</strong>. ${seen}${accuracy}`;
}

function showSpot(index) {
  const m = state.manifest;
  const payload = state.sections.get(state.section);
  if (!payload || index < 0) return;

  const layout = m.sprites[state.section];
  const preview = $("patch-preview");
  if (layout) {
    const style = spriteStyle({ ...layout, section: state.section }, index,
      preview.clientWidth || 220);
    preview.textContent = "";
    Object.assign(preview.style, style);
  }

  const trueLayer = payload.true[index];
  const predLayer = payload.pred ? payload.pred[index] : -1;
  const rows = [
    ["Barcode index", index],
    ["Array position", `${payload.array_row[index]}, ${payload.array_col[index]}`],
    ["Total UMI", fmt.int(payload.total_counts[index])],
    ["Genes detected", fmt.int(payload.n_genes[index])],
    ["Annotation", trueLayer >= 0
      ? `<span style="color:${m.layer_colors[m.layers[trueLayer]]}">${m.layers[trueLayer]}</span>`
      : "none"],
  ];
  if (predLayer >= 0) {
    rows.push(["Predicted", `<span style="color:${m.layer_colors[m.layers[predLayer]]}">${m.layers[predLayer]}</span>`]);
    rows.push(["Confidence", fmt.pct(payload.confidence[index])]);
  }
  if (payload.gate_expression && payload.gate_expression[index] !== null) {
    rows.push(["Gate → expression", fmt.pct(payload.gate_expression[index])]);
  }
  if (payload.genes[state.gene]) {
    rows.push([`${state.gene} (z)`, fmt.n(payload.genes[state.gene][index], 2)]);
  }
  $("spot-facts").innerHTML = rows
    .map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");

  if (payload.probabilities) {
    $("spot-probs").innerHTML = m.layers.map((layer, i) => {
      const p = payload.probabilities[i][index];
      return `<div class="probrow"><span>${layer}</span>
        <span class="bar"><i style="width:${(p * 100).toFixed(1)}%;background:${m.layer_colors[layer]}"></i></span>
        <span class="val">${(p * 100).toFixed(0)}</span></div>`;
    }).join("");
  } else {
    $("spot-probs").innerHTML = `<span style="font-size:.78rem;color:var(--text-faint)">no predictions exported for this section</span>`;
  }
}

/* ---------------------------------------------------------------- results */

const COLUMNS = [
  { key: "label", name: "Model", type: "text" },
  { key: "macro_f1", name: "Macro F1", best: "max" },
  { key: "balanced_accuracy", name: "Bal. acc", best: "max" },
  { key: "accuracy", name: "Accuracy", best: "max" },
  { key: "kappa", name: "κ", best: "max" },
  { key: "ari", name: "ARI", best: "max" },
  { key: "adjacent_accuracy", name: "±1 layer", best: "max" },
  { key: "spatial_coherence", name: "Coherence", best: "max" },
  { key: "ece", name: "ECE", best: "min" },
  { key: "parameters", name: "Params", format: (v) => fmt.compact(v) },
];

let sortKey = "macro_f1";
let sortAsc = false;

function buildModelsTable() {
  const m = state.manifest;
  if (!m.models.length) {
    $("models-table").innerHTML = `<tbody><tr><td>No completed runs found.</td></tr></tbody>`;
    return;
  }
  state.selectedModel = state.selectedModel
    || (m.models.find((x) => x.tag === m.headline_model) || m.models[0]).tag;
  renderModelsTable();
  renderModelDetail();
}

function renderModelsTable() {
  const m = state.manifest;
  const rows = [...m.models].sort((a, b) => {
    const av = a[sortKey];
    const bv = b[sortKey];
    if (typeof av === "string") return sortAsc ? av.localeCompare(bv) : bv.localeCompare(av);
    return sortAsc ? (av ?? -Infinity) - (bv ?? -Infinity) : (bv ?? -Infinity) - (av ?? -Infinity);
  });

  const bests = {};
  for (const column of COLUMNS) {
    if (!column.best) continue;
    const values = m.models.map((x) => x[column.key]).filter((v) => v !== null && v !== undefined);
    if (values.length) bests[column.key] = column.best === "max" ? Math.max(...values) : Math.min(...values);
  }

  const head = COLUMNS.map((c) =>
    `<th data-key="${c.key}" class="${sortKey === c.key ? "sorted" + (sortAsc ? " asc" : "") : ""}">${c.name}</th>`).join("");

  const body = rows.map((row) => {
    const cells = COLUMNS.map((c) => {
      if (c.type === "text") {
        const badge = row.ssl_init ? `<span class="tag ssl">SSL</span>` : "";
        return `<td>${row.label} ${badge}</td>`;
      }
      const v = row[c.key];
      const isBest = c.best && v !== null && Math.abs(v - bests[c.key]) < 1e-9;
      return `<td class="num ${isBest ? "best" : ""}">${c.format ? c.format(v) : fmt.n(v, 3)}</td>`;
    }).join("");
    return `<tr data-tag="${row.tag}" class="${row.tag === state.selectedModel ? "sel" : ""}">${cells}</tr>`;
  }).join("");

  $("models-table").innerHTML = `<thead><tr>${head}</tr></thead><tbody>${body}</tbody>`;

  for (const th of $("models-table").querySelectorAll("th")) {
    th.addEventListener("click", () => {
      const key = th.dataset.key;
      if (key === sortKey) sortAsc = !sortAsc;
      else { sortKey = key; sortAsc = key === "label" || key === "ece"; }
      renderModelsTable();
    });
  }
  for (const tr of $("models-table").querySelectorAll("tbody tr")) {
    tr.addEventListener("click", () => {
      state.selectedModel = tr.dataset.tag;
      renderModelsTable();
      renderModelDetail();
    });
  }
}

function renderModelDetail() {
  const m = state.manifest;
  const model = m.models.find((x) => x.tag === state.selectedModel);
  if (!model) return;

  $("cm-caption").textContent = `${model.label} · row-normalised %`;
  heatmap($("confusion"), { matrix: model.confusion_matrix, labels: m.layers, cell: 44 });

  const reference = m.models.find((x) => x.tag === "expression_only");
  const series = [{
    name: model.label, color: cssVar("--accent"),
    values: m.layers.map((l) => model.per_class_f1[l]),
  }];
  if (reference && reference.tag !== model.tag) {
    series.push({
      name: reference.label, color: cssVar("--text-faint"),
      values: m.layers.map((l) => reference.per_class_f1[l]),
    });
  }
  groupedBars($("perclass"), {
    groups: m.layers, series, max: 1, height: 230, yLabel: "F1", yDigits: 1,
  });

  const curve = m.curves[model.tag];
  if (curve && curve.epoch) {
    lineChart($("curves"), {
      height: 230, xLabel: "epoch", yLabel: "macro F1",
      series: [
        { name: "train", x: curve.epoch, y: curve.train_macro_f1,
          color: cssVar("--text-faint"), dash: "4 3" },
        { name: "validation", x: curve.epoch, y: curve.val_macro_f1,
          color: cssVar("--accent"), area: true },
      ],
    });
  } else {
    $("curves").innerHTML = `<div class="loading">no curve recorded</div>`;
  }

  if (model.gate_by_layer) {
    barChart($("gate-bars"), {
      items: m.layers.filter((l) => model.gate_by_layer[l] !== undefined).map((l) => ({
        label: l, value: model.gate_by_layer[l], color: m.layer_colors[l],
        tip: `<div class="t">${l}</div><div class="r">weight on expression<b>${fmt.pct(model.gate_by_layer[l])}</b></div>` +
          `<div class="r">weight on histology<b>${fmt.pct(1 - model.gate_by_layer[l])}</b></div>`,
      })),
      max: 1, format: (v) => fmt.pct(v, 0), rowHeight: 26,
    });
    const values = Object.entries(model.gate_by_layer);
    const most = values.reduce((a, b) => (b[1] > a[1] ? b : a));
    const least = values.reduce((a, b) => (b[1] < a[1] ? b : a));
    $("gate-note").innerHTML =
      `Averaged over the held-out donor the gate puts <strong>${fmt.pct(model.gate_mean_expression, 0)}</strong>
       of its weight on expression. It leans on expression most in <strong>${most[0]}</strong>
       (${fmt.pct(most[1], 0)}) and least in <strong>${least[0]}</strong> (${fmt.pct(least[1], 0)}),
       where the histology carries relatively more of the decision.`;
  } else {
    $("gate-bars").innerHTML = `<div class="loading">this model has no fusion gate</div>`;
    $("gate-note").textContent = "Select a gated fusion model to see the modality weights.";
  }
}

/* ---------------------------------------------------------------- SSL */

function buildSslPanels() {
  const m = state.manifest;
  const ssl = m.ssl && m.ssl.ssl;
  if (!ssl) {
    $("ssl-stats").innerHTML = `<div class="stat"><div class="k">status</div>
      <div class="v">—</div><div class="s">no pretraining run exported</div></div>`;
    return;
  }
  const final = ssl.probes[ssl.probes.length - 1];
  $("ssl-stats").innerHTML = [
    { k: "Spots pretrained on", v: fmt.compact(ssl.n_spots),
      s: "no layer annotation used" },
    { k: "In-batch match", v: fmt.pct(ssl.history.batch_accuracy[ssl.history.batch_accuracy.length - 1], 0),
      s: `right partner out of ${ssl.batch_size}, during training` },
    { k: "Median rank", v: fmt.int(final.median_rank),
      s: `of ${fmt.compact(final.n_gallery)} on the held-out donor (chance ${fmt.compact(Math.round(final.n_gallery / 2))})` },
    { k: "kNN layer F1", v: fmt.n(final.knn_expression_f1, 3),
      s: `frozen expression embedding, up from ${fmt.n(ssl.probes[0].knn_expression_f1, 3)} at init` },
    { k: "Pretraining time", v: `${fmt.n(ssl.minutes, 0)} min`,
      s: `${ssl.epochs} epochs, batch ${ssl.batch_size}` },
  ].map(statCard).join("");

  lineChart($("ssl-loss"), {
    height: 250, xLabel: "epoch", yLabel: "InfoNCE loss",
    series: [
      { name: "loss", x: ssl.history.epoch, y: ssl.history.loss,
        color: cssVar("--accent"), area: true },
    ],
  });

  const probes = ssl.probes;
  const epochs = probes.map((p) => p.epoch);

  lineChart($("ssl-probe"), {
    height: 250, xLabel: "pretraining epoch", yLabel: "layer macro F1", yMin: 0,
    series: [
      { name: "expression embedding", x: epochs,
        y: probes.map((p) => p.knn_expression_f1),
        color: cssVar("--accent"), markers: true },
      { name: "image embedding", x: epochs, y: probes.map((p) => p.knn_image_f1),
        color: cssVar("--accent-2"), markers: true },
    ],
  });

  const first = probes[0];
  $("probe-note").innerHTML =
    `Freeze the encoders, take a 25-nearest-neighbour vote in embedding space and fit nothing.
     Over pretraining the expression embedding goes from <strong>${fmt.n(first.knn_expression_f1, 3)}</strong>
     to <strong>${fmt.n(final.knn_expression_f1, 3)}</strong> macro-F1 on the held-out donor, and the
     image embedding from ${fmt.n(first.knn_image_f1, 3)} to ${fmt.n(final.knn_image_f1, 3)}.
     No layer label was involved in producing those embeddings, so the laminar structure is
     something the pairing objective found on its own.`;

  lineChart($("ssl-retrieval"), {
    height: 230, xLabel: "pretraining epoch", yLabel: "recall", yMin: 0,
    // Axis digits follow the tick spacing, so a run where every recall sits
    // near zero still gets distinguishable labels.
    yFormat: (v, digits) => fmt.pct(v, Math.max(0, digits - 2)),
    series: [
      { name: "R@1", x: epochs, y: probes.map((p) => p["recall@1"]),
        color: cssVar("--accent"), markers: true },
      { name: "R@5", x: epochs, y: probes.map((p) => p["recall@5"]),
        color: cssVar("--accent-2"), markers: true },
      { name: "R@10", x: epochs, y: probes.map((p) => p["recall@10"]),
        color: cssVar("--text-faint"), markers: true },
      { name: "chance R@10", x: epochs,
        y: epochs.map(() => 10 / final.n_gallery),
        color: cssVar("--warn"), dash: "4 3", width: 1.5 },
    ],
  });

  const chance10 = 10 / final.n_gallery;
  const lift = final["recall@10"] / chance10;
  const atChance = lift < 2;
  const trained = ssl.history.batch_accuracy[ssl.history.batch_accuracy.length - 1];

  $("retrieval-note").innerHTML = atChance
    ? `Given one spot's expression vector the model ranks ${fmt.compact(final.n_gallery)}
       H&amp;E crops from the held-out donor. The correct partner lands at median rank
       <strong>${fmt.int(final.median_rank)}</strong> against a chance median of
       ${fmt.int(Math.round(final.n_gallery / 2))}, and R@10 is ${fmt.pct(final["recall@10"], 2)}
       where chance is ${fmt.pct(chance10, 2)}. Spot-level matching across modalities does not
       transfer to a new donor at all.
       <br><br>
       The objective did train. By the last epoch the model picks the right partner out of its
       own batch <strong>${fmt.pct(trained, 0)}</strong> of the time, and the frozen embeddings
       carry layer information, as the probes above show. Whatever lets it pair one
       <em>particular</em> spot with one <em>particular</em> crop seems tied to donor-specific
       staining and morphology, while the coarser laminar structure survives the move to a
       second brain.`
    : `Given one spot's expression vector, the model ranks all ${fmt.compact(final.n_gallery)}
       H&amp;E crops on the held-out donor and the correct one lands in the top 10
       <strong>${fmt.pct(final["recall@10"], 1)}</strong> of the time, about
       ${fmt.n(lift, 1)}× chance. Immediate spatial neighbours are excluded, so this is not the
       model retrieving a near-duplicate.`;
}

function buildEfficiency() {
  const m = state.manifest;
  const points = m.efficiency || [];
  if (!points.length) {
    $("efficiency-curve").innerHTML = `<div class="loading">no label-efficiency runs exported</div>`;
    $("efficiency-table").innerHTML = "";
    return;
  }
  const scratch = points.filter((p) => p.arm === "scratch");
  const ssl = points.filter((p) => p.arm === "ssl");

  lineChart($("efficiency-curve"), {
    height: 280, xLabel: "labelled training spots", yLabel: "macro F1",
    xTicks: scratch.map((p) => Math.log10(p.n_train)),
    xFormat: (v) => fmt.compact(Math.round(Math.pow(10, v))),
    series: [
      { name: "from scratch", x: scratch.map((p) => Math.log10(p.n_train)),
        y: scratch.map((p) => p.macro_f1), color: cssVar("--text-faint"), markers: true },
      { name: "SSL initialised", x: ssl.map((p) => Math.log10(p.n_train)),
        y: ssl.map((p) => p.macro_f1), color: cssVar("--accent"), markers: true },
    ],
  });

  const rows = scratch.map((s) => {
    const partner = ssl.find((x) => x.percent === s.percent);
    const delta = partner ? partner.macro_f1 - s.macro_f1 : null;
    return `<tr>
      <td>${s.percent}%</td>
      <td class="num">${fmt.int(s.n_train)}</td>
      <td class="num">${fmt.n(s.macro_f1, 3)}</td>
      <td class="num">${partner ? fmt.n(partner.macro_f1, 3) : "—"}</td>
      <td class="num ${delta > 0 ? "best" : ""}">${delta === null ? "—" : (delta > 0 ? "+" : "") + fmt.n(delta, 3)}</td>
    </tr>`;
  }).join("");
  $("efficiency-table").innerHTML = `
    <thead><tr><th>Labels</th><th>Spots</th><th>Scratch</th><th>SSL init</th><th>Δ</th></tr></thead>
    <tbody>${rows}</tbody>`;

  const gains = scratch.map((s) => {
    const partner = ssl.find((x) => x.percent === s.percent);
    return partner ? { percent: s.percent, delta: partner.macro_f1 - s.macro_f1 } : null;
  }).filter(Boolean);
  const smallest = gains[0];
  const full = gains.find((g) => g.percent === 100);
  // Where the SSL arm stops being ahead. This is the number worth quoting:
  // the two arms cross somewhere, and where they cross is the finding.
  const crossover = gains.find((g) => g.delta < 0);

  let verdict = "";
  if (smallest && smallest.delta > 0.02) {
    verdict = `At <strong>${smallest.percent}%</strong> of the annotations
      (${fmt.int(scratch[0].n_train)} labelled spots) the pretrained model is worth
      <strong>+${fmt.n(smallest.delta, 3)}</strong> macro-F1, which is the largest effect
      anywhere in this project.`;
    if (crossover) {
      verdict += ` The advantage shrinks as labels accumulate and reverses at
        <strong>${crossover.percent}%</strong>.`;
    }
    if (full && full.delta < 0) {
      verdict += ` With every label available, starting from the self-supervised
        checkpoint is <strong>${fmt.n(full.delta, 3)}</strong> macro-F1 <em>worse</em> than
        starting from ImageNet weights. Pretraining here buys label efficiency, not a
        better ceiling, and past a few thousand annotated spots it is actively the wrong
        choice.`;
    }
  } else if (smallest) {
    verdict = `The pretrained and from-scratch arms stay within
      ${fmt.n(Math.abs(smallest.delta), 3)} macro-F1 of each other even at the smallest
      label budget, so pretraining is not buying label efficiency on this dataset.`;
  }
  $("efficiency-note").innerHTML = verdict;
}

function buildAblation() {
  const m = state.manifest;
  const tags = ["probe_ssl", "probe_ssl_no_imagenet", "probe_ssl_no_neighbour_mask"];
  const rows = tags.map((tag) => m.models.find((x) => x.tag === tag)).filter(Boolean);
  if (!rows.length) {
    $("ssl-ablation").innerHTML = `<div class="loading">no ablation runs exported</div>`;
    $("ablation-note").textContent = "";
    return;
  }
  barChart($("ssl-ablation"), {
    items: rows.map((row) => ({
      label: row.label.replace("Frozen SSL", "").replace("encoders + linear head", "full recipe")
        .replace(/^[,\s]+/, "") || "full recipe",
      value: row.macro_f1,
      color: row.tag === "probe_ssl" ? cssVar("--accent") : cssVar("--text-faint"),
      tip: `<div class="t">${row.label}</div><div class="r">macro F1<b>${fmt.n(row.macro_f1, 3)}</b></div>` +
        `<div class="r">accuracy<b>${fmt.pct(row.accuracy)}</b></div>`,
    })),
    max: Math.max(...rows.map((r) => r.macro_f1)) * 1.18,
    format: (v) => fmt.n(v, 3), rowHeight: 30, margin: { left: 190 },
  });

  const full = rows.find((r) => r.tag === "probe_ssl");
  const noMask = rows.find((r) => r.tag === "probe_ssl_no_neighbour_mask");
  const noImagenet = rows.find((r) => r.tag === "probe_ssl_no_imagenet");
  const parts = [];
  if (full && noMask) {
    const d = full.macro_f1 - noMask.macro_f1;
    parts.push(`Dropping the neighbour mask changes macro-F1 by <strong>${(d >= 0 ? "−" : "+") + fmt.n(Math.abs(d), 3)}</strong>,
      so treating a spot's neighbours as negatives ${d > 0.005 ? "does measurably hurt" : "makes little difference to"} the representation.`);
  }
  if (full && noImagenet) {
    const d = full.macro_f1 - noImagenet.macro_f1;
    parts.push(`Starting the image encoder from random weights instead of ImageNet costs
      <strong>${fmt.n(d, 3)}</strong> macro-F1, so ${d > 0.03
        ? "the contrastive objective alone does not fully replace generic visual pretraining on this amount of tissue"
        : "the contrastive objective recovers most of what ImageNet initialisation provides"}.`);
  }
  $("ablation-note").innerHTML = parts.join(" ");
}

/* ---------------------------------------------------------------- cross-modal */

function buildRegression() {
  const m = state.manifest;
  const block = m.regression && m.regression.expr_from_image;
  if (!block) {
    $("regression-stats").innerHTML = `<div class="stat"><div class="k">status</div>
      <div class="v">—</div><div class="s">no regression run exported</div></div>`;
    return;
  }
  $("regression-stats").innerHTML = [
    { k: "Mean r", v: fmt.n(block.mean_r, 3), s: `across ${fmt.compact(m.dataset.n_genes_modelled)} genes` },
    { k: "Median r", v: fmt.n(block.median_r, 3), s: "held-out donor" },
    { k: "Genes above r = 0.3", v: fmt.pct(block.above_03, 1), s: "recoverable from morphology" },
    { k: "Genes above r = 0.5", v: fmt.pct(block.above_05, 1), s: "strongly recoverable" },
    { k: "Best gene", v: block.top_genes[0].gene, s: `r = ${fmt.n(block.top_genes[0].r, 2)}` },
  ].map(statCard).join("");

  histogram($("gene-hist"), {
    counts: block.histogram.counts, edges: block.histogram.edges, height: 280,
    xLabel: "Pearson r", yLabel: "genes",
    colorAt: (v) => ramp.viridis(Math.max(0, Math.min(1, (v + 0.2) / 1.0))),
    markers: [
      { value: block.mean_r, label: `mean ${fmt.n(block.mean_r, 2)}`, color: cssVar("--warn") },
    ],
  });

  barChart($("top-genes"), {
    items: block.top_genes.slice(0, 24).map((g) => ({
      label: g.gene, value: g.r, color: ramp.viridis(Math.min(1, g.r / 0.85)),
      tip: `<div class="t">${g.gene}</div><div class="r">Pearson r<b>${fmt.n(g.r, 3)}</b></div>`,
    })),
    max: Math.max(...block.top_genes.map((g) => g.r)) * 1.1,
    format: (v) => fmt.n(v, 2), rowHeight: 23, margin: { left: 78, right: 40 },
  });

  const scatters = block.scatter || [];
  $("gene-scatters").innerHTML = scatters.map((s, i) =>
    `<div><div style="font-size:.8rem;font-weight:600;margin-bottom:4px">${s.gene}
      <span style="color:var(--text-faint);font-weight:400">r = ${fmt.n(s.r, 2)}</span></div>
      <div id="scatter-${i}"></div></div>`).join("");
  scatters.forEach((s, i) => {
    scatter($(`scatter-${i}`), {
      x: s.measured, y: s.predicted, height: 190, radius: 1.6, opacity: 0.4,
      xLabel: "measured (z)", yLabel: "predicted", diagonal: true,
      colors: s.measured.map((v) => ramp.magma(Math.max(0, Math.min(1, (v + 2) / 5)))),
    });
  });
}

function buildUmap() {
  const m = state.manifest;
  if (!m.embedding) {
    $("umap").innerHTML = `<div class="loading">no embedding exported</div>`;
    return;
  }
  const e = m.embedding;
  scatter($("umap"), {
    x: e.x, y: e.y, height: 320, radius: 1.9, opacity: 0.65,
    colors: e.label.map((v) => m.layer_colors[m.layers[v]] || "#bbb"),
    xLabel: "UMAP 1", yLabel: "UMAP 2",
  });
  $("umap").insertAdjacentHTML("beforeend",
    `<div class="legend" style="margin-top:10px">` +
    m.layers.map((l) => `<span class="item" style="cursor:default">
      <span class="sw" style="background:${m.layer_colors[l]}"></span>${l}</span>`).join("") +
    `</div>`);
}

/* ---------------------------------------------------------------- chrome */

function buildFooter() {
  const m = state.manifest;
  const total = m.models.reduce((a, b) => a + (b.minutes || 0), 0)
    + (m.ssl && m.ssl.ssl ? m.ssl.ssl.minutes : 0);
  $("runtime-note").innerHTML =
    `The full set of runs on this page took about ${fmt.n(total / 60, 1)} hours on a single
     GTX 1080 Ti. Downloading the raw data is the slow part; the twelve
     full-resolution slide images are roughly 6.4 GB.`;
  $("footer-text").innerHTML =
    `Data: Maynard et al. 2021, spatialLIBD. Figures generated from the exported run
     artefacts in <code>docs/data/</code>. Page is static and needs no server.`;
}

function initTheme() {
  const stored = localStorage.getItem("theme");
  const preferred = stored
    || (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  document.documentElement.dataset.theme = preferred;
  $("theme-toggle").addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    localStorage.setItem("theme", next);
    redrawAll();
  });
}

function redrawAll() {
  renderModelDetail();
  buildSslPanels();
  buildEfficiency();
  buildAblation();
  buildRegression();
  buildUmap();
  buildLayerBars();
  for (const map of [mapMain, mapLeft, mapRight]) if (map && map.section) map.draw();
}

function initScrollSpy() {
  const links = [...document.querySelectorAll(".navlinks a")];
  const sections = links.map((a) => document.querySelector(a.getAttribute("href")));
  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      const i = sections.indexOf(entry.target);
      links.forEach((a, j) => a.classList.toggle("active", i === j));
    }
  }, { rootMargin: "-45% 0px -50% 0px" });
  for (const section of sections) if (section) observer.observe(section);
}

boot().catch((error) => {
  document.querySelector("main").innerHTML =
    `<div class="wrap"><div class="loading">${error.message}</div></div>`;
  console.error(error);
});
