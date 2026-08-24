/* Small SVG charting layer.
 *
 * The page has no build step and no third-party dependencies, so the handful of
 * chart types it needs are drawn directly. Everything here takes plain arrays
 * and returns nothing; call it again to redraw.
 */

const SVG_NS = "http://www.w3.org/2000/svg";

export function el(tag, attrs = {}, parent = null) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v !== null && v !== undefined) node.setAttribute(k, v);
  }
  if (parent) parent.appendChild(node);
  return node;
}

export const fmt = {
  n: (v, d = 3) => (v === null || v === undefined || Number.isNaN(v) ? "—" : (+v).toFixed(d)),
  pct: (v, d = 1) => (v === null || v === undefined ? "—" : (100 * v).toFixed(d) + "%"),
  int: (v) => (v === null || v === undefined ? "—" : (+v).toLocaleString("en-US")),
  compact: (v) => {
    if (v === null || v === undefined) return "—";
    if (v >= 1e6) return (v / 1e6).toFixed(1) + "M";
    if (v >= 1e4) return Math.round(v / 1e3) + "k";
    if (v >= 1e3) return (v / 1e3).toFixed(1) + "k";
    return String(v);
  },
};

/* ---------- tooltip ---------- */

let tip;
export function tooltip() {
  if (!tip) {
    tip = document.createElement("div");
    tip.className = "tooltip";
    document.body.appendChild(tip);
  }
  return tip;
}

export function showTip(html, event, offset = 14) {
  const node = tooltip();
  node.innerHTML = html;
  node.style.opacity = "1";
  const box = node.getBoundingClientRect();
  let x = event.clientX + offset;
  let y = event.clientY + offset;
  if (x + box.width > window.innerWidth - 8) x = event.clientX - box.width - offset;
  if (y + box.height > window.innerHeight - 8) y = event.clientY - box.height - offset;
  node.style.left = Math.max(6, x) + "px";
  node.style.top = Math.max(6, y) + "px";
}

export function hideTip() {
  if (tip) tip.style.opacity = "0";
}

/* ---------- colour ---------- */

export function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/* Perceptually reasonable sequential ramp, dark-blue to warm-yellow. */
const VIRIDIS = [
  [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142], [38, 130, 142],
  [31, 158, 137], [53, 183, 121], [109, 205, 89], [180, 222, 44], [253, 231, 37],
];

const MAGMA = [
  [0, 0, 4], [28, 16, 68], [79, 18, 123], [129, 37, 129], [181, 54, 122],
  [229, 80, 100], [251, 135, 97], [254, 194, 135], [252, 253, 191],
];

function rampLookup(stops, t) {
  const clamped = Math.max(0, Math.min(1, t));
  const scaled = clamped * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(scaled));
  const f = scaled - i;
  const a = stops[i];
  const b = stops[i + 1];
  return `rgb(${Math.round(a[0] + f * (b[0] - a[0]))},${Math.round(a[1] + f * (b[1] - a[1]))},${Math.round(a[2] + f * (b[2] - a[2]))})`;
}

export const ramp = {
  viridis: (t) => rampLookup(VIRIDIS, t),
  magma: (t) => rampLookup(MAGMA, t),
  /* Diverging ramp for gate weights and anything else centred on a midpoint. */
  balance: (t) => {
    const c = Math.max(0, Math.min(1, t));
    if (c < 0.5) {
      const f = c / 0.5;
      return `rgb(${Math.round(33 + f * 210)},${Math.round(102 + f * 141)},${Math.round(172 + f * 71)})`;
    }
    const f = (c - 0.5) / 0.5;
    return `rgb(${Math.round(243 - f * 65)},${Math.round(243 - f * 158)},${Math.round(243 - f * 200)})`;
  },
};

export function rampCss(fn, stops = 12) {
  const parts = [];
  for (let i = 0; i < stops; i++) parts.push(fn(i / (stops - 1)));
  return `linear-gradient(to right, ${parts.join(",")})`;
}

/* ---------- axes ---------- */

function niceTicks(min, max, count = 5) {
  if (min === max) return [min];
  const span = max - min;
  const raw = span / count;
  const magnitude = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / magnitude;
  const step = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * magnitude;
  const out = [];
  for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) {
    out.push(Math.round(v / step) * step);
  }
  return out;
}

/* Decimal places that keep neighbouring ticks distinguishable. Without this a
 * narrow range prints the same label several times down the axis. */
function tickDigits(ticks) {
  if (ticks.length < 2) return 2;
  const step = Math.abs(ticks[1] - ticks[0]);
  if (!step) return 2;
  return Math.max(0, Math.min(6, Math.ceil(-Math.log10(step))));
}

function frame(node, opts) {
  const width = opts.width || node.clientWidth || 520;
  const height = opts.height || 260;
  node.innerHTML = "";
  const svg = el("svg", {
    class: "chart", viewBox: `0 0 ${width} ${height}`,
    preserveAspectRatio: "xMidYMid meet",
  }, node);
  return { svg, width, height };
}

/* ---------- line chart ---------- */

export function lineChart(node, opts) {
  const m = Object.assign({ top: 14, right: 16, bottom: 34, left: 48 }, opts.margin);
  const { svg, width, height } = frame(node, opts);
  const iw = width - m.left - m.right;
  const ih = height - m.top - m.bottom;

  const series = opts.series.filter((s) => s.visible !== false && s.y.some((v) => v !== null));
  if (!series.length) return;

  const xs = series.flatMap((s) => s.x);
  const ys = series.flatMap((s) => s.y).filter((v) => v !== null && Number.isFinite(v));
  let x0 = opts.xMin ?? Math.min(...xs);
  let x1 = opts.xMax ?? Math.max(...xs);
  let y0 = opts.yMin ?? Math.min(...ys);
  let y1 = opts.yMax ?? Math.max(...ys);
  if (y0 === y1) { y0 -= 0.5; y1 += 0.5; }
  if (opts.padY !== false) {
    const pad = (y1 - y0) * 0.08;
    y0 -= pad; y1 += pad;
  }
  if (x0 === x1) { x0 -= 0.5; x1 += 0.5; }

  const sx = (v) => m.left + ((v - x0) / (x1 - x0)) * iw;
  const sy = (v) => m.top + ih - ((v - y0) / (y1 - y0)) * ih;

  const grid = el("g", { class: "grid" }, svg);
  const yTicks = niceTicks(y0, y1, opts.yTicks || 5);
  const yDigits = opts.yDigits ?? tickDigits(yTicks);
  for (const t of yTicks) {
    el("line", { x1: m.left, x2: m.left + iw, y1: sy(t), y2: sy(t) }, grid);
    el("text", { x: m.left - 8, y: sy(t) + 3.5, "text-anchor": "end" }, svg)
      .textContent = opts.yFormat ? opts.yFormat(t, yDigits) : fmt.n(t, yDigits);
  }
  const xTicks = opts.xTicks || niceTicks(x0, x1, 6);
  for (const t of xTicks) {
    el("text", { x: sx(t), y: m.top + ih + 17, "text-anchor": "middle" }, svg)
      .textContent = opts.xFormat ? opts.xFormat(t) : String(Math.round(t));
  }
  el("line", { x1: m.left, x2: m.left + iw, y1: m.top + ih, y2: m.top + ih,
    stroke: cssVar("--border-strong") }, svg);

  if (opts.xLabel) {
    el("text", { x: m.left + iw / 2, y: height - 2, "text-anchor": "middle",
      class: "axlabel" }, svg).textContent = opts.xLabel;
  }
  if (opts.yLabel) {
    el("text", { x: 11, y: m.top + ih / 2, "text-anchor": "middle", class: "axlabel",
      transform: `rotate(-90 11 ${m.top + ih / 2})` }, svg).textContent = opts.yLabel;
  }

  for (const s of series) {
    const points = [];
    for (let i = 0; i < s.x.length; i++) {
      if (s.y[i] === null || !Number.isFinite(s.y[i])) continue;
      points.push(`${sx(s.x[i]).toFixed(2)},${sy(s.y[i]).toFixed(2)}`);
    }
    if (!points.length) continue;
    if (s.area) {
      el("polygon", {
        points: `${sx(s.x[0])},${sy(y0)} ${points.join(" ")} ${sx(s.x[s.x.length - 1])},${sy(y0)}`,
        fill: s.color, opacity: 0.1,
      }, svg);
    }
    el("polyline", {
      points: points.join(" "), fill: "none", stroke: s.color,
      "stroke-width": s.width || 2, "stroke-linejoin": "round",
      "stroke-linecap": "round", "stroke-dasharray": s.dash || null,
    }, svg);
    if (s.markers) {
      for (let i = 0; i < s.x.length; i++) {
        if (s.y[i] === null || !Number.isFinite(s.y[i])) continue;
        el("circle", { cx: sx(s.x[i]), cy: sy(s.y[i]), r: 3.2, fill: s.color,
          stroke: cssVar("--bg-panel"), "stroke-width": 1.4 }, svg);
      }
    }
  }

  /* One shared crosshair rather than a hit target per point. */
  if (opts.hover !== false) {
    const line = el("line", { y1: m.top, y2: m.top + ih, stroke: cssVar("--border-strong"),
      "stroke-dasharray": "3 3", opacity: 0 }, svg);
    const overlay = el("rect", { x: m.left, y: m.top, width: iw, height: ih,
      fill: "transparent" }, svg);
    overlay.addEventListener("mousemove", (event) => {
      const box = svg.getBoundingClientRect();
      const px = ((event.clientX - box.left) / box.width) * width;
      const value = x0 + ((px - m.left) / iw) * (x1 - x0);
      let best = null;
      for (const s of series) {
        let bi = 0;
        for (let i = 0; i < s.x.length; i++) {
          if (Math.abs(s.x[i] - value) < Math.abs(s.x[bi] - value)) bi = i;
        }
        if (best === null || Math.abs(s.x[bi] - value) < Math.abs(best.xv - value)) {
          best = { xv: s.x[bi], index: bi };
        }
      }
      if (!best) return;
      line.setAttribute("x1", sx(best.xv));
      line.setAttribute("x2", sx(best.xv));
      line.setAttribute("opacity", 1);
      const rows = series.map((s) => {
        const i = s.x.indexOf(best.xv);
        const v = i >= 0 ? s.y[i] : null;
        return `<div class="r"><span style="color:${s.color}">●</span>&nbsp;${s.name}<b>${
          v === null ? "—" : (opts.yFormat ? opts.yFormat(v) : fmt.n(v, opts.tipDigits ?? 4))}</b></div>`;
      }).join("");
      showTip(`<div class="t">${opts.xLabel || "x"} ${
        opts.xFormat ? opts.xFormat(best.xv) : best.xv}</div>${rows}`, event);
    });
    overlay.addEventListener("mouseleave", () => { line.setAttribute("opacity", 0); hideTip(); });
  }
}

/* ---------- horizontal bars ---------- */

export function barChart(node, opts) {
  const m = Object.assign({ top: 8, right: 44, bottom: 26, left: 62 }, opts.margin);
  const rowHeight = opts.rowHeight || 24;
  const height = opts.height || m.top + m.bottom + opts.items.length * rowHeight;
  const { svg, width } = frame(node, { ...opts, height });
  const iw = width - m.left - m.right;

  const max = opts.max ?? Math.max(...opts.items.map((d) => d.value), 0.0001);
  opts.items.forEach((d, i) => {
    const y = m.top + i * rowHeight;
    const w = Math.max(1, (d.value / max) * iw);
    el("text", { x: m.left - 9, y: y + rowHeight / 2 + 3.5, "text-anchor": "end" }, svg)
      .textContent = d.label;
    el("rect", { x: m.left, y: y + 3, width: iw, height: rowHeight - 10, rx: 3,
      fill: cssVar("--bg-sunken") }, svg);
    const bar = el("rect", { x: m.left, y: y + 3, width: w, height: rowHeight - 10, rx: 3,
      fill: d.color || cssVar("--accent") }, svg);
    el("text", { x: m.left + w + 7, y: y + rowHeight / 2 + 3.5 }, svg)
      .textContent = opts.format ? opts.format(d.value) : fmt.n(d.value, 3);
    if (d.tip) {
      bar.addEventListener("mousemove", (e) => showTip(d.tip, e));
      bar.addEventListener("mouseleave", hideTip);
    }
  });
}

/* ---------- grouped vertical bars ---------- */

export function groupedBars(node, opts) {
  const m = Object.assign({ top: 12, right: 12, bottom: 40, left: 46 }, opts.margin);
  const { svg, width, height } = frame(node, opts);
  const iw = width - m.left - m.right;
  const ih = height - m.top - m.bottom;

  const groups = opts.groups;
  const series = opts.series;
  const max = opts.max ?? Math.max(...series.flatMap((s) => s.values), 0.0001);
  const groupWidth = iw / groups.length;
  const barWidth = Math.min(26, (groupWidth * 0.74) / series.length);

  const grid = el("g", { class: "grid" }, svg);
  for (const t of niceTicks(0, max, 4)) {
    const y = m.top + ih - (t / max) * ih;
    el("line", { x1: m.left, x2: m.left + iw, y1: y, y2: y }, grid);
    el("text", { x: m.left - 8, y: y + 3.5, "text-anchor": "end" }, svg)
      .textContent = fmt.n(t, opts.yDigits ?? 2);
  }

  groups.forEach((g, gi) => {
    const cx = m.left + groupWidth * (gi + 0.5);
    el("text", { x: cx, y: m.top + ih + 16, "text-anchor": "middle" }, svg).textContent = g;
    series.forEach((s, si) => {
      const v = s.values[gi];
      if (v === null || v === undefined) return;
      const h = Math.max(1, (v / max) * ih);
      const x = cx - (series.length * barWidth) / 2 + si * barWidth;
      const rect = el("rect", { x: x + 1, y: m.top + ih - h, width: barWidth - 2, height: h,
        rx: 2, fill: s.color }, svg);
      rect.addEventListener("mousemove", (e) => showTip(
        `<div class="t">${g}</div><div class="r">${s.name}<b>${fmt.n(v, 3)}</b></div>`, e));
      rect.addEventListener("mouseleave", hideTip);
    });
  });
  el("line", { x1: m.left, x2: m.left + iw, y1: m.top + ih, y2: m.top + ih,
    stroke: cssVar("--border-strong") }, svg);
  if (opts.yLabel) {
    el("text", { x: 11, y: m.top + ih / 2, "text-anchor": "middle", class: "axlabel",
      transform: `rotate(-90 11 ${m.top + ih / 2})` }, svg).textContent = opts.yLabel;
  }
}

/* ---------- confusion matrix ---------- */

export function heatmap(node, opts) {
  const labels = opts.labels;
  const n = labels.length;
  const cell = opts.cell || 42;
  const m = { top: 26, right: 10, bottom: 34, left: 46 };
  const width = m.left + m.right + n * cell;
  const height = m.top + m.bottom + n * cell;
  const { svg } = frame(node, { width, height });

  const matrix = opts.matrix;
  const rowTotals = matrix.map((row) => row.reduce((a, b) => a + b, 0));

  for (let i = 0; i < n; i++) {
    el("text", { x: m.left - 8, y: m.top + i * cell + cell / 2 + 3.5, "text-anchor": "end",
      "font-weight": 600 }, svg).textContent = labels[i];
    el("text", { x: m.left + i * cell + cell / 2, y: m.top - 9, "text-anchor": "middle",
      "font-weight": 600 }, svg).textContent = labels[i];
  }

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const count = matrix[i][j];
      const share = rowTotals[i] ? count / rowTotals[i] : 0;
      const x = m.left + j * cell;
      const y = m.top + i * cell;
      const rect = el("rect", { x: x + 1, y: y + 1, width: cell - 2, height: cell - 2,
        rx: 4, fill: ramp.viridis(share), opacity: share < 0.008 ? 0.16 : 1 }, svg);
      if (share >= 0.06) {
        el("text", { x: x + cell / 2, y: y + cell / 2 + 3.5, "text-anchor": "middle",
          fill: share > 0.55 ? "#10240f" : "#ffffff", "font-size": 10.5,
          "font-weight": 600 }, svg).textContent = (share * 100).toFixed(0);
      }
      rect.addEventListener("mousemove", (e) => showTip(
        `<div class="t">true ${labels[i]} → predicted ${labels[j]}</div>` +
        `<div class="r">spots<b>${fmt.int(count)}</b></div>` +
        `<div class="r">row share<b>${fmt.pct(share)}</b></div>`, e));
      rect.addEventListener("mouseleave", hideTip);
    }
  }
  el("text", { x: m.left + (n * cell) / 2, y: height - 4, "text-anchor": "middle",
    class: "axlabel" }, svg).textContent = "predicted";
  el("text", { x: 11, y: m.top + (n * cell) / 2, "text-anchor": "middle", class: "axlabel",
    transform: `rotate(-90 11 ${m.top + (n * cell) / 2})` }, svg).textContent = "annotated";
}

/* ---------- scatter ---------- */

export function scatter(node, opts) {
  const m = Object.assign({ top: 12, right: 12, bottom: 34, left: 46 }, opts.margin);
  const { svg, width, height } = frame(node, opts);
  const iw = width - m.left - m.right;
  const ih = height - m.top - m.bottom;

  const xs = opts.x;
  const ys = opts.y;
  const x0 = opts.xMin ?? Math.min(...xs);
  const x1 = opts.xMax ?? Math.max(...xs);
  const y0 = opts.yMin ?? Math.min(...ys);
  const y1 = opts.yMax ?? Math.max(...ys);
  const sx = (v) => m.left + ((v - x0) / (x1 - x0 || 1)) * iw;
  const sy = (v) => m.top + ih - ((v - y0) / (y1 - y0 || 1)) * ih;

  const grid = el("g", { class: "grid" }, svg);
  const yTicks = niceTicks(y0, y1, 4);
  const xTicks = niceTicks(x0, x1, 4);
  for (const t of yTicks) {
    el("line", { x1: m.left, x2: m.left + iw, y1: sy(t), y2: sy(t) }, grid);
    el("text", { x: m.left - 8, y: sy(t) + 3.5, "text-anchor": "end" }, svg)
      .textContent = fmt.n(t, tickDigits(yTicks));
  }
  for (const t of xTicks) {
    el("text", { x: sx(t), y: m.top + ih + 16, "text-anchor": "middle" }, svg)
      .textContent = fmt.n(t, tickDigits(xTicks));
  }

  if (opts.diagonal) {
    const lo = Math.max(x0, y0);
    const hi = Math.min(x1, y1);
    el("line", { x1: sx(lo), y1: sy(lo), x2: sx(hi), y2: sy(hi),
      stroke: cssVar("--border-strong"), "stroke-dasharray": "4 4" }, svg);
  }

  const group = el("g", {}, svg);
  const radius = opts.radius || 2;
  for (let i = 0; i < xs.length; i++) {
    el("circle", { cx: sx(xs[i]).toFixed(1), cy: sy(ys[i]).toFixed(1), r: radius,
      fill: opts.colors ? opts.colors[i] : cssVar("--accent"),
      opacity: opts.opacity ?? 0.55 }, group);
  }

  if (opts.xLabel) {
    el("text", { x: m.left + iw / 2, y: height - 2, "text-anchor": "middle",
      class: "axlabel" }, svg).textContent = opts.xLabel;
  }
  if (opts.yLabel) {
    el("text", { x: 11, y: m.top + ih / 2, "text-anchor": "middle", class: "axlabel",
      transform: `rotate(-90 11 ${m.top + ih / 2})` }, svg).textContent = opts.yLabel;
  }
}

/* ---------- histogram ---------- */

export function histogram(node, opts) {
  const m = Object.assign({ top: 12, right: 12, bottom: 36, left: 48 }, opts.margin);
  const { svg, width, height } = frame(node, opts);
  const iw = width - m.left - m.right;
  const ih = height - m.top - m.bottom;

  const counts = opts.counts;
  const edges = opts.edges;
  const max = Math.max(...counts);
  const x0 = edges[0];
  const x1 = edges[edges.length - 1];
  const sx = (v) => m.left + ((v - x0) / (x1 - x0)) * iw;

  const grid = el("g", { class: "grid" }, svg);
  for (const t of niceTicks(0, max, 4)) {
    const y = m.top + ih - (t / max) * ih;
    el("line", { x1: m.left, x2: m.left + iw, y1: y, y2: y }, grid);
    el("text", { x: m.left - 8, y: y + 3.5, "text-anchor": "end" }, svg)
      .textContent = fmt.compact(t);
  }

  counts.forEach((count, i) => {
    const left = sx(edges[i]);
    const right = sx(edges[i + 1]);
    const h = (count / max) * ih;
    const mid = (edges[i] + edges[i + 1]) / 2;
    const rect = el("rect", {
      x: left, y: m.top + ih - h, width: Math.max(0.7, right - left - 0.6), height: h,
      fill: opts.colorAt ? opts.colorAt(mid) : cssVar("--accent"),
    }, svg);
    rect.addEventListener("mousemove", (e) => showTip(
      `<div class="t">${opts.xLabel || "bin"} ${fmt.n(edges[i], 2)} to ${fmt.n(edges[i + 1], 2)}</div>` +
      `<div class="r">genes<b>${fmt.int(count)}</b></div>`, e));
    rect.addEventListener("mouseleave", hideTip);
  });

  const xTicks = niceTicks(x0, x1, 6);
  for (const t of xTicks) {
    el("text", { x: sx(t), y: m.top + ih + 16, "text-anchor": "middle" }, svg)
      .textContent = fmt.n(t, tickDigits(xTicks));
  }
  el("line", { x1: m.left, x2: m.left + iw, y1: m.top + ih, y2: m.top + ih,
    stroke: cssVar("--border-strong") }, svg);

  for (const marker of opts.markers || []) {
    el("line", { x1: sx(marker.value), x2: sx(marker.value), y1: m.top, y2: m.top + ih,
      stroke: marker.color || cssVar("--warn"), "stroke-width": 1.5,
      "stroke-dasharray": "4 3" }, svg);
    el("text", { x: sx(marker.value) + 5, y: m.top + 11, fill: marker.color || cssVar("--warn"),
      "font-size": 10, "font-weight": 600 }, svg).textContent = marker.label;
  }

  if (opts.xLabel) {
    el("text", { x: m.left + iw / 2, y: height - 2, "text-anchor": "middle",
      class: "axlabel" }, svg).textContent = opts.xLabel;
  }
  if (opts.yLabel) {
    el("text", { x: 11, y: m.top + ih / 2, "text-anchor": "middle", class: "axlabel",
      transform: `rotate(-90 11 ${m.top + ih / 2})` }, svg).textContent = opts.yLabel;
  }
}
