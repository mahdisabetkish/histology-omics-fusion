/* Canvas renderer for a Visium section.
 *
 * Roughly four thousand spots are drawn per section and the colouring changes
 * whenever a control moves, so each spot is a filled arc on a canvas rather
 * than an SVG node. Hit testing for the hover preview goes through a uniform
 * bucket grid, which keeps the cursor responsive without a spatial index
 * library.
 *
 * Spot coordinates arrive in full-resolution slide pixels. The backdrop JPEG
 * was produced from the same slide at a known scale factor, so one multiply
 * puts the spots on top of the tissue they came from.
 */

export class SpotMap {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.section = null;
    this.backdrop = null;
    this.showBackdrop = true;
    this.radius = 3.2;
    this.colorOf = () => "#888";
    this.dimmed = () => false;
    this.highlight = -1;
    this.pinned = -1;
    this._grid = null;
    this._resize = () => this.draw();
    window.addEventListener("resize", this._resize);
  }

  destroy() {
    window.removeEventListener("resize", this._resize);
  }

  setSection(payload, tissueInfo, backdropUrl) {
    this.section = payload;
    this.tissue = tissueInfo;
    this.scale = tissueInfo ? tissueInfo.scale : 1;

    this.px = payload.px_col.map((v) => v * this.scale);
    this.py = payload.px_row.map((v) => v * this.scale);

    if (tissueInfo) {
      this.width = tissueInfo.width;
      this.height = tissueInfo.height;
    } else {
      this.width = Math.max(...this.px) + 20;
      this.height = Math.max(...this.py) + 20;
    }

    this._buildGrid();
    this.backdrop = null;
    if (backdropUrl) {
      const image = new Image();
      image.onload = () => { this.backdrop = image; this.draw(); };
      image.src = backdropUrl;
    }
    return this;
  }

  _buildGrid() {
    const cell = 24;
    const grid = new Map();
    for (let i = 0; i < this.px.length; i++) {
      const key = `${Math.floor(this.px[i] / cell)},${Math.floor(this.py[i] / cell)}`;
      if (!grid.has(key)) grid.set(key, []);
      grid.get(key).push(i);
    }
    this._grid = { cell, grid };
  }

  setColoring(colorOf, dimmed) {
    this.colorOf = colorOf;
    this.dimmed = dimmed || (() => false);
    return this;
  }

  draw() {
    if (!this.section) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const cssWidth = this.canvas.clientWidth || this.width;
    const cssHeight = cssWidth * (this.height / this.width);

    this.canvas.width = Math.round(cssWidth * dpr);
    this.canvas.height = Math.round(cssHeight * dpr);
    this.canvas.style.height = cssHeight + "px";

    const ctx = this.ctx;
    const k = (cssWidth / this.width) * dpr;
    this._k = k;

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    const style = getComputedStyle(document.documentElement);
    ctx.fillStyle = style.getPropertyValue("--bg-sunken").trim() || "#eee";
    ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

    if (this.backdrop && this.showBackdrop) {
      ctx.globalAlpha = 1;
      ctx.drawImage(this.backdrop, 0, 0, this.canvas.width, this.canvas.height);
      ctx.globalAlpha = 0.16;
      ctx.fillStyle = style.getPropertyValue("--bg-panel").trim() || "#fff";
      ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
      ctx.globalAlpha = 1;
    }

    const r = this.radius * k;
    const n = this.px.length;
    for (let i = 0; i < n; i++) {
      const colour = this.colorOf(i);
      if (!colour) continue;
      ctx.globalAlpha = this.dimmed(i) ? 0.12 : 0.92;
      ctx.fillStyle = colour;
      ctx.beginPath();
      ctx.arc(this.px[i] * k, this.py[i] * k, r, 0, 6.2832);
      ctx.fill();
    }
    ctx.globalAlpha = 1;

    for (const [index, ring] of [[this.pinned, 2.6], [this.highlight, 2.0]]) {
      if (index < 0 || index >= n) continue;
      ctx.beginPath();
      ctx.arc(this.px[index] * k, this.py[index] * k, r * 2.1, 0, 6.2832);
      ctx.lineWidth = ring * (dpr / 1.4);
      ctx.strokeStyle = style.getPropertyValue("--text").trim() || "#000";
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(this.px[index] * k, this.py[index] * k, r * 2.1, 0, 6.2832);
      ctx.lineWidth = ring * (dpr / 2.6);
      ctx.strokeStyle = style.getPropertyValue("--bg-panel").trim() || "#fff";
      ctx.stroke();
    }
  }

  /* Nearest spot to a client-space point, or -1 when the cursor is off tissue. */
  hitTest(clientX, clientY) {
    if (!this.section) return -1;
    const box = this.canvas.getBoundingClientRect();
    const x = ((clientX - box.left) / box.width) * this.width;
    const y = ((clientY - box.top) / box.height) * this.height;

    const { cell, grid } = this._grid;
    const cx = Math.floor(x / cell);
    const cy = Math.floor(y / cell);
    let best = -1;
    let bestDistance = Infinity;
    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        const bucket = grid.get(`${cx + dx},${cy + dy}`);
        if (!bucket) continue;
        for (const i of bucket) {
          const d = (this.px[i] - x) ** 2 + (this.py[i] - y) ** 2;
          if (d < bestDistance) { bestDistance = d; best = i; }
        }
      }
    }
    const limit = (this.radius * 2.4) ** 2;
    return bestDistance <= limit ? best : -1;
  }
}

/* Position within a sprite sheet, as a CSS background shorthand.
 * Applied inline, so the URL resolves against the document, not the stylesheet. */
export function spriteStyle(layout, index, boxPx) {
  if (!layout) return {};
  const zoom = boxPx / layout.tile;
  return {
    backgroundImage: `url(assets/patches/${layout.section}.jpg)`,
    backgroundSize: `${layout.columns * layout.tile * zoom}px ${layout.rows * layout.tile * zoom}px`,
    backgroundPosition: `-${(index % layout.columns) * layout.tile * zoom}px -${
      Math.floor(index / layout.columns) * layout.tile * zoom}px`,
    imageRendering: zoom > 2 ? "pixelated" : "auto",
  };
}
