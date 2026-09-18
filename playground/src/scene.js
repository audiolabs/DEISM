/*
 * Copyright (c) 2022-2026 Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V.
 * Part of DEISM: a JavaScript port of the DEISM Python package. Subject to the
 * Fraunhofer Software Copyright License (see LICENSE in the package root).
 * Requires a separate license from Fraunhofer beyond internal, non-commercial
 * use for evaluation, testing, and academic research.
 */
/**
 * 3D room view on a 2D canvas: azimuth/elevation camera, room edges,
 * image sources, draggable source/receiver/vertex handles.
 */

export class Scene {
  constructor(canvas, callbacks) {
    this.canvas = canvas;
    this.cb = callbacks; // {onMove(kind, index, xyz), onSelectVertex(i)}
    this.az = -0.7;
    this.el = 0.55;
    this.drag = null;
    this.proj = null;
    this.data = null;
    canvas.addEventListener("pointerdown", (e) => this.onDown(e));
    canvas.addEventListener("pointermove", (e) => this.onMove(e));
    canvas.addEventListener("pointerup", () => (this.drag = null));
    canvas.addEventListener("pointerleave", () => (this.drag = null));
  }

  /** data: {corners, edges, faces:[[idx...]], src, rec, images:[[x,y,z,order]], vertices|null, selVert} */
  set(data) {
    this.data = data;
    this.draw();
  }

  ctx() {
    const r = this.canvas.getBoundingClientRect();
    if (r.width < 2 || r.height < 2) return null;
    const dpr = window.devicePixelRatio || 1;
    if (this.canvas.width !== Math.round(r.width * dpr) || this.canvas.height !== Math.round(r.height * dpr)) {
      this.canvas.width = Math.round(r.width * dpr);
      this.canvas.height = Math.round(r.height * dpr);
    }
    const ctx = this.canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, r.width, r.height);
    return { ctx, w: r.width, h: r.height };
  }

  setProj(w, h) {
    const d = this.data;
    const c = d.center;
    const az = this.az,
      el = this.el;
    const raw = (p) => {
      const dx = p[0] - c[0],
        dy = p[1] - c[1],
        dz = p[2] - c[2];
      const X = dx * Math.cos(az) - dy * Math.sin(az);
      const Y = dx * Math.sin(az) + dy * Math.cos(az);
      return [X, Y * Math.sin(el) + dz * Math.cos(el)];
    };
    let mnX = Infinity,
      mxX = -Infinity,
      mnY = Infinity,
      mxY = -Infinity;
    for (const p of d.corners) {
      const [X, Z] = raw(p);
      mnX = Math.min(mnX, X);
      mxX = Math.max(mxX, X);
      mnY = Math.min(mnY, -Z);
      mxY = Math.max(mxY, -Z);
    }
    const S = Math.min((w * 0.72) / (mxX - mnX || 1), (h * 0.72) / (mxY - mnY || 1));
    this.proj = { S, ox: w / 2 - (S * (mnX + mxX)) / 2, oy: h / 2 - (S * (mnY + mxY)) / 2, az, el, c };
  }

  project(p) {
    const { S, ox, oy, az, el, c } = this.proj;
    const dx = p[0] - c[0],
      dy = p[1] - c[1],
      dz = p[2] - c[2];
    const X = dx * Math.cos(az) - dy * Math.sin(az);
    const Y = dx * Math.sin(az) + dy * Math.cos(az);
    return { sx: ox + S * X, sy: oy - S * (Y * Math.sin(el) + dz * Math.cos(el)), depth: Y * Math.cos(el) - dz * Math.sin(el) };
  }

  unproject(sx, sy, z) {
    const { S, ox, oy, az, el, c } = this.proj;
    const X = (sx - ox) / S,
      Zt = (oy - sy) / S,
      dz = z - c[2];
    const Y = (Zt - dz * Math.cos(el)) / Math.sin(el);
    return [Math.cos(az) * X + Math.sin(az) * Y + c[0], -Math.sin(az) * X + Math.cos(az) * Y + c[1], z];
  }

  draw() {
    const cc = this.ctx();
    if (!cc || !this.data) return;
    const { ctx, w, h } = cc;
    const d = this.data;
    this.setProj(w, h);
    // faces: fill the floor lightly, other faces faintly by depth
    ctx.lineJoin = "round";
    for (const [fi, f] of d.faces.entries()) {
      ctx.beginPath();
      f.forEach((i, k) => {
        const p = this.project(d.corners[i]);
        k ? ctx.lineTo(p.sx, p.sy) : ctx.moveTo(p.sx, p.sy);
      });
      ctx.closePath();
      ctx.fillStyle = fi === d.selectedWall ? "rgba(245,180,80,0.5)" : "rgba(45,91,255,0.045)";
      ctx.fill();
    }
    const deps = d.edges.map(([a, b]) => (this.project(d.corners[a]).depth + this.project(d.corners[b]).depth) / 2);
    const dmn = Math.min(...deps),
      dmx = Math.max(...deps);
    ctx.lineWidth = 1.2;
    d.edges.forEach(([a, b], i) => {
      const pa = this.project(d.corners[a]),
        pb = this.project(d.corners[b]);
      const t = dmx === dmn ? 0.5 : (deps[i] - dmn) / (dmx - dmn);
      ctx.strokeStyle = `rgba(255,255,255,${(0.65 - 0.45 * t).toFixed(3)})`;
      ctx.beginPath();
      ctx.moveTo(pa.sx, pa.sy);
      ctx.lineTo(pb.sx, pb.sy);
      ctx.stroke();
    });
    // image sources
    const srcP = this.project(d.src),
      recP = this.project(d.rec);
    for (const im of d.images) {
      if (im[3] === 0) continue;
      const p = this.project(im);
      ctx.fillStyle = `rgba(120,150,255,${Math.max(0.12, 0.55 - im[3] * 0.12).toFixed(3)})`;
      ctx.beginPath();
      ctx.arc(p.sx, p.sy, Math.max(1.5, 3.5 - im[3] * 0.6), 0, 7);
      ctx.fill();
    }
    // first-order paths
    ctx.strokeStyle = "rgba(255,255,255,0.10)";
    ctx.lineWidth = 1;
    d.images
      .filter((im) => im[3] === 1)
      .slice(0, 8)
      .forEach((im) => {
        const p = this.project(im);
        ctx.beginPath();
        ctx.moveTo(recP.sx, recP.sy);
        ctx.lineTo(p.sx, p.sy);
        ctx.stroke();
      });
    ctx.strokeStyle = "rgba(45,91,255,0.55)";
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1.1;
    ctx.beginPath();
    ctx.moveTo(srcP.sx, srcP.sy);
    ctx.lineTo(recP.sx, recP.sy);
    ctx.stroke();
    ctx.setLineDash([]);
    // vertex handles
    if (d.vertices) {
      d.vertices.forEach((v, i) => {
        const p = this.project(v);
        const sel = i === d.selVert;
        ctx.fillStyle = sel ? "#fff" : "rgba(255,255,255,0.85)";
        ctx.strokeStyle = sel ? "#2D5BFF" : "rgba(20,23,28,0.6)";
        ctx.lineWidth = sel ? 2 : 1;
        ctx.beginPath();
        ctx.rect(p.sx - 4, p.sy - 4, 8, 8);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = "rgba(255,255,255,0.55)";
        ctx.font = '10px "IBM Plex Mono", monospace';
        ctx.fillText(`V${i + 1}`, p.sx + 6, p.sy - 5);
      });
    }
    // facing arrows
    const arrow = (pos, facing, col) => {
      const tip = [pos[0] + 0.5 * facing[0], pos[1] + 0.5 * facing[1], pos[2] + 0.5 * facing[2]];
      const a = this.project(pos),
        b = this.project(tip);
      ctx.strokeStyle = col;
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      ctx.moveTo(a.sx, a.sy);
      ctx.lineTo(b.sx, b.sy);
      ctx.stroke();
      const dx = b.sx - a.sx, dy = b.sy - a.sy;
      const length = Math.hypot(dx, dy);
      if (length > 2) {
        const ux = dx / length, uy = dy / length;
        const head = Math.min(10, length * 0.4);
        ctx.fillStyle = col;
        ctx.beginPath();
        ctx.moveTo(b.sx, b.sy);
        ctx.lineTo(b.sx - head * ux + head * 0.5 * uy, b.sy - head * uy - head * 0.5 * ux);
        ctx.lineTo(b.sx - head * ux - head * 0.5 * uy, b.sy - head * uy + head * 0.5 * ux);
        ctx.closePath();
        ctx.fill();
      }
    };
    if (d.srcFacing) arrow(d.src, d.srcFacing, "rgba(157,180,255,0.8)");
    if (d.recFacing) arrow(d.rec, d.recFacing, "rgba(255,179,160,0.8)");
    const dot = (p, col) => {
      ctx.strokeStyle = col;
      ctx.globalAlpha = 0.5;
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.arc(p.sx, p.sy, 11, 0, 7);
      ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.fillStyle = col;
      ctx.beginPath();
      ctx.arc(p.sx, p.sy, 6.5, 0, 7);
      ctx.fill();
    };
    dot(srcP, "#2D5BFF");
    dot(recP, "#FF6A45");
  }

  pointer(e) {
    const r = this.canvas.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  }

  onDown(e) {
    if (!this.proj || !this.data) return;
    this.canvas.setPointerCapture(e.pointerId);
    const [mx, my] = this.pointer(e);
    const near = (p, r) => Math.hypot(p.sx - mx, p.sy - my) < r;
    const d = this.data;
    if (d.vertices) {
      for (let i = 0; i < d.vertices.length; i++) {
        if (near(this.project(d.vertices[i]), 10)) {
          this.drag = { kind: "vert", i };
          this.cb.onSelectVertex(i);
          return;
        }
      }
    }
    if (near(this.project(d.src), 15)) this.drag = { kind: "src" };
    else if (near(this.project(d.rec), 15)) this.drag = { kind: "rec" };
    else this.drag = { kind: "orbit", x: mx, y: my, az: this.az, el: this.el };
  }

  onMove(e) {
    if (!this.drag || !this.proj) return;
    const [mx, my] = this.pointer(e);
    const dr = this.drag;
    if (dr.kind === "orbit") {
      this.az = dr.az + (mx - dr.x) * 0.01;
      this.el = Math.min(1.4, Math.max(0.1, dr.el + (my - dr.y) * 0.005));
      this.draw();
      return;
    }
    const d = this.data;
    if (dr.kind === "vert") {
      const v = d.vertices[dr.i];
      const w = this.unproject(mx, my, v[2]);
      this.cb.onMove("vert", dr.i, [w[0], w[1], v[2]]);
      return;
    }
    const cur = dr.kind === "src" ? d.src : d.rec;
    const w = this.unproject(mx, my, cur[2]);
    this.cb.onMove(dr.kind, 0, [w[0], w[1], cur[2]]);
  }
}
