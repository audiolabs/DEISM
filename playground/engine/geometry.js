/*
 * Copyright (c) 2022-2026 Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V.
 * Part of DEISM: a JavaScript port of the DEISM Python package. Subject to the
 * Fraunhofer Software Copyright License (see LICENSE in the package root).
 * Requires a separate license from Fraunhofer beyond internal, non-commercial
 * use for evaluation, testing, and academic research.
 */
/*
 * The convex-room geometry helpers below (wall primitives, segment/plane
 * intersections, image-source visibility) originated from the libroom core of
 * https://github.com/LCAV/pyroomacoustics and were ported to JavaScript and
 * modified by Fraunhofer. The original code was obtained under the MIT
 * license, reproduced here as it requires:
 *
 * Copyright (C) 2019  Robin Scheibler, Cyril Cadoux
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
/**
 * Geometry: coordinate conversions, rotations, and the convex-room image
 * source model ported from deism/core_deism_arg.py (Room_deism_python, the
 * reference producer of the compact ARG geometry) and the libroom C++ wall
 * primitives it mirrors.
 */

import {
  cross3,
  dot3,
  sub3,
  add3,
  scale3,
  norm3,
  normalize3,
  mat3mul,
  identity3,
} from "./linalg.js";

export const LIBROOM_EPS = 1e-5;

/** cart2sph as in deism.utilities: returns [az, el, r] with el the elevation. */
export function cart2sph(x, y, z) {
  const hxy = Math.hypot(x, y);
  return [Math.atan2(y, x), Math.atan2(z, hxy), Math.hypot(hxy, z)];
}

export function sph2cart(az, el, r) {
  const rc = r * Math.cos(el);
  return [rc * Math.cos(az), rc * Math.sin(az), r * Math.sin(el)];
}

/** [azimuth, inclination, radius] of the vector v, the DEISM kernel convention. */
export function toKernelSph(v) {
  const [az, el, r] = cart2sph(v[0], v[1], v[2]);
  return [az, Math.PI / 2 - el, r];
}

/** Z-X-Z Euler rotation matrix as used by COMSOL and deism.shared_utils. */
export function rotationMatrixZXZ(alpha, beta, gamma) {
  const ca = Math.cos(alpha),
    sa = Math.sin(alpha),
    cb = Math.cos(beta),
    sb = Math.sin(beta),
    cg = Math.cos(gamma),
    sg = Math.sin(gamma);
  return [
    [ca * cg - sa * cb * sg, -ca * sg - sa * cb * cg, sb * sa],
    [sa * cg + ca * cb * sg, -sa * sg + ca * cb * cg, -sb * ca],
    [sb * sg, sb * cg, cb],
  ];
}

/** Compose room and transducer rotations; geometry is already in world coordinates. */
export function directivityRotation(orientDeg, roomDeg = null) {
  const R = rotationMatrixZXZ(...orientDeg.map((v) => v * Math.PI / 180));
  return roomDeg ? mat3mul(rotationMatrixZXZ(...roomDeg.map((v) => v * Math.PI / 180)), R) : R;
}

// ---------------------------------------------------------------------------
// 2D polygon helpers (libroom geometry.cpp)
// ---------------------------------------------------------------------------

function ccw3p(p1, p2, p3) {
  const d = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]);
  if (Math.abs(d) < LIBROOM_EPS) return 0;
  return d > 0 ? 1 : -1;
}

/**
 * Collinear point within the segment's bounding box, with the libroom
 * tolerance. libroom compares strictly, in single precision; a ray through
 * a wall edge (a common degenerate case in rooms with parallel walls) then
 * lands on or off the segment by rounding luck. The tolerance keeps such
 * points on the boundary, where the solver counts them as visible.
 */
function onSegment(c1, c2, p) {
  const xd = Math.min(c1[0], c2[0]) - LIBROOM_EPS,
    xu = Math.max(c1[0], c2[0]) + LIBROOM_EPS,
    yd = Math.min(c1[1], c2[1]) - LIBROOM_EPS,
    yu = Math.max(c1[1], c2[1]) + LIBROOM_EPS;
  return xd <= p[0] && p[0] <= xu && yd <= p[1] && p[1] <= yu;
}

function isLeft(p0, p1, p2) {
  const t = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1]);
  if (Math.abs(t) < LIBROOM_EPS) return 0;
  return t > 0 ? 1 : -1;
}

/** Winding-number point-in-polygon test: -1 outside, 0 inside, 1 on boundary. */
export function isInside2dPolygon(p, corners) {
  const n = corners.length;
  let wn = 0;
  for (let i = 0, j = n - 1; i < n; j = i++) {
    const cj = corners[j],
      ci = corners[i];
    if (ccw3p(cj, ci, p) === 0 && onSegment(cj, ci, p)) return 1;
    if (cj[1] <= p[1]) {
      if (ci[1] > p[1] && isLeft(cj, ci, p) > 0) wn++;
    } else if (ci[1] <= p[1] && isLeft(cj, ci, p) < 0) wn--;
  }
  return wn === 0 ? -1 : 0;
}

function area2dPolygon(corners) {
  let area = 0;
  for (let c1 = 0; c1 < corners.length; c1++) {
    const c2 = c1 === corners.length - 1 ? 0 : c1 + 1;
    const base = 0.5 * (corners[c2][1] + corners[c1][1]);
    const height = corners[c2][0] - corners[c1][0];
    area -= height * base;
  }
  return area;
}

function intersection3dSegmentPlane(a1, a2, p, normal) {
  const u = sub3(a2, a1);
  const denom = dot3(normal, u);
  if (Math.abs(denom) > LIBROOM_EPS) {
    const w = sub3(a1, p);
    const s = -dot3(normal, w) / denom;
    if (-LIBROOM_EPS <= s && s <= 1 + LIBROOM_EPS) {
      const pt = add3(scale3(u, s), a1);
      if (Math.abs(s) < LIBROOM_EPS || Math.abs(s - 1) < LIBROOM_EPS) return [1, pt];
      return [0, pt];
    }
  }
  return [-1, null];
}

// ---------------------------------------------------------------------------
// Wall
// ---------------------------------------------------------------------------

export class Wall {
  /**
   * @param {number[][]} points  polygon corners (unordered)
   * @param {number[]} centroid  room centroid; the normal is oriented away from it
   * @param {number} materialIndex row of the impedance table for this wall
   */
  constructor(points, centroid, materialIndex) {
    let normal = normalize3(cross3(sub3(points[1], points[0]), sub3(points[2], points[0])));
    this.origin = points[0].slice();
    if (dot3(normal, sub3(centroid, points[0])) > 0) normal = scale3(normal, -1);
    this.normal = normal;
    this.points = this.orderPoints(points);
    this.materialIndex = materialIndex;
    // In-plane orthonormal basis (any basis works for the 2D tests)
    const e1 = normalize3(sub3(this.points[1], this.points[0]));
    const e2 = cross3(normal, e1);
    let basis = [e1, e2];
    let flat = this.points.map((p) => {
      const d = sub3(p, this.origin);
      return [dot3(basis[0], d), dot3(basis[1], d)];
    });
    if (area2dPolygon(flat) < 0) {
      basis = [e2, e1];
      flat = flat.map((f) => [f[1], f[0]]);
    }
    this.basis = basis;
    this.flatCorners = flat;
    const n = normal;
    this.reflectionMatrix = [
      [1 - 2 * n[0] * n[0], -2 * n[0] * n[1], -2 * n[0] * n[2]],
      [-2 * n[1] * n[0], 1 - 2 * n[1] * n[1], -2 * n[1] * n[2]],
      [-2 * n[2] * n[0], -2 * n[2] * n[1], 1 - 2 * n[2] * n[2]],
    ];
    this.center = this.points.reduce((a, p) => add3(a, p), [0, 0, 0]).map((v) => v / this.points.length);
    // Polygon area from the ordered corners (shoelace in the plane)
    this.area = Math.abs(area2dPolygon(flat));
  }

  orderPoints(points) {
    const c = points.reduce((a, p) => add3(a, p), [0, 0, 0]).map((v) => v / points.length);
    const ref = sub3(points[0], c);
    const angles = points.map((p) => {
      const d = sub3(p, c);
      const cr = cross3(d, ref);
      let ang = Math.atan2(norm3(cr), dot3(d, ref));
      if (dot3(cr, this.normal) <= 0) ang = 2 * Math.PI - ang;
      return ang;
    });
    const idx = points.map((_, i) => i).sort((a, b) => angles[a] - angles[b]);
    return idx.map((i) => points[i].slice());
  }

  /** Reflect a point; flag 1 (in front), -1 (behind), 0 (on the wall). */
  reflect(point) {
    const d = dot3(this.normal, sub3(this.origin, point));
    const r = add3(point, scale3(this.normal, 2 * d));
    if (d > LIBROOM_EPS) return [r, 1];
    if (d < -LIBROOM_EPS) return [r, -1];
    return [r, 0];
  }

  /** Segment/polygon intersection: [-1 | flags, point]. */
  intersection(p1, p2) {
    const [ret1, pt] = intersection3dSegmentPlane(p1, p2, this.origin, this.normal);
    if (ret1 === -1) return [-1, null];
    let ret = ret1 === 1 ? 1 : 0;
    const d = sub3(pt, this.origin);
    const flat = [dot3(this.basis[0], d), dot3(this.basis[1], d)];
    const ret2 = isInside2dPolygon(flat, this.flatCorners);
    if (ret2 < 0) return [-1, pt];
    if (ret2 === 1) ret |= 2;
    return [ret, pt];
  }
}

// ---------------------------------------------------------------------------
// Convex hull faces (replaces scipy.spatial.ConvexHull for small vertex sets)
// ---------------------------------------------------------------------------

function roundKey(n) {
  return n.map((v) => {
    const r = Math.round(v * 1e5) / 1e5;
    return r === 0 ? 0 : r; // drop negative zero
  });
}

/**
 * Faces of the convex hull of `vertices`, grouped by outward normal and
 * sorted lexicographically by the rounded normal, exactly like
 * find_wall_centers / generate_walls_convex. Each face: {normal, points}.
 */
export function convexHullFaces(vertices) {
  const n = vertices.length;
  if (n < 4) throw new Error("A room needs at least 4 vertices");
  const centroid = vertices.reduce((a, p) => add3(a, p), [0, 0, 0]).map((v) => v / n);
  const faces = new Map();
  const tol = 1e-7;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      for (let k = j + 1; k < n; k++) {
        let normal = cross3(sub3(vertices[j], vertices[i]), sub3(vertices[k], vertices[i]));
        const len = norm3(normal);
        if (len < 1e-12) continue;
        normal = scale3(normal, 1 / len);
        let pos = 0,
          neg = 0;
        for (let q = 0; q < n; q++) {
          if (q === i || q === j || q === k) continue;
          const d = dot3(normal, sub3(vertices[q], vertices[i]));
          if (d > tol) pos++;
          else if (d < -tol) neg++;
        }
        if (pos > 0 && neg > 0) continue; // not a supporting plane
        if (pos > 0) normal = scale3(normal, -1); // make outward
        if (dot3(normal, sub3(vertices[i], centroid)) < 0) normal = scale3(normal, -1);
        const key = roundKey(normal).join(",");
        if (!faces.has(key)) faces.set(key, { normal: roundKey(normal), points: new Map() });
        const f = faces.get(key);
        for (const q of [i, j, k]) f.points.set(q, vertices[q]);
      }
    }
  }
  const out = [...faces.values()].map((f) => ({
    normal: f.normal,
    points: [...f.points.keys()].sort((a, b) => a - b).map((q) => vertices[q].slice()),
  }));
  out.sort((a, b) => {
    for (let d = 0; d < 3; d++) if (a.normal[d] !== b.normal[d]) return a.normal[d] - b.normal[d];
    return 0;
  });
  if (out.length < 4) throw new Error("Vertices do not form a closed convex polyhedron");
  return out;
}

/** Wall centers in hull-face order (find_wall_centers). */
export function findWallCenters(vertices) {
  return convexHullFaces(vertices).map((f) =>
    f.points.reduce((a, p) => add3(a, p), [0, 0, 0]).map((v) => v / f.points.length),
  );
}

/** Volume and per-face areas in hull-face order (convex_room_volume_and_areas). */
export function convexRoomVolumeAndAreas(vertices) {
  const faces = convexHullFaces(vertices);
  const centroid = vertices.reduce((a, p) => add3(a, p), [0, 0, 0]).map((v) => v / vertices.length);
  let volume = 0;
  const areas = faces.map((f) => {
    const w = new Wall(f.points, centroid, 0);
    volume += (w.area * Math.abs(dot3(w.normal, sub3(f.points[0], centroid)))) / 3;
    return w.area;
  });
  return { volume, areas };
}

// ---------------------------------------------------------------------------
// Convex room image-source model (Room_deism_python)
// ---------------------------------------------------------------------------

class ImageSource {
  constructor(loc) {
    this.loc = loc;
    this.order = 0;
    this.genWall = -1;
    this.parent = null;
    this.wallSequence = [];
    this.incidenceCos = [];
  }

  /** Accumulated reflection matrix W_k ... W_1 (computed only for visible images). */
  accumulateReflection(walls) {
    let R = identity3();
    for (let node = this; node.parent !== null; node = node.parent) R = mat3mul(R, walls[node.genWall].reflectionMatrix);
    return R;
  }
}

export class ConvexRoom {
  /**
   * @param {number[][]} vertices  convex room vertices
   * @param {number[][]|null} wallCenters material row centers; default hull order
   */
  constructor(vertices, wallCenters = null) {
    this.vertices = vertices.map((v) => v.slice());
    this.centroid = vertices.reduce((a, p) => add3(a, p), [0, 0, 0]).map((v) => v / vertices.length);
    const faces = convexHullFaces(vertices);
    const centers = wallCenters || faces.map((f) => f.points.reduce((a, p) => add3(a, p), [0, 0, 0]).map((v) => v / f.points.length));
    this.walls = faces.map((f) => {
      const fc = f.points.reduce((a, p) => add3(a, p), [0, 0, 0]).map((v) => v / f.points.length);
      let best = 0,
        bestD = Infinity;
      centers.forEach((c, i) => {
        const d = norm3(sub3(c, fc));
        if (d < bestD) {
          bestD = d;
          best = i;
        }
      });
      if (bestD > 0.001) throw new Error("The face center is not close enough to any wall center");
      return new Wall(f.points, this.centroid, best);
    });
    const va = convexRoomVolumeAndAreas(vertices);
    this.volume = va.volume;
    this.areas = va.areas;
  }

  /** True when p is strictly inside the room (all walls behind their normals). */
  contains(p, margin = 0) {
    return this.walls.every((w) => dot3(w.normal, sub3(p, w.origin)) < -margin);
  }

  /**
   * Run the image-source DFS. Returns {sources, orders, reflectionMatrix,
   * wallSequence, incidenceCos} with one entry per visible image, in the
   * solver's push order.
   */
  imageSources(source, receiver, maxOrder) {
    this.source = source;
    this.receiver = receiver;
    this.visible = [];
    this.maxOrder = maxOrder;
    this.dfs(new ImageSource(source.slice()), maxOrder);
    const out = {
      count: this.visible.length,
      sources: this.visible.map((s) => s.loc),
      orders: this.visible.map((s) => s.order),
      reflectionMatrix: this.visible.map((s) => s.reflectMatrix),
      wallSequence: this.visible.map((s) => {
        const row = new Array(maxOrder).fill(-1);
        s.wallSequence.forEach((w, i) => (row[i] = w));
        return row;
      }),
      incidenceCos: this.visible.map((s) => {
        const row = new Array(maxOrder).fill(NaN);
        s.incidenceCos.forEach((c, i) => (row[i] = c));
        return row;
      }),
    };
    return out;
  }

  dfs(is, maxOrder) {
    const segs = [];
    if (this.isVisibleDfs(this.receiver, is, segs)) {
      const [ws, ic] = this.compactPath(is, segs);
      is.wallSequence = ws;
      is.incidenceCos = ic;
      is.reflectMatrix = is.accumulateReflection(this.walls);
      this.visible.push(is);
    }
    if (maxOrder === 0) return;
    const walls = this.walls;
    for (let wi = 0; wi < walls.length; wi++) {
      const [r, flag] = walls[wi].reflect(is.loc);
      if (flag <= 0) continue;
      const child = new ImageSource(r);
      child.order = is.order + 1;
      child.genWall = wi;
      child.parent = is;
      this.dfs(child, maxOrder - 1);
    }
  }

  /**
   * Walk from p back through the generating walls of `is` and its parents
   * (libroom is_visible_dfs), collecting the segments from each intersection
   * point to its image in `segs`. Iterative: the recursion of the reference
   * only ever continues at the tail.
   */
  isVisibleDfs(p, is, segs) {
    let node = is;
    while (node.parent !== null) {
      const [ret, pt] = this.walls[node.genWall].intersection(p, node.loc);
      if (ret < 0) return false;
      segs.push(sub3(node.loc, pt));
      p = pt;
      node = node.parent;
    }
    return true;
  }

  compactPath(is, segs) {
    const ws = [],
      ic = [];
    let node = is;
    for (const seg of segs) {
      const wallId = node.genWall;
      if (wallId < 0) break;
      const wall = this.walls[wallId];
      const nrm = norm3(seg);
      if (nrm <= LIBROOM_EPS) throw new Error("zero-length reflection segment in compact ARG path");
      ws.push(wall.materialIndex);
      ic.push(Math.min(1, Math.abs(dot3(scale3(seg, 1 / nrm), wall.normal))));
      node = node.parent;
    }
    return [ws, ic];
  }
}
