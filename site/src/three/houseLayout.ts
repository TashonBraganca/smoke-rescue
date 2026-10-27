// Pure data for the 3D house: rooms, walls, furniture, doors and the dogs' routes, all in scene units.
// Plan coordinates are metres from the south-west corner, rect = [x0, z0, x1, z1]. The scene puts the
// footprint centre at the origin and maps plan z to -Z, so "north" points away from a default camera.
import * as THREE from "three";
import type { Plan } from "@/lib/plans";

export const WALL_H = 1.2; // a cut-away: low walls so the camera can see into every room
const FRONT = 0.85; // door approach points sit this far either side of a door, so the dog is straight before it turns

export type RoomInfo = {
  id: string; name: string; kind: string; fire: boolean;
  x0: number; x1: number; z0: number; z1: number; // scene rect, min/max
  centre: THREE.Vector3;
};
export type DoorInfo = { id: string; a: string; b: string; p: THREE.Vector3; n: THREE.Vector3; width: number };
export type WallSeg = { x: number; z: number; lx: number; lz: number };
export type FurnitureSpot = { kind: string; x: number; z: number; ry: number };
export type Layout = {
  rooms: RoomInfo[]; byId: Record<string, RoomInfo>; walls: WallSeg[]; furniture: FurnitureSpot[];
  fires: THREE.Vector3[]; doors: DoorInfo[];
  entry: { room: string; door: THREE.Vector3; out: THREE.Vector3; side: THREE.Vector3 };
  w: number; d: number;
};

export function buildLayout(p: Plan): Layout {
  const fp = p.footprint, cx = (fp[0] + fp[2]) / 2, cz = (fp[1] + fp[3]) / 2;
  const P = (x: number, z: number) => new THREE.Vector3(x - cx, 0, -(z - cz));
  const ids = new Set(p.rooms.map((r) => r.id)), fireRooms = new Set(p.fires.map((f) => f.room));
  const rooms: RoomInfo[] = p.rooms.map((r) => {
    const [x0, z0, x1, z1] = r.rect;
    return { id: r.id, name: r.name, kind: r.kind, fire: fireRooms.has(r.id), x0: x0 - cx, x1: x1 - cx, z0: -(z1 - cz), z1: -(z0 - cz), centre: P((x0 + x1) / 2, (z0 + z1) / 2) };
  });
  const byId = Object.fromEntries(rooms.map((r) => [r.id, r]));

  // walls: one wall per shared line (no doubled walls between neighbours), door gaps cut out, exterior walls thicker
  const walls: WallSeg[] = [];
  type Line = { ax: "x" | "z"; at: number; spans: [number, number][] };
  const lines: Line[] = [];
  const addEdge = (ax: "x" | "z", at: number, a0: number, a1: number) => {
    let l = lines.find((q) => q.ax === ax && Math.abs(q.at - at) < 0.04);
    if (!l) { l = { ax, at, spans: [] }; lines.push(l); }
    l.spans.push([a0, a1]);
  };
  for (const r of p.rooms) { const [x0, z0, x1, z1] = r.rect; addEdge("z", z0, x0, x1); addEdge("z", z1, x0, x1); addEdge("x", x0, z0, z1); addEdge("x", x1, z0, z1); }
  const TI = 0.13, TE = 0.22;
  for (const l of lines) {
    const ext = l.ax === "z" ? Math.abs(l.at - fp[1]) < 0.04 || Math.abs(l.at - fp[3]) < 0.04 : Math.abs(l.at - fp[0]) < 0.04 || Math.abs(l.at - fp[2]) < 0.04;
    const T = ext ? TE : TI;
    // centre exterior walls just inside the footprint so the outer face is flush with it
    const lo = l.ax === "z" ? fp[1] : fp[0];
    const centre = ext ? l.at + (Math.abs(l.at - lo) < 0.04 ? T / 2 : -T / 2) : l.at;
    // union of spans
    const sp = l.spans.slice().sort((a, b) => a[0] - b[0]), merged: [number, number][] = [];
    for (const [a, b] of sp) { const m = merged[merged.length - 1]; if (m && a <= m[1] + 0.05) m[1] = Math.max(m[1], b); else merged.push([a, b]); }
    const gaps = p.doors
      .filter((d) => d.wall === l.ax && Math.abs((l.ax === "z" ? d.at[1] : d.at[0]) - l.at) < 0.3)
      .map((d) => { const u = l.ax === "z" ? d.at[0] : d.at[1]; return [u - d.width / 2, u + d.width / 2] as [number, number]; })
      .sort((a, b) => a[0] - b[0]);
    for (const [m0, m1] of merged) {
      const lo2 = l.ax === "z" ? fp[0] : fp[1], hi2 = l.ax === "z" ? fp[2] : fp[3];
      let s = Math.max(lo2, m0 - TE / 2); const end = Math.min(hi2, m1 + TE / 2); const segs: [number, number][] = [];
      for (const [g0, g1] of gaps) { if (g1 <= s || g0 >= end) continue; if (g0 > s) segs.push([s, g0]); s = Math.max(s, g1); }
      if (end > s) segs.push([s, end]);
      for (const [u0, u1] of segs) {
        if (u1 - u0 < 0.05) continue;
        const um = (u0 + u1) / 2, L = u1 - u0;
        if (l.ax === "z") { const q = P(um, centre); walls.push({ x: q.x, z: q.z, lx: L, lz: T }); }
        else { const q = P(centre, um); walls.push({ x: q.x, z: q.z, lx: T, lz: L }); }
      }
    }
  }
  const furniture = p.landmarks.map((l) => { const q = P(l.at[0], l.at[1]); return { kind: l.kind, x: q.x, z: q.z, ry: -((l.yaw ?? 0) * Math.PI) / 180 }; });
  const fires = p.fires.map((f) => P(f.at[0], f.at[1]));

  const doors: DoorInfo[] = [];
  let entry: Layout["entry"] | null = null;
  for (const d of p.doors) {
    const q = P(d.at[0], d.at[1]);
    const n = d.wall === "z" ? new THREE.Vector3(0, 0, 1) : new THREE.Vector3(1, 0, 0);
    const [a, b] = d.between;
    if (ids.has(a) && ids.has(b)) doors.push({ id: d.id, a, b, p: q, n, width: d.width });
    else if (!entry) {
      const room = ids.has(a) ? a : b, c = byId[room].centre;
      const out = n.clone(); if (out.dot(q.clone().sub(c)) < 0) out.negate();
      entry = { room, door: q, out, side: new THREE.Vector3(-out.z, 0, out.x) };
    }
  }
  if (!entry) { const r = rooms[0]; entry = { room: r.id, door: r.centre.clone(), out: new THREE.Vector3(0, 0, 1), side: new THREE.Vector3(1, 0, 0) }; }
  return { rooms, byId, walls, furniture, fires, doors, entry, w: fp[2] - fp[0], d: fp[3] - fp[1] };
}

const inside = (r: RoomInfo, v: THREE.Vector3, pad = 0) => v.x > r.x0 + pad && v.x < r.x1 - pad && v.z > r.z0 + pad && v.z < r.z1 - pad;
export function roomAt(L: Layout, v: THREE.Vector3, pad = 0.12) { for (const r of L.rooms) if (inside(r, v, pad)) return r; return null; }

/** The approach point on room `room`'s side of a door. */
function front(L: Layout, d: DoorInfo, room: string) {
  const a = d.p.clone().addScaledVector(d.n, FRONT);
  return inside(L.byId[room], a) ? a : d.p.clone().addScaledVector(d.n, -FRONT);
}

/** Shortest walk from a point in room `from` to the centre of room `to`, through doors (Dijkstra on a tiny graph). */
function walk(L: Layout, fromRoom: string, fromPt: THREE.Vector3, to: string): { pts: THREE.Vector3[]; len: number } {
  // nodes: door ids; start and goal are virtual
  const dist = new Map<string, number>(), prev = new Map<string, { node: string; room: string }>();
  const doorsOf = (room: string) => L.doors.filter((d) => d.a === room || d.b === room);
  type Q = { node: string; room: string; pos: THREE.Vector3; cost: number };
  const queue: Q[] = [{ node: "start", room: fromRoom, pos: fromPt, cost: 0 }];
  dist.set("start", 0);
  let goal: { cost: number; via: string; room: string } | null = null;
  const done = new Set<string>();
  while (queue.length) {
    queue.sort((a, b) => a.cost - b.cost);
    const cur = queue.shift()!;
    if (done.has(cur.node + "@" + cur.room)) continue;
    done.add(cur.node + "@" + cur.room);
    if (cur.room === to) {
      const c = cur.cost + cur.pos.distanceTo(L.byId[to].centre);
      if (!goal || c < goal.cost) goal = { cost: c, via: cur.node, room: cur.room };
    }
    for (const d of doorsOf(cur.room)) {
      if (d.id === cur.node) continue;
      const other = d.a === cur.room ? d.b : d.a, c = cur.cost + cur.pos.distanceTo(d.p);
      const key = d.id + "@" + other;
      if (c < (dist.get(key) ?? Infinity)) { dist.set(key, c); prev.set(key, { node: cur.node, room: cur.room }); queue.push({ node: d.id, room: other, pos: d.p, cost: c }); }
    }
  }
  if (!goal) return { pts: [L.byId[to].centre.clone()], len: fromPt.distanceTo(L.byId[to].centre) };
  // unwind the chain of doors
  const chain: { door: string; into: string }[] = [];
  let k = goal.via + "@" + goal.room;
  while (k.split("@")[0] !== "start") {
    const [door, into] = k.split("@");
    chain.unshift({ door, into });
    const pv = prev.get(k)!; k = pv.node + "@" + pv.room;
  }
  const pts: THREE.Vector3[] = [];
  let room = fromRoom;
  for (const { door, into } of chain) {
    const d = L.doors.find((x) => x.id === door)!;
    pts.push(front(L, d, room), d.p.clone(), front(L, d, into));
    room = into;
  }
  pts.push(L.byId[to].centre.clone());
  return { pts, len: goal.cost };
}

export type Route = {
  len: number; // metres
  pos: Float32Array; // x,z every STEP metres
  dir: Float32Array; // unit heading x,z
  cam: Float32Array; // low-passed heading for a follow camera
  marks: { room: string; s: number }[]; // where the dog first steps into each room
};
export const STEP = 0.05;

/** Sample a smooth curve through waypoints by arc length, so the dog can be placed at any distance with no allocation. */
function sample(L: Layout, wp: THREE.Vector3[]): Route {
  const pts: THREE.Vector3[] = [];
  for (const v of wp) if (!pts.length || pts[pts.length - 1].distanceTo(v) > 0.08) pts.push(v);
  const curve = new THREE.CatmullRomCurve3(pts, false, "centripetal");
  let rough = 0; for (let i = 1; i < pts.length; i++) rough += pts[i].distanceTo(pts[i - 1]);
  curve.arcLengthDivisions = Math.max(200, Math.ceil(rough * 30));
  const len = curve.getLength(), n = Math.max(2, Math.ceil(len / STEP) + 1);
  const pos = new Float32Array(n * 2), dir = new Float32Array(n * 2), cam = new Float32Array(n * 2);
  const v = new THREE.Vector3(), t = new THREE.Vector3();
  const marks: Route["marks"] = []; const seen = new Set<string>();
  let cx = 0, cz = 0; const a = Math.exp(-STEP / 2.2);
  for (let i = 0; i < n; i++) {
    const u = Math.min(1, (i * STEP) / len);
    curve.getPointAt(u, v); curve.getTangentAt(u, t);
    const tl = Math.hypot(t.x, t.z) || 1;
    pos[i * 2] = v.x; pos[i * 2 + 1] = v.z; dir[i * 2] = t.x / tl; dir[i * 2 + 1] = t.z / tl;
    if (i === 0) { cx = dir[0]; cz = dir[1]; } else { cx = cx * a + dir[i * 2] * (1 - a); cz = cz * a + dir[i * 2 + 1] * (1 - a); }
    const cl = Math.hypot(cx, cz) || 1; cam[i * 2] = cx / cl; cam[i * 2 + 1] = cz / cl;
    const r = roomAt(L, v);
    if (r && !seen.has(r.id)) { seen.add(r.id); marks.push({ room: r.id, s: i * STEP }); }
  }
  return { len, pos, dir, cam, marks };
}

/** Read a route at distance s (clamped). Writes into out = [x, z, dirX, dirZ, camX, camZ]. */
export function at(r: Route, s: number, out: Float32Array) {
  const n = r.pos.length / 2, f = Math.min(n - 1, Math.max(0, s / STEP)), i = Math.min(n - 2, Math.floor(f)), k = f - i;
  for (const [src, o] of [[r.pos, 0], [r.dir, 2], [r.cam, 4]] as const) {
    out[o] = src[i * 2] + (src[i * 2 + 2] - src[i * 2]) * k;
    out[o + 1] = src[i * 2 + 1] + (src[i * 2 + 3] - src[i * 2 + 1]) * k;
  }
  // extrapolate backwards before the start, so a follow camera can sit behind a dog that has not moved yet
  if (s < 0) { out[0] += r.dir[0] * s; out[1] += r.dir[1] * s; }
  return out;
}

/**
 * Plan a sweep for `n` dogs entering through the front door. Each step, the dog that has walked least picks the
 * nearest room nobody has entered yet; rooms it passes through count as entered. Dog k starts k * LAG metres later.
 */
export const LAG = 6; // metres the second dog waits before it starts walking
export function planSweep(L: Layout, n: number): Route[] {
  const e = L.entry, entryRoom = e.room;
  const outside = e.door.clone().addScaledVector(e.out, FRONT), inFront = e.door.clone().addScaledVector(e.out, -FRONT);
  const dogs = Array.from({ length: n }, (_, k) => {
    const start = e.door.clone().addScaledVector(e.out, 3.2 - k * 0.7).addScaledVector(e.side, k ? 1.25 : 0);
    const lead = e.door.clone().addScaledVector(e.out, 1.7).addScaledVector(e.side, k ? 0.35 : 0);
    return { wp: [start, lead, outside.clone(), e.door.clone(), inFront.clone()], room: entryRoom, pos: inFront.clone(), acc: k * LAG };
  });
  const entered = new Set<string>([entryRoom]);
  const todo = () => L.rooms.filter((r) => !entered.has(r.id));
  let guard = 0;
  while (todo().length && guard++ < 64) {
    const d = dogs.reduce((a, b) => (b.acc < a.acc ? b : a));
    let best: { room: string; w: ReturnType<typeof walk> } | null = null;
    for (const r of todo()) { const w = walk(L, d.room, d.pos, r.id); if (!best || w.len < best.w.len) best = { room: r.id, w }; }
    if (!best) break;
    // a short swirl at the room centre so the dog turns round smoothly instead of spinning on the spot
    const pts = best.w.pts, c = pts[pts.length - 1], prev = pts.length > 1 ? pts[pts.length - 2] : d.pos;
    const din = c.clone().sub(prev).setY(0); if (din.lengthSq() < 1e-6) din.set(1, 0, 0); din.normalize();
    const q = new THREE.Vector3(-din.z, 0, din.x);
    const r = L.byId[best.room], room = Math.min(r.x1 - r.x0, r.z1 - r.z0), rad = Math.min(0.55, room * 0.16);
    pts.splice(pts.length - 1, 1, c.clone().addScaledVector(din, rad * 0.6).addScaledVector(q, rad), c.clone().addScaledVector(din, -rad * 0.2).addScaledVector(q, -rad));
    // mark every room the walk passes through
    let from = d.room;
    for (const v of pts) { const rr = roomAt(L, v, 0.02); if (rr && rr.id !== from) { entered.add(rr.id); from = rr.id; } }
    entered.add(best.room);
    d.wp.push(...pts); d.acc += best.w.len + 1.5; d.room = best.room; d.pos = pts[pts.length - 1].clone();
  }
  return dogs.map((d) => sample(L, d.wp));
}

/** Camera placement that fits the whole house for a given aspect, trying 24 directions and preferring the door side. */
export function fitOverview(L: Layout, aspect: number, fovDeg: number, pitchDeg: number, margin = 1.12) {
  const tanV = Math.tan((fovDeg * Math.PI) / 360), tanH = tanV * aspect, pitch = (pitchDeg * Math.PI) / 180;
  const hw = L.w / 2 + 0.3, hd = L.d / 2 + 0.3;
  const corners: THREE.Vector3[] = [];
  for (const x of [-hw, hw]) for (const z of [-hd, hd]) for (const y of [0, WALL_H]) corners.push(new THREE.Vector3(x, y, z));
  let best: { d: number; score: number; dir: THREE.Vector3 } | null = null;
  const fwd = new THREE.Vector3(), right = new THREE.Vector3(), up = new THREE.Vector3(), Y = new THREE.Vector3(0, 1, 0);
  for (let i = 0; i < 24; i++) {
    const az = (i / 24) * Math.PI * 2;
    // direction from target to camera
    const dir = new THREE.Vector3(Math.sin(az) * Math.cos(pitch), Math.sin(pitch), Math.cos(az) * Math.cos(pitch));
    fwd.copy(dir).negate(); right.crossVectors(fwd, Y).normalize(); up.crossVectors(right, fwd);
    let d = 0;
    for (const c of corners) {
      const depth = c.dot(fwd);
      d = Math.max(d, (Math.abs(c.dot(right)) * margin) / tanH - depth, (Math.abs(c.dot(up)) * margin) / tanV - depth);
    }
    const door = Math.max(0, new THREE.Vector3(dir.x, 0, dir.z).normalize().dot(L.entry.out));
    const score = d * (1.12 - 0.12 * door);
    if (!best || score < best.score) best = { d, score, dir };
  }
  const target = new THREE.Vector3(0, 0, 0);
  return { pos: best!.dir.clone().multiplyScalar(best!.d).add(target), target };
}

/**
 * A drone-height follow camera that keeps the overview's compass direction, so it never swings round behind walls:
 * steeper and closer than the overview, high enough that the low walls never fill the frame.
 */
export function followView(over: { pos: THREE.Vector3; target: THREE.Vector3 }, narrow: boolean) {
  const d = over.pos.clone().sub(over.target), az = Math.atan2(d.x, d.z), pitch = ((narrow ? 58 : 52) * Math.PI) / 180;
  const dist = THREE.MathUtils.clamp(d.length() * (narrow ? 0.62 : 0.55), narrow ? 10 : 8.5, narrow ? 16 : 13);
  return { dir: new THREE.Vector3(Math.sin(az) * Math.cos(pitch), Math.sin(pitch), Math.cos(az) * Math.cos(pitch)), dist };
}
