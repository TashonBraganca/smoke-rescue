"use client";
import { useEffect, useRef, useState } from "react";
import { motion, useReducedMotion } from "motion/react";
import RiseHead from "@/components/RiseHead";
import { missionStore, useMission } from "@/lib/store";
import { SKINS, SKIN_NAMES, type SkinName } from "@/lib/skins";
import { useCanvasStage, inferno, rrect, hexA } from "@/lib/useCanvasStage";

/* ---------- the house: a 40 x 24 cell grid, walls = 1 ---------- */
const GW = 40, GH = 24, N4 = [[1, 0], [-1, 0], [0, 1], [0, -1]] as const;
const GRID = (() => {
  const g = new Uint8Array(GW * GH), w = (x: number, y: number) => { if (x >= 0 && y >= 0 && x < GW && y < GH) g[y * GW + x] = 1; };
  for (let x = 0; x < GW; x++) { w(x, 0); w(x, GH - 1); w(x, 8); w(x, 11); }
  for (let y = 0; y < GH; y++) { w(0, y); w(GW - 1, y); }
  for (const x of [10, 21, 31]) for (let y = 0; y <= 8; y++) w(x, y);
  for (const x of [9, 19, 28]) for (let y = 11; y < GH; y++) w(x, y);
  for (let x = 1; x < 9; x++) w(x, 17);
  for (let x = 29; x < GW - 1; x++) w(x, 16);
  for (const [x, y] of [[5, 8], [15, 8], [26, 8], [35, 8], [4, 11], [14, 11], [23, 11], [33, 11], [21, 5], [5, 17], [34, 16], [0, 9], [0, 10]]) g[y * GW + x] = 0;
  return g;
})();
const START = [[1, 9], [1, 10], [2, 9], [2, 10]];
const inside = (x: number, y: number) => x >= 0 && y >= 0 && x < GW && y < GH;
// reachable floor, and each cell's walking distance from the front door (drives the warm gradient)
const { REACH, NREACH, WARM } = (() => {
  const dist = new Int32Array(GW * GH).fill(-1), q = [9 * GW + 1, 10 * GW + 1];
  for (const k of q) dist[k] = 0;
  for (let qi = 0; qi < q.length; qi++) {
    const k = q[qi], x = k % GW, y = (k / GW) | 0;
    for (const [dx, dy] of N4) { const X = x + dx, Y = y + dy; if (!inside(X, Y)) continue; const kk = Y * GW + X; if (dist[kk] < 0 && !GRID[kk]) { dist[kk] = dist[k] + 1; q.push(kk); } }
  }
  const max = Math.max(...dist), warm: [number, number, number][] = [];
  for (let k = 0; k < GW * GH; k++) warm.push(inferno(0.3 + 0.6 * Math.max(0, dist[k]) / max));
  return { REACH: dist.map((d) => (d >= 0 ? 1 : 0)), NREACH: q.length, WARM: warm };
})();

const RADIO_EVERY = 2, RING_SPEED = 15, RING_MAX = 14, SPEED = 7;
type Dog = { i: number; x: number; y: number; dx: number; dy: number; known: Uint8Array; path: number[]; tgt: number | null };
type Ring = { x: number; y: number; c: number; t0: number };

function dogColours(skin: SkinName) {
  const own = SKINS[skin].hex;
  return [own, skin === "Ember" ? "#FCFFA4" : "#FCB040", "#F2E6D8", "#E0607E"];
}

class Sim {
  n = 3; radio = true; dogs: Dog[] = []; seenAt = new Float32Array(GW * GH); owner = new Int8Array(GW * GH);
  mergeAt = new Float32Array(GW * GH); t = 0; clock = 0; gen = 0; done = false; doneT = 0; rings: Ring[] = []; radioT = 0; pulses = 0;
  reset(n: number, radio: boolean) {
    this.gen++; this.n = n; this.radio = radio; this.t = 0; this.done = false; this.doneT = 0; this.rings = []; this.radioT = 0;
    this.seenAt.fill(-1); this.owner.fill(-1); this.mergeAt.fill(Infinity);
    this.dogs = START.slice(0, n).map(([x, y], i) => ({ i, x: x + 0.5, y: y + 0.5, dx: x + 0.5, dy: y + 0.5, known: new Uint8Array(GW * GH), path: [], tgt: null }));
    for (const d of this.dogs) this.reveal(d);
  }
  reveal(d: Dog) {
    for (let r = 0; r < 72; r++) {
      const a = (r / 72) * Math.PI * 2, ca = Math.cos(a), sa = Math.sin(a);
      for (let s = 0; s < 6.5; s += 0.3) {
        const X = Math.floor(d.x + ca * s), Y = Math.floor(d.y + sa * s);
        if (!inside(X, Y)) break;
        const k = Y * GW + X;
        d.known[k] = GRID[k] ? 2 : 1;
        if (this.seenAt[k] < 0) { this.seenAt[k] = this.clock; this.owner[k] = d.i; }
        if (GRID[k]) break;
      }
    }
  }
  plan(d: Dog) {
    const k0 = Math.floor(d.y) * GW + Math.floor(d.x), par = new Int32Array(GW * GH).fill(-2), q = [k0];
    par[k0] = -1; let best = -1, fb = -1;
    const claimed = this.dogs.filter((o) => o !== d && o.tgt != null).map((o) => o.tgt as number);
    for (let qi = 0; qi < q.length; qi++) {
      const k = q[qi], x = k % GW, y = (k / GW) | 0;
      let frontier = false;
      for (const [dx, dy] of N4) { const X = x + dx, Y = y + dy; if (inside(X, Y) && !d.known[Y * GW + X]) frontier = true; }
      if (frontier && k !== k0) {
        if (fb < 0) fb = k;
        if (claimed.every((c) => Math.abs((c % GW) - x) + Math.abs(((c / GW) | 0) - y) > 7)) { best = k; break; }
      }
      for (const [dx, dy] of N4) {
        const X = x + dx, Y = y + dy; if (!inside(X, Y)) continue;
        const kk = Y * GW + X; if (par[kk] === -2 && d.known[kk] === 1) { par[kk] = k; q.push(kk); }
      }
    }
    if (best < 0) best = fb;
    if (best < 0) { d.tgt = null; d.path = []; return; }
    d.tgt = best; const p: number[] = [];
    for (let k = best; k !== k0 && k >= 0; k = par[k]) p.push(k);
    d.path = p.reverse();
  }
  share() {
    const m = new Uint8Array(GW * GH);
    for (const d of this.dogs) for (let k = 0; k < m.length; k++) if (d.known[k]) m[k] = d.known[k];
    for (const d of this.dogs) { d.known.set(m); d.tgt = null; d.path = []; }
    // the shared map spreads out from each dog at the speed of the drawn ripple
    for (let k = 0; k < m.length; k++) {
      if (m[k] !== 1 || this.mergeAt[k] !== Infinity) continue;
      const x = (k % GW) + 0.5, y = ((k / GW) | 0) + 0.5;
      let dist = RING_MAX;
      for (const d of this.dogs) dist = Math.min(dist, Math.hypot(x - d.x, y - d.y));
      this.mergeAt[k] = this.clock + dist / RING_SPEED;
    }
    for (const d of this.dogs) this.rings.push({ x: d.x, y: d.y, c: d.i, t0: this.clock });
    this.pulses++;
  }
  frac(a: Uint8Array) { let n = 0; for (let k = 0; k < a.length; k++) if (REACH[k] && a[k] === 1) n++; return n / NREACH; }
  tgtDone(d: Dog) { const x = (d.tgt as number) % GW, y = ((d.tgt as number) / GW) | 0; return N4.every(([dx, dy]) => !inside(x + dx, y + dy) || d.known[(y + dy) * GW + x + dx]); }
  step(dt: number) {
    this.clock += dt;
    this.rings = this.rings.filter((r) => (this.clock - r.t0) * RING_SPEED < RING_MAX);
    if (this.done) { this.doneT += dt; return; }
    this.t += dt;
    for (const d of this.dogs) {
      if (!d.path.length || (d.tgt != null && this.tgtDone(d))) this.plan(d);
      let mv = SPEED * dt;
      while (mv > 0 && d.path.length) {
        const k = d.path[0], tx = (k % GW) + 0.5, ty = ((k / GW) | 0) + 0.5, ex = tx - d.x, ey = ty - d.y, L = Math.hypot(ex, ey);
        if (L <= mv) { d.x = tx; d.y = ty; d.path.shift(); mv -= L; } else { d.x += (ex / L) * mv; d.y += (ey / L) * mv; mv = 0; }
      }
      this.reveal(d);
      const f = 1 - Math.exp(-dt * 14);
      d.dx += (d.x - d.dx) * f; d.dy += (d.y - d.dy) * f;
    }
    if (this.radio) { this.radioT += dt; if (this.radioT >= RADIO_EVERY) { this.radioT = 0; this.share(); } }
    const u = new Uint8Array(GW * GH);
    for (const d of this.dogs) for (let k = 0; k < u.length; k++) if (d.known[k] === 1) u[k] = 1;
    if (this.frac(u) > 0.995) this.done = true;
  }
}

// colours pre-blended over the stage background, so overlapping shapes never double up into beads
const BG = [14, 8, 23], INK = [247, 238, 223];
const over = (c: readonly number[], a: number) => `rgb(${(BG[0] + (c[0] - BG[0]) * a) | 0},${(BG[1] + (c[1] - BG[1]) * a) | 0},${(BG[2] + (c[2] - BG[2]) * a) | 0})`;
const ease = (x: number) => 1 - Math.pow(1 - Math.max(0, Math.min(1, x)), 3);

export default function Team() {
  const reduce = useReducedMotion();
  const { dog } = useMission();
  const n = 3;
  const [radio, setRadio] = useState(true);
  const cv = useRef<HTMLCanvasElement>(null);
  const sim = useRef<Sim | null>(null);
  const cols = useRef(dogColours(dog));
  const view = useRef({ s: 1, ox: 0, oy: 0 });
  const bars = useRef<(HTMLSpanElement | null)[]>([]);
  const pcts = useRef<(HTMLSpanElement | null)[]>([]);
  const status = useRef<HTMLSpanElement>(null);
  const ping = useRef<HTMLSpanElement>(null);
  const shown = useRef({ pulses: 0, pct: [-1, -1, -1, -1], status: "" });
  // Two cached layers so each frame only redraws what is still animating: `base` (background, unseen walls and
  // floor dots, drawn once) and `done` (floor cells and walls whose fade-in has finished, added once each).
  const layer = useRef<{ key: string; base: HTMLCanvasElement; done: HTMLCanvasElement; dctx: CanvasRenderingContext2D; cell: Uint8Array; seg: Uint8Array } | null>(null);

  const finishNow = (s: Sim) => { let i = 0; while (!s.done && i++ < 6000) s.step(1 / 30); s.clock += 10; };

  const stage = useCanvasStage(cv, (ctx, S, dt) => {
    const s0 = sim.current;
    if (!s0) return;
    const m = s0, { s, ox, oy } = view.current, X = (x: number) => ox + x * s, Y = (y: number) => oy + y * s;
    if (dt > 0) { m.step(dt); if (m.done && m.doneT > 3.5) m.reset(m.n, m.radio); }
    const now = m.clock, C = cols.current;

    const known = (k: number) => m.seenAt[k] >= 0 && m.dogs.some((d) => d.known[k] === 2);
    const wallA = (k: number) => (known(k) ? 0.1 + 0.55 * ease((now - m.seenAt[k]) / 0.35) : 0.055);
    const BASE_WALL = 0.055, DONE_WALL = 0.65;
    const key = `${m.gen}|${S.w}x${S.h}@${S.dpr}|${s}`;
    let L = layer.current;
    if (!L || L.key !== key) {
      const mk = () => { const c = document.createElement("canvas"); c.width = Math.round(S.w * S.dpr); c.height = Math.round(S.h * S.dpr); const x = c.getContext("2d")!; x.setTransform(S.dpr, 0, 0, S.dpr, 0, 0); return [c, x] as const; };
      const [base, bx] = mk(), [done, dctx] = mk();
      bx.fillStyle = "#0e0817"; bx.fillRect(0, 0, S.w, S.h);
      bx.fillStyle = "rgba(247,238,223,0.07)";
      for (let k = 0; k < GW * GH; k++) if (!GRID[k] && REACH[k]) { bx.beginPath(); bx.arc(X((k % GW) + 0.5), Y(((k / GW) | 0) + 0.5), Math.max(0.8, s * 0.06), 0, Math.PI * 2); bx.fill(); }
      bx.lineCap = dctx.lineCap = "round"; bx.lineWidth = dctx.lineWidth = Math.max(2, s * 0.42); bx.strokeStyle = bx.fillStyle = over(INK, BASE_WALL);
      for (let y = 0; y < GH; y++) for (let x = 0; x < GW; x++) {
        if (!GRID[y * GW + x]) continue;
        let joined = false;
        for (const [dx, dy] of [[1, 0], [0, 1]] as const) { if (!inside(x + dx, y + dy) || !GRID[(y + dy) * GW + x + dx]) continue; joined = true; bx.beginPath(); bx.moveTo(X(x + 0.5), Y(y + 0.5)); bx.lineTo(X(x + dx + 0.5), Y(y + dy + 0.5)); bx.stroke(); }
        if (!joined) { bx.beginPath(); bx.arc(X(x + 0.5), Y(y + 0.5), bx.lineWidth / 2, 0, Math.PI * 2); bx.fill(); }
      }
      L = layer.current = { key, base, done, dctx, cell: new Uint8Array(GW * GH), seg: new Uint8Array(GW * GH * 2) };
    }
    ctx.drawImage(L.base, 0, 0, S.w, S.h);
    ctx.drawImage(L.done, 0, 0, S.w, S.h);
    for (let y = 0; y < GH; y++) for (let x = 0; x < GW; x++) {
      const k = y * GW + x;
      if (GRID[k] || L.cell[k]) continue;
      const cx = X(x + 0.5), cy = Y(y + 0.5), ma = m.mergeAt[k];
      if (now >= ma) {
        // shared map: each cell grows from a dot into a seamless warm floor; once grown it moves to the cached layer
        const e = ease((now - ma) / 0.5), sz = s * (0.3 + 0.7 * e) + e * 0.6;
        if (e >= 1) L.cell[k] = 1;                               // finished: also goes into the cached layer for later frames
        for (const tgt of e >= 1 ? [ctx, L.dctx] : [ctx]) { tgt.fillStyle = over(WARM[k], 0.5 - 0.12 * e); rrect(tgt, cx - sz / 2, cy - sz / 2, sz, sz, sz * 0.5 * (1 - e)); tgt.fill(); }
      } else if (m.seenAt[k] >= 0) {
        const e = ease((now - m.seenAt[k]) / 0.3), sz = s * 0.3 * (0.4 + 0.6 * e);
        ctx.fillStyle = hexA(C[Math.max(0, m.owner[k])], 0.8 * e);
        ctx.beginPath(); ctx.arc(cx, cy, sz / 2, 0, Math.PI * 2); ctx.fill();
      }
    }
    // walls as continuous rounded strokes, like a floor plan, fading in as the dogs see them
    ctx.lineCap = "round"; ctx.lineWidth = Math.max(2, s * 0.42);
    for (let y = 0; y < GH; y++) for (let x = 0; x < GW; x++) {
      const k = y * GW + x;
      if (!GRID[k] || !known(k)) continue;                    // unseen walls are already in the base layer
      const a = wallA(k), cx = X(x + 0.5), cy = Y(y + 0.5);
      let joined = false;
      for (const [dx, dy, j] of [[1, 0, 0], [0, 1, 1]] as const) {
        const X2 = x + dx, Y2 = y + dy;
        if (!inside(X2, Y2) || !GRID[Y2 * GW + X2]) continue;
        joined = true;
        if (L.seg[k * 2 + j]) continue;
        const al = Math.min(a, wallA(Y2 * GW + X2));
        if (al <= BASE_WALL) continue;
        if (al >= DONE_WALL) L.seg[k * 2 + j] = 1;
        for (const tgt of al >= DONE_WALL ? [ctx, L.dctx] : [ctx]) { tgt.strokeStyle = over(INK, al); tgt.beginPath(); tgt.moveTo(cx, cy); tgt.lineTo(X(X2 + 0.5), Y(Y2 + 0.5)); tgt.stroke(); }
      }
      if (!joined && !L.seg[k * 2]) {
        if (a >= DONE_WALL) L.seg[k * 2] = 1;
        for (const tgt of a >= DONE_WALL ? [ctx, L.dctx] : [ctx]) { tgt.fillStyle = over(INK, a); tgt.beginPath(); tgt.arc(cx, cy, ctx.lineWidth / 2, 0, Math.PI * 2); tgt.fill(); }
      }
    }

    // radio ripples: two soft rings per dog per pulse
    for (const r of m.rings) {
      const rad = (now - r.t0) * RING_SPEED, a = Math.max(0, 1 - rad / RING_MAX);
      for (const [f, w, al] of [[1, Math.max(5, s * 0.5), 0.14], [1, 1.75, 0.95], [0.7, 1, 0.4]] as const) {
        ctx.strokeStyle = hexA(C[r.c], a * al); ctx.lineWidth = w;
        ctx.beginPath(); ctx.arc(X(r.x), Y(r.y), Math.max(0, rad * f * s), 0, Math.PI * 2); ctx.stroke();
      }
    }
    // where each dog is heading
    ctx.setLineDash([1, Math.max(3, s * 0.55)]); ctx.lineCap = "round"; ctx.lineWidth = Math.max(1.5, s * 0.14);
    for (const d of m.dogs) {
      if (!d.path.length) continue;
      ctx.strokeStyle = hexA(C[d.i], 0.55); ctx.beginPath(); ctx.moveTo(X(d.dx), Y(d.dy));
      for (const k of d.path) ctx.lineTo(X((k % GW) + 0.5), Y(((k / GW) | 0) + 0.5));
      ctx.stroke();
    }
    ctx.setLineDash([]);
    // the dogs, with a soft glow
    for (const d of m.dogs) {
      const x = X(d.dx), y = Y(d.dy), r = Math.max(4.5, s * 0.5);
      const g = ctx.createRadialGradient(x, y, 0, x, y, r * 3.4);
      g.addColorStop(0, hexA(C[d.i], 0.55)); g.addColorStop(1, hexA(C[d.i], 0));
      ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, r * 3.4, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = C[d.i]; ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = "rgba(247,238,223,0.9)"; ctx.lineWidth = 1.25; ctx.stroke();
    }
    // "Front door" label fades once the dogs are inside
    const la = 1 - ease((m.t - 1.4) / 0.8);
    if (la > 0.01) {
      ctx.save(); ctx.globalAlpha = la;
      ctx.font = `800 11px ${S.font}`; ctx.textBaseline = "middle";
      const tw = ctx.measureText("Front door").width + 18, bx = X(0.15), by = Y(7.6) - 22;
      rrect(ctx, bx, by, tw, 22, 11); ctx.fillStyle = "rgba(11,6,18,0.85)"; ctx.fill();
      ctx.strokeStyle = "rgba(252,255,164,0.6)"; ctx.lineWidth = 1; ctx.stroke();
      ctx.fillStyle = "#fcffa4"; ctx.fillText("Front door", bx + 9, by + 11.5);
      ctx.strokeStyle = "rgba(252,255,164,0.6)"; ctx.beginPath(); ctx.moveTo(X(0.5), by + 22); ctx.lineTo(X(0.5), Y(9)); ctx.stroke();
      ctx.restore();
    }

    // side panel: each dog's share of the house, radio pings, timer
    const sh = shown.current;
    m.dogs.forEach((d, i) => {
      const p = Math.round(m.frac(d.known) * 100);
      if (p === sh.pct[i]) return;
      sh.pct[i] = p;
      const b = bars.current[i], t = pcts.current[i];
      if (b) b.style.transform = `scaleX(${p / 100})`;
      if (t) t.textContent = `${p}%`;
    });
    for (let i = m.dogs.length; i < 4; i++) if (sh.pct[i] !== -1) {
      sh.pct[i] = -1; const b = bars.current[i], t = pcts.current[i];
      if (b) b.style.transform = "scaleX(0)"; if (t) t.textContent = "";
    }
    if (m.pulses !== sh.pulses) {
      sh.pulses = m.pulses;
      if (!S.reduce) ping.current?.animate([{ transform: "scale(1)", opacity: 0.9 }, { transform: "scale(3.2)", opacity: 0 }], { duration: 900, easing: "cubic-bezier(0.2,0.9,0.25,1)" });
    }
    const st = m.done ? `Mapped in ${m.t.toFixed(1)} s` : `${m.t.toFixed(1)} s`;
    if (st !== sh.status && status.current) { sh.status = st; status.current.textContent = st; status.current.dataset.done = String(m.done); }
  }, (S) => {
    const pad = 10, s = Math.min((S.w - pad * 2) / GW, (S.h - pad * 2) / GH);
    view.current = { s, ox: (S.w - GW * s) / 2, oy: (S.h - GH * s) / 2 };
  });

  useEffect(() => {
    const m = (sim.current ??= new Sim());
    m.reset(n, radio);
    shown.current.pct = [-1, -1, -1, -1];
    if (matchMedia("(prefers-reduced-motion: reduce)").matches) finishNow(m);
    stage.current.redraw();
  }, [n, radio, stage]);
  useEffect(() => { cols.current = dogColours(dog); stage.current.redraw(); }, [dog, stage]);

  const C = dogColours(dog);
  const facts = [["2 s", "between radio messages"], ["12 m", "radio range"], ["1 wall", "it gets through"]];

  return (
    <section id="team" className="gutter py-[clamp(80px,11vw,140px)]">
      <div className="mx-auto max-w-[1200px]">
        <RiseHead eyebrow="Team" title="Every dog maps. The radio shares it." className="mb-[clamp(36px,5vw,64px)]" titleClass="max-w-[15ch]" />

        <motion.div
          initial={{ opacity: 0, y: 48 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-6% 0px" }}
          transition={reduce ? { duration: 0 } : { type: "spring", stiffness: 70, damping: 20 }}
          className="grid gap-5 lg:grid-cols-[minmax(0,1.75fr)_minmax(0,1fr)]"
        >
          <div className="relative min-w-0 overflow-hidden rounded-[28px] border border-line bg-stage">
            <canvas
              ref={cv}
              className="block aspect-[5/3] w-full"
              role="img"
              aria-label="Dogs exploring a house. Each dog fills in only what it has seen; on every radio pulse their maps merge into one."
            />
          </div>

          <div className="grid min-w-0 content-between gap-7 rounded-[28px] border border-line bg-stage p-6 sm:p-7">
            <div className="grid gap-5">
              <div className="flex items-baseline justify-between gap-3">
                <span className="text-footnote font-extrabold uppercase tracking-[0.16em] text-mute">Each dog&apos;s map</span>
                <span ref={status} className="text-sm font-bold tabular-nums text-sub data-[done=true]:text-amber">0.0 s</span>
              </div>
              <ul className="grid gap-4">
                {[0, 1, 2].map((i) => (
                  <li key={i} className="grid grid-cols-[4.5rem_minmax(0,1fr)_2.75rem] items-center gap-3">
                    <span className="flex items-center gap-2 text-sm font-semibold text-ink">
                      <i className="h-2.5 w-2.5 rounded-full" style={{ background: C[i], boxShadow: i < n ? `0 0 10px ${C[i]}` : "none" }} />
                      Dog {"ABCD"[i]}
                    </span>
                    <span className="relative h-2 overflow-hidden rounded-full bg-white/[0.08]">
                      <span
                        ref={(el) => { bars.current[i] = el; }}
                        className="absolute inset-0 origin-left rounded-full transition-transform duration-500 ease-[var(--ease-spring)]"
                        style={{ background: C[i], transform: "scaleX(0)" }}
                      />
                    </span>
                    <span ref={(el) => { pcts.current[i] = el; }} className="text-right text-sm tabular-nums text-sub" />
                  </li>
                ))}
              </ul>
            </div>

            <div className="grid gap-6 border-t border-line pt-6">
              <div className="grid gap-3" role="radiogroup" aria-label="Lead dog">
                <span className="text-sm font-semibold text-sub">Lead dog</span>
                <div className="flex flex-wrap gap-2">
                  {SKIN_NAMES.map((name) => (
                    <button key={name} type="button" role="radio" aria-checked={dog === name} onClick={() => missionStore.set({ dog: name })}
                      className={`flex items-center gap-2 rounded-full border px-3.5 py-2 text-sm font-semibold transition-colors ${dog === name ? "border-heat bg-heat/15 text-ink" : "border-line2 text-sub hover:text-ink"}`}>
                      <i className="h-2.5 w-2.5 rounded-full" style={{ background: SKINS[name].hex }} />
                      {name}
                    </button>
                  ))}
                </div>
              </div>

              <button
                type="button" role="switch" aria-checked={radio} onClick={() => setRadio((r) => !r)}
                className="flex items-center justify-between gap-4 text-left"
              >
                <span className="flex items-center gap-3 text-sm font-semibold text-sub">
                  <span className="relative grid h-2.5 w-2.5 place-items-center">
                    <span ref={ping} className={`absolute inset-0 rounded-full ${radio ? "bg-amber" : "bg-transparent"}`} />
                    <span className={`h-2.5 w-2.5 rounded-full transition-colors ${radio ? "bg-amber" : "bg-white/20"}`} />
                  </span>
                  Radio {radio ? "on" : "off"}
                </span>
                <span className={`relative h-7 w-12 shrink-0 rounded-full transition-colors duration-300 ${radio ? "bg-heat" : "bg-white/15"}`}>
                  <span className={`absolute left-0.5 top-0.5 h-6 w-6 rounded-full bg-ink shadow-[0_2px_6px_rgba(0,0,0,0.45)] transition-transform duration-300 ease-[var(--ease-spring)] ${radio ? "translate-x-5" : "translate-x-0"}`} />
                </span>
              </button>
              <p className="text-footnote text-mute">Simplified model.</p>
            </div>
          </div>
        </motion.div>

        <div className="mt-[clamp(48px,7vw,88px)] grid grid-cols-3 gap-4 border-t border-line pt-8">
          {facts.map(([big, small], i) => (
            <motion.div
              key={big}
              initial={{ opacity: 0, y: 28 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-8% 0px" }}
              transition={reduce ? { duration: 0 } : { type: "spring", stiffness: 80, damping: 20, delay: i * 0.1 }}
              className="grid min-w-0 content-start gap-2"
            >
              <span className="whitespace-nowrap text-[clamp(2rem,6.4vw,5rem)] font-extrabold leading-none tracking-[-0.04em] text-amber">{big}</span>
              <span className="text-[clamp(0.8rem,1.6vw,1.0625rem)] font-semibold leading-snug text-sub">{small}</span>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
