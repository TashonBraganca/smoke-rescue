"use client";
import { useEffect, useRef, useState, type ReactNode } from "react";
import {
  motion, type MotionStyle, type MotionValue, useMotionTemplate, useMotionValue, useMotionValueEvent, useReducedMotion, useScroll, useTransform,
} from "motion/react";
import RiseHead from "@/components/RiseHead";
import { useMission } from "@/lib/store";
import { SKINS } from "@/lib/skins";
import { useCanvasStage, inferno, rrect, hexA } from "@/lib/useCanvasStage";

export default function Smoke() {
  return (
    <section id="smoke" className="gutter py-[clamp(80px,11vw,140px)]">
      <div className="mx-auto max-w-[1200px]">
        <RiseHead eyebrow="Through the smoke" title="Colour sees smoke. Heat sees people." className="mb-[clamp(36px,5vw,64px)]" titleClass="max-w-[14ch]" />
        <div className="grid gap-5 md:grid-cols-2">
          <Panel
            className="md:col-span-2"
            caption={<><b className="font-semibold text-ink">Same hallway, two cameras.</b> Scroll or drag.</>}
          >
            <Compare />
          </Panel>
          <Panel delay={0.05} caption={<><b className="font-semibold text-ink">It sweeps the room.</b> A body-warm shape gets flagged.</>}>
            <Cone />
          </Panel>
          <Panel delay={0.12} caption={<><b className="font-semibold text-ink">OpenCV traces the warm shape.</b> That outline is what the dog reports.</>}>
            <Contour />
          </Panel>
        </div>
      </div>
    </section>
  );
}

function Panel({ children, caption, className = "", delay = 0 }: { children: ReactNode; caption: ReactNode; className?: string; delay?: number }) {
  const reduce = useReducedMotion();
  return (
    <motion.figure
      initial={{ opacity: 0, y: 48 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-6% 0px" }}
      transition={reduce ? { duration: 0 } : { type: "spring", stiffness: 70, damping: 20, mass: 1, delay }}
      className={`m-0 min-w-0 overflow-hidden rounded-[28px] border border-line bg-stage ${className}`}
    >
      {children}
      <figcaption className="px-6 py-5 text-[15px] leading-snug text-sub">{caption}</figcaption>
    </motion.figure>
  );
}

function Chip({ children, className = "", style }: { children: ReactNode; className?: string; style?: MotionStyle }) {
  return (
    <motion.span style={style} className={`pointer-events-none absolute top-4 rounded-full border border-white/10 bg-bg/70 px-3 py-1.5 text-[11px] font-extrabold uppercase tracking-[0.14em] text-whiteheat ${className}`}>
      {children}
    </motion.span>
  );
}

/* ---------- (a) colour vs thermal: the divider follows the scroll; drag or arrow keys to take over ---------- */
const clampN = (lo: number, hi: number, v: number) => Math.max(lo, Math.min(hi, v));
const fromScroll = (p: number) => clampN(6, 94, 92 - p * 84); // p = 0: panel centre at the bottom edge, 1: at the top

function Compare() {
  const box = useRef<HTMLDivElement>(null);
  // driven straight from scroll progress: Lenis already smooths the scroll, a spring on top only adds lag
  const { scrollYProgress } = useScroll({ target: box, offset: ["center end", "center start"] });
  const cut = useMotionValue(fromScroll(0));
  // after a drag or key press the hand-set position holds until the page scrolls, then melts back into the
  // scroll position over the next stretch of scrolling (by distance scrolled, not by time, so it never lags)
  const st = useRef({ manual: false, held: false, offset: 0, lastP: 0 });
  const thermalX = useMotionTemplate`${cut}%`;
  const imageX = useTransform(cut, (v) => `${-v}%`);
  const colourA = useTransform(cut, [8, 20], [0, 1]);
  const heatA = useTransform(cut, [80, 92], [1, 0]);

  useEffect(() => { const p = scrollYProgress.get(); st.current.lastP = p; cut.set(fromScroll(p)); }, [scrollYProgress, cut]);
  useMotionValueEvent(scrollYProgress, "change", (p) => {
    const s = st.current;
    if (s.held) { s.lastP = p; return; } // a finger or the mouse is on the divider
    if (s.manual) { s.manual = false; s.offset = cut.get() - fromScroll(p); s.lastP = p; return; }
    s.offset *= Math.exp(-Math.abs(p - s.lastP) * 10); s.lastP = p;
    if (Math.abs(s.offset) < 0.05) s.offset = 0;
    cut.set(clampN(0, 100, fromScroll(p) + s.offset));
  });
  useMotionValueEvent(cut, "change", (v) => box.current?.setAttribute("aria-valuenow", String(Math.round(v))));

  const setFrom = (x: number) => {
    const r = box.current!.getBoundingClientRect();
    cut.set(clampN(0, 100, ((x - r.left) / r.width) * 100));
    st.current.manual = true;
  };
  // a mouse takes over on press; a finger only after clear sideways movement, so vertical page scrolls pass through
  const drag = useRef({ id: -1, x0: 0, y0: 0, on: false });
  const onDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    drag.current = { id: e.pointerId, x0: e.clientX, y0: e.clientY, on: false };
    if (e.pointerType === "mouse") { drag.current.on = true; st.current.held = true; e.currentTarget.setPointerCapture(e.pointerId); setFrom(e.clientX); }
  };
  const onMove = (e: React.PointerEvent<HTMLDivElement>) => {
    const d = drag.current;
    if (e.pointerId !== d.id) return;
    if (!d.on) {
      const dx = Math.abs(e.clientX - d.x0), dy = Math.abs(e.clientY - d.y0);
      if (dy > 10 && dy > dx) { d.id = -1; return; } // a vertical swipe: leave it to the page
      if (dx < 8 || dx < dy * 1.2) return;
      d.on = true; st.current.held = true; e.currentTarget.setPointerCapture(e.pointerId);
    }
    setFrom(e.clientX);
  };
  const onUp = (e: React.PointerEvent<HTMLDivElement>) => {
    if (e.pointerId !== drag.current.id) return;
    drag.current.id = -1; drag.current.on = false; st.current.held = false;
  };

  return (
    <div
      ref={box}
      role="slider"
      tabIndex={0}
      aria-label="Divider between the colour and thermal views"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(fromScroll(0))}
      onPointerDown={onDown}
      onPointerMove={onMove}
      onPointerUp={onUp}
      onPointerCancel={onUp}
      onKeyDown={(e) => {
        const step = e.key === "ArrowLeft" ? -5 : e.key === "ArrowRight" ? 5 : e.key === "Home" ? -100 : e.key === "End" ? 100 : 0;
        if (!step) return;
        e.preventDefault();
        cut.set(clampN(0, 100, cut.get() + step));
        st.current.manual = true;
      }}
      className="relative aspect-[4/3] cursor-ew-resize touch-pan-y select-none overflow-hidden bg-black md:aspect-[16/9]"
    >
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src="/img/pov_rgb.jpg" alt="Hallway seen by the colour camera" width={640} height={480} draggable={false} className="absolute inset-0 h-full w-full object-cover" />
      {/* the thermal layer slides as a whole and its image slides back the same amount: transforms only, no repaint per frame */}
      <motion.div style={{ x: thermalX }} className="absolute inset-0 overflow-hidden will-change-transform">
        <motion.div style={{ x: imageX }} className="absolute inset-0 will-change-transform">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src="/img/pov_th.jpg" alt="The same hallway seen by the thermal camera" width={640} height={480} draggable={false} className="absolute inset-0 h-full w-full object-cover" />
        </motion.div>
      </motion.div>
      <motion.div style={{ x: thermalX }} className="pointer-events-none absolute inset-0 will-change-transform">
        <div className="absolute inset-y-0 -left-px w-[2px] bg-whiteheat shadow-[0_0_18px_2px_rgba(249,140,10,0.75)]" />
        <div className="absolute left-0 top-1/2 grid h-11 w-11 -translate-x-1/2 -translate-y-1/2 place-items-center rounded-full border border-whiteheat/70 bg-bg/85 shadow-[0_0_24px_rgba(249,140,10,0.45)]">
          <svg width="20" height="12" viewBox="0 0 20 12" fill="none" aria-hidden="true">
            <path d="M6 1 1 6l5 5M14 1l5 5-5 5" stroke="#fcffa4" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </div>
      </motion.div>
      <Chip style={{ opacity: colourA }} className="left-4">Colour</Chip>
      <Chip style={{ opacity: heatA }} className="right-4">Heat</Chip>
    </div>
  );
}

/* ---------- (b) top-down thermal cone ---------- */
type Seg = [number, number, number, number];
const RW = 8, RH = 6; // room, metres
const DOG = { x: 0.9, y: 3.4 }, PERSON = { x: 4.25, y: 4.95 }, FIRE = { x: 7.0, y: 0.95 };
const FURN: [number, number, number, number][] = [[4.9, 2.3, 6.5, 3.2], [0.3, 0.3, 2.3, 1.0]];
function wallH(y: number, x0: number, x1: number, gaps: [number, number][] = []): Seg[] {
  const o: Seg[] = []; let s = x0;
  for (const [a, b] of [...gaps].sort((p, q) => p[0] - q[0])) { if (a > s) o.push([s, y, a, y]); s = b; }
  if (x1 > s) o.push([s, y, x1, y]); return o;
}
function wallV(x: number, y0: number, y1: number, gaps: [number, number][] = []): Seg[] {
  return wallH(x, y0, y1, gaps).map(([a, , b]) => [x, a, x, b] as Seg);
}
const SHELL: Seg[] = [...wallH(0, 0, RW), ...wallH(RH, 0, RW), ...wallV(0, 0, RH), ...wallV(RW, 0, RH, [[2.4, 3.4]]), [3.0, 0, 3.0, 1.4]];
const WALLS: Seg[] = [...SHELL, ...FURN.flatMap(([a, b, c, d]) => [[a, b, c, b], [c, b, c, d], [c, d, a, d], [a, d, a, b]] as Seg[])];
function raySeg(ox: number, oy: number, dx: number, dy: number, g: Seg) {
  const ex = g[2] - g[0], ey = g[3] - g[1], den = dx * ey - dy * ex;
  if (Math.abs(den) < 1e-9) return Infinity;
  const t = ((g[0] - ox) * ey - (g[1] - oy) * ex) / den, u = ((g[0] - ox) * dy - (g[1] - oy) * dx) / den;
  return t > 1e-6 && u >= 0 && u <= 1 ? t : Infinity;
}
function cast(ox: number, oy: number, a: number, max: number) {
  const dx = Math.cos(a), dy = Math.sin(a); let best = max;
  for (const g of WALLS) { const t = raySeg(ox, oy, dx, dy, g); if (t < best) best = t; }
  return best;
}
const wrapA = (d: number) => { while (d > Math.PI) d -= 2 * Math.PI; while (d < -Math.PI) d += 2 * Math.PI; return d; };
const smooth = (a: number, b: number, x: number) => { const t = Math.max(0, Math.min(1, (x - a) / (b - a))); return t * t * (3 - 2 * t); };

let fieldCache: { heat: HTMLCanvasElement; smoke: HTMLCanvasElement } | null = null;
function fields() {
  if (fieldCache) return fieldCache;
  const GW = 320, GH = 240, heat = document.createElement("canvas");
  heat.width = GW; heat.height = GH;
  const hx = heat.getContext("2d")!, id = hx.createImageData(GW, GH);
  let seed = 7; const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  for (let j = 0; j < GH; j++) for (let i = 0; i < GW; i++) {
    const x = ((i + 0.5) / GW) * RW, y = ((j + 0.5) / GH) * RH, df = (x - FIRE.x) ** 2 + (y - FIRE.y) ** 2;
    let T = 21 + 520 * Math.exp(-df / 0.245) + 70 * Math.exp(-df / 3.38) + 14 * Math.exp(-df / 18);
    // a person lying on the floor: a body capsule plus a head
    const sx0 = PERSON.x - 0.75, sx1 = PERSON.x + 0.5, cx = Math.max(sx0, Math.min(sx1, x));
    const dseg = Math.hypot(x - cx, (y - PERSON.y) * (x < PERSON.x - 0.1 ? 1.25 : 1));
    T += 27 * Math.exp(-((dseg / 0.19) ** 2));
    T += 26 * Math.exp(-((Math.hypot(x - (PERSON.x + 0.78), y - (PERSON.y - 0.02)) / 0.13) ** 2));
    T += 6 * Math.exp(-(((x - 5.7) / 0.7) ** 2 + ((y - 2.75) / 0.4) ** 2));
    T += (rnd() - 0.5) * 1.6;
    const c = inferno((T - 15) / 45), k = (j * GW + i) * 4;
    id.data[k] = c[0]; id.data[k + 1] = c[1]; id.data[k + 2] = c[2]; id.data[k + 3] = 255;
  }
  hx.putImageData(id, 0, 0);
  const SW = 160, SH = 120, smoke = document.createElement("canvas");
  smoke.width = SW; smoke.height = SH;
  const sx = smoke.getContext("2d")!, sd = sx.createImageData(SW, SH);
  for (let j = 0; j < SH; j++) for (let i = 0; i < SW; i++) {
    const x = ((i + 0.5) / SW) * RW, y = ((j + 0.5) / SH) * RH;
    const n = 0.5 + 0.22 * Math.sin(x * 1.7 + Math.sin(y * 2.1) * 2) + 0.18 * Math.sin(y * 1.3 - x * 0.8 + Math.cos(x * 2.4)) + 0.1 * Math.sin(x * 4.1 + y * 3.3);
    const g = 30 + n * 34, k = (j * SW + i) * 4;
    sd.data[k] = g + 10; sd.data[k + 1] = g + 3; sd.data[k + 2] = g + 14; sd.data[k + 3] = 255;
  }
  sx.putImageData(sd, 0, 0);
  return (fieldCache = { heat, smoke });
}

function chip(ctx: CanvasRenderingContext2D, font: string, text: string, x: number, y: number, a: number, hot = false) {
  if (a <= 0.01) return;
  ctx.save(); ctx.globalAlpha = a;
  ctx.font = `800 11px ${font}`;
  const w = ctx.measureText(text).width + 18, h = 22;
  rrect(ctx, x, y - h, w, h, 11);
  ctx.fillStyle = "rgba(11,6,18,0.82)"; ctx.fill();
  ctx.strokeStyle = hot ? "rgba(252,255,164,0.7)" : "rgba(247,238,223,0.18)"; ctx.lineWidth = 1; ctx.stroke();
  ctx.fillStyle = hot ? "#fcffa4" : "#f7eedf"; ctx.textBaseline = "middle"; ctx.fillText(text, x + 9, y - h / 2 + 0.5);
  ctx.restore();
}

function Cone() {
  const cv = useRef<HTMLCanvasElement>(null);
  const { dog } = useMission();
  const skin = useRef<string>(SKINS[dog].hex);
  const view = useRef({ s: 1, ox: 0, oy: 0 });
  const st = useRef({ head: 0, phase: 0, flag: 0, pin: 0, fire: 0, aim: null as null | [number, number], aimT: -9 });
  const HALF = (21 * Math.PI) / 180, RANGE = 7.6;
  const personA = Math.atan2(PERSON.y - DOG.y, PERSON.x - DOG.x);

  const stage = useCanvasStage(cv, (ctx, S, dt) => {
    const { s, ox, oy } = view.current, X = (x: number) => ox + x * s, Y = (y: number) => oy + y * s;
    const q = st.current, F = fields();
    if (S.reduce) { q.head = personA - 0.12; q.flag = 1; q.pin = 1; }
    else if (dt > 0) {
      let target: number;
      if (q.aim && S.t - q.aimT < 2.5) target = Math.atan2(q.aim[1] - DOG.y, q.aim[0] - DOG.x);
      else {
        // slow down while the person is in the cone, as if taking a longer look
        q.phase += dt * 0.6 * (0.22 + 0.78 * smooth(0.04, 0.42, Math.abs(wrapA(q.head - personA))));
        target = Math.sin(q.phase) * 0.82;
      }
      q.head += wrapA(target - q.head) * Math.min(1, dt * 4.5);
    }
    const seen = (p: { x: number; y: number }, r: number) => {
      const a = Math.atan2(p.y - DOG.y, p.x - DOG.x), dist = Math.hypot(p.x - DOG.x, p.y - DOG.y);
      return Math.abs(wrapA(a - q.head)) < HALF && dist < RANGE && cast(DOG.x, DOG.y, a, RANGE) > dist - r;
    };
    const pSeen = seen(PERSON, 0.4), fSeen = seen(FIRE, 0.3);
    if (!S.reduce) {
      const k = Math.min(1, dt * 5);
      q.flag += ((pSeen ? 1 : 0) - q.flag) * k;
      q.fire += ((fSeen ? 1 : 0) - q.fire) * k;
      if (pSeen) q.pin = Math.min(1, q.pin + dt * 1.5);
    } else q.fire = fSeen ? 1 : 0;

    ctx.fillStyle = "#0e0817"; ctx.fillRect(0, 0, S.w, S.h);
    // smoke on the floor: what the colour camera would see
    ctx.save();
    rrect(ctx, X(0), Y(0), RW * s, RH * s, 6); ctx.clip();
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(F.smoke, X(0), Y(0), RW * s, RH * s);
    ctx.globalAlpha = 0.45;
    ctx.drawImage(F.smoke, X(0) + Math.sin(S.t * 0.21) * 0.5 * s - 0.5 * s, Y(0) + Math.cos(S.t * 0.17) * 0.3 * s - 0.3 * s, (RW + 1) * s, (RH + 0.6) * s);
    ctx.globalAlpha = 1;
    // firelight bleeding through the smoke: the colour camera sees a glow, not the room
    const fl = 0.85 + (S.reduce ? 0 : 0.15 * Math.sin(S.t * 2.3) * Math.sin(S.t * 0.7));
    const fg = ctx.createRadialGradient(X(FIRE.x), Y(FIRE.y), 0, X(FIRE.x), Y(FIRE.y), 3.2 * s);
    fg.addColorStop(0, `rgba(249,140,10,${0.42 * fl})`); fg.addColorStop(0.45, `rgba(188,55,84,${0.16 * fl})`); fg.addColorStop(1, "rgba(87,16,110,0)");
    ctx.fillStyle = fg; ctx.fillRect(X(0), Y(0), RW * s, RH * s);
    const vg = ctx.createRadialGradient(X(RW / 2), Y(RH / 2), 1.5 * s, X(RW / 2), Y(RH / 2), 5.4 * s);
    vg.addColorStop(0, "rgba(14,8,23,0)"); vg.addColorStop(1, "rgba(14,8,23,0.55)");
    ctx.fillStyle = vg; ctx.fillRect(X(0), Y(0), RW * s, RH * s);
    ctx.restore();

    // the thermal cone, clipped by walls
    const pts: [number, number][] = [];
    for (let i = 0; i <= 72; i++) {
      const a = q.head - HALF + (2 * HALF * i) / 72, d = cast(DOG.x, DOG.y, a, RANGE);
      pts.push([X(DOG.x + Math.cos(a) * d), Y(DOG.y + Math.sin(a) * d)]);
    }
    ctx.save();
    ctx.beginPath(); ctx.moveTo(X(DOG.x), Y(DOG.y)); for (const [x, y] of pts) ctx.lineTo(x, y); ctx.closePath(); ctx.clip();
    ctx.drawImage(F.heat, X(0), Y(0), RW * s, RH * s);
    const fade = ctx.createRadialGradient(X(DOG.x), Y(DOG.y), RANGE * 0.55 * s, X(DOG.x), Y(DOG.y), RANGE * s);
    fade.addColorStop(0, "rgba(14,8,23,0)"); fade.addColorStop(1, "rgba(14,8,23,0.75)");
    ctx.fillStyle = fade; ctx.fillRect(0, 0, S.w, S.h);
    ctx.restore();
    // cone edges fade out along their length
    for (const e of [pts[0], pts[pts.length - 1]]) {
      const g = ctx.createLinearGradient(X(DOG.x), Y(DOG.y), e[0], e[1]);
      g.addColorStop(0, "rgba(252,255,164,0.7)"); g.addColorStop(1, "rgba(252,255,164,0)");
      ctx.strokeStyle = g; ctx.lineWidth = 1.25; ctx.beginPath(); ctx.moveTo(X(DOG.x), Y(DOG.y)); ctx.lineTo(e[0], e[1]); ctx.stroke();
    }

    // walls and furniture
    ctx.lineCap = "round"; ctx.lineJoin = "round";
    for (const [a, b, c, d] of FURN) {
      rrect(ctx, X(a), Y(b), (c - a) * s, (d - b) * s, 4);
      ctx.fillStyle = "rgba(247,238,223,0.035)"; ctx.fill();
      ctx.strokeStyle = "rgba(247,238,223,0.28)"; ctx.lineWidth = 1.25; ctx.stroke();
    }
    ctx.strokeStyle = "rgba(247,238,223,0.55)"; ctx.lineWidth = 2.5; ctx.beginPath();
    for (const g of SHELL) { ctx.moveTo(X(g[0]), Y(g[1])); ctx.lineTo(X(g[2]), Y(g[3])); }
    ctx.stroke();

    // flagged pin stays after the first sighting
    if (q.pin > 0) {
      ctx.save(); ctx.globalAlpha = q.pin * (1 - q.flag * 0.6);
      const px = X(PERSON.x), py = Y(PERSON.y), pr = Math.max(4, 0.09 * s);
      const pulse = S.reduce ? 0.5 : (S.t * 0.8) % 1;
      ctx.strokeStyle = `rgba(252,176,64,${0.6 * (1 - pulse)})`; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(px, py, pr + pulse * 0.45 * s, 0, Math.PI * 2); ctx.stroke();
      ctx.fillStyle = "#fcb040"; ctx.beginPath(); ctx.arc(px, py, pr, 0, Math.PI * 2); ctx.fill();
      ctx.restore();
    }
    // detection brackets grow in while the shape is inside the cone
    if (q.flag > 0.01) {
      const x0 = X(PERSON.x - 0.95), y0 = Y(PERSON.y - 0.42), x1 = X(PERSON.x + 1.0), y1 = Y(PERSON.y + 0.42);
      const L = Math.min(x1 - x0, y1 - y0) * 0.38 * q.flag;
      ctx.save(); ctx.globalAlpha = q.flag;
      ctx.strokeStyle = "#fcffa4"; ctx.lineWidth = 2; ctx.shadowColor = "rgba(249,140,10,0.9)"; ctx.shadowBlur = 10;
      ctx.beginPath();
      ctx.moveTo(x0, y0 + L); ctx.lineTo(x0, y0); ctx.lineTo(x0 + L, y0);
      ctx.moveTo(x1 - L, y0); ctx.lineTo(x1, y0); ctx.lineTo(x1, y0 + L);
      ctx.moveTo(x1, y1 - L); ctx.lineTo(x1, y1); ctx.lineTo(x1 - L, y1);
      ctx.moveTo(x0 + L, y1); ctx.lineTo(x0, y1); ctx.lineTo(x0, y1 - L);
      ctx.stroke(); ctx.restore();
      chip(ctx, S.font, "Body-warm shape", x0, y0 - 8, q.flag, true);
    }
    chip(ctx, S.font, "Fire", X(FIRE.x - 0.45), Y(FIRE.y + 1.05), q.fire);

    // our dog, in the visitor's chosen paint
    const dx = X(DOG.x), dy = Y(DOG.y), r = Math.max(6, 0.17 * s), col = skin.current;
    const glow = ctx.createRadialGradient(dx, dy, 0, dx, dy, r * 4);
    glow.addColorStop(0, hexA(col, 0.5)); glow.addColorStop(1, hexA(col, 0));
    ctx.fillStyle = glow; ctx.beginPath(); ctx.arc(dx, dy, r * 4, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = col; ctx.beginPath(); ctx.arc(dx, dy, r, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = "rgba(247,238,223,0.9)"; ctx.lineWidth = 1.5; ctx.stroke();
    ctx.strokeStyle = "#fcffa4"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(dx + Math.cos(q.head) * r * 1.4, dy + Math.sin(q.head) * r * 1.4); ctx.lineTo(dx + Math.cos(q.head) * r * 2.3, dy + Math.sin(q.head) * r * 2.3); ctx.stroke();
  }, (S) => {
    const pad = 14, s = Math.min((S.w - pad * 2) / RW, (S.h - pad * 2) / RH);
    view.current = { s, ox: (S.w - RW * s) / 2, oy: (S.h - RH * s) / 2 };
  });

  useEffect(() => { skin.current = SKINS[dog].hex; stage.current.redraw(); }, [dog, stage]);

  const aim = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (e.pointerType !== "mouse") return;
    const r = e.currentTarget.getBoundingClientRect(), v = view.current;
    st.current.aim = [(e.clientX - r.left - v.ox) / v.s, (e.clientY - r.top - v.oy) / v.s];
    st.current.aimT = stage.current.stage?.t ?? 0;
  };

  return (
    <div className="relative">
      <canvas
        ref={cv}
        onPointerMove={aim}
        className="block aspect-[4/3] w-full touch-pan-y"
        role="img"
        aria-label="Top view of a smoky room. The dog's heat camera sweeps a cone across it and flags a body-warm shape on the floor."
      />
      <Chip className="left-4">Top view</Chip>
    </div>
  );
}

/* ---------- (c) the real OpenCV contour on a thermal frame, drawn by the scroll ---------- */
type CV = { opencv: string; warm: { poly: [number, number][]; box: [number, number, number, number]; area_px: number; kp?: [number, number, number][] } };

// one ORB keypoint on the body: it pops in as the scroll passes its turn (cv.json warm.kp, OpenCV 5 ORB on this frame)
function Kp({ p, x, y, a, at }: { p: MotionValue<number>; x: number; y: number; a: number; at: number }) {
  const o = useTransform(p, [at, at + 0.06], [0, 1]);
  const r = (a * Math.PI) / 180;
  return (
    <motion.g style={{ opacity: o }}>
      <circle cx={x} cy={y} r={6.5} fill="none" stroke="#2bd46a" strokeWidth={1.5} opacity={0.75} />
      <line x1={x} y1={y} x2={x + Math.cos(r) * 9} y2={y + Math.sin(r) * 9} stroke="#2bd46a" strokeWidth={1.5} />
      <circle cx={x} cy={y} r={2.2} fill="#c9ffd9" />
    </motion.g>
  );
}

function Contour() {
  const wrap = useRef<HTMLDivElement>(null);
  const path = useRef<SVGPathElement>(null);
  const tip = useRef<SVGCircleElement>(null);
  const reduce = useReducedMotion();
  const [data, setData] = useState<CV | null>(null);
  useEffect(() => {
    let live = true;
    fetch("/data/cv.json").then((r) => r.json()).then((d: CV) => { if (live) setData(d); }).catch(() => {});
    return () => { live = false; };
  }, []);
  const { scrollYProgress } = useScroll({ target: wrap, offset: ["start 0.9", "center 0.62"] });
  const one = useMotionValue(1);
  const p = reduce ? one : scrollYProgress; // straight from scroll: Lenis already smooths it
  const len = useRef(0);
  const boxA = useTransform(p, [0.9, 1], [0, 1]);
  const fillA = useTransform(p, [0.92, 1], [0, 0.14]);

  useMotionValueEvent(p, "change", (v) => {
    const el = path.current, t = tip.current;
    if (!el || !t) return;
    const L = len.current || (len.current = el.getTotalLength()), pt = el.getPointAtLength(Math.min(1, v) * L);
    t.setAttribute("cx", String(pt.x)); t.setAttribute("cy", String(pt.y));
    t.style.opacity = v > 0.01 && v < 0.985 ? "1" : "0";
  });

  const d = data ? "M" + data.warm.poly.map(([x, y]) => `${x} ${y}`).join("L") + "Z" : "";
  const b = data?.warm.box;
  return (
    <div ref={wrap} className="relative aspect-[4/3] overflow-hidden bg-black">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src="/img/warm_th.jpg" alt="Thermal frame with a person-sized warm shape on the floor" width={640} height={480} className="absolute inset-0 h-full w-full object-cover" />
      {data && b && (
        <svg viewBox="0 0 640 480" className="absolute inset-0 h-full w-full" aria-hidden="true">
          <motion.path d={d} fill="#fcffa4" style={{ opacity: fillA }} />
          <motion.path d={d} fill="none" stroke="#f98c0a" strokeWidth={12} strokeLinejoin="round" strokeLinecap="round" style={{ pathLength: p, opacity: 0.35 }} />
          <motion.path ref={path} d={d} fill="none" stroke="#fcffa4" strokeWidth={3.5} strokeLinejoin="round" strokeLinecap="round" style={{ pathLength: p }} />
          <motion.rect x={b[0] - 10} y={b[1] - 10} width={b[2] + 20} height={b[3] + 20} rx={10} fill="none" stroke="#fcb040" strokeWidth={2.5} strokeDasharray="10 8" style={{ opacity: boxA }} />
          <circle ref={tip} r={7} fill="#fcffa4" style={{ opacity: 0, filter: "drop-shadow(0 0 6px #f98c0a)" }} />
          {data.warm.kp?.map(([x, y, a], i, all) => <Kp key={i} p={p} x={x} y={y} a={a} at={0.3 + (0.62 * i) / all.length} />)}
        </svg>
      )}
      <Chip className="left-4">Real OpenCV {data?.opencv.split(".")[0] ?? "5"} output</Chip>
      {data && (
        <Chip style={{ opacity: boxA }} className="right-4 tabular-nums">{data.warm.kp ? `${data.warm.kp.length} ORB points` : `${data.warm.area_px.toLocaleString("en")} px`}</Chip>
      )}
    </div>
  );
}
