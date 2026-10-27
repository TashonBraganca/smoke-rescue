"use client";
import { Canvas } from "@react-three/fiber";
import { Suspense, useEffect, useRef, useState } from "react";
import { useScroll, useMotionValueEvent, useTransform, useInView, motion, type MotionValue } from "motion/react";
import * as THREE from "three";
import StoryScene from "@/three/StoryScene";
import { useAdaptiveDpr, useLowPower } from "@/three/quality";
import WordRise from "@/components/WordRise";
import { useMission } from "@/lib/store";

// Beats of the locked story, by scroll progress through the section (k = small label above the headline).
const BEATS: { at: number; k: string; h: string; s: string }[] = [
  { at: 0.035, k: "Why", h: "In a fire, crews search blind.", s: "Thick smoke, no map, and every minute counts." },
  { at: 0.1, k: "The dog", h: "Send in the first one.", s: "A robot dog walks in before the crew." },
  { at: 0.17, k: "Lidar", h: "It maps the walls with lidar.", s: "720 laser beams, ten times a second." },
  { at: 0.29, k: "SLAM", h: "It always knows where it is.", s: "Each new scan lines up with the map so far." },
  { at: 0.38, k: "Cameras", h: "Then it looks.", s: "A colour camera and a heat camera, side by side." },
  { at: 0.49, k: "ORB-SLAM3 · Colour camera", h: "Smoke hides almost everything.", s: "164 ORB corners: the features ORB-SLAM3 tracks to know where it is." },
  { at: 0.565, k: "Heat camera", h: "Heat sees right through it.", s: "Same room. Same moment." },
  { at: 0.635, k: "Reflex", h: "Too hot? It backs off on its own.", s: "Its reflexes act at once, without waiting for the AI." },
  { at: 0.715, k: "OpenCV 5", h: "A person. Right there.", s: "OpenCV traces the warm shape." },
  { at: 0.815, k: "Team", h: "Another dog joins in.", s: "It maps the hall on its way in." },
  { at: 0.885, k: "Radio", h: "Now they share one map.", s: "Over the radio, every 2 seconds." },
  { at: 0.945, k: "The crew", h: "The crew goes in knowing the way.", s: "Who is inside, where, and the safe path to them." },
];
const beatAt = (p: number) => { let b = -1; for (let i = 0; i < BEATS.length; i++) if (p >= BEATS[i].at) b = i; return b; };
// Section progress -> the 3D scene's own timeline (its camera keys and effects are written against this).
const SEC = [0, 0.035, 0.1, 0.17, 0.29, 0.38, 0.48, 0.8, 0.83, 0.88, 0.94, 1];
const SCN = [0, 0, 0.05, 0.2, 0.33, 0.44, 0.66, 0.86, 0.89, 0.95, 0.975, 1];
const EYE_IN = 0.475, EYE_ON = 0.49, EYE_OFF = 0.795, EYE_OUT = 0.815;

type CV = { orb_room: { kp: number[][]; n: number }; warm: { poly: number[][]; box: number[] }; hot: { box: number[] } };
const sm = (a: number, b: number, x: number) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };

// Full-screen "through its eyes" view of ONE room: the smoky colour frame with real ORB corners, a wipe to the
// heat frame of the same room and camera, a push in on the hot doorway (reflex), then on the person, whose real
// OpenCV contour traces itself. Coordinates in cv.json are 640x480 image pixels.
function EyeView({ progress, active }: { progress: MotionValue<number>; active: boolean }) {
  const cv = useRef<HTMLCanvasElement>(null);
  const [data, setData] = useState<CV | null>(null);
  const rgb = useRef<HTMLImageElement | null>(null), warm = useRef<HTMLImageElement | null>(null);
  useEffect(() => {
    fetch("/data/cv.json").then((r) => (r.ok ? r.json() : null)).then(setData).catch(() => setData(null));
    const a = new Image(); a.src = "/img/room_rgb.jpg"; rgb.current = a;
    const b = new Image(); b.src = "/img/warm_th.jpg"; warm.current = b;
  }, []);
  useEffect(() => {
    let raf = 0, lastKey = "";
    if (!active) return;
    const draw = () => {
      raf = requestAnimationFrame(draw);
      const c = cv.current, p = progress.get();
      if (!c || !data) return;
      const key = `${p.toFixed(4)}|${c.clientWidth}x${c.clientHeight}|${rgb.current?.complete}${warm.current?.complete}|${p > 0.635 && p < 0.72 ? Math.floor(performance.now() / 50) % 2 : 0}`;   // the time term only animates the hot-zone pulse
      if (key === lastKey) return; lastKey = key;
      c.style.opacity = String(sm(EYE_IN, EYE_ON, p) * (1 - sm(EYE_OFF, EYE_OUT, p)));
      if (p < EYE_IN || p > EYE_OUT) return;
      const dpr = Math.min(window.matchMedia("(pointer: coarse)").matches ? 1.5 : 2, devicePixelRatio || 1), W = c.clientWidth, H = c.clientHeight;
      if (c.width !== Math.round(W * dpr) || c.height !== Math.round(H * dpr)) { c.width = Math.round(W * dpr); c.height = Math.round(H * dpr); }
      const ctx = c.getContext("2d")!; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const wide = W > 900, s0 = Math.max(W / 640, H / 480) * (wide ? 1.05 : 1);
      // camera moves inside the frame: push in on the doorway, back off, then push in on the person
      const [hx, hy, hw, hh] = data.hot.box, [bx, by, bw, bh] = data.warm.box;
      const hot = sm(0.64, 0.675, p) * (1 - sm(0.69, 0.715, p)), per = sm(0.72, 0.76, p);
      const z = 1 + 0.45 * hot + (wide ? 0.25 : 0) * per;
      const fx = hot > 0 ? hx + hw / 2 : bx + bw / 2, fy = hot > 0 ? hy + hh / 2 : by + bh / 2, amt = Math.max(hot, per);
      const S = s0 * z, base = { x: (W - 640 * s0) / 2 + (wide ? W * 0.1 : 0), y: (H - 480 * s0) / 2 };
      const tx = wide ? W * 0.64 : W * 0.5, ty = wide ? H * 0.47 : H * 0.36;
      const px = base.x + fx * s0 + (tx - (base.x + fx * s0)) * amt, py = base.y + fy * s0 + (ty - (base.y + fy * s0)) * amt;
      // keep the frame covering the screen: never slide an edge into view
      const ox = Math.min(0, Math.max(W - 640 * S, px - fx * S)), oy = Math.min(0, Math.max(H - 480 * S, py - fy * S));
      const X = (x: number) => ox + x * S, Y = (y: number) => oy + y * S;
      const wipe = sm(0.565, 0.6, p);
      ctx.fillStyle = "#000"; ctx.fillRect(0, 0, W, H);
      if (rgb.current?.complete) ctx.drawImage(rgb.current, ox, oy, 640 * S, 480 * S);
      if (warm.current?.complete && wipe > 0) {
        const cut = W * wipe;
        ctx.save(); ctx.beginPath(); ctx.rect(0, 0, cut, H); ctx.clip(); ctx.drawImage(warm.current, ox, oy, 640 * S, 480 * S); ctx.restore();
        if (wipe < 1) { ctx.fillStyle = "#fcffa4"; ctx.fillRect(cut - 1.5, 0, 3, H); }
      }
      // ORB corners on the colour frame, strongest first; they fade as the heat view takes over. The frame dims
      // under them, each corner gets a dark halo, a ring, a centre dot and a tick along its ORB orientation.
      const kp = data.orb_room.kp, n = Math.round(kp.length * sm(0.5, 0.555, p)), ka = 1 - wipe;
      if (ka > 0) {
        ctx.globalAlpha = ka * 0.35 * sm(0.49, 0.51, p); ctx.fillStyle = "#000"; ctx.fillRect(0, 0, W, H);
        const r = Math.max(6, Math.min(10, 3 * S));
        for (let i = 0; i < n; i++) {
          const [x, y, ang] = kp[i], fresh = i > n - 20, cx = X(x), cy = Y(y), rr = fresh ? r * 1.35 : r;
          const t = ((ang ?? 0) * Math.PI) / 180, col = fresh ? "#fcffa4" : "#f98c0a";
          ctx.globalAlpha = ka;
          ctx.strokeStyle = "rgba(11,6,18,.75)"; ctx.lineWidth = 5; ctx.beginPath(); ctx.arc(cx, cy, rr, 0, 7); ctx.stroke();
          if (fresh) { ctx.shadowColor = "#fcffa4"; ctx.shadowBlur = 12; }
          ctx.strokeStyle = col; ctx.lineWidth = fresh ? 2.6 : 2.2; ctx.beginPath(); ctx.arc(cx, cy, rr, 0, 7);
          ctx.moveTo(cx, cy); ctx.lineTo(cx + Math.cos(t) * (rr + 5), cy + Math.sin(t) * (rr + 5)); ctx.stroke();
          ctx.shadowBlur = 0; ctx.fillStyle = col; ctx.beginPath(); ctx.arc(cx, cy, 1.8, 0, 7); ctx.fill();
        }
        // name the points where they are, on the image side away from the headline scrim
        if (n > 0) { ctx.globalAlpha = ka * sm(0.5, 0.52, p); tag(ctx, `ORB-SLAM3 keypoints · ${n}`, wide ? W * 0.56 : 16, wide ? H * 0.2 : H * 0.16, "#fcffa4"); }
      }
      ctx.globalAlpha = 1;
      // reflex: the hot doorway is boxed and pulses while the view backs away from it
      if (hot > 0.05) {
        const pulse = 0.65 + 0.35 * Math.sin(performance.now() / 160);
        ctx.globalAlpha = Math.min(1, hot * 1.6);
        ctx.strokeStyle = `rgba(255,74,61,${pulse})`; ctx.lineWidth = 3; ctx.strokeRect(X(hx) - 6, Y(hy) - 6, hw * S + 12, hh * S + 12);
        tag(ctx, "Too hot. Backing off.", X(hx) - 6, Y(hy + hh) + 14, "#ff4a3d");
        ctx.globalAlpha = 1;
      }
      // the person: real OpenCV contour traces itself, then the box and a label
      const tr = sm(0.735, 0.78, p);
      if (tr > 0) {
        const poly = data.warm.poly, k = Math.max(2, Math.round(poly.length * tr));
        ctx.strokeStyle = "#fcffa4"; ctx.lineWidth = 3; ctx.shadowColor = "#f98c0a"; ctx.shadowBlur = 14; ctx.beginPath();
        for (let i = 0; i < k; i++) { const [x, y] = poly[i]; if (i) ctx.lineTo(X(x), Y(y)); else ctx.moveTo(X(x), Y(y)); }
        if (k === poly.length) ctx.closePath(); ctx.stroke(); ctx.shadowBlur = 0;
        if (tr > 0.95) { ctx.setLineDash([10, 8]); ctx.strokeStyle = "#f98c0a"; ctx.lineWidth = 2; ctx.strokeRect(X(bx) - 8, Y(by) - 8, bw * S + 16, bh * S + 16); ctx.setLineDash([]);
          tag(ctx, "Person, body heat", X(bx) - 8, Y(by) - 22, "#f98c0a"); }
      }
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [data, progress, active]);
  return <canvas ref={cv} aria-hidden="true" className="pointer-events-none absolute inset-0 z-[1] h-full w-full opacity-0" />;
}
function tag(ctx: CanvasRenderingContext2D, text: string, x: number, y: number, color: string) {
  ctx.font = "800 13px Manrope, system-ui, sans-serif"; const w = ctx.measureText(text).width + 20;
  ctx.fillStyle = "rgba(11,6,18,.85)"; ctx.beginPath(); ctx.roundRect(x, y - 13, w, 26, 13); ctx.fill();
  ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.stroke(); ctx.fillStyle = "#f7eedf"; ctx.fillText(text, x + 10, y + 5);
}

export default function Story() {
  const ref = useRef<HTMLElement>(null);
  const { scrollYProgress } = useScroll({ target: ref, offset: ["start start", "end end"] });
  const [beat, setBeat] = useState(-1);
  const [ready, setReady] = useState(false);
  useEffect(() => { const id = setTimeout(() => setReady(true), 250); return () => clearTimeout(id); }, []);
  // while the full-screen camera view is fully opaque (opacity 1 between 0.66 and 0.86) the 3D scene behind it is hidden: stop rendering it
  const [covered, setCovered] = useState(false);
  useMotionValueEvent(scrollYProgress, "change", (p) => { setBeat(beatAt(p)); setCovered(p > EYE_ON + 0.003 && p < EYE_OFF - 0.003); });
  const sceneP = useTransform(scrollYProgress, SEC, SCN);
  const { dog } = useMission();
  // the backdrop photo is pre-blurred and pre-darkened; scrolling only fades a dark layer over it (compositor-only, no filter re-paint)
  const bgShade = useTransform(scrollYProgress, [0, 0.035, 0.1, 0.18], [0, 0.3, 0.5, 0.84]);
  const low = useLowPower();
  const [dpr, monitor] = useAdaptiveDpr(low);
  const heroOpacity = useTransform(scrollYProgress, [0, 0.03], [1, 0]);
  const heroY = useTransform(scrollYProgress, [0, 0.035], [0, -40]);
  const heroPointer = useTransform(heroOpacity, (o) => (o < 0.05 ? "none" : "auto"));
  const inView = useInView(ref);

  return (
    <section ref={ref} id="top" className="relative h-[600vh]" aria-label="The robot dog, shown part by part as you scroll">
      <div className="sticky top-0 h-svh min-h-[600px] overflow-hidden">
        <div aria-hidden className="absolute inset-0 -z-0 bg-cover bg-[center_62%]" style={{ backgroundImage: "url(/img/hero_bg_soft.jpg)" }} />
        <motion.div aria-hidden className="absolute inset-0 -z-0 bg-bg" style={{ opacity: bgShade }} />
        <div className="absolute inset-0">
          <Canvas
            shadows={low ? false : "percentage"} dpr={dpr} frameloop={inView && !covered ? "always" : "never"} gl={{ antialias: low, alpha: true, powerPreference: "high-performance", toneMapping: THREE.ACESFilmicToneMapping, toneMappingExposure: 0.95 }}
            camera={{ fov: 26, position: [0.55, 1.15, 6.9], near: 0.02, far: 100 }}
          >
            {monitor}
            <Suspense fallback={null}><StoryScene progress={sceneP} skin={dog} low={low} /></Suspense>
          </Canvas>
        </div>
        <EyeView progress={scrollYProgress} active={inView} />
        {/* edges fade into the page */}
        <div aria-hidden className="pointer-events-none absolute inset-0 z-[2] bg-[radial-gradient(ellipse_80%_75%_at_58%_50%,transparent_50%,rgba(11,6,18,.55)_78%,#0b0612_100%),linear-gradient(90deg,rgba(11,6,18,.88)_0%,rgba(11,6,18,.45)_30%,transparent_52%),linear-gradient(180deg,rgba(11,6,18,.6)_0%,transparent_15%,transparent_78%,#0b0612_100%)]" />

        <div className="gutter absolute inset-0 z-[3]">
          <div className="relative mx-auto grid h-full max-w-[1200px] items-center">
            <motion.div style={{ opacity: heroOpacity, y: heroY, pointerEvents: heroPointer }} className="grid max-w-[620px] gap-6 pt-24 max-md:self-start max-md:pt-28">
              <span className="text-xs font-extrabold uppercase tracking-[.2em] text-heat">Search robots for house fires</span>
              <WordRise as="h1" text="Know the house before you walk in." show={ready} className="text-display font-extrabold" stagger={0.07} />
              <p className="max-w-[46ch] text-lg text-sub">Robot dogs map a burning house, find the people inside, and tell the crew where to go.</p>
              <div className="flex flex-wrap gap-3">
                <a href="#mission" className="rounded-full bg-heat px-6 py-3.5 font-extrabold text-heat-ink transition-[background,transform] duration-300 hover:-translate-y-0.5 hover:bg-amber">Choose a mission</a>
                <a href="#built" className="rounded-full border border-white/20 bg-white/5 px-6 py-3.5 font-extrabold text-ink transition-colors hover:bg-white/10">How it&apos;s built</a>
              </div>
            </motion.div>
            {BEATS.map((b, i) => (
              <div key={i} className="pointer-events-none absolute bottom-[clamp(56px,12vh,120px)] left-0 grid max-w-[640px] gap-3" aria-hidden={beat !== i}>
                <motion.span className="text-xs font-extrabold uppercase tracking-[.2em] text-heat" animate={{ opacity: beat === i ? 1 : 0, y: beat === i ? 0 : 8 }} transition={{ duration: 0.4 }}>{b.k}</motion.span>
                <WordRise as="h2" text={b.h} show={beat === i} className="text-title1 font-extrabold" stagger={0.06} />
                {b.s && <WordRise as="p" text={b.s} show={beat === i} delay={0.18} stagger={0.025} className="text-lg font-semibold text-sub" />}
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
