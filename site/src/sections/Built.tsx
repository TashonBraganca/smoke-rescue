"use client";
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { motion, useInView, useReducedMotion } from "motion/react";
import RiseHead from "@/components/RiseHead";

const LOOP = [
  { k: "01", t: "Sense", d: "Lidar, depth and heat cameras in our Unity simulator." },
  { k: "02", t: "See", d: "", cv: [["RGB-D odometry", "cv2.Odometry"], ["Warm-shape contours", "findContours"], ["Thermal colour", "applyColorMap"], ["Keypoints for SLAM", "ORB"]] },
  { k: "03", t: "Decide", d: "A language-model agent picks the next room or the next look." },
  { k: "04", t: "Act and share", d: "Reflexes back off from heat at once. It can spray water on a small fire. The radio merges maps." },
] as const;

const AWS = [
  ["AWS Amplify", "Hosts this site"],
  ["AWS Lambda on Graviton", "OpenCV 5 vision API, arm64"],
  ["Amazon ECS on Fargate", "The robots' agent service, arm64"],
] as const;

type Geo = { w: number; h: number; links: string[]; heads: string[]; back: string; backHead: string; flow: string; label: { x: number; y: number } | null };
const spring = { type: "spring" as const, stiffness: 80, damping: 20 };

export default function Built() {
  const reduce = useReducedMotion();
  const wrap = useRef<HTMLDivElement>(null);
  const boxes = useRef<(HTMLDivElement | null)[]>([]);
  const grid = useRef<HTMLDivElement>(null);
  const inView = useInView(wrap, { once: true, margin: "0px 0px -15% 0px" });
  const [geo, setGeo] = useState<Geo | null>(null);

  const measure = useCallback(() => {
    const el = wrap.current, gr = grid.current;
    if (!el || !gr || boxes.current.some((b) => !b)) return;
    // layout offsets, not getBoundingClientRect: the boxes start translated down while they rise in
    const W = { width: el.clientWidth, height: el.clientHeight };
    const r = boxes.current.map((b) => {
      const l = gr.offsetLeft + b!.offsetLeft, t = gr.offsetTop + b!.offsetTop, w = b!.offsetWidth, h = b!.offsetHeight;
      return { l, t, r: l + w, b: t + h, cx: l + w / 2, cy: t + h / 2 };
    });
    const row = r[1].t < r[0].b - 1, g = 7, R = 16;
    const links: string[] = [], heads: string[] = [];
    let back: string, backHead: string, flow: string, label: Geo["label"] = null;
    if (row) {
      const y = Math.min(...r.map((q) => q.t)) + 62;
      for (let i = 0; i < 3; i++) {
        const x1 = r[i].r + g, x2 = r[i + 1].l - g;
        links.push(`M${x1} ${y}H${x2}`); heads.push(`M${x2 - 6} ${y - 5}L${x2} ${y}L${x2 - 6} ${y + 5}`);
      }
      const bot = Math.max(...r.map((q) => q.b)), yb = bot + 44, x4 = r[3].cx, x1 = r[0].cx;
      back = `M${x4} ${r[3].b + g}V${yb - R}Q${x4} ${yb} ${x4 - R} ${yb}H${x1 + R}Q${x1} ${yb} ${x1} ${yb - R}V${r[0].b + g}`;
      backHead = `M${x1 - 5} ${r[0].b + g + 6}L${x1} ${r[0].b + g}L${x1 + 5} ${r[0].b + g + 6}`;
      flow = `M${x1} ${y}H${x4}V${yb - R}Q${x4} ${yb} ${x4 - R} ${yb}H${x1 + R}Q${x1} ${yb} ${x1} ${yb - R}Z`;
      label = { x: (x1 + x4) / 2, y: yb };
    } else {
      const x = r[0].l + 30;
      for (let i = 0; i < 3; i++) {
        const y1 = r[i].b + g, y2 = r[i + 1].t - g;
        links.push(`M${x} ${y1}V${y2}`); heads.push(`M${x - 5} ${y2 - 6}L${x} ${y2}L${x + 5} ${y2 - 6}`);
      }
      const xr = W.width - 12, y4 = r[3].t + 34, y0 = r[0].t + 34, xs = r[3].r + g;
      back = `M${xs} ${y4}H${xr - R}Q${xr} ${y4} ${xr} ${y4 - R}V${y0 + R}Q${xr} ${y0} ${xr - R} ${y0}H${r[0].r + g}`;
      backHead = `M${r[0].r + g + 6} ${y0 - 5}L${r[0].r + g} ${y0}L${r[0].r + g + 6} ${y0 + 5}`;
      flow = `M${x} ${y0}V${y4}H${xr - R}Q${xr} ${y4} ${xr} ${y4 - R}V${y0 + R}Q${xr} ${y0} ${xr - R} ${y0}Z`;
    }
    setGeo({ w: W.width, h: W.height, links, heads, back, backHead, flow, label });
  }, []);

  useLayoutEffect(() => { measure(); }, [measure]);
  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const ro = new ResizeObserver(() => measure());
    ro.observe(el);
    boxes.current.forEach((b) => b && ro.observe(b));
    document.fonts?.ready.then(measure);
    return () => ro.disconnect();
  }, [measure]);

  const show = inView || !!reduce;
  const draw = (delay: number) => ({
    initial: { pathLength: 0, opacity: 0 },
    animate: show ? { pathLength: 1, opacity: 1 } : undefined,
    transition: reduce ? { duration: 0 } : { pathLength: { duration: 0.7, ease: [0.25, 1, 0.5, 1] as const, delay }, opacity: { duration: 0.2, delay } },
  });
  const pop = (delay: number) => ({
    initial: { opacity: 0 },
    animate: show ? { opacity: 1 } : undefined,
    transition: reduce ? { duration: 0 } : { duration: 0.3, delay },
  });

  return (
    <section id="built" className="gutter py-[clamp(80px,11vw,140px)]">
      <div className="mx-auto max-w-[1200px]">
        <RiseHead eyebrow="How it's built" title="OpenCV 5 sees. An agent decides. AWS runs it." className="mb-[clamp(36px,5vw,64px)]" titleClass="max-w-[17ch]" />

        <div className="grid gap-5">
          <div className="rounded-[28px] border border-line bg-stage p-5 sm:p-8">
            <span className="text-footnote font-extrabold uppercase tracking-[0.16em] text-mute">On every dog</span>
            <div ref={wrap} className="relative mt-6 pr-9 lg:pr-0 lg:pb-[76px]">
              {geo && (
                <svg width={geo.w} height={geo.h} className="pointer-events-none absolute left-0 top-0 overflow-visible" aria-hidden="true" fill="none">
                  <defs>
                    <linearGradient id="built-back" x1="1" x2="0" y1="0" y2="0">
                      <stop offset="0" stopColor="#bc3754" /><stop offset="1" stopColor="#f98c0a" />
                    </linearGradient>
                  </defs>
                  {geo.links.map((d, i) => (
                    <motion.path key={"l" + i} d={d} stroke="#5e4a72" strokeWidth={2} strokeLinecap="round" {...draw(0.55 + i * 0.22)} />
                  ))}
                  {geo.heads.map((d, i) => (
                    <motion.path key={"h" + i} d={d} stroke="#8f7e95" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" {...pop(1.05 + i * 0.22)} />
                  ))}
                  <motion.path d={geo.back} stroke="url(#built-back)" strokeWidth={2} strokeLinecap="round" {...draw(1.3)} />
                  <motion.path d={geo.backHead} stroke="#f98c0a" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" {...pop(2.0)} />
                  {show && !reduce && (
                    <g style={{ filter: "drop-shadow(0 0 6px #f98c0a)" }}>
                      {[0, 1].map((i) => (
                        <motion.circle key={i} r={3.5} fill="#fcffa4" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 2.2 }}>
                          <animateMotion dur="7s" begin={`${-i * 3.5}s`} repeatCount="indefinite" path={geo.flow} />
                        </motion.circle>
                      ))}
                    </g>
                  )}
                </svg>
              )}

              <div ref={grid} className="relative grid gap-11 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.45fr)_minmax(0,1fr)_minmax(0,1fr)] lg:gap-12">
                {LOOP.map((b, i) => {
                  const cv = "cv" in b;
                  return (
                    <motion.div
                      key={b.k}
                      ref={(el) => { boxes.current[i] = el; }}
                      initial={{ opacity: 0, y: 36 }}
                      animate={show ? { opacity: 1, y: 0 } : undefined}
                      transition={reduce ? { duration: 0 } : { ...spring, delay: i * 0.1 }}
                      className={`grid min-w-0 content-start gap-3 rounded-[20px] border bg-panel p-5 ${cv ? "border-heat/55 shadow-[0_24px_70px_-28px_rgba(249,140,10,0.55)]" : "border-line2"}`}
                    >
                      <span className="text-[11px] font-extrabold tracking-[0.16em] text-mute">{b.k}</span>
                      <h3 className="flex flex-wrap items-center gap-x-2.5 gap-y-1 text-title3 font-extrabold text-ink">
                        {b.t}
                        {cv && <span className="rounded-full bg-heat px-2.5 py-1 text-[11px] font-extrabold uppercase tracking-[0.1em] text-heat-ink">with OpenCV 5</span>}
                      </h3>
                      {cv ? (
                        <ul className="grid">
                          {b.cv.map(([what, fn]) => (
                            <li key={fn} className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-0.5 border-t border-line py-2 first:border-t-0 first:pt-0">
                              <span className="text-[14px] leading-snug text-sub">{what}</span>
                              <code className="font-mono text-[12.5px] font-bold text-amber">{fn}</code>
                            </li>
                          ))}
                        </ul>
                      ) : (
                        <p className="text-[15px] leading-snug text-sub">{b.d}</p>
                      )}
                    </motion.div>
                  );
                })}
              </div>

              {geo?.label && (
                <motion.span
                  {...pop(1.7)}
                  className="absolute hidden -translate-x-1/2 -translate-y-1/2 whitespace-nowrap bg-stage px-4 text-[14px] font-semibold text-sub lg:block"
                  style={{ left: geo.label.x, top: geo.label.y }}
                >
                  What it sees changes what it does next
                </motion.span>
              )}
            </div>
            <motion.p {...pop(1.6)} className="mt-6 flex items-center gap-2.5 text-[14px] font-semibold text-sub lg:hidden">
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true"><path d="M13 8a5 5 0 1 1-1.6-3.7M13 2.5v2.8h-2.8" stroke="#f98c0a" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" /></svg>
              What it sees changes what it does next
            </motion.p>
          </div>

          <div className="rounded-[28px] border border-line bg-stage p-5 sm:p-8">
            <span className="text-footnote font-extrabold uppercase tracking-[0.16em] text-mute">On AWS</span>
            <div className="mt-6 grid gap-3 sm:grid-cols-3">
              {AWS.map(([name, role], i) => (
                <motion.div
                  key={name}
                  initial={{ opacity: 0, y: 28 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true, margin: "-6% 0px" }}
                  transition={reduce ? { duration: 0 } : { ...spring, delay: 0.1 + i * 0.07 }}
                  className="grid min-w-0 content-start gap-1.5 rounded-[18px] border border-line2 bg-panel p-5"
                >
                  <span className="text-[16px] font-extrabold leading-tight text-ink">{name}</span>
                  <span className="text-[14px] leading-snug text-sub">{role}</span>
                </motion.div>
              ))}
            </div>
          </div>

        </div>
      </div>
    </section>
  );
}
