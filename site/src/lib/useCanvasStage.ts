"use client";
import { useEffect, useLayoutEffect, useRef, type RefObject } from "react";

// One 2D canvas stage: sharp at devicePixelRatio (max 2), refit on resize, paused while off screen.
// The frame callback gets dt in seconds (0 for a still redraw). Under reduced motion the loop never
// runs; the frame is drawn once with dt = 0 and again whenever redraw() is called.
export type Stage = { w: number; h: number; t: number; dpr: number; reduce: boolean; font: string };
export type Frame = (ctx: CanvasRenderingContext2D, s: Stage, dt: number) => void;

export function useCanvasStage(
  ref: RefObject<HTMLCanvasElement | null>,
  frame: Frame,
  onFit?: (s: Stage) => void,
) {
  const frameRef = useRef(frame);
  const fitRef = useRef(onFit);
  const api = useRef({ redraw: () => {}, stage: null as Stage | null });
  useLayoutEffect(() => { frameRef.current = frame; fitRef.current = onFit; });

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
    const S: Stage = { w: 0, h: 0, t: 0, dpr: 1, reduce, font: getComputedStyle(cv).fontFamily || "sans-serif" };
    api.current.stage = S;
    let raf = 0, last = 0, vis = false;
    const draw = (dt: number) => { if (S.w > 0 && S.h > 0) frameRef.current(ctx, S, dt); };
    const fit = () => {
      const r = cv.getBoundingClientRect(), d = Math.min(window.matchMedia("(pointer: coarse)").matches ? 1.5 : 2, window.devicePixelRatio || 1);
      S.w = r.width; S.h = r.height; S.dpr = d;
      cv.width = Math.max(1, Math.round(S.w * d)); cv.height = Math.max(1, Math.round(S.h * d));
      ctx.setTransform(d, 0, 0, d, 0, 0);
      fitRef.current?.(S);
      draw(0);
    };
    const loop = (now: number) => {
      const dt = Math.max(0, Math.min(0.05, (now - last) / 1000));
      last = now; S.t += dt; draw(dt);
      if (vis) raf = requestAnimationFrame(loop);
    };
    api.current.redraw = () => draw(0);
    const ro = new ResizeObserver(fit); ro.observe(cv);
    const io = new IntersectionObserver((es) => {
      vis = es[es.length - 1].isIntersecting; // several entries can arrive at once; the last one is the current state
      cancelAnimationFrame(raf);
      if (vis && !reduce) { last = performance.now(); raf = requestAnimationFrame(loop); }
      else if (vis) draw(0);
    }, { rootMargin: "80px 0px" });
    io.observe(cv);
    document.fonts?.ready.then(() => { S.font = getComputedStyle(cv).fontFamily || S.font; draw(0); });
    return () => { cancelAnimationFrame(raf); ro.disconnect(); io.disconnect(); };
  }, [ref]);

  return api;
}

// Inferno colour map, the same scale OpenCV's COLORMAP_INFERNO uses for our thermal frames.
const INF: [number, number, number, number][] = [
  [0, 0, 0, 4], [0.13, 27, 12, 65], [0.25, 74, 12, 107], [0.38, 120, 28, 109], [0.5, 165, 44, 96],
  [0.63, 207, 68, 70], [0.75, 237, 105, 37], [0.88, 251, 155, 6], [0.95, 247, 209, 61], [1, 252, 255, 164],
];
export function inferno(v: number): [number, number, number] {
  v = Math.max(0, Math.min(1, v));
  for (let i = 1; i < INF.length; i++) {
    if (v <= INF[i][0]) {
      const a = INF[i - 1], b = INF[i], t = (v - a[0]) / (b[0] - a[0]);
      return [a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t, a[3] + (b[3] - a[3]) * t];
    }
  }
  return [252, 255, 164];
}

export function rrect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  ctx.beginPath();
  if (ctx.roundRect) ctx.roundRect(x, y, w, h, r);
  else ctx.rect(x, y, w, h);
}

export function hexA(hex: string, a: number) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}
