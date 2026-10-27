"use client";
import { useEffect, useRef, useState } from "react";
import { useInView, useReducedMotion } from "motion/react";
import RiseHead from "@/components/RiseHead";

// The visual SLAM moment from a recorded run, told with pictures. Two loops cut from slam_moment.mp4 (seed 934):
// the camera pair with ORB matches drawing in (clip 0 to 6 s), and the two maps snapping together (clip 6 to 14 s).
// In the map loop the launch-pad guess holds to 2 s, the link applies from 2 to 5 s, then holds; the counters
// follow the clip's own status line (72% to 97% walls agree, 2.39 to 0.23 degrees heading error).
const lerp = (a: number, b: number, p: number) => a + (b - a) * p;

function Loop({ src, poster, play, onTime, label, className = "" }: { src: string; poster: string; play: boolean; onTime?: (t: number) => void; label: string; className?: string }) {
  const v = useRef<HTMLVideoElement>(null);
  useEffect(() => { const el = v.current; if (!el) return; if (play) el.play().catch(() => {}); else el.pause(); }, [play]);
  return <video ref={v} src={src} poster={poster} muted loop playsInline preload="metadata" aria-label={label}
    onTimeUpdate={onTime ? (e) => onTime(e.currentTarget.currentTime) : undefined} className={`block w-full ${className}`} />;
}

const chip = "pointer-events-none absolute rounded-full px-3 py-1 text-[12px] font-extrabold uppercase tracking-[0.12em]";

export function SlamMoment() {
  const reduce = useReducedMotion();
  const box = useRef<HTMLElement>(null);
  const inView = useInView(box, { margin: "0px 0px -15% 0px" });
  const [p, setP] = useState(reduce ? 1 : 0);
  const play = inView && !reduce;
  const pct = Math.round(lerp(72, 97, p)), deg = lerp(2.39, 0.23, p).toFixed(2);

  return (
    <section id="slam" ref={box} className="gutter py-[clamp(80px,11vw,140px)]" aria-label="Visual SLAM: two robots recognise the same place and merge their maps">
      <div className="mx-auto grid max-w-[1200px] gap-6">
        <RiseHead eyebrow="Visual SLAM" title="Two robots, one map." />

        <div className="relative overflow-hidden rounded-3xl border border-line bg-black">
          <Loop src="/evidence/slam_pair_loop.mp4" poster="/evidence/slam_pair_poster.jpg" play={play} label="Robot B's camera and robot A's camera see the same hall; green lines join 24 matched ORB features" />
          <span className={`${chip} left-3 top-3 bg-[#1d9bf0]/90 text-white`}>Robot B</span>
          <span className={`${chip} left-[calc(50.8%+12px)] top-3 bg-amber/90 text-heat-ink`}>Robot A</span>
          <span className={`${chip} bottom-3 left-1/2 -translate-x-1/2 bg-[#2bd46a]/90 text-[#062010]`}>24 ORB matches</span>
        </div>

        <div className="grid gap-6 md:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)]">
          <div className="relative overflow-hidden rounded-3xl border border-line bg-[#14161a]">
            <Loop src="/evidence/slam_map_loop.mp4" poster="/evidence/slam_map_poster.jpg" play={play}
              onTime={(t) => setP(Math.min(1, Math.max(0, (t - 2) / 3)))} label="Robot B's map in blue moves onto robot A's map in amber until the walls agree and turn white" />
            <span className="pointer-events-none absolute left-3 top-3 flex gap-3 rounded-full bg-black/70 px-3 py-1.5 text-[12px] font-bold text-sub">
              <span className="flex items-center gap-1.5"><i className="size-2 rounded-full bg-[#1d9bf0]" />B</span>
              <span className="flex items-center gap-1.5"><i className="size-2 rounded-full bg-amber" />A</span>
              <span className="flex items-center gap-1.5"><i className="size-2 rounded-full bg-white" />agree</span>
            </span>
          </div>
          <div className="grid content-center gap-4">
            <div className="grid gap-1 rounded-3xl border border-line bg-panel/60 p-6">
              <span className={`font-stencil text-[clamp(3rem,7vw,5.5rem)] leading-none tabular-nums transition-colors duration-500 ${p >= 1 ? "text-[#2bd46a]" : "text-whiteheat"}`}>{pct}%</span>
              <span className="text-callout text-sub">walls agree</span>
            </div>
            <div className="grid gap-1 rounded-3xl border border-line bg-panel/60 p-6">
              <span className={`font-stencil text-[clamp(3rem,7vw,5.5rem)] leading-none tabular-nums transition-colors duration-500 ${p >= 1 ? "text-[#2bd46a]" : "text-whiteheat"}`}>{deg}°</span>
              <span className="text-callout text-sub">heading error</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
