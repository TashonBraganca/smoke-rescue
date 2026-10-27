"use client";
import { useRef, useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import RiseHead from "@/components/RiseHead";
import { assessVisibility, CLEAR_REF, shrink, type Visibility } from "@/lib/api";

const SAMPLES = [
  { src: "/img/room_rgb.jpg", label: "Robot camera, smoky room" },
  { src: "/img/robot_clear.jpg", label: "Robot camera, clear air" },
] as const;

// The three measures, named plainly, with which direction is better. No pass or fail: the numbers sit next to the
// robot's own clear-air frame so a visitor can compare.
const MEASURES = [
  { k: "laplacian_var", name: "Sharpness", how: "Laplacian variance", better: "higher is sharper", up: true, fmt: (v: number) => v.toFixed(1) },
  { k: "rms_contrast", name: "Contrast", how: "RMS, 0 to 255", better: "higher is clearer", up: true, fmt: (v: number) => v.toFixed(1) },
  { k: "dark_channel", name: "Haze", how: "dark-channel prior, 0 to 1", better: "lower is clearer", up: false, fmt: (v: number) => v.toFixed(3) },
] as const;

type State = { phase: "idle" } | { phase: "busy"; url: string } | { phase: "done"; url: string; v: Visibility; rt: number } | { phase: "error"; url?: string; msg: string };

export default function TryIt() {
  const reduce = useReducedMotion();
  const input = useRef<HTMLInputElement>(null);
  const [s, setS] = useState<State>({ phase: "idle" });
  const [drag, setDrag] = useState(false);

  async function run(blob: Blob) {
    let url: string | undefined;
    try {
      const small = await shrink(blob);
      url = URL.createObjectURL(small);
      setS({ phase: "busy", url });
      const { v, roundTripMs } = await assessVisibility(small);
      setS({ phase: "done", url, v, rt: roundTripMs });
    } catch (e) {
      setS({ phase: "error", url, msg: (e as Error).message });
    }
  }
  async function sample(src: string) {
    const r = await fetch(src);
    run(await r.blob());
  }
  const busy = s.phase === "busy";
  const shown = s.phase !== "idle" ? s.url : undefined;

  return (
    <section id="try" className="gutter py-[clamp(96px,14vw,176px)]" aria-label="Try OpenCV 5 on your own photo">
      <div className="mx-auto grid max-w-[1200px] gap-10">
        <RiseHead eyebrow="Try it, live on AWS" title="Run OpenCV 5 on your own photo." />
        <p className="max-w-[60ch] text-body text-sub">
          This is one of the OpenCV checks the robots&apos; AI runs on a camera frame (509 times in our recorded runs): can it trust what it sees, or is smoke hiding too much?
          Pick a photo, or one of the robot&apos;s own frames. It goes to our server on AWS, OpenCV 5 measures it, and the numbers come back here.
        </p>

        <div className="grid gap-6 lg:grid-cols-[1.1fr_1fr]">
          <div
            onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
            onDragLeave={() => setDrag(false)}
            onDrop={(e) => { e.preventDefault(); setDrag(false); const f = e.dataTransfer.files[0]; if (f && !busy) run(f); }}
            className={`relative grid min-h-[300px] place-items-center overflow-hidden rounded-3xl border bg-panel transition-colors ${drag ? "border-heat" : "border-line"}`}
          >
            {shown ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={shown} alt="The photo being measured" className={`h-full max-h-[460px] w-full object-contain transition-opacity ${busy ? "opacity-50" : ""}`} />
            ) : (
              <div className="grid justify-items-center gap-2 p-8 text-center">
                <span className="text-title3 font-extrabold text-ink">Drop a photo here</span>
                <span className="text-callout text-mute">or use the buttons below</span>
              </div>
            )}
            {busy && <span className="absolute bottom-4 left-4 rounded-full bg-stage/90 px-3 py-1.5 text-footnote font-bold text-amber">Measuring on AWS…</span>}
          </div>

          <div className="grid content-start gap-5">
            <div className="flex flex-wrap gap-3">
              <input ref={input} type="file" accept="image/*" className="sr-only" onChange={(e) => { const f = e.target.files?.[0]; if (f) run(f); e.target.value = ""; }} />
              <button type="button" disabled={busy} onClick={() => input.current?.click()}
                className="rounded-full bg-heat px-5 py-3 text-callout font-extrabold text-heat-ink transition-colors hover:bg-amber disabled:opacity-50">
                Choose a photo
              </button>
              {SAMPLES.map((x) => (
                <button key={x.src} type="button" disabled={busy} onClick={() => sample(x.src)}
                  className="rounded-full border border-line2 px-4 py-3 text-callout font-semibold text-sub transition-colors hover:border-heat hover:text-ink disabled:opacity-50">
                  {x.label}
                </button>
              ))}
            </div>

            <div className="grid gap-3" aria-live="polite">
              {MEASURES.map((m) => {
                const val = s.phase === "done" ? s.v[m.k] : null, ref = CLEAR_REF[m.k];
                const better = val === null ? null : m.up ? val >= ref : val <= ref;
                return (
                  <div key={m.k} className="grid gap-1 rounded-2xl border border-line bg-panel/60 px-5 py-4">
                    <div className="flex items-baseline justify-between gap-4">
                      <span className="text-callout font-extrabold text-ink">{m.name} <span className="font-semibold text-mute">({m.how})</span></span>
                      <AnimatePresence mode="wait">
                        <motion.span key={val ?? "none"} initial={reduce ? false : { opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }}
                          className="font-stencil text-title3 tabular-nums text-whiteheat">
                          {val === null ? "–" : m.fmt(val)}
                        </motion.span>
                      </AnimatePresence>
                    </div>
                    <span className="text-footnote text-mute">
                      {m.better[0].toUpperCase() + m.better.slice(1)}. Robot camera in clear air: {m.fmt(ref)}
                      {better !== null && <> · this photo is <span className={better ? "text-amber" : "text-coal"}>{better ? (m.up ? "higher" : "lower") : (m.up ? "lower" : "higher")}</span></>}
                    </span>
                  </div>
                );
              })}
            </div>

            {s.phase === "done" && (
              <p className="text-callout text-sub">
                Ran on AWS Graviton (arm64), OpenCV {s.v.opencv}: {s.v.ms.toFixed(1)} ms of OpenCV work, {Math.round(s.rt)} ms round trip from your browser.
              </p>
            )}
            {s.phase === "error" && <p role="alert" className="text-callout font-semibold text-coal">{s.msg}</p>}
            <p className="text-footnote text-mute">Your photo is shrunk to 1280 pixels in your browser and sent over HTTPS to our server to be measured.</p>
          </div>
        </div>
      </div>
    </section>
  );
}
