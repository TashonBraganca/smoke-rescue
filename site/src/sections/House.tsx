"use client";
import dynamic from "next/dynamic";
import { AnimatePresence, motion } from "motion/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import WordRise from "@/components/WordRise";
import { HOUSES, loadPlan, type HouseId, type Plan } from "@/lib/plans";
import type { SkinName } from "@/lib/skins";
import { missionStore, useMission } from "@/lib/store";
import { buildLayout, type Layout } from "@/three/houseLayout";
import { RoomLabels } from "@/three/HouseModel";
import type { WalkEvent } from "@/three/HouseScene";

const HouseScene = dynamic(() => import("@/three/HouseScene"), { ssr: false });
const EASE = [0.2, 0.9, 0.25, 1] as const;

/** What the dogs' agent would say as a room is first entered, in plain words, taken from the floor plan. */
function caption(layout: Layout, e: WalkEvent): string | null {
  if (e.type === "done") return e.n === e.total ? "Every room checked." : `${e.n} of ${e.total} rooms checked.`;
  if (e.type !== "room") return null;
  const room = layout.byId[e.room].name;
  if (e.fire) return `${room}: heat ahead. Going in slowly.`;
  if (!e.next) return e.n === e.total ? `${room}: clear. That was the last room.` : `${room}: clear.`;
  const next = layout.byId[e.next].name;
  return `${room}: clear. Next, ${next === room ? "another" : "the"} ${next.toLowerCase()}.`;
}

// "How they walk in": one screen that plays on its own like a short film when it scrolls into view. Two dogs walk in
// through the front door and search every room while a caption shows what their agent decides; then it loops.
export default function House() {
  const { dog, house } = useMission();
  const ref = useRef<HTMLElement>(null);
  const labels = useRef<(HTMLElement | null)[]>([]);
  const [plans, setPlans] = useState<Partial<Record<HouseId, Plan>>>({});
  const [failed, setFailed] = useState(false);
  const [active, setActive] = useState(false);
  const [ready, setReady] = useState(false);
  const [count, setCount] = useState({ n: 0, total: 0 });
  const [line, setLine] = useState<{ id: number; text: string } | null>(null);
  const [dark, setDark] = useState(false);

  useEffect(() => {
    let live = true;
    HOUSES.forEach((h) => loadPlan(h.id).then((p) => live && setPlans((s) => ({ ...s, [h.id]: p })), () => live && setFailed(true)));
    return () => { live = false; };
  }, []);
  useEffect(() => {
    const el = ref.current; if (!el) return;
    // play only while a real part of the section is on screen; frameloop "never" otherwise
    const io = new IntersectionObserver((es) => setActive(es[es.length - 1].isIntersecting), { threshold: 0.2 });
    io.observe(el); return () => io.disconnect();
  }, []);

  const plan = plans[house];
  const layout = useMemo(() => (plan ? buildLayout(plan) : null), [plan]);
  const other: SkinName = dog === "Ash" ? "Ember" : "Ash";
  const skins = useMemo<[SkinName, SkinName]>(() => [dog, other], [dog, other]);
  const onReady = useCallback(() => setReady(true), []);
  const onEvent = useCallback((e: WalkEvent) => {
    if (!layout) return;
    if (e.type === "start") { setDark(false); setCount({ n: 0, total: layout.rooms.length }); setLine(null); return; }
    if (e.type === "fade") { setDark(true); return; }
    if (e.type === "room") setCount({ n: e.n, total: e.total });
    if (e.type === "done") setCount({ n: e.n, total: e.total });
    const text = caption(layout, e);
    if (text) setLine((l) => ({ id: (l?.id ?? 0) + 1, text }));
  }, [layout]);

  return (
    <section id="house" ref={ref} aria-label="How they walk in" className="relative h-[100svh] min-h-[600px] overflow-hidden bg-stage">
      <div className={`absolute inset-0 transition-opacity duration-1000 ease-[var(--ease-spring)] ${ready ? "opacity-100" : "opacity-0"}`}>
        {layout && (
          <>
            <HouseScene layout={layout} skins={skins} active={active} onEvent={onEvent} onReady={onReady} labels={labels} />
            <RoomLabels layout={layout} els={labels} />
          </>
        )}
      </div>
      {failed && !layout && (
        <p className="absolute inset-0 grid place-items-center px-6 text-center text-body text-sub">The house plan did not load. Reload the page to try again.</p>
      )}
      {/* a quick dip to dark when the house changes and before each replay, so the cut never pops */}
      <AnimatePresence>
        <motion.div key={house} initial={{ opacity: 1 }} animate={{ opacity: 0 }} transition={{ duration: 0.6, ease: EASE }} className="pointer-events-none absolute inset-0 bg-stage" />
      </AnimatePresence>
      <div className={`pointer-events-none absolute inset-0 bg-stage transition-opacity duration-500 ${dark ? "opacity-100" : "opacity-0"}`} />
      <div className="pointer-events-none absolute inset-x-0 top-0 h-64 bg-gradient-to-b from-bg via-bg/70 to-transparent" />
      <div className="pointer-events-none absolute inset-x-0 bottom-0 h-64 bg-gradient-to-t from-bg via-bg/75 to-transparent" />

      <div className="gutter pointer-events-none absolute inset-x-0 top-[calc(env(safe-area-inset-top,0px)+88px)] text-center">
        <WordRise as="h2" text="How they walk in." className="text-title1 font-extrabold text-ink" />
        <p className="mx-auto mt-3 max-w-[34ch] text-body text-sub">Two dogs, no driver. Each one picks the next room on its own.</p>
      </div>

      <div className="gutter absolute inset-x-0 bottom-[calc(env(safe-area-inset-bottom,0px)+24px)] flex flex-col items-center gap-3.5">
        {/* the agent's running commentary: one short line per room */}
        <div className="relative grid min-h-[2.75rem] w-full max-w-xl place-items-center">
          <AnimatePresence mode="popLayout" initial={false}>
            {line && (
              <motion.p
                key={line.id}
                initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.45, ease: EASE }}
                className="col-start-1 row-start-1 flex max-w-full items-center gap-2.5 rounded-2xl border border-white/10 bg-stage/85 px-4 py-2.5 text-callout font-semibold text-ink shadow-[0_12px_40px_-18px_rgba(0,0,0,.8)] motion-reduce:transition-none"
              >
                <span className="shrink-0 rounded-full bg-heat/15 px-2 py-0.5 text-[11px] font-extrabold uppercase tracking-[0.14em] text-heat">Agent</span>
                <span className="min-w-0">{line.text}</span>
              </motion.p>
            )}
          </AnimatePresence>
        </div>
        <div className="flex items-center gap-3 rounded-full border border-white/10 bg-stage/80 px-4 py-2 text-callout font-semibold text-sub" aria-live="polite">
          <span className="tabular-nums text-ink">{count.n}</span>
          <span className="relative h-1.5 w-28 overflow-hidden rounded-full bg-white/10 sm:w-40">
            <span className="absolute inset-y-0 left-0 rounded-full bg-heat transition-[width] duration-500 ease-[var(--ease-spring)]" style={{ width: `${count.total ? (count.n / count.total) * 100 : 0}%` }} />
          </span>
          <span className="tabular-nums">{count.total} rooms</span>
        </div>
        <div role="group" aria-label="Choose a house" className="flex flex-wrap justify-center gap-2">
          {HOUSES.map((h) => (
            <button
              key={h.id} type="button" aria-pressed={h.id === house} onClick={() => missionStore.set({ house: h.id })}
              className={`rounded-full border px-4 py-2 text-footnote font-bold transition-colors duration-300 ${h.id === house ? "border-heat bg-heat text-heat-ink" : "border-line2 bg-panel/80 text-ink hover:border-mute"}`}
            >
              {h.name}
            </button>
          ))}
        </div>
      </div>
    </section>
  );
}
