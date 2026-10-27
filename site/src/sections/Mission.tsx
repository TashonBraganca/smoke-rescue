"use client";
import dynamic from "next/dynamic";
import { AnimatePresence, motion, useInView, useReducedMotion } from "motion/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import MissionOverlay from "@/components/MissionOverlay";
import PlanThumb from "@/components/PlanThumb";
import WordRise from "@/components/WordRise";
import { HOUSES, loadPlan, planInfo, type HouseId, type Plan } from "@/lib/plans";
import { SKINS, SKIN_NAMES } from "@/lib/skins";
import { missionStore, useMission } from "@/lib/store";
import { buildLayout } from "@/three/houseLayout";

// Choose a mission: the two dogs on a turntable (drag to spin), the picked house as a turning 3D model with its
// floor plan in the corner, five houses to pick from, then the live 3D run in the full-screen player.
const Turntable = dynamic(() => import("@/three/Turntable"), { ssr: false });
const HousePreview = dynamic(() => import("@/three/HousePreview"), { ssr: false });
const EASE = [0.2, 0.9, 0.25, 1] as const;
const TEAM = SKIN_NAMES.slice(0, 2);
// The judge flow lives on /rescue/: the picked house opens there and its mission starts at once (its film, or a live AWS run).
const RESCUE_ID: Record<HouseId, string> = { family_ranch: 'family_ranch', long_ranch: 'long_ranch', '934': 'onedoor_934', '5008': 'hx_46bfe067cc', '5009': 'hx_cb9d3d6726' };

export default function Mission() {
  const { house } = useMission();
  const reduce = !!useReducedMotion();
  const [plans, setPlans] = useState<Partial<Record<HouseId, Plan>>>({});
  const [failed, setFailed] = useState<Partial<Record<HouseId, boolean>>>({});
  const [ready, setReady] = useState(false);
  const [shown, setShown] = useState(false);
  const [open, setOpen] = useState(false);
  const head = useRef<HTMLElement>(null), stage = useRef<HTMLDivElement>(null);
  const headIn = useInView(head, { once: true, margin: "-10% 0px" });
  const stageIn = useInView(stage, { margin: "0px 0px -10% 0px" });
  useEffect(() => {
    let live = true;
    HOUSES.forEach((h) => loadPlan(h.id).then(
      (p) => live && setPlans((s) => ({ ...s, [h.id]: p })),
      () => live && setFailed((s) => ({ ...s, [h.id]: true })),
    ));
    return () => { live = false; };
  }, []);
  useEffect(() => { missionStore.set({ count: 2, dog: "Cinder" }); }, []);
  const onReady = useCallback(() => setReady(true), []);
  const onShown = useCallback(() => setShown(true), []);
  const close = useCallback(() => setOpen(false), []);
  const plan = plans[house];
  const layout = useMemo(() => (plan ? buildLayout(plan) : null), [plan]);
  const info = plan ? planInfo(house, plan) : null;

  return (
    <section id="mission" aria-label="Pick a house and send in the dogs" className="relative overflow-hidden py-[clamp(72px,10vw,128px)]">
      <div className="gutter mx-auto max-w-6xl">
        <header ref={head} className="text-center">
          <WordRise as="h2" text="Pick a house. Send in the dogs." show={headIn || reduce} className="text-title1 font-extrabold text-ink" />
        </header>

        {/* the two dogs */}
        <div className="relative left-1/2 h-[clamp(280px,46vh,460px)] w-screen -translate-x-1/2">
          <div aria-hidden="true" className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_42%_48%_at_50%_62%,rgba(249,140,10,.12),rgba(87,16,110,.10)_55%,transparent_78%)]" />
          <div className={`absolute inset-0 transition-opacity duration-1000 ease-[var(--ease-spring)] [mask-image:linear-gradient(to_bottom,transparent,black_12%,black_82%,transparent)] ${ready ? "opacity-100" : "opacity-0"}`}>
            <Turntable reduce={reduce} onReady={onReady} paused={open} />
          </div>
        </div>
        <p className="-mt-2 flex items-center justify-center gap-5 text-body">
          {TEAM.map((n) => (
            <span key={n} className="flex items-center gap-2 font-semibold text-ink">
              <span className="size-2.5 rounded-full" style={{ background: SKINS[n].hex, boxShadow: `0 0 10px ${SKINS[n].hex}` }} aria-hidden="true" />{n}
            </span>
          ))}
          <span className="flex items-center gap-1.5 text-footnote text-mute" aria-hidden="true">
            <svg width="18" height="12" viewBox="0 0 18 12" fill="none"><path d="M5 2 1 6l4 4M13 2l4 4-4 4M1 6h16" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /></svg>
            drag
          </span>
        </p>

        {/* the picked house, in 3D, with its floor plan in the corner */}
        <div ref={stage} className="relative mt-12 aspect-[16/10] overflow-hidden rounded-[28px] border border-line bg-[radial-gradient(ellipse_at_50%_40%,#22122c,#0b0612_70%)] sm:aspect-[16/8]">
          <div className={`absolute inset-0 cursor-grab transition-opacity duration-700 active:cursor-grabbing ${shown ? "opacity-100" : "opacity-0"}`}>
            {layout && <HousePreview layout={layout} active={stageIn && !open} reduce={reduce} onReady={onShown} />}
          </div>
          {failed[house] && !layout && <p className="absolute inset-0 grid place-items-center text-body text-sub">This house did not load. Reload to try again.</p>}
          <div className="pointer-events-none absolute left-4 top-4 flex items-center gap-2.5 sm:left-6 sm:top-6">
            <span className="flex items-center gap-1.5 rounded-full bg-coal/90 px-2.5 py-1 text-[11px] font-extrabold uppercase tracking-[0.14em] text-ink">
              <span className="size-1.5 animate-pulse rounded-full bg-white" />Live
            </span>
            <AnimatePresence mode="wait">
              <motion.span key={house} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.3, ease: EASE }}
                className="text-title3 font-extrabold text-ink [text-shadow:0_2px_12px_rgba(0,0,0,.8)]">
                {HOUSES.find((h) => h.id === house)!.name}
              </motion.span>
            </AnimatePresence>
          </div>
          {info && (
            <div className="pointer-events-none absolute left-4 bottom-4 flex gap-2 text-footnote font-bold sm:left-6 sm:bottom-6">
              <span className="rounded-full border border-white/10 bg-stage/85 px-3 py-1.5 text-ink">{info.rooms} rooms</span>
              <span className="rounded-full border border-coal/40 bg-stage/85 px-3 py-1.5 text-amber">Fire: {info.fire}</span>
            </div>
          )}
          {plan && (
            <AnimatePresence mode="wait">
              <motion.div key={house} initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.35, ease: EASE }}
                className="pointer-events-none absolute bottom-4 right-4 w-[min(30%,220px)] rounded-2xl border border-white/10 bg-bg/80 p-2 shadow-[0_16px_40px_-16px_rgba(0,0,0,.9)] backdrop-blur-sm sm:bottom-6 sm:right-6">
                <PlanThumb plan={plan} id={`pv${house}`} className="block h-auto w-full" />
              </motion.div>
            </AnimatePresence>
          )}
        </div>

        {/* the five houses */}
        <div role="group" aria-label="Choose a house" className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-5">
          {HOUSES.map((h) => {
            const p = plans[h.id], on = h.id === house;
            return (
              <button key={h.id} type="button" aria-pressed={on} onClick={() => missionStore.set({ house: h.id })}
                className={`group grid gap-2 rounded-2xl border p-3 text-left transition-[border-color,background-color,box-shadow] duration-300 ${on ? "border-heat bg-panel shadow-[0_18px_44px_-24px_rgba(249,140,10,.7)]" : "border-line bg-panel/40 hover:border-line2 hover:bg-panel"}`}>
                <span className={`grid h-16 place-items-center transition-opacity ${on ? "opacity-100" : "opacity-60 group-hover:opacity-90"}`}>
                  {p && <PlanThumb plan={p} id={`tile${h.id}`} className="block h-full w-auto" />}
                </span>
                <span className={`text-footnote font-extrabold ${on ? "text-ink" : "text-sub"}`}>{h.name}</span>
              </button>
            );
          })}
        </div>

        <div className="mt-10 flex justify-center">
          <motion.button
            type="button" onClick={() => { window.location.href = `/rescue/?house=${RESCUE_ID[house]}&robots=2&start=1`; }}
            whileHover={{ scale: reduce ? 1 : 1.03 }} whileTap={{ scale: reduce ? 1 : 0.97 }}
            transition={{ type: "spring", stiffness: 400, damping: 26 }}
            className="flex items-center gap-3 rounded-full bg-heat px-9 py-4 text-[1.1875rem] font-extrabold text-heat-ink shadow-[0_20px_60px_-20px_rgba(249,140,10,.8)] transition-colors hover:bg-amber disabled:opacity-50 sm:px-11 sm:py-5"
          >
            <svg width="14" height="16" viewBox="0 0 14 16" aria-hidden="true"><path d="M1.5 1.6v12.8a.8.8 0 0 0 1.2.7l10.6-6.4a.8.8 0 0 0 0-1.4L2.7.9a.8.8 0 0 0-1.2.7z" fill="currentColor" /></svg>
            Start mission
          </motion.button>
        </div>
      </div>
      <MissionOverlay open={open} onClose={close} count={2} house={house} plan={plan} failed={!!failed[house]} reduce={reduce} />
    </section>
  );
}
