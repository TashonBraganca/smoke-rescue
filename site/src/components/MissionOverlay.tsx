"use client";
import dynamic from "next/dynamic";
import { AnimatePresence, motion } from "motion/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import PlanThumb from "@/components/PlanThumb";
import WordRise from "@/components/WordRise";
import { HOUSES, type HouseId, type Plan } from "@/lib/plans";
import { SKINS } from "@/lib/skins";
import type { TeamSize } from "@/lib/store";
import { dogsWord, teamOf, teamPhrase } from "@/lib/team";
import { buildLayout } from "@/three/houseLayout";
import { RoomLabels } from "@/three/HouseModel";
import type { RunStatus } from "@/three/LiveRun";
import { planTeam } from "@/three/teamRun";

const LiveRun = dynamic(() => import("@/three/LiveRun"), { ssr: false });
const EASE = [0.2, 0.9, 0.25, 1] as const;
// Houses that would play the rendered film instead of the live 3D run. Every house runs live now.
const FILMS: string[] = [];
const FILM_SRC = "/film/rescue.mp4";

type Props = { open: boolean; onClose: () => void; count: TeamSize; house: HouseId; plan?: Plan; failed?: boolean; reduce: boolean };

// Full-screen mission player. Plays the rendered film for this house if one exists, otherwise runs it live in 3D.
export default function MissionOverlay(props: Props) {
  return <AnimatePresence>{props.open && <Player key="player" {...props} />}</AnimatePresence>;
}

function Player({ onClose, count, house, plan, failed, reduce }: Props) {
  const name = HOUSES.find((h) => h.id === house)!.name;
  const closeBtn = useRef<HTMLButtonElement>(null);
  const box = useRef<HTMLDivElement>(null);
  const listed = FILMS.includes(house);
  const who = teamPhrase(count), names = teamOf(count);
  // decided up front when there is no film to check, so no effect has to set it
  const [film, setFilm] = useState<"checking" | "yes" | "no">(() => (listed ? "checking" : "no"));
  const replayBtn = useRef<HTMLButtonElement>(null);
  const [status, setStatus] = useState<RunStatus>({ n: 0, total: 0, done: false, t: 0, perDog: [] });
  const [runKey, setRunKey] = useState(0);
  const [ready, setReady] = useState(false);
  const labels = useRef<(HTMLElement | null)[]>([]);
  const layout = useMemo(() => (plan ? buildLayout(plan) : null), [plan]);
  // plan (not play) all three team sizes: the chosen one plays, the others give the comparison line at the end
  const teams = useMemo(() => (layout ? ([1, 2, 3] as const).map((n) => planTeam(layout, n)) : null), [layout]);
  const plan1 = teams?.[0], team = teams?.[count - 1];
  // play back faster than real time so even one dog in the biggest house takes well under a minute; same rate for every team size
  const rate = plan1 ? Math.min(2.5, Math.max(1, Math.round((plan1.total / 32) * 2) / 2)) : 1;
  const src = FILM_SRC;

  // a film only counts if the server really returns a video (static hosts can answer 200 with the HTML page)
  useEffect(() => {
    if (!listed) return;
    let live = true;
    fetch(src, { method: "HEAD" })
      .then((r) => live && setFilm(r.ok && (r.headers.get("content-type") ?? "").startsWith("video") ? "yes" : "no"))
      .catch(() => live && setFilm("no"));
    return () => { live = false; };
  }, [src, listed]);

  // focus in, Escape closes, Tab stays inside, page behind does not scroll
  useEffect(() => {
    const prev = document.activeElement as HTMLElement | null;
    closeBtn.current?.focus({ preventScroll: true });
    const html = document.documentElement, was = html.style.overflow;
    html.style.overflow = "hidden";
    const key = (e: KeyboardEvent) => {
      if (e.key === "Escape") { e.preventDefault(); onClose(); return; }
      if (e.key !== "Tab" || !box.current) return;
      const f = [...box.current.querySelectorAll<HTMLElement>("button, video, [href], [tabindex]:not([tabindex='-1'])")].filter((el) => !el.hasAttribute("disabled"));
      if (!f.length) return;
      const first = f[0], last = f[f.length - 1], at = document.activeElement;
      // focus that has escaped the dialog (a click on the page behind, a canvas) is pulled back in
      if (!at || !box.current.contains(at)) { e.preventDefault(); (e.shiftKey ? last : first).focus(); return; }
      if (e.shiftKey && at === first) { e.preventDefault(); last.focus(); }
      else if (!e.shiftKey && at === last) { e.preventDefault(); first.focus(); }
    };
    window.addEventListener("keydown", key);
    return () => { window.removeEventListener("keydown", key); html.style.overflow = was; prev?.focus({ preventScroll: true }); };
  }, [onClose]);

  const onStatus = useCallback((s: RunStatus) => setStatus(s), []);
  const onReady = useCallback(() => setReady(true), []);
  // Replay disables itself while the run plays, so focus moves to Close instead of being dropped on the body
  const replay = () => { closeBtn.current?.focus({ preventScroll: true }); setRunKey((k) => k + 1); setStatus((s) => ({ ...s, n: 0, t: 0, done: false, perDog: [] })); };

  return (
    <motion.div
      ref={box}
      role="dialog" aria-modal="true" aria-label={`Mission: ${who} in the ${name}`}
      data-lenis-prevent
      className="fixed inset-0 z-[100] overflow-hidden bg-bg"
      initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
      transition={{ duration: reduce ? 0 : 0.45, ease: EASE }}
    >
      <motion.div
        className="absolute inset-0"
        initial={{ scale: reduce ? 1 : 1.04 }} animate={{ scale: 1 }} exit={{ scale: reduce ? 1 : 1.02 }}
        transition={{ duration: reduce ? 0 : 0.8, ease: EASE }}
      >
        {film === "yes" && (
          <video src={src} controls autoPlay playsInline className="h-full w-full bg-black object-contain" />
        )}
        {film === "no" && layout && team && (
          <div className={`absolute inset-0 transition-opacity duration-700 ${ready ? "opacity-100" : "opacity-0"}`}>
            <LiveRun layout={layout} team={team} rate={rate} reduce={reduce} runKey={runKey} onStatus={onStatus} labels={labels} onReady={onReady} />
            <RoomLabels layout={layout} els={labels} />
          </div>
        )}
      </motion.div>

      {failed && !layout && film === "no" && (
        <div className="absolute inset-0 grid place-items-center px-6 text-center">
          <div>
            <p className="text-title3 font-bold text-ink">This house did not load.</p>
            <p className="mt-2 text-body text-sub">Check your connection, close this and try again.</p>
          </div>
        </div>
      )}

      {/* top bar */}
      <div className="pointer-events-none absolute inset-x-0 top-0 h-36 bg-gradient-to-b from-bg/90 to-transparent" />
      <div className="gutter absolute inset-x-0 top-[calc(env(safe-area-inset-top,0px)+18px)] flex items-start justify-between gap-4">
        <div>
          <p className="text-footnote font-extrabold uppercase tracking-[0.18em] text-heat">Mission</p>
          <p className="mt-1 text-title3 font-bold text-ink">{who} in the {name.replace(/^[A-Z](?!-)/, (c) => c.toLowerCase())}</p>
        </div>
        <button
          ref={closeBtn} type="button" onClick={onClose} aria-label="Close"
          className="grid size-11 shrink-0 place-items-center rounded-full border border-white/15 bg-panel/80 text-ink transition-colors hover:bg-panel"
        >
          <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><path d="M3 3l10 10M13 3L3 13" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" /></svg>
        </button>
      </div>

      {film === "no" && plan && (
        <div className="pointer-events-none absolute right-[max(16px,env(safe-area-inset-right,0px))] top-[calc(env(safe-area-inset-top,0px)+80px)] w-[min(30vw,240px)] rounded-2xl border border-white/10 bg-bg/80 p-2 shadow-[0_16px_40px_-16px_rgba(0,0,0,.9)] backdrop-blur-sm">
          <PlanThumb plan={plan} id="overlay-plan" className="block h-auto w-full" />
        </div>
      )}

      {film === "no" && layout && teams && team && (
        <>
          <div className="pointer-events-none absolute inset-x-0 bottom-0 h-64 bg-gradient-to-t from-bg/95 via-bg/60 to-transparent" />
          <div className="gutter absolute inset-x-0 bottom-[calc(env(safe-area-inset-bottom,0px)+24px)] flex flex-col items-center gap-4 text-center">
            <div className="flex flex-col items-center gap-2">
              <WordRise as="p" text={`Every room checked in ${Math.round(team.total)}\u00a0s with ${dogsWord(count)}.`} show={status.done} className="text-title3 font-extrabold text-amber sm:text-title2" />
              <motion.p
                className="text-footnote text-sub"
                animate={{ opacity: status.done ? 1 : 0 }} transition={{ duration: reduce ? 0 : 0.5, delay: status.done && !reduce ? 0.5 : 0 }}
                aria-hidden={!status.done}
              >
                {teams.map((p, i) => (
                  <span key={i} className={i + 1 === count ? "font-bold text-ink" : ""}>
                    {i > 0 && <span className="px-1.5 text-mute">·</span>}
                    <span className="whitespace-nowrap">{dogsWord(i + 1)}: {Math.round(p.total)}&nbsp;s</span>
                  </span>
                ))}
              </motion.p>
            </div>
            <div className="flex flex-wrap items-center justify-center gap-2.5">
              {/* simulated clock: the run plays back faster than real time, and says so */}
              <div className="flex items-baseline gap-2 rounded-full border border-white/10 bg-stage/80 px-4 py-2 text-callout font-semibold text-sub">
                <span className="text-[11px] font-bold uppercase tracking-[0.14em] text-mute">Simulated time</span>
                <span className="min-w-[3.2ch] text-right tabular-nums text-ink">{status.t} s</span>
                {rate > 1 && <span className="text-[11px] text-mute">played at {rate}x</span>}
              </div>
              <div className="flex items-center gap-3 rounded-full border border-white/10 bg-stage/80 px-4 py-2 text-callout font-semibold text-sub" aria-live="polite">
                {team.dogs.map((d, k) => (
                  <span key={k} className="flex items-center gap-1.5" aria-label={`${names[k]}: ${status.perDog[k] ?? 0} of ${d.rooms} rooms`}>
                    <span className="size-2.5 rounded-full" style={{ background: SKINS[names[k]].hex }} aria-hidden="true" />
                    <span className="hidden text-ink sm:inline" aria-hidden="true">{names[k]}</span>
                    <span className="tabular-nums" aria-hidden="true">{status.perDog[k] ?? 0}</span>
                  </span>
                ))}
                <span className="text-mute">of {layout.rooms.length} rooms</span>
              </div>
              <motion.button
                ref={replayBtn} type="button" onClick={replay}
                animate={{ opacity: status.done ? 1 : 0, scale: status.done ? 1 : 0.94 }}
                transition={{ duration: 0.4, ease: EASE }}
                disabled={!status.done} aria-hidden={!status.done}
                className="rounded-full bg-heat px-5 py-2.5 text-callout font-extrabold text-heat-ink transition-colors hover:bg-amber disabled:pointer-events-none"
              >
                Replay
              </motion.button>
            </div>
          </div>
        </>
      )}
    </motion.div>
  );
}
