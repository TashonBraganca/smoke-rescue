'use client';
import dynamic from 'next/dynamic';
import Link from 'next/link';
import { useCallback, useEffect, useRef, useState } from 'react';
import { advanceClock, caption, exitCaption, keyFor, loadRecording, timeLabel, type House, type Recording } from './data';
import { startLive } from './live';
import type { MissionConfig } from './missions';
import LiveDash from './LiveDash';
import Setup from './Setup';
import Stage, { ResultCard } from './Stage';
const Scene = dynamic(() => import('./Scene'), { ssr: false, loading: () => <p className="p-8 text-sub">Loading the house...</p> });
const button = 'rounded-full border border-line2 px-5 py-3 text-sm font-bold hover:border-heat';
const chip = 'pointer-events-auto rounded-full border border-white/10 bg-stage/80 px-4 py-2 text-[13px] font-bold text-sub backdrop-blur hover:border-heat hover:text-ink';
const noop = () => {};
// Quiet stretch: no event and no new find in the next 6 s of the replay (frames are 2 per second); play it 2x faster.
const quiet = (run: Recording | null, t: number) => !!run && run.frames.slice(Math.floor(t * 2), Math.floor(t * 2) + 12).every(f => f.events.length === 0 && f.found.length === 0);
export default function LiveMission({ house, robots, config, onReset }: { house: House; robots: number; config: MissionConfig; onReset: () => void }) {
  const [run, setRun] = useState<Recording | null>(null), [setupDone, setSetupDone] = useState(false), [playing, setPlaying] = useState(true), [overview, setOverview] = useState(false), [reduced, setReduced] = useState(false);
  const [t, setT] = useState(0), [rate] = useState(8), [replay, setReplay] = useState(false), [error, setError] = useState(''), [firstFrame, setFirstFrame] = useState(0);
  const clock = useRef(0), current = useRef<Recording | null>(null);
  const onFrame = () => { if (!firstFrame) setFirstFrame(performance.now()); };
  const finishSetup = useCallback(() => setSetupDone(true), []);
  useEffect(() => {
    const media = matchMedia('(prefers-reduced-motion: reduce)');
    const change = () => { setReduced(media.matches); setOverview(media.matches); };
    change(); media.addEventListener('change', change);
    return () => media.removeEventListener('change', change);
  }, []);
  useEffect(() => {
    const controller = new AbortController();
    const update = (r: Recording) => { if (!controller.signal.aborted) { current.current = r; setRun(r); } };
    startLive(house, robots, config.engine, controller.signal, update).catch(async () => {
      if (controller.signal.aborted) return;
      setReplay(true);
      try {
        if (!config.engine || !house.available_replays?.includes(robots)) throw Error('A checked replay is not ready.');
        const fallback = await loadRecording(keyFor(house, robots));
        if (fallback.gate?.engine_git !== config.engine) throw Error('The replay was recorded with a different engine version.');
        if (controller.signal.aborted) return;
        clock.current = 0; setT(0); setPlaying(true); update(fallback);
      } catch {
        if (!controller.signal.aborted) { current.current = null; setRun(null); setError('The live run is unavailable, and a checked replay of this setup is not ready yet. Please choose another house or try again later.'); }
      }
    });
    return () => controller.abort();
  }, [house, robots, config.engine]);
  const visible = setupDone && !!run && !error;
  useEffect(() => {
    if (!visible || !playing) return;
    let id = 0, previous = performance.now(), lastUi = 0;
    const tick = (now: number) => {
      const latest = current.current;
      const end = latest?.frames.at(-1)?.t || 0;
      const elapsed = document.hidden ? 0 : (now - previous) / 1000;
      // A live stream arrives at about real time: follow it smoothly instead of racing ahead and stalling.
      // A recorded replay plays at about the pace of a live run (3x, quiet stretches 6x); a finished live run catches up at 8x.
      const base = replay ? 3 : rate;
      const speed = latest?.validated ? base * (quiet(latest, clock.current) ? 2 : 1) : Math.max(1, Math.min(8, (end - clock.current) * 0.8));
      previous = now; clock.current = Math.min(end, advanceClock(clock.current, elapsed, speed, latest?.duration_s || 180));
      if (now - lastUi > 80 || clock.current >= (latest?.duration_s || 180)) { setT(clock.current); lastUi = now; }
      if (clock.current >= (latest?.duration_s || 180) && latest?.validated) setPlaying(false); else id = requestAnimationFrame(tick);
    };
    id = requestAnimationFrame(tick); return () => cancelAnimationFrame(id);
  }, [visible, playing, rate, replay]);
  const again = () => { clock.current = 0; setT(0); setPlaying(true); };
  const found = run?.frames.slice(0, Math.floor(t * 2) + 1).flatMap(f => f.found) || [];
  const foundCount = house.people_truth.filter(person => found.some(p => Math.hypot(p.x - person.x, p.y - person.y) <= 1.5)).length;
  const done = visible && t >= (run?.duration_s || 180) && !!run?.validated;
  const waiting = visible && playing && !run.validated && t >= (run.frames.at(-1)?.t || 0);
  return <Stage title={house.name} status={error ? 'error' : !setupDone ? 'setup' : done ? 'done' : 'running'} clock={done && run ? run.score.time_s : t} tag={`${replay ? 'Recorded replay of this setup' : 'Live on AWS'} · ${robots} ${robots === 1 ? 'robot' : 'robots'}`} onReset={onReset}>
    <div className="absolute inset-0 flex flex-col gap-3 overflow-y-auto p-3 md:grid md:grid-rows-[minmax(0,1.35fr)_minmax(0,1fr)] md:overflow-hidden" data-setup={visible ? run?.setup : undefined} data-first-frame={firstFrame} style={{ opacity: visible ? 1 : 0, transition: 'opacity 0.9s var(--ease-out-quart)' }}>
      <div className="relative min-h-[46dvh] overflow-hidden rounded-2xl border border-white/[0.07] bg-[#0d0a15] md:min-h-0">
        {run && <Scene house={house} run={run} robots={robots} clock={clock} time={t} overview={overview} running reduced={reduced} onReady={noop} onFrame={onFrame} />}
        {visible && run && <>
          <div className="pointer-events-none absolute left-4 top-4 flex items-center gap-2">{!replay && !done && <span className="flex items-center gap-2 rounded-full bg-[#c0392b] px-3 py-1.5 text-[12px] font-extrabold tracking-wide text-white"><span className="h-2 w-2 animate-pulse rounded-full bg-white" />LIVE · AWS Fargate · us-east-1</span>}<span className="rounded-full border border-white/10 bg-stage/80 px-4 py-1.5 text-[13px] font-extrabold text-whiteheat backdrop-blur">{done ? run.score.people_found : foundCount} / {house.people} people found</span></div>
          <div className="pointer-events-none absolute right-4 top-4 flex gap-2"><button className={chip} onClick={() => setOverview(!overview)}>{overview ? 'Follow robots' : 'Full house'}</button></div>
          {!done && <div className="pointer-events-none absolute inset-x-0 bottom-0 px-5 pb-4 pt-12" style={{ background: 'linear-gradient(to top, #0d0a15, transparent)' }}>
            <p data-caption aria-live="polite" className="max-w-[70ch] text-lg font-semibold">{caption(house, run, t)}</p>
            <p data-source className="mt-1 text-[12px] text-mute">{waiting ? 'Receiving the next frames from AWS...' : replay ? 'The live run was unavailable. This is a precomputed brain-only simulation with the same house, people, fire and robot count.' : 'Robot paths, confirmations and fire stream from the simulator on AWS. Smoke haze and furniture height are illustrations.'} {house.source === 'HouseExpo' && <>Floor plan from HouseExpo (MIT), <a className="pointer-events-auto underline" href="/rescue-data/houseexpo-license.txt">license</a>.</>}</p>
          </div>}
        </>}
      </div>
      {run && <LiveDash house={house} run={run} t={t} live={!replay} />}
    </div>
    {!setupDone && <Setup house={house} robots={robots} live ready={!!run || !!error} onDone={finishSetup} />}
    {setupDone && error && <div role="alert" className="absolute inset-0 flex flex-col items-center justify-center gap-5 p-6 text-center"><p className="max-w-[60ch] text-callout text-sub">{error}</p><button className={button} onClick={onReset}>Choose another house</button></div>}
    {done && run && <ResultCard title={exitCaption(robots, run.score.robots_out)} stats={[[`${run.score.people_found}/${house.people}`, 'People found'], [timeLabel(run.score.time_s), 'Mission time'], [`${run.rooms_searched}/${house.rooms}`, 'Rooms searched']]}
      note={<p className="text-footnote text-sub">A room counts as searched when the robot maps cover at least 80% of its floor.</p>}>
      <button className={button} onClick={again}>Watch again</button><button className={button} onClick={onReset}>Try another house</button><Link className={button} href="/#built">How it works</Link>
    </ResultCard>}
  </Stage>;
}
