'use client';
import dynamic from 'next/dynamic';
import Link from 'next/link';
import { useEffect, useRef, useState } from 'react';
import { advanceClock, caption, exitCaption, keyFor, loadRecording, timeLabel, type House, type Recording } from './data';
import { startLive } from './live';
import type { MissionConfig } from './missions';
const Scene = dynamic(() => import('./Scene'), { ssr: false, loading: () => <p className="p-8 text-sub">Loading the house...</p> });
const button = 'rounded-full border border-line2 px-4 py-2 text-sm font-bold text-sub hover:border-heat hover:text-ink';
const noop = () => {};
export default function LiveMission({ house, robots, config, onReset }: { house: House; robots: number; config: MissionConfig; onReset: () => void }) {
  const [run, setRun] = useState<Recording | null>(null), [introDone, setIntroDone] = useState(false), [playing, setPlaying] = useState(true), [overview, setOverview] = useState(false), [reduced, setReduced] = useState(false);
  const [t, setT] = useState(0), [rate, setRate] = useState(4), [source, setSource] = useState('Live simulation on AWS'), [reason, setReason] = useState(''), [error, setError] = useState(''), [firstFrame, setFirstFrame] = useState(0);
  const clock = useRef(0), current = useRef<Recording | null>(null), video = useRef<HTMLVideoElement>(null);
  const onFrame = () => { if (!firstFrame) setFirstFrame(performance.now()); };
  useEffect(() => {
    const media = matchMedia('(prefers-reduced-motion: reduce)');
    const change = () => { setReduced(media.matches); setOverview(media.matches); };
    change(); media.addEventListener('change', change);
    return () => media.removeEventListener('change', change);
  }, []);
  useEffect(() => {
    const controller = new AbortController();
    const update = (r: Recording) => { if (!controller.signal.aborted) { current.current = r; setRun(r); } };
    startLive(house, robots, config.engine, controller.signal, update).catch(async e => {
      if (controller.signal.aborted) return;
      setReason(e.message || 'The live connection stopped.');
      setSource('Recorded replay of this exact setup');
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
  const visible = introDone && !!run && !error;
  useEffect(() => {
    if (!visible || !playing) return;
    let id = 0, previous = performance.now(), lastUi = 0;
    const tick = (now: number) => {
      const latest = current.current;
      const end = latest?.frames.at(-1)?.t || 0;
      const elapsed = document.hidden ? 0 : (now - previous) / 1000;
      previous = now; clock.current = Math.min(end, advanceClock(clock.current, elapsed, rate, latest?.duration_s || 180));
      if (now - lastUi > 80 || clock.current >= (latest?.duration_s || 180)) { setT(clock.current); lastUi = now; }
      if (clock.current >= (latest?.duration_s || 180) && latest?.validated) setPlaying(false); else id = requestAnimationFrame(tick);
    };
    id = requestAnimationFrame(tick); return () => cancelAnimationFrame(id);
  }, [visible, playing, rate]);
  const seek = (value: number) => { clock.current = value; setT(value); if (value >= (run?.duration_s || 180)) setPlaying(false); };
  const found = run?.frames.slice(0, Math.floor(t * 2) + 1).flatMap(f => f.found) || [];
  const foundCount = house.people_truth.filter(person => found.some(p => Math.hypot(p.x - person.x, p.y - person.y) <= 1.5)).length;
  const done = visible && t >= (run?.duration_s || 180) && run?.validated;
  return <section className="gutter mx-auto max-w-[1440px] py-10" aria-label="Live rescue" data-setup={visible ? run?.setup : undefined} data-first-frame={firstFrame}>
    <div className="mb-5 flex flex-wrap items-center justify-between gap-3"><h1 className="font-stencil text-title2">{house.name}</h1><button className={button} onClick={onReset}>Try another house</button></div>
    {!introDone && <><p className="mb-3 text-footnote text-sub">Cinematic intro · AWS starts your simulation while the robots arrive</p><div className="aspect-video overflow-hidden rounded-3xl border border-line bg-stage"><video ref={video} src={config.intro} poster={config.poster} autoPlay playsInline controls preload="metadata" onEnded={() => setIntroDone(true)} onError={() => setIntroDone(true)} className="h-full w-full object-contain" /></div><button className={`${button} mt-4`} onClick={() => setIntroDone(true)}>Skip cinematic intro</button></>}
    {introDone && !visible && !error && <div className="relative aspect-video overflow-hidden rounded-3xl border border-line bg-stage"><div className="absolute inset-0 bg-cover bg-center opacity-40" style={{ backgroundImage: `url(${config.poster})` }} /><p role="status" className="relative p-8 text-callout text-ink">The robots have arrived. Waiting for the first AWS frames...</p><button className={`${button} relative mx-8`} onClick={onReset}>Choose another house</button></div>}
    {introDone && error && <div role="alert" className="rounded-3xl border border-line bg-panel p-8"><p className="max-w-[70ch] text-callout text-sub">{error}</p><button className={`${button} mt-5`} onClick={onReset}>Choose another house</button></div>}
    {visible && run && <><div className="overflow-hidden rounded-3xl border border-line bg-stage">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-line p-4"><div><p data-source className="text-callout font-extrabold">{source}</p>{reason && <p className="mt-1 max-w-[70ch] text-footnote text-sub">The live run was unavailable. This is a precomputed brain-only simulation with the same house, people, fire and robot count.</p>}</div><button className={button} onClick={() => setOverview(!overview)}>{overview ? 'Follow robots' : 'Full house'}</button></div>
      <div className="relative h-[clamp(380px,62vh,700px)]"><Scene house={house} run={run} robots={robots} clock={clock} time={t} overview={overview} running reduced={reduced} onReady={noop} onFrame={onFrame} /><p className="pointer-events-none absolute left-4 top-4 rounded-full bg-stage/90 px-3 py-2 text-footnote font-extrabold text-whiteheat">{done ? run.score.people_found : foundCount} / {house.people} people found</p></div>
      <p data-caption aria-live="polite" className="min-h-16 border-t border-line px-5 py-4 text-callout font-semibold">{caption(house, run, t)}</p><div className="grid gap-3 p-5"><div className="flex flex-wrap items-center gap-2"><button className={button} onClick={() => { if (t >= (run?.duration_s || 180)) seek(0); setPlaying(!playing); }}>{playing ? 'Pause' : t >= (run?.duration_s || 180) ? 'Replay' : 'Play'}</button><button className={button} aria-label={`Playback speed ${rate}x`} onClick={() => setRate(rate === 4 ? 1 : rate === 1 ? 2 : 4)}>{rate}x</button><p className="ml-auto text-footnote text-sub">{timeLabel(t)} / {timeLabel(run.duration_s || 180)}</p></div><input aria-label="Replay time" type="range" min={0} max={run.frames.at(-1)?.t || 0} step={0.5} value={t} onChange={e => seek(Number(e.target.value))} className="w-full accent-heat" />{playing && !run.validated && t >= (run.frames.at(-1)?.t || 0) && <p role="status" className="text-footnote text-sub">Waiting for the next live frames...</p>}</div>
    </div>{done && <section data-end-card aria-label="Rescue result" className="mt-5 rounded-3xl border border-line bg-panel p-6"><h2 className="font-stencil text-title2">{exitCaption(robots, run.score.robots_out)}</h2><p className="mt-5 text-callout text-sub">{run.score.people_found} / {house.people} people found. {timeLabel(run.score.time_s)} mission time. {run.rooms_searched} / {house.rooms} rooms searched.</p><p className="mt-3 text-footnote text-sub">A room counts as searched when the robot maps cover at least 80% of its floor.</p><div className="mt-6 flex flex-wrap gap-3"><button className={button} onClick={onReset}>Try another house</button><Link className={button} href="/#built">How it works</Link></div></section>}
    <p className="mt-3 text-footnote text-sub">Robot paths, confirmations and fire come from the simulator. Smoke haze and furniture height are illustrations. floor plan from HouseExpo (MIT). <a className="underline" href="/rescue-data/houseexpo-license.txt">Read the license</a>.</p></>}
  </section>;
}
