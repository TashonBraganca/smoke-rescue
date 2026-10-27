'use client';
import Link from 'next/link';
import { useEffect, useRef, useState } from 'react';
import type { House } from './data';
import { timeLabel } from './data';
import type { MissionConfig } from './missions';
import Setup from './Setup';
import Stage, { ResultCard, type StageStatus } from './Stage';
const button = 'rounded-full border border-line2 px-5 py-3 text-sm font-bold hover:border-heat';
// The run films are rendered at 8x simulation speed. The spliced film opens with the cinematic and plays the run
// slowed by judge.slow, so mission time is (film time - runStart) / slow * 8.
const FILM_RATE = 8;
export default function FilmPlayer({ house, config, onReset }: { house: House; config: MissionConfig; onReset: () => void }) {
  const film = config.films[house.id], judge = film?.judge, src = judge?.src || film?.src;
  const missionTime = (t: number) => judge ? Math.min(film?.missionSeconds ?? Infinity, Math.max(0, (t - judge.runStart) / judge.slow * FILM_RATE)) : t * FILM_RATE;
  const video = useRef<HTMLVideoElement | null>(null), canvas = useRef<HTMLCanvasElement>(null);
  const [status, setStatus] = useState<StageStatus>('setup'), [ready, setReady] = useState(false), [blocked, setBlocked] = useState(false), [error, setError] = useState(''), [clock, setClock] = useState(0);
  // Some browsers never buffer a paused video; start anyway after the setup and let the clock wait for frames.
  useEffect(() => { const id = setTimeout(() => setReady(true), 7000); return () => clearTimeout(id); }, []);
  // The film plays from a video element that never enters the page; the stage draws its frames on a canvas.
  useEffect(() => {
    if (!src) return;
    const v = document.createElement('video');
    const canPlay = () => setReady(true), ended = () => setStatus('done'), failed = () => { setReady(true); setError('This mission could not load.'); };
    v.muted = true; v.playsInline = true; v.preload = 'auto'; v.src = src;
    v.addEventListener('canplay', canPlay); v.addEventListener('ended', ended); v.addEventListener('error', failed);
    video.current = v;
    return () => { v.pause(); v.removeEventListener('canplay', canPlay); v.removeEventListener('ended', ended); v.removeEventListener('error', failed); v.removeAttribute('src'); v.load(); video.current = null; };
  }, [src]);
  useEffect(() => {
    if (status !== 'running') return;
    let id = 0;
    const context = canvas.current?.getContext('2d');
    const tick = () => {
      const v = video.current;
      if (v) { setClock(Math.floor(missionTime(v.currentTime))); if (context && v.readyState >= 2) context.drawImage(v, 0, 0, 1920, 1080); }
      id = requestAnimationFrame(tick);
    };
    id = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(id);
  }, [status]);
  // Try with sound first (the cinematic has a score); browsers that refuse it get the muted film.
  const play = () => {
    const v = video.current; setStatus('running'); if (!v) return;
    v.muted = false;
    void v.play().then(() => setBlocked(false)).catch(() => { v.muted = true; void v.play().then(() => setBlocked(false)).catch(() => setBlocked(true)); });
  };
  const missionClock = status === 'done' ? film?.missionSeconds ?? clock : clock;
  return <Stage title={house.name} status={error ? 'error' : status} clock={missionClock} tag="Unity recording · 2 robots" onReset={onReset}>
    {src ? <canvas ref={canvas} width={1920} height={1080} aria-label={`${house.name} mission`} className="pointer-events-none absolute inset-0 h-full w-full object-contain" style={{ opacity: status === 'setup' ? 0 : 1, transition: 'opacity 0.9s var(--ease-out-quart)' }} />
      : <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 p-6 text-center"><h2 className="font-stencil text-title3">Mission coming soon</h2><p className="max-w-[48ch] text-sub">{film?.note || 'This house recording is being prepared.'}</p><button className={button} onClick={onReset}>Try another house</button></div>}
    {src && status === 'setup' && <Setup house={house} robots={2} live={false} ready={ready} onDone={play} />}
    {blocked && !error && <div className="absolute inset-0 flex items-center justify-center"><button className="rounded-full bg-heat px-8 py-4 text-callout font-extrabold text-heat-ink hover:bg-amber" onClick={play}>Start the mission</button></div>}
    {error && <div role="alert" className="absolute inset-0 flex flex-col items-center justify-center gap-4 p-6 text-center"><p className="text-sub">{error}</p><div className="flex gap-3"><button className={button} onClick={() => { setError(''); video.current?.load(); play(); }}>Retry</button><button className={button} onClick={onReset}>Try another house</button></div></div>}
    {status === 'done' && film && <ResultCard title={film.allOut === true ? 'Both robots are out.' : film.allOut === false ? 'The mission ended before both robots were out.' : 'Exit result awaiting the recording.'}
      stats={[[film.peopleFound != null ? `${film.peopleFound}/${film.peopleTotal}` : '-', 'People found'], [film.missionSeconds != null ? timeLabel(film.missionSeconds) : '-', 'Mission time'], [film.allOut ? '2/2' : '-', 'Robots out']]}
      note={<><p className="text-callout text-sub">On 39 real floor plans it never saw, two robots sharing one map found 99% of people; one robot found 92% (p = 0.002).</p><p className="mt-2 text-footnote text-mute">Headless simulation benchmark, 25 October 2026. <a className="underline" href="/rescue-data/benchmark.txt">Read the results</a>.</p></>}>
      <a className={`${button} border-heat bg-heat text-heat-ink hover:bg-amber`} href="/rescue/?mode=live">Run a live simulation on AWS</a><button className={button} onClick={onReset}>Try another house</button><Link className={button} href="/#built">How it works</Link>
    </ResultCard>}
  </Stage>;
}
