'use client';
import Link from 'next/link';
import { useEffect, useRef, useState } from 'react';
import type { House } from './data';
import { timeLabel } from './data';
import type { MissionConfig } from './missions';
const button = 'rounded-full border border-line2 px-5 py-3 text-sm font-bold hover:border-heat';
export default function FilmPlayer({ house, config, onReset }: { house: House; config: MissionConfig; onReset: () => void }) {
  const [phase, setPhase] = useState<'intro' | 'mission' | 'outro' | 'end'>('intro');
  const [error, setError] = useState('');
  const [blocked, setBlocked] = useState(false);
  const video = useRef<HTMLVideoElement>(null);
  const film = config.films[house.id];
  const src = phase === 'intro' ? config.intro : phase === 'mission' ? film?.src : phase === 'outro' ? config.outro : null;
  const next = () => { setError(''); setPhase(phase === 'intro' ? 'mission' : phase === 'mission' ? 'outro' : 'end'); };
  useEffect(() => {
    if (!src) return;
    let current = true;
    video.current?.play().catch(() => { if (current) setBlocked(true); });
    return () => { current = false; };
  }, [src]);
  return <section className="gutter mx-auto max-w-[1440px] py-10" aria-label="Filmed rescue" data-film-phase={phase}>
    <div className="mb-5 flex flex-wrap items-center justify-between gap-3"><h1 className="font-stencil text-title2">{house.name}</h1><button className={button} onClick={onReset}>Try another house</button></div>
    {phase !== 'end' && <><div className="mb-3 flex flex-wrap justify-between gap-3 text-footnote text-sub"><p>{phase === 'intro' ? 'Cinematic intro' : phase === 'mission' ? 'Recorded Unity mission · 2 robots' : 'Cinematic outro'}</p>{src && <button className="underline underline-offset-4" onClick={next}>{phase === 'intro' ? 'Skip cinematic intro' : phase === 'mission' ? 'View ending' : 'View results'}</button>}</div>
      <div className="relative aspect-video overflow-hidden rounded-3xl border border-line bg-stage">
        {src ? <video ref={video} key={src} src={src} poster={config.poster} playsInline controls autoPlay preload="metadata" onPlaying={() => setBlocked(false)} onEnded={next} onError={() => setError('This film could not load. Retry it or choose another house.')} className="h-full w-full object-contain" /> : <div className="flex h-full flex-col items-center justify-center gap-4 p-6 text-center"><h2 className="font-stencil text-title3">Mission film coming soon</h2><p className="max-w-[48ch] text-sub">{film?.note || 'This house recording is being prepared.'}</p><p className="max-w-[48ch] text-footnote text-sub">The cinematic intro is an illustration. The house mission will appear here when its recording is ready.</p><button className={button} onClick={onReset}>Try another house</button></div>}
      </div>
      {blocked && !error && <button className={`${button} mt-4`} onClick={() => { void video.current?.play().catch(() => setError('Use the video controls to start playback.')); }}>Play film</button>}
      {error && <p role="alert" className="mt-4 text-sub">{error} <button className="underline" onClick={() => { setError(''); video.current?.load(); }}>Retry film</button></p>}
      {phase === 'outro' && <p className="mt-3 text-footnote text-sub">Illustrated return to the crew. The mission result below comes from the Unity recording.</p>}
    </>}
    {phase === 'end' && <section data-end-card aria-label="Mission result" className="rounded-3xl border border-line bg-panel p-6 sm:p-10">
      <p className="mb-3 text-footnote font-bold uppercase tracking-widest text-heat">Mission result</p><h2 className="font-stencil text-title2">{film?.allOut === true ? 'Both robots are out.' : film?.allOut === false ? 'The team did not both return.' : 'Exit result awaiting the recording.'}</h2>
      <div className="my-8 flex flex-wrap gap-x-16 gap-y-5"><p><span className="block font-stencil text-title2">{film?.peopleFound ?? 'Pending'}{film?.peopleFound != null ? ` / ${film.peopleTotal}` : ''}</span><span className="text-sub">People found</span></p><p><span className="block font-stencil text-title2">{film?.missionSeconds != null ? timeLabel(film.missionSeconds) : 'Pending'}</span><span className="text-sub">Mission time</span></p></div>
      <p className="max-w-[70ch] text-callout text-sub">On 39 real floor plans it never saw, two robots sharing one map found 99% of people; one robot found 92% (p = 0.002).</p><p className="mt-3 text-footnote text-sub">Headless simulation benchmark, 25 October 2026. <a className="underline" href="/rescue-data/benchmark.txt">Read the results</a>.</p>
      <div className="mt-8 flex flex-wrap gap-3"><a className={`${button} bg-heat text-heat-ink`} href="/rescue/?mode=live">Run a live simulation on AWS</a><button className={button} onClick={onReset}>Try another house</button><Link className={button} href="/#built">How it works</Link></div><p className="mt-4 text-footnote text-sub">Live simulations use the two extra floor plans. Our three houses play their recorded missions.</p>
    </section>}
  </section>;
}
