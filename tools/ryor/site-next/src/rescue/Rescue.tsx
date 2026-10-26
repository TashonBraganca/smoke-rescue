'use client';
import dynamic from 'next/dynamic';
import Link from 'next/link';
import { useCallback, useEffect, useRef, useState } from 'react';
import Nav from '@/components/Nav';
import { KIND_COLOR } from '@/lib/plans';
import { advanceClock, caption, exitCaption, keyFor, loadRecording, timeLabel, type House, type Recording } from './data';

const Scene = dynamic(() => import('./Scene'), { ssr: false, loading: () => <p className="p-8 text-sub">Loading the house...</p> });
const button = 'rounded-full border border-line2 px-4 py-2 text-sm font-bold text-sub transition-colors hover:border-heat hover:text-ink disabled:opacity-40';
function Thumb({ house }: { house: House }) {
  const [x0, y0, x1, y1] = house.bounds;
  return <svg className="h-14 w-[76px] shrink-0" viewBox={`${x0 - 0.5} ${-y1 - 0.5} ${x1 - x0 + 1} ${y1 - y0 + 1}`} aria-hidden="true">
    {house.rooms_poly.map(r => <polygon key={r.id} points={r.poly.map(p => `${p[0]},${-p[1]}`).join(' ')} fill={KIND_COLOR[r.kind] || '#3a2350'} />)}
    {house.walls.map((w, i) => <line key={i} x1={w[0]} y1={-w[1]} x2={w[2]} y2={-w[3]} stroke="#cdbcc9" strokeWidth={0.09} />)}
    <circle cx={0} cy={0} r={0.22} fill="#f98c0a" />
  </svg>;
}
export default function Rescue() {
  const [houses, setHouses] = useState<House[]>([]);
  const [selected, setSelected] = useState(0);
  const [robots, setRobots] = useState(2);
  const [run, setRun] = useState<Recording | null>(null);
  const [active, setActive] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [overview, setOverview] = useState(false);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState('');
  const [reduced, setReduced] = useState(false);
  const [t, setT] = useState(0);
  const [rate, setRate] = useState(4);
  const [firstFrame, setFirstFrame] = useState(0);
  const clock = useRef(0);
  const painted = useRef(false);
  const stage = useRef<HTMLDivElement>(null);
  const house = houses[selected];
  const onReady = useCallback(() => setReady(true), []);
  const onFrame = useCallback(() => {
    if (!painted.current) { painted.current = true; requestAnimationFrame(() => setFirstFrame(performance.now())); }
  }, []);
  useEffect(() => {
    const media = matchMedia('(prefers-reduced-motion: reduce)');
    const change = () => setReduced(media.matches);
    change(); media.addEventListener('change', change);
    fetch('/rescue-data/gallery.json').then(async r => {
      if (!r.ok) throw new Error('The houses could not load. Reload this page to try again.');
      const gallery = await r.json();
      setHouses(gallery.houses);
      // All fifteen recordings are small static files. No mission server is in the playback path.
      for (const h of gallery.houses as House[]) for (const n of [1, 2, 3]) void loadRecording(keyFor(h, n)).catch(() => {});
    }).catch(e => setError(e.message));
    return () => media.removeEventListener('change', change);
  }, []);
  useEffect(() => {
    if (!house) return;
    let current = true;
    loadRecording(keyFor(house, robots)).then(data => { if (current) setRun(data); }).catch(e => { if (current) setError(e.message); });
    return () => { current = false; };
  }, [house, robots]);
  useEffect(() => {
    if (!playing) return;
    let id = 0, previous = performance.now(), lastUi = 0;
    const tick = (now: number) => {
      // A hidden tab pauses the clock instead of skipping the rescue.
      const dt = document.hidden ? 0 : Math.min((now - previous) / 1000, 0.1);
      previous = now;
      clock.current = advanceClock(clock.current, dt, rate);
      if (now - lastUi >= 80 || clock.current >= 180) { setT(clock.current); lastUi = now; }
      if (clock.current >= 180) setPlaying(false); else id = requestAnimationFrame(tick);
    };
    id = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(id);
  }, [playing, rate]);
  const start = () => {
    if (!run || !ready) return;
    clock.current = 0; setT(0); setFirstFrame(0); painted.current = false;
    setActive(true); setPlaying(true); setOverview(reduced);
    if (window.innerWidth < 1024) requestAnimationFrame(() => stage.current?.scrollIntoView({ behavior: reduced ? 'instant' : 'smooth', block: 'start' }));
  };
  const reset = () => { setPlaying(false); setActive(false); clock.current = 0; setT(0); setFirstFrame(0); window.scrollTo({ top: 0, behavior: 'instant' }); };
  const seek = (value: number) => { clock.current = value; setT(value); if (value >= 180) setPlaying(false); };
  const frame = run?.frames[Math.min(360, Math.floor(t * 2))];
  const done = active && t >= 180;
  const markers = run?.frames.slice(0, Math.floor(t * 2) + 1).flatMap(f => f.found) || [];
  const foundCount = house?.people_truth.filter(person => markers.some(p => Math.hypot(p.x - person.x, p.y - person.y) <= 1.5)).length || 0;
  const rooms = frame?.mapped.filter(p => p >= 80).length || 0;
  const one = house?.scores['1'];
  return <main className="min-h-dvh pb-14">
    <Nav rescue />
    <div className="gutter mx-auto max-w-[1440px] pt-28 md:pt-32">
      <header className="mb-8 grid gap-4 lg:grid-cols-[1fr_340px] lg:items-end">
        <div><p className="mb-3 text-footnote font-extrabold uppercase tracking-[0.16em] text-heat">Run your own rescue</p><h1 className="max-w-[850px] font-stencil text-title1 text-ink">Pick a house.<br />Send in the dogs.</h1></div>
        <p className="max-w-[48ch] text-callout text-sub">Five houses. One, two or three robots. Watch the robot brains search for five people as fire spreads.</p>
      </header>
      {error && <p role="alert" className="mb-4 rounded-2xl border border-coal bg-panel p-4 text-ink">{error} <button className="underline" onClick={() => window.location.reload()}>Reload</button></p>}
      {!house && !error && <p className="py-16 text-sub">Loading the five houses...</p>}
      {house && <div className={`grid gap-5 ${active ? '' : 'lg:grid-cols-[310px_1fr]'}`}>
        {!active && <section aria-label="Choose your rescue" className="grid content-start gap-3">
          <h2 className="text-callout font-extrabold">Choose a house</h2>
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-1">
            {houses.map((h, i) => <button key={h.id} data-house={h.id} aria-pressed={selected === i} onClick={() => { if (selected !== i) { setRun(null); setError(''); setSelected(i); } clock.current = 0; setT(0); }} className={`flex min-h-[76px] items-center gap-3 rounded-2xl border px-3 py-2 text-left transition-colors ${selected === i ? 'border-heat bg-heat/10' : 'border-line bg-panel/60 hover:border-line2'}`}>
              <Thumb house={h} /><span><span className="block text-sm font-extrabold text-ink">{h.name}</span><span className="block text-footnote text-mute">{h.rooms} rooms</span>{h.source === 'HouseExpo' && <span className="block text-[10px] leading-4 text-sub">floor plan from HouseExpo (MIT)</span>}</span>
            </button>)}
          </div>
          <div className="mt-2"><h2 className="mb-2 text-callout font-extrabold">How many robots?</h2><div className="flex gap-2">{[1, 2, 3].map(n => <button key={n} aria-pressed={robots === n} className={`${button} flex-1 px-2 ${robots === n ? 'border-heat bg-heat/10 text-ink' : ''}`} onClick={() => { if (robots !== n) { setRun(null); setError(''); setRobots(n); } }}>{n} {n === 1 ? 'robot' : 'robots'}</button>)}</div></div>
          <button data-start disabled={!run || !ready} onClick={start} className="mt-2 rounded-full bg-heat px-6 py-3.5 text-callout font-extrabold text-heat-ink transition-colors hover:bg-amber disabled:opacity-50">{run && ready ? 'Start rescue' : 'Loading rescue...'}</button>
          <p className="text-footnote text-mute">Same people and fire each time. Light smoke. The EASY setup is fixed.</p>
        </section>}
        <section ref={stage} className="min-w-0 scroll-mt-24" aria-label="Rescue replay" data-setup={active ? run?.setup : undefined} data-first-frame={firstFrame}>
          <div className="overflow-hidden rounded-3xl border border-line bg-stage">
            <div className="flex min-h-[60px] flex-wrap items-center justify-between gap-2 border-b border-line px-5 py-3">
              <div><span className="text-callout font-extrabold">{house.name}</span><span className="ml-3 text-footnote text-sub">{active ? 'Recorded simulation' : `${house.rooms} rooms · 5 people`}</span></div>
              {active && <button className={button} onClick={() => setOverview(!overview)} aria-pressed={overview}>{overview ? 'Follow robots' : 'Full house'}</button>}
            </div>
            <div className={`relative w-full ${active ? 'h-[clamp(380px,62vh,700px)]' : 'h-[430px] lg:h-[570px]'}`}>
              <Scene house={house} run={run} robots={robots} clock={clock} time={t} overview={overview} running={active} reduced={reduced} onReady={onReady} onFrame={onFrame} />
              {!active && <div className="pointer-events-none absolute bottom-5 left-5 rounded-full border border-line2 bg-stage/90 px-4 py-2 text-footnote font-bold text-sub">The team starts at the front door</div>}
              {active && <div className="pointer-events-none absolute left-4 top-4 flex flex-wrap gap-2 text-footnote font-extrabold"><span className="rounded-full bg-stage/90 px-3 py-1.5 text-whiteheat">{done ? run?.score.people_found : foundCount} / 5 people found</span><span className="rounded-full bg-stage/90 px-3 py-1.5 text-sub">{rooms} / {house.rooms} rooms searched</span></div>}
            </div>
            {active && run && <>
              <p data-caption aria-live="polite" className="min-h-[60px] border-t border-line bg-panel/70 px-5 py-4 text-callout font-semibold text-ink">{caption(house, run, t)}</p>
              <div className="grid gap-3 px-5 py-4">
                <div className="flex flex-wrap items-center gap-2"><button className={button} onClick={() => { if (t >= 180) seek(0); setPlaying(!playing); }}>{playing ? 'Pause' : t >= 180 ? 'Replay' : 'Play'}</button><button className={button} onClick={() => setRate(rate === 4 ? 1 : rate === 1 ? 2 : 4)} aria-label={`Playback speed ${rate}x`}>{rate}x</button><span className="ml-auto text-footnote tabular-nums text-sub">{timeLabel(t)} / 3:00</span></div>
                <input aria-label="Replay time" type="range" min={0} max={180} step={0.5} value={t} onChange={e => seek(Number(e.target.value))} className="w-full cursor-pointer accent-heat" />
              </div>
            </>}
          </div>
          {done && run && <section data-end-card aria-label="Rescue result" className="mt-5 rounded-3xl border border-line bg-panel/60 p-6 sm:p-8">
            <p className="mb-2 text-footnote font-extrabold uppercase tracking-[0.14em] text-heat">Rescue result</p><h2 className="font-stencil text-title2">{exitCaption(robots, run.score.robots_out)}</h2>
            <div className="my-6 grid grid-cols-3 gap-3"><Metric value={`${run.score.people_found} / 5`} label="People found" /><Metric value={timeLabel(run.score.time_s)} label={run.score.all_out ? 'Time to exit' : 'Run duration'} /><Metric value={`${run.rooms_searched} / ${house.rooms}`} label="Rooms searched" /></div>
            {robots > 1 && one && <div className="border-t border-line pt-5"><h3 className="mb-3 text-callout font-extrabold">The same house with 1 robot</h3><div className="grid grid-cols-3 gap-3"><Metric value={`${one.people_found} / 5`} label="People found" /><Metric value={timeLabel(one.time_s)} label={one.all_out ? 'Time to exit' : 'Run duration'} /><Metric value={`${one.rooms_searched} / ${house.rooms}`} label="Rooms searched" /></div><p className="mt-3 text-footnote text-sub">{exitCaption(1, one.robots_out)} Same people, fire and EASY setting.</p></div>}
            <p className="mt-5 text-footnote text-mute">A room counts as searched when the robot maps cover at least 80% of its floor. Finding someone records their location for the crew.</p>
          </section>}
          {active && <button className={`${button} mt-5`} onClick={reset}>Choose another house</button>}
          <p className="mt-3 max-w-[90ch] text-footnote text-mute">Robot paths, person confirmations and fire come from a recorded headless simulation. Smoke haze and furniture height are illustrations. <Link className="text-sub underline underline-offset-4" href="/#mission">Watch the filmed mission</Link>.</p>
          {house.source === 'HouseExpo' && <p className="mt-2 text-footnote text-mute">floor plan from HouseExpo (MIT). <a className="underline underline-offset-4" href="/rescue-data/houseexpo-license.txt">Read the license</a>.</p>}
        </section>
      </div>}
    </div>
  </main>;
}
function Metric({ value, label }: { value: string; label: string }) { return <div><p className="font-stencil text-title3 text-whiteheat sm:text-title2">{value}</p><p className="mt-1 text-footnote text-sub">{label}</p></div>; }
