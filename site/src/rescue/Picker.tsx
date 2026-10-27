'use client';
import { useEffect, useState } from 'react';
import { KIND_COLOR } from '@/lib/plans';
import { fireRoom, isFilmHouse, missionHref } from './missions';
import type { House } from './data';
export function Thumb({ house, large = false }: { house: House; large?: boolean }) {
  const [x0, y0, x1, y1] = house.bounds;
  return <svg className={large ? 'h-full w-full' : 'h-16 w-24 shrink-0'} viewBox={`${x0 - 1} ${-y1 - 1} ${x1 - x0 + 2} ${y1 - y0 + 2}`} aria-hidden="true">
    {house.rooms_poly.map(r => <polygon key={r.id} points={r.poly.map(p => `${p[0]},${-p[1]}`).join(' ')} fill={KIND_COLOR[r.kind] || '#3a2350'} />)}
    {house.walls.map((w, i) => <line key={i} x1={w[0]} y1={-w[1]} x2={w[2]} y2={-w[3]} stroke="#cdbcc9" strokeWidth={0.1} />)}
    <circle cx={house.pads[0][0]} cy={-house.pads[0][1]} r={0.23} fill="#f98c0a" />
  </svg>;
}
export default function Picker({ onStart, initialHouse, initialRobots = 2 }: { onStart?: (house: House, robots: number) => void; initialHouse?: string; initialRobots?: number }) {
  const [houses, setHouses] = useState<House[]>([]), [selected, setSelected] = useState(initialHouse || ''), [robots, setRobots] = useState(initialRobots);
  const [error, setError] = useState('');
  useEffect(() => {
    const controller = new AbortController();
    fetch('/rescue-data/gallery.json', { signal: controller.signal }).then(r => { if (!r.ok) throw Error('The houses could not load. Reload to try again.'); return r.json(); }).then(g => setHouses(g.houses)).catch(e => { if (e.name !== 'AbortError') setError(e.message); });
    return () => controller.abort();
  }, []);
  const house = houses.find(h => h.id === selected) || houses[0];
  const filmed = house && isFilmHouse(house);
  // Only robot counts with a checked replay are offered; the rest would end on "not ready".
  const counts = house && !filmed && house.available_replays?.length ? house.available_replays : [1, 2, 3];
  useEffect(() => { if (house && !filmed && !counts.includes(robots)) setRobots(counts.includes(2) ? 2 : counts[0]); }, [house, filmed, counts, robots]);
  const Heading = onStart ? 'h1' : 'h2';
  return <section aria-label="Choose a mission" className="gutter mx-auto max-w-[1440px] py-16 sm:py-24">
    <header className="mb-10 max-w-[800px]"><p className="mb-3 text-footnote font-extrabold uppercase tracking-[0.16em] text-heat">Choose a mission</p><Heading className="font-stencil text-title1 text-ink">Pick a house.<br />Send in the dogs.</Heading><p className="mt-5 max-w-[60ch] text-callout text-sub">Watch a filmed mission in one of our three houses, or run a live simulation in a real floor plan.</p></header>
    {error && <p role="alert">{error}</p>}
    {!house && !error && <p role="status">Loading the five houses...</p>}
    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">{houses.map(h => <button key={h.id} data-house={h.id} aria-pressed={house?.id === h.id} onClick={() => setSelected(h.id)} className={`grid grid-cols-[100px_1fr] items-center sm:block min-w-0 overflow-hidden rounded-3xl border text-left transition-colors ${house?.id === h.id ? 'border-heat bg-panel' : 'border-line bg-stage hover:border-line2'}`}>
      <span className="flex aspect-[4/3] items-center justify-center bg-stage p-3 sm:p-6"><Thumb house={h} large /></span>
      <span className="block p-4"><span className="block text-callout font-extrabold">{h.name}</span><span className="mt-2 block text-footnote text-sub">{h.rooms} rooms, {h.people} people inside. {fireRoom(h)}.</span><span className="mt-3 inline-block rounded-full border border-line2 px-2.5 py-1 text-[11px] font-bold text-whiteheat">{isFilmHouse(h) ? 'Full film' : 'Live on AWS'}</span>{h.placeholder && <span className="mt-2 block text-[11px] text-sub">Plan preview awaiting the eight-room rebuild.</span>}{h.source === 'HouseExpo' && <span className="mt-2 block text-[10px] text-sub">floor plan from HouseExpo (MIT)</span>}</span>
    </button>)}</div>
    {house && <div className="mt-8 flex flex-wrap items-end justify-between gap-6">
      {filmed ? <p data-fixed-team className="max-w-[42ch] text-callout text-sub">Two robots. A recorded Unity mission, from arrival to exit.</p> : <div><p className="mb-3 text-callout font-bold">How many robots?</p><div className="flex gap-2" aria-label="Robot count">{counts.map(n => <button key={n} aria-pressed={robots === n} onClick={() => setRobots(n)} className={`min-h-11 rounded-full border px-5 py-2 font-bold ${robots === n ? 'border-heat bg-heat/10 text-ink' : 'border-line2 text-sub'}`}>{n} {n === 1 ? 'robot' : 'robots'}</button>)}</div><p className="mt-3 text-footnote text-sub">Light smoke. Same people and fire each time.</p></div>}
      {onStart ? <button data-start className="rounded-full bg-heat px-8 py-4 text-callout font-extrabold text-heat-ink hover:bg-amber" onClick={() => onStart(house, filmed ? 2 : robots)}>Send them in</button> : <a data-start className="rounded-full bg-heat px-8 py-4 text-callout font-extrabold text-heat-ink hover:bg-amber" href={missionHref(house, robots)}>Send them in</a>}
    </div>}
  </section>;
}
