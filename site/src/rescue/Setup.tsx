'use client';
import { useEffect, useRef, useState } from 'react';
import { KIND_COLOR } from '@/lib/plans';
import { fireRoom } from './missions';
import type { House } from './data';
export const ROBOT_COLOR = ['#f6a31c', '#2f9be0', '#3fd28a'];
const TOTAL_MS = 5200, HOLD_MS = 650;
// Each line names a real input of this mission. The sequence runs while the film buffers or AWS starts the run.
export function setupSteps(house: House, robots: number, live: boolean) {
  return [
    live ? 'Starting the simulation on AWS' : 'Loading the Unity house',
    `Floor plan: ${house.rooms} rooms, ${house.people} people inside`,
    fireRoom(house),
    ...Array.from({ length: robots }, (_, i) => `Robot ${'ABC'[i]} online: camera, thermal, LiDAR`),
    ...(robots > 1 ? ['Radio link up: one shared map'] : []),
    'Robots at the front door',
  ];
}
export default function Setup({ house, robots, live, ready, onDone }: { house: House; robots: number; live: boolean; ready: boolean; onDone: () => void }) {
  const steps = setupSteps(house, robots, live), [shown, setShown] = useState(1), done = useRef(onDone);
  done.current = onDone;
  useEffect(() => {
    const fast = matchMedia('(prefers-reduced-motion: reduce)').matches;
    const id = setInterval(() => setShown(n => Math.min(steps.length, n + 1)), fast ? 200 : TOTAL_MS / steps.length);
    return () => clearInterval(id);
  }, [steps.length]);
  const complete = shown >= steps.length;
  useEffect(() => {
    if (!complete || !ready) return;
    const id = setTimeout(() => done.current(), HOLD_MS);
    return () => clearTimeout(id);
  }, [complete, ready]);
  const [x0, y0, x1, y1] = house.bounds, fire = house.fire_origins?.[0], fireStep = 3, robotStep = (i: number) => 4 + i, radioStep = 4 + robots;
  return <div data-setup-screen className="absolute inset-0 flex items-center justify-center overflow-hidden px-5 py-6 sm:px-10">
    <div className="grid w-full max-w-[1180px] items-center gap-8 md:grid-cols-[1.25fr_1fr] md:gap-14">
      <svg className="mx-auto max-h-[38dvh] w-full md:max-h-[68dvh]" viewBox={`${x0 - 1.2} ${-y1 - 1.2} ${x1 - x0 + 2.4} ${y1 - y0 + 2.4}`} aria-hidden="true">
        <defs><pattern id="setup-dots" width="1" height="1" patternUnits="userSpaceOnUse"><circle cx="0.5" cy="0.5" r="0.045" fill="#2b1c3a" /></pattern></defs>
        <rect x={x0 - 1.2} y={-y1 - 1.2} width={x1 - x0 + 2.4} height={y1 - y0 + 2.4} fill="url(#setup-dots)" />
        {house.rooms_poly.map((r, i) => <polygon key={r.id} points={r.poly.map(p => `${p[0]},${-p[1]}`).join(' ')} fill={KIND_COLOR[r.kind] || '#3a2350'} style={{ animation: `setup-fade 0.9s var(--ease-out-quart) ${0.9 + i * 0.06}s both` }} />)}
        {house.walls.map((w, i) => <line key={i} x1={w[0]} y1={-w[1]} x2={w[2]} y2={-w[3]} pathLength={1} strokeDasharray="1" stroke="#cdbcc9" strokeWidth={0.1} strokeLinecap="round" style={{ animation: `setup-draw 1.1s var(--ease-out-quart) ${Math.min(0.9, i * 0.015)}s both` }} />)}
        {fire && shown >= fireStep && <g style={{ animation: 'setup-fade 0.5s both' }}><circle className="setup-ping" cx={fire[0]} cy={-fire[1]} r={0.35} fill="none" stroke="#f04a1c" strokeWidth={0.08} /><circle cx={fire[0]} cy={-fire[1]} r={0.32} fill="#f04a1c" /></g>}
        {Array.from({ length: robots }, (_, i) => {
          const pad = house.pads[i] || house.pads[0];
          return shown >= robotStep(i) && <g key={i} style={{ animation: 'setup-rise 0.6s var(--ease-spring) both' }}>
            {shown >= radioStep && robots > 1 && <circle className="setup-ping" cx={pad[0]} cy={-pad[1]} r={0.3} fill="none" stroke={ROBOT_COLOR[i]} strokeWidth={0.06} style={{ animationDelay: `${i * 0.4}s` }} />}
            <circle cx={pad[0]} cy={-pad[1]} r={0.3} fill={ROBOT_COLOR[i]} />
            <text x={pad[0] + 0.55} y={-pad[1] + 0.22} fontSize={0.62} fontWeight={800} fill={ROBOT_COLOR[i]}>{'ABC'[i]}</text>
          </g>;
        })}
      </svg>
      <div>
        <p className="mb-2 text-footnote font-extrabold uppercase tracking-[0.18em] text-heat">Mission setup</p>
        <h2 className="font-stencil text-title2">{house.name}</h2>
        <ol className="mt-6 space-y-2.5 sm:mt-8 sm:space-y-3">{steps.map((s, i) => {
          const visible = i < shown, finished = i < shown - 1 || (complete && ready);
          return <li key={s} className="flex items-center gap-3 text-callout" style={{ opacity: visible ? 1 : 0, transform: visible ? 'none' : 'translateY(6px)', transition: 'opacity .45s var(--ease-out-quart), transform .45s var(--ease-out-quart)' }}>
            <span className="flex h-5 w-5 shrink-0 items-center justify-center">{finished ? <svg viewBox="0 0 16 16" className="h-4 w-4 text-whiteheat" aria-hidden="true"><path d="m3 8.5 3.2 3L13 4.5" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" /></svg> : <span className="h-4 w-4 rounded-full border-2 border-heat border-t-transparent" style={{ animation: 'setup-spin 0.8s linear infinite' }} />}</span>
            <span className={finished ? 'text-sub' : 'font-semibold text-ink'}>{s}</span>
          </li>;
        })}</ol>
        {complete && !ready && <p role="status" className="mt-5 text-footnote text-sub">{live ? 'Waiting for the first frames from AWS...' : 'Loading the mission...'}</p>}
        <div className="mt-7 h-[3px] overflow-hidden rounded-full bg-white/[0.07]"><div className="h-full rounded-full bg-heat" style={{ width: `${(100 * Math.min(shown, steps.length - (complete && ready ? 0 : 0.5))) / steps.length}%`, transition: 'width .6s var(--ease-out-quart)' }} /></div>
      </div>
    </div>
  </div>;
}
