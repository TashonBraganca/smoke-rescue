'use client';
import { useEffect, useMemo, useRef } from 'react';
import { inside, type House, type Recording } from './data';
import { ROBOT_COLOR } from './Setup';
import { clockLabel } from './Stage';
// Film-style panels for a live run. Every number comes from the streamed frames: robot poses, per-room searched
// percentages, fire, confirmations and the robots' own radio events.
const area = (poly: number[][]) => Math.abs(poly.reduce((s, p, i) => { const q = poly[(i + 1) % poly.length]; return s + p[0] * q[1] - q[0] * p[1]; }, 0)) / 2;
const letter = (id: string) => id.replace('robot_', '');
const colour = (id: string) => ROBOT_COLOR['ABC'.indexOf(letter(id))] || '#cdbcc9';
const TAG: Record<string, [string, string]> = {
  room_searched: ['room done', 'text-[#3fd28a] bg-[#3fd28a]/10'], door_taken: ['doorway claimed', 'text-amber bg-amber/10'],
  survivor_from_peer: ['radio relay', 'text-[#7cc4f5] bg-[#2f9be0]/10'], map_shared: ['map shared', 'text-[#7cc4f5] bg-[#2f9be0]/10'],
  mission_complete: ['heading out', 'text-whiteheat bg-whiteheat/10'], room_marked_fire: ['fire', 'text-[#ff7a59] bg-[#f04a1c]/15'],
  agent_decision: ['Terra', 'text-[#c9a6ff] bg-[#8b5cf6]/15'], agent_order: ['Terra', 'text-[#c9a6ff] bg-[#8b5cf6]/15'], agent_unavailable: ['Terra', 'text-[#c9a6ff] bg-[#8b5cf6]/15'],
};
const panel = 'flex min-h-0 flex-col overflow-hidden rounded-2xl border border-white/[0.07] bg-[#100d19] p-4 sm:p-5';
export default function LiveDash({ house, run, t, live }: { house: House; run: Recording; t: number; live: boolean }) {
  const idx = Math.min(run.frames.length - 1, Math.floor(t * 2));
  const rooms = useMemo(() => house.rooms_poly.map(r => area(r.poly)), [house]);
  const total = rooms.reduce((a, b) => a + b, 0);
  const searchedAt = (i: number) => rooms.reduce((s, a, k) => s + a * Math.min(100, run.frames[i]?.mapped[k] || 0) / 100, 0);
  const frame = run.frames[idx];
  const searched = frame ? searchedAt(idx) : 0;
  const spark = useMemo(() => { const pts: number[] = []; for (let i = 0; i <= idx; i += Math.max(1, Math.floor(idx / 60))) pts.push(searchedAt(i)); return pts; }, [idx, run]); // eslint-disable-line react-hooks/exhaustive-deps
  const trails = useMemo(() => {
    const out: Record<string, number[][]> = {};
    for (let i = 0; i <= idx; i += 2) for (const p of run.frames[i].robots) if (!p.out) (out[p.id] ||= []).push([p.x, -p.y]);
    return out;
  }, [idx, run]);
  const found = useMemo(() => {
    const list: { x: number; y: number; by: string }[] = [];
    for (const f of run.frames.slice(0, idx + 1)) for (const p of f.found) if (!list.some(q => Math.hypot(q.x - p.x, q.y - p.y) < 1)) list.push(p);
    return list;
  }, [idx, run]);
  const events = useMemo(() => run.frames.slice(0, idx + 1).flatMap(f => f.events.map(e => ({ e, f }))).slice(-40), [idx, run]);
  const radio = useRef<HTMLOListElement>(null);
  useEffect(() => { radio.current?.scrollTo({ top: radio.current.scrollHeight, behavior: 'smooth' }); }, [events.length]);
  const [x0, y0, x1, y1] = house.bounds;
  const notSearched = Math.max(0, total - searched);
  const max = Math.max(1, total);
  const line = (e: (typeof events)[number]) => {
    const { e: ev, f } = e, who = letter(ev.robot);
    if (ev.kind === 'room_searched') {
      const p = f.robots.find(r => r.id === ev.robot), k = p ? house.rooms_poly.findIndex(r => inside(r.poly, p.x, p.y)) : -1;
      if (k >= 0) return `${who} searched a ${Math.round(rooms[k])} m² ${house.rooms_poly[k].name.toLowerCase().replaceAll('_', ' ')}, ${Math.min(100, f.mapped[k])}% seen`;
    }
    if (ev.kind === 'door_taken') return `${who} takes the next doorway; the team leaves that room to it`;
    if (ev.kind === 'survivor_from_peer') return `${who} heard about a person over the radio`;
    if (ev.kind === 'map_shared') return `${who} linked its map with its partner`;
    if (ev.kind === 'mission_complete') return `${who} is walking out along its trail`;
    return ev.text.replace(/^Robot ([ABC])/, '$1');
  };
  return <div className="grid min-h-0 gap-3 md:grid-cols-2">
    <section className={panel} aria-label="Shared map">
      <div className="flex min-h-0 flex-1 gap-4">
        <svg className="h-full min-h-[160px] w-[48%] shrink-0" viewBox={`${x0 - 0.8} ${-y1 - 0.8} ${x1 - x0 + 1.6} ${y1 - y0 + 1.6}`} aria-hidden="true">
          <defs><pattern id="dash-dots" width="0.5" height="0.5" patternUnits="userSpaceOnUse"><circle cx="0.25" cy="0.25" r="0.035" fill="#2b2240" /></pattern></defs>
          <rect x={x0 - 0.8} y={-y1 - 0.8} width={x1 - x0 + 1.6} height={y1 - y0 + 1.6} fill="url(#dash-dots)" />
          {house.rooms_poly.map((r, k) => <polygon key={r.id} points={r.poly.map(p => `${p[0]},${-p[1]}`).join(' ')} fill="#8b3fd6" opacity={0.08 + 0.55 * Math.min(100, frame?.mapped[k] || 0) / 100} style={{ transition: 'opacity .6s' }} />)}
          {house.walls.map((w, i) => <line key={i} x1={w[0]} y1={-w[1]} x2={w[2]} y2={-w[3]} stroke="#d9d0e0" strokeWidth={0.12} strokeLinecap="round" />)}
          {frame?.fire.map((p, i) => <g key={i}><circle className="setup-ping" cx={p[0]} cy={-p[1]} r={0.3} fill="none" stroke="#f04a1c" strokeWidth={0.06} /><circle cx={p[0]} cy={-p[1]} r={Math.max(0.25, p[2] || 0.3)} fill="#f04a1c" opacity={0.75} /></g>)}
          {Object.entries(trails).map(([id, pts]) => <polyline key={id} points={pts.map(p => p.join(',')).join(' ')} fill="none" stroke={colour(id)} strokeWidth={0.09} strokeDasharray="0.08 0.22" strokeLinecap="round" opacity={0.8} />)}
          {found.map((p, i) => <g key={i}><circle cx={p.x} cy={-p.y} r={0.42} fill="none" stroke="#3fd28a" strokeWidth={0.1} /><circle cx={p.x} cy={-p.y} r={0.16} fill="#3fd28a" /></g>)}
          {frame?.robots.filter(p => !p.out).map(p => <g key={p.id}><circle cx={p.x} cy={-p.y} r={0.34} fill={colour(p.id)} stroke="#0a0910" strokeWidth={0.08} /><text x={p.x + 0.45} y={-p.y - 0.35} fontSize={0.62} fontWeight={800} fill={colour(p.id)}>{letter(p.id)}</text></g>)}
        </svg>
        <div className="flex min-w-0 flex-1 flex-col">
          <p className="text-[15px] font-bold uppercase tracking-wide">Shared map</p>
          <p className="mt-1 text-[12px] leading-snug text-mute">The team map both robots build and share over radio.</p>
          <p className="mt-3 flex items-baseline gap-2"><span className="text-[clamp(1.6rem,3vw,2.3rem)] font-bold tabular-nums leading-none">{Math.round(searched)} m²</span><span className="text-[13px] text-sub">of {Math.round(total)} m² searched</span></p>
          <svg className="mt-3 h-6 w-full" viewBox={`0 0 ${Math.max(1, spark.length - 1)} ${max}`} preserveAspectRatio="none" aria-hidden="true"><polyline points={spark.map((v, i) => `${i},${max - v}`).join(' ')} fill="none" stroke="#a46be8" strokeWidth={2} vectorEffect="non-scaling-stroke" /></svg>
          <ul className="mt-3 space-y-1.5 text-[13px]">
            <li className="flex justify-between gap-2"><span className="flex items-center gap-2"><span className="h-3 w-3 rounded-sm bg-[#8b3fd6]" />Searched by the team</span><span className="tabular-nums text-sub">{Math.round(searched)} m²</span></li>
            <li className="flex justify-between gap-2"><span className="flex items-center gap-2"><span className="h-3 w-3 rounded-sm bg-[#2b2240]" />Not searched yet</span><span className="tabular-nums text-sub">{Math.round(notSearched)} m²</span></li>
            <li className="flex justify-between gap-2"><span className="flex items-center gap-2"><span className="h-3 w-3 rounded-full bg-[#f04a1c]" />Fire on the robots&apos; maps</span><span className="tabular-nums text-sub">{frame?.fire.length || 0}</span></li>
            <li className="flex justify-between gap-2"><span className="flex items-center gap-2"><span className="h-3 w-3 rounded-full border-2 border-[#3fd28a]" />People found</span><span className="tabular-nums text-sub">{found.length}</span></li>
          </ul>
          <p className="mt-auto pt-2 text-[10.5px] leading-snug text-mute">Walls come from the true house plan, drawn for viewers. The robots never see it.</p>
        </div>
      </div>
    </section>
    <section className={panel} aria-label="Team radio">
      <div className="flex items-baseline justify-between gap-3 border-b border-white/[0.07] pb-3"><p className="flex items-baseline gap-3"><span className="text-[15px] font-bold uppercase tracking-wide">Team radio</span><span className="hidden text-[12px] text-mute sm:inline">what the robots tell each other</span></p><p className="flex items-center gap-1.5 text-[12px] text-sub"><span className={`h-2 w-2 rounded-full ${live ? 'bg-[#3fd28a]' : 'bg-mute'}`} />{live ? 'streaming from AWS' : 'radio link up'}</p></div>
      <ol ref={radio} className="mt-2 min-h-0 flex-1 space-y-1 overflow-y-auto pr-1">
        {events.length === 0 && <li className="py-2 text-[13px] text-mute">Waiting for the first radio message...</li>}
        {events.map((item, i) => {
          const { e } = item, fade = i < events.length - 8 ? 'opacity-50' : '';
          if (e.kind === 'survivor_confirmed') return <li key={i} className={`flex items-center gap-3 rounded-lg border-l-4 border-[#3fd28a] bg-[#3fd28a]/10 px-3 py-1.5 ${fade}`} style={{ animation: 'setup-rise .5s both' }}><span className="font-mono text-[12px] text-[#3fd28a]/80 tabular-nums">{clockLabel(e.t)}</span><span className="text-[15px] font-bold text-[#3fd28a]">SURVIVOR CONFIRMED by {letter(e.robot)}</span></li>;
          const [tag, tone] = TAG[e.kind] || [e.kind.replaceAll('_', ' '), 'text-sub bg-white/5'];
          return <li key={i} className={`flex items-center gap-3 px-1 py-1 ${fade}`} style={{ animation: 'setup-rise .5s both' }}>
            <span className="font-mono text-[12px] text-mute tabular-nums">{clockLabel(e.t)}</span>
            <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-[11px] font-extrabold text-[#0a0910]" style={{ background: colour(e.robot) }}>{letter(e.robot)}</span>
            <span className="min-w-0 flex-1 truncate text-[14px]">{line(item)}</span>
            <span className={`hidden shrink-0 rounded px-2 py-0.5 text-[11px] font-bold sm:inline ${tone}`}>{tag}</span>
          </li>;
        })}
      </ol>
    </section>
  </div>;
}
