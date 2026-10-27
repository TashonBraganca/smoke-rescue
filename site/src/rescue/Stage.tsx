'use client';
import { useEffect, type ReactNode } from 'react';
// The films are drawn on #0a0910, so the stage uses the same colour and the letterbox disappears.
export const STAGE_BG = '#0a0910';
export type StageStatus = 'setup' | 'running' | 'done' | 'error';
const STATUS: Record<StageStatus, string> = { setup: 'Setting up', running: 'Mission running', done: 'Mission complete', error: 'Stopped' };
export const clockLabel = (s: number) => { const t = Math.max(0, Math.floor(s)); return `${String(Math.floor(t / 60)).padStart(2, '0')}:${String(t % 60).padStart(2, '0')}`; };
const chip = 'inline-flex h-8 items-center gap-2 rounded-full border border-white/10 px-3 text-[13px] font-bold text-sub transition-colors hover:border-heat hover:text-ink';
export default function Stage({ title, status, clock, tag, onReset, children }: { title: string; status: StageStatus; clock: number; tag: string; onReset: () => void; children: ReactNode }) {
  useEffect(() => {
    const html = document.documentElement, before = html.style.overflow;
    html.style.overflow = 'hidden';
    return () => { html.style.overflow = before; };
  }, []);
  const live = status === 'running';
  return <section aria-label={`${title} mission`} data-stage={status} className="fixed inset-0 z-50 flex flex-col text-ink" style={{ background: STAGE_BG }}>
    <header className="flex h-14 shrink-0 items-center gap-3 border-b border-white/[0.06] px-3 sm:gap-5 sm:px-5">
      <button onClick={onReset} className={chip} aria-label="Back to the houses"><svg viewBox="0 0 16 16" className="h-3.5 w-3.5" aria-hidden="true"><path d="M10 3 5 8l5 5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" /></svg><span className="hidden sm:inline">Houses</span></button>
      <p className="min-w-0 truncate font-stencil text-lg tracking-wide sm:text-xl">{title}</p>
      <p className="hidden items-center gap-2 text-[13px] font-bold text-sub md:flex" role="status">
        <span className="relative flex h-2.5 w-2.5">{live && <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-heat opacity-60" />}<span className={`relative inline-flex h-2.5 w-2.5 rounded-full ${status === 'done' ? 'bg-whiteheat' : status === 'error' ? 'bg-coal' : status === 'setup' ? 'bg-amber' : 'bg-heat'}`} /></span>{STATUS[status]}
      </p>
      <div className="ml-auto flex items-center gap-2 sm:gap-3">
        <p className="hidden text-[12px] font-semibold text-mute lg:block">{tag}</p>
        <p className="flex items-baseline gap-2"><span className="hidden text-[11px] font-bold uppercase tracking-[0.14em] text-mute sm:inline">Mission time</span><span data-clock className="font-mono text-xl font-bold tabular-nums tracking-tight sm:text-2xl">{clockLabel(clock)}</span></p>
      </div>
    </header>
    <div className="relative min-h-0 flex-1">{children}</div>
  </section>;
}
// Shared result card shown over the last frame of a mission.
export function ResultCard({ title, stats, note, children }: { title: string; stats: [string, string][]; note?: ReactNode; children: ReactNode }) {
  return <div className="absolute inset-0 flex items-center justify-center p-4" style={{ background: 'radial-gradient(ellipse at center, rgba(10,9,16,0.72), rgba(10,9,16,0.9))', animation: 'setup-fade 0.7s var(--ease-out-quart) both' }}>
    <section data-end-card aria-label="Mission result" className="w-full max-w-[640px] rounded-3xl border border-white/10 bg-panel/90 p-6 shadow-2xl backdrop-blur sm:p-9" style={{ animation: 'setup-rise 0.8s var(--ease-spring) 0.15s both' }}>
      <p className="mb-3 text-footnote font-extrabold uppercase tracking-[0.16em] text-heat">Mission complete</p>
      <h2 className="font-stencil text-title2">{title}</h2>
      <div className="my-7 grid grid-cols-3 gap-4">{stats.map(([value, label]) => <p key={label}><span className="block font-stencil text-title2 tabular-nums">{value}</span><span className="text-footnote text-sub">{label}</span></p>)}</div>
      {note}
      <div className="mt-7 flex flex-wrap gap-3">{children}</div>
    </section>
  </div>;
}
