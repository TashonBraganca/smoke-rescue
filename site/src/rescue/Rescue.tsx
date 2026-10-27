'use client';
import { useEffect, useState } from 'react';
import Nav from '@/components/Nav';
import Picker from './Picker';
import FilmPlayer from './FilmPlayer';
import LiveMission from './LiveMission';
import { STAGE_BG } from './Stage';
import { isFilmHouse, type MissionConfig } from './missions';
import type { House } from './data';
export default function Rescue() {
  const [mission, setMission] = useState<{ house: House; robots: number } | null>(null), [config, setConfig] = useState<MissionConfig | null>(null), [error, setError] = useState(''), [initial, setInitial] = useState<{ house?: string; robots: number }>({ robots: 2 });
  const [launching, setLaunching] = useState(false);
  useEffect(() => {
    const controller = new AbortController(), params = new URLSearchParams(location.search);
    setLaunching(params.get('start') === '1');
    const robots = [1, 2, 3].includes(Number(params.get('robots'))) ? Number(params.get('robots')) : 2;
    Promise.all([fetch('/rescue-data/missions.json', { signal: controller.signal }).then(r => { if (!r.ok) throw Error('The mission settings could not load.'); return r.json(); }), fetch('/rescue-data/gallery.json', { signal: controller.signal }).then(r => { if (!r.ok) throw Error('The houses could not load.'); return r.json(); })]).then(([c, g]) => {
      setConfig(c);
      setInitial({ house: params.get('house') || undefined, robots });
      const h = (g.houses as House[]).find(h => h.id === params.get('house'));
      if (h && params.get('start') === '1') setMission({ house: h, robots: isFilmHouse(h) ? 2 : robots });
      else { setLaunching(false); if (params.get('mode') === 'live') setInitial({ house: (g.houses as House[]).find(h => !isFilmHouse(h))?.id, robots: 2 }); }
    }).catch(e => { if (e.name !== 'AbortError') setError(e.message); });
    return () => controller.abort();
  }, []);
  const start = (house: House, robots: number) => { setMission({ house, robots }); history.replaceState(null, '', `?house=${encodeURIComponent(house.id)}&robots=${robots}`); window.scrollTo({ top: 0, behavior: 'instant' }); };
  const reset = () => { setMission(null); setLaunching(false); history.replaceState(null, '', '/rescue/'); window.scrollTo({ top: 0, behavior: 'instant' }); };
  // A mission takes the whole screen; while a direct link loads, show the empty stage instead of the page.
  if (mission && config) return <main>{isFilmHouse(mission.house) ? <FilmPlayer house={mission.house} config={config} onReset={reset} /> : <LiveMission house={mission.house} robots={mission.robots} config={config} onReset={reset} />}</main>;
  if (launching && !error) return <main className="fixed inset-0" style={{ background: STAGE_BG }} />;
  return <main className="min-h-dvh pb-14 pt-20"><Nav rescue />{error ? <p role="alert" className="gutter py-20">{error} Reload to try again.</p> : !config ? <p role="status" className="gutter py-20">Loading missions...</p> : <Picker key={`${initial.house}-${initial.robots}`} initialHouse={initial.house} initialRobots={initial.robots} onStart={start} />}</main>;
}
