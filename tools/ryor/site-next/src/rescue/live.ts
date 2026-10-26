import { API } from '@/lib/api';
import { inside, type Frame, type House, type Recording, type Score } from './data';
export type Status = { setup: string; state: string; chunks: number; validated?: boolean; score?: Score; engine?: { git: string; image: string }; sim_t?: number; error?: string };
export type Chunk = { chunk: number; frames: (Omit<Frame, 'mapped'> & { seen: Record<string, number[]> })[] };
const emptyScore = (n: number, total: number): Score => ({ people_found: 0, people_total: total, time_s: 180, robots_out: 0, robots: n, all_out: false, coverage_pct: 0, rooms_searched: 0 });
export function streamBuilder(house: House, n: number, key: string) {
  const union = new Set<number>(), frames: Frame[] = [];
  const rooms = house.rooms_poly.map(room => {
    const cells = new Set<number>(); let total = 0;
    const xs = room.poly.map(p => p[0]), ys = room.poly.map(p => p[1]);
    for (let y = Math.floor((Math.min(...ys) + 25) * 10); y <= Math.ceil((Math.max(...ys) + 25) * 10); y++) for (let x = Math.floor((Math.min(...xs) + 25) * 10); x <= Math.ceil((Math.max(...xs) + 25) * 10); x++) {
      if (inside(room.poly, -25 + (x + .5) / 10, -25 + (y + .5) / 10)) { total++; if (x >= 0 && x < 500 && y >= 0 && y < 500) cells.add(y * 500 + x); }
    }
    return { cells, total };
  });
  let next = 0;
  return { get next() { return next; }, append(chunks: Chunk[], status: Status): Recording {
    if (status.setup !== key) throw Error('AWS returned a different setup.');
    for (const chunk of chunks) {
      if (chunk.chunk < next) continue;
      if (chunk.chunk !== next) throw Error('The live recording has a missing chunk.');
      for (const f of chunk.frames) {
        if (f.t !== frames.length / 2 || f.robots.length !== n || f.robots.some((p, i) => p.id !== `robot_${'ABC'[i]}` || ![p.x, p.y, p.yaw].every(Number.isFinite))) throw Error('The live recording has invalid frames.');
        for (const cells of Object.values(f.seen)) for (const c of cells) union.add(c);
        const mapped = rooms.map(r => Math.floor(100 * [...r.cells].filter(c => union.has(c)).length / Math.max(1, r.total)));
        frames.push({ t: f.t, robots: f.robots, fire: f.fire, found: f.found, events: f.events, mapped });
      }
      next++;
    }
    const score = status.score ? { ...status.score, robots: n, all_out: status.score.robots_out === n } : emptyScore(n, house.people);
    const complete = status.state === 'done' && next >= status.chunks;
    if (complete && (!status.validated || frames.length !== Math.round((status.sim_t || 180) * 2) + 1 || score.false_confirmations !== 0)) throw Error('The completed live recording failed its checks.');
    return { setup: key, duration_s: status.state === 'done' ? status.sim_t || 180 : house.mission_s || 180, difficulty: 'easy', frames: [...frames], score, rooms_searched: frames.at(-1)?.mapped.filter(p => p >= 80).length || 0, validated: complete, engine: status.engine };
  } };
}
async function request(path: string, signal: AbortSignal, body?: unknown) {
  const timeout = AbortSignal.timeout(25000);
  const r = await fetch(`${API}/api/mission${path}`, { signal: AbortSignal.any([signal, timeout]), ...(body ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } : {}) });
  if (!r.ok) throw Error(`AWS answered ${r.status}.`);
  return r.json();
}
export async function startLive(house: House, n: number, engine: string | null, signal: AbortSignal, update: (r: Recording) => void) {
  if (!engine) throw Error('The updated engine is not ready yet.');
  const key = `${house.id}__r${n}__easy`, stream = streamBuilder(house, n, key);
  const started = performance.now();
  const launch = await request('', signal, { house: house.id, robots: n, difficulty: 'easy', terra: true });
  if (launch.setup !== key || launch.source !== 'live' || !/^[a-z0-9]{8,40}$/.test(launch.mission)) throw Error('AWS could not start a live mission.');
  while (!signal.aborted) {
    if (performance.now() - started >= 600000) throw Error('The live mission timed out.');
    const reply = await request(`/${launch.mission}?from=${stream.next}`, signal);
    if (reply.status.state === 'failed') throw Error(reply.status.error || 'The live mission stopped.');
    if (reply.status.engine?.git && reply.status.engine.git !== engine) throw Error('AWS is running a different engine version.');
    const run = stream.append(reply.chunks, reply.status);
    if (run.frames.length) update(run);
    if (run.validated) return;
    await new Promise<void>((resolve, reject) => {
      const stop = () => { clearTimeout(timer); reject(signal.reason); };
      const timer = setTimeout(() => { signal.removeEventListener('abort', stop); resolve(); }, 1500);
      signal.addEventListener('abort', stop, { once: true });
    });
  }
}
