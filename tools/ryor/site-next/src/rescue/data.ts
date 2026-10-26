export type Pose = { id: string; x: number; y: number; yaw: number; state: string; out: boolean };
export type Event = { robot: string; kind: string; text: string; t: number };
export type Found = { id: string; x: number; y: number; by: string; t: number };
export type Frame = { t: number; robots: Pose[]; fire: number[][]; found: Found[]; events: Event[]; mapped: number[] };
export type Score = { people_found: number; people_total: number; time_s: number; robots_out: number; robots: number; all_out: boolean; coverage_pct: number; rooms_searched: number };
export type House = { id: string; name: string; rooms: number; people: number; source: string; bounds: number[]; walls: number[][]; furniture: number[][][]; pads: number[][]; people_truth: { id: string; x: number; y: number }[]; rooms_poly: { id: string; name: string; kind: string; poly: number[][] }[]; scores: Record<string, Score> };
export type Recording = { setup: string; difficulty: 'easy'; frames: Frame[]; score: Score; rooms_searched: number; validated: boolean };
export const keyFor = (house: House, n: number) => `${house.id}__r${n}__easy`;
const recordings = new Map<string, Promise<Recording>>();
export function loadRecording(key: string) {
  let promise = recordings.get(key);
  if (!promise) {
    promise = fetch(`/rescue-data/${key}.json`).then(async r => {
      if (!r.ok) throw new Error('The recording could not load. Please try again.');
      const data: Recording = await r.json();
      if (!data.validated || data.difficulty !== 'easy' || data.setup !== key || data.frames.length !== 361) throw new Error('This recording failed its checks. Please try again.');
      return data;
    }).catch(e => { recordings.delete(key); throw e; });
    recordings.set(key, promise);
  }
  return promise;
}
export function inside(poly: number[][], x: number, y: number) {
  let result = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [a, b] = poly[i], [c, d] = poly[j];
    if ((b > y) !== (d > y) && x < (c - a) * (y - b) / (d - b) + a) result = !result;
  }
  return result;
}
export const robotName = (id: string) => id.replace('robot_', 'Robot ');
export const timeLabel = (t: number) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, '0')}`;
export function roomAt(house: House, x: number, y: number) {
  const room = house.rooms_poly.find(r => inside(r.poly, x, y));
  return room?.name?.toLowerCase().replace(/_/g, ' ') || 'house';
}
export function exitCaption(n: number, out: number) {
  if (out === n) return n === 1 ? 'The robot is out.' : n === 2 ? 'Both robots are out.' : 'All three robots are out.';
  return `${out} of ${n} ${n === 1 ? 'robot' : 'robots'} returned after entering the house.`;
}
// Captions use recorded events and poses. Internal brain reasons never enter the page.
export function caption(house: House, run: Recording, t: number) {
  if (t >= 180) return exitCaption(run.score.robots, run.score.robots_out);
  const frames = run.frames.slice(0, Math.floor(t * 2) + 1);
  const events = frames.flatMap(f => f.events.map(e => ({ e, f }))).filter(({ e }) => e.t <= t);
  const meaningful = events.filter(({ e }) => ['survivor_confirmed', 'mission_complete', 'room_searched', 'room_marked_fire', 'agent_decision', 'agent_unavailable', 'agent_order'].includes(e.kind));
  const last = meaningful.at(-1);
  if (last) {
    const { e, f } = last;
    const p = f.robots.find(r => r.id === e.robot);
    const room = p ? roomAt(house, p.x, p.y) : 'house';
    const who = robotName(e.robot);
    if (e.kind === 'survivor_confirmed') {
      const person = f.found.find(p => p.by === e.robot && Math.abs(p.t - e.t) < 0.1);
      return person ? `${who} found someone in the ${roomAt(house, person.x, person.y)}.` : `${who} confirmed a person's location.`;
    }
    if (e.kind === 'mission_complete') return p?.out ? `${who} is outside.` : `${who} is heading back to the front door.`;
    if (e.kind === 'room_searched') return `${who} finished a room search.`;
    if (e.kind === 'room_marked_fire') return `${who} detected fire in the ${room}.`;
    if (e.kind === 'agent_decision') return `Terra reviewed ${who}'s report.`;
    if (e.kind === 'agent_unavailable') return 'Terra did not answer. The robots kept searching.';
    if (e.kind === 'agent_order') return `${who} received a search instruction from Terra.`;
  }
  return t < 6 ? 'The robots enter through the front door.' : 'The robots search the rooms and share what they find.';
}

// An animation frame timestamp can precede the effect that scheduled it.
export const advanceClock = (t: number, elapsed: number, rate: number) => Math.min(180, Math.max(0, t) + Math.max(0, Math.min(elapsed, 0.1)) * rate);
