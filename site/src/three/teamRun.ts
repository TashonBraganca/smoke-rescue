// The mission timeline for a team of 1 to 3 dogs, as pure data. planSweep splits the rooms (nearest-first greedy
// from the front door, the dog that has walked least picks next). Then every dog walks its route at the same speed
// and stops for LOOK seconds in each room it is the first to reach. The same numbers drive the live run, the
// timer and the "other team sizes" line, so what plays and what is reported cannot drift apart.
import { LAG, planSweep, type Layout, type Route } from "./houseLayout";

export const WALK = 1.4; // m/s, the dog's natural trot (one 1.4 m stride per second)
export const LOOK = 2; // seconds a dog stands in a room it reached first, scanning it
const LOOK_IN = 0.9; // metres past the doorway where it stops to scan

export type Stop = { s: number; room: number };
export type DogPlan = { route: Route; start: number; stops: Stop[]; end: number; rooms: number };
export type TeamPlan = {
  dogs: DogPlan[];
  owner: Int8Array; // room index -> dog that searched it (-1 if no route reaches it)
  litAt: Float32Array; // room index -> sim second the room is first entered
  total: number; // sim seconds until the last room has been scanned
};

/** Plan (not play) a team of n dogs through this house. */
export function planTeam(layout: Layout, n: number): TeamPlan {
  const routes = planSweep(layout, n);
  const index = new Map(layout.rooms.map((r, i) => [r.id, i]));
  const R = layout.rooms.length;
  const owner = new Int8Array(R).fill(-1), litAt = new Float32Array(R).fill(Infinity);
  // dogs file in through one front door, so dog k starts LAG metres of walking behind dog k - 1
  const dogs: DogPlan[] = routes.map((route, k) => ({ route, start: (k * LAG) / WALK, stops: [], end: 0, rooms: 0 }));
  const next = dogs.map(() => 0);
  const timeAt = (d: DogPlan, s: number) => d.start + s / WALK + LOOK * d.stops.filter((q) => q.s < s).length;
  // event order: always advance the dog whose next room entry comes first in time
  for (;;) {
    let best = -1, bt = Infinity;
    dogs.forEach((d, k) => {
      const m = d.route.marks[next[k]];
      if (m) { const t = timeAt(d, m.s); if (t < bt) { bt = t; best = k; } }
    });
    if (best < 0) break;
    const d = dogs[best], m = d.route.marks[next[best]++], i = index.get(m.room);
    if (i === undefined || owner[i] >= 0) continue;
    owner[i] = best; litAt[i] = bt; d.rooms++;
    d.stops.push({ s: Math.min(d.route.len, m.s + LOOK_IN), room: i });
  }
  for (const d of dogs) d.end = d.start + d.route.len / WALK + d.stops.length * LOOK;
  let total = 0;
  for (let i = 0; i < R; i++) if (owner[i] >= 0) total = Math.max(total, litAt[i] + LOOK);
  return { dogs, owner, litAt, total };
}

/** Where dog d is at sim time t: writes distance along its route and whether it is walking. */
export function placeAt(d: DogPlan, t: number, out: { s: number; v: number }) {
  let tc = d.start, s = 0;
  if (t <= tc) { out.s = 0; out.v = 0; return out; }
  for (const q of d.stops) {
    const walk = (q.s - s) / WALK;
    if (t < tc + walk) { out.s = s + (t - tc) * WALK; out.v = WALK; return out; }
    tc += walk; s = q.s;
    if (t < tc + LOOK) { out.s = s; out.v = 0; return out; }
    tc += LOOK;
  }
  const s2 = s + (t - tc) * WALK;
  out.s = Math.min(d.route.len, s2); out.v = s2 < d.route.len ? WALK : 0;
  return out;
}
