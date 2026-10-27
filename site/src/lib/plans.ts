// Floor plans exported by the Unity simulator (config/floorplan_random_*.json). Units are metres,
// origin at the south-west corner; rect = [x0, z0, x1, z1].
export type Room = { id: string; name: string; kind: string; rect: [number, number, number, number] };
export type Door = { id: string; between: [string, string]; wall: "x" | "z"; at: [number, number]; width: number };
export type Landmark = { room: string; kind: string; at: [number, number]; yaw?: number };
export type Plan = { footprint: [number, number, number, number]; rooms: Room[]; doors: Door[]; landmarks: Landmark[]; fires: { room: string; at: [number, number] }[] };

export const HOUSES = [
  { id: "family_ranch", name: "Three-bed ranch" },
  { id: "long_ranch", name: "L-shaped ranch" },
  { id: "934", name: "One-door house" },
  { id: "5008", name: "Bungalow" },
  { id: "5009", name: "Long house" },
] as const;
export type HouseId = (typeof HOUSES)[number]["id"];

const cache: Partial<Record<HouseId, Promise<Plan>>> = {};
export function loadPlan(id: HouseId): Promise<Plan> {
  // a failed load is not cached, so the next caller tries again instead of getting the same rejection forever
  return (cache[id] ??= fetch(`/data/plan_${id}.json`)
    .then((r) => { if (!r.ok) throw new Error(`plan ${id}: HTTP ${r.status}`); return r.json() as Promise<Plan>; })
    .catch((e: unknown) => { delete cache[id]; throw e; }));
}

export function planInfo(id: HouseId, p: Plan) {
  const byId = Object.fromEntries(p.rooms.map((r) => [r.id, r]));
  const fire = p.fires[0] && byId[p.fires[0].room] ? byId[p.fires[0].room].name.toLowerCase() : "";
  return { name: HOUSES.find((h) => h.id === id)!.name, rooms: p.rooms.length, w: p.footprint[2] - p.footprint[0], d: p.footprint[3] - p.footprint[1], fire };
}

export const KIND_COLOR: Record<string, string> = {
  living: "#4A2346", kitchen: "#4E3326", bedroom: "#3A2350", bathroom: "#3A3340",
  study: "#50303A", laundry: "#343038", closet: "#2B2433", corridor: "#241B2C",
};
