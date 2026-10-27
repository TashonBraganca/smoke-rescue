"use client";
import { useSyncExternalStore } from "react";
import type { SkinName } from "./skins";
import type { HouseId } from "./plans";

// The visitor's mission choice, shared by the mission picker, the house walk-in and the film player.
// `count` is how many dogs go in (dog 1 Cinder, 2 Ember, 3 Ash). `dog` is the lead dog's paint, read by other sections.
export type TeamSize = 1 | 2 | 3;
type State = { dog: SkinName; house: HouseId; count: TeamSize };
let state: State = { dog: "Cinder", house: "family_ranch", count: 2 };
const subs = new Set<() => void>();
export const missionStore = {
  get: () => state,
  set: (patch: Partial<State>) => { state = { ...state, ...patch }; subs.forEach((f) => f()); },
  subscribe: (f: () => void) => { subs.add(f); return () => { subs.delete(f); }; },
};
export function useMission() {
  return useSyncExternalStore(missionStore.subscribe, missionStore.get, missionStore.get);
}
