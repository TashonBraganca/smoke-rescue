"use client";
// Shared bits for the two house scenes (the scroll walk-in and the mission run): lights, a follow camera, and
// helpers to turn a dog's distance along its route into a pose and a set of searched rooms.
import { useEffect } from "react";
import * as THREE from "three";
import { at, type Layout, type Route } from "@/three/houseLayout";

export const smooth = (a: number, b: number, x: number) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
export const smoother = (a: number, b: number, x: number) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * t * (t * (t * 6 - 15) + 10); };

/** Follow camera behind a dog at distance s: low at the door, rising to drone height once inside. */
export function chase(route: Route, s: number, buf: Float32Array, pos: THREE.Vector3, target: THREE.Vector3, narrow: boolean) {
  at(route, s, buf);
  const inside = smooth(1.5, 7, s);
  const back = (narrow ? 6.2 : 5.6) - inside * 1.2, h = 2.6 + inside * 0.1;
  pos.set(buf[0] - buf[4] * back, h, buf[1] - buf[5] * back);
  target.set(buf[0] + buf[4] * 1.4, 1.05 - inside * 0.75, buf[1] + buf[5] * 1.4);
  return buf;
}

/** Rooms entered by any dog so far, written into lit (1 or 0 per room, in layout order). Returns the count. */
export function searched(layout: Layout, routes: Route[], s: number[], lit: Float32Array) {
  let n = 0;
  for (let i = 0; i < layout.rooms.length; i++) {
    const id = layout.rooms[i].id;
    let on = 0;
    for (let k = 0; k < routes.length; k++) { const m = routes[k].marks.find((x) => x.room === id); if (m && s[k] >= m.s) { on = 1; break; } }
    lit[i] = on; n += on;
  }
  return n;
}

export const yawOf = (dx: number, dz: number) => Math.atan2(-dz, dx);

export function HouseLights({ layout, shadows = true }: { layout: Layout; shadows?: boolean }) {
  const span = Math.max(layout.w, layout.d) / 2 + 2;
  // key light from the front-door side and a little to one side, so the dogs at the door are lit, not silhouettes
  const o = layout.entry.out, sd = layout.entry.side;
  const key: [number, number, number] = [(o.x * 0.8 + sd.x * 0.6) * span, span * 1.5, (o.z * 0.8 + sd.z * 0.6) * span];
  return (
    <>
      <hemisphereLight args={["#f3e6dc", "#1a1024", 0.85]} />
      <directionalLight
        position={key} intensity={1.7} color="#ffe9d6" castShadow={shadows}
        shadow-mapSize={[1024, 1024]} shadow-bias={-0.0005} shadow-normalBias={0.03}
        shadow-camera-left={-span} shadow-camera-right={span} shadow-camera-top={span} shadow-camera-bottom={-span}
        shadow-camera-near={1} shadow-camera-far={span * 5}
      />
      <directionalLight position={[-o.x * span, span * 0.6, -o.z * span]} intensity={0.5} color="#fcb040" />
    </>
  );
}

/** Calls back once the Suspense boundary around it has resolved (models loaded). */
export function Ready({ onReady }: { onReady: () => void }) {
  useEffect(() => { onReady(); }, [onReady]);
  return null;
}
