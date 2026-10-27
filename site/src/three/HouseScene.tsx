"use client";
// The House section's scene, played like a short film: two dogs walk in through the front door and search the house
// room by room, a drone-height camera follows the lead dog loosely, then lifts to a three-quarter view of the whole
// house. A short hold, a dip to dark, and it plays again. Off screen the canvas does not render at all.
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Suspense, useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import type { SkinName } from "@/lib/skins";
import { DogRig, type DogRigHandle } from "@/three/DogRig";
import { HouseModel, type LabelEls } from "@/three/HouseModel";
import { at, fitOverview, followView, LAG, planSweep, roomAt, type Layout } from "@/three/houseLayout";
import { useAdaptiveDpr, useLowPower } from "@/three/quality";
import { HouseLights, Ready, smooth, smoother, yawOf } from "@/three/sceneKit";

export type WalkEvent =
  | { type: "start" }
  | { type: "room"; room: string; next: string | null; fire: boolean; n: number; total: number }
  | { type: "done"; n: number; total: number }
  | { type: "fade" };

type Props = {
  layout: Layout; skins: [SkinName, SkinName]; active: boolean;
  onEvent: (e: WalkEvent) => void; onReady: () => void; labels: LabelEls;
};

const PRE = 1.4; // seconds the dogs stand outside before they move
const RISE = 3.4; // seconds of the final lift to the overview
const HOLD = 3.6; // seconds on the overview before the loop restarts
const DIP = 0.6; // seconds of the fade to dark before the restart
const FIRE_SLOW = 0.5; // the dogs walk at half speed in the fire room

function Scene({ layout, skins, onEvent, labels, reduce }: Omit<Props, "active" | "onReady"> & { reduce: boolean }) {
  const low = useLowPower();
  const routes = useMemo(() => planSweep(layout, 2), [layout]);
  const total = layout.rooms.length;
  const index = useMemo(() => Object.fromEntries(layout.rooms.map((r, i) => [r.id, i])), [layout]);
  // pace the walk so the whole search takes about 20 s, whatever the size of the house
  const speed = useMemo(() => Math.min(2.8, Math.max(1.5, Math.max(...routes.map((r, k) => r.len + k * LAG)) / 19)), [routes]);
  const lit = useRef(new Float32Array(total));
  const rigs = [useRef<DogRigHandle>(null), useRef<DogRigHandle>(null)];
  const lamp = useRef<THREE.PointLight>(null);
  const { camera, size } = useThree();
  const aspect = size.width / size.height, narrow = aspect < 0.9, fov = narrow ? 50 : 38;
  useEffect(() => { const c = camera as THREE.PerspectiveCamera; c.fov = fov; c.updateProjectionMatrix(); }, [camera, fov]);
  const over = useMemo(() => fitOverview(layout, aspect, fov, narrow ? 60 : 46, narrow ? 1.2 : 1.26), [layout, aspect, fov, narrow]);
  const follow = useMemo(() => followView(over, narrow), [over, narrow]);

  const fresh = () => ({ t: 0, s: [0, 0], v: [0, 0], mi: [0, 0], n: 0, end: -1, faded: false, snap: true, lead: 0 });
  const st = useRef(fresh());
  const tmp = useMemo(() => ({ buf: new Float32Array(6), p: new THREE.Vector3(), tgt: new THREE.Vector3(), want: new THREE.Vector3(), pos: new THREE.Vector3(), look: new THREE.Vector3() }), []);
  useEffect(() => { st.current = fresh(); lit.current = new Float32Array(total); onEvent({ type: "start" }); }, [layout, total, onEvent]);

  useFrame((_, rawDt) => {
    const dt = Math.min(rawDt, 0.05), S = st.current, L = lit.current;
    if (reduce) {
      // one still frame: every dog at the end of its route, every room lit, the overview
      for (let k = 0; k < 2; k++) { const r = routes[k]; at(r, r.len, tmp.buf); rigs[k].current?.drive(tmp.buf[0], tmp.buf[1], yawOf(tmp.buf[2], tmp.buf[3]), r.len, 0, dt); }
      if (S.end < 0) { L.fill(1); S.end = 0; onEvent({ type: "done", n: total, total }); }
      camera.position.copy(over.pos); camera.lookAt(over.target);
      return;
    }
    S.t += dt;
    let moving = 0;
    for (let k = 0; k < 2; k++) {
      const r = routes[k], startAt = PRE + (k * LAG) / speed;
      at(r, S.s[k], tmp.buf);
      const walking = S.t > startAt && S.s[k] < r.len;
      if (walking) {
        moving++;
        const room = roomAt(layout, tmp.p.set(tmp.buf[0], 0, tmp.buf[1]));
        const want = speed * (room?.fire ? FIRE_SLOW : 1) * smooth(0, 1.4, r.len - S.s[k] + 0.2); // ease off at the very end
        S.v[k] += (want - S.v[k]) * (1 - Math.exp(-dt * 3));
        S.s[k] = Math.min(r.len, S.s[k] + Math.max(0.05, S.v[k]) * dt);
        at(r, S.s[k], tmp.buf);
      } else S.v[k] = 0;
      rigs[k].current?.drive(tmp.buf[0], tmp.buf[1], yawOf(tmp.buf[2], tmp.buf[3]), S.s[k], walking ? S.v[k] : 0, dt);
      // first steps into rooms nobody has entered yet become the agent's captions
      const marks = r.marks;
      while (S.mi[k] < marks.length && marks[S.mi[k]].s <= S.s[k]) {
        const id = marks[S.mi[k]++].room, i = index[id];
        if (L[i]) continue;
        L[i] = 1; S.n++;
        let next: string | null = null;
        for (let j = S.mi[k]; j < marks.length; j++) if (!L[index[marks[j].room]]) { next = marks[j].room; break; }
        onEvent({ type: "room", room: id, next, fire: layout.byId[id].fire, n: S.n, total });
      }
    }
    if (S.end < 0 && !moving && S.t > PRE + LAG / speed + 0.5) { S.end = S.t; onEvent({ type: "done", n: S.n, total }); }

    // camera: follow the dog that is still walking (the lead first), loosely, then lift to the overview
    const o = 1 - S.lead;
    if (S.s[S.lead] >= routes[S.lead].len && S.s[o] < routes[o].len && S.t > PRE + LAG / speed) S.lead = o;
    at(routes[S.lead], S.s[S.lead], tmp.buf);
    tmp.want.set(tmp.buf[0] + tmp.buf[2] * 0.9, 0.3, tmp.buf[1] + tmp.buf[3] * 0.9);
    const door = 1 - smooth(1, 5, S.s[0]); // at the start, frame the dogs and the front door together
    tmp.want.x += (layout.entry.door.x - tmp.want.x) * door * 0.5; tmp.want.z += (layout.entry.door.z - tmp.want.z) * door * 0.5;
    if (S.snap) tmp.tgt.copy(tmp.want); else tmp.tgt.lerp(tmp.want, 1 - Math.exp(-dt * 1.5));
    lamp.current?.position.set(tmp.buf[0], 2.4, tmp.buf[1]);
    const push = 1 + 0.5 * (1 - smoother(0, PRE + 3, S.t)); // opens wide, then eases in on the dogs
    tmp.pos.copy(tmp.tgt).addScaledVector(follow.dir, follow.dist * push);
    tmp.look.copy(tmp.tgt);
    const e = S.end < 0 ? 0 : smoother(S.end, S.end + RISE, S.t);
    tmp.pos.lerp(over.pos, e); tmp.look.lerp(over.target, e);
    camera.position.copy(tmp.pos); camera.lookAt(tmp.look);
    S.snap = false;

    if (S.end >= 0) {
      const loopAt = S.end + RISE + HOLD;
      if (!S.faded && S.t > loopAt - DIP) { S.faded = true; onEvent({ type: "fade" }); }
      if (S.t > loopAt) { st.current = fresh(); L.fill(0); onEvent({ type: "start" }); }
    }
  }, -1); // before HouseModel's frame, so the room pills are projected with this frame's camera

  return (
    <>
      <HouseLights layout={layout} shadows={!low} />
      {/* warm fill so interiors and the dark dog paint read, plus a lamp that rides above the lead dog */}
      <ambientLight intensity={0.5} color="#ffd8b8" />
      <hemisphereLight args={["#ffe8d2", "#3a2430", 0.55]} />
      <pointLight ref={lamp} position={[0, 2.4, 0]} intensity={reduce ? 0 : 9} distance={7.5} decay={1.4} color="#ffd2a0" />
      <HouseModel layout={layout} lit={lit} reduce={reduce} labels={labels} />
      <DogRig ref={rigs[0]} skin={skins[0]} shadows={!low} />
      <DogRig ref={rigs[1]} skin={skins[1]} shadows={!low} />
    </>
  );
}

export default function HouseScene(props: Props) {
  const { active, onReady, ...rest } = props;
  // client only (loaded with ssr: false), so reading the media query here cannot cause a hydration mismatch
  const reduce = useMemo(() => typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches, []);
  const low = useLowPower();
  const [dpr, monitor] = useAdaptiveDpr(low);
  return (
    <Canvas
      shadows={low ? false : "percentage"}
      dpr={dpr}
      frameloop={active ? (reduce ? "demand" : "always") : "never"}
      camera={{ fov: 38, near: 0.1, far: 220, position: [0, 14, 18] }}
      gl={{ antialias: true, powerPreference: "high-performance" }}
      resize={{ offsetSize: true }}
      style={{ touchAction: "pan-y" }}
      aria-hidden="true"
    >
      {monitor}
      <color attach="background" args={["#120b1b"]} />
      <fog attach="fog" args={["#120b1b", 38, 95]} />
      <Suspense fallback={null}>
        <Scene {...rest} reduce={reduce} />
        <Ready onReady={onReady} />
      </Suspense>
    </Canvas>
  );
}
