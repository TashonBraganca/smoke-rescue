"use client";
// The mission, played live with the visitor's team of 1 to 3 dogs. The rooms are split between the dogs (teamRun.ts);
// each dog trots in through the front door and searches its share in parallel. A room takes the colour of the dog
// that searched it. The camera follows the team, then lifts over the whole house when every room is checked.
// Everything runs on simulated time from the plan, played back `rate` times faster, so the timer is honest.
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Suspense, useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { SKINS } from "@/lib/skins";
import { teamOf } from "@/lib/team";
import { DogRig, type DogRigHandle } from "@/three/DogRig";
import { HouseModel, type LabelEls } from "@/three/HouseModel";
import { at, fitOverview, followView, type Layout } from "@/three/houseLayout";
import { useAdaptiveDpr, useLowPower } from "@/three/quality";
import { HouseLights, Ready, smooth, smoother, yawOf } from "@/three/sceneKit";
import { placeAt, type TeamPlan } from "@/three/teamRun";

export type RunStatus = { n: number; total: number; done: boolean; t: number; perDog: number[] };
type Props = { layout: Layout; team: TeamPlan; rate: number; reduce: boolean; runKey: number; onStatus: (s: RunStatus) => void; labels: LabelEls; onReady: () => void };

const START = 0.8, LIFT = 3.2; // real seconds of standing at the door, real seconds of the final crane-up

function Run({ layout, team, rate, reduce, runKey, onStatus, labels }: Omit<Props, "onReady">) {
  const low = useLowPower();
  const n = team.dogs.length, R = layout.rooms.length;
  const skins = useMemo(() => teamOf(n), [n]);
  const lit = useRef(new Float32Array(R));
  const rigs = useRef<(DogRigHandle | null)[]>([]);
  const lamps = useRef<(THREE.PointLight | null)[]>([]);
  const { camera, size } = useThree();
  const narrow = size.width / size.height < 0.9, fov = narrow ? 52 : 40;
  useEffect(() => { const c = camera as THREE.PerspectiveCamera; c.fov = fov; c.updateProjectionMatrix(); }, [camera, fov]);
  const over = useMemo(() => {
    const o = fitOverview(layout, size.width / size.height, fov, narrow ? 64 : 50, narrow ? 1.2 : 1.1);
    // slide the view towards the camera so the house sits a little higher, clear of the end card's text
    const toCam = o.pos.clone().sub(o.target).setY(0).normalize().multiplyScalar(narrow ? 2.2 : 0.9);
    o.pos.add(toCam); o.target.add(toCam);
    return o;
  }, [layout, size.width, size.height, fov, narrow]);
  const follow = useMemo(() => followView(over, narrow), [over, narrow]);

  // a tint in the searching dog's paint, laid over each room's floor
  const tints = useMemo(() => layout.rooms.map((r, i) => {
    const o = team.owner[i];
    const mat = new THREE.MeshBasicMaterial({ color: o >= 0 ? SKINS[skins[o]].hex : "#ffffff", transparent: true, opacity: 0, depthWrite: false, toneMapped: false });
    return { r, mat, w: r.x1 - r.x0 - 0.1, d: r.z1 - r.z0 - 0.1, max: o >= 0 && skins[o] === "Ash" ? 0.4 : 0.3 };
  }), [layout, team, skins]);
  useEffect(() => () => tints.forEach((t) => t.mat.dispose()), [tints]);

  const fresh = () => ({ t: 0, n: -1, sec: -1, done: false, end: -1, snap: true, per: "" });
  const st = useRef(fresh());
  const tmp = useMemo(() => ({ buf: new Float32Array(6), place: { s: 0, v: 0 }, cp: new THREE.Vector3(), ct: new THREE.Vector3(), pos: new THREE.Vector3(), look: new THREE.Vector3(), c: new THREE.Vector3() }), []);
  useEffect(() => {
    st.current = fresh(); lit.current = new Float32Array(R);
    tints.forEach((t) => { t.mat.opacity = 0; });
    labels.current?.forEach((el) => { if (el) { el.style.borderColor = ""; el.style.boxShadow = ""; } });
  }, [runKey, R, tints, labels]);

  useFrame((_, rawDt) => {
    const dt = Math.min(rawDt, 0.05), S = st.current, L = lit.current;
    S.t += dt;
    // simulated seconds since the dogs set off
    const sim = reduce ? team.total + 60 : Math.max(0, (S.t - START) * rate);
    // dogs: place each one on its route at this moment of the plan
    tmp.c.set(0, 0, 0);
    let walking = 0, spread = 0;
    const per: number[] = [];
    for (let k = 0; k < n; k++) {
      const d = team.dogs[k], p = placeAt(d, sim, tmp.place);
      at(d.route, p.s, tmp.buf);
      rigs.current[k]?.drive(tmp.buf[0], tmp.buf[1], yawOf(tmp.buf[2], tmp.buf[3]), p.s, p.v * rate, dt);
      lamps.current[k]?.position.set(tmp.buf[0], 2.4, tmp.buf[1]);
      if (sim < d.end) { tmp.c.x += tmp.buf[0]; tmp.c.z += tmp.buf[1]; walking++; }
      per.push(0);
    }
    // camera target: the middle of the dogs still searching (the whole team once they are all done)
    if (!walking) for (let k = 0; k < n; k++) { at(team.dogs[k].route, team.dogs[k].route.len, tmp.buf); tmp.c.x += tmp.buf[0]; tmp.c.z += tmp.buf[1]; walking++; }
    tmp.c.multiplyScalar(1 / walking);
    for (let k = 0; k < n; k++) {
      const d = team.dogs[k]; if (sim >= d.end && walking < n) continue;
      at(d.route, placeAt(d, sim, tmp.place).s, tmp.buf);
      spread = Math.max(spread, Math.hypot(tmp.buf[0] - tmp.c.x, tmp.buf[1] - tmp.c.z));
    }
    // rooms: lit from the moment their dog steps in
    let count = 0;
    for (let i = 0; i < R; i++) {
      const on = sim >= team.litAt[i] ? 1 : 0;
      if (on && !L[i]) {
        const el = labels.current?.[i], hex = SKINS[skins[team.owner[i]]].hex;
        if (el && !layout.rooms[i].fire) { el.style.borderColor = hex; el.style.boxShadow = `0 0 0 1px ${hex}`; }
      }
      L[i] = on; count += on;
      if (on) per[team.owner[i]]++;
      const t = tints[i]; t.mat.opacity += (on * t.max - t.mat.opacity) * (reduce ? 1 : 1 - Math.exp(-dt * 5));
    }
    const done = sim >= team.total;
    if (done && S.end < 0) S.end = S.t;
    const sec = done ? Math.round(team.total) : Math.floor(sim), perKey = per.join(",");
    if (count !== S.n || done !== S.done || sec !== S.sec || perKey !== S.per) {
      S.n = count; S.done = done; S.sec = sec; S.per = perKey;
      onStatus({ n: count, total: R, done, t: sec, perDog: per });
    }

    // camera: high follow view from the overview's side, pulled back as the team spreads out, lifting at the end
    tmp.ct.set(tmp.c.x, 0.3, tmp.c.z);
    if (n === 1) { at(team.dogs[0].route, placeAt(team.dogs[0], sim, tmp.place).s, tmp.buf); tmp.ct.x += tmp.buf[2] * 0.9; tmp.ct.z += tmp.buf[3] * 0.9; }
    const door = (1 - smooth(1, 6, sim * 1.4)) * 0.5;
    tmp.ct.x += (layout.entry.door.x - tmp.ct.x) * door; tmp.ct.z += (layout.entry.door.z - tmp.ct.z) * door;
    tmp.cp.copy(tmp.ct).addScaledVector(follow.dir, follow.dist * (1 + 0.5 * (1 - smoother(0, START + 3, S.t))));
    const wide = smooth(1.5, 7, spread) * 0.75;
    const e = reduce ? 1 : Math.max(wide, S.end < 0 ? 0 : smoother(S.end, S.end + LIFT, S.t));
    tmp.cp.lerp(over.pos, e); tmp.ct.lerp(over.target, e);
    // ease the camera towards where it wants to be, so turns in the routes never jerk the view
    const k = S.snap ? 1 : 1 - Math.exp(-dt * (2.2 + e * 4));
    S.snap = false;
    tmp.pos.lerp(tmp.cp, k); tmp.look.lerp(tmp.ct, k);
    if (k === 1) { tmp.pos.copy(tmp.cp); tmp.look.copy(tmp.ct); }
    camera.position.copy(tmp.pos); camera.lookAt(tmp.look);
  }, -1);

  return (
    <>
      <HouseLights layout={layout} shadows={!low} />
      {/* warm fill so interiors and the dogs read, plus a lamp that rides above each dog (one on the light path) */}
      <ambientLight intensity={0.5} color="#ffd8b8" />
      <hemisphereLight args={["#ffe8d2", "#3a2430", 0.55]} />
      {skins.slice(0, low ? 1 : n).map((s, k) => (
        <pointLight key={s} ref={(l) => { lamps.current[k] = l; }} position={[0, 2.4, 0]} intensity={reduce ? 0 : n > 1 ? 7 : 9} distance={7.5} decay={1.4} color="#ffd2a0" />
      ))}
      <HouseModel layout={layout} lit={lit} reduce={reduce} labels={labels} />
      {tints.map((t) => (
        <mesh key={t.r.id} rotation-x={-Math.PI / 2} position={[t.r.centre.x, 0.014, t.r.centre.z]} material={t.mat} renderOrder={1}>
          <planeGeometry args={[t.w, t.d]} />
        </mesh>
      ))}
      {skins.map((s, k) => <DogRig key={s} ref={(r) => { rigs.current[k] = r; }} skin={s} shadows={!low} />)}
    </>
  );
}

export default function LiveRun(props: Props) {
  const low = useLowPower();
  const [dpr, monitor] = useAdaptiveDpr(low);
  const { onReady, ...rest } = props;
  return (
    // offsetSize: measure the layout box, not the transformed one, because the overlay opens with a scale animation
    <Canvas shadows={low ? false : "percentage"} dpr={dpr} resize={{ offsetSize: true }} camera={{ fov: 40, near: 0.1, far: 220, position: [0, 6, 14] }} gl={{ antialias: true, powerPreference: "high-performance" }} aria-hidden="true">
      {monitor}
      <color attach="background" args={["#0e0817"]} />
      <fog attach="fog" args={["#0e0817", 34, 90]} />
      <Suspense fallback={null}>
        <Run {...rest} />
        <Ready onReady={onReady} />
      </Suspense>
    </Canvas>
  );
}
