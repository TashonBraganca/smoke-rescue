"use client";
import { useFrame, useThree } from "@react-three/fiber";
import { ContactShadows, Environment, Lightformer } from "@react-three/drei";
import { EffectComposer, Bloom, Vignette, SMAA } from "@react-three/postprocessing";
import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import type { MotionValue } from "motion/react";
import { Dog, Firefighter, dogAnchors, type DogClip } from "./models";
import { BlobShadow } from "./quality";
import type { SkinName } from "@/lib/skins";

const SENSOR_OFFSETS: [number, number, number][] = [[0, 0.025, 0.07], [0, 0.025, -0.07]];
// The second dog walks in down the hall, through the doorway at z = 1.9, and stops beside the first one.
const D2_FROM = new THREE.Vector3(0.85, 0, 6.2), D2_DOOR = new THREE.Vector3(0.85, 0, 1.55), D2_TO = new THREE.Vector3(-1.05, 0, 1.3);

const sm = (a: number, b: number, x: number) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
const V = (x: number, y: number, z: number) => new THREE.Vector3(x, y, z);
const DOG_YAW = -0.5; // dog front (+X) turned a little toward the camera

// Camera path: holds at each beat, glides between them (smoothstep per segment, then damped).
type Key = { p: number; pos: THREE.Vector3; tgt: THREE.Vector3 | "face"; fov: number };
function keys(face: THREE.Vector3): Key[] {
  const out = face.clone().sub(V(0, face.y, 0)).normalize();          // horizontal direction out of the face
  const lens = face.clone().add(out.clone().multiplyScalar(0.09)).add(V(0, 0.005, 0));
  return [
    { p: 0.0, pos: V(0.55, 1.15, 6.9), tgt: V(0.0, 0.85, 0), fov: 26 },
    { p: 0.06, pos: V(0.55, 1.15, 6.9), tgt: V(0.0, 0.85, 0), fov: 26 },
    { p: 0.17, pos: V(1.6, 1.0, 3.2), tgt: V(0.2, 0.36, 0), fov: 28 },
    { p: 0.24, pos: V(-2.9, 3.0, 3.6), tgt: V(0.1, 0.3, 0), fov: 34 },
    { p: 0.36, pos: V(-4.0, 4.4, -1.4), tgt: V(0.1, 0.2, 0), fov: 36 },
    { p: 0.44, pos: face.clone().add(out.clone().multiplyScalar(1.6)).add(V(0, 0.1, 0)), tgt: "face", fov: 30 },
    { p: 0.55, pos: face.clone().add(out.clone().multiplyScalar(1.15)).add(V(0.15, 0.02, 0.2)), tgt: "face", fov: 28 },
    { p: 0.66, pos: lens, tgt: "face", fov: 22 },
    { p: 0.86, pos: lens, tgt: "face", fov: 22 },
    { p: 0.9, pos: V(3.4, 4.6, 6.6), tgt: V(0.55, 0.1, 1.6), fov: 34 },
    { p: 1.0, pos: V(2.6, 5.2, 7.2), tgt: V(0.55, 0.1, 1.7), fov: 34 },
  ];
}

// A rectangular room with a doorway, sampled as lidar returns on five scan lines.
function lidarCloud() {
  const W = 2.4, D = 1.9, door: [number, number] = [0.4, 1.3], N = 720, H = [0.12, 0.35, 0.6, 0.85, 1.1];
  const pts: number[] = [];
  for (let i = 0; i < N; i++) {
    const a = (i / N) * Math.PI * 2, dx = Math.cos(a), dz = Math.sin(a);
    let t = Math.min(dx ? (dx > 0 ? W : -W) / dx : 1e9, dz ? (dz > 0 ? D : -D) / dz : 1e9);
    const hx = dx * t, hz = dz * t;
    if (hz > D - 0.01 && hx > door[0] && hx < door[1]) t *= 1.7;
    for (let k = 0; k < H.length; k++) { if ((i + k) % 2) continue; pts.push(dx * t, H[k] + (Math.random() - 0.5) * 0.02, dz * t); }
  }
  return new Float32Array(pts);
}
// The hall the second dog walks through: two walls from the doorway out to z = 5.4, plus its far end.
function hallCloud() {
  const H = [0.12, 0.35, 0.6, 0.85, 1.1], pts: number[] = [];
  for (let z = 1.95; z <= 5.4; z += 0.035) for (const x of [0.4, 1.3]) for (const h of H) pts.push(x, h + (Math.random() - 0.5) * 0.02, z);
  return new Float32Array(pts);
}

export default function StoryScene({ progress, skin, low }: { progress: MotionValue<number>; skin: SkinName; low: boolean }) {
  const { camera, size } = useThree();
  const cam = camera as THREE.PerspectiveCamera;
  const dogGroup = useRef<THREE.Group>(null);
  const ffGroup = useRef<THREE.Group>(null);
  const [anch, setAnch] = useState<ReturnType<typeof dogAnchors> | null>(null);
  const onModel = useCallback((m: THREE.Object3D) => setAnch(dogAnchors(m)), []);
  const [clip, setClip] = useState<DogClip>("idle");

  // world-space face point (the dog sits at the origin, turned by DOG_YAW)
  const face = useMemo(() => { if (!anch) return V(0.5, 0.42, 0); return anch.face.clone().applyAxisAngle(V(0, 1, 0), DOG_YAW); }, [anch]);
  const back = useMemo(() => (anch ? anch.back.clone().applyAxisAngle(V(0, 1, 0), DOG_YAW) : V(0, 0.55, 0)), [anch]);
  const K = useMemo(() => keys(face), [face]);

  // an occasional small hop while the hero is on screen
  useEffect(() => {
    const id = setInterval(() => { if (progress.get() < 0.05) { setClip("hop"); setTimeout(() => setClip("idle"), 1500); } }, 5200);
    return () => clearInterval(id);
  }, [progress]);

  const cloud = useMemo(() => {
    const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.BufferAttribute(lidarCloud(), 3)); g.setDrawRange(0, 0); return g;
  }, []);
  const hall = useMemo(() => {
    const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.BufferAttribute(hallCloud(), 3)); g.setDrawRange(0, 0); return g;
  }, []);
  const hallMat = useMemo(() => new THREE.PointsMaterial({ size: 0.03, color: new THREE.Color("#fcffa4").multiplyScalar(2), transparent: true, depthWrite: false, toneMapped: false }), []);
  const amber = useMemo(() => new THREE.Color("#f98c0a").multiplyScalar(2.2), []), whiteHot = useMemo(() => new THREE.Color("#fcffa4").multiplyScalar(2), []);
  const dog2 = useRef<THREE.Group>(null);
  const [clip2, setClip2] = useState<DogClip>("trot");
  const ringsB = useRef<THREE.Group>(null);
  const link = useMemo(() => { const g = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]); return new THREE.Line(g, new THREE.LineBasicMaterial({ color: new THREE.Color("#fcffa4").multiplyScalar(2), transparent: true, toneMapped: false })); }, []);
  const pulse = useRef<THREE.Mesh>(null);
  // closing beat: the safe route the crew takes, from the hall to the person the dogs found
  const route = useMemo(() => {
    const curve = new THREE.CatmullRomCurve3([V(0.85, 0.05, 5.4), V(0.85, 0.05, 2.4), V(0.9, 0.05, 1.5), V(1.35, 0.05, 0.3), V(1.75, 0.05, -1.0)], false, "catmullrom", 0.2);
    return new THREE.TubeGeometry(curve, 120, 0.028, 8, false);
  }, []);
  const routeMat = useMemo(() => new THREE.MeshBasicMaterial({ color: new THREE.Color("#fcffa4").multiplyScalar(2.5), toneMapped: false, transparent: true }), []);
  const pin = useRef<THREE.Group>(null);
  useEffect(() => () => { route.dispose(); routeMat.dispose(); }, [route, routeMat]);
  const cloudMat = useMemo(() => new THREE.PointsMaterial({ size: 0.03, color: new THREE.Color("#f98c0a").multiplyScalar(2.2), transparent: true, depthWrite: false, toneMapped: false }), []);
  useEffect(() => () => { cloud.dispose(); hall.dispose(); cloudMat.dispose(); hallMat.dispose(); link.geometry.dispose(); (link.material as THREE.Material).dispose(); }, [cloud, hall, cloudMat, hallMat, link]);
  const sweep = useRef<THREE.Mesh>(null);
  const glowA = useRef<THREE.Mesh>(null), glowB = useRef<THREE.Mesh>(null), faceLight = useRef<THREE.PointLight>(null);
  const rings = useRef<THREE.Group>(null);
  const fire = useRef<THREE.PointLight>(null), rim = useRef<THREE.DirectionalLight>(null);
  const ffMats = useRef<THREE.Material[]>([]);
  const cur = useRef({ pos: K[0].pos.clone(), tgt: (K[0].tgt as THREE.Vector3).clone(), fov: 26, off: 0 });
  const tmpP = useMemo(() => new THREE.Vector3(), []), tmpT = useMemo(() => new THREE.Vector3(), []);

  useFrame((state, dt) => {
    const p = progress.get(), t = state.clock.elapsedTime;
    // camera target from the keyframes
    let i = 0; while (i < K.length - 2 && p > K[i + 1].p) i++;
    const a = K[i], b = K[i + 1], s = sm(a.p, b.p, p);
    tmpP.copy(a.pos).lerp(b.pos, s);
    const ta = a.tgt === "face" ? face : a.tgt, tb = b.tgt === "face" ? face : b.tgt;
    tmpT.copy(ta).lerp(tb, s);
    const fov = a.fov + (b.fov - a.fov) * s;
    const k = 1 - Math.exp(-dt * 4.5);                                     // critically damped follow, no jitter
    cur.current.pos.lerp(tmpP, k); cur.current.tgt.lerp(tmpT, k); cur.current.fov += (fov - cur.current.fov) * k;
    // a slow breathing drift so held frames are never dead still
    const drift = 0.035 * (1 - sm(0.6, 0.66, p) + sm(0.86, 0.9, p));
    cam.position.copy(cur.current.pos).add(tmpP.set(Math.sin(t * 0.3) * drift, Math.sin(t * 0.23) * drift * 0.5, 0));
    // phones: a wider lens at the start so the pair fits below the headline
    const narrow = size.width <= 760 ? 1 + 0.45 * (1 - sm(0.05, 0.16, p)) + 0.5 * sm(0.86, 0.92, p) : 1;   // phones: also wider at the end so both dogs fit
    cam.fov = cur.current.fov * narrow; cam.lookAt(cur.current.tgt);
    // hero framing: subjects sit right of the headline on wide screens
    const wide = size.width > 760, offT = (wide ? -0.17 : 0) * (1 - sm(0.05, 0.16, p));
    cur.current.off += (offT - cur.current.off) * k;
    if (Math.abs(cur.current.off) > 0.001 || !wide) cam.setViewOffset(size.width, size.height, size.width * cur.current.off, wide ? 0 : -size.height * 0.23 * (1 - sm(0.05, 0.16, p)), size.width, size.height); else cam.clearViewOffset();
    cam.updateProjectionMatrix();

    // firefighter steps out of the shot
    const out = sm(0.05, 0.13, p);
    if (ffGroup.current) { ffGroup.current.position.x = -0.35 - out * 1.2; ffGroup.current.visible = out < 0.99;
      if (!ffMats.current.length) ffGroup.current.traverse((o) => { const m = o as THREE.Mesh; if (m.isMesh) { const mat = m.material as THREE.MeshStandardMaterial; mat.transparent = true; ffMats.current.push(mat); } });
      ffMats.current.forEach((m) => { m.opacity = 1 - out; }); }

    // lidar: returns build up scan by scan, a sweep turns
    const lid = sm(0.2, 0.36, p), lidOut = sm(0.42, 0.5, p);
    const total = cloud.attributes.position.count;
    cloud.setDrawRange(0, Math.floor(total * lid));
    cloudMat.opacity = 1 - lidOut * 0.85;
    if (sweep.current) { sweep.current.visible = lid > 0 && lid < 1 && lidOut < 0.5; sweep.current.rotation.z = t * 2.6; }

    // heat cameras glow as the camera arrives at the face
    const heat = sm(0.4, 0.5, p) * (1 - sm(0.86, 0.92, p));
    for (const g of [glowA.current, glowB.current]) if (g) { g.visible = heat > 0.01; (g.material as THREE.MeshBasicMaterial).opacity = heat; }
    if (faceLight.current) faceLight.current.intensity = heat * 2.5;

    // a second dog walks in down the hall, mapping it as it comes; then the two swap maps over the radio
    const walk2 = sm(0.885, 0.94, p), merged = sm(0.955, 0.975, p);
    if (dog2.current) {
      dog2.current.visible = p > 0.875;
      // down the hall to the doorway, then left into the room, then turn to face the first dog
      const leg1 = Math.min(1, walk2 / 0.72), leg2 = Math.max(0, (walk2 - 0.72) / 0.28);
      if (leg2 <= 0) dog2.current.position.lerpVectors(D2_FROM, D2_DOOR, leg1); else dog2.current.position.lerpVectors(D2_DOOR, D2_TO, leg2);
      const headHall = Math.PI / 2, headIn = Math.atan2(-(D2_TO.z - D2_DOOR.z), D2_TO.x - D2_DOOR.x), headDog = Math.atan2(D2_TO.z, -D2_TO.x);
      const turnIn = sm(0.66, 0.78, walk2), turnDog = sm(0.94, 0.955, p);
      dog2.current.rotation.y = headHall + (headIn - headHall) * turnIn + (headDog - headIn) * turnDog;
      const want: DogClip = walk2 > 0 && walk2 < 1 ? "trot" : "idle";
      if (want !== clip2) setClip2(want);
    }
    hall.setDrawRange(0, Math.floor(hall.attributes.position.count * walk2));
    hallMat.color.copy(whiteHot).lerp(amber, merged);
    cloudMat.opacity = 1 - lidOut * 0.85 + sm(0.88, 0.93, p) * 0.75;
    const radio = sm(0.945, 0.96, p);
    for (const [grp, x, z] of [[rings.current, 0, 0], [ringsB.current, D2_TO.x, D2_TO.z]] as const) if (grp) {
      grp.visible = radio > 0.01; grp.position.set(x, 0.04, z);
      grp.children.forEach((m, j) => { const ph = (t * 0.6 + j / 3) % 1; m.scale.setScalar(0.3 + ph * 1.9); ((m as THREE.Mesh).material as THREE.MeshBasicMaterial).opacity = radio * 0.55 * (1 - ph); });
    }
    link.visible = radio > 0.01; (link.material as THREE.LineBasicMaterial).opacity = radio * 0.6;
    const lp = link.geometry.attributes.position as THREE.BufferAttribute; lp.setXYZ(0, 0, 0.55, 0); lp.setXYZ(1, D2_TO.x, 0.55, D2_TO.z); lp.needsUpdate = true;
    const crew = sm(0.972, 0.995, p), idx = route.index!.count;
    route.setDrawRange(0, Math.floor((idx * crew) / 6) * 6); routeMat.opacity = crew > 0 ? 1 : 0;
    if (pin.current) { pin.current.visible = crew > 0.6; const ph = (t * 0.8) % 1; const ring = pin.current.children[1]; ring.scale.setScalar(1 + ph * 3); ((ring as THREE.Mesh).material as THREE.MeshBasicMaterial).opacity = 0.9 * (1 - ph); }
    if (pulse.current) { pulse.current.visible = radio > 0.01; const f = (Math.sin(t * 2.2) + 1) / 2; pulse.current.position.set(D2_TO.x * f, 0.55, D2_TO.z * f); }

    if (fire.current) fire.current.intensity = (7 + Math.sin(t * 11) * 0.7 + Math.sin(t * 23) * 0.5) * (1 - sm(0.12, 0.25, p) * 0.7);
    if (rim.current) rim.current.intensity = 1.4 * (1 - sm(0.12, 0.25, p) * 0.4);
  });

  return (
    <>
      {/* soft studio panels, generated locally (no HDR download) */}
      <Environment resolution={256} environmentIntensity={0.35}>
        <Lightformer form="rect" intensity={2} color="#ffe2c4" position={[3, 3, 4]} scale={[4, 2, 1]} />
        <Lightformer form="rect" intensity={3} color="#ff7a1a" position={[-3, 2, -4]} scale={[5, 2, 1]} />
        <Lightformer form="ring" intensity={0.6} color="#57106e" position={[0, 5, 0]} scale={4} />
      </Environment>
      <hemisphereLight args={["#2e2622", "#0b0612", 0.55]} />
      <directionalLight position={[3, 5, 5]} intensity={1.0} castShadow={!low} shadow-mapSize={[1024, 1024]} shadow-bias={-0.0004} />
      <directionalLight ref={rim} position={[-1.5, 2.2, -4]} intensity={1.4} color="#ff7a1a" />
      <pointLight ref={fire} position={[0.4, 1.4, -2.6]} intensity={7} distance={16} decay={1.6} color="#ff6a10" />

      <group ref={ffGroup} position={[-0.35, 0, -0.2]} rotation={[0, 0.4, 0]}>
        <Suspense fallback={null}><Firefighter clip="idle" /></Suspense>
        {low && <BlobShadow w={0.75} d={0.55} />}
      </group>
      <group ref={dogGroup} rotation={[0, DOG_YAW, 0]} position={[0, 0, 0]}>
        <Dog skin={skin} clip={clip} onModel={onModel} />
        {low && <BlobShadow w={1.25} d={0.7} />}
        {/* heat-camera glow on the face panel */}
        {anch && SENSOR_OFFSETS.map((o, j) => (
          <mesh key={j} ref={j ? glowB : glowA} position={[anch.face.x + 0.01 + o[0], anch.face.y + o[1], anch.face.z + o[2]]} visible={false}>
            <sphereGeometry args={[0.022, 16, 12]} />
            <meshBasicMaterial color={new THREE.Color("#fcffa4").multiplyScalar(3)} transparent toneMapped={false} />
          </mesh>
        ))}
      </group>
      <pointLight ref={faceLight} position={face.toArray()} intensity={0} distance={1.2} decay={2} color="#fcb040" />

      <points geometry={cloud} material={cloudMat} />
      <mesh ref={sweep} position={[back.x, 0.45, back.z]} rotation={[-Math.PI / 2, 0, 0]} visible={false}>
        <circleGeometry args={[2.4, 48, 0, 0.32]} />
        <meshBasicMaterial color="#f98c0a" transparent opacity={0.05} side={THREE.DoubleSide} depthWrite={false} />
      </mesh>
      {[rings, ringsB].map((r, i) => (
        <group key={i} ref={r} visible={false}>
          {[0, 1, 2].map((j) => (
            <mesh key={j} rotation={[-Math.PI / 2, 0, 0]}>
              <ringGeometry args={[0.97, 1, 96]} />
              <meshBasicMaterial color={new THREE.Color("#fcffa4").multiplyScalar(2)} transparent depthWrite={false} toneMapped={false} side={THREE.DoubleSide} />
            </mesh>
          ))}
        </group>
      ))}
      <group ref={dog2} visible={false} position={D2_FROM.toArray()}>
        <Dog skin={skin === "Ash" ? "Ember" : "Ash"} clip={clip2} />
        {low && <BlobShadow w={1.25} d={0.7} />}
      </group>
      <points geometry={hall} material={hallMat} />
      <primitive object={link} />
      <mesh geometry={route} material={routeMat} />
      <group ref={pin} position={[1.75, 0.06, -1.0]} visible={false}>
        <mesh><sphereGeometry args={[0.07, 20, 14]} /><meshBasicMaterial color={new THREE.Color("#f98c0a").multiplyScalar(3)} toneMapped={false} /></mesh>
        <mesh rotation={[-Math.PI / 2, 0, 0]}><ringGeometry args={[0.1, 0.12, 48]} /><meshBasicMaterial color={new THREE.Color("#fcffa4").multiplyScalar(2)} transparent toneMapped={false} side={THREE.DoubleSide} depthWrite={false} /></mesh>
      </group>
      <mesh ref={pulse} visible={false}>
        <sphereGeometry args={[0.035, 16, 12]} />
        <meshBasicMaterial color={new THREE.Color("#fcffa4").multiplyScalar(4)} toneMapped={false} />
      </mesh>

      {!low && <ContactShadows position={[0, 0.001, 0]} opacity={0.55} scale={8} blur={2.4} far={2} resolution={512} />}
      {!low && <EffectComposer multisampling={0}>
        <SMAA />
        <Bloom mipmapBlur intensity={0.7} luminanceThreshold={0.85} luminanceSmoothing={0.2} />
        <Vignette eskil={false} offset={0.25} darkness={0.7} />
      </EffectComposer>}
    </>
  );
}
