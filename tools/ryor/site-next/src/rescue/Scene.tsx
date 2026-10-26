'use client';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { Billboard, Html } from '@react-three/drei';
import { Suspense, useEffect, useMemo, useRef, type MutableRefObject } from 'react';
import * as THREE from 'three';
import { DogRig, type DogRigHandle } from '@/three/DogRig';
import { KIND_COLOR } from '@/lib/plans';
import { type House, type Recording } from './data';

type Props = { house: House; run: Recording | null; robots: number; clock: MutableRefObject<number>; time: number; overview: boolean; running: boolean; reduced: boolean; onReady: () => void; onFrame: () => void };
const COLORS = ['#f98c0a', '#fcffa4', '#cdbcc9'];
const SKINS = ['Cinder', 'Ember', 'Ash'] as const;
function polyShape(poly: number[][]) {
  const shape = new THREE.Shape();
  poly.forEach(([x, y], i) => i ? shape.lineTo(x, y) : shape.moveTo(x, y));
  shape.closePath();
  return shape;
}
function Floor({ poly, color, opacity = 1, level = 0 }: { poly: number[][]; color: string; opacity?: number; level?: number }) {
  const geometry = useMemo(() => new THREE.ShapeGeometry(polyShape(poly)), [poly]);
  useEffect(() => () => geometry.dispose(), [geometry]);
  return <mesh geometry={geometry} rotation-x={-Math.PI / 2} position-y={level} receiveShadow><meshStandardMaterial color={color} transparent={opacity < 1} opacity={opacity} roughness={0.95} side={THREE.DoubleSide} /></mesh>;
}
function Box({ at, size, color, rotation = 0 }: { at: [number, number, number]; size: [number, number, number]; color: string; rotation?: number }) {
  return <mesh position={at} rotation-y={rotation} castShadow receiveShadow><boxGeometry args={size} /><meshStandardMaterial color={color} roughness={0.8} /></mesh>;
}
function Figure({ x, y, found }: { x: number; y: number; found: boolean }) {
  const color = found ? '#fcffa4' : '#78687b';
  return <group position={[x, 0.03, -y]}>
    <mesh position-y={0.98}><sphereGeometry args={[0.15, 12, 10]} /><meshStandardMaterial color={color} emissive={color} emissiveIntensity={found ? 0.9 : 0} /></mesh>
    <mesh position-y={0.61}><capsuleGeometry args={[0.15, 0.34, 4, 8]} /><meshStandardMaterial color={color} emissive={color} emissiveIntensity={found ? 0.5 : 0} /></mesh>
    {[-0.09, 0.09].map(a => <mesh key={a} position={[a, 0.23, 0]}><capsuleGeometry args={[0.065, 0.3, 4, 6]} /><meshStandardMaterial color={color} /></mesh>)}
    {[-0.23, 0.23].map(a => <mesh key={a} position={[a, 0.57, 0]} rotation-z={a}><capsuleGeometry args={[0.055, 0.3, 4, 6]} /><meshStandardMaterial color={color} /></mesh>)}
    {found && <><mesh rotation-x={-Math.PI / 2} position-y={0.02}><ringGeometry args={[0.38, 0.45, 32]} /><meshBasicMaterial color={color} transparent opacity={0.85} /></mesh><Html position={[0, 1.4, 0]} center style={{ pointerEvents: 'none' }}><span className="whitespace-nowrap rounded-full bg-whiteheat px-2 py-0.5 text-[10px] font-extrabold text-heat-ink">Person found</span></Html></>}
  </group>;
}
function Haze({ fire, t, reduced }: { fire: number[][]; t: number; reduced: boolean }) {
  const texture = useMemo(() => {
    const c = document.createElement('canvas'); c.width = c.height = 64;
    const ctx = c.getContext('2d')!;
    const g = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
    g.addColorStop(0, 'rgba(150,129,145,0.6)'); g.addColorStop(1, 'rgba(100,80,100,0)');
    ctx.fillStyle = g; ctx.fillRect(0, 0, 64, 64);
    return new THREE.CanvasTexture(c);
  }, []);
  useEffect(() => () => texture.dispose(), [texture]);
  return <>{fire.slice(0, 6).map((p, k) => <group key={k} position={[p[0], 0, -p[1]]}>
    <mesh position-y={0.36} scale-y={reduced ? 1 : 1 + Math.sin(t * 9 + k) * 0.12}><coneGeometry args={[Math.max(0.12, p[2]), 0.8, 7]} /><meshBasicMaterial color="#f98c0a" transparent opacity={0.85} /></mesh>
    <mesh position-y={0.25}><coneGeometry args={[Math.max(0.06, p[2] * 0.5), 0.5, 7]} /><meshBasicMaterial color="#fcffa4" /></mesh>
    {k < 3 && <pointLight position={[0, 0.8, 0]} intensity={3.5} distance={4} color="#f98c0a" decay={1.5} />}
    {[0, 1, 2].map(i => <Billboard key={i} position={[Math.sin(k + i) * 0.4, 0.8 + i * 0.55 + (reduced ? 0 : (t * 0.15 + i * 0.15) % 0.5), 0]}><mesh><planeGeometry args={[2.4 + i * 0.5, 1.6]} /><meshBasicMaterial map={texture} transparent opacity={0.26} depthWrite={false} /></mesh></Billboard>)}
  </group>)}</>;
}
function World(props: Props) {
  const { house, run, robots, clock, overview, running, reduced, onReady, onFrame } = props;
  const rigs = useRef<(DogRigHandle | null)[]>([]);
  const { camera, size } = useThree();
  const [x0, y0, x1, y1] = house.bounds;
  const centre = useMemo(() => new THREE.Vector3((x0 + x1) / 2, 0, -(y0 + y1) / 2), [x0, y0, x1, y1]);
  const overviewCamera = useMemo(() => {
    const direction = new THREE.Vector3(0.42, 1.05, 0.85).normalize();
    const test = new THREE.PerspectiveCamera(42, size.width / size.height, 0.1, 250);
    const corners = [x0 - 1, x1 + 1].flatMap(x => [y0 - 1, y1 + 1].map(y => new THREE.Vector3(x, 0, -y)));
    let lo = 1, hi = 180;
    for (let j = 0; j < 25; j++) {
      const dist = (lo + hi) / 2;
      test.position.copy(centre).addScaledVector(direction, dist); test.lookAt(centre); test.updateMatrixWorld();
      const edge = Math.max(...corners.flatMap(v => { const p = v.clone().project(test); return [Math.abs(p.x), Math.abs(p.y)]; }));
      if (edge > 0.86) lo = dist; else hi = dist;
    }
    return centre.clone().addScaledVector(direction, hi);
  }, [centre, size.width, size.height, x0, y0, x1, y1]);
  const scratchRef = useRef({ look: centre.clone(), want: new THREE.Vector3(), target: new THREE.Vector3(), dist: [0, 0, 0], prior: [new THREE.Vector2(), new THREE.Vector2(), new THREE.Vector2()], initialized: false });
  useEffect(onReady, [onReady]);
  // Use the convention whose headings agree with the recorded direction of travel.
  const heading = useMemo(() => {
    if (!run) return 1;
    const candidates = [(y: number) => y, (y: number) => 90 - y, (y: number) => 90 + y, (y: number) => y - 90];
    const errors = [0, 0, 0, 0];
    for (let i = 1; i < run.frames.length; i++) run.frames[i].robots.forEach((r, k) => {
      const p = run.frames[i - 1].robots[k];
      const dx = r.x - p.x, dy = r.y - p.y;
      if (Math.hypot(dx, dy) > 0.1) {
        const motion = Math.atan2(dy, dx) * 180 / Math.PI;
        candidates.forEach((fn, j) => errors[j] += Math.min(90, Math.abs(((fn(r.yaw) - motion) % 360 + 540) % 360 - 180)));
      }
    });
    const best = errors.indexOf(Math.min(...errors));
    return best;
  }, [run]);
  const t = props.time;
  const index = Math.min(360, Math.floor(t * 2));
  const frame = run?.frames[index];
  const found = run?.frames.slice(0, index + 1).flatMap(f => f.found) || [];
  useFrame((_, rawDt) => {
    const scratch = scratchRef.current;
    const dt = Math.min(rawDt, 0.05), sim = clock.current;
    const i = Math.min(360, Math.floor(sim * 2));
    const a = run?.frames[i], b = run?.frames[Math.min(i + 1, 360)];
    const alpha = sim * 2 - i;
    const points = [];
    for (let k = 0; k < robots; k++) {
      const p = a?.robots[k], q = b?.robots[k];
      const x = p && q ? THREE.MathUtils.lerp(p.x, q.x, alpha) : house.pads[k][0];
      const y = p && q ? THREE.MathUtils.lerp(p.y, q.y, alpha) : house.pads[k][1];
      const speed = p && q ? Math.hypot(q.x - p.x, q.y - p.y) * 2 : 0;
      const prior = scratch.prior[k];
      const moved = prior.distanceTo(new THREE.Vector2(x, y));
      if (moved < 1) scratch.dist[k] += moved;
      prior.set(x, y);
      // dog.glb faces +X. The simulator's north maps to negative scene Z.
      rigs.current[k]?.drive(x, -y, ([p?.yaw || 0, 90 - (p?.yaw || 0), 90 + (p?.yaw || 0), (p?.yaw || 0) - 90][heading]) * Math.PI / 180, scratch.dist[k], speed, dt);
      points.push(new THREE.Vector3(x, 0.3, -y));
    }
    scratch.target.set(0, 0, 0); points.forEach(p => scratch.target.add(p)); scratch.target.divideScalar(robots);
    if (overview || !running || sim >= 180 || reduced) {
      scratch.target.copy(centre); scratch.want.copy(overviewCamera);
    } else {
      const spread = Math.max(...points.map(p => p.distanceTo(scratch.target)));
      const halfView = Math.atan(Math.tan(21 * Math.PI / 180) * Math.min(1, size.width / size.height));
      const distance = Math.max(size.width < 600 ? 13 : 11, spread / Math.sin(halfView) + 4);
      scratch.want.copy(scratch.target).add(new THREE.Vector3(0.35, 1.05, 0.95).normalize().multiplyScalar(distance));
    }
    const blend = scratch.initialized && !reduced ? 1 - Math.exp(-dt * 3) : 1;
    camera.position.lerp(scratch.want, blend); scratch.look.lerp(scratch.target, blend); camera.lookAt(scratch.look); scratch.initialized = true;
    if (running) onFrame();
  });
  return <>
    <color attach="background" args={['#0e0817']} />
    <fog attach="fog" args={['#0e0817', 38, 105]} />
    <hemisphereLight args={['#ffe8d2', '#3a2430', 1.8]} />
    <directionalLight position={[5, 18, 10]} intensity={2.4} color="#ffe9d6" castShadow shadow-mapSize={[1024, 1024]} shadow-camera-left={-35} shadow-camera-right={35} shadow-camera-top={35} shadow-camera-bottom={-35} shadow-normalBias={0.04} />
    <directionalLight position={[-8, 10, -10]} intensity={0.8} color="#cdbcc9" />
    <mesh rotation-x={-Math.PI / 2} position={[centre.x, -0.13, centre.z]} receiveShadow><planeGeometry args={[150, 150]} /><meshStandardMaterial color="#170d22" roughness={1} /></mesh>
    {house.rooms_poly.map((room, i) => <group key={room.id}>
      <Floor poly={room.poly} color={KIND_COLOR[room.kind] || '#3a2350'} />
      {(frame?.mapped[i] || 0) > 0 && <Floor poly={room.poly} color="#f98c0a" opacity={(frame!.mapped[i] / 100) * 0.28} level={0.008} />}
    </group>)}
    {house.walls.map(([a, b, c, d], i) => <Box key={i} at={[(a + c) / 2, 0.27, -(b + d) / 2]} size={[Math.hypot(c - a, d - b), 0.54, 0.1]} color="#b9a5ad" rotation={Math.atan2(d - b, c - a)} />)}
    {house.furniture.map((poly, i) => {
      const xs = poly.map(p => p[0]), ys = poly.map(p => p[1]);
      const w = Math.max(...xs) - Math.min(...xs), d = Math.max(...ys) - Math.min(...ys);
      return <Box key={i} at={[(Math.min(...xs) + Math.max(...xs)) / 2, 0.22, -(Math.min(...ys) + Math.max(...ys)) / 2]} size={[Math.max(0.08, w), 0.44, Math.max(0.08, d)]} color={i % 2 ? '#604251' : '#796070'} />;
    })}
    {house.people_truth.map(person => <Figure key={person.id} x={person.x} y={person.y} found={found.some(p => Math.hypot(p.x - person.x, p.y - person.y) <= 1.5)} />)}
    {SKINS.slice(0, robots).map((skin, i) => <group key={skin}>
      <DogRig ref={rig => { rigs.current[i] = rig; }} skin={skin} />
      <RobotLabel index={i} run={run} house={house} clock={clock} />
    </group>)}
    <Haze fire={frame?.fire || run?.frames[0].fire || []} t={t} reduced={reduced} />
  </>;
}
function RobotLabel({ index, run, house, clock }: { index: number; run: Recording | null; house: House; clock: MutableRefObject<number> }) {
  const group = useRef<THREE.Group>(null);
  useFrame(() => {
    const i = Math.max(0, Math.min(360, Math.floor(clock.current * 2)));
    const p = run?.frames[i].robots[index], q = run?.frames[Math.min(i + 1, 360)].robots[index];
    const a = clock.current * 2 - i;
    group.current?.position.set(p && q ? THREE.MathUtils.lerp(p.x, q.x, a) : house.pads[index][0], 1.15, -(p && q ? THREE.MathUtils.lerp(p.y, q.y, a) : house.pads[index][1]));
  });
  return <group ref={group}><Html center style={{ pointerEvents: 'none' }}><span className="whitespace-nowrap rounded-full border bg-stage/90 px-2 py-0.5 text-[11px] font-extrabold" style={{ color: COLORS[index], borderColor: COLORS[index] }}>Robot {'ABC'[index]}</span></Html></group>;
}
export default function Scene(props: Props) {
  return <Canvas shadows dpr={[1, 1.5]} camera={{ position: [12, 20, 18], fov: 42, near: 0.1, far: 250 }} gl={{ antialias: true, powerPreference: 'high-performance' }} fallback={<p className="p-8 text-sub">This browser cannot draw the 3D replay. Open this page in a browser with WebGL enabled.</p>} aria-label="Recorded robot rescue in a 3D cutaway house">
    <Suspense fallback={null}><World {...props} /></Suspense>
  </Canvas>;
}
