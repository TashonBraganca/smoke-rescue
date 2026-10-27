"use client";
// The two dogs, Cinder and Ember, on a softly lit round platform. It turns slowly on its own; drag to spin it.
// The platform follows a critically damped spring (no overshoot, no jitter).
import { ContactShadows, Environment, Lightformer } from "@react-three/drei";
import { BlobShadow, useAdaptiveDpr, useLowPower } from "./quality";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { SKINS, SKIN_NAMES } from "@/lib/skins";
import { Dog } from "@/three/models";
import { softDot } from "@/three/HouseModel";
import { Ready } from "@/three/sceneKit";

const DOGS = SKIN_NAMES.slice(0, 2), STEP = Math.PI, R = 1.05, PLATE = 2.05;
// platform angle that puts the middle of the standing team at the front
const FACE = -STEP / 2;

/** Unity-style SmoothDamp: frame-rate independent critically damped spring. */
function damp(cur: number, target: number, v: { v: number }, time: number, dt: number) {
  const o = 2 / time, x = o * dt, e = 1 / (1 + x + 0.48 * x * x + 0.235 * x * x * x);
  const ch = cur - target, tmp = (v.v + o * ch) * dt;
  v.v = (v.v - o * tmp) * e;
  return target + (ch + tmp) * e;
}

type Spin = { cur: number; vel: { v: number }; target: number; time: number; drag: boolean };

function Platform({ spin, reduce, low }: { spin: React.RefObject<Spin>; reduce: boolean; low: boolean }) {
  const group = useRef<THREE.Group>(null!);
  const riders = useRef<(THREE.Group | null)[]>([]);
  const rings = useRef<(THREE.MeshBasicMaterial | null)[]>([]);
  const glow = useMemo(() => softDot(), []);
  useFrame((_, rawDt) => {
    const s = spin.current, dt = Math.min(rawDt, 0.05);
    if (!reduce && !s.drag) s.target += dt * 0.22;
    s.cur = reduce ? s.target : damp(s.cur, s.target, s.vel, s.time, dt);
    group.current.rotation.y = s.cur;
    const k = reduce ? 1 : 1 - Math.exp(-dt * 6);
    // each dog keeps a three-quarter pose towards the viewer as it rides round, turning only a little with the platform
    for (let i = 0; i < DOGS.length; i++) {
      const g = riders.current[i]; if (!g) continue;
      const w = i * STEP + s.cur; // the dog's angle around the platform in world space
      g.rotation.y = -Math.PI / 4 + 0.35 * Math.sin(w) - s.cur;
      // a ring of the dog's own paint under each dog on the team
      const m = rings.current[i]; if (m) m.opacity += (0.85 - m.opacity) * k;
    }
  });
  return (
    <group>
      {/* soft pool of warm light the platform floats on */}
      <mesh rotation-x={-Math.PI / 2} position={[0, -0.16, 0]}>
        <planeGeometry args={[7.5, 7.5]} />
        <meshBasicMaterial map={glow} color="#f98c0a" transparent opacity={0.2} depthWrite={false} blending={THREE.AdditiveBlending} toneMapped={false} />
      </mesh>
      <mesh position={[0, -0.075, 0]}>
        <cylinderGeometry args={[PLATE, PLATE + 0.05, 0.15, 128]} />
        <meshStandardMaterial color="#3a2c40" metalness={0.35} roughness={0.5} envMapIntensity={0.7} />
      </mesh>
      {/* a soft pool of light on the top */}
      <mesh rotation-x={-Math.PI / 2} position={[0, 0.0015, 0.25]}>
        <circleGeometry args={[PLATE * 0.95, 96]} />
        <meshBasicMaterial map={glow} color="#ffd9b0" transparent opacity={0.16} depthWrite={false} blending={THREE.AdditiveBlending} toneMapped={false} />
      </mesh>
      {/* a thin heat-orange edge */}
      <mesh rotation-x={-Math.PI / 2} position={[0, 0.001, 0]}>
        <ringGeometry args={[PLATE - 0.035, PLATE - 0.012, 160]} />
        <meshBasicMaterial color="#f98c0a" toneMapped={false} transparent opacity={0.9} />
      </mesh>
      <group ref={group}>
        {DOGS.map((name, i) => {
          const a = i * STEP;
          return (
            <group key={name} ref={(g) => { riders.current[i] = g; }} position={[R * Math.sin(a), 0, R * Math.cos(a)]}>
              <mesh rotation-x={-Math.PI / 2} position={[0, 0.004, 0]}>
                <ringGeometry args={[0.5, 0.56, 64]} />
                <meshBasicMaterial ref={(m) => { rings.current[i] = m; }} color={SKINS[name].hex} transparent opacity={0.85} depthWrite={false} toneMapped={false} />
              </mesh>
              <Dog skin={name} clip="idle" fade={0.6} scale={0.95} />
              {low && <BlobShadow w={1.2} d={0.7} />}
            </group>
          );
        })}
      </group>
      {!low && <ContactShadows position={[0, 0.003, 0]} scale={PLATE * 2.1} resolution={512} blur={2.6} far={1.4} opacity={0.8} color="#05020a" />}
    </group>
  );
}

function Rig({ reduce }: { reduce: boolean }) {
  const { camera, size } = useThree();
  const look = useMemo(() => new THREE.Vector3(0, 0.38, 0.2), []);
  useFrame(({ clock }) => {
    const aspect = size.width / size.height, cam = camera as THREE.PerspectiveCamera;
    // keep both side dogs in frame on narrow screens by stepping back
    const tan = Math.tan((cam.fov * Math.PI) / 360);
    const dist = Math.max(5.5, 2.1 / (tan * aspect));
    const sway = reduce ? 0 : Math.sin(clock.elapsedTime * 0.25) * 0.05;
    camera.position.set(Math.sin(sway) * dist, 1.25 + dist * 0.16, Math.cos(sway) * dist);
    camera.lookAt(look);
  });
  return null;
}

export default function Turntable({ reduce, onReady, paused = false }: { reduce: boolean; onReady: () => void; paused?: boolean }) {
  const low = useLowPower();
  const [dpr, monitor] = useAdaptiveDpr(low);
  const spin = useRef<Spin>({ cur: FACE, vel: { v: 0 }, target: FACE, time: 0.55, drag: false });
  const lastX = useRef(0);
  const wrapEl = useRef<HTMLDivElement>(null);
  const [active, setActive] = useState(true);

  useEffect(() => {
    const el = wrapEl.current; if (!el) return;
    const io = new IntersectionObserver(([e]) => setActive(e.isIntersecting), { rootMargin: "120px 0px" });
    io.observe(el); return () => io.disconnect();
  }, []);

  return (
    <div ref={wrapEl} className="absolute inset-0 cursor-grab touch-pan-y active:cursor-grabbing"
      onPointerDown={(e) => { spin.current.drag = true; lastX.current = e.clientX; (e.target as HTMLElement).setPointerCapture?.(e.pointerId); }}
      onPointerMove={(e) => { if (!spin.current.drag) return; spin.current.target += (e.clientX - lastX.current) * 0.012; lastX.current = e.clientX; }}
      onPointerUp={() => { spin.current.drag = false; }} onPointerCancel={() => { spin.current.drag = false; }}>
      <Canvas
        dpr={dpr}
        frameloop={active && !paused ? "always" : "never"} // paused under the full-screen mission player
        camera={{ fov: 26, near: 0.1, far: 60, position: [0, 2, 7.4] }}
        gl={{ antialias: true, alpha: true, powerPreference: "high-performance" }}
        aria-hidden="true"
      >
        <hemisphereLight args={["#fff1e0", "#2a1530", 0.45]} />
        <directionalLight position={[3.2, 5, 4.2]} intensity={2.4} color="#ffe2c4" />
        <directionalLight position={[-3.5, 2.4, -4]} intensity={3.2} color="#ff7a1a" />
        <directionalLight position={[-4, 1.2, 3]} intensity={0.45} color="#bc3754" />
        <Environment resolution={256} environmentIntensity={0.55}>
          <Lightformer form="rect" intensity={2.2} color="#ffd9b0" position={[0, 5, 3]} rotation-x={Math.PI / 2.4} scale={[8, 3, 1]} />
          <Lightformer form="rect" intensity={3} color="#f98c0a" position={[-5, 1.5, -3]} rotation-y={Math.PI / 3} scale={[3, 4, 1]} />
          <Lightformer form="ring" intensity={1.2} color="#fcb040" position={[5, 2, 1]} rotation-y={-Math.PI / 2} scale={2} />
        </Environment>
        {monitor}
        <Suspense fallback={null}>
          <Platform spin={spin} reduce={reduce} low={low} />
          <Ready onReady={onReady} />
        </Suspense>
        <Rig reduce={reduce} />
      </Canvas>
    </div>
  );
}
