"use client";
// The picked house as a small 3D model: roof off, fire burning, every room lit, slowly turning. Drag to orbit.
import { OrbitControls } from "@react-three/drei";
import { Canvas, useThree } from "@react-three/fiber";
import { Suspense, useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { HouseModel } from "@/three/HouseModel";
import { fitOverview, type Layout } from "@/three/houseLayout";
import { useAdaptiveDpr, useLowPower } from "@/three/quality";
import { HouseLights, Ready } from "@/three/sceneKit";

function Fit({ layout }: { layout: Layout }) {
  const { camera, size } = useThree();
  const controls = useThree((s) => s.controls) as unknown as { target: THREE.Vector3; update: () => void } | null;
  useEffect(() => {
    const aspect = size.width / size.height, c = camera as THREE.PerspectiveCamera;
    c.fov = 34; c.updateProjectionMatrix();
    const o = fitOverview(layout, aspect, 34, 48, 1.08);
    camera.position.copy(o.pos);
    if (controls) { controls.target.copy(o.target); controls.update(); } else camera.lookAt(o.target);
  }, [layout, camera, size, controls]);
  return null;
}

export default function HousePreview({ layout, active, reduce, onReady }: { layout: Layout; active: boolean; reduce: boolean; onReady: () => void }) {
  const low = useLowPower();
  const [dpr, monitor] = useAdaptiveDpr(low);
  const lit = useRef(new Float32Array(layout.rooms.length).fill(1));
  useMemo(() => { lit.current = new Float32Array(layout.rooms.length).fill(1); }, [layout]);
  return (
    <Canvas dpr={dpr} shadows={!low} frameloop={active ? "always" : "never"} camera={{ fov: 34, near: 0.1, far: 200 }}
      gl={{ antialias: true, alpha: true, powerPreference: "high-performance", toneMapping: THREE.ACESFilmicToneMapping }}>
      {monitor}
      <HouseLights layout={layout} shadows={!low} />
      <Suspense fallback={null}>
        <HouseModel layout={layout} lit={lit} reduce={reduce} />
        <Ready onReady={onReady} />
      </Suspense>
      <OrbitControls makeDefault enablePan={false} enableZoom={false} autoRotate={!reduce} autoRotateSpeed={0.7}
        minPolarAngle={0.25} maxPolarAngle={1.25} enableDamping />
      <Fit layout={layout} />
    </Canvas>
  );
}
