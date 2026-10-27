"use client";
import { PerformanceMonitor } from "@react-three/drei";
import { useState, useSyncExternalStore } from "react";
import * as THREE from "three";

// Phones and small or weak machines get the light render path: no post effects, no shadow maps, blob shadows,
// lower pixel density. Their GPUs are many times slower than a laptop's, and the full path made them stutter and heat up.
function lowPower() {
  const coarse = window.matchMedia("(pointer: coarse)").matches;
  const small = Math.min(window.screen.width, window.screen.height) < 820;
  const cores = navigator.hardwareConcurrency || 8;
  const mem = (navigator as Navigator & { deviceMemory?: number }).deviceMemory ?? 8;
  return coarse || small || cores <= 4 || mem <= 4;
}
const noop = () => () => {};
/** True on phones and weak devices. Always false in the server HTML; canvases render on the client only. */
export function useLowPower() {
  return useSyncExternalStore(noop, lowPower, () => false);
}

/** Pixel density that follows the device: starts modest, steps down when frames run slow, back up when there is room.
 *  Returns the value for <Canvas dpr> and a monitor element to place inside the Canvas. */
export function useAdaptiveDpr(low: boolean) {
  const max = low ? 1.5 : 2;
  const [dpr, setDpr] = useState(low ? 1.25 : 1.75);
  const monitor = (
    <PerformanceMonitor
      flipflops={3}
      onIncline={() => setDpr((d) => Math.min(max, +(d + 0.25).toFixed(2)))}
      onDecline={() => setDpr((d) => Math.max(0.75, +(d - 0.25).toFixed(2)))}
      onFallback={() => setDpr(1)}
    />
  );
  return [[0.75, Math.min(dpr, max)] as [number, number], monitor] as const;
}

// A soft dark ellipse on the floor: the light path's stand-in for contact shadows (one quad, no extra scene render).
let blobTex: THREE.Texture | null = null;
function blobTexture() {
  if (blobTex) return blobTex;
  const c = document.createElement("canvas"); c.width = c.height = 128;
  const g = c.getContext("2d")!, r = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  r.addColorStop(0, "rgba(0,0,0,0.75)"); r.addColorStop(0.55, "rgba(0,0,0,0.35)"); r.addColorStop(1, "rgba(0,0,0,0)");
  g.fillStyle = r; g.fillRect(0, 0, 128, 128);
  blobTex = new THREE.CanvasTexture(c); return blobTex;
}
export function BlobShadow({ w = 1.3, d = 0.75, opacity = 0.85, y = 0.004 }: { w?: number; d?: number; opacity?: number; y?: number }) {
  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, y, 0]} scale={[w, d, 1]} renderOrder={-1}>
      <planeGeometry />
      <meshBasicMaterial map={blobTexture()} transparent opacity={opacity} depthWrite={false} toneMapped={false} />
    </mesh>
  );
}
