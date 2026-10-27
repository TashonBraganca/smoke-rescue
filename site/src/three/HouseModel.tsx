"use client";
// The house, built from the simulator's floor plan: muted floors by room kind, low warm-grey walls with a lighter
// top edge (an architect's cut-away), simple furniture, a glowing fire room, and small name pills over each room.
// Rooms "light up" when `lit.current[i]` goes to 1; this component eases the glow itself, so callers just set targets.
import { useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef, type RefObject } from "react";
import * as THREE from "three";
import { mergeGeometries } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import { KIND_COLOR } from "@/lib/plans";
import { WALL_H, type Layout } from "@/three/houseLayout";

const WALL_SIDE = new THREE.Color("#b5a698"), WALL_TOP = new THREE.Color("#ece2d4");
const WARM = new THREE.Color("#8c6a54"); // floors lean towards warm timber so the rooms read lit, not like a cellar
const LIT = new THREE.Color("#fcb040"), HEAT = new THREE.Color("#f98c0a");

type Part = { g: THREE.BufferGeometry; c: string };
const box = (w: number, h: number, d: number, x: number, y: number, z: number, c: string): Part => ({ g: new THREE.BoxGeometry(w, h, d).translate(x, y, z), c });
const cyl = (r: number, h: number, x: number, y: number, z: number, c: string, rt = r, seg = 20): Part => ({ g: new THREE.CylinderGeometry(rt, r, h, seg).translate(x, y, z), c });
const ball = (r: number, x: number, y: number, z: number, c: string): Part => ({ g: new THREE.SphereGeometry(r, 20, 14).translate(x, y, z), c });

/** Furniture in its own frame: width along X, depth along Z, front facing -Z, feet at y = 0. */
function furniturePieces(kind: string): Part[] {
  const wood = "#6b4a35", dark = "#3d2b22", cream = "#ddd3c6", steel = "#bdb8b0", white = "#ece7df";
  switch (kind) {
    case "blue_double_bed": { const w = 1.6, d = 2.0; return [
      box(w, 0.26, d, 0, 0.13, 0, dark), box(w - 0.08, 0.16, d - 0.12, 0, 0.34, -0.02, cream),
      box(w - 0.04, 0.05, d * 0.56, 0, 0.44, -d * 0.2, "#6b4a72"),
      box(0.56, 0.1, 0.34, -0.38, 0.47, d / 2 - 0.32, white), box(0.56, 0.1, 0.34, 0.38, 0.47, d / 2 - 0.32, white),
      box(w + 0.04, 0.8, 0.08, 0, 0.4, d / 2 - 0.02, "#4a3226")]; }
    case "apartment_modern_sofa": { const w = 2.1, d = 0.9; return [
      box(w, 0.4, d, 0, 0.2, 0, "#55465a"), box(w, 0.42, 0.22, 0, 0.61, d / 2 - 0.11, "#4b3e50"),
      box(0.2, 0.22, d, -w / 2 + 0.1, 0.51, 0, "#4b3e50"), box(0.2, 0.22, d, w / 2 - 0.1, 0.51, 0, "#4b3e50"),
      box(w / 2 - 0.22, 0.1, d - 0.3, -(w / 4 - 0.06), 0.45, -0.08, "#685870"), box(w / 2 - 0.22, 0.1, d - 0.3, w / 4 - 0.06, 0.45, -0.08, "#685870")]; }
    case "tall_wardrobe": { const w = 1.2, d = 0.6, h = 1.1; return [
      box(w, h, d, 0, h / 2, 0, wood), box(0.012, h - 0.12, 0.01, 0, h / 2, -d / 2 - 0.005, dark),
      box(0.03, 0.16, 0.03, -0.07, h * 0.55, -d / 2 - 0.02, "#c9a46a"), box(0.03, 0.16, 0.03, 0.07, h * 0.55, -d / 2 - 0.02, "#c9a46a")]; }
    case "wooden_desk_with_lamp": { const w = 1.2, d = 0.6, h = 0.75; const legs = [-1, 1].flatMap((sx) => [-1, 1].map((sz) => box(0.05, h - 0.05, 0.05, sx * (w / 2 - 0.05), (h - 0.05) / 2, sz * (d / 2 - 0.05), "#5a3c28"))); return [
      box(w, 0.05, d, 0, h - 0.025, 0, "#7a5232"), ...legs,
      cyl(0.08, 0.025, w / 2 - 0.16, h + 0.012, d / 2 - 0.16, dark), cyl(0.012, 0.32, w / 2 - 0.16, h + 0.17, d / 2 - 0.16, dark),
      cyl(0.11, 0.14, w / 2 - 0.16, h + 0.36, d / 2 - 0.16, "#f2c27a", 0.05)]; }
    case "antique_globe": return [cyl(0.17, 0.04, 0, 0.02, 0, "#5a3c28"), cyl(0.035, 0.56, 0, 0.3, 0, "#6a4a2e"), ball(0.26, 0, 0.84, 0, "#a07a3a"), { g: new THREE.TorusGeometry(0.29, 0.012, 6, 40).translate(0, 0.84, 0), c: "#c9a46a" }];
    case "utility_washer": { const g = new THREE.CylinderGeometry(0.19, 0.19, 0.03, 28).rotateX(Math.PI / 2).translate(0, 0.46, -0.31); return [box(0.6, 0.85, 0.6, 0, 0.425, 0, "#d9d4cc"), { g, c: "#3a3640" }, box(0.5, 0.06, 0.04, 0, 0.78, -0.3, "#9e978d")]; }
    case "clawfoot_bathtub": { const w = 1.7, d = 0.8; return [box(w, 0.55, d, 0, 0.3, 0, white), box(w - 0.16, 0.02, d - 0.16, 0, 0.571, 0, "#a9a39b"), ...[-1, 1].flatMap((sx) => [-1, 1].map((sz) => cyl(0.05, 0.06, sx * (w / 2 - 0.12), 0.03, sz * (d / 2 - 0.1), "#c9a46a")))]; }
    case "sink_with_mirror": return [box(0.6, 0.82, 0.45, 0, 0.41, 0, "#e3ded6"), box(0.46, 0.03, 0.3, 0, 0.83, -0.03, "#a9a39b"), cyl(0.015, 0.14, 0, 0.9, 0.12, steel)];
    case "console_table_with_vase": { const w = 1.1, d = 0.4, h = 0.78; return [box(w, 0.04, d, 0, h - 0.02, 0, "#6a4a32"), ...[-1, 1].flatMap((sx) => [-1, 1].map((sz) => box(0.04, h - 0.04, 0.04, sx * (w / 2 - 0.05), (h - 0.04) / 2, sz * (d / 2 - 0.05), "#5a3c28"))), cyl(0.06, 0.26, 0.25, h + 0.13, 0, "#bc3754", 0.035)]; }
    case "stove_and_hood": return [box(0.8, 0.9, 0.65, 0, 0.45, 0, "#c9c4bc"), box(0.74, 0.012, 0.58, 0, 0.906, 0, "#26222c"), box(0.6, 0.04, 0.02, 0, 0.75, -0.33, "#8d877e")];
    case "tall_steel_fridge": return [box(0.8, 1.15, 0.75, 0, 0.575, 0, steel), box(0.78, 0.012, 0.01, 0, 0.78, -0.38, "#77726b"), box(0.03, 0.35, 0.03, 0.3, 0.55, -0.4, "#77726b")];
    case "supply_boxes": return [box(0.5, 0.36, 0.4, -0.12, 0.18, 0, "#8a6a45"), box(0.42, 0.3, 0.36, 0.18, 0.15, 0.02, "#7a5c3a"), box(0.36, 0.26, 0.3, -0.08, 0.49, 0.02, "#957550")];
    default: return [box(0.6, 0.6, 0.6, 0, 0.3, 0, wood)];
  }
}

function withColour(g: THREE.BufferGeometry, c: THREE.Color, top?: THREE.Color) {
  const n = g.attributes.position.count, arr = new Float32Array(n * 3), nor = g.attributes.normal;
  for (let i = 0; i < n; i++) { const col = top && nor.getY(i) > 0.9 ? top : c; arr[i * 3] = col.r; arr[i * 3 + 1] = col.g; arr[i * 3 + 2] = col.b; }
  g.setAttribute("color", new THREE.BufferAttribute(arr, 3));
  return g;
}

let softTex: THREE.Texture | null = null;
/** A soft round dot, drawn once, for fire glow, floor glow and sparks. */
export function softDot() {
  if (softTex) return softTex;
  const c = document.createElement("canvas"); c.width = c.height = 128;
  const x = c.getContext("2d")!, g = x.createRadialGradient(64, 64, 0, 64, 64, 64);
  g.addColorStop(0, "rgba(255,255,255,1)"); g.addColorStop(0.35, "rgba(255,255,255,.55)"); g.addColorStop(1, "rgba(255,255,255,0)");
  x.fillStyle = g; x.fillRect(0, 0, 128, 128);
  softTex = new THREE.CanvasTexture(c); softTex.colorSpace = THREE.SRGBColorSpace;
  return softTex;
}

/** Fire: a flickering light, a tall layered flame of glow sprites, a glow on the floor and rising sparks. */
export function FireGlow({ at, reduce, power = 1 }: { at: THREE.Vector3; reduce?: boolean; power?: number }) {
  const light = useRef<THREE.PointLight>(null!);
  const fl = [useRef<THREE.Sprite>(null!), useRef<THREE.Sprite>(null!), useRef<THREE.Sprite>(null!), useRef<THREE.Sprite>(null!)];
  const N = 40;
  const sparks = useMemo(() => {
    const p = new Float32Array(N * 3), v = new Float32Array(N);
    for (let i = 0; i < N; i++) { p[i * 3] = (Math.random() - 0.5) * 0.8; p[i * 3 + 1] = Math.random() * 2; p[i * 3 + 2] = (Math.random() - 0.5) * 0.8; v[i] = 0.4 + Math.random() * 0.6; }
    const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.BufferAttribute(p, 3));
    return { g, v };
  }, []);
  useEffect(() => () => sparks.g.dispose(), [sparks]);
  const tex = useMemo(() => softDot(), []);
  // [colour, width, height, y, opacity] from the outer glow to the white-hot core
  const LAYERS: [string, number, number, number, number][] = [["#bc3754", 3.6, 3.2, 0.9, 0.42], ["#f98c0a", 1.15, 2.1, 0.85, 0.85], ["#fcb040", 0.75, 1.35, 0.6, 0.9], ["#fcffa4", 0.42, 0.7, 0.38, 0.9]];
  useFrame(({ clock }, dt) => {
    const t = clock.elapsedTime, m = reduce ? 0 : 1;
    const f = 1 - m * (0.18 - 0.1 * Math.sin(t * 11.3) - 0.08 * Math.sin(t * 23.7 + 1.3));
    light.current.intensity = 30 * f * power;
    for (let i = 0; i < 4; i++) {
      const [, w, h, y] = LAYERS[i], s = fl[i].current, ph = i * 1.7;
      const fx = 1 + m * 0.08 * Math.sin(t * (7 + i * 2.3) + ph), fy = 1 + m * 0.14 * Math.sin(t * (9 + i * 1.9) + ph * 2);
      s.scale.set(w * fx, h * fy, 1); s.position.set(m * 0.04 * Math.sin(t * 3.1 + ph), y * (0.96 + 0.04 * fy), 0);
    }
    if (reduce) return;
    const a = sparks.g.attributes.position.array as Float32Array;
    for (let i = 0; i < N; i++) {
      a[i * 3 + 1] += sparks.v[i] * Math.min(dt, 0.05); a[i * 3] += Math.sin(t * 1.7 + i) * 0.002;
      if (a[i * 3 + 1] > 2.2) { a[i * 3 + 1] = 0.2; a[i * 3] = (Math.random() - 0.5) * 0.8; a[i * 3 + 2] = (Math.random() - 0.5) * 0.8; }
    }
    sparks.g.attributes.position.needsUpdate = true;
  });
  return (
    <group position={[at.x, 0, at.z]}>
      <pointLight ref={light} position={[0, 1.0, 0]} color="#ff6a10" intensity={30} distance={11} decay={1.6} />
      <mesh rotation-x={-Math.PI / 2} position={[0, 0.02, 0]}>
        <planeGeometry args={[4.6, 4.6]} />
        <meshBasicMaterial map={tex} color="#f98c0a" transparent opacity={0.5} depthWrite={false} blending={THREE.AdditiveBlending} toneMapped={false} />
      </mesh>
      {LAYERS.map(([c, , , , o], i) => (
        <sprite key={i} ref={fl[i]}><spriteMaterial map={tex} color={c} transparent opacity={o} depthWrite={false} blending={THREE.AdditiveBlending} toneMapped={false} /></sprite>
      ))}
      <points geometry={sparks.g}><pointsMaterial map={tex} color="#fcb040" size={0.08} transparent depthWrite={false} blending={THREE.AdditiveBlending} toneMapped={false} /></points>
    </group>
  );
}

export type LabelEls = RefObject<(HTMLElement | null)[]>;

export function HouseModel({ layout, lit, labels, reduce }: { layout: Layout; lit: RefObject<Float32Array>; labels?: LabelEls; reduce?: boolean }) {
  const { walls, furniture } = useMemo(() => {
    const wg = layout.walls.map((s) => withColour(new THREE.BoxGeometry(s.lx, WALL_H, s.lz).translate(s.x, WALL_H / 2, s.z), WALL_SIDE, WALL_TOP));
    const walls = mergeGeometries(wg); wg.forEach((g) => g.dispose());
    const parts: THREE.BufferGeometry[] = [];
    const m = new THREE.Matrix4();
    for (const f of layout.furniture) {
      m.makeRotationY(f.ry).setPosition(f.x, 0, f.z);
      for (const p of furniturePieces(f.kind)) { p.g.applyMatrix4(m); parts.push(withColour(p.g, new THREE.Color(p.c))); }
    }
    for (const g of parts) for (const k of Object.keys(g.attributes)) if (!["position", "normal", "uv", "color"].includes(k)) g.deleteAttribute(k);
    const flat = parts.map((g) => (g.index ? g.toNonIndexed() : g));
    const furniture = flat.length ? mergeGeometries(flat) : null;
    parts.forEach((g) => g.dispose()); flat.forEach((g) => g.dispose());
    return { walls, furniture };
  }, [layout]);
  useEffect(() => () => { walls?.dispose(); furniture?.dispose(); }, [walls, furniture]);

  // floors: dim until searched, then they brighten to their own muted colour with a faint warm glow
  const floors = useMemo(() => layout.rooms.map((r) => {
    const base = new THREE.Color(KIND_COLOR[r.kind] ?? "#2b2433").lerp(WARM, 0.5).multiplyScalar(1.45);
    const dim = base.clone().multiplyScalar(0.62);
    const mat = new THREE.MeshStandardMaterial({ color: dim.clone(), roughness: 0.92, metalness: 0, emissive: r.fire ? HEAT : LIT, emissiveIntensity: 0 });
    return { r, mat, base, dim, w: r.x1 - r.x0 - 0.02, d: r.z1 - r.z0 - 0.02 };
  }), [layout]);
  useEffect(() => () => floors.forEach((f) => f.mat.dispose()), [floors]);

  const cur = useMemo(() => new Float32Array(layout.rooms.length), [layout]);
  const size = useThree((s) => s.size);
  const v = useMemo(() => new THREE.Vector3(), []);
  useFrame(({ clock, camera }, dt) => {
    const k = 1 - Math.exp(-Math.min(dt, 0.05) * 4), t = clock.elapsedTime, L = lit.current;
    const els = labels?.current;
    for (let i = 0; i < floors.length; i++) {
      const target = L ? L[i] ?? 0 : 0;
      cur[i] += (target - cur[i]) * (reduce ? 1 : k);
      const f = floors[i], c = cur[i];
      f.mat.color.copy(f.dim).lerp(f.base, c);
      f.mat.emissiveIntensity = f.r.fire ? 0.22 + 0.06 * Math.sin(t * 3.1) * (reduce ? 0 : 1) : c * 0.07;
      const el = els?.[i];
      if (el) {
        v.copy(f.r.centre).setY(0.3).project(camera);
        const o = v.z > 1 ? 0 : f.r.fire ? 1 : c;
        el.style.opacity = o.toFixed(3);
        el.style.transform = `translate3d(${((v.x + 1) / 2 * size.width).toFixed(1)}px, ${((1 - v.y) / 2 * size.height + (1 - o) * 6).toFixed(1)}px, 0) translate(-50%, -50%)`;
      }
    }
  });

  const pad = 0.3;
  const doorYaw = Math.atan2(layout.entry.out.x, layout.entry.out.z);
  return (
    <group>
      {/* plinth under the house and the dark ground around it */}
      <mesh position={[0, -0.06, 0]} receiveShadow>
        <boxGeometry args={[layout.w + pad * 2, 0.12, layout.d + pad * 2]} />
        <meshStandardMaterial color="#2a2030" roughness={0.95} />
      </mesh>
      <mesh rotation-x={-Math.PI / 2} position={[0, -0.12, 0]} receiveShadow>
        <planeGeometry args={[160, 160]} />
        <meshStandardMaterial color="#140c1c" roughness={1} />
      </mesh>
      {floors.map((f) => (
        <mesh key={f.r.id} rotation-x={-Math.PI / 2} position={[f.r.centre.x, 0.005, f.r.centre.z]} receiveShadow material={f.mat}>
          <planeGeometry args={[f.w, f.d]} />
        </mesh>
      ))}
      <mesh geometry={walls} castShadow receiveShadow>
        <meshStandardMaterial vertexColors roughness={0.86} />
      </mesh>
      {furniture && (
        <mesh geometry={furniture} castShadow receiveShadow>
          <meshStandardMaterial vertexColors roughness={0.7} />
        </mesh>
      )}
      {/* a warm strip across the front door */}
      <mesh position={[layout.entry.door.x, 0.012, layout.entry.door.z]} rotation={[-Math.PI / 2, 0, doorYaw]}>
        <planeGeometry args={[1.0, 0.16]} />
        <meshBasicMaterial color="#fcb040" toneMapped={false} />
      </mesh>
      {layout.fires.map((f, i) => <FireGlow key={i} at={f} reduce={reduce} />)}
    </group>
  );
}

/** The room name pills, as plain DOM over the canvas. HouseModel moves them every frame after the camera. */
export function RoomLabels({ layout, els }: { layout: Layout; els: LabelEls }) {
  return (
    <div aria-hidden="true" className="pointer-events-none absolute inset-0 overflow-hidden">
      {layout.rooms.map((r, i) => (
        <span
          key={r.id}
          ref={(el) => { if (els.current) els.current[i] = el; }}
          style={{ opacity: 0, transform: "translate3d(-999px,-999px,0)" }}
          className={`absolute left-0 top-0 flex items-center gap-1.5 whitespace-nowrap rounded-full border px-2.5 py-1 text-[11px] font-bold leading-none tracking-wide will-change-transform ${r.fire ? "border-heat/70 bg-[#1a0a00]/85 text-amber" : "border-white/12 bg-stage/85 text-ink"}`}
        >
          {r.fire && <span className="size-1.5 rounded-full bg-heat shadow-[0_0_8px_2px_rgba(249,140,10,.7)]" />}
          {r.name}
        </span>
      ))}
    </div>
  );
}
