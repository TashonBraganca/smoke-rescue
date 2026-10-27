"use client";
import { useGLTF, useAnimations, useTexture } from "@react-three/drei";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { clone as cloneSkinned } from "three/examples/jsm/utils/SkeletonUtils.js";
import { SKINS, type SkinName } from "@/lib/skins";

export const DOG_URL = "/models/dog.glb";
export const FF_URL = "/models/firefighter.glb";
useGLTF.preload(DOG_URL);
useGLTF.preload(FF_URL);

export type DogClip = "idle" | "trot" | "sit" | "hop";
type GroupProps = Omit<React.ComponentProps<"group">, "ref">;

/** The robot dog: metric, front at +X, feet at y = 0, clips idle (2 s), trot (1 s), sit (1.5 s, once), hop (1.5 s). */
export function Dog({ skin = "Cinder", clip = "idle", fade = 0.3, onModel, ...props }: GroupProps & { skin?: SkinName; clip?: DogClip; fade?: number; onModel?: (m: THREE.Object3D) => void }) {
  const { scene, animations } = useGLTF(DOG_URL);
  const model = useMemo(() => {
    const m = cloneSkinned(scene);
    m.traverse((o) => {
      const mesh = o as THREE.Mesh;
      if (!mesh.isMesh) return;
      mesh.castShadow = true; mesh.frustumCulled = false;
      mesh.material = (mesh.material as THREE.MeshStandardMaterial).clone();
    });
    return m;
  }, [scene]);
  useEffect(() => {
    const s = SKINS[skin];
    model.traverse((o) => {
      const mesh = o as THREE.Mesh;
      if (!mesh.isMesh) return;
      const mat = mesh.material as THREE.MeshStandardMaterial;
      if (mat.name === "BodyShell") { mat.color.set(s.hex).multiplyScalar(s.shade); mat.metalness = 0.12; mat.roughness = 0.5; }
      if (mat.name === "LegSteel") mat.color.setHex(s.legs);
    });
  }, [model, skin]);
  useEffect(() => { onModel?.(model); }, [model, onModel]);
  const group = useRef<THREE.Group>(null);
  const { actions } = useAnimations(animations, group);
  useEffect(() => {
    const a = actions[clip];
    if (!a) return;
    if (clip === "sit") { a.setLoop(THREE.LoopOnce, 1); a.clampWhenFinished = true; }
    a.reset().fadeIn(fade).play();
    return () => { a.fadeOut(fade); };
  }, [actions, clip, fade]);
  return <group ref={group} {...props}><primitive object={model} /></group>;
}

/** Points on the dog in its own frame (lidar on the back, cameras on the face panel), measured from its painted parts. */
export function dogAnchors(model: THREE.Object3D) {
  // measure with the model detached so a turned parent group cannot leak into the numbers
  const parent = model.parent; parent?.remove(model); model.updateMatrixWorld(true);
  try { return measure(model); } finally { parent?.add(model); model.updateMatrixWorld(true); }
}
function measure(model: THREE.Object3D) {
  const box = (name: string) => { const b = new THREE.Box3(); model.traverse((o) => { const m = o as THREE.Mesh; if (m.isMesh && (m.material as THREE.Material).name === name) b.expandByObject(m); }); return b; };
  const body = box("BodyShell"), face = box("FaceGraphite");
  const bc = body.getCenter(new THREE.Vector3());
  const fc = face.isEmpty() ? new THREE.Vector3(body.max.x, bc.y, bc.z) : face.getCenter(new THREE.Vector3());
  return {
    back: new THREE.Vector3(bc.x, body.max.y + 0.02, bc.z),
    face: new THREE.Vector3(face.isEmpty() ? body.max.x : face.max.x, fc.y, fc.z),
    centre: bc,
    size: body.getSize(new THREE.Vector3()),
  };
}

/** The firefighter, rigged on a Mixamo skeleton. Clips idle, talk_radio, point, kneel. The web export lost its
 *  opacity maps (mask visor, belt straps), so they are re-applied here by material name. Faces +Z. */
export function Firefighter({ clip = "idle", ...props }: GroupProps & { clip?: "idle" | "talk_radio" | "point" | "kneel" }) {
  const { scene, animations } = useGLTF(FF_URL);
  const [oxy, belt] = useTexture(["/tex/oxygen_opacity.jpg", "/tex/belt_opacity.jpg"]);
  const model = useMemo(() => {
    for (const t of [oxy, belt]) { t.flipY = false; t.colorSpace = THREE.NoColorSpace; }
    const m = cloneSkinned(scene);
    m.traverse((o) => {
      const mesh = o as THREE.Mesh;
      if (!mesh.isMesh) return;
      mesh.castShadow = true; mesh.frustumCulled = false;
      const mat = (mesh.material as THREE.MeshStandardMaterial).clone();
      if (mat.name.startsWith("Oxygen system")) { mat.alphaMap = oxy; mat.alphaTest = 0.5; }
      if (mat.name.startsWith("Belt")) { mat.alphaMap = belt; mat.alphaTest = 0.5; }
      mesh.material = mat;
    });
    return m;
  }, [scene, oxy, belt]);
  const group = useRef<THREE.Group>(null);
  const { actions } = useAnimations(animations, group);
  useEffect(() => { const a = actions[clip]; a?.reset().fadeIn(0.4).play(); return () => { a?.fadeOut(0.4); }; }, [actions, clip]);
  return <group ref={group} {...props}><primitive object={model} /></group>;
}
