"use client";
// A dog whose legs are driven by distance walked, not by the clock: scrubbing the page forwards or backwards
// moves the trot cycle exactly with the body, so the feet never slide. The parent's useFrame calls drive().
import { useGLTF } from "@react-three/drei";
import { forwardRef, useEffect, useImperativeHandle, useMemo, useRef } from "react";
import * as THREE from "three";
import { clone as cloneSkinned } from "three/examples/jsm/utils/SkeletonUtils.js";
import { DOG_URL } from "@/three/models";
import { SKINS, type SkinName } from "@/lib/skins";

const STRIDE = 1.4; // metres per trot cycle (the 1 s clip at the dog's natural 1.4 m/s)

export type DogRigHandle = {
  root: THREE.Group;
  /** place the dog and pose it: dist = metres walked so far, speed in m/s (decides trot vs idle), dt in seconds */
  drive: (x: number, z: number, yaw: number, dist: number, speed: number, dt: number) => void;
};

function paint(model: THREE.Object3D, skin: SkinName) {
  const s = SKINS[skin];
  model.traverse((o) => {
    const mesh = o as THREE.Mesh;
    if (!mesh.isMesh) return;
    const mat = mesh.material as THREE.MeshStandardMaterial;
    if (mat.name === "BodyShell") { mat.color.set(s.hex).multiplyScalar(s.shade); mat.metalness = 0.12; mat.roughness = 0.5; }
    if (mat.name === "LegSteel") mat.color.setHex(s.legs);
  });
}

export const DogRig = forwardRef<DogRigHandle, { skin: SkinName; shadows?: boolean }>(function DogRig({ skin, shadows = true }, ref) {
  const { scene, animations } = useGLTF(DOG_URL);
  const model = useMemo(() => {
    const m = cloneSkinned(scene);
    m.traverse((o) => {
      const mesh = o as THREE.Mesh;
      if (!mesh.isMesh) return;
      mesh.castShadow = shadows; mesh.frustumCulled = false;
      mesh.material = (mesh.material as THREE.MeshStandardMaterial).clone();
    });
    return m;
  }, [scene, shadows]);
  useEffect(() => paint(model, skin), [model, skin]);
  useEffect(() => () => model.traverse((o) => { const m = o as THREE.Mesh; if (m.isMesh) (m.material as THREE.Material).dispose(); }), [model]);

  const root = useRef<THREE.Group>(null!);
  const rig = useMemo(() => {
    const mixer = new THREE.AnimationMixer(model);
    const clip = (n: string) => animations.find((a) => a.name === n)!;
    const idle = mixer.clipAction(clip("idle")), trot = mixer.clipAction(clip("trot"));
    idle.play(); trot.play(); trot.timeScale = 0; trot.setEffectiveWeight(0);
    const dur = trot.getClip().duration;
    let w = 0;
    return {
      mixer,
      drive(x: number, z: number, yaw: number, dist: number, speed: number, dt: number) {
        const g = root.current; if (!g) return;
        g.position.set(x, 0, z); g.rotation.y = yaw;
        const want = speed > 0.12 ? 1 : 0;
        w += (want - w) * (1 - Math.exp(-dt * 9));
        trot.time = ((((dist / STRIDE) % 1) + 1) % 1) * dur;
        trot.setEffectiveWeight(w); idle.setEffectiveWeight(1 - w);
        mixer.update(dt);
      },
    };
  }, [model, animations]);
  useEffect(() => () => { rig.mixer.stopAllAction(); rig.mixer.uncacheRoot(model); }, [rig, model]);
  useImperativeHandle(ref, () => ({ get root() { return root.current; }, drive: rig.drive }), [rig]);
  return <group ref={root}><primitive object={model} /></group>;
});
