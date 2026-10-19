"""DINOv2 value facets, neutral-set VLAD and compact place fingerprints.

The fixed COCO128 vocabulary is shared before deployment. No recording images
enter vocabulary fitting. L/14 uses zero-based block 19 of 24; B/14 uses block
9 of 12. Both are late attention value facets, rather than CLS tokens. Features
are unit normalized. VLAD residuals receive intra and global L2 normalization,
then a seeded sparse random projection to 512 float16 values.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import time
import warnings

import cv2
import numpy as np
import torch
from scipy import sparse

from brain.io import read_associations


# Pin model code. The hub loads the public LVD-142M pretrained weights.
DINO_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
IMAGE_SIZE = (322, 238)


def unit(array: np.ndarray, axis: int = -1) -> np.ndarray:
    return array / np.maximum(np.linalg.norm(array, axis=axis, keepdims=True), 1e-12)


class VLAD:
    def __init__(self, centers: np.ndarray, seed: int = 1729):
        self.centers = np.asarray(centers, dtype=np.float32)
        if self.centers.ndim != 2 or not np.isfinite(self.centers).all():
            raise ValueError("centers must be a finite matrix")
        # Three signed entries per input coordinate. This is a shared sparse
        # random projection, not a PCA fitted on either evaluation trajectory.
        rng = np.random.default_rng(seed)
        size = self.centers.size
        rows = np.repeat(np.arange(size), 3)
        cols = rng.integers(0, 512, size=size * 3)
        values = rng.choice([-1., 1.], size=size * 3) / np.sqrt(3.)
        self.projection = sparse.csr_matrix((values, (rows, cols)), shape=(size, 512), dtype=np.float32)

    def aggregate(self, patches: np.ndarray) -> np.ndarray:
        patches = np.asarray(patches, dtype=np.float32)
        if (patches.ndim != 2 or patches.shape[1] != self.centers.shape[1]
                or not len(patches) or not np.isfinite(patches).all()):
            raise ValueError("patches must be a nonempty finite matrix matching the centers")
        distances = (np.sum(patches ** 2, axis=1)[:, None]
                     + np.sum(self.centers ** 2, axis=1)[None, :]
                     - 2 * patches @ self.centers.T)
        labels = np.argmin(distances, axis=1)
        residuals = np.zeros_like(self.centers)
        np.add.at(residuals, labels, patches - self.centers[labels])
        return unit(unit(residuals).ravel())

    def describe(self, patches: np.ndarray) -> np.ndarray:
        projected = np.asarray(self.aggregate(patches) @ self.projection).ravel()
        return unit(projected).astype("<f2")


def retrieve(query: np.ndarray, database: np.ndarray, k: int = 5,
             times: np.ndarray | None = None, available_at: float | None = None) -> list[int]:
    query, database = np.asarray(query, dtype=np.float32), np.asarray(database, dtype=np.float32)
    if (query.ndim != 1 or database.ndim != 2 or database.shape[1] != len(query)
            or not np.isfinite(query).all() or not np.isfinite(database).all() or k < 1):
        raise ValueError("retrieval requires finite, dimensionally compatible descriptors and k > 0")
    indices = np.arange(len(database))
    if times is not None:
        times = np.asarray(times)
        if len(times) != len(database) or available_at is None or not np.isfinite(times).all():
            raise ValueError("candidate timestamps require a delivery time")
        indices = indices[times <= available_at]
    scores = unit(database[indices]) @ unit(query)
    return indices[np.argsort(-scores, kind="stable")[:k]].tolist()


def fingerprint(sender: str, msg_id: str, t_send: float, keyframe_id: int,
                t_keyframe: float, descriptor: np.ndarray) -> dict:
    descriptor = np.asarray(descriptor, dtype="<f2")
    if (descriptor.shape != (512,) or not np.isfinite(descriptor).all()
            or not np.isfinite([t_send, t_keyframe]).all() or t_keyframe > t_send):
        raise ValueError("fingerprint needs 512 finite values and a keyframe no newer than the send time")
    return {"msg_id": msg_id, "from": sender, "to": "broadcast", "t_send": t_send,
            "kind": "place_fingerprint", "bytes": 1096,
            "payload": {"keyframe_id": keyframe_id, "t_keyframe": t_keyframe,
                        "desc_f16_b64": base64.b64encode(descriptor.tobytes()).decode("ascii")}}


class DinoValueExtractor:
    def __init__(self, model: str = "dinov2_vitl14", device: str | None = None,
                 model_cache: Path | str | None = None):
        if model not in ("dinov2_vitl14", "dinov2_vitb14"):
            raise ValueError("supported backbones are DINOv2 ViT-L/14 and ViT-B/14")
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable in this Python process")
        if device == "cpu":
            warnings.warn("MPS is unavailable or CPU was selected; using CPU. MPS acceptance is untested.", stacklevel=2)
        if model_cache is not None:
            torch.hub.set_dir(str(Path(model_cache).resolve()))
        self.device, self.name = device, model
        self.layer = 19 if model == "dinov2_vitl14" else 9
        self.model = torch.hub.load(f"facebookresearch/dinov2:{DINO_REVISION}", model,
                                    trust_repo=True, skip_validation=True).eval().to(device)
        self.mean = torch.tensor([.485, .456, .406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([.229, .224, .225], device=device).view(1, 3, 1, 1)

    @torch.inference_mode()
    def __call__(self, image: np.ndarray | Path | str) -> np.ndarray:
        if isinstance(image, (Path, str)):
            bgr = cv2.imread(str(image))
            if bgr is None:
                raise ValueError(f"cannot read RGB image: {image}")
            image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("expected an RGB uint8 image")
        resized = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized.copy()).permute(2, 0, 1).unsqueeze(0).to(self.device).float() / 255
        tokens = self.model.prepare_tokens_with_masks((tensor - self.mean) / self.std)
        for block in self.model.blocks[:self.layer]:
            tokens = block(tokens)
        # Exact value facet at the selected block's QKV projection. Nothing
        # downstream of this projection contributes, so later blocks are skipped.
        block = self.model.blocks[self.layer]
        qkv = block.attn.qkv(block.norm1(tokens))
        dim = qkv.shape[-1] // 3
        values = torch.nn.functional.normalize(qkv[:, 1:, 2 * dim:], dim=-1)
        return values[0].cpu().numpy()

    def synchronize(self):
        if self.device == "mps":
            torch.mps.synchronize()

    def config(self) -> dict:
        return {"model": self.name, "revision": DINO_REVISION, "layer_zero_based": self.layer,
                "facet": "value", "image_width": IMAGE_SIZE[0], "image_height": IMAGE_SIZE[1],
                "device": self.device, "patch_normalization": "L2", "projection_seed": 1729,
                "projection": "sparse random, 3 signed entries per input coordinate", "clusters": 32}


def benchmark(extractor: DinoValueExtractor, encoder: VLAD, images: list[Path],
              warmup: int = 3, repeats: int = 20) -> dict:
    for i in range(warmup):
        encoder.describe(extractor(images[i % len(images)]))
    extractor.synchronize()
    samples = []
    for i in range(repeats):
        extractor.synchronize()
        start = time.perf_counter()
        descriptor = encoder.describe(extractor(images[i % len(images)]))
        extractor.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
        if descriptor.nbytes != 1024:
            raise ValueError("descriptor is not 1024 bytes")
    return {"device": extractor.device, "warmup": warmup, "repeats": repeats,
            "mean_ms": float(np.mean(samples)), "median_ms": float(np.median(samples)),
            "p95_ms": float(np.percentile(samples, 95)), "samples_ms": samples,
            "scope": "PNG read, resize, normalize, dense value facets, device transfer, VLAD and projection"}


def fit_vocabulary(extractor: DinoValueExtractor, neutral_images: list[Path], out: Path) -> VLAD:
    """Fit a fixed vocabulary from an explicit neutral image manifest, never a map."""
    if not neutral_images:
        raise ValueError("neutral vocabulary image set is empty")
    digest = hashlib.sha256(json.dumps(extractor.config(), sort_keys=True).encode())
    manifest = []
    for path in neutral_images:
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest.append({"name": path.name, "sha256": sha})
        digest.update(sha.encode())
    signature = digest.hexdigest()
    out.mkdir(parents=True, exist_ok=True)
    centers_path, metadata_path = out / "centers.npy", out / "vocabulary.json"
    if metadata_path.exists() and centers_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("signature") == signature:
            return VLAD(np.load(centers_path))
    # Sample every fourth dense patch, deterministically, from each neutral image.
    samples = []
    for i, path in enumerate(neutral_images):
        samples.append(extractor(path)[::4])
        if i % 16 == 0:
            print(f"Neutral vocabulary features {i + 1}/{len(neutral_images)}", flush=True)
    features = torch.from_numpy(np.concatenate(samples)).to(extractor.device)
    rng = np.random.default_rng(1729)
    centers = features[rng.choice(len(features), 32, replace=False).tolist()].clone()
    # Lloyd iterations on the selected device. Empty clusters retain their last center.
    for _ in range(40):
        distances = (features.square().sum(1)[:, None] + centers.square().sum(1)[None, :]
                     - 2 * features @ centers.T)
        labels = distances.argmin(1)
        updated = torch.stack([features[labels == i].mean(0) if (labels == i).any()
                               else centers[i] for i in range(32)])
        delta = (updated - centers).square().sum().item()
        centers = updated
        if delta < 1e-6:
            break
    centers_array = centers.cpu().numpy()
    np.save(centers_path, centers_array)
    metadata = {"signature": signature, "config": extractor.config(), "images": manifest,
                "fit_image_count": len(manifest), "fit_patch_count": len(features),
                "set": "COCO128 neutral images, first 128 COCO train2017 images; no TUM frames",
                "source": "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip",
                "fit": "40 Lloyd iterations maximum, seed 1729, every fourth patch"}
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return VLAD(centers_array)


def describe_map(robot_dir: Path, robot_out: Path, extractor: DinoValueExtractor, encoder: VLAD) -> dict:
    manifest = json.loads((robot_out / "keyframes.json").read_text())
    associations = read_associations(robot_dir / "associations.txt")
    descriptors = []
    for i, keyframe in enumerate(manifest):
        index = keyframe["frame_index"]
        association = associations[index]
        if abs(association.rgb_timestamp - keyframe["timestamp"]) > 1e-5:
            raise ValueError("keyframe manifest disagrees with recording")
        descriptors.append(encoder.describe(extractor(robot_dir / association.rgb_path)))
        if i % 25 == 0:
            print(f"{robot_dir.name} descriptors {i + 1}/{len(manifest)}", flush=True)
    array = np.array(descriptors, dtype="<f2")
    np.save(robot_out / "descriptors.npy", array)
    metadata = {**extractor.config(), "count": len(array), "dimension": 512,
                "dtype": "float16", "bytes_per_descriptor": 1024,
                "keyframe_ids": [row["keyframe_id"] for row in manifest],
                "timestamps": [row["timestamp"] for row in manifest],
                "centers_sha256": hashlib.sha256(encoder.centers.tobytes()).hexdigest()}
    (robot_out / "descriptors.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def acceptance_gate(recall: dict, timing: dict) -> bool:
    return bool(recall["eligible_queries"] > 0 and recall.get("truth_labels_complete", True)
                and recall["recall_at_5"] >= .8
                and timing["device"] == "mps" and timing["mean_ms"] < 150)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--neutral-images", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "mps"])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    neutral = sorted(args.neutral_images.glob("*.jpg"))
    attempts = []
    for model in ("dinov2_vitl14", "dinov2_vitb14"):
        extractor = DinoValueExtractor(model, args.device, args.assets / "torch_hub")
        encoder = fit_vocabulary(extractor, neutral, args.assets / model)
        robot_a = args.recording / "robot_A"
        manifest = json.loads((args.out / "robot_A/keyframes.json").read_text())
        images = [robot_a / row["rgb"] for row in manifest[:20]]
        timing = benchmark(extractor, encoder, images)
        attempts.append({"config": extractor.config(), "timing": timing})
        (args.out / "c8_timing.json").write_text(json.dumps(attempts, indent=2) + "\n")
        print(json.dumps(attempts[-1], indent=2), flush=True)
        if timing["mean_ms"] < 150 or model == "dinov2_vitb14":
            break
        print("ViT-L/14 exceeded 150 ms; selecting ViT-B/14. Device is recorded in the timing evidence.", flush=True)
        del extractor
    metadata = json.loads((args.recording / "meta.json").read_text())
    for robot in metadata["robots"]:
        describe_map(args.recording / robot, args.out / robot, extractor, encoder)
    print("Descriptor extraction finished. Retrieval acceptance must be evaluated separately.", flush=True)


if __name__ == "__main__":
    main()
