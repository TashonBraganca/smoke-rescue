"""Ground-truth readers used only by evaluation and viewing code."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Trajectory:
    timestamps: np.ndarray
    positions: np.ndarray
    quaternions_xyzw: np.ndarray


def truth_folder(recording: Path | str) -> Path:
    return Path(recording) / "oracle"


def load_trajectory(recording: Path | str, robot: str) -> Trajectory:
    path = truth_folder(recording) / f"{robot}_groundtruth.txt"
    rows = np.loadtxt(path, comments="#", dtype=np.float64, ndmin=2)
    if rows.shape[0] == 0 or rows.shape[1] != 8:
        raise ValueError(f"invalid TUM ground-truth trajectory: {path}")
    return Trajectory(
        timestamps=rows[:, 0],
        positions=rows[:, 1:4],
        quaternions_xyzw=rows[:, 4:8],
    )


def load_optional_json(recording: Path | str, filename: str) -> object | None:
    path = truth_folder(recording) / filename
    if not path.exists():
        return None
    return json.loads(path.read_text())


def load_survivors(recording: Path | str) -> object | None:
    return load_optional_json(recording, "survivors.json")


def load_floorplan(recording: Path | str) -> object | None:
    return load_optional_json(recording, "floorplan_used.json")


def load_collisions(recording: Path | str) -> dict | None:
    """Aggregate seconds stopped against geometry, not individual contacts."""
    return load_optional_json(recording, "collisions.json")


def load_house_points(recording: Path | str) -> np.ndarray:
    """Read the writer's XYZ PLY without loading a rendering library."""
    path = truth_folder(recording) / "house_truth.ply"
    with path.open("rb") as stream:
        header = []
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"unterminated PLY header: {path}")
            header.append(line.decode("ascii").strip())
            if header[-1] == "end_header":
                break
        count = int(next(s.split()[-1] for s in header if s.startswith("element vertex ")))
        properties = [s for s in header if s.startswith("property ")]
        if properties != ["property float x", "property float y", "property float z"]:
            raise ValueError("expected XYZ float PLY from OracleWriter")
        if "format binary_little_endian 1.0" in header:
            points = np.frombuffer(stream.read(count * 12), dtype="<f4").reshape(-1, 3)
        elif "format ascii 1.0" in header:
            points = np.loadtxt(stream, max_rows=count, ndmin=2)
        else:
            raise ValueError("unsupported PLY encoding")
    if points.shape != (count, 3) or not np.isfinite(points).all():
        raise ValueError(f"invalid PLY points: {path}")
    return points.astype(np.float64)
