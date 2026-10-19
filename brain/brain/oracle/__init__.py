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
