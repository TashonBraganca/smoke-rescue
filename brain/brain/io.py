"""Readers for robot-side recording data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import yaml


@dataclass(frozen=True)
class Camera:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    fps: float
    depth_factor: float


@dataclass(frozen=True)
class Association:
    rgb_timestamp: float
    rgb_path: Path
    depth_timestamp: float
    depth_path: Path


@dataclass(frozen=True)
class Frame:
    rgb_timestamp: float
    depth_timestamp: float
    rgb: np.ndarray
    depth: np.ndarray
    depth_m: np.ndarray


def load_opencv_yaml(path: Path) -> dict[str, object]:
    lines = path.read_text().splitlines()
    if lines and lines[0].startswith("%YAML:"):
        lines = lines[1:]
    document = yaml.safe_load("\n".join(lines))
    if not isinstance(document, dict):
        raise ValueError(f"camera settings are not a YAML mapping: {path}")
    return document


def required_number(settings: dict[str, object], *keys: str) -> float:
    for key in keys:
        value = settings.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    raise ValueError(f"camera settings are missing numeric key: {' or '.join(keys)}")


def read_camera(path: Path) -> Camera:
    settings = load_opencv_yaml(path)
    return Camera(
        fx=required_number(settings, "Camera1.fx", "Camera.fx"),
        fy=required_number(settings, "Camera1.fy", "Camera.fy"),
        cx=required_number(settings, "Camera1.cx", "Camera.cx"),
        cy=required_number(settings, "Camera1.cy", "Camera.cy"),
        width=int(required_number(settings, "Camera.width")),
        height=int(required_number(settings, "Camera.height")),
        fps=required_number(settings, "Camera.fps"),
        depth_factor=required_number(
            settings,
            "RGBD.DepthMapFactor",
            "DepthMapFactor",
        ),
    )


def read_associations(path: Path) -> list[Association]:
    associations: list[Association] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) != 4:
            raise ValueError(f"{path}:{line_number}: expected four fields")
        rgb_path = Path(fields[1])
        depth_path = Path(fields[3])
        if rgb_path.is_absolute() or depth_path.is_absolute():
            raise ValueError(f"{path}:{line_number}: image paths must be relative")
        if ".." in rgb_path.parts or ".." in depth_path.parts:
            raise ValueError(f"{path}:{line_number}: image paths leave the robot folder")
        associations.append(
            Association(
                rgb_timestamp=float(fields[0]),
                rgb_path=rgb_path,
                depth_timestamp=float(fields[2]),
                depth_path=depth_path,
            )
        )
    if not associations:
        raise ValueError(f"{path} contains no associated frames")
    return associations


class RecordingReader:
    def __init__(self, robot_folder: Path | str) -> None:
        self.robot_folder = Path(robot_folder)
        self.camera = read_camera(self.robot_folder / "camera.yaml")
        self.associations = read_associations(
            self.robot_folder / "associations.txt"
        )

    def __len__(self) -> int:
        return len(self.associations)

    def read_frame(self, index: int) -> Frame:
        association = self.associations[index]
        rgb_path = self.robot_folder / association.rgb_path
        depth_path = self.robot_folder / association.depth_path
        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if bgr is None:
            raise ValueError(f"failed to read RGB image: {rgb_path}")
        if depth is None:
            raise ValueError(f"failed to read depth image: {depth_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.dtype != np.uint8 or rgb.shape != (
            self.camera.height,
            self.camera.width,
            3,
        ):
            raise ValueError(f"unexpected RGB image shape or dtype: {rgb_path}")
        if depth.dtype != np.uint16 or depth.shape != (
            self.camera.height,
            self.camera.width,
        ):
            raise ValueError(f"unexpected depth image shape or dtype: {depth_path}")
        depth_m = depth.astype(np.float32) / self.camera.depth_factor
        return Frame(
            rgb_timestamp=association.rgb_timestamp,
            depth_timestamp=association.depth_timestamp,
            rgb=rgb,
            depth=depth,
            depth_m=depth_m,
        )

    def frames(self) -> Iterator[Frame]:
        for index in range(len(self)):
            yield self.read_frame(index)
