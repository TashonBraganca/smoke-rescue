#!/usr/bin/env python3
"""Convert the TUM fr3 long-office sequence into a Smoke Rescue recording."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np


RUN_ID = "tum_fr3_long_office_2halves"
ROBOT_NAMES = ("robot_A", "robot_B")
MAX_ASSOCIATION_DELTA_S = 0.02


@dataclass(frozen=True)
class TimestampedPath:
    timestamp: float
    path: Path


@dataclass(frozen=True)
class AssociatedPair:
    rgb: TimestampedPath
    depth: TimestampedPath


@dataclass(frozen=True)
class Summary:
    frames_per_robot: dict[str, int]
    time_spans: dict[str, tuple[float, float]]
    overlap_seconds: float
    gt_rows_per_robot: dict[str, int]
    median_depth_m: float

    def format(self) -> str:
        frame_text = ", ".join(
            f"{robot}={count}" for robot, count in self.frames_per_robot.items()
        )
        span_text = ", ".join(
            f"{robot}={start:.6f}..{end:.6f}s"
            for robot, (start, end) in self.time_spans.items()
        )
        gt_text = ", ".join(
            f"{robot}={count}" for robot, count in self.gt_rows_per_robot.items()
        )
        return "\n".join(
            [
                f"Frames: {frame_text}",
                f"Time spans: {span_text}",
                f"Overlap: {self.overlap_seconds:.6f}s",
                f"Ground truth rows: {gt_text}",
                f"Median valid depth: {self.median_depth_m:.6f}m",
            ]
        )


def read_tum_index(path: Path) -> list[TimestampedPath]:
    entries: list[TimestampedPath] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) != 2:
            raise ValueError(f"{path}:{line_number}: expected timestamp and filename")
        entries.append(TimestampedPath(float(fields[0]), Path(fields[1])))
    if not entries:
        raise ValueError(f"{path} contains no data rows")
    return entries


def associate_frames(
    rgb_entries: Sequence[TimestampedPath],
    depth_entries: Sequence[TimestampedPath],
    max_delta_s: float = MAX_ASSOCIATION_DELTA_S,
) -> list[AssociatedPair]:
    candidates = sorted(
        (
            abs(rgb.timestamp - depth.timestamp),
            rgb_index,
            depth_index,
        )
        for rgb_index, rgb in enumerate(rgb_entries)
        for depth_index, depth in enumerate(depth_entries)
        if abs(rgb.timestamp - depth.timestamp) <= max_delta_s
    )
    used_rgb: set[int] = set()
    used_depth: set[int] = set()
    matches: list[AssociatedPair] = []
    for _, rgb_index, depth_index in candidates:
        if rgb_index in used_rgb or depth_index in used_depth:
            continue
        used_rgb.add(rgb_index)
        used_depth.add(depth_index)
        matches.append(
            AssociatedPair(rgb_entries[rgb_index], depth_entries[depth_index])
        )
    matches.sort(key=lambda pair: pair.rgb.timestamp)
    if not matches:
        raise ValueError("no RGB and depth frames matched within 0.02 seconds")
    return matches


def read_groundtruth(path: Path) -> list[tuple[float, tuple[float, ...]]]:
    rows: list[tuple[float, tuple[float, ...]]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) != 8:
            raise ValueError(f"{path}:{line_number}: expected eight TUM trajectory fields")
        values = tuple(float(field) for field in fields)
        rows.append((values[0], values[1:]))
    if not rows:
        raise ValueError(f"{path} contains no ground-truth rows")
    return rows


def camera_yaml() -> str:
    return """%YAML:1.0
---
File.version: "1.0"
Camera.type: "PinHole"

Camera1.fx: 535.4
Camera1.fy: 539.2
Camera1.cx: 320.1
Camera1.cy: 247.6
Camera1.k1: 0.0
Camera1.k2: 0.0
Camera1.p1: 0.0
Camera1.p2: 0.0
Camera1.k3: 0.0
Camera.width: 640
Camera.height: 480
Camera.fps: 30
Camera.RGB: 1

Stereo.ThDepth: 40.0
Stereo.b: 0.07732
RGBD.DepthMapFactor: 1000.0

ORBextractor.nFeatures: 1250
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7

Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500.0
"""


def write_index(path: Path, title: str, rows: Iterable[tuple[float, str]]) -> None:
    lines = [
        f"# {title}",
        "# Smoke Rescue shared simulation clock",
        "# timestamp filename",
    ]
    lines.extend(f"{timestamp:.6f} {filename}" for timestamp, filename in rows)
    path.write_text("\n".join(lines) + "\n")


def write_associations(path: Path, pairs: Sequence[AssociatedPair], origin: float) -> None:
    lines = [
        "# associated RGB-D frames",
        "# Smoke Rescue shared simulation clock",
        "# t_rgb rgb_path t_depth depth_path",
    ]
    for pair in pairs:
        rgb_time = pair.rgb.timestamp - origin
        depth_time = pair.depth.timestamp - origin
        lines.append(
            f"{rgb_time:.6f} rgb/{rgb_time:.6f}.png "
            f"{depth_time:.6f} depth/{depth_time:.6f}.png"
        )
    path.write_text("\n".join(lines) + "\n")


def write_sensor_stub(path: Path, sensor: str) -> None:
    path.write_text(
        f"# {sensor} data\n"
        "# TUM RGB-D fr3/long_office_household does not provide this sensor\n"
        "# no data rows\n"
    )


def convert_robot(
    source: Path,
    robot_folder: Path,
    pairs: Sequence[AssociatedPair],
    origin: float,
) -> None:
    rgb_folder = robot_folder / "rgb"
    depth_folder = robot_folder / "depth"
    rgb_folder.mkdir(parents=True)
    depth_folder.mkdir()

    rgb_index_rows: list[tuple[float, str]] = []
    depth_index_rows: list[tuple[float, str]] = []
    seen_rgb_names: set[str] = set()
    seen_depth_names: set[str] = set()

    for pair in pairs:
        rgb_time = pair.rgb.timestamp - origin
        depth_time = pair.depth.timestamp - origin
        rgb_name = f"{rgb_time:.6f}.png"
        depth_name = f"{depth_time:.6f}.png"
        if rgb_name in seen_rgb_names or depth_name in seen_depth_names:
            raise ValueError("six-decimal timestamp rounding produced a duplicate filename")
        seen_rgb_names.add(rgb_name)
        seen_depth_names.add(depth_name)

        source_rgb = source / pair.rgb.path
        rgb = cv2.imread(str(source_rgb), cv2.IMREAD_UNCHANGED)
        if rgb is None or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"RGB image is not 8-bit three-channel PNG: {source_rgb}")
        if not cv2.imwrite(
            str(rgb_folder / rgb_name),
            rgb,
            [cv2.IMWRITE_PNG_COMPRESSION, 3],
        ):
            raise OSError(f"failed to write RGB image {rgb_folder / rgb_name}")

        source_depth = source / pair.depth.path
        raw_depth = cv2.imread(str(source_depth), cv2.IMREAD_UNCHANGED)
        if raw_depth is None or raw_depth.dtype != np.uint16 or raw_depth.ndim != 2:
            raise ValueError(f"depth image is not uint16 single-channel PNG: {source_depth}")
        depth_mm = np.rint(raw_depth.astype(np.float64) / 5.0).astype(np.uint16)
        depth_mm[raw_depth == 0] = 0
        destination_depth = depth_folder / depth_name
        if not cv2.imwrite(
            str(destination_depth),
            depth_mm,
            [cv2.IMWRITE_PNG_COMPRESSION, 6],
        ):
            raise OSError(f"failed to write depth image {destination_depth}")
        written_depth = cv2.imread(str(destination_depth), cv2.IMREAD_UNCHANGED)
        if written_depth is None or written_depth.dtype != np.uint16:
            raise ValueError(f"written depth image is not uint16: {destination_depth}")

        rgb_index_rows.append((rgb_time, f"rgb/{rgb_name}"))
        depth_index_rows.append((depth_time, f"depth/{depth_name}"))

    write_index(robot_folder / "rgb.txt", "color images", rgb_index_rows)
    write_index(robot_folder / "depth.txt", "depth maps in millimetres", depth_index_rows)
    write_associations(robot_folder / "associations.txt", pairs, origin)
    write_sensor_stub(robot_folder / "imu.txt", "IMU")
    write_sensor_stub(robot_folder / "odom.txt", "odometry")
    (robot_folder / "camera.yaml").write_text(camera_yaml())


def write_groundtruth(
    path: Path,
    groundtruth: Sequence[tuple[float, tuple[float, ...]]],
    origin: float,
    start: float,
    end: float,
) -> int:
    selected = [row for row in groundtruth if start <= row[0] <= end]
    lines = [
        "# ground truth trajectory",
        "# t tx ty tz qx qy qz qw",
        "# camera optical pose in the source TUM world frame",
    ]
    for timestamp, pose in selected:
        pose_text = " ".join(f"{value:.9f}" for value in pose)
        lines.append(f"{timestamp - origin:.6f} {pose_text}")
    path.write_text("\n".join(lines) + "\n")
    return len(selected)


def tick_times(start: float, end: float) -> Iterable[tuple[int, float]]:
    tick = 0
    while True:
        timestamp = start + tick * 2.0
        if timestamp > end + 1e-9:
            return
        yield tick, timestamp
        tick += 1


def write_radio(path: Path, spans: dict[str, tuple[float, float]]) -> None:
    rows: list[tuple[float, str]] = []
    for sender, receiver in (("robot_A", "robot_B"), ("robot_B", "robot_A")):
        sender_start, sender_end = spans[sender]
        receiver_start, receiver_end = spans[receiver]
        for tick, timestamp in tick_times(sender_start, sender_end):
            if receiver_start <= timestamp <= receiver_end:
                sender_id = sender.removeprefix("robot_")
                line = (
                    f"{timestamp:.6f} {timestamp + 0.050:.6f} {sender} {receiver} "
                    f"{sender_id}-{tick:06d} 1 ok"
                )
                rows.append((timestamp, line))
    rows.sort(key=lambda item: (item[0], item[1]))
    headers = [
        "# synthetic radio delivery log",
        "# t_send t_deliver from to msg_id delivered reason",
        "# always-connected link while both stand-in robots exist",
    ]
    path.write_text("\n".join(headers + [row[1] for row in rows]) + "\n")


def data_rows(path: Path) -> list[list[str]]:
    return [
        line.split()
        for line in path.read_text().splitlines()
        if line and not line.startswith("#")
    ]


def evenly_spaced_sample(paths: Sequence[Path], count: int) -> list[Path]:
    if len(paths) <= count:
        return list(paths)
    indices = np.linspace(0, len(paths) - 1, count, dtype=int)
    return [paths[int(index)] for index in indices]


def validate_recording(output: Path) -> Summary:
    meta = json.loads((output / "meta.json").read_text())
    robots = tuple(meta["robots"])
    frames_per_robot: dict[str, int] = {}
    spans: dict[str, tuple[float, float]] = {}
    gt_counts: dict[str, int] = {}
    all_depth_paths: list[Path] = []

    for robot in robots:
        robot_folder = output / robot
        rgb_rows = data_rows(robot_folder / "rgb.txt")
        depth_rows = data_rows(robot_folder / "depth.txt")
        association_rows = data_rows(robot_folder / "associations.txt")
        rgb_files = sorted((robot_folder / "rgb").glob("*.png"))
        depth_files = sorted((robot_folder / "depth").glob("*.png"))
        counts = {
            len(rgb_rows),
            len(depth_rows),
            len(association_rows),
            len(rgb_files),
            len(depth_files),
        }
        if len(counts) != 1:
            raise ValueError(
                f"{robot} frame count mismatch: rgb.txt={len(rgb_rows)}, "
                f"depth.txt={len(depth_rows)}, associations.txt={len(association_rows)}, "
                f"rgb files={len(rgb_files)}, depth files={len(depth_files)}"
            )
        if not association_rows:
            raise ValueError(f"{robot} has no associated frames")

        for row in association_rows:
            if len(row) != 4:
                raise ValueError(f"{robot} has a malformed association row")
            if not (robot_folder / row[1]).is_file() or not (
                robot_folder / row[3]
            ).is_file():
                raise ValueError(f"{robot} association references a missing image")

        for depth_path in depth_files:
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if depth is None or depth.dtype != np.uint16 or depth.ndim != 2:
                raise ValueError(f"depth PNG is not uint16: {depth_path}")

        timestamps = [float(row[0]) for row in association_rows]
        frames_per_robot[robot] = len(association_rows)
        spans[robot] = (timestamps[0], timestamps[-1])
        all_depth_paths.extend(depth_files)

        gt_path = output / "oracle" / f"{robot}_groundtruth.txt"
        gt_count = len(data_rows(gt_path))
        if gt_count == 0:
            raise ValueError(f"{robot} ground-truth row count is zero")
        gt_counts[robot] = gt_count

    overlap = min(span[1] for span in spans.values()) - max(
        span[0] for span in spans.values()
    )
    if overlap <= 0:
        raise ValueError(f"robot time spans do not overlap: {overlap:.6f} seconds")

    valid_depth_values: list[np.ndarray] = []
    for depth_path in evenly_spaced_sample(all_depth_paths, 20):
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        valid = depth[depth > 0]
        if valid.size:
            valid_depth_values.append(valid.astype(np.float32) / 1000.0)
    if not valid_depth_values:
        raise ValueError("20-frame depth sample has no valid pixels")
    median_depth_m = float(np.median(np.concatenate(valid_depth_values)))
    if not 0.3 <= median_depth_m <= 5.0:
        raise ValueError(
            f"median valid depth {median_depth_m:.6f}m is outside 0.3 to 5.0m"
        )

    return Summary(frames_per_robot, spans, overlap, gt_counts, median_depth_m)


def convert_tum_sequence(source: Path, output: Path) -> Summary:
    source = source.resolve()
    output = output.resolve()
    required = ("rgb.txt", "depth.txt", "groundtruth.txt", "rgb", "depth")
    missing = [name for name in required if not (source / name).exists()]
    if missing:
        raise FileNotFoundError(f"source sequence is missing: {', '.join(missing)}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output folder is not empty: {output}")

    rgb_entries = read_tum_index(source / "rgb.txt")
    depth_entries = read_tum_index(source / "depth.txt")
    pairs = associate_frames(rgb_entries, depth_entries)
    if len(pairs) < 2:
        raise ValueError("at least two associated frames are required for an overlap")
    groundtruth = read_groundtruth(source / "groundtruth.txt")
    origin = pairs[0].rgb.timestamp
    frames_per_half = math.ceil(len(pairs) * 0.60)
    robot_pairs = {
        "robot_A": pairs[:frames_per_half],
        "robot_B": pairs[-frames_per_half:],
    }

    output.mkdir(parents=True, exist_ok=True)
    truth_folder = output / "oracle"
    truth_folder.mkdir()
    absolute_spans: dict[str, tuple[float, float]] = {}
    rebased_spans: dict[str, tuple[float, float]] = {}

    for robot, selected_pairs in robot_pairs.items():
        convert_robot(source, output / robot, selected_pairs, origin)
        absolute_span = (
            selected_pairs[0].rgb.timestamp,
            selected_pairs[-1].rgb.timestamp,
        )
        absolute_spans[robot] = absolute_span
        rebased_spans[robot] = (
            absolute_span[0] - origin,
            absolute_span[1] - origin,
        )
        write_groundtruth(
            truth_folder / f"{robot}_groundtruth.txt",
            groundtruth,
            origin,
            *absolute_span,
        )

    write_radio(output / "radio.txt", rebased_spans)
    meta = {
        "run_id": RUN_ID,
        "scenario": "tum_standin",
        "seed": None,
        "unity_version": None,
        "sim_rate_hz": 30,
        "robots": list(ROBOT_NAMES),
        "smoke_level": 0.0,
        "recorded_offline": True,
        "source": "TUM RGB-D fr3/long_office_household",
        "imu": "not available in the source TUM RGB-D sequence",
        "odom": "not available in the source TUM RGB-D sequence",
        "radio": "synthetic always-connected link for the TUM stand-in",
    }
    (output / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    return validate_recording(output)


def copy_png_losslessly(source: Path, destination: Path, expected_dtype: np.dtype) -> None:
    image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    if image is None or image.dtype != expected_dtype:
        raise ValueError(f"unexpected PNG dtype in fixture source: {source}")
    parameters = [cv2.IMWRITE_PNG_COMPRESSION, 9]
    if expected_dtype == np.dtype(np.uint8):
        parameters.extend(
            [cv2.IMWRITE_PNG_STRATEGY, cv2.IMWRITE_PNG_STRATEGY_RLE]
        )
    if not cv2.imwrite(str(destination), image, parameters):
        raise OSError(f"failed to write fixture image: {destination}")


def select_truth_rows(path: Path, start: float, end: float) -> list[str]:
    selected: list[str] = []
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        timestamp = float(line.split()[0])
        if start <= timestamp <= end:
            selected.append(line)
    return selected


def create_tiny_fixture(
    recording: Path,
    fixture: Path,
    frame_limit: int = 15,
) -> None:
    """Copy the first frames per robot into a small, self-contained recording."""
    if frame_limit <= 0:
        raise ValueError("frame_limit must be positive")
    if fixture.exists() and any(fixture.iterdir()):
        raise FileExistsError(f"fixture folder is not empty: {fixture}")

    source_meta = json.loads((recording / "meta.json").read_text())
    robots = tuple(source_meta["robots"])
    fixture.mkdir(parents=True, exist_ok=True)
    fixture_truth = fixture / "oracle"
    fixture_truth.mkdir()
    spans: dict[str, tuple[float, float]] = {}

    for robot in robots:
        source_robot = recording / robot
        fixture_robot = fixture / robot
        (fixture_robot / "rgb").mkdir(parents=True)
        (fixture_robot / "depth").mkdir()
        associations = data_rows(source_robot / "associations.txt")[:frame_limit]
        if len(associations) < frame_limit:
            raise ValueError(
                f"{robot} has {len(associations)} frames, fewer than {frame_limit}"
            )

        association_lines: list[str] = []
        rgb_rows: list[tuple[float, str]] = []
        depth_rows: list[tuple[float, str]] = []
        for row in associations:
            rgb_time, rgb_relative, depth_time, depth_relative = row
            copy_png_losslessly(
                source_robot / rgb_relative,
                fixture_robot / rgb_relative,
                np.dtype(np.uint8),
            )
            copy_png_losslessly(
                source_robot / depth_relative,
                fixture_robot / depth_relative,
                np.dtype(np.uint16),
            )
            association_lines.append(" ".join(row))
            rgb_rows.append((float(rgb_time), rgb_relative))
            depth_rows.append((float(depth_time), depth_relative))

        write_index(fixture_robot / "rgb.txt", "color images", rgb_rows)
        write_index(
            fixture_robot / "depth.txt",
            "depth maps in millimetres",
            depth_rows,
        )
        association_headers = [
            "# associated RGB-D frames",
            "# Smoke Rescue shared simulation clock",
            "# t_rgb rgb_path t_depth depth_path",
        ]
        (fixture_robot / "associations.txt").write_text(
            "\n".join(association_headers + association_lines) + "\n"
        )
        for filename in ("camera.yaml", "imu.txt", "odom.txt"):
            shutil.copyfile(source_robot / filename, fixture_robot / filename)

        start = float(associations[0][0])
        end = float(associations[-1][0])
        spans[robot] = (start, end)
        source_truth = recording / "oracle" / f"{robot}_groundtruth.txt"
        truth_rows = select_truth_rows(source_truth, start, end)
        if not truth_rows:
            raise ValueError(f"fixture selection has no ground truth for {robot}")
        truth_headers = [
            "# ground truth trajectory",
            "# t tx ty tz qx qy qz qw",
            "# camera optical pose in the source TUM world frame",
        ]
        (fixture_truth / f"{robot}_groundtruth.txt").write_text(
            "\n".join(truth_headers + truth_rows) + "\n"
        )

    for optional_name in ("survivors.json", "floorplan_used.json"):
        source_optional = recording / "oracle" / optional_name
        if source_optional.exists():
            shutil.copyfile(source_optional, fixture_truth / optional_name)

    radio_headers = [
        "# synthetic radio delivery log",
        "# t_send t_deliver from to msg_id delivered reason",
        "# rows retained only when both fixture robots exist",
    ]
    radio_lines: list[str] = []
    for row in data_rows(recording / "radio.txt"):
        timestamp = float(row[0])
        sender = row[2]
        receiver = row[3]
        if spans[sender][0] <= timestamp <= spans[sender][1] and spans[receiver][
            0
        ] <= timestamp <= spans[receiver][1]:
            radio_lines.append(" ".join(row))
    (fixture / "radio.txt").write_text(
        "\n".join(radio_headers + radio_lines) + "\n"
    )

    fixture_meta = dict(source_meta)
    fixture_meta["run_id"] = "tiny_2robots"
    fixture_meta["fixture_of"] = source_meta["run_id"]
    (fixture / "meta.json").write_text(json.dumps(fixture_meta, indent=2) + "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        nargs="?",
        type=Path,
        default=repo_root / "data/public/rgbd_dataset_freiburg3_long_office_household",
    )
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=repo_root / "data/recordings" / RUN_ID,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = convert_tum_sequence(args.source, args.output)
    except Exception as error:
        print(f"tum_to_recording: {error}", file=sys.stderr)
        return 1
    print(summary.format())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
