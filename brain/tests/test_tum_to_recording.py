from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest


REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.tum_to_recording import (
    create_tiny_fixture,
    convert_tum_sequence,
    validate_recording,
)


def write_synthetic_tum(root: Path, frame_count: int = 10) -> None:
    (root / "rgb").mkdir(parents=True)
    (root / "depth").mkdir()
    base = 1_341_847_980.0
    rgb_rows = ["# color images", "# synthetic sequence", "# timestamp filename"]
    depth_rows = ["# depth maps", "# synthetic sequence", "# timestamp filename"]
    groundtruth_rows = [
        "# ground truth trajectory",
        "# synthetic sequence",
        "# timestamp tx ty tz qx qy qz qw",
    ]

    for index in range(frame_count):
        rgb_time = base + index / 30.0
        depth_time = rgb_time + 0.005
        rgb_name = f"rgb/{rgb_time:.6f}.png"
        depth_name = f"depth/{depth_time:.6f}.png"
        rgb = np.full((480, 640, 3), index, dtype=np.uint8)
        raw_depth = np.full((480, 640), 5_000 + index * 5, dtype=np.uint16)
        assert cv2.imwrite(str(root / rgb_name), rgb)
        assert cv2.imwrite(str(root / depth_name), raw_depth)
        rgb_rows.append(f"{rgb_time:.6f} {rgb_name}")
        depth_rows.append(f"{depth_time:.6f} {depth_name}")
        groundtruth_rows.append(
            f"{rgb_time:.6f} {index / 100:.3f} 0 0 0 0 0 1"
        )

    (root / "rgb.txt").write_text("\n".join(rgb_rows) + "\n")
    (root / "depth.txt").write_text("\n".join(depth_rows) + "\n")
    (root / "groundtruth.txt").write_text("\n".join(groundtruth_rows) + "\n")
    (root / "accelerometer.txt").write_text(
        "# accelerometer data is not available\n# synthetic sequence\n# empty\n"
    )


def data_rows(path: Path) -> list[str]:
    return [line for line in path.read_text().splitlines() if line and not line.startswith("#")]


def test_converter_writes_two_overlapping_robot_recordings(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "recording"
    write_synthetic_tum(source)

    summary = convert_tum_sequence(source, output)

    assert summary.frames_per_robot == {"robot_A": 6, "robot_B": 6}
    assert summary.overlap_seconds > 0
    assert summary.gt_rows_per_robot == {"robot_A": 6, "robot_B": 6}
    assert summary.median_depth_m == pytest.approx(1.0045, abs=0.001)

    robot_a_associations = data_rows(output / "robot_A" / "associations.txt")
    robot_b_associations = data_rows(output / "robot_B" / "associations.txt")
    assert robot_a_associations[0].split() == [
        "0.000000",
        "rgb/0.000000.png",
        "0.005000",
        "depth/0.005000.png",
    ]
    assert robot_b_associations[0].split()[0] == "0.133333"

    converted_depth = cv2.imread(
        str(output / "robot_A" / "depth" / "0.005000.png"),
        cv2.IMREAD_UNCHANGED,
    )
    assert converted_depth.dtype == np.uint16
    assert int(converted_depth[0, 0]) == 1_000

    camera_yaml = (output / "robot_A" / "camera.yaml").read_text()
    assert camera_yaml.startswith("%YAML:1.0\n")
    assert 'File.version: "1.0"' in camera_yaml
    assert "Camera1.fx: 535.4" in camera_yaml
    assert "Camera.RGB: 1" in camera_yaml
    assert "RGBD.DepthMapFactor: 1000.0" in camera_yaml
    assert "ORBextractor.nFeatures: 1250" in camera_yaml

    meta = json.loads((output / "meta.json").read_text())
    assert meta["run_id"] == "tum_fr3_long_office_2halves"
    assert meta["scenario"] == "tum_standin"
    assert meta["source"] == "TUM RGB-D fr3/long_office_household"
    assert meta["radio"] == "synthetic always-connected link for the TUM stand-in"

    radio_rows = data_rows(output / "radio.txt")
    assert radio_rows == [
        "0.133333 0.183333 robot_B robot_A B-000000 1 ok"
    ]
    assert data_rows(output / "robot_A" / "imu.txt") == []
    assert data_rows(output / "robot_A" / "odom.txt") == []


def test_validation_rejects_non_uint16_depth(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "recording"
    write_synthetic_tum(source)
    convert_tum_sequence(source, output)
    depth_path = next((output / "robot_A" / "depth").glob("*.png"))
    assert cv2.imwrite(
        str(depth_path),
        np.ones((480, 640), dtype=np.uint8),
    )

    with pytest.raises(ValueError, match="uint16"):
        validate_recording(output)


def test_validation_rejects_frame_count_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "recording"
    write_synthetic_tum(source)
    convert_tum_sequence(source, output)
    next((output / "robot_A" / "rgb").glob("*.png")).unlink()

    with pytest.raises(ValueError, match="frame count mismatch"):
        validate_recording(output)


def test_validation_rejects_depth_median_outside_range(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "recording"
    write_synthetic_tum(source)
    convert_tum_sequence(source, output)
    for depth_path in output.glob("robot_*/depth/*.png"):
        assert cv2.imwrite(
            str(depth_path),
            np.full((480, 640), 6_000, dtype=np.uint16),
        )

    with pytest.raises(ValueError, match="outside 0.3 to 5.0m"):
        validate_recording(output)


def test_validation_rejects_zero_groundtruth_rows(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "recording"
    write_synthetic_tum(source)
    convert_tum_sequence(source, output)
    (output / "oracle" / "robot_A_groundtruth.txt").write_text(
        "# no data rows\n"
    )

    with pytest.raises(ValueError, match="row count is zero"):
        validate_recording(output)


def test_validation_rejects_non_overlapping_robot_spans(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "recording"
    write_synthetic_tum(source)
    convert_tum_sequence(source, output)
    associations = output / "robot_B" / "associations.txt"
    shifted_lines: list[str] = []
    for line in associations.read_text().splitlines():
        if not line or line.startswith("#"):
            shifted_lines.append(line)
            continue
        fields = line.split()
        fields[0] = f"{float(fields[0]) + 100.0:.6f}"
        fields[2] = f"{float(fields[2]) + 100.0:.6f}"
        shifted_lines.append(" ".join(fields))
    associations.write_text("\n".join(shifted_lines) + "\n")

    with pytest.raises(ValueError, match="do not overlap"):
        validate_recording(output)


def test_tiny_fixture_keeps_first_frames_for_each_robot(tmp_path: Path) -> None:
    source = tmp_path / "source"
    recording = tmp_path / "recording"
    fixture = tmp_path / "fixture"
    write_synthetic_tum(source)
    convert_tum_sequence(source, recording)

    create_tiny_fixture(recording, fixture, frame_limit=3)

    for robot in ("robot_A", "robot_B"):
        assert len(data_rows(fixture / robot / "associations.txt")) == 3
        assert len(list((fixture / robot / "rgb").glob("*.png"))) == 3
        assert len(list((fixture / robot / "depth").glob("*.png"))) == 3
        assert data_rows(fixture / "oracle" / f"{robot}_groundtruth.txt")
    fixture_meta = json.loads((fixture / "meta.json").read_text())
    assert fixture_meta["run_id"] == "tiny_2robots"
    assert fixture_meta["fixture_of"] == "tum_fr3_long_office_2halves"
