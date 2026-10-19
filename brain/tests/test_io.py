from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from brain.io import RecordingReader


def write_robot_fixture(root: Path) -> None:
    (root / "rgb").mkdir(parents=True)
    (root / "depth").mkdir()
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    depth = np.full((480, 640), 1_250, dtype=np.uint16)
    assert cv2.imwrite(str(root / "rgb" / "0.000000.png"), rgb)
    assert cv2.imwrite(str(root / "depth" / "0.000000.png"), depth)
    (root / "associations.txt").write_text(
        "# associated RGB-D frames\n"
        "# timestamp filename timestamp filename\n"
        "# shared simulation clock\n"
        "0.000000 rgb/0.000000.png 0.000000 depth/0.000000.png\n"
    )
    (root / "camera.yaml").write_text(
        "%YAML:1.0\n"
        "---\n"
        'File.version: "1.0"\n'
        'Camera.type: "PinHole"\n'
        "Camera1.fx: 535.4\n"
        "Camera1.fy: 539.2\n"
        "Camera1.cx: 320.1\n"
        "Camera1.cy: 247.6\n"
        "Camera.width: 640\n"
        "Camera.height: 480\n"
        "Camera.fps: 30\n"
        "RGBD.DepthMapFactor: 1000.0\n"
    )


def test_recording_reader_uses_yaml_intrinsics_and_reads_numpy_frames(
    tmp_path: Path,
) -> None:
    robot = tmp_path / "robot_A"
    write_robot_fixture(robot)

    reader = RecordingReader(robot)
    frame = reader.read_frame(0)

    assert reader.camera.fx == 535.4
    assert reader.camera.fy == 539.2
    assert reader.camera.cx == 320.1
    assert reader.camera.cy == 247.6
    assert reader.camera.width == 640
    assert reader.camera.height == 480
    assert reader.camera.fps == 30
    assert reader.camera.depth_factor == 1000.0
    assert frame.rgb.shape == (480, 640, 3)
    assert frame.rgb.dtype == np.uint8
    assert frame.depth.shape == (480, 640)
    assert frame.depth.dtype == np.uint16
    assert frame.depth_m[0, 0] == 1.25


def test_committed_fixture_has_expected_frame_shape_and_camera_values() -> None:
    fixture = Path(__file__).parents[2] / "data" / "fixtures" / "tiny_2robots"
    reader = RecordingReader(fixture / "robot_A")

    frame = reader.read_frame(0)

    assert len(reader) == 15
    assert frame.rgb.shape == (480, 640, 3)
    assert frame.rgb.dtype == np.uint8
    assert frame.depth.shape == (480, 640)
    assert frame.depth.dtype == np.uint16
    assert (reader.camera.fx, reader.camera.fy) == (535.4, 539.2)
    assert (reader.camera.cx, reader.camera.cy) == (320.1, 247.6)
    total_bytes = sum(path.stat().st_size for path in fixture.rglob("*") if path.is_file())
    assert total_bytes < 15_000_000
