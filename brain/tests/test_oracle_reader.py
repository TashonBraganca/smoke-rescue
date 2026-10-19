from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from brain.oracle import load_floorplan, load_survivors, load_trajectory


def test_oracle_loads_truth_files_and_allows_optional_json(tmp_path: Path) -> None:
    truth = tmp_path / "oracle"
    truth.mkdir()
    (truth / "robot_A_groundtruth.txt").write_text(
        "# t tx ty tz qx qy qz qw\n0.000000 1 2 3 0 0 0 1\n"
    )
    (truth / "survivors.json").write_text(json.dumps({"survivors": [{"id": "s1"}]}))

    trajectory = load_trajectory(tmp_path, "robot_A")

    assert trajectory.timestamps.tolist() == [0.0]
    assert np.array_equal(trajectory.positions, np.array([[1.0, 2.0, 3.0]]))
    assert np.array_equal(trajectory.quaternions_xyzw, np.array([[0.0, 0.0, 0.0, 1.0]]))
    assert load_survivors(tmp_path) == {"survivors": [{"id": "s1"}]}
    assert load_floorplan(tmp_path) is None
