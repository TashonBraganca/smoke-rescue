import numpy as np
from types import SimpleNamespace

from brain.detect import place_box


def test_box_centre_lands_at_its_depth_in_the_map_frame():
    cam = SimpleNamespace(fx=415.692, fy=415.692, cx=319.5, cy=239.5)
    depth = np.full((480, 640), 2.0, np.float32)
    # Camera 1 m along map x, turned 90 degrees about y: its optical z axis points along map +x.
    T = np.eye(4)
    T[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    T[:3, 3] = [1.0, 0.0, 0.0]
    p = place_box((300.0, 200.0, 339.0, 279.0), depth, cam, T)
    assert np.allclose(p, [3.0, 0.0, 0.0], atol=1e-6)


def test_box_without_valid_depth_is_skipped():
    cam = SimpleNamespace(fx=415.692, fy=415.692, cx=319.5, cy=239.5)
    depth = np.zeros((480, 640), np.float32)
    assert place_box((100.0, 100.0, 200.0, 200.0), depth, cam, np.eye(4)) is None
