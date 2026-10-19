"""TUM pose validation and association against recorded image timestamps."""

from pathlib import Path

import numpy as np


def validate_tum(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != 8 or not len(rows):
        raise ValueError("expected a nonempty TUM trajectory with eight columns")
    if not np.isfinite(rows).all() or np.any(np.diff(rows[:, 0]) <= 0):
        raise ValueError("trajectory must be finite with strictly increasing timestamps")
    if not np.allclose(np.linalg.norm(rows[:, 4:8], axis=1), 1, atol=1e-3):
        raise ValueError("trajectory quaternions must have unit length")
    return rows


def read_tum(path: Path | str) -> np.ndarray:
    return validate_tum(np.loadtxt(path, comments="#", ndmin=2))


def match_timestamps(frame_times: np.ndarray, pose_times: np.ndarray) -> np.ndarray:
    """Match exported timestamps to frames within ten microseconds, without reuse."""
    frame_times = np.asarray(frame_times)
    pose_times = np.asarray(pose_times)
    if (not len(frame_times) or not np.isfinite(frame_times).all()
            or not np.isfinite(pose_times).all()
            or np.any(np.diff(frame_times) <= 0)
            or np.any(np.diff(pose_times) <= 0)):
        raise ValueError("timestamps must be finite, unique and increasing")
    right = np.clip(np.searchsorted(frame_times, pose_times), 0, len(frame_times) - 1)
    left = np.maximum(right - 1, 0)
    indices = np.where(abs(frame_times[left] - pose_times)
                       <= abs(frame_times[right] - pose_times), left, right)
    if (np.any(abs(frame_times[indices] - pose_times) > 1e-5)
            or len(np.unique(indices)) != len(indices)):
        raise ValueError("trajectory timestamps do not match distinct recording frames")
    return indices


def tracking_summary(frame_times: np.ndarray, trajectory: np.ndarray, fps: float) -> dict:
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("camera fps must be positive")
    indices = match_timestamps(frame_times, trajectory[:, 0])
    tracked = np.zeros(len(frame_times), dtype=bool)
    tracked[indices] = True
    if tracked.sum() < .5 * len(frame_times):
        raise ValueError(f"trajectory has {tracked.sum()}/{len(frame_times)} frames, below 50%")
    # A missing frame owns the interval until the next input frame. The final
    # frame owns one nominal camera period. This includes startup without a pose.
    intervals = np.r_[np.diff(frame_times), 1 / fps]
    return {
        "frames": len(frame_times),
        "tracked_frames": int(tracked.sum()),
        "tracking_lost_frames": int((~tracked).sum()),
        "tracking_lost_s": float(intervals[~tracked].sum()),
    }
