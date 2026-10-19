"""Retrieval evaluation must count misses and use translation and rotation."""
import numpy as np
from scipy.spatial.transform import Rotation

from brain import evaluator
from brain.oracle import Trajectory


def test_place_recall_counts_misses_and_excludes_opposite_views():
    assert hasattr(evaluator, "compute_place_recall"), "C8 retrieval evaluation is missing"
    truth_a = Trajectory(np.arange(6.), np.column_stack((np.arange(6.), np.zeros((6, 2)))),
                         np.tile([0., 0., 0., 1.], (6, 1)))
    truth_b = Trajectory(np.arange(3.), np.array([[.1, 0, 0], [3.1, 0, 0], [1.1, 0, 0]]),
                         Rotation.from_euler("z", [[0], [0], [180]], degrees=True).as_quat())
    descriptors_a = np.eye(6, dtype=np.float16)
    descriptors_b = np.array([[0, 5, 4, 3, 2, 1], [0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0]], dtype=np.float16)
    result = evaluator.compute_place_recall(truth_a, truth_b, truth_a.timestamps, truth_b.timestamps,
                                             descriptors_a, descriptors_b)
    assert result["eligible_queries"] == 2
    assert result["hits_at_5"] == 1
    assert result["recall_at_5"] == .5
    assert result["queries"][0]["hit"] is False
    assert result["queries"][0]["true_matches"] == [0]
    assert result["queries"][2]["eligible"] is False


def test_missing_startup_truth_is_reported_without_inventing_a_pose():
    truth_a = Trajectory(np.array([.067, 1., 2.]), np.array([[10., 0, 0], [20., 0, 0], [30., 0, 0]]),
                         np.tile([0., 0., 0., 1.], (3, 1)))
    truth_b = Trajectory(np.array([1.]), np.array([[20., 0, 0]]), np.array([[0., 0., 0., 1.]]))
    result = evaluator.compute_place_recall(truth_a, truth_b, np.array([0., 1., 2.]), np.array([1.]),
                                             np.eye(3), np.array([[0., 1., 0.]]))
    assert result["eligible_queries"] == 1
    assert result["hits_at_5"] == 1
    assert result["unlabelled_database_indices"] == [0]
    assert result["truth_labels_complete"] is False
    assert result["queries"][0]["top_5_distances_m"][1] is None
