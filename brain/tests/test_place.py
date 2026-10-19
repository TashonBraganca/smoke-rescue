"""C8 tests catch encoding drift, future-frame retrieval and invalid payloads."""
import base64
import importlib.util

import numpy as np
import pytest


def place_module():
    assert importlib.util.find_spec("brain.place") is not None, "C8 place module is missing"
    from brain import place
    return place


def test_vlad_intra_normalizes_each_cluster_before_global_normalization():
    place = place_module()
    encoder = place.VLAD(np.array([[0., 0.], [10., 10.]], dtype=np.float32))
    actual = encoder.aggregate(np.array([[1., 0.], [0., 1.], [11., 10.]], dtype=np.float32))
    np.testing.assert_allclose(actual, [0.5, 0.5, 2 ** -0.5, 0], atol=1e-6)
    assert np.isfinite(encoder.aggregate(np.zeros((2, 2)))).all()


def test_projection_is_shared_deterministic_and_fingerprint_is_1024_bytes():
    place = place_module()
    rng = np.random.default_rng(8)
    centers = rng.normal(size=(32, 24)).astype(np.float32)
    patches = rng.normal(size=(40, 24)).astype(np.float32)
    a = place.VLAD(centers).describe(patches)
    b = place.VLAD(centers.copy()).describe(patches)
    assert a.shape == (512,) and a.dtype == np.float16
    np.testing.assert_array_equal(a, b)
    assert np.linalg.norm(a.astype(np.float32)) == pytest.approx(1., abs=.001)
    message = place.fingerprint("robot_A", "A-1", 4., 7, 3.5, a)
    decoded = np.frombuffer(base64.b64decode(message["payload"]["desc_f16_b64"]), dtype="<f2")
    assert decoded.nbytes == 1024
    np.testing.assert_array_equal(decoded, a)
    assert message["payload"]["keyframe_id"] == 7
    assert message["payload"]["t_keyframe"] == 3.5
    with pytest.raises(ValueError):
        place.fingerprint("robot_A", "A-1", 3., 7, 3.5, a)


def test_retrieval_filters_future_keyframes_and_handles_empty_maps():
    place = place_module()
    database = np.array([[1., 0.], [.8, .6], [0., 1.]], dtype=np.float16)
    query = np.array([1., 0.], dtype=np.float16)
    assert place.retrieve(query, database, times=np.array([4., 1., 2.]), available_at=3.) == [1, 2]
    assert place.retrieve(query, database, k=2) == [0, 1]
    assert place.retrieve(query, database[:0]) == []
    with pytest.raises(ValueError):
        place.retrieve(np.array([np.nan, 1]), database)


def test_timing_gate_cannot_pass_cpu_or_an_empty_overlap_set():
    place = place_module()
    assert place.acceptance_gate({"recall_at_5": .9, "eligible_queries": 10},
                                 {"device": "cpu", "mean_ms": 20.}) is False
    assert place.acceptance_gate({"recall_at_5": .9, "eligible_queries": 10},
                                 {"device": "mps", "mean_ms": 149.}) is True
    assert place.acceptance_gate({"recall_at_5": .9, "eligible_queries": 0},
                                 {"device": "mps", "mean_ms": 20.}) is False
    assert place.acceptance_gate({"recall_at_5": .79, "eligible_queries": 100},
                                 {"device": "mps", "mean_ms": 20.}) is False


def test_incomplete_truth_labels_cannot_pass_the_acceptance_gate():
    place = place_module()
    assert place.acceptance_gate({"recall_at_5": 1., "eligible_queries": 113,
                                  "truth_labels_complete": False},
                                 {"device": "mps", "mean_ms": 20.}) is False
