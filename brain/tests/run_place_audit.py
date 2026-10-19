"""Independently inspect perfect C8 recall; never imported by robot modules."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from brain.oracle import load_trajectory


def audit(recording, out):
    recall = json.loads((out / "c8_recall.json").read_text())
    manifests = [json.loads((out / robot / "keyframes.json").read_text())
                 for robot in ("robot_A", "robot_B")]
    truths = [load_trajectory(recording, robot) for robot in ("robot_A", "robot_B")]
    positions, rotations, valid = [], [], []
    for manifest, truth in zip(manifests, truths):
        # Brute-force timestamp association, independent of the evaluator's
        # binary search. Retain the same explicit twenty millisecond validity.
        times = np.array([row["timestamp"] for row in manifest])
        dt = abs(times[:, None] - truth.timestamps[None, :])
        indices = np.argmin(dt, axis=1)
        positions.append(truth.positions[indices])
        rotations.append(truth.quaternions_xyzw[indices])
        valid.append(dt.min(1) <= .02)
    distances = np.linalg.norm(positions[1][:, None, :] - positions[0][None, :, :], axis=2)
    qa, qb = rotations
    qa = qa / np.linalg.norm(qa, axis=1, keepdims=True)
    qb = qb / np.linalg.norm(qb, axis=1, keepdims=True)
    angles = 2 * np.degrees(np.arccos(np.clip(abs(qb @ qa.T), 0, 1)))
    rankings = [q["top_5"] for q in recall["queries"]]
    masks = (distances <= .5) & (angles <= 30) & valid[0][None, :] & valid[1][:, None]
    eligible = masks.any(1)
    independent_hits = np.array([masks[i, ids].any() for i, ids in enumerate(rankings)])
    assert int(eligible.sum()) == recall["eligible_queries"]
    assert int(independent_hits[eligible].sum()) == recall["hits_at_5"]
    strict = (distances <= .1) & (angles <= 10) & valid[0][None, :] & valid[1][:, None]
    strict_eligible = strict.any(1)
    strict_hits = np.array([strict[i, ids].any() for i, ids in enumerate(rankings)])
    rng = np.random.default_rng(1729)
    controls = []
    for _ in range(100):
        mapping = rng.permutation(len(manifests[0]))
        controls.append(sum(masks[i, mapping[ids]].any() for i, ids in enumerate(rankings) if eligible[i]) / eligible.sum())
    eligible_indices = np.flatnonzero(eligible)
    post = [i for i in eligible_indices if manifests[1][i]["timestamp"] > manifests[0][-1]["timestamp"]]
    true_counts = masks[eligible].sum(1)
    first_rank = [next(rank + 1 for rank, index in enumerate(rankings[i]) if masks[i, index])
                  for i in eligible_indices]
    fingerprints = []
    for robot in ("robot_A", "robot_B"):
        desc = np.load(out / robot / "descriptors.npy")
        fingerprints.append({"robot": robot, "rows": len(desc),
                             "unique_descriptor_rows": len(np.unique(desc, axis=0)),
                             "sha256": hashlib.sha256(desc.tobytes()).hexdigest(),
                             "norm_min": float(np.linalg.norm(desc.astype(np.float32), axis=1).min()),
                             "norm_max": float(np.linalg.norm(desc.astype(np.float32), axis=1).max())})
    result = {"independent_brute_force_hits_at_5": int(independent_hits[eligible].sum()),
              "independent_eligible_queries": int(eligible.sum()),
              "top_1_hits": int(sum(masks[i, rankings[i][0]] for i in eligible_indices)),
              "true_match_count_per_query": {"min": int(true_counts.min()), "median": float(np.median(true_counts)), "max": int(true_counts.max())},
              "first_true_match_rank_counts": {str(i): first_rank.count(i) for i in range(1, 6)},
              "eligible_queries_after_A_recording_end": len(post),
              "hits_after_A_recording_end": int(independent_hits[post].sum()),
              "strict_0_1m_10deg_eligible": int(strict_eligible.sum()),
              "strict_0_1m_10deg_hits_at_5": int(strict_hits[strict_eligible].sum()),
              "random_assignment_control": {"seed": 1729, "repeats": 100, "mean_recall": float(np.mean(controls)), "min": float(min(controls)), "max": float(max(controls))},
              "descriptor_arrays": fingerprints,
              "unlabelled_A_keyframes": recall["unlabelled_database_indices"],
              "interpretation": "Both halves come from the same physical TUM sequence. The 0.5 m and 30 degree labels allow many positive A keyframes per B image. This is a stand-in retrieval result, not an independent multi-robot generalization test."}
    (out / "c8_audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    audit(args.recording, args.out)
