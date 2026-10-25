#!/usr/bin/env python3
"""Production-domain check for place_fit: the planar fit (ransac_rigid) against PnP (pnp_sqpnp) on Unity recordings.

The TUM harness (tools/reloc_compare.py) decides the ship rule. This checks that switching place_fit does not hurt the
simulated robots: each pair runs through place.match_keyframes exactly as the service does, the partner side as a
decoded radio payload (no pixels), the own side as place.make_keyframe stores it (with own pixels and the fixed mount).
Truth comes from tools/place_eval.py (truth_planar reads oracle/ on the evaluation side; the matcher never does).

Pairs (place_eval definitions): keyframes at 1 Hz from both robots; positives within 1.5 m and 30 deg, same-robot
revisits at least 10 s apart (capped at 300 per direction, seeded); negatives over 6 m apart, a seeded sample of
NEG_PER_REC per recording. A wrong pose is an accepted positive off by more than 5 deg or 0.5 m (the TUM definition).

    nice -n 10 brain/.venv/bin/python -I tools/reloc_unity_check.py [--no-write]
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import place_eval as pe  # noqa: E402
from place_eval import place, vision  # noqa: E402
import real_data_check as rdc  # noqa: E402

# The recordings on this machine that keep RGB frames and truth for both robots: two runs of the seed 934 house (the
# first is the one config/perception.json's place reasons cite) and the family ranch house (seed 2510).
RECORDINGS = ["unity_house_onedoor_seed934_brain_final_1954", "unity_house_onedoor_seed934_brain_allfixes_0007",
              "unity_family_ranch_agents"]
NEG_PER_REC = 3000
STEP_S = 1.0
WRONG_DEG, WRONG_M = 5.0, 0.5
DROPPED: dict = {}


def keyframes_for(rec, robot, settings):
    folder = pe.ROOT / "data/recordings" / rec / robot
    stamps = [float(line.split()[0]) for line in (folder / "rgb.txt").read_text().splitlines()
              if line.strip() and not line.startswith("#")]
    kfs, t_frames = [], []
    for t in np.arange(0.0, stamps[-1] + 1e-6, STEP_S):
        frame = vision.load_frame(str(folder), float(t))
        kf, _ = place.make_keyframe(frame.rgb, frame.depth_m, frame.K, settings)
        kfs.append(kf)
        t_frames.append(float(frame["stamps"]["rgb"]))
    # Samples whose frame has no truth stamp within 1 ms are dropped (the ranch recording has frames past its truth).
    truth_t = pe.load_trajectory(pe.ROOT / "data/recordings" / rec, robot).timestamps
    t_frames = np.array(t_frames)
    i = np.clip(np.searchsorted(truth_t, t_frames), 1, len(truth_t) - 1)
    keep = np.minimum(np.abs(truth_t[i - 1] - t_frames), np.abs(truth_t[i] - t_frames)) < 1e-3
    return [k for k, ok in zip(kfs, keep) if ok], t_frames[keep], int((~keep).sum())


def inliers_of(m):
    if m["matched"]:
        return m["inliers"]
    reason = m["reason"]
    return int(reason.split(" inliers")[0].split()[-1]) if reason.startswith("best candidate had") else 0


def main():
    t0 = time.time()
    settings = place.load_settings()
    s_rig = dict(settings, place_fit="ransac_rigid")
    s_pnp = dict(settings, place_fit="pnp_sqpnp")
    present = [r for r in RECORDINGS if (pe.ROOT / "data/recordings" / r / "oracle").exists()]
    pe.RECORDINGS = present
    kfs, truth = {}, {}
    for rec in present:
        for robot in pe.ROBOTS:
            k, tf, dropped = keyframes_for(rec, robot, settings)
            DROPPED[f"{rec}/{robot}"] = dropped
            o, h, _, _ = pe.truth_planar(rec, robot, tf, settings["mount"])
            kfs[(rec, robot)], truth[(rec, robot)] = k, (o, h, tf)
    # Convention gate: an own keyframe against its own payload is the identity under both fits.
    probe = next(k for k in kfs[(present[0], "robot_A")][40:] if k is not None)
    for s in (s_rig, s_pnp):
        m = place.match_keyframes(place.decode_payload(place.encode_payload(probe.desc, probe.xyh)), [(0, probe)], s)
        if not (m["matched"] and abs(m["dyaw_deg"]) < 0.5 and math.hypot(m["dx_m"], m["dy_m"]) < 0.03 and m["fit"] == s["place_fit"]):
            raise SystemExit(f"self-match is not the identity under {s['place_fit']}: {m}")
    pairs = pe.make_pairs(truth, kfs, settings, 10.0, 1.5, 30.0, 6.0, np.random.default_rng(0), 300)
    rng = np.random.default_rng(1)
    keep = []
    for rec in present:
        pos = [p for p in pairs if p[0] == rec and p[2] == "pos"]
        neg = [p for p in pairs if p[0] == rec and p[2] == "neg"]
        if len(neg) > NEG_PER_REC:
            neg = [neg[i] for i in sorted(rng.choice(len(neg), NEG_PER_REC, replace=False))]
        keep += pos + neg
    rows = []
    for rec, kind, label, i, j, _, _ in keep:
        pr, mr = {"cross": ("robot_B", "robot_A"), "cross_rev": ("robot_A", "robot_B"),
                  "same_A": ("robot_A", "robot_A"), "same_B": ("robot_B", "robot_B")}[kind]
        own = kfs[(rec, mr)][j]
        partner = place.decode_payload(place.encode_payload(kfs[(rec, pr)][i].desc, kfs[(rec, pr)][i].xyh))
        po, ph, _ = truth[(rec, pr)]
        mo, mh, _ = truth[(rec, mr)]
        true = pe.true_relative(po[i], ph[i], mo[j], mh[j])
        row = {"rec": rec, "cls": "cross" if kind.startswith("cross") else "same", "label": label}
        for name, s in (("ransac_rigid", s_rig), ("pnp_sqpnp", s_pnp)):
            t1 = time.perf_counter()
            m = place.match_keyframes(partner, [(0, own)], s)
            row["t_" + name] = time.perf_counter() - t1
            if m["matched"] and m["fit"] != name:
                raise SystemExit(f"place_fit={name} produced a {m['fit']} match")
            row[name] = (m["matched"], inliers_of(m),
                         pe.pose_error((m["dx_m"], m["dy_m"], m["dyaw_deg"]), true) if m["matched"] else None)
        rows.append(row)

    L = ["", f"UNITY CHECK of place_fit (tools/reloc_unity_check.py, {time.strftime('%Y-%m-%d %H:%M %Z')}, {time.time() - t0:.0f} s)",
         "Recordings: " + ", ".join(present) + f". 1 Hz keyframes of both robots; positives <= 1.5 m and < 30 deg (same-robot",
         f">= 10 s apart, <= 300 per direction); negatives > 6 m, {NEG_PER_REC} sampled per recording. Through place.match_keyframes",
         f"with partner = decoded payload, own = place.make_keyframe. PnP thresholds from config: {settings['place_pnp_reproj_px']} px,",
         f"{settings['place_pnp_min_inliers']} inliers, tilt <= {settings['place_pnp_max_tilt_deg']} deg. Self-match identity gate passed for both fits.",
         "Samples dropped for lack of a truth stamp: " + ", ".join(f"{k} {v}" for k, v in DROPPED.items() if v) + ".",
         "  fit / pairs                  pos  recall (Wilson 95 % CI)  rot med/p90 deg  trans med/p90 m  wrong/acc  false/neg maxneg  ms/match"]
    for cls in ("cross", "same", "all"):
        sub = [r for r in rows if cls == "all" or r["cls"] == cls]
        for name in ("ransac_rigid", "pnp_sqpnp"):
            pos = [r[name] for r in sub if r["label"] == "pos"]
            neg = [r[name] for r in sub if r["label"] == "neg"]
            acc = [p[2] for p in pos if p[0]]
            k, n = len(acc), len(pos)
            lo, hi = rdc.wilson(k, n)
            rot, tr = [a[1] for a in acc], [a[0] for a in acc]
            wrong = sum(a[1] > WRONG_DEG or a[0] > WRONG_M for a in acc)
            ms = 1000 * np.mean([r["t_" + name] for r in sub])
            L.append(f"  {name + ' ' + cls:<26} {n:>5} {100 * k / max(n, 1):5.1f} % ({100 * lo:4.1f}-{100 * hi:4.1f}) "
                     f"{rdc.pct(rot, 50):5.2f}/{rdc.pct(rot, 90):5.2f}  {rdc.pct(tr, 50):.3f}/{rdc.pct(tr, 90):.3f}  "
                     f"{wrong:>4}/{k:<5} {sum(x[0] for x in neg):>3}/{len(neg):<5} {max([x[1] for x in neg], default=0):>3}  {ms:6.2f}")
    text = "\n".join(L) + "\n"
    print(text)
    if "--no-write" not in sys.argv[1:]:
        with open(rdc.EVIDENCE, "a") as fh:
            fh.write(text)


if __name__ == "__main__":
    main()
