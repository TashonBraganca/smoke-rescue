using System;
using System.Collections.Generic;
using SmokeRescue.Sensors;
using UnityEngine;

namespace SmokeRescue.Brain
{
    /// <summary>
    /// Linking two robots' maps without a shared frame. Each robot maps in its own frame, starting at its own origin.
    /// Every keyframe_spacing_m it keeps a keyframe (pose in its frame, a 360 beam LiDAR sweep, a Scan Context
    /// fingerprint) and radios keyframes with its map. Camera RANSAC poses seed the same own-map fit and uniqueness
    /// checks using a nearby partner LiDAR sweep. A receiver compares each partner keyframe with its own; a
    /// close fingerprint gives a candidate place and turn, which is checked by placing the partner's sweep into the
    /// receiver's own LiDAR map with the scan matcher. The verified pose gives the transform from the partner's frame
    /// to this one. Two candidates from places at least link_min_separation_m apart must agree before the link is
    /// accepted. Until then the partner's map, claims, survivors and heat are held back, since their coordinates mean
    /// nothing here. With frames_shared_at_start (entry doors known to an operator) the link is the identity.
    /// </summary>
    public sealed partial class RobotBrain
    {
        sealed class Keyframe { public int id; public BrainPose pose; public LidarScan scan; public ScanContext context; }
        readonly List<Keyframe> keyframes = new List<Keyframe>();
        readonly SortedDictionary<string, List<Keyframe>> partnerKeyframes = new SortedDictionary<string, List<Keyframe>>(StringComparer.Ordinal);
        readonly SortedDictionary<string, List<(BrainPose link, Vector2 theirs, int mineId, int theirsId, float fit, string source)>> linkCandidates =
            new SortedDictionary<string, List<(BrainPose, Vector2, int, int, float, string)>>(StringComparer.Ordinal);
        readonly SortedDictionary<string, BrainPose> links = new SortedDictionary<string, BrainPose>(StringComparer.Ordinal);
        int nextKeyframeId, keyframeRoundRobin, newestSentKeyframe = -1;
        /// <summary>Log every rejected fingerprint comparison (tests and tuning only; noisy in a run).</summary>
        public bool LinkDiagnostics { get; set; }

        /// <summary>Transform from a partner's map frame into this robot's frame, once linked.</summary>
        public bool TryPartnerLink(string partner, out BrainPose toOwn)
        {
            // visual_slam_live: a link refined by camera place matches replaces the operator's identity link.
            if (links.TryGetValue(partner ?? "", out toOwn)) return true;
            if (c.frames_shared_at_start) { toOwn = new BrainPose(); return partner != null && partner != id; }
            return false;
        }
        /// <summary>A partner's position, converted into this robot's frame when the frames are linked.</summary>
        public bool PartnerToOwn(string partner, Vector2 p, out Vector2 own)
        {
            if (!TryPartnerLink(partner, out var link)) { own = default; return false; }
            own = link.Transform(p.x, p.y); return true;
        }
        static Vector2 Inverse(BrainPose link, Vector2 p)
        {
            Vector2 d = p - link.position; float cos = Mathf.Cos(-link.yawRad), sin = Mathf.Sin(-link.yawRad);
            return new Vector2(cos * d.x - sin * d.y, sin * d.x + cos * d.y);
        }

        void KeepKeyframe(LidarScan scan, BrainPose pose, double t)
        {
            if (keyframes.Count > 0)
            {
                var last = keyframes[keyframes.Count - 1].pose;
                if (Vector2.Distance(last.position, pose.position) < c.keyframe_spacing_m &&
                    Mathf.Abs(BrainMath.Angle(last.yawRad, pose.yawRad)) < c.keyframe_turn_deg) return;
            }
            // The kept sweep is exactly what goes on the radio (every k-th beam, whole centimetres), so both robots
            // compare the same numbers.
            int stride = Math.Max(1, scan.RangesM.Length / c.keyframe_beams), n = scan.RangesM.Length / stride;
            var ranges = new float[n];
            for (int i = 0; i < n; i++) ranges[i] = Mathf.Round(scan.RangesM[i * stride] * 100f) / 100f;
            var kept = new LidarScan(scan.Timestamp, scan.AngleMinRad, scan.AngleIncrementRad * stride, scan.MinM, scan.MaxM,
                scan.MountForwardM, scan.MountHeightM, ranges);
            var kf = new Keyframe { id = nextKeyframeId++, pose = pose, scan = kept, context = new ScanContext(kept.AngleMinRad, kept.AngleIncrementRad, ranges) };
            keyframes.Add(kf);
            if (keyframes.Count > c.keyframe_max) keyframes.RemoveAt(0);
            // A place the partner saw earlier is recognised when this robot gets there (closest fingerprints only).
            // With shared frames (visual_slam_live) partner sweeps only verify camera matches; the camera proposes.
            if (!c.frames_shared_at_start)
                foreach (var pair in partnerKeyframes)
                    if (!links.ContainsKey(pair.Key)) foreach (var theirs in Shortlist(kf, pair.Value)) TryLink(pair.Key, theirs, kf, t);
        }

        void ReceiveKeyframes(string sender, BrainMessage m, double t)
        {
            if ((c.frames_shared_at_start && !c.visual_slam_live) || m.Keyframes.Length == 0) return;
            if (!partnerKeyframes.TryGetValue(sender, out var list)) partnerKeyframes[sender] = list = new List<Keyframe>();
            foreach (var k in m.Keyframes)
            {
                if (list.Exists(x => x.id == k.id)) continue;
                var ranges = new float[k.rangesCm.Length]; for (int i = 0; i < ranges.Length; i++) ranges[i] = k.rangesCm[i] / 100f;
                var scan = new LidarScan(t, k.angleMinRad, k.incrementRad, k.minM, k.maxM, k.mountForwardM, 0f, ranges);
                var kf = new Keyframe { id = k.id, pose = k.pose, scan = scan, context = new ScanContext(k.angleMinRad, k.incrementRad, ranges) };
                list.Add(kf);
                if (list.Count > c.keyframe_max) list.RemoveAt(0);
                if (!links.ContainsKey(sender) && !c.frames_shared_at_start) foreach (var mine in Shortlist(kf, keyframes)) TryLink(sender, kf, mine, t);
            }
        }

        List<Keyframe> Shortlist(Keyframe probe, List<Keyframe> pool)
        {
            var ranked = new List<(float d, Keyframe k)>();
            foreach (var other in pool)
                if (other.context.ReturnShare >= c.keyframe_min_return_share) ranked.Add((other.context.RingKeyDistance(probe.context), other));
            ranked.Sort((a, b) => a.d.CompareTo(b.d));
            var result = new List<Keyframe>();
            for (int i = 0; i < ranked.Count && i < c.link_shortlist; i++) result.Add(ranked[i].k);
            return result;
        }

        double verifyBudgetAt = double.NaN; int verificationsLeft;
        void TryLink(string sender, Keyframe theirs, Keyframe mine, double t)
        {
            if (links.ContainsKey(sender)) return;
            // Map verification is the expensive part; bound it per step so linking never stalls the robot's loop.
            if (verifyBudgetAt != t) { verifyBudgetAt = t; verificationsLeft = c.link_verifications_per_step; }
            if (mine.context.ReturnShare < c.keyframe_min_return_share || theirs.context.ReturnShare < c.keyframe_min_return_share) return;
            // Column j here matches column j + shift there, so the partner's bearings are turned by shift sectors:
            // placed at heading h in this frame, h = mine.yaw - shift * sector. The map check picks among the best turns.
            ScanMatcher.Result placed = default; float distance = 1; int shift = 0; float triedBest = 1, fitBest = 0;
            var turns = mine.context.BestShifts(theirs.context, c.link_turn_candidates, 3);
            // Rectangular rooms look alike turned 180 degrees: always let the map check try the opposite turn as well.
            if (turns.Count > 0)
            {
                int opposite = (turns[0].shift + ScanContext.Sectors / 2) % ScanContext.Sectors;
                if (!turns.Exists(x => x.shift == opposite)) turns.Add((mine.context.DistanceAt(theirs.context, opposite), opposite));
            }
            var tried = LinkDiagnostics ? new System.Text.StringBuilder() : null;
            foreach (var (d, s) in turns)
            {
                triedBest = Mathf.Min(triedBest, d);
                if (d > c.link_max_descriptor_distance || verificationsLeft <= 0) continue;
                verificationsLeft--;
                var prior = new BrainPose { position = mine.pose.position, yawRad = mine.pose.yawRad - s * ScanContext.SectorRad };
                var wide = matcher.Match(theirs.scan, prior, c.link_window_m, c.link_step_m, c.link_window_deg, c.link_step_deg);
                tried?.Append($"[s{s} d{d:F2} wide{wide.fit:F2}@({wide.pose.position.x:F2},{wide.pose.position.y:F2},{wide.pose.yawRad * Mathf.Rad2Deg:F0})]");
                // The coarse grid only has to land near the right pose; the bar applies after fine refinement.
                if (!wide.applied || wide.fit < c.link_coarse_min_fit) continue;
                var fine = matcher.Match(theirs.scan, wide.pose);
                var best = fine.applied && fine.fit >= wide.fit ? fine : wide;
                fitBest = Mathf.Max(fitBest, best.fit);
                if (best.fit < c.link_min_fit) continue;
                if (!placed.applied || best.fit > placed.fit) { placed = best; distance = d; shift = s; }
            }
            if (placed.applied && !Unique(theirs.scan, placed)) { if (LinkDiagnostics) Emit(t, "place_rejected", "Match is not unique: the sweep fits nearly as well shifted, as along one straight wall.", "fit", placed.fit); return; }
            if (!placed.applied)
            {
                if (LinkDiagnostics) Emit(t, "place_rejected", "Fingerprint comparison did not verify: " + tried, "descriptor_distance", triedBest, "fit", fitBest,
                    "mine_x_m", mine.pose.position.x, "mine_y_m", mine.pose.position.y, "theirs_x_m", theirs.pose.position.x, "theirs_y_m", theirs.pose.position.y);
                return;
            }
            AddLinkCandidate(sender, placed.pose, theirs.pose, mine.pose, mine.id, theirs.id, placed.fit, "LiDAR", t,
                "descriptor_distance", distance, "shift", shift);
        }

        /// <summary>
        /// A verified place match from either sensor: their keyframe (pose there) lies at placedTheirs in this frame. The
        /// transform from the partner's frame follows; two candidates from places at least link_min_separation_m apart
        /// that agree within link_agree_m and link_agree_deg link the frames, whichever sensor each came from.
        /// </summary>
        void AddLinkCandidate(string sender, BrainPose placedTheirs, BrainPose theirsPose, BrainPose minePose, int mineId, int theirsId,
                              float score, string source, double t, params object[] extra)
        {
            if (links.ContainsKey(sender)) return;
            // Partner frame -> this frame: the verified pose of their keyframe, composed with the inverse of its pose there.
            // Normalised: two headings near +-180 deg differ by a small turn, not by 360 deg (seed 934 live, 24 Oct: -363.5 deg).
            float yaw = BrainMath.Angle(theirsPose.yawRad, placedTheirs.yawRad) * Mathf.Deg2Rad, cos = Mathf.Cos(yaw), sin = Mathf.Sin(yaw);
            var p = theirsPose.position;
            var link = new BrainPose { yawRad = yaw, position = placedTheirs.position - new Vector2(cos * p.x - sin * p.y, sin * p.x + cos * p.y) };
            bool refine = c.frames_shared_at_start;
            float agreeM = refine ? c.visual_refine_agree_m : c.link_agree_m, agreeDeg = refine ? c.visual_refine_agree_deg : c.link_agree_deg;
            if (refine && (Mathf.Abs(BrainMath.Angle(0, link.yawRad)) > c.visual_refine_max_deg || link.position.magnitude > c.visual_refine_max_m))
            {
                // The operator's launch pads bound how far off the shared frame can be; a larger implied correction is a wrong match.
                Emit(t, "visual_link_rejected", "A camera place match implies a correction of the shared frame larger than the launch pads allow: ignored.",
                    "dx_m", link.position.x, "dy_m", link.position.y, "dyaw_deg", link.yawRad * Mathf.Rad2Deg, "score", score);
                return;
            }
            if (!linkCandidates.TryGetValue(sender, out var candidates)) linkCandidates[sender] = candidates = new List<(BrainPose, Vector2, int, int, float, string)>();
            foreach (var other in candidates)
            {
                if (Vector2.Distance(other.theirs, theirsPose.position) < c.link_min_separation_m) continue;
                if (Vector2.Distance(other.link.position, link.position) > agreeM ||
                    Mathf.Abs(BrainMath.Angle(other.link.yawRad, link.yawRad)) > agreeDeg) continue;
                var agreed = new BrainPose { position = (other.link.position + link.position) / 2,
                    yawRad = other.link.yawRad + BrainMath.Angle(other.link.yawRad, link.yawRad) * Mathf.Deg2Rad / 2 };
                links[sender] = agreed;
                if (refine)
                    Emit(t, "frame_link_refined", "Two camera place matches with " + sender + ", each checked against this robot's LiDAR map, agree on how its frame sits in this one: its map, claims and finds now convert through the measured link instead of the launch pads.",
                        "dx_m", agreed.position.x, "dy_m", agreed.position.y, "dyaw_deg", agreed.yawRad * Mathf.Rad2Deg,
                        "score", score, "candidates", candidates.Count + 1, "visual", (source == "camera" ? 1 : 0) + (other.source == "camera" ? 1 : 0));
                else Emit(t, "frame_linked", "Two of " + sender + "'s fingerprints (" + other.source + " and " + source + ") matched places in this map and agree on one transform; its map, claims and finds now convert into this frame.",
                    "dx_m", agreed.position.x, "dy_m", agreed.position.y, "dyaw_deg", agreed.yawRad * Mathf.Rad2Deg,
                    "score", score, "candidates", candidates.Count + 1, "visual", (source == "camera" ? 1 : 0) + (other.source == "camera" ? 1 : 0));
                if (peers.TryGetValue(sender, out var state) && state.message != null) ApplyPeerContent(sender, state, t);
                return;
            }
            candidates.Add((link, theirsPose.position, mineId, theirsId, score, source));
            if (candidates.Count > c.link_max_candidates) candidates.RemoveAt(0);
            var values = new List<object> { "score", score, "mine_x_m", minePose.position.x, "mine_y_m", minePose.position.y,
                "theirs_x_m", theirsPose.position.x, "theirs_y_m", theirsPose.position.y,
                "link_x_m", link.position.x, "link_y_m", link.position.y, "link_yaw_deg", link.yawRad * Mathf.Rad2Deg };
            values.AddRange(extra);
            Emit(t, "place_recognised", "A " + source + " fingerprint from " + sender + " matched a place in this map; waiting for a second, independent match before linking.",
                values.ToArray());
        }

        // ---- visual keyframes: ORB features with body-frame positions, made by this robot's own OpenCV 5 place service
        sealed class VisualKeyframe { public int id; public BrainPose pose; public byte[] payload; public double t = double.NaN; }
        readonly List<VisualKeyframe> visualKeyframes = new List<VisualKeyframe>();
        readonly Dictionary<string, SortedDictionary<int, BrainVisualKeyframe>> partnerVisualFrames = new Dictionary<string, SortedDictionary<int, BrainVisualKeyframe>>(StringComparer.Ordinal);
        readonly Dictionary<string, Dictionary<int, BrainPose>> partnerVisualPoses = new Dictionary<string, Dictionary<int, BrainPose>>(StringComparer.Ordinal);
        readonly List<(string sender, BrainVisualKeyframe keyframe)> partnerVisualPending = new List<(string, BrainVisualKeyframe)>();
        int newestSentVisual = -1, visualRoundRobin;

        public bool VisualPlacesEnabled => c.visual_place_linking && (!c.frames_shared_at_start || c.visual_slam_live);
        public int VisualKeyframeCapacity => c.visual_keyframe_max;
        public float VisualKeyframePeriodS => c.visual_keyframe_period_s;
        /// <summary>A camera keyframe from this robot's own place service. The caller assigns a stable radio id even if
        /// the service restarts and reuses its own ids; pose is the brain's estimate at capture.</summary>
        public void AddVisualKeyframe(int keyframeId, BrainPose pose, byte[] payload) => AddVisualKeyframe(keyframeId, pose, payload, double.NaN);
        /// <summary>The same, with its capture time, so later views can close a loop on it (visual_slam_live).</summary>
        public void AddVisualKeyframe(int keyframeId, BrainPose pose, byte[] payload, double t)
        {
            if (keyframeId < 0 || !Finite(pose.position) || !BrainMath.Finite(pose.yawRad) || payload == null || payload.Length == 0 || payload.Length > c.visual_keyframe_max_payload_bytes) throw new ArgumentException("Invalid visual keyframe payload.");
            if (!VisualPlacesEnabled || visualKeyframes.Exists(k => k.id == keyframeId)) return;
            visualKeyframes.Add(new VisualKeyframe { id = keyframeId, pose = pose, payload = (byte[])payload.Clone(), t = t });
            if (visualKeyframes.Count > c.visual_keyframe_max) visualKeyframes.RemoveAt(0);
            foreach (var pair in partnerVisualFrames)
                if (!links.ContainsKey(pair.Key)) foreach (var k in pair.Value.Values) QueueVisual(pair.Key, k);
        }

        /// <summary>A restarted place worker cannot keep the old service ids. Radio ids remain monotone in the caller.</summary>
        public void ResetVisualKeyframes()
        {
            visualKeyframes.Clear(); newestSentVisual = -1; visualRoundRobin = 0;
        }

        /// <summary>Partner camera keyframes queued for matching, including retries when an own view arrives.
        /// Only unlinked frames need matching. The returned payloads are detached from the retained fingerprints.</summary>
        public List<(string sender, BrainVisualKeyframe keyframe)> DrainPartnerVisualKeyframes()
        {
            var list = new List<(string, BrainVisualKeyframe)>();
            foreach (var item in partnerVisualPending)
            {
                var k = item.keyframe; k.payload = (byte[])k.payload.Clone(); list.Add((item.sender, k));
            }
            partnerVisualPending.Clear(); return list;
        }

        void QueueVisual(string sender, BrainVisualKeyframe k)
        {
            if (partnerVisualPending.Exists(x => x.sender == sender && x.keyframe.id == k.id)) return;
            partnerVisualPending.Add((sender, k));
            if (partnerVisualPending.Count > c.visual_keyframe_max) partnerVisualPending.RemoveAt(0);
        }
        void ReceiveVisualKeyframes(string sender, BrainMessage m)
        {
            if (!VisualPlacesEnabled || links.ContainsKey(sender) || m.VisualKeyframes.Length == 0) return;
            if (!partnerVisualFrames.TryGetValue(sender, out var frames)) partnerVisualFrames[sender] = frames = new SortedDictionary<int, BrainVisualKeyframe>();
            if (!partnerVisualPoses.TryGetValue(sender, out var poses)) partnerVisualPoses[sender] = poses = new Dictionary<int, BrainPose>();
            foreach (var v in m.VisualKeyframes)
            {
                if (frames.ContainsKey(v.id)) continue;
                frames[v.id] = v; poses[v.id] = v.pose; QueueVisual(sender, v);
                while (frames.Count > c.visual_keyframe_max)
                {
                    int oldest = -1; foreach (int key in frames.Keys) { oldest = key; break; }
                    frames.Remove(oldest); poses.Remove(oldest);
                    partnerVisualPending.RemoveAll(x => x.sender == sender && x.keyframe.id == oldest);
                }
            }
        }

        /// <summary>
        /// This robot's place service recognised a partner camera keyframe as its own keyframe mineId: the partner keyframe's
        /// body pose in that own keyframe's body frame is (dx right, dy forward, dyaw counter-clockwise), from ORB matches
        /// verified by RANSAC on their 3D positions. It becomes a link candidate like a verified LiDAR match.
        /// </summary>
        public void OnVisualPlaceMatch(string sender, int theirsId, int mineId, float dxM, float dyM, float dyawDeg, int inliers, double t)
        {
            BrainMath.Time(t);
            if (!BrainMath.Finite(dxM) || !BrainMath.Finite(dyM) || !BrainMath.Finite(dyawDeg)) throw new ArgumentException("Nonfinite visual match.");
            if (!VisualPlacesEnabled || links.ContainsKey(sender) || inliers < c.visual_link_min_inliers) return;
            var mine = visualKeyframes.Find(k => k.id == mineId);
            if (mine == null || !partnerVisualPoses.TryGetValue(sender, out var poses) || !poses.TryGetValue(theirsId, out var theirsPose)) return;
            var placed = new BrainPose { position = mine.pose.Transform(dxM, dyM), yawRad = mine.pose.yawRad + dyawDeg * Mathf.Deg2Rad };
            if (!VerifyVisualPlacement(sender, theirsPose, placed, t, out var sweepHere, out var sweepThere)) return;
            // Refining the launch pads' shared frame needs better than the camera's turn: RANSAC on 200 features scattered
            // 1.7 to 3.5 deg live (seed 934, 24 Oct 23:58) around a 2.4 deg error. The camera finds the place; the partner's
            // LiDAR sweep placed in the own map (0.5 deg steps) measures it, as a LiDAR fingerprint match does.
            if (c.frames_shared_at_start)
                AddLinkCandidate(sender, sweepHere, sweepThere, mine.pose, mineId, theirsId, inliers, "camera", t, "inliers", inliers,
                    "camera_yaw_deg", BrainMath.Angle(theirsPose.yawRad, placed.yawRad));
            else AddLinkCandidate(sender, placed, theirsPose, mine.pose, mineId, theirsId, inliers, "camera", t, "inliers", inliers);
        }

        // RANSAC measures a body-to-body pose. A nearby partner LiDAR sweep must also fit this robot's own map at that
        // pose, without moving farther than the same agreement limits used by LiDAR candidates. The uniqueness check
        // rejects the repeated-corridor case even when ORB produces enough inliers (place_eval database negatives).
        bool VerifyVisualPlacement(string sender, BrainPose cameraThere, BrainPose cameraHere, double t) =>
            VerifyVisualPlacement(sender, cameraThere, cameraHere, t, out _, out _);
        /// <summary>The same, giving the verified LiDAR placement: the partner sweep's pose in this map and its pose there.</summary>
        bool VerifyVisualPlacement(string sender, BrainPose cameraThere, BrainPose cameraHere, double t, out BrainPose sweepHere, out BrainPose sweepThere)
        {
            sweepHere = sweepThere = default;
            Keyframe nearest = null; float distance = c.keyframe_spacing_m;
            if (partnerKeyframes.TryGetValue(sender, out var scans)) foreach (var k in scans)
            {
                float d = Vector2.Distance(k.pose.position, cameraThere.position);
                if (d <= distance && k.context.ReturnShare >= c.keyframe_min_return_share) { nearest = k; distance = d; }
            }
            if (nearest == null) return RejectVisual(t, "No nearby partner LiDAR sweep can verify the camera pose.");
            if (verifyBudgetAt != t) { verifyBudgetAt = t; verificationsLeft = c.link_verifications_per_step; }
            if (verificationsLeft <= 0) return RejectVisual(t, "Map verification budget is exhausted for this step.");
            verificationsLeft--;
            float yaw = cameraHere.yawRad - cameraThere.yawRad;
            Vector2 d0 = nearest.pose.position - cameraThere.position;
            float cos = Mathf.Cos(yaw), sin = Mathf.Sin(yaw);
            var prior = new BrainPose { position = cameraHere.position + new Vector2(cos * d0.x - sin * d0.y, sin * d0.x + cos * d0.y),
                yawRad = nearest.pose.yawRad + yaw };
            var wide = matcher.Match(nearest.scan, prior, c.link_window_m, c.link_step_m, c.link_window_deg, c.link_step_deg);
            if (!wide.applied || wide.fit < c.link_coarse_min_fit) return RejectVisual(t, "Camera pose did not fit the own LiDAR map.");
            var fine = matcher.Match(nearest.scan, wide.pose);
            var placed = fine.applied && fine.fit >= wide.fit ? fine : wide;
            if (placed.fit < c.link_min_fit) return RejectVisual(t, "Camera pose did not reach the LiDAR link fit threshold.");
            if (Vector2.Distance(placed.pose.position, prior.position) > c.link_agree_m ||
                Mathf.Abs(BrainMath.Angle(placed.pose.yawRad, prior.yawRad)) > c.link_agree_deg)
                return RejectVisual(t, "Camera RANSAC pose disagrees with its LiDAR map placement.");
            if (!Unique(nearest.scan, placed)) return RejectVisual(t, "Camera placement is not unique in the LiDAR map.");
            sweepHere = placed.pose; sweepThere = nearest.pose;
            return true;
        }
        /// <summary>Whether camera matches with this partner can still change anything: frames not yet linked by measurement.</summary>
        public bool VisualMatchWanted(string partner) => VisualPlacesEnabled && partner != null && partner != id && !links.ContainsKey(partner);

        /// <summary>visual_slam_live: own camera keyframes near where the partner stood (its keyframe pose through the current
        /// link) and facing within visual_prior_turn_deg, the only views its keyframe is compared with. Null without a link,
        /// where every own view is a candidate.</summary>
        public List<int> VisualCandidates(string sender, BrainPose theirs)
        {
            if (!c.visual_slam_live || !TryPartnerLink(sender, out var link)) return null;
            var at = new BrainPose { position = link.Transform(theirs.position.x, theirs.position.y), yawRad = theirs.yawRad + link.yawRad };
            return Near(at, k => true);
        }
        /// <summary>visual_slam_live: own keyframes taken at least visual_loop_min_age_s before keyframe newId, near its pose and
        /// facing the same way: the earlier views of this place a loop closure can be found among.</summary>
        public List<int> LoopCandidates(int newId)
        {
            var now = visualKeyframes.Find(k => k.id == newId);
            if (!c.visual_slam_live || now == null || double.IsNaN(now.t)) return new List<int>();
            return Near(now.pose, k => !double.IsNaN(k.t) && now.t - k.t >= c.visual_loop_min_age_s);
        }
        List<int> Near(BrainPose at, Predicate<VisualKeyframe> keep)
        {
            var ids = new List<int>();
            foreach (var k in visualKeyframes)
                if (keep(k) && Vector2.Distance(k.pose.position, at.position) <= c.visual_prior_radius_m &&
                    Mathf.Abs(BrainMath.Angle(k.pose.yawRad, at.yawRad)) <= c.visual_prior_turn_deg) ids.Add(k.id);
            return ids;
        }

        /// <summary>
        /// visual_slam_live loop closure: this robot's newest view newId was recognised as its own earlier view oldId, with the
        /// new body pose (dx right, dy forward, dyaw) in the old body frame. Where the old pose puts the new view, against
        /// where the brain thought it was, is the drift since. The correction is applied, as a LiDAR match is, only when it is
        /// within the visual_refine bounds and the newest LiDAR sweep fits the own map at least as well after it.
        /// </summary>
        public void OnVisualLoopClosure(int newId, int oldId, float dxM, float dyM, float dyawDeg, int inliers, double t)
        {
            BrainMath.Time(t);
            if (!BrainMath.Finite(dxM) || !BrainMath.Finite(dyM) || !BrainMath.Finite(dyawDeg)) throw new ArgumentException("Nonfinite loop closure.");
            if (!c.visual_slam_live || inliers < c.visual_link_min_inliers) return;
            VisualKeyframe now = visualKeyframes.Find(k => k.id == newId), then = visualKeyframes.Find(k => k.id == oldId);
            if (now == null || then == null || newId == oldId) return;
            var implied = new BrainPose { position = then.pose.Transform(dxM, dyM), yawRad = then.pose.yawRad + dyawDeg * Mathf.Deg2Rad };
            var delta = ScanMatcher.Delta(now.pose, implied);
            float driftM = Vector2.Distance(implied.position, now.pose.position), driftDeg = BrainMath.Angle(now.pose.yawRad, implied.yawRad);
            bool within = driftM <= c.visual_refine_max_m && Mathf.Abs(driftDeg) <= c.visual_refine_max_deg, applied = false;
            float fitBefore = float.NaN, fitAfter = float.NaN;
            if (within && latestScan != null)
            {
                var prior = motion.At(latestScan.Timestamp);
                fitBefore = matcher.Match(latestScan, prior, 0, c.link_step_m, 0, c.link_step_deg).priorFit;
                fitAfter = matcher.Match(latestScan, DeadReckoning.Correct(prior, delta), 0, c.link_step_m, 0, c.link_step_deg).priorFit;
                if (fitAfter >= fitBefore) { motion.ApplyCorrection(delta); teamMap.ApplyCorrection(delta); applied = true; }
            }
            Emit(t, "visual_loop_closure", applied
                    ? "The camera recognised a place this robot saw earlier; the measured drift since then is corrected (the LiDAR map agrees)."
                    : within ? "The camera recognised a place this robot saw earlier; the LiDAR map fits worse after the correction, so the pose is kept."
                             : "The camera recognised a place this robot saw earlier, but the implied drift is beyond the correction bound: pose kept.",
                "age_s", now.t - then.t, "drift_m", driftM, "drift_deg", driftDeg, "inliers", inliers, "applied", applied ? 1 : 0,
                "fit_before", fitBefore, "fit_after", fitAfter, "x_m", now.pose.position.x, "y_m", now.pose.position.y);
        }

        bool RejectVisual(double t, string reason)
        {
            if (LinkDiagnostics) Emit(t, "place_rejected", reason);
            return false;
        }

        BrainVisualKeyframe[] VisualKeyframesToSend()
        {
            if (!VisualPlacesEnabled || visualKeyframes.Count == 0 || c.visual_keyframes_per_message < 1) return Array.Empty<BrainVisualKeyframe>();
            var chosen = new List<VisualKeyframe>();
            var newest = visualKeyframes[visualKeyframes.Count - 1];
            if (newest.id > newestSentVisual) { chosen.Add(newest); newestSentVisual = newest.id; }
            for (int tries = 0; tries < visualKeyframes.Count && chosen.Count < c.visual_keyframes_per_message; tries++)
            {
                var k = visualKeyframes[visualRoundRobin++ % visualKeyframes.Count];
                if (!chosen.Contains(k)) chosen.Add(k);
            }
            var result = new BrainVisualKeyframe[chosen.Count];
            for (int i = 0; i < chosen.Count; i++) result[i] = new BrainVisualKeyframe { id = chosen[i].id, pose = chosen[i].pose, payload = (byte[])chosen[i].payload.Clone() };
            return result;
        }

        /// <summary>A placement counts only if moving it link_unique_shift_m in any direction scores clearly worse; a
        /// sweep that sees one flat wall fits anywhere along it and would link the maps at a wrong offset.</summary>
        bool Unique(LidarScan scan, ScanMatcher.Result placed)
        {
            float d = c.link_unique_shift_m;
            foreach (var o in new[] { new Vector2(d, 0), new Vector2(-d, 0), new Vector2(0, d), new Vector2(0, -d) })
            {
                var shifted = new BrainPose { position = placed.pose.position + o, yawRad = placed.pose.yawRad };
                var score = matcher.Match(scan, shifted, 0, c.link_step_m, 0, c.link_step_deg);
                if (score.priorFit > c.link_unique_ratio * placed.fit) return false;
            }
            return true;
        }

        /// <summary>Keyframes for the next packet: the newest not yet sent, then older ones in turn, so a lost packet's
        /// keyframes are sent again later.</summary>
        BrainKeyframe[] KeyframesToSend(int maximum, BrainVisualKeyframe[] visual)
        {
            if ((c.frames_shared_at_start && !c.visual_slam_live) || keyframes.Count == 0) return Array.Empty<BrainKeyframe>();
            var chosen = new List<Keyframe>();
            // A delayed camera reply may describe an older capture. Carry a sweep near that capture pose, so the
            // receiver can verify it even while the robot keeps moving. The same keyframe_spacing_m bounds pairing.
            foreach (var v in visual)
            {
                Keyframe nearest = null; float distance = c.keyframe_spacing_m;
                foreach (var k in keyframes)
                {
                    float d = Vector2.Distance(k.pose.position, v.pose.position);
                    if (d <= distance) { distance = d; nearest = k; }
                }
                if (nearest != null && !chosen.Contains(nearest) && chosen.Count < maximum) chosen.Add(nearest);
            }
            var newest = keyframes[keyframes.Count - 1];
            if (chosen.Count < maximum && newest.id > newestSentKeyframe) chosen.Add(newest);
            foreach (var k in chosen) newestSentKeyframe = Math.Max(newestSentKeyframe, k.id);
            for (int tries = 0; tries < keyframes.Count && chosen.Count < maximum; tries++)
            {
                var k = keyframes[keyframeRoundRobin++ % keyframes.Count];
                if (!chosen.Contains(k)) chosen.Add(k);
            }
            var result = new BrainKeyframe[chosen.Count];
            for (int i = 0; i < chosen.Count; i++)
            {
                var k = chosen[i]; var cm = new ushort[k.scan.RangesM.Length];
                for (int j = 0; j < cm.Length; j++) cm[j] = (ushort)Mathf.Clamp(Mathf.RoundToInt(k.scan.RangesM[j] * 100f), 0, ushort.MaxValue);
                result[i] = new BrainKeyframe { id = k.id, pose = k.pose, angleMinRad = k.scan.AngleMinRad, incrementRad = k.scan.AngleIncrementRad,
                    minM = k.scan.MinM, maxM = k.scan.MaxM, mountForwardM = k.scan.MountForwardM, rangesCm = cm };
            }
            return result;
        }
    }
}
