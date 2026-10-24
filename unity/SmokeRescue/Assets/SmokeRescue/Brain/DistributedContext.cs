using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using UnityEngine;

namespace SmokeRescue.Brain
{
    public sealed partial class RobotBrain
    {
        sealed class PeerState
        {
            public BrainMessage message;
            public byte[] explored;
            public long acknowledgedOurs, completeMapSequence;
            public bool overlapLogged;
            // In this robot's frame; null or unset until the frames are linked.
            public BrainClaim claim;
            public BrainRoomClaim room;
            public BrainPose pose; public bool poseKnown;
            public double poseT; public Vector2 velocity;   // when the pose was sent, and the motion between the last two
            public Vector2? goal;
        }
        readonly SortedDictionary<string, PeerState> peers = new SortedDictionary<string, PeerState>(StringComparer.Ordinal);
        readonly List<BrainSurvivor> confirmed = new List<BrainSurvivor>();
        readonly List<BrainCarried> carriedOut = new List<BrainCarried>();
        /// <summary>A person was strapped onto this robot's sled: the nearest confirmed person goes off the to-rescue list
        /// here and, by radio, on linked partners.</summary>
        void MarkCarried(double t)
        {
            int best = -1; float bestM = float.PositiveInfinity;
            for (int i = 0; i < survivors.Count; i++) { float m = Vector2.Distance(survivors[i], motion.Pose.position); if (m < bestM) { bestM = m; best = i; } }
            if (best < 0) return;
            var p = survivors[best];
            carriedOut.Add(new BrainCarried { position = p, at = t, sourceRobot = id });
            ForgetSurvivorNear(p);
            Emit(t, "survivor_carried", "The person confirmed here is on this robot's sled; off the to-rescue list.", "x_m", p.x, "y_m", p.y);
        }
        /// <summary>This robot's own rejections from the last reject_share_s, newest first, at most BrainMessage.MaxRejected.</summary>
        BrainRejected[] RejectedToSend(double t) => tracks
            .Where(k => k.rejected && !k.peerRejected && !k.confirmed && t - k.rejectedAt <= c.reject_share_s)
            .OrderByDescending(k => k.rejectedAt).Take(BrainMessage.MaxRejected)
            .Select(k => new BrainRejected { position = k.position, from = k.rejectedFrom, at = k.rejectedAt }).ToArray();
        /// <summary>A partner's agent inspected this warm spot and found no person. It stays quiet here too unless this robot
        /// sees it from reject_new_view_deg round from where the partner stood, or from half the partner's distance, until
        /// reject_share_s after the rejection (seed 928: the two robots raised and inspected the same spot four times).
        /// A spot this robot is inspecting, has confirmed or rejected itself is left alone.</summary>
        void ReceiveRejected(BrainRejected rejected, string sender, BrainPose link, double t)
        {
            if (t - rejected.at > c.reject_share_s) return;
            var p = link.Transform(rejected.position.x, rejected.position.y);
            foreach (var s in survivors) if (Vector2.Distance(s, p) <= c.survivor_merge_m) return;
            Track nearest = null; float best = c.survivor_merge_m;
            foreach (var k in tracks) { float d = Vector2.Distance(k.position, p); if (!k.confirmed && d <= best) { best = d; nearest = k; } }
            if (nearest == null) { nearest = new Track { position = p, lastSeen = double.NegativeInfinity }; tracks.Add(nearest); }
            if (nearest == active || nearest.candidate || (nearest.rejected && !nearest.peerRejected) ||
                (nearest.peerRejected && nearest.rejectedAt >= rejected.at)) return;
            NoteRejectedSpot(p, rejected.at);
            nearest.rejected = true; nearest.peerRejected = true; nearest.rejectedAt = rejected.at;
            nearest.rejectedUntil = rejected.at + c.reject_share_s;
            nearest.rejectedFrom = link.Transform(rejected.from.x, rejected.from.y);
            Emit(t, "peer_rejection_received", sender + "'s agent inspected the warm spot here and found no person; it stays quiet unless seen from a new side.",
                "x_m", p.x, "y_m", p.y);
        }
        void ForgetSurvivorNear(Vector2 p)
        {
            survivors.RemoveAll(s => Vector2.Distance(s, p) <= c.survivor_merge_m);
            confirmed.RemoveAll(s => Vector2.Distance(s.position, p) <= c.survivor_merge_m);
        }
        readonly long[] observedSequence;
        Vector2 home, frontierCentre;
        float frontierRadius;
        bool frontierGoal;
        long sequence;
        double lastFull = double.NegativeInfinity, lastBuild = double.NegativeInfinity;
        double lastPeerReceived = double.NegativeInfinity;
        /// <summary>Where this robot stood when it last heard a partner: the radio reached between there and the partner.</summary>
        Vector2? contactAt;
        byte[] lastState;
        int SharedSize => Mathf.CeilToInt(grid.Size * grid.CellM / BrainMessage.ResolutionM);

        /// <summary>Left-compose a building-frame rigid delta, with left-positive radians in yawRad.
        /// Corrects current and retained capture poses; discards the current path.
        /// The operator's entry door remains the return anchor in building coordinates.
        /// The confidence map's source transform moves its shared history with the corrected pose.
        /// Legacy planner cells and survivor coordinates keep their current behaviour.</summary>
        public void ApplyCorrection(BrainPose delta)
        {
            if (!Finite(delta.position) || !BrainMath.Finite(delta.yawRad)) throw new ArgumentException("Nonfinite pose correction.");
            if (delta.position == Vector2.zero && delta.yawRad == 0) return;
            ResetStuckRecovery();
            motion.ApplyCorrection(delta);
            teamMap.ApplyCorrection(delta);
            ClearPath();
        }

        /// <summary>Call after Step at each 2-second broadcast boundary. The recorder alone decides
        /// delivery and queues newest pending state per sender/receiver. This method never reads radio geometry.
        /// Acknowledgements travel inside received peer messages; no recorder ACK call is needed.</summary>
        /// <summary>Own map cells (shared grid) observed since the oldest partner acknowledgement: what a partner has not got.</summary>
        int UnsharedCells()
        {
            long baseline = sequence;
            foreach (var peer in peers.Values) baseline = Math.Min(baseline, peer.acknowledgedOurs);
            int n = 0; foreach (long s in observedSequence) if (s > baseline) n++;
            return n;
        }

        public BrainMessage BuildMessage(double t)
        {
            BrainMath.Time(t);
            if (t < lastBuild || t < lastPeerReceived || (!double.IsNaN(lastStep) && t < lastStep))
                throw new ArgumentException("Build messages at the current simulation time after Step.");
            lastBuild = t;
            long baseline = sequence;
            if (peers.Count == 0) baseline = 0;
            foreach (var peer in peers.Values) baseline = Math.Min(baseline, peer.acknowledgedOurs);
            bool full = t - lastFull >= c.full_snapshot_s;
            int cells = SharedSize * SharedSize, bytes = (cells + 7) / 8;
            var visual = VisualKeyframesToSend();
            var m = new BrainMessage { Sender = id, BaseSequence = full ? 0 : baseline, GridSize = SharedSize,
                Origin = grid.Origin, Pose = motion.Pose, Goal = goal,
                Explored = new byte[bytes], Occupied = new byte[bytes], Survivors = confirmed.ToArray(), Keyframes = KeyframesToSend(visual.Length > 0 ? c.visual_lidar_keyframes_per_message : c.keyframes_per_message, visual), CarriedOut = carriedOut.ToArray(), VisualKeyframes = visual, Rejected = RejectedToSend(t),
                Air = airTrack.Skip(Math.Max(0, airTrack.Count - BrainMessage.MaxAir)).Select(a => new BrainAirSample { position = a.position, c = a.c, at = a.t }).ToArray() };
            bool searching = frontierGoal && frontierRadius > 0 && goal.HasValue && !completed && !returning && !retreating && active == null;
            if (searching)
                m.Claim = new BrainClaim { centre = frontierCentre, radiusM = frontierRadius, expiresAt = t + c.peer_claim_s };
            m.RoomClaim = RoomClaimToSend(t, searching);
            if (c.door_claims) m.Doors = DoorsToSend();
            for (int i = 0; i < grid.Count; i++)
            {
                int shared = m.Cell(grid.Centre(i));
                if (shared < 0) continue;
                if (grid.OwnObserved(i) && observedSequence[shared] == 0) observedSequence[shared] = sequence + 1;
                if (grid.OwnOccupancy(i) == 255) BrainMessage.Set(m.Occupied, shared);
            }
            for (int i = 0; i < cells; i++)
                if (observedSequence[i] > 0 && (full || observedSequence[i] > baseline)) BrainMessage.Set(m.Explored, i);
            m.Hotspots = LocalHotspots(m, t);
            var acknowledgements = new List<BrainAck>();
            foreach (var p in peers) acknowledgements.Add(new BrainAck { sender = p.Key, sequence = p.Value.completeMapSequence });
            m.Acknowledgements = acknowledgements.ToArray();
            if (teamMap.HasOwnEvidence || m.Hotspots.Length > 0) m.SentAt = t;
            teamMap.FillMessage(m);
            // Ignore send time when comparing idle states. Hot cells always refresh and need the
            // real send time here because dense heat encodes sample ages relative to it.
            if (m.Hotspots.Length > 0) m.SentAt = t;
            byte[] state = m.ToBytes();
            if (!full && m.Claim == null && m.RoomClaim == null && m.Hotspots.Length == 0 && m.Keyframes.Length == 0 && m.VisualKeyframes.Length == 0 && Equal(lastState, state)) return null;
            m.Sequence = sequence + 1; m.SentAt = t; m.FullSnapshot = full;
            m.ToBytes(); // Check the complete packet before advancing sender state.
            sequence = m.Sequence; lastState = state; if (full) lastFull = t;
            return m;
        }

        /// <summary>Radio decode boundary. Old or incomplete packets are logged and dropped before they touch peer state.</summary>
        public void OnPeerPacket(byte[] packet, double tReceived)
        {
            BrainMath.Time(tReceived);
            try { OnPeerMessage(BrainMessage.FromBytes(packet), tReceived); }
            catch (InvalidDataException e) { Emit(tReceived, "peer_message_rejected", "Dropped radio packet: " + e.Message); }
            catch (IOException e) { Emit(tReceived, "peer_message_rejected", "Dropped radio packet: " + e.Message); }
            catch (ArgumentException e) { Emit(tReceived, "peer_message_rejected", "Dropped radio packet: " + e.Message); }
        }

        /// <summary>Call on the simulation thread when a RadioModel-approved packet's delay has elapsed,
        /// before the next Step. Late map evidence remains useful; claims use send-time expiry and heat
        /// loses temperature for its full age. Duplicate and older sender sequences have no effect.</summary>
        public void OnPeerMessage(BrainMessage incoming, double tReceived)
        {
            BrainMath.Time(tReceived);
            if (incoming == null) throw new ArgumentNullException(nameof(incoming));
            if (incoming.Sender == id) return;
            if (incoming.Sender != null && peers.TryGetValue(incoming.Sender, out var existing) && incoming.Sequence <= existing.message.Sequence) return;
            if (tReceived < lastPeerReceived || (!double.IsNaN(lastStep) && tReceived < lastStep))
                throw new ArgumentException("Deliver peer packets in receive-time order before Step.");
            // Detach caller-owned buffers. A queued packet must itself remain unchanged until delivered.
            var bytes = incoming.ToBytes(); var m = BrainMessage.FromBytes(bytes);
            // A packet that fails validation is dropped and logged; it must never stop this robot.
            try { ValidateMessage(m, tReceived); }
            catch (InvalidDataException e)
            {
                Emit(tReceived, "peer_message_rejected", $"{m.Sender}: {e.Message}", "sequence", m.Sequence);
                return;
            }
            if (!peers.TryGetValue(m.Sender, out var peer))
            { peer = new PeerState { explored = new byte[m.Explored.Length] }; peers.Add(m.Sender, peer); }
            // Observation is monotone. OR also retains prior rooms if a newer delta follows a dropped full snapshot.
            for (int i = 0; i < peer.explored.Length; i++) peer.explored[i] |= m.Explored[i];
            // A new receiver may first hear a delta whose baseline it never received.
            // It can use the included cells, but must ACK zero until its explored history is complete.
            if (m.FullSnapshot || m.BaseSequence <= peer.completeMapSequence) peer.completeMapSequence = m.Sequence;
            peer.message = m;
            foreach (var ack in m.Acknowledgements)
                if (ack.sender == id && ack.sequence <= sequence) peer.acknowledgedOurs = Math.Max(peer.acknowledgedOurs, ack.sequence);
            lastPeerReceived = tReceived;
            contactAt = motion.Pose.position;
            grid.DecayTo(tReceived);
            ReceiveKeyframes(m.Sender, m, tReceived);
            ReceiveVisualKeyframes(m.Sender, m);
            Emit(tReceived, "peer_message_received", "Accepted sensor-derived state from " + m.Sender + (TryPartnerLink(m.Sender, out _) ? "." : "; frames not linked yet, so only its fingerprints are used."),
                "sequence", m.Sequence, "age_s", Math.Max(0, tReceived - m.SentAt), "bytes", bytes.Length, "keyframes", m.Keyframes.Length, "visual_keyframes", m.VisualKeyframes.Length);
            ApplyPeerContent(m.Sender, peer, tReceived);
        }

        /// <summary>The partner's map, claim, survivors and heat, converted into this frame. Nothing is applied before
        /// the frames are linked: a coordinate from another frame would put walls, fires and people in the wrong place.</summary>
        void ApplyPeerContent(string sender, PeerState peer, double t)
        {
            var m = peer.message;
            if (!TryPartnerLink(sender, out var link)) { peer.claim = null; peer.room = null; peer.goal = null; return; }
            grid.ReplacePeerLayer(sender, m, peer.explored, link, OwnBodyWasAt);
            teamMap.Receive(m, link);
            var before = peer.poseKnown ? peer.pose.position : (Vector2?)null; double beforeT = peer.poseT;
            peer.pose = new BrainPose { position = link.Transform(m.Pose.position.x, m.Pose.position.y), yawRad = m.Pose.yawRad + link.yawRad };
            peer.poseKnown = true; peer.poseT = m.SentAt;
            if (c.partner_rooms) planner.Rooms.MarkPartner(peer.pose.position, c.body_radius_m);
            double dtPose = m.SentAt - beforeT;
            peer.velocity = before.HasValue && dtPose > 0 && dtPose <= c.teammate_predict_max_s && Vector2.Distance(before.Value, peer.pose.position) <= c.teammate_track_max_m
                ? (peer.pose.position - before.Value) / (float)dtPose : Vector2.zero;
            // Where the partner stands, and the straight line it reported moving along since its last pose, is floor.
            float ends = Mathf.Max(0, c.body_half_length_m - c.body_radius_m); var f = peer.pose.Forward;
            grid.MarkTeammateFree(peer.pose.position - f * ends, peer.pose.position + f * ends, c.body_radius_m);
            if (before.HasValue && Vector2.Distance(before.Value, peer.pose.position) <= c.teammate_track_max_m)
                grid.MarkTeammateFree(before.Value, peer.pose.position, c.body_radius_m);
            peer.goal = m.Goal.HasValue ? link.Transform(m.Goal.Value.x, m.Goal.Value.y) : (Vector2?)null;
            peer.claim = m.Claim == null ? null : new BrainClaim { centre = link.Transform(m.Claim.centre.x, m.Claim.centre.y), radiusM = m.Claim.radiusM, expiresAt = m.Claim.expiresAt };
            peer.room = m.RoomClaim == null ? null : new BrainRoomClaim { anchor = link.Transform(m.RoomClaim.anchor.x, m.RoomClaim.anchor.y),
                radiusM = m.RoomClaim.radiusM, bidS = m.RoomClaim.bidS, since = m.RoomClaim.since, expiresAt = m.RoomClaim.expiresAt };
            if (c.door_claims && m.Doors.Length > 0)
            {
                var turn = new BrainPose { position = Vector2.zero, yawRad = link.yawRad };
                ReceiveDoors(Array.ConvertAll(m.Doors, d => new BrainDoor { position = link.Transform(d.position.x, d.position.y),
                    inward = turn.Transform(d.inward.x, d.inward.y), owner = d.owner, state = d.state, since = d.since }), t);
            }
            foreach (var hot in m.Hotspots)
            {
                // Fill every local cell under the shared 20 cm cell, preserving its full hazard extent.
                Vector2 centre = link.Transform(m.Centre(hot.cell).x, m.Centre(hot.cell).y);
                float half = BrainMessage.ResolutionM * .75f;
                for (float y = -half; y <= half; y += grid.CellM) for (float x = -half; x <= half; x += grid.CellM)
                    grid.ObservePeerHeat(grid.Cell(centre + new Vector2(x, y)), hot.temperatureC, hot.observedAt, t);
            }
            foreach (var carried in m.CarriedOut)
            {
                var p = link.Transform(carried.position.x, carried.position.y);
                bool known = false; foreach (var k in carriedOut) if (Vector2.Distance(k.position, p) <= c.survivor_merge_m) known = true;
                if (known) continue;
                carriedOut.Add(new BrainCarried { position = p, at = carried.at, sourceRobot = carried.sourceRobot });
                ForgetSurvivorNear(p);
                Emit(t, "survivor_carried_by_peer", carried.sourceRobot + " carried out the person confirmed here.", "x_m", p.x, "y_m", p.y);
            }
            foreach (var rejected in m.Rejected) ReceiveRejected(rejected, sender, link, t);
            foreach (var air in m.Air)
            {
                // A partner's reading of the air where it stood: the retreat uses the newest reading near each spot.
                var p = link.Transform(air.position.x, air.position.y);
                if (peerAir.Exists(a => a.t == air.at && Vector2.Distance(a.position, p) < .05f)) continue;
                peerAir.Add((p, air.c, air.at));
                if (peerAir.Count > 2000) peerAir.RemoveAt(0);
            }
            foreach (var survivor in m.Survivors)
            {
                var s = survivor; s.position = link.Transform(survivor.position.x, survivor.position.y);
                bool gone = false; foreach (var k in carriedOut) if (Vector2.Distance(k.position, s.position) <= c.survivor_merge_m) gone = true;
                if (!gone) ReceiveSurvivor(s, sender, t);
            }
            InvalidatePeerGoal(t); YieldLostRoom(t); CheckMapOverlap(t);
        }

        BrainHotspot[] LocalHotspots(BrainMessage m, double t)
        {
            var hottest = new SortedDictionary<int, BrainHotspot>();
            for (int i = 0; i < grid.Count; i++)
            {
                double at = grid.LocalHeatTime(i); if (double.IsNegativeInfinity(at)) continue;
                float temperature = grid.LocalHeat(i), effective = temperature - c.heat_decay_c_per_s * (float)Math.Max(0, t - at);
                if (effective < c.fire_c) continue;
                int cell = m.Cell(grid.Centre(i)); if (cell < 0) continue;
                if (!hottest.TryGetValue(cell, out var old) || effective > old.temperatureC - c.heat_decay_c_per_s * (float)(t - old.observedAt))
                    hottest[cell] = new BrainHotspot { cell = cell, temperatureC = temperature, observedAt = at };
            }
            var result = new List<BrainHotspot>(); foreach (var p in hottest) result.Add(p.Value); return result.ToArray();
        }
        void ReceiveSurvivor(BrainSurvivor survivor, string sender, double t)
        {
            foreach (var known in survivors) if (Vector2.Distance(known, survivor.position) <= c.survivor_merge_m) return;
            foreach (var w in withdrawn) if (Vector2.Distance(w, survivor.position) <= c.survivor_merge_m) return;
            survivors.Add(survivor.position); confirmed.Add(survivor);
            foreach (var track in tracks)
                if (Vector2.Distance(track.position, survivor.position) <= c.survivor_merge_m) track.confirmed = true;
            if (active != null && active.confirmed) { active = null; dwell = 0; inspecting = false; ClearPath(); }
            Emit(t, "survivor_from_peer", "Confirmation received from " + sender + ", originally confirmed by " + survivor.sourceRobot + ".",
                "x_m", survivor.position.x, "y_m", survivor.position.y, "confirmed_at_s", survivor.confirmedAt);
        }
        void CheckMapOverlap(double t)
        {
            foreach (var pair in peers)
            {
                if (pair.Value.overlapLogged || !grid.OwnOverlap(pair.Key)) continue;
                pair.Value.overlapLogged = true;
                Emit(t, "map_shared", "Local and received explored cells first overlap with " + pair.Key + ".", "sequence", pair.Value.message.Sequence);
            }
        }
        /// <summary>Is p on a partner's body at time t? Within the body's half length plus teammate_mask_margin_m of the stretch
        /// it can have walked since its last radioed pose, for at most teammate_predict_max_s: toward the goal it radioed, at up
        /// to open_speed_mps, or at the velocity between its last two poses when it radioed none. On one-door seed 924's real
        /// LiDAR (A, 8 to 13.5 s, B walking in behind it) the velocity alone left 27 % of the returns on B's body unmasked, all
        /// in the second after B set off from standing; along its goal 1 % were left, with 0.7 % of other returns skipped.
        /// A partner not heard from for longer may be anywhere, so nothing is masked for it.</summary>
        bool OnTeammate(Vector2 p, double t)
        {
            foreach (var peer in peers.Values)
            {
                if (!peer.poseKnown) continue;
                double age = t - peer.poseT; if (age > c.teammate_predict_max_s) continue;
                float since = (float)Math.Max(0, age); Vector2 a = peer.pose.position, b = a + peer.velocity * since;
                if (peer.goal.HasValue)
                {
                    Vector2 to = peer.goal.Value - a; float far = to.magnitude;
                    b = far > 1e-4f ? a + to * (Mathf.Min(c.open_speed_mps * since, far) / far) : a;
                }
                Vector2 d = b - a;
                float u = d.sqrMagnitude < 1e-9f ? 0 : Mathf.Clamp01(Vector2.Dot(p - a, d) / d.sqrMagnitude);
                if (Vector2.Distance(p, a + d * u) <= c.body_half_length_m + c.teammate_mask_margin_m) return true;
            }
            return false;
        }
        bool ClaimBlocks(Vector2 p, double t)
        {
            foreach (var pair in peers)
            {
                var claim = pair.Value.claim;
                // Same overlap rule as InvalidatePeerGoal: a frontier here would carry our own claim radius, so it is
                // blocked whenever the two circles would overlap. A narrower test let a robot pick a goal that the
                // yield check dropped on the next step, dozens of times a second (check run, robot B, 7.4 s).
                if (claim == null || claim.expiresAt <= t || Vector2.Distance(p, claim.centre) > claim.radiusM + c.peer_claim_radius_m) continue;
                // Keep our existing contested claim only when our ID wins. Avoid all other live claims.
                if (frontierGoal && goal.HasValue && Vector2.Distance(frontierCentre, claim.centre) <= claim.radiusM + frontierRadius && CompareIds(id, pair.Key) < 0) continue;
                return true;
            }
            return false;
        }
        void InvalidatePeerGoal(double t)
        {
            if (!frontierGoal || !goal.HasValue) return;
            bool keepsContested = false;
            foreach (var pair in peers)
            {
                var claim = pair.Value.claim;
                if (claim == null || claim.expiresAt <= t || Vector2.Distance(frontierCentre, claim.centre) > claim.radiusM + frontierRadius) continue;
                if (CompareIds(id, pair.Key) < 0) { keepsContested = true; continue; }
                Emit(t, "goal_yielded", "The lower robot ID " + pair.Key + " keeps the contested frontier.", "x_m", goal.Value.x, "y_m", goal.Value.y);
                // An agent's frontier order inside the partner's claim yields too, and is dropped: kept, it was set again
                // the next tick and yielded again, 118 times in 2 s while robot B stood still (one-door seed 924, 52 s).
                if (AgentGoalActive(t) && agentGoalIsFrontier)
                {
                    Emit(t, "replan", "Agent goal is the partner's claimed frontier; back to exploring.", "x_m", agentGoal.Value.x, "y_m", agentGoal.Value.y);
                    agentGoal = null;
                }
                ClearPath(); return;
            }
            // Existing simultaneous claims use the ID tie-break even when both mapped the approach cell.
            // New frontier choices still exclude every peer-explored cell.
            if (keepsContested) return;
            if (grid.PeerExplored(grid.Cell(goal.Value)))
            {
                Emit(t, "replan", "A peer has already observed the current frontier.");
                // The same for an agent's frontier order on floor the partner already searched: kept, it was set again the
                // next tick and cleared again, 290 times in 5 s while robot A stood still (one-door seed 928, 61 to 66 s).
                if (AgentGoalActive(t) && agentGoalIsFrontier)
                {
                    Emit(t, "replan", "Agent goal is floor the partner already searched; back to exploring.", "x_m", agentGoal.Value.x, "y_m", agentGoal.Value.y);
                    agentGoal = null;
                }
                ClearPath();
            }
        }
        // Numeric suffixes compare numerically, so robot_2 wins over robot_10. Other IDs use ordinal order.
        static int CompareIds(string a, string b)
        {
            int ai = a.Length, bi = b.Length;
            while (ai > 0 && char.IsDigit(a[ai - 1])) ai--;
            while (bi > 0 && char.IsDigit(b[bi - 1])) bi--;
            if (ai < a.Length && bi < b.Length && a.Substring(0, ai) == b.Substring(0, bi) &&
                ulong.TryParse(a.Substring(ai), out var an) && ulong.TryParse(b.Substring(bi), out var bn) && an != bn) return an.CompareTo(bn);
            return StringComparer.Ordinal.Compare(a, b);
        }
        PeerSnapshot[] CopyPeers()
        {
            double t = Math.Max(double.IsNaN(lastStep) ? 0 : lastStep, lastPeerReceived);
            var result = new List<PeerSnapshot>();
            foreach (var p in peers)
            {
                var m = p.Value.message; var claim = p.Value.claim; var room = p.Value.room; bool linked = TryPartnerLink(p.Key, out _);
                result.Add(new PeerSnapshot { robot = p.Key, sequence = m.Sequence, sentAt = m.SentAt, pose = p.Value.pose, goal = p.Value.goal, linked = linked,
                    claim = claim == null || claim.expiresAt <= t ? null : new BrainClaim { centre = claim.centre, radiusM = claim.radiusM, expiresAt = claim.expiresAt },
                    room = room == null || room.expiresAt <= t ? null : new BrainRoomClaim { anchor = room.anchor, radiusM = room.radiusM, bidS = room.bidS, since = room.since, expiresAt = room.expiresAt } });
            }
            return result.ToArray();
        }
        SurvivorSnapshot[] CopySurvivors()
        {
            var result = new List<SurvivorSnapshot>();
            foreach (var s in confirmed) result.Add(new SurvivorSnapshot { position = s.position, confirmedAt = s.confirmedAt, sourceRobot = s.sourceRobot, source = s.sourceRobot == id ? "own" : "peer" });
            return result.ToArray();
        }
        void ValidateMessage(BrainMessage m, double received)
        {
            int n = (SharedSize * SharedSize + 7) / 8;
            if (string.IsNullOrWhiteSpace(m.Sender) || Encoding.UTF8.GetByteCount(m.Sender) > 256 || m.Sequence <= 0 || m.BaseSequence < 0 || m.BaseSequence >= m.Sequence ||
                !ValidTime(m.SentAt, received) || m.GridSize != SharedSize || m.Origin != grid.Origin ||
                m.Explored.Length != n || m.Occupied.Length != n || !Finite(m.Pose.position) || !BrainMath.Finite(m.Pose.yawRad) ||
                (m.Goal.HasValue && !Finite(m.Goal.Value))) throw new InvalidDataException("Invalid or incompatible building-frame brain state.");
            if (m.Claim != null && (!Finite(m.Claim.centre) || !BrainMath.Finite(m.Claim.radiusM) || m.Claim.radiusM <= 0 ||
                !ValidTime(m.Claim.expiresAt, m.SentAt + c.peer_claim_s + 1e-6) || m.Claim.expiresAt < m.SentAt)) throw new InvalidDataException("Invalid peer claim.");
            if (m.RoomClaim != null && (!Finite(m.RoomClaim.anchor) || !BrainMath.Finite(m.RoomClaim.radiusM) || m.RoomClaim.radiusM < 0 ||
                !BrainMath.Finite(m.RoomClaim.bidS) || m.RoomClaim.bidS < 0 || !ValidTime(m.RoomClaim.since, m.SentAt) ||
                !ValidTime(m.RoomClaim.expiresAt, m.SentAt + c.peer_claim_s + 1e-6) || m.RoomClaim.expiresAt < m.SentAt)) throw new InvalidDataException("Invalid peer room claim.");
            foreach (var s in m.Survivors)
                if (!Finite(s.position) || !ValidTime(s.confirmedAt, m.SentAt) || string.IsNullOrWhiteSpace(s.sourceRobot)) throw new InvalidDataException("Invalid peer confirmation.");
            foreach (var h in m.Hotspots)
                if (h.cell < 0 || h.cell >= SharedSize * SharedSize || !BrainMath.Finite(h.temperatureC) || h.temperatureC < c.fire_c || !ValidTime(h.observedAt, m.SentAt)) throw new InvalidDataException("Invalid peer hotspot.");
            foreach (var v in m.VisualKeyframes)
                if (v.id < 0 || !Finite(v.pose.position) || !BrainMath.Finite(v.pose.yawRad) || v.payload == null ||
                    v.payload.Length == 0 || v.payload.Length > c.visual_keyframe_max_payload_bytes)
                    throw new InvalidDataException("Invalid peer visual keyframe.");
            foreach (var k in m.Keyframes)
                if (k.id < 0 || !Finite(k.pose.position) || !BrainMath.Finite(k.pose.yawRad) ||
                    !BrainMath.Finite(k.angleMinRad) || !BrainMath.Finite(k.incrementRad) || !(k.incrementRad > 0) ||
                    !BrainMath.Finite(k.minM) || !BrainMath.Finite(k.maxM) || !(k.maxM > k.minM) || k.minM < 0 ||
                    !BrainMath.Finite(k.mountForwardM) || k.rangesCm.Length < 8)
                    throw new InvalidDataException("Invalid peer LiDAR keyframe.");
            foreach (var a in m.Acknowledgements)
                if (string.IsNullOrWhiteSpace(a.sender) || a.sequence < 0) throw new InvalidDataException("Invalid peer acknowledgement.");
        }
        static bool ValidTime(double t, double latest) => !double.IsNaN(t) && !double.IsInfinity(t) && t >= 0 && t <= latest + 1e-8;
        static bool Finite(Vector2 p) => BrainMath.Finite(p.x) && BrainMath.Finite(p.y);
        static bool Equal(byte[] a, byte[] b)
        {
            if (a == null || a.Length != b.Length) return false;
            for (int i = 0; i < a.Length; i++) if (a[i] != b[i]) return false;
            return true;
        }
    }
}
