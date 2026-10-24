using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using NUnit.Framework;
using SmokeRescue.Brain;
using SmokeRescue.Sensors;
using UnityEngine;

namespace SmokeRescue.Tests.EditMode
{
    public sealed partial class BrainTests
    {
        static byte[] VisualPayload(int features = 96)
        {
            var p = new byte[5 + 38 * features]; p[0] = (byte)'V'; p[1] = (byte)'P'; p[2] = 1;
            p[3] = (byte)features; p[4] = (byte)(features >> 8);
            for (int i = 5; i < p.Length; i++) p[i] = (byte)(i * 7);
            return p;
        }
        sealed class CameraFixture
        {
            public RobotBrain brain;
            public BrainConfig config;
            public BrainPose truth;
            public BrainPose[] own, partner;
            public void Match(int i, float offsetM = 0, int inliers = 30)
            {
                brain.OnVisualPlaceMatch("A", i + 1, 10 + i, offsetM, 0, 0, inliers, 1 + i);
            }
        }
        static CameraFixture CameraScene(bool map = true, bool enabled = true, bool corridor = false, float separationM = 1.5f,
                                         bool padsLive = false, float padErrorDeg = 0)
        {
            var c = Settings(); c.grid_size_m = 16; c.frames_shared_at_start = padsLive; c.visual_slam_live = padsLive;
            c.visual_place_linking = enabled;
            // With launch pads the partner's frame is the shared frame turned by the pads' placement error.
            var f = new CameraFixture { brain = new RobotBrain(Calibration(), c, "B", 0) { LinkDiagnostics = true }, config = c,
                truth = padsLive ? new BrainPose { yawRad = padErrorDeg * Mathf.Deg2Rad } : new BrainPose { position = new Vector2(3, -2), yawRad = Mathf.PI / 2 },
                own = new[] { new BrainPose(), new BrainPose { position = new Vector2(0, separationM) } }, partner = new BrainPose[2] };
            var walls = corridor ? Box(-1, -20, 1, 20) : RoomOne;
            var sender = new RobotBrain(Calibration(), c, "A", 0); var message = sender.BuildMessage(0);
            message.Keyframes = new BrainKeyframe[2]; message.VisualKeyframes = new BrainVisualKeyframe[2];
            message.Survivors = new[] { new BrainSurvivor { position = Vector2.zero, sourceRobot = "A", confirmedAt = 0 } };
            for (int i = 0; i < 2; i++)
            {
                var d = f.own[i].position - f.truth.position;
                float cos = Mathf.Cos(-f.truth.yawRad), sin = Mathf.Sin(-f.truth.yawRad);
                f.partner[i] = new BrainPose { position = new Vector2(cos * d.x - sin * d.y, sin * d.x + cos * d.y), yawRad = -f.truth.yawRad };
                var scan = SegmentScan(0, f.own[i].position, 0, walls);
                if (map) Map(f.brain).IntegrateLidar(scan, f.own[i]);
                message.Keyframes[i] = new BrainKeyframe { id = i + 1, pose = f.partner[i], angleMinRad = scan.AngleMinRad,
                    incrementRad = scan.AngleIncrementRad, minM = scan.MinM, maxM = scan.MaxM, mountForwardM = scan.MountForwardM,
                    rangesCm = scan.RangesM.Select(r => (ushort)Mathf.RoundToInt(r * 100)).ToArray() };
                f.brain.AddVisualKeyframe(10 + i, f.own[i], VisualPayload());
                message.VisualKeyframes[i] = new BrainVisualKeyframe { id = i + 1, pose = f.partner[i], payload = VisualPayload() };
            }
            f.brain.OnPeerMessage(Wire(message), .5); f.brain.DrainPartnerVisualKeyframes(); f.brain.DrainEvents();
            return f;
        }

        [Test]
        public void CameraMatchesWithoutALidarMapNeverBecomeLinkCandidates()
        {
            var absent = CameraScene(map: false); absent.Match(0); absent.Match(1);
            Assert.That(absent.brain.TryPartnerLink("A", out _), Is.False, "RANSAC alone cannot link frames");
            Assert.That(absent.brain.DrainEvents().Any(e => e.kind == "place_recognised"), Is.False);
            var present = CameraScene(); present.Match(0); present.Match(1);
            Assert.That(present.brain.TryPartnerLink("A", out _), Is.True, "control: own LiDAR map verifies both views");
        }

        [Test]
        public void RepeatingCorridorCameraMatchesFailTheLidarUniquenessTest()
        {
            var corridor = CameraScene(corridor: true); corridor.Match(0); corridor.Match(1);
            Assert.That(corridor.brain.TryPartnerLink("A", out _), Is.False);
            Assert.That(corridor.brain.DrainEvents().Any(e => e.kind == "place_rejected" && e.reason.Contains("unique")), Is.True);
            var room = CameraScene(); room.Match(0); room.Match(1);
            Assert.That(room.brain.TryPartnerLink("A", out _), Is.True, "control: off-centre pillars fix the placement");
        }

        [Test]
        public void CameraRansacDisagreeingWithALidarCandidateCannotLink()
        {
            var f = CameraScene();
            var method = typeof(RobotBrain).GetMethod("AddLinkCandidate", BindingFlags.NonPublic | BindingFlags.Instance);
            method.Invoke(f.brain, new object[] { "A", f.own[0], f.partner[0], f.own[0], 0, 1, 1f, "LiDAR", .6, Array.Empty<object>() });
            f.brain.DrainEvents();
            f.Match(0, .5f); f.Match(1, .5f);
            Assert.That(f.brain.TryPartnerLink("A", out _), Is.False, "two camera transforms agree with each other but disagree with LiDAR by 0.5 m");
            var rejected = f.brain.DrainEvents();
            Assert.That(rejected.Any(e => e.kind == "place_recognised"), Is.False);
            Assert.That(rejected.Any(e => e.kind == "place_rejected" && e.reason.Contains("disagrees")), Is.True);
            f.Match(1);
            Assert.That(f.brain.TryPartnerLink("A", out _), Is.True, "control: the agreeing camera candidate links");
        }

        [Test]
        public void CameraMatchCannotReplaceAnExistingFrameLink()
        {
            var f = CameraScene(); f.Match(0); f.Match(1);
            Assert.That(f.brain.TryPartnerLink("A", out var before), Is.True);
            f.Match(0, .1f); f.Match(1, .1f);
            // The shared candidate accumulator also owns immutability if another sensor path calls it after linking.
            var add = typeof(RobotBrain).GetMethod("AddLinkCandidate", BindingFlags.NonPublic | BindingFlags.Instance);
            var worse = f.own[1]; worse.position += new Vector2(.1f, 0);
            add.Invoke(f.brain, new object[] { "A", worse, f.partner[1], f.own[1], 11, 2, 1f, "camera", 3.0, Array.Empty<object>() });
            Assert.That(f.brain.TryPartnerLink("A", out var after), Is.True);
            Assert.That(after.position, Is.EqualTo(before.position)); Assert.That(after.yawRad, Is.EqualTo(before.yawRad));
            Assert.That(f.brain.DrainEvents().Count(e => e.kind == "frame_linked"), Is.EqualTo(1));
            var off = CameraScene(enabled: false); off.Match(0); off.Match(1);
            Assert.That(off.brain.TryPartnerLink("A", out _), Is.False, "control: no established link with the feature off");
        }

        [Test]
        public void CameraMatchesNeedSeparatedPartnerPlacesAndEnoughInliers()
        {
            var close = CameraScene(separationM: .2f); close.Match(0); close.Match(1);
            Assert.That(close.brain.TryPartnerLink("A", out _), Is.False, "two views at one place are not independent");
            var weak = CameraScene(); weak.Match(0, inliers: 1); weak.Match(1, inliers: 1);
            Assert.That(weak.brain.TryPartnerLink("A", out _), Is.False);
            var good = CameraScene(); good.Match(0); good.Match(1);
            Assert.That(good.brain.TryPartnerLink("A", out _), Is.True, "control: separated, verified views");
        }

        [Test]
        public void OldAndTruncatedRadioPacketsAreDroppedAndLogged()
        {
            var c = Settings(); var bytes = new RobotBrain(Calibration(), c, "A", 0).BuildMessage(0).ToBytes();
            var old = (byte[])bytes.Clone(); old[0] = 3;
            var bad = new[] { old, Array.Empty<byte>(), bytes.Take(1).ToArray(), bytes.Take(bytes.Length / 2).ToArray(), bytes.Take(bytes.Length - 1).ToArray() };
            var b = new RobotBrain(Calibration(), c, "B", 0);
            foreach (var packet in bad)
            {
                Assert.Catch(() => BrainMessage.FromBytes(packet), "control: the old recorder decode path throws on the same packet");
                Assert.DoesNotThrow(() => b.OnPeerPacket(packet, 1.0));
            }
            Assert.That(b.Snapshot().peers, Is.Empty);
            Assert.That(b.DrainEvents().Count(e => e.kind == "peer_message_rejected"), Is.EqualTo(bad.Length));
            b.OnPeerPacket(bytes, 1.0);
            Assert.That(b.Snapshot().peers.Length, Is.EqualTo(1), "control: the complete current packet is accepted");
        }
        [Test]
        public void CameraPacketsCarryANearbyLidarSweepWithinTheByteBudget()
        {
            BrainMessage Packet(bool visual, float captureY = 0)
            {
                var c = Settings(); c.frames_shared_at_start = false;
                var b = new RobotBrain(Calibration(), c, "A", 0);
                var keep = typeof(RobotBrain).GetMethod("KeepKeyframe", BindingFlags.NonPublic | BindingFlags.Instance);
                for (int i = 0; i < 2; i++)
                {
                    var pose = new BrainPose { position = new Vector2(0, i * 1.5f) };
                    keep.Invoke(b, new object[] { SegmentScan(i, pose.position, 0, RoomOne), pose, (double)i });
                }
                if (visual) b.AddVisualKeyframe(0, new BrainPose { position = new Vector2(0, captureY) }, VisualPayload());
                return Wire(b.BuildMessage(2));
            }
            var camera = Packet(true); var lidar = Packet(false);
            Assert.That(camera.VisualKeyframes.Length, Is.EqualTo(1));
            Assert.That(camera.Keyframes.Length, Is.EqualTo(1));
            Assert.That(camera.Keyframes[0].id, Is.EqualTo(0), "a delayed camera reply gets its older nearby sweep");
            Assert.That(Packet(true, 1.5f).Keyframes[0].id, Is.EqualTo(1), "a fresh capture gets the newest sweep");
            Assert.That(lidar.Keyframes.Length, Is.EqualTo(2), "control: LiDAR-only messages retain both sweeps");
            Assert.That(lidar.VisualKeyframes, Is.Empty);
        }

        [Test]
        public void VisualOnlyPacketsAreResentAndCallerPayloadsAreDetached()
        {
            var c = Settings(); c.frames_shared_at_start = false;
            var b = new RobotBrain(Calibration(), c, "A", 0); var payload = VisualPayload();
            b.AddVisualKeyframe(0, new BrainPose(), payload); payload[5] ^= 1;
            var first = b.BuildMessage(0);
            Assert.That(first.VisualKeyframes.Single().payload, Is.EqualTo(VisualPayload()), "caller mutation cannot change a fingerprint");
            first.VisualKeyframes[0].payload[5] ^= 1;
            var second = b.BuildMessage(2);
            Assert.That(second.VisualKeyframes.Single().payload, Is.EqualTo(VisualPayload()), "editing a built packet cannot change the stored keyframe");
            Assert.That(second, Is.Not.Null, "a lost visual-only packet is retried");
            Assert.That(second.VisualKeyframes.Length, Is.EqualTo(1));
            var off = new RobotBrain(Calibration(), c, "off", 0); off.BuildMessage(0);
            Assert.That(off.BuildMessage(2), Is.Null, "control: an idle robot without camera keyframes stays quiet");
        }

        [Test]
        public void PartnerVisualWorkIsBoundedAndRetriedWhenAnOwnViewArrives()
        {
            var c = Settings(); c.frames_shared_at_start = false; c.visual_keyframe_max = 3;
            var a = new RobotBrain(Calibration(), c, "A", 0); var b = new RobotBrain(Calibration(), c, "B", 0);
            var m = a.BuildMessage(0);
            m.VisualKeyframes = Enumerable.Range(0, 5).Select(i => new BrainVisualKeyframe { id = i, pose = new BrainPose(), payload = VisualPayload() }).ToArray();
            b.OnPeerMessage(Wire(m), 0);
            var pending = b.DrainPartnerVisualKeyframes();
            Assert.That(pending.Count, Is.LessThanOrEqualTo(3), "bound work received before the place service is ready");
            pending[0].keyframe.payload[5] ^= 1;
            Assert.That(b.DrainPartnerVisualKeyframes(), Is.Empty, "control: draining alone does not requeue work");
            b.AddVisualKeyframe(10, new BrainPose(), VisualPayload());
            var retries = b.DrainPartnerVisualKeyframes();
            Assert.That(retries.Select(x => x.keyframe.id), Is.EquivalentTo(new[] { 2, 3, 4 }),
                "the robot may reach a previously radioed place later");
            Assert.That(retries.All(x => x.keyframe.payload.SequenceEqual(VisualPayload())), Is.True,
                "a matcher cannot mutate the retained partner fingerprints");
        }
        [Test]
        public void InvalidVisualPoseOrOversizePayloadIsRejectedBeforeMatching()
        {
            var c = Settings(); c.frames_shared_at_start = false;
            var a = new RobotBrain(Calibration(), c, "A", 0);
            foreach (bool badPose in new[] { true, false })
            {
                var b = new RobotBrain(Calibration(), c, "B", 0); var m = a.BuildMessage(badPose ? 0 : 10);
                m.VisualKeyframes = new[] { new BrainVisualKeyframe { id = 0,
                    pose = new BrainPose { yawRad = badPose ? float.NaN : 0 }, payload = VisualPayload(badPose ? 96 : 97) } };
                Assert.DoesNotThrow(() => b.OnPeerPacket(m.ToBytes(), m.SentAt));
                Assert.That(b.DrainPartnerVisualKeyframes(), Is.Empty);
                Assert.That(b.DrainEvents().Any(e => e.kind == "peer_message_rejected"), Is.True);
            }
            var valid = CameraScene();
            Assert.That(valid.brain.Snapshot().peers.Length, Is.EqualTo(1), "control: valid camera and LiDAR metadata is accepted");
        }
        [Test]
        public void CameraVerificationUsesTheSamePerStepBudgetAsLidar()
        {
            var f = CameraScene(); f.config.link_verifications_per_step = 1;
            f.Match(0);
            f.brain.OnVisualPlaceMatch("A", 2, 11, 0, 0, 0, 30, 1);
            Assert.That(f.brain.TryPartnerLink("A", out _), Is.False, "the second verification waits for another step");
            Assert.That(f.brain.DrainEvents().Any(e => e.kind == "place_rejected" && e.reason.Contains("budget")), Is.True);
            f.Match(1);
            Assert.That(f.brain.TryPartnerLink("A", out _), Is.True, "control: a fresh step admits the second candidate");
        }
    
        [Test]
        public void WithLaunchPadsTwoAgreeingCameraMatchesRefineTheSharedFrame()
        {
            // visual_slam_live (owner, 24 Oct: visual SLAM live, not after the run). The partner's frame is the shared frame
            // turned 2.4 deg (robot B's measured launch error in seed 934). Two verified camera matches 1.5 m apart refine
            // the link to the measured turn. Control: one match leaves the pads' identity link.
            var one = CameraScene(padsLive: true, padErrorDeg: 2.4f); one.Match(0);
            Assert.That(one.brain.TryPartnerLink("A", out var identity), Is.True);
            Assert.That(Mathf.Abs(identity.yawRad), Is.LessThan(1e-6f), "control: one match changes nothing");
            var two = CameraScene(padsLive: true, padErrorDeg: 2.4f); two.Match(0); two.Match(1);
            Assert.That(two.brain.TryPartnerLink("A", out var link), Is.True);
            Assert.That(Mathf.Abs(link.yawRad * Mathf.Rad2Deg - 2.4f), Is.LessThan(.3f), "the link carries the partner frame's measured 2.4 deg turn: " + link.yawRad * Mathf.Rad2Deg);
            Assert.That(two.brain.DrainEvents().Count(e => e.kind == "frame_link_refined"), Is.EqualTo(1));
            Assert.That(two.brain.VisualMatchWanted("A"), Is.False, "a measured link ends the camera work for that partner");
        }

        [Test]
        public void ACameraMatchImplyingMoreThanThePadsAllowIsIgnored()
        {
            var wild = CameraScene(padsLive: true, padErrorDeg: 8f); wild.Match(0); wild.Match(1);
            Assert.That(wild.brain.TryPartnerLink("A", out var link), Is.True);
            Assert.That(Mathf.Abs(link.yawRad), Is.LessThan(1e-6f), "8 deg is beyond visual_refine_max_deg: identity kept");
            Assert.That(wild.brain.DrainEvents().Count(e => e.kind == "visual_link_rejected"), Is.EqualTo(2));
            var fair = CameraScene(padsLive: true, padErrorDeg: 4f); fair.Match(0); fair.Match(1);
            fair.brain.TryPartnerLink("A", out var fixedLink);
            Assert.That(Mathf.Abs(fixedLink.yawRad * Mathf.Rad2Deg - 4f), Is.LessThan(.3f), "control: 4 deg is within the bound and applied");
        }

        [Test]
        public void TwoCameraMatchesTwoDegreesApartDoNotRefineTheSharedFrame()
        {
            var f = CameraScene(padsLive: true, padErrorDeg: 2.4f); f.Match(0);
            f.brain.OnVisualPlaceMatch("A", 2, 11, 0, 0, 2f, 30, 2);   // the second view's camera turn is off by 2 deg
            Assert.That(f.brain.TryPartnerLink("A", out var link), Is.True);
            Assert.That(Mathf.Abs(link.yawRad), Is.LessThan(1e-6f), "2 deg apart is beyond visual_refine_agree_deg: identity kept");
            var loose = CameraScene(padsLive: true, padErrorDeg: 2.4f); loose.config.visual_refine_agree_deg = 4; loose.Match(0);
            loose.brain.OnVisualPlaceMatch("A", 2, 11, 0, 0, 2f, 30, 2);
            loose.brain.TryPartnerLink("A", out var looseLink);
            Assert.That(Mathf.Abs(looseLink.yawRad), Is.GreaterThan(.01f), "control: the old 4 deg tolerance would have accepted it");
        }

        [Test]
        public void OnlyOwnViewsNearThePartnerOrAnEarlierVisitAreCandidates()
        {
            var c = Settings(); c.frames_shared_at_start = true; c.visual_slam_live = true; c.visual_place_linking = true;
            var b = new RobotBrain(Calibration(), c, "B", 0);
            b.AddVisualKeyframe(1, new BrainPose { position = new Vector2(0, 0) }, VisualPayload(), 0);
            b.AddVisualKeyframe(2, new BrainPose { position = new Vector2(1, 1) }, VisualPayload(), 10);
            b.AddVisualKeyframe(3, new BrainPose { position = new Vector2(8, 0) }, VisualPayload(), 12);
            b.AddVisualKeyframe(4, new BrainPose { position = new Vector2(0, 1), yawRad = Mathf.PI }, VisualPayload(), 14);
            b.AddVisualKeyframe(5, new BrainPose { position = new Vector2(.5f, .5f) }, VisualPayload(), 30);
            CollectionAssert.AreEquivalent(new[] { 1, 2, 5 }, b.VisualCandidates("A", new BrainPose { position = new Vector2(.5f, .5f) }),
                "near and facing the same way: not the far view 3 or the opposite view 4");
            CollectionAssert.AreEquivalent(new[] { 1, 2 }, b.LoopCandidates(5), "earlier than visual_loop_min_age_s (20 s) before view 5, near it");
            var unlinked = Settings(); unlinked.frames_shared_at_start = false; unlinked.visual_slam_live = true;
            Assert.That(new RobotBrain(Calibration(), unlinked, "B", 0).VisualCandidates("A", new BrainPose()), Is.Null, "control: no link, no prior");
        }

        [Test]
        public void AVisualLoopClosureMeasuresTheDriftSinceTheEarlierView()
        {
            var c = Settings(); c.frames_shared_at_start = true; c.visual_slam_live = true; c.visual_place_linking = true;
            var b = new RobotBrain(Calibration(), c, "B", 0);
            b.AddVisualKeyframe(1, new BrainPose { position = new Vector2(0, 2) }, VisualPayload(), 0);
            b.AddVisualKeyframe(2, new BrainPose { position = new Vector2(0, 2) }, VisualPayload(), 40);
            b.DrainEvents();
            // The camera says the new view stands 0.2 m right of the old one, turned 1 deg: that is the drift since.
            b.OnVisualLoopClosure(2, 1, .2f, 0, 1f, 30, 40);
            var e = b.DrainEvents().Single(x => x.kind == "visual_loop_closure");
            Assert.That(e.numbers["drift_m"], Is.EqualTo(.2).Within(.01));
            Assert.That(e.numbers["drift_deg"], Is.EqualTo(1).Within(.05));
            Assert.That(e.numbers["age_s"], Is.EqualTo(40).Within(1e-6));
            b.OnVisualLoopClosure(2, 1, .2f, 0, 1f, 5, 41);
            Assert.That(b.DrainEvents().Any(x => x.kind == "visual_loop_closure"), Is.False, "control: too few inliers is not a loop closure");
        }
    }
}
