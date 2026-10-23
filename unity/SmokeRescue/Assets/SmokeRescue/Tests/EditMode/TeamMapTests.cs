using System;
using System.Linq;
using NUnit.Framework;
using SmokeRescue.Brain;
using SmokeRescue.Sensors;
using UnityEngine;

namespace SmokeRescue.Tests.EditMode
{
    public sealed partial class BrainTests
    {
        [Test]
        public void GyroIntervalsDoNotDependOnBrainStepPhase()
        {
            var motion = new DeadReckoning(.2f, 2, new BrainPose());
            double sum = 0; int ii = 0, oi = 0;
            for (int tick = 0; tick <= 600; tick++)
            {
                double t = tick / 60.0;
                while (ii / 200.0 <= t + 1e-10)
                {
                    double at = ii / 200.0;
                    // A short turn alternates with rest. Rates are means of their 5 ms intervals.
                    float rate = ii % 40 < 7 ? 1.1f : 0;
                    if (ii > 0) sum += rate / 200.0;
                    motion.Add(new ImuSample(at, Vector3.zero, new Vector3(0, -rate, 0))); ii++;
                }
                while (oi / 50.0 <= t + 1e-10) motion.Add(new OdometrySample(oi++ / 50.0, 0, 0, 0));
                motion.Advance(t);
            }
            Assert.That(motion.Pose.yawRad, Is.EqualTo(sum).Within(.0001), "every complete sample interval contributes exactly once");
        }

        [Test]
        public void GyroIntervalsStayBoundedDuringAnOdometryOutage()
        {
            var motion = new DeadReckoning(.2f, 1, new BrainPose());
            motion.Add(new OdometrySample(0, 0, 0, 0));
            for (int i = 0; i <= 12000; i++)
            {
                double t = i / 200.0;
                motion.Add(new ImuSample(t, Vector3.zero, new Vector3(0, -.01f, 0))); motion.Advance(t);
            }
            var flags = System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance;
            var samples = (System.Collections.ICollection)typeof(DeadReckoning).GetField("imu", flags).GetValue(motion);
            Assert.That(samples.Count, Is.LessThan(1100), "a stopped stream must not retain the other sensor's whole run");
            Assert.That(motion.Healthy(60), Is.False);
            Assert.That(motion.Pose.yawRad, Is.EqualTo(.6).Within(.001));
        }

        static TeamMap Belief(string id = "A") => new TeamMap(id, 40, new Vector2(-4, -4), Settings());
        static void See(TeamMap map, double t, Vector2 p, bool wall, bool searched = false)
        { map.BeginObservation(t); map.Observe(p, wall, searched); map.EndObservation(); }
        static BrainMessage BeliefPacket(TeamMap map, long sequence, double t)
        {
            var m = new BrainMessage { Sender = map.Robot, Sequence = sequence, SentAt = t, GridSize = 40,
                Origin = new Vector2(-4, -4), Explored = new byte[200], Occupied = new byte[200] };
            map.FillMessage(m); return BrainMessage.FromBytes(m.ToBytes());
        }

        [Test]
        public void TeamMapFusesEachSourceOnceAndNeverForwardsFusedBelief()
        {
            var a = Belief(); var b = Belief("B"); var p = new Vector2(.1f, 1.1f);
            a.BeginObservation(1); for (int i = 0; i < 10; i++) a.Observe(p, true, false); a.EndObservation();
            var m = BeliefPacket(a, 1, 1); b.Receive(m, new BrainPose());
            float once = b.Export(1).occupancyProbability[b.Cell(p)];
            See(a, 1, p, true);
            Assert.That(a.Export(1).occupancyProbability[a.Cell(p)], Is.EqualTo(once), "duplicate camera captures do not add evidence");
            for (int i = 0; i < 10; i++) b.Receive(m, new BrainPose());
            Assert.That(b.Export(1).occupancyProbability[b.Cell(p)], Is.EqualTo(once));
            Assert.That(once, Is.EqualTo(1 / (1 + Math.Exp(-Settings().occupied_log_odds))).Within(.001));
            var returned = BeliefPacket(b, 1, 2);
            Assert.That(returned.TeamCells, Is.Empty, "B must send only B's observations, never A's fused belief");
            a.Receive(returned, new BrainPose());
            Assert.That(a.Export(2).occupancyProbability[a.Cell(p)], Is.EqualTo(once).Within(.001));
        }

        [Test]
        public void TeamMapSearchRequiresCameraFloorAndRetainsWhoAndWhen()
        {
            var a = Belief(); var b = Belief("B"); var p = new Vector2(.1f, 1.1f);
            See(a, 1, p, false); Assert.That(a.Export(1).searchedConfidence[a.Cell(p)], Is.Zero);
            See(a, 2, p, false, true); b.Receive(BeliefPacket(a, 1, 3), new BrainPose());
            var fresh = b.Export(3); var old = b.Export(63); int i = b.Cell(p);
            Assert.That(fresh.searchedSourceMask[i], Is.EqualTo(2));
            Assert.That(fresh.searchAgeS[i], Is.EqualTo(1).Within(.11));
            Assert.That(old.searchedConfidence[i], Is.EqualTo(fresh.searchedConfidence[i] * .5).Within(.001));
            Assert.That(old.sources[1], Is.EqualTo("A"));
        }

        [Test]
        public void TeamMapFreshLocalFreeKeepsUncertainPartnerWallOutOfDoorway()
        {
            var a = Belief(); var b = Belief("B"); var p = new Vector2(.1f, 1.1f);
            for (int i = 0; i < 10; i++) See(a, i * .1, p, true);
            a.Uncertainty = new PoseUncertainty { positionStdM = .3f, yawStdRad = .05f };
            b.Receive(BeliefPacket(a, 1, 1), new BrainPose()); See(b, 1.1, p, false, true);
            Assert.That(b.Export(1.1).occupancyProbability[b.Cell(p)], Is.LessThan(.5));
            Assert.That(new BrainConfig().team_map_planner, Is.False);
        }

        [Test]
        public void TeamMapCorrectionMovesPreviouslySharedEvidenceWithoutAnotherObservation()
        {
            var a = Belief(); var b = Belief("B"); var p = new Vector2(.1f, 1.1f);
            See(a, 1, p, true); b.Receive(BeliefPacket(a, 1, 1), new BrainPose());
            var delta = new BrainPose { position = new Vector2(1, 0) }; a.ApplyCorrection(delta);
            b.Receive(BeliefPacket(a, 2, 2), new BrainPose()); var map = b.Export(2);
            Assert.That(map.sourceMask[b.Cell(p)], Is.Zero, "the old shared cell must disappear");
            Assert.That(map.occupancyProbability[b.Cell(p + Vector2.right)], Is.GreaterThan(.5));
        }

        [Test]
        public void TeamMapRadioCarriesPoseUncertaintyAndBoundsItsEvidenceBatch()
        {
            var a = Belief();
            a.BeginObservation(1);
            for (int i = 0; i < 800; i++) a.Observe(new Vector2(-3.9f + (i % 40) * .2f, -3.9f + (i / 40) * .2f), true, false);
            a.EndObservation();
            a.Uncertainty = new PoseUncertainty { positionStdM = .15f, yawStdRad = .03f };
            var m = BeliefPacket(a, 1, 2);
            Assert.That(m.PoseUncertainty.positionStdM, Is.EqualTo(.15f));
            Assert.That(m.TeamCells.Length, Is.LessThanOrEqualTo(BrainMessage.MaxTeamCells));
            Assert.That(m.EstimatedBytes, Is.LessThan(BrainMessage.MaxBytes));
        }

        [Test]
        public void HeadlessMotionErrorsAreOptInSeededAndApplyKnownHeadingAndBias()
        {
            var off = new HeadlessMotionErrors(null, 1);
            Assert.That(off.Gyro(1), Is.EqualTo(1)); Assert.That(off.InitialHeadingDeg, Is.Zero);
            var options = Newtonsoft.Json.Linq.JObject.Parse("{\"gyro_bias_dps\":0.2,\"gyro_noise_dps\":0,\"initial_heading_error_deg\":3}");
            var on = new HeadlessMotionErrors(options, 1);
            var motion = new DeadReckoning(.2f, 2, new EntryPose(0, 0, on.InitialHeadingDeg).pose);
            for (int i = 0; i <= 600; i++)
            { double t = i / 60.0; motion.Add(new OdometrySample(t, 0, 0, 0)); motion.Add(new ImuSample(t, Vector3.zero, new Vector3(0, -on.Gyro(0), 0))); motion.Advance(t); }
            Assert.That(motion.Pose.yawRad * Mathf.Rad2Deg, Is.EqualTo(-1).Within(.001));
            options["gyro_noise_dps"] = .1;
            var a = new HeadlessMotionErrors(options, 7); var b = new HeadlessMotionErrors(options, 7);
            Assert.That(Enumerable.Range(0, 20).Select(i => a.Gyro(0)).ToArray(), Is.EqualTo(Enumerable.Range(0, 20).Select(i => b.Gyro(0)).ToArray()));
        }

        [Test]
        public void TeamMapLostBatchesReturnAndNewerAbsoluteBeliefsReplaceOldOnes()
        {
            var a = Belief(); var b = Belief("B");
            a.BeginObservation(1);
            for (int i = 0; i < 1200; i++) a.Observe(new Vector2(-3.89f + (i % 40) * .2f, -3.89f + (i / 40) * .2f), true, false);
            a.EndObservation();
            for (int i = 1; i <= 12; i++) { var m = BeliefPacket(a, i, i + 1); if (i != 2 && i != 4) b.Receive(m, new BrainPose()); }
            Assert.That(b.Export(14).sourceMask.Count(v => v != 0), Is.EqualTo(a.Export(14).sourceMask.Count(v => v != 0)));
            var p = new Vector2(.1f, .1f);
            for (int i = 0; i < 20; i++) See(a, 15 + i, p, false, true);
            for (int i = 13; i <= 20; i++) b.Receive(BeliefPacket(a, i, 40 + i), new BrainPose());
            Assert.That(b.Export(60).occupancyProbability[b.Cell(p)], Is.LessThan(.5));
            Assert.That(BeliefPacket(a, 21, 61).EstimatedBytes, Is.LessThan(3000), "confidence batch is compressed before radio transport");
        }

        [Test]
        public void TeamMapRobotCorrectionsAndScanMatchingUpdateTheSharedTransform()
        {
            var b = new RobotBrain(Calibration(), Settings(), "A", 0, new EntryPose(0, -1, 0));
            for (int tick = 0; tick <= 40; tick++)
            {
                double t = tick / 20.0; Motion(b, t, .36f);
                if (tick % 2 == 0) b.OnLidar(RoomScan(t, new Vector2(0, -1 + .3f * (float)t), 0));
                b.Step(t, .05f);
            }
            var before = b.BuildMessage(2);
            Assert.That(before.TeamCells, Is.Not.Empty, "real LiDAR integration feeds the confidence map");
            Assert.That(b.ExportTeamMap(2).searchedSourceMask.All(v => v == 0), Is.True, "LiDAR and the initial footprint cannot search floor");
            Assert.That(before.MapCorrection.position.magnitude, Is.GreaterThan(.01f), "own-map scan corrections must move shared evidence");
            var delta = new BrainPose { position = new Vector2(1, .5f), yawRad = .1f }; b.ApplyCorrection(delta);
            var after = b.BuildMessage(4);
            Assert.That(Vector2.Distance(after.MapCorrection.position, delta.Transform(before.MapCorrection.position.x, before.MapCorrection.position.y)), Is.LessThan(.0001));
            Assert.That(after.MapCorrection.yawRad, Is.EqualTo(before.MapCorrection.yawRad + .1f).Within(.0001));
        }

        [Test]
        public void TeamMapCorrectionsRetainEvidenceWithoutGridAllocations()
        {
            var a = Belief(); var p = new Vector2(.1f, 1.1f); See(a, 1, p, true);
            var forward = new BrainPose { position = new Vector2(.01f, 0) }; var back = new BrainPose { position = new Vector2(-.01f, 0) };
            a.ApplyCorrection(forward); a.ApplyCorrection(back);
            var clock = System.Diagnostics.Stopwatch.StartNew();
#if !UNITY_5_3_OR_NEWER
            long allocated = GC.GetAllocatedBytesForCurrentThread();
#endif
            for (int i = 0; i < 10000; i++) { a.ApplyCorrection(forward); a.ApplyCorrection(back); }
            clock.Stop();
#if !UNITY_5_3_OR_NEWER
            allocated = GC.GetAllocatedBytesForCurrentThread() - allocated;
            Assert.That(allocated, Is.Zero, "pose correction must not allocate grid buffers");
            TestContext.WriteLine("20000 corrections allocated " + allocated + " bytes.");
#endif
            TestContext.WriteLine("20000 corrections took " + clock.Elapsed.TotalMilliseconds + " ms on the standalone runtime.");
            Assert.That(a.Export(2).occupancyProbability[a.Cell(p)], Is.GreaterThan(.5));
            string fixture = Environment.GetEnvironmentVariable("SMOKE_M1_EXPORT_FIXTURE");
            if (!string.IsNullOrEmpty(fixture)) a.Export(2).Write(fixture);
        }

        [Test]
        public void TeamMapCameraFloorIntegrationRecordsSearchWithoutCountingWallHits()
        {
            var s = Calibration(); var c = Settings(); var b = new RobotBrain(s, c, "A", 0);
            Assert.That(b.ExportTeamMap(0).searchedSourceMask.All(v => v == 0), Is.True);
            var depth = new ushort[s.camera.width * s.camera.height];
            float pitch = s.camera.mount.pitch_down_deg * Mathf.Deg2Rad;
            for (int v = 0; v < s.camera.height; v++) for (int u = 0; u < s.camera.width; u++)
            {
                float down = (v - s.camera.cy) / s.camera.fy;
                float denom = Mathf.Sin(pitch) + down * Mathf.Cos(pitch);
                float range = denom > 0 ? s.camera.mount.height_m / denom : 100;
                if (range <= s.depth.max_m) depth[v * s.camera.width + u] = (ushort)Math.Round(range * 1000);
            }
            b.OnRgbd(new RgbdFrame(0, s.camera.width, s.camera.height, new byte[s.camera.width * s.camera.height * 3], depth));
            Motion(b, 0); b.Step(0, 1f / 60);
            var export = b.ExportTeamMap(0);
            Assert.That(export.searchedSourceMask.Count(v => v != 0), Is.GreaterThan(10));
            Assert.That(export.searchedSourceMask.Where(v => v != 0).All(v => v == 1), Is.True);
        }
    }
}
