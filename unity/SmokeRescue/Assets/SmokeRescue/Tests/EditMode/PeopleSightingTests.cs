using System.Collections.Generic;
using System.Collections;
using System.Linq;
using NUnit.Framework;
using SmokeRescue.Brain;
using UnityEngine;

namespace SmokeRescue.Tests.EditMode
{
    public sealed partial class BrainTests
    {
        static bool HasCandidate(RobotBrain b) => ((IEnumerable)typeof(RobotBrain).GetField("tracks", Private).GetValue(b)).Cast<object>()
            .Any(k => (bool)k.GetType().GetField("candidate").GetValue(k));

        [Test]
        public void OneStrongColourSightingMakesACandidateAndWeakerOnesStillNeedTwo()
        {
            // Seed 934, robot B: a person on a bed in view for 1.2 s at 0.54 to 0.86; the once-a-second scan caught one frame and
            // the two-sightings rule dropped her.
            var c = AuctionSettings(); c.person_rgb_strong_confidence = .7f;
            var b = new RobotBrain(Calibration(), c, "robot_B", 0); Room(b, -6, 6, -6, 6);
            b.OnPersonSighting(1, 0, 2, .6f);
            Assert.That(HasCandidate(b), Is.False, "control: one sighting under the strong level is not yet a candidate");
            b.OnPersonSighting(1.1, 30, 3, .82f);
            Assert.That(HasCandidate(b), Is.True, "one sighting at 0.82 is a candidate");
            c.person_rgb_strong_confidence = 1.01f; var off = new RobotBrain(Calibration(), c, "robot_B", 0); Room(off, -6, 6, -6, 6);
            off.OnPersonSighting(1.1, 30, 3, .82f);
            Assert.That(HasCandidate(off), Is.False, "control: switched off, one sighting is never a candidate");
        }

        [Test]
        public void AnySightingStopsTheTurnAndHoldsTheCameraOnItWithAFasterScan()
        {
            var c = AuctionSettings(); c.person_glance = true;
            var b = new RobotBrain(Calibration(), c, "robot_B", 0); Room(b, -6, 6, -6, 6);
            var step = typeof(RobotBrain).GetMethod("GlanceStep", Private);
            BrainCommand? Glance(double t) => (BrainCommand?)step.Invoke(b, new object[] { t, 1f / 60 });
            Assert.That(b.WantsPeopleBoost(0), Is.False, "control: nothing seen, normal scan rate");
            b.OnPersonSighting(1, 60, 2, .3f);   // weak hit to the right, below the candidate level
            Assert.That(HasCandidate(b), Is.False, "a weak hit is no candidate");
            Assert.That(b.WantsPeopleBoost(1.5), Is.True, "but the colour scan speeds up");
            var cmd = Glance(1.5);
            Assert.That(cmd.HasValue, Is.True, "and the robot holds on it");
            Assert.That(cmd.Value.forwardMps, Is.EqualTo(0).Within(.05f), "standing");
            Assert.That(cmd.Value.yawRateDeg, Is.LessThan(0), "turning right, toward the sighting");
            Assert.That(Glance(3.5).HasValue, Is.False, "the hold ends after person_glance_s");
            b.OnPersonSighting(5, 60, 2, .3f);
            Assert.That(Glance(5.1).HasValue, Is.False, "the same spot does not stop the robot again within the repeat window");
            c.person_glance = false; var off = new RobotBrain(Calibration(), c, "robot_B", 0); Room(off, -6, 6, -6, 6);
            off.OnPersonSighting(1, 60, 2, .3f);
            Assert.That(off.WantsPeopleBoost(1.5) || ((BrainCommand?)step.Invoke(off, new object[] { 1.5, 1f / 60 })).HasValue, Is.False,
                "control: switched off, a weak hit changes nothing");
        }

        [Test]
        public void ARejectedWarmAreaIsNotRaisedAgainFromANewSideButAColourPersonThereStillIs()
        {
            // Seed 934, robot A in the study: 12 warm patches beside a desk lamp and a globe rejected, each a 10 to 27 s agent step,
            // and they kept coming back from new sides or as new tracks when the blob centre moved past survivor_merge_m.
            bool Raised(bool memory, Vector2 again, out RobotBrain brain)
            {
                var c = AuctionSettings(); c.reject_area_memory = memory; c.person_rgb_strong_confidence = .7f;
                var b = brain = new RobotBrain(Calibration(), c, "robot_A", 0); Room(b, -6, 6, -6, 6);
                var g = Map(b);   // a mapped wall along the north side, y 5.85 to 6.05
                for (int i = 0; i < g.Count; i++) { var p = g.Centre(i); if (Mathf.Abs(p.x) <= 6 && p.y >= 5.85f && p.y <= 6.05f) g.ObserveOccupied(i); }
                var observe = typeof(RobotBrain).GetMethod("ObservePeople", Private);
                void Frames(Vector2 at, double t0) { for (int k = 0; k < 4; k++) observe.Invoke(b, new object[] { t0 + k * .07, new List<PersonDetection> { new PersonDetection(at, .4f) } }); }
                var wall = new Vector2(2, 5.7f);   // warm patch on the room's north wall (y = 6)
                Frames(wall, 1);
                Assert.That(HasCandidate(b), Is.True, "control: the warm patch is raised the first time");
                b.AgentRejectCandidate(wall, 2, "e1");
                Assert.That(HasCandidate(b), Is.False);
                Frames(again, 5);
                return HasCandidate(b);
            }
            var shifted = new Vector2(3.2f, 5.7f);   // 1.2 m along the wall: past survivor_merge_m, so a new track
            Assert.That(Raised(false, shifted, out _), Is.True, "control: without the memory the shifted patch comes back as new");
            Assert.That(Raised(true, shifted, out var b2), Is.False, "with it, a warm patch within reject_area_m stays quiet");
            Assert.That(Raised(true, new Vector2(2, -3), out _), Is.True, "a warm patch far from any rejection is still raised");
            Assert.That(Raised(true, new Vector2(2.6f, 4.4f), out _), Is.True, "a warm shape on open floor near a rejected wall is still raised");
            b2.OnPersonSighting(6, 0, 3.5f, .85f);
            Assert.That(HasCandidate(b2), Is.True, "a colour person sighting in the rejected area is still a candidate");
        }

        [Test]
        public void BeforeDeclaringNothingLeftTheRobotRetriesFrontiersTheFireMarginSetAside()
        {
            // Seed 934, robot A: 21 frontiers set aside for the fire margin, then "no reachable unsearched floor" at 209 s with six
            // house rooms unsearched.
            bool Retry(bool on, out RobotBrain brain)
            {
                var c = AuctionSettings(); c.fire_cut_retry = on;
                var b = brain = new RobotBrain(Calibration(), c, "robot_A", 0); Room(b, -6, 6, -6, 6);
                var blacklist = typeof(RobotBrain).GetMethod("BlacklistGoal", Private);
                var reason = (string)typeof(RobotBrain).GetField("FireCutReason", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static).GetValue(null);
                blacklist.Invoke(b, new object[] { new Vector2(3, 3), 1.0, 0, 40.0, reason, 0f });
                blacklist.Invoke(b, new object[] { new Vector2(-3, 3), 1.0, 0, 40.0, "Repeated stalls near one location exclude this goal area until the retry time.", 0f });
                return (bool)typeof(RobotBrain).GetMethod("RetryFireCuts", Private).Invoke(b, new object[] { 5.0 });
            }
            bool Blocked(RobotBrain b, Vector2 p) => (bool)typeof(RobotBrain).GetMethod("GoalBlacklisted", Private).Invoke(b, new object[] { p, 6.0 });
            Assert.That(Retry(false, out var off), Is.False, "control: switched off, nothing is retried");
            Assert.That(Blocked(off, new Vector2(3, 3)), Is.True);
            Assert.That(Retry(true, out var on), Is.True, "with it, the fire-cut frontiers are offered again");
            Assert.That(Blocked(on, new Vector2(3, 3)), Is.False, "the fire-cut exclusion is gone");
            Assert.That(Blocked(on, new Vector2(-3, 3)), Is.True, "a stall exclusion stays");
            Assert.That((bool)typeof(RobotBrain).GetMethod("RetryFireCuts", Private).Invoke(on, new object[] { 10.0 }), Is.False, "at most once per fire_cut_retry_s");
        }

        [Test]
        public void APinnedRobotYieldsToItsPartnerByMovingStraightAwayWithoutTurning()
        {
            // Headless house 5003: two robots in contact at the entrance; every turn in place swung into the partner and was refused.
            var c = AuctionSettings(); c.pinned_watchdog = true; c.pinned_yield = true;
            var b = new RobotBrain(Calibration(), c, "robot_B", 0); Room(b, -6, 6, -6, 6);   // at the origin, facing +y
            var m = new RobotBrain(Calibration(), c, "robot_A", 0).BuildMessage(0); m.Sequence = 1;
            b.OnPeerMessage(Wire(m), .1);
            var contact = typeof(RobotBrain).GetMethod("PartnerInContact", Private);
            var args = new object[] { .2, null, null };
            Assert.That((bool)contact.Invoke(b, args), Is.True, "the partner's radio pose is within pinned_yield_m");
            Assert.That((string)args[2], Is.EqualTo("robot_A"));
            var step = typeof(RobotBrain).GetMethod("YieldStep", Private);
            void From(Vector2 p) { typeof(RobotBrain).GetField("yieldFrom", Private).SetValue(b, p); typeof(RobotBrain).GetField("yieldUntil", Private).SetValue(b, 10.0); }
            From(new Vector2(0, .5f));   // partner just ahead
            var back = (BrainCommand?)step.Invoke(b, new object[] { .3, 1f / 60 });
            Assert.That(back.HasValue && back.Value.forwardMps <= 0 && back.Value.yawRateDeg == 0, Is.True, "partner ahead: reverse straight back, no turn");
            From(new Vector2(0, -.5f));  // partner just behind
            var ahead = (BrainCommand?)step.Invoke(b, new object[] { .4, 1f / 60 });
            Assert.That(ahead.HasValue && ahead.Value.forwardMps >= 0 && ahead.Value.yawRateDeg == 0, Is.True, "partner behind: drive straight forward");
            From(new Vector2(0, 2));   // already clear
            Assert.That(((BrainCommand?)step.Invoke(b, new object[] { .5, 1f / 60 })).HasValue, Is.False, "clear of the partner: the yield ends");
        }

        [Test]
        public void AWalkBackToMeetThePartnerIsDroppedAfterItsTimeLimit()
        {
            // Seed 934, robot B: 30 s walking back to where it last heard robot A, which had moved on.
            bool StillMeeting(float cap)
            {
                var c = AuctionSettings(); c.rendezvous_walk_max_s = cap;
                var b = new RobotBrain(Calibration(), c, "robot_B", 0); Room(b, -6, 6, -6, 6);
                var m = new RobotBrain(Calibration(), c, "robot_A", 0).BuildMessage(0); m.Sequence = 1;
                b.OnPeerMessage(Wire(m), .1);   // heard once: contactAt is set and a peer exists
                var type = typeof(RobotBrain);
                type.GetField("meetGoal", Private).SetValue(b, (Vector2?)new Vector2(4, 4));
                type.GetField("meetSince", Private).SetValue(b, .1);
                type.GetField("meetWalkFrom", Private).SetValue(b, 50.0);
                type.GetField("meetArrived", Private).SetValue(b, double.NaN);
                type.GetMethod("RendezvousStep", Private).Invoke(b, new object[] { 61.0, 1f / 60 });
                return ((Vector2?)type.GetField("meetGoal", Private).GetValue(b)).HasValue;
            }
            Assert.That(StillMeeting(0), Is.True, "control: without a cap the walk back goes on after 11 s");
            Assert.That(StillMeeting(10), Is.False, "with a 10 s cap it is dropped and the search goes on");
        }
    }
}
