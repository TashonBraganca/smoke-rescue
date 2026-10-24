using System;
using System.Collections.Generic;
using NUnit.Framework;
using SmokeRescue.Brain;
using UnityEngine;

namespace SmokeRescue.Tests.EditMode
{
    public sealed partial class BrainTests
    {
        /// <summary>Whether StartsViewingStop holds the robot at (0, 0), 3.5 m from a candidate at (0, 3.5), inside the 3 to 4 m
        /// viewing band. wall: a mapped wall across the line at y = 1.5; atGoal: the robot stands at the end of an approach
        /// path.</summary>
        static bool Stops(bool sightLine, bool wall, bool atGoal = false)
        {
            var c = Settings(); c.grid_size_m = 12; c.survivor_view_sight_line = sightLine; c.survivor_standoff_m = 3.3f; c.survivor_stop_min_m = 3; c.survivor_stop_max_m = 4;   // shipped band
            var b = new RobotBrain(Calibration(), c, "A", 0, new EntryPose(0, 0, 0)); Room(b, -4, 4, -4, 5);
            var g = Map(b);
            if (wall) for (int i = 0; i < g.Count; i++) { var p = g.Centre(i); if (Mathf.Abs(p.y - 1.55f) < .06f && Mathf.Abs(p.x) < 2) g.ObserveOccupied(i); }
            var track = Activator.CreateInstance(typeof(RobotBrain).GetNestedType("Track", Private | System.Reflection.BindingFlags.Public));
            track.GetType().GetField("position").SetValue(track, new Vector2(0, 3.5f));
            typeof(RobotBrain).GetField("active", Private).SetValue(b, track);
            if (atGoal) typeof(RobotBrain).GetMethod("SetPath", Private).Invoke(b, new object[] { new List<Vector2> { new Vector2(0, .05f) }, false });
            return (bool)typeof(RobotBrain).GetMethod("StartsViewingStop", Private).Invoke(b, new object[] { 3.5f });
        }

        /// <summary>survivor_view_sight_line. Headless house 5022 (23 Oct final), robot A stopped 4.00 m from s1 with a wall between
        /// them and stood 35.5 s until the approach limit; s1 was never confirmed. With the switch the stop needs a mapped clear
        /// line. Controls: the switch off stops behind the wall as before; no wall stops; at the approach path's end it stops.</summary>
        [Test]
        public void TheViewingStopWaitsForAClearLineToThePerson()
        {
            Assert.That(Stops(false, true), Is.True, "control: the range alone stops the robot behind the wall");
            Assert.That(Stops(true, true), Is.False, "a wall on the line: keep walking the approach");
            Assert.That(Stops(true, false), Is.True, "control: a clear line stops");
            Assert.That(Stops(true, true, atGoal: true), Is.True, "control: at the planned viewpoint it stops");
        }
        /// <summary>survivor_close_view_min_m. Headless house 5027: s5 lay in a 3.1 x 3.0 m bathroom; no standable 3 to 4 m spot had
        /// a clear line to her and she was never confirmed. Here only floor within 2.9 m of the person is mapped, so the band has
        /// no spot. With the closer view the approach ends 2.0 to 3.0 m away with a clear line and the robot stops there to view.
        /// Controls: off, there is no approach (as before); a robot 2.5 m away that is not at its approach's end does not stop.</summary>
        [Test]
        public void ASmallRoomIsViewedFromCloserWhenTheViewingBandHasNoClearSpot()
        {
            var person = new Vector2(0, 4);
            (RobotBrain b, GridPlanner planner) Brain(float closeMin)
            {
                var c = Settings(); c.survivor_standoff_m = 3.3f; c.survivor_stop_min_m = 3; c.survivor_stop_max_m = 4; c.survivor_close_view_min_m = closeMin;
                var b = new RobotBrain(Calibration(), c, "A", 0); var g = Map(b);
                for (int i = 0; i < g.Count; i++) { var p = g.Centre(i); if (Vector2.Distance(p, person) < 2.9f && Mathf.Abs(p.x) < 4) g.ObserveFree(i); }
                return (b, (GridPlanner)typeof(RobotBrain).GetField("planner", Private).GetValue(b));
            }
            var from = new Vector2(0, 1.4f);
            Assert.That(Brain(0).planner.ApproachPath(from, person), Is.Empty, "control: the band alone has no spot");
            var on = Brain(2);
            var path = on.planner.ApproachPath(from, person);
            Assert.That(path, Is.Not.Empty, "a closer viewpoint");
            float range = Vector2.Distance(path[path.Count - 1], person);
            Assert.That(range, Is.InRange(2f, 3f), $"viewed from closer than the band, got {range}");
            // At the approach's end the closer range is a viewing stop; 2.5 m away without having arrived it is not.
            var track = Activator.CreateInstance(typeof(RobotBrain).GetNestedType("Track", Private | System.Reflection.BindingFlags.Public));
            track.GetType().GetField("position").SetValue(track, person);
            typeof(RobotBrain).GetField("active", Private).SetValue(on.b, track);
            var stops = typeof(RobotBrain).GetMethod("StartsViewingStop", Private);
            Assert.That(stops.Invoke(on.b, new object[] { 2.5f }), Is.False, "control: not at the approach's end");
            typeof(RobotBrain).GetMethod("SetPath", Private).Invoke(on.b, new object[] { new List<Vector2> { new Vector2(0, .05f) }, false });
            Assert.That(stops.Invoke(on.b, new object[] { 2.5f }), Is.True, "at the end of a closer approach it views");
        }
    }
}
