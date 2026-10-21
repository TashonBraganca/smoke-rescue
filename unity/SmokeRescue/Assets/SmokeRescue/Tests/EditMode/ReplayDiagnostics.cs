using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using SmokeRescue.Brain;
using UnityEngine;

namespace SmokeRescue.Tests.EditMode
{
    /// <summary>Replays a robot's recorded brain map (5 s snapshots), its own hot cells and pose into a fresh grid with
    /// the real config, and reports what the planner and the path check see there. Diagnostic, run by hand: it reads a
    /// recording that is not in the repository.</summary>
    public sealed class ReplayDiagnostics
    {
        const BindingFlags Hidden = BindingFlags.NonPublic | BindingFlags.Instance;

        [Test, Explicit("diagnostic replay of a recorded freeze; needs the recording")]
        public void RobotBFreezeInHeldOutSeed914()
        {
            string dir = Path.GetFullPath(Path.Combine(Application.dataPath, "..", "..", "..", "data", "recordings", "unity_house_heldout_seed914", "robot_B"));
            var c = BrainConfig.Load();
            var states = File.ReadAllLines(Path.Combine(dir, "brain_state.jsonl")).Select(JObject.Parse).ToList();
            var report = new System.Text.StringBuilder();
            foreach (double snap in new[] { 145.0, 150.0, 155.0, 160.0 })
            foreach (float clear in new[] { c.emergency_fire_clearance_m, c.fire_clearance_m })
            {
                var g = new OccupancyGrid(c);
                var tex = new Texture2D(2, 2); tex.LoadImage(File.ReadAllBytes(Path.Combine(dir, "brain_map", $"{snap:F6}.png")));
                var px = tex.GetPixels32();
                Assert.That(tex.width, Is.EqualTo(g.Size), "snapshot matches the grid");
                for (int i = 0; i < px.Length; i++)
                {
                    byte v = px[i].r;
                    if (v == 0) { g.ObserveOccupied(i); g.ObserveOccupied(i); }
                    else if (v == 255) g.ObserveFree(i);
                }
                var s = states.OrderBy(x => Math.Abs((double)x["t"] - snap)).First();
                foreach (var f in (JArray)s["fires"]) g.ObserveHeat(g.Cell(new Vector2((float)f[0], (float)f[1])), 500, snap);
                g.DecayTo(snap);
                typeof(OccupancyGrid).GetProperty("FireClearance", Hidden).SetValue(g, clear);
                var planner = new GridPlanner(g, c);
                bool Walk(int i) => (bool)typeof(OccupancyGrid).GetMethod("Walkable", Hidden).Invoke(g, new object[] { i });
                bool Pass(int i) => (bool)typeof(OccupancyGrid).GetMethod("Passable", Hidden).Invoke(g, new object[] { i });
                float Obs(int i) => (float)typeof(OccupancyGrid).GetMethod("ObstacleDistance", Hidden).Invoke(g, new object[] { i });
                float Fire(int i) => (float)typeof(OccupancyGrid).GetMethod("FireDistance", Hidden).Invoke(g, new object[] { i });
                var pos = new Vector2((float)s["x"], (float)s["y"]); int start = g.Cell(pos);
                var route = planner.Plan(pos, Vector2.zero);
                var leadArgs = new object[] { pos, 0, null };
                bool lead = (bool)typeof(GridPlanner).GetMethod("Lead", Hidden).Invoke(planner, leadArgs);
                int firstUnsafe = -1;
                for (int k = 0; k < route.Count && firstUnsafe < 0; k++) if (!planner.PathSafe(route.Take(k + 1).ToList(), 0)) firstUnsafe = k;
                int home = g.Cell(Vector2.zero);
                report.AppendLine($"t={snap} clearance={clear} pose=({pos.x:F2},{pos.y:F2}) route={route.Count} safe={(route.Count > 0 && planner.PathSafe(route, 0))} firstUnsafe={firstUnsafe} " +
                    $"start: walkable={Walk(start)} passable={Pass(start)} obstacle={Obs(start):F2} fire={Fire(start):F2}; lead={lead} ({((List<int>)leadArgs[2])?.Count} cells); " +
                    $"home walkable={Walk(home)} obstacle={Obs(home):F2}");
                if (firstUnsafe >= 0)
                {
                    var p = route[firstUnsafe]; int cell = g.Cell(p);
                    var prev = firstUnsafe > 0 ? route[firstUnsafe - 1] : pos;
                    report.AppendLine($"   first unsafe point {firstUnsafe} ({p.x:F2},{p.y:F2}) from ({prev.x:F2},{prev.y:F2}): walkable={Walk(cell)} passable={Pass(cell)} obstacle={Obs(cell):F2} fire={Fire(cell):F2} occupancy={g.Occupancy(cell)}");
                }
                // The neighbourhood of the robot: walkable (W), passable only (p), free but tight (.), occupied (#), unknown ( ).
                var sb = new System.Text.StringBuilder();
                for (int dy = 6; dy >= -6; dy--)
                {
                    for (int dx = -6; dx <= 6; dx++)
                    {
                        int x = start % g.Size + dx, y = start / g.Size + dy, i = y * g.Size + x;
                        char ch = dx == 0 && dy == 0 ? 'R' : g.Occupancy(i) == 255 ? '#' : g.Occupancy(i) == 0 ? ' ' : Walk(i) ? 'W' : Pass(i) ? 'p' : '.';
                        sb.Append(ch);
                    }
                    sb.Append('|');
                }
                if (clear == c.emergency_fire_clearance_m) report.AppendLine("   around the robot (north up, 0.1 m cells): " + sb);
            }
            Assert.Fail(report.ToString());
        }
    }
}
