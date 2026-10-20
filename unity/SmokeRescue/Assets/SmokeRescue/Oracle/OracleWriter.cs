using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using Newtonsoft.Json;
using SmokeRescue.World;
using UnityEngine;

namespace SmokeRescue.Oracle
{
    /// <summary>
    /// The only Unity code that writes ground truth. Everything goes to the recording's oracle/ folder,
    /// in the house frame (Unity world with y flipped, INTERFACES 1.1). Robot-side assemblies never reference this.
    /// </summary>
    public sealed class OracleWriter
    {
        readonly string dir;
        readonly Dictionary<string, StringBuilder> trajectories = new Dictionary<string, StringBuilder>();

        public OracleWriter(string recordingDir)
        {
            dir = Path.Combine(recordingDir, "oracle");
            Directory.CreateDirectory(dir);
        }

        /// <summary>Camera optical pose in the house frame, camera-to-world, TUM line.</summary>
        public void CameraPose(string robotId, double t, Vector3 unityPos, Quaternion unityRot)
        {
            if (!trajectories.TryGetValue(robotId, out var sb))
                trajectories[robotId] = sb = new StringBuilder("# ground truth camera optical pose, house frame, camera-to-world\n# t tx ty tz qx qy qz qw\n");
            var p = new Vector3(unityPos.x, -unityPos.y, unityPos.z);
            var q = unityRot.normalized;
            q = new Quaternion(-q.x, q.y, -q.z, q.w);
            sb.Append(string.Format(CultureInfo.InvariantCulture, "{0:F6} {1:F6} {2:F6} {3:F6} {4:F6} {5:F6} {6:F6} {7:F6}\n", t, p.x, p.y, p.z, q.x, q.y, q.z, q.w));
        }

        readonly Dictionary<string, StringBuilder> heat = new Dictionary<string, StringBuilder>();

        /// <summary>True air temperature at a robot's body and the hottest floor surface within 1 m, for scoring heat
        /// exposure. Never available to the robots.</summary>
        public void Heat(string robotId, double t, float airC, float hottestSurfaceC)
        {
            if (!heat.TryGetValue(robotId, out var sb))
                heat[robotId] = sb = new StringBuilder("# ground truth heat at the robot body, house frame\n# t air_c hottest_surface_within_1m_c\n");
            sb.Append(string.Format(CultureInfo.InvariantCulture, "{0:F6} {1:F2} {2:F2}\n", t, airC, hottestSurfaceC));
        }

        /// <summary>Fire timeline from the fire simulation: collapses and room flashover times.</summary>
        public void FireEvents(object events) =>
            File.WriteAllText(Path.Combine(dir, "fire_events.json"), JsonConvert.SerializeObject(events, Formatting.Indented));

        public void Finish(Scenario scenario, string floorplanPath)
        {
            foreach (var kv in heat)
                File.WriteAllText(Path.Combine(dir, $"{kv.Key}_heat.txt"), kv.Value.ToString());
            foreach (var kv in trajectories)
                File.WriteAllText(Path.Combine(dir, $"{kv.Key}_groundtruth.txt"), kv.Value.ToString());
            var survivors = new List<object>();
            foreach (var s in scenario.survivors)
                survivors.Add(new { s.id, s.room, position_house = new[] { s.at[0], 0f, s.at[1] }, s.yaw, s.pose });
            File.WriteAllText(Path.Combine(dir, "survivors.json"), JsonConvert.SerializeObject(survivors, Formatting.Indented));
            File.Copy(floorplanPath, Path.Combine(dir, "floorplan_used.json"), true);
        }

        /// <summary>
        /// Samples the visible surfaces of the given roots (house and furniture) into house_truth.ply, house frame,
        /// area-weighted at about `pointsPerM2`, for the evaluator's map accuracy metric.
        /// </summary>
        public int ExportHouseTruth(IEnumerable<GameObject> roots, int layerToSkip, float pointsPerM2 = 400f, int seed = 1)
        {
            var rng = new System.Random(seed);
            var pts = new List<Vector3>();
            foreach (var root in roots)
            {
                if (root == null) continue;
                foreach (var mf in root.GetComponentsInChildren<MeshFilter>())
                {
                    if (mf.gameObject.layer == layerToSkip || mf.sharedMesh == null || !mf.sharedMesh.isReadable) continue;
                    var m = mf.sharedMesh;
                    var v = m.vertices;
                    var tr = m.triangles;
                    var l2w = mf.transform.localToWorldMatrix;
                    for (int i = 0; i < tr.Length; i += 3)
                    {
                        Vector3 a = l2w.MultiplyPoint3x4(v[tr[i]]), b = l2w.MultiplyPoint3x4(v[tr[i + 1]]), c = l2w.MultiplyPoint3x4(v[tr[i + 2]]);
                        float area = 0.5f * Vector3.Cross(b - a, c - a).magnitude;
                        double expected = area * pointsPerM2;
                        int n = (int)expected + (rng.NextDouble() < expected - (int)expected ? 1 : 0);
                        for (int k = 0; k < n; k++)
                        {
                            double r1 = rng.NextDouble(), r2 = rng.NextDouble();
                            if (r1 + r2 > 1) { r1 = 1 - r1; r2 = 1 - r2; }
                            var p = a + (float)r1 * (b - a) + (float)r2 * (c - a);
                            pts.Add(new Vector3(p.x, -p.y, p.z));
                        }
                    }
                }
            }
            using (var w = new BinaryWriter(File.Create(Path.Combine(dir, "house_truth.ply"))))
            {
                w.Write(Encoding.ASCII.GetBytes($"ply\nformat binary_little_endian 1.0\ncomment Smoke Rescue ground truth surfaces, house frame (x east, y down, z north), metres\nelement vertex {pts.Count}\nproperty float x\nproperty float y\nproperty float z\nend_header\n"));
                foreach (var p in pts) { w.Write(p.x); w.Write(p.y); w.Write(p.z); }
            }
            return pts.Count;
        }
    }
}
