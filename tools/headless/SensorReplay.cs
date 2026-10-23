// Evaluator-side replay. RobotBrain receives recorded sensors and nominal entry poses only.
using System;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text.RegularExpressions;
using Newtonsoft.Json.Linq;
using SmokeRescue.Brain;
using SmokeRescue.Sensors;
using UnityEngine;

static class SensorReplay
{
    static readonly CultureInfo Inv = CultureInfo.InvariantCulture;
    static double[][] Table(string path) => File.ReadLines(path).Where(l => l.Length > 0 && l[0] != '#')
        .Select(l => l.Split((char[])null, StringSplitOptions.RemoveEmptyEntries).Select(s => double.Parse(s, Inv)).ToArray()).ToArray();

    public static void Run(JObject request)
    {
        string dir = (string)request["robot_dir"], output = (string)request["out"];
        var first = JObject.Parse(File.ReadLines(Path.Combine(dir, "brain_state.jsonl")).First());
        var cal = SensorConfig.Load(); var cfg = BrainConfig.Load();
        cfg.frames_shared_at_start = true; cfg.mission_time_limit_s = 86400;
        if (request["scan_yaw_gain"] != null) cfg.scan_yaw_gain = (float)request["scan_yaw_gain"];
        var entry = new EntryPose((float)first["x"], (float)first["y"], -(float)first["yaw_rad"] * Mathf.Rad2Deg);
        var brain = new RobotBrain(cal, cfg, new DirectoryInfo(dir).Name, 934, entry);
        var motion = (DeadReckoning)typeof(RobotBrain).GetField("motion", BindingFlags.NonPublic | BindingFlags.Instance).GetValue(brain);
        var odom = Table(Path.Combine(dir, "odom.txt")); var imu = Table(Path.Combine(dir, "imu.txt"));
        bool useLidar = (bool?)request["lidar"] ?? true;
        var scans = useLidar ? Table(Path.Combine(dir, "lidar.txt")) : Array.Empty<double[]>();
        string header = string.Join("\n", File.ReadLines(Path.Combine(dir, "lidar.txt")).TakeWhile(l => l.StartsWith("#")));
        float Field(string name) => float.Parse(Regex.Match(header, @"\b" + name + @"\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)").Groups[1].Value, Inv);
        int oi = 0, ii = 0, li = 0; double duration = (double?)request["duration_s"] ?? imu.Last()[0];
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output)));
        using (var log = new StreamWriter(output))
        for (int tick = 0; tick <= (int)Math.Round(duration * 60); tick++)
        {
            double t = tick / 60.0;
            while (oi < odom.Length && odom[oi][0] <= t + 1e-7)
            { var s = odom[oi++]; brain.OnOdometry(new OdometrySample(Math.Min(s[0], t), (float)s[1], (float)s[2], (float)s[3])); }
            while (ii < imu.Length && imu[ii][0] <= t + 1e-7)
            { var s = imu[ii++]; brain.OnImu(new ImuSample(Math.Min(s[0], t), Vector3.zero, new Vector3(0, (float)s[5], 0))); }
            while (li < scans.Length && scans[li][0] <= t + 1e-7)
            {
                var s = scans[li++]; var ranges = s.Skip(1).Select(v => v > 0 ? (float)(v * .001) : float.PositiveInfinity).ToArray();
                brain.OnLidar(new LidarScan(Math.Min(s[0], t), Field("angle_min_rad"), Field("increment_rad"), Field("min_m"), Field("max_m"), Field("mount_forward_m"), Field("mount_height_m"), ranges));
            }
            brain.Step(t, 1f / 60);
            brain.DrainEvents();
            if (tick % 30 == 0)
            {
                var p = motion.Pose;
                log.WriteLine(new JObject { ["t"] = t, ["x"] = p.position.x, ["y"] = p.position.y, ["yaw_rad"] = p.yawRad }.ToString(Newtonsoft.Json.Formatting.None));
            }
        }
        Console.WriteLine(output);
    }
}
