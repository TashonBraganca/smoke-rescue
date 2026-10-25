#!/usr/bin/env python3
"""Write the cloud engine's copy of HeadlessSim.cs with two request options added; unity/ itself is never edited.

    python3 tools/ryor/patch_headless.py <out.cs>

Options added (both off unless a request sets them, so a patched build runs every old request the same way):
  replay_every_s  every this many simulated seconds, log each robot's newly searched cells on its own brain map
                  (kind replay_map) and the fire's flame regions in the gate frame (kind replay_fire), then flush the
                  trace, so the replay can be cut into chunks while the run is still going. Viewer only.
  visibility_m    smoke: the world lets the camera see a person only within this range. The brain is not told.
  parallel_robots each 1/60 s step runs in three phases: odometry, IMU and air for every robot in order (the fire solver is
                  not thread-safe); then every robot's depth, LiDAR, sightings and brain step at once on its own thread; then
                  motion, contact and logging in robot order. Every robot's sensors therefore see the other robots where
                  they stood at the start of the step (serial: robots earlier in the list have already moved, by at most
                  one step, 1 cm). Still deterministic. Off, the step runs exactly as before; the sensing code is moved into
                  a method unchanged, and its log lines are written right after it, in the same order.
Every anchor must match exactly once, so an upstream change to HeadlessSim.cs stops the build instead of patching wrong.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "unity/SmokeRescue/Assets/SmokeRescue/Tests/EditMode/HeadlessSim.cs"

EDITS = [
    ("public AirTemperatureSensor sensor; public HeadlessMotionErrors motionError; }",
     "public AirTemperatureSensor sensor; public HeadlessMotionErrors motionError; public byte[] replaySeen;\n"
     "            public BrainCommand cmd; public readonly List<string> lines = new List<string>(); public readonly List<(string key, double t)> sights = new List<(string, double)>(); }"),
    ("        /// <summary>House (x, z) of a gate-frame point.</summary>\n",
     "        /// <summary>Gate-frame point of a house (x, z) point: the inverse of House.</summary>\n"
     "        Vector2 Gate(Vector2 h) { Vector2 f = new Vector2(Mathf.Sin(gateYaw), Mathf.Cos(gateYaw)), r = new Vector2(Mathf.Cos(gateYaw), -Mathf.Sin(gateYaw)), d = h - gateOrigin; return new Vector2(Vector2.Dot(d, r), Vector2.Dot(d, f)); }\n"
     "        /// <summary>House (x, z) of a gate-frame point.</summary>\n"),
    ("        SensorConfig cal; BrainConfig cfg;",
     "        SensorConfig cal; BrainConfig cfg; double replayEvery; float visibility; bool parallelRobots;"),
    ("            if (run[\"probes\"] != null)",
     "            replayEvery = (double?)run[\"replay_every_s\"] ?? 0; visibility = (float?)run[\"visibility_m\"] ?? float.MaxValue;\n"
     "            parallelRobots = (bool?)run[\"parallel_robots\"] ?? false;\n"
     "            if (run[\"probes\"] != null)"),
    ("range > cfg.person_rgb_max_range_m || below",
     "range > cfg.person_rgb_max_range_m || range > visibility || below"),
    ("                if (t + 1e-9 >= nextRadio)",
     "                if (replayEvery > 0 && k % Math.Max(1, (int)Math.Round(replayEvery / dt)) == 0) LogReplay(log, t, robots);\n"
     "                if (t + 1e-9 >= nextRadio)"),
    ("        string Run(string root, JObject run)",
     """        void LogReplay(StreamWriter log, double t, List<Body> robots)
        {
            foreach (var r in robots)
            {
                if (t < r.start) continue;
                var own = r.brain.Snapshot().ownSearched; var cells = new JArray();
                for (int c = 0; c < own.Length; c++) if (own[c] != 0 && (r.replaySeen == null || r.replaySeen[c] == 0)) cells.Add(c);
                r.replaySeen = own;
                log.WriteLine(new JObject { ["t"] = t, ["kind"] = "replay_map", ["robot"] = r.id, ["cells"] = cells }.ToString(Newtonsoft.Json.Formatting.None));
            }
            var flames = new JArray();
            if (heat != null)
                foreach (var f in heat.Flames((float)t))
                    if (f.radius > 0) { var g = Gate(new Vector2(f.centre.x, f.centre.z)); flames.Add(new JArray(Math.Round(g.x, 2), Math.Round(g.y, 2), Math.Round(f.radius, 2))); }
            log.WriteLine(new JObject { ["t"] = t, ["kind"] = "replay_fire", ["flames"] = flames }.ToString(Newtonsoft.Json.Formatting.None));
            log.Flush();
        }

        string Run(string root, JObject run)"""),
]


BASICS_START = "                    r.brain.OnOdometry(new OdometrySample(t, r.v, 0, r.w));\n"
SENSE_START = "                    renderClock.Start(); if (k % 4 == 0) r.brain.OnRgbd(Depth(t, r));\n"
STEP_END = "                    brainClock.Start(); var cmd = r.brain.Step(t, (float)dt); brainClock.Stop();\n"
STEP_POSE = "                    var stepPosition = r.p; float stepYaw = r.yaw;\n"
SIGHT_LOG = "if (!sighted.ContainsKey(pid + \"|\" + r.id)) sighted[pid + \"|\" + r.id] = t;"
LOOP_HEAD = "                foreach (var r in robots)\n                {\n                    if (t < r.start) continue;\n" + BASICS_START


def split_step(s):
    """Move the per-robot sensing out of the step loop into Basics() and SenseStep(), text unchanged apart from the log
    lines and first sightings, which SenseStep buffers on the body; the loop writes them right after the call."""
    for a in (BASICS_START, SENSE_START, STEP_END, STEP_POSE, LOOP_HEAD):
        if s.count(a) != 1:
            raise SystemExit(f"patch_headless: step anchor found {s.count(a)} times: {a[:70]!r}")
    s = s.replace(LOOP_HEAD,
        "                if (parallelRobots)\n"
        "                {\n"
        "                    var active = robots.Where(q => t >= q.start).ToList();\n"
        "                    foreach (var r in active) Basics(r, t, dt);\n"
        "                    System.Threading.Tasks.Parallel.ForEach(active, r => r.cmd = SenseStep(r, t, k, dt, robots, false));\n"
        "                }\n"
        "                foreach (var r in robots)\n                {\n                    if (t < r.start) continue;\n" + BASICS_START)
    if s.count("if (parallelRobots)\n") != 1:
        raise SystemExit("patch_headless: parallel phase not inserted")
    i0, i1, i2 = s.index(BASICS_START), s.index(SENSE_START), s.index(STEP_END)
    basics, sense = s[i0:i1], s[i1:i2]
    if sense.count(STEP_POSE) != 1 or sense.count(SIGHT_LOG) != 1 or sense.count("log.WriteLine(") != 1:
        raise SystemExit("patch_headless: sensing block changed shape")
    sense = sense.replace(STEP_POSE, "").replace(SIGHT_LOG, "r.sights.Add((pid + \"|\" + r.id, t));").replace("log.WriteLine(", "r.lines.Add(")
    sense_par = sense.replace("renderClock.Start(); ", "").replace("renderClock.Stop();\n", "\n")
    loop = (STEP_POSE +
            "                    if (!parallelRobots) { Basics(r, t, dt); r.cmd = SenseStep(r, t, k, dt, robots, true); }\n"
            "                    var cmd = r.cmd;\n"
            "                    foreach (var line in r.lines) log.WriteLine(line);\n"
            "                    foreach (var (key, at) in r.sights) if (!sighted.ContainsKey(key)) sighted[key] = at;\n"
            "                    r.lines.Clear(); r.sights.Clear();\n")
    s = s[:i0] + loop + s[i2 + len(STEP_END):]
    methods = ("        void Basics(Body r, double t, double dt)\n        {\n" + basics + "        }\n\n"
               "        BrainCommand SenseStep(Body r, double t, int k, double dt, List<Body> robots, bool clocks)\n        {\n"
               "            if (clocks)\n            {\n" + sense +
               "                brainClock.Start(); var c = r.brain.Step(t, (float)dt); brainClock.Stop(); return c;\n            }\n"
               + sense_par + "            return r.brain.Step(t, (float)dt);\n        }\n\n")
    anchor = "        void LogReplay("
    return s.replace(anchor, methods + anchor)


def main():
    s = SRC.read_text()
    for old, new in EDITS:
        n = s.count(old)
        if n != 1:
            raise SystemExit(f"patch_headless: anchor found {n} times (need 1): {old[:80]!r}")
        s = s.replace(old, new)
    s = split_step(s)
    # Optional agent bridge. It reads only detached brain snapshots and public brain order methods.
    # A single simulation thread applies orders before sensing; worker/model threads never touch a brain.
    bridge = (ROOT / 'tools/ryor/AgentBridge.cs').read_text()
    s = s.replace('        void LogReplay(', bridge + '\n        void LogReplay(')
    s = s.replace('        SensorConfig cal;', '        string agentDir;\n        SensorConfig cal;')
    anchor = '            var log = new StreamWriter(Path.Combine(outDir, "trace.jsonl"));'
    if s.count(anchor) != 1:
        raise SystemExit('patch_headless: agent directory anchor changed')
    s = s.replace(anchor, '            agentDir = (bool?)run["terra"] == true ? outDir : null;\n' + anchor)
    anchor = '                if (parallelRobots)\n'
    s = s.replace(anchor, '                if (agentDir != null) AgentTick(log, robots, t, k);\n' + anchor)
    out = Path(sys.argv[1])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("// GENERATED by tools/ryor/patch_headless.py from HeadlessSim.cs. Do not edit.\n" + s)
    print(f"patched {SRC.relative_to(ROOT)} -> {out}")


if __name__ == "__main__":
    main()
