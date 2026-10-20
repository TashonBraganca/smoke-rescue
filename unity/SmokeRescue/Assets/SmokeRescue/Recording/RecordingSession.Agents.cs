using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using SmokeRescue.Radio;
using SmokeRescue.Robots;
using SmokeRescue.Sensors;
using SmokeRescue.World;
using UnityEngine;
using Debug = UnityEngine.Debug;

namespace SmokeRescue.Recording
{
    /// <summary>
    /// The agent layer in a recording. On brain events worth a decision (a warm candidate, poor visibility, rising
    /// heat, being stuck) and every 20 s, the recorder pauses the sim clock, flushes the robot's frames to disk and
    /// asks the agent server (brain/brain/agent/server.py) for one decision step. Orders come back cited to OpenCV
    /// evidence and are applied: brain orders, torch, arm, radio text. Agent texts and commander orders travel over
    /// the same RadioModel as everything else, so walls and distance delay or drop them. The commander stands at the
    /// command post outside and decides every 30 s from whatever reports have reached it.
    /// Doors and debris run in every recording, with or without agents, so both see the same world.
    /// </summary>
    public static partial class RecordingSession
    {
        /// <summary>Agent server URL, for example http://localhost:8080. Null records without agents.</summary>
        public static string AgentUrl;
        static readonly HttpClient Http = new HttpClient { Timeout = TimeSpan.FromSeconds(300) };
        static readonly HashSet<string> AgentTriggers = new HashSet<string>
            { "survivor_candidate", "sensor_switch", "avoid_fire", "retreat", "stuck" };   // collisions stay with the reflex layer
        const double AgentPeriodS = 20, AgentMinGapS = 4, CommanderPeriodS = 30, OrderHoldS = 15;

        partial class RobotRig
        {
            public RobotActions actions;
            public double lastAgent = double.NegativeInfinity;
            public JObject trigger;
            public readonly List<JObject> inbox = new List<JObject>();     // delivered texts and orders, for the next step
            public readonly List<JObject> outbox = new List<JObject>();    // texts waiting for a radio link
            public readonly List<double> thermalTimes = new List<double>();
            public readonly StringBuilder agentLog = new StringBuilder();
        }

        sealed class AgentWorld
        {
            public DoorRuntime doors;
            public ObstacleRuntime obstacles;
            public Vector3 post;
            public double lastCommander = double.NegativeInfinity;
            public readonly List<JObject> postInbox = new List<JObject>();
            public readonly List<(string to, JObject order)> postOutbox = new List<(string, JObject)>();
            public int steps, orders, refusals, errors;
        }

        static AgentWorld AgentSetup(FloorPlan plan, Scenario scenario, List<RobotRig> rigs)
        {
            var w = new AgentWorld();
            var house = GameObject.Find(HouseBuilder.RootName);
            if (house != null)
                try { w.doors = new DoorRuntime(house.transform, plan, scenario); }
                catch (Exception e) { Debug.LogWarning($"[SmokeRescue] doors not actuated: {e.Message}"); }
            var obstacles = GameObject.Find("Obstacles");
            if (obstacles != null) w.obstacles = new ObstacleRuntime(obstacles.transform);
            foreach (var rig in rigs) rig.actions = new RobotActions(rig.robot, w.doors, w.obstacles);
            var at = scenario.robots[0].spawn.at;    // the incident commander stands at the first entry
            w.post = new Vector3(at[0], 0f, at[1] - 2f);
            return w;
        }

        static void AgentWorldStep(AgentWorld w, List<RobotRig> rigs, float dt)
        {
            w.doors?.Step(dt);
            w.obstacles?.Step(dt);
            foreach (var rig in rigs)
            {
                if (rig.actions == null || !rig.actions.IsRunning) continue;
                rig.actions.Step(dt);
                if (!rig.actions.IsRunning)
                    rig.brain?.LogAgentAction(LastT, $"{rig.actions.Action} {rig.actions.Status}" +
                        (rig.actions.FailureReason != null ? ": " + rig.actions.FailureReason : ""), "target", rig.actions.TargetId ?? "");
            }
        }

        static double LastT;

        static void NoteAgentTrigger(RobotRig rig, Brain.BrainEvent e)
        {
            if (!AgentTriggers.Contains(e.kind)) return;
            if (rig.trigger != null && rig.trigger.Value<string>("kind") == "survivor_candidate") return;   // keep the most important
            rig.trigger = JObject.FromObject(new { e.kind, e.reason, e.numbers });
        }

        static void AgentDecisions(AgentWorld w, List<RobotRig> rigs, double t, Queue<Task> pending, SensorConfig cfg,
                                   Scenario scenario, string root)
        {
            LastT = t;
            foreach (var rig in rigs)
            {
                if (rig.brain == null || rig.missionDone || (rig.actions != null && rig.actions.IsRunning)) continue;
                bool due = (rig.trigger != null && t - rig.lastAgent >= AgentMinGapS) || t - rig.lastAgent >= AgentPeriodS;
                if (!due || rig.thermalTimes.Count == 0) continue;
                var ev = rig.trigger ?? JObject.FromObject(new { kind = "periodic", reason = "Regular check-in.", numbers = new { } });
                rig.trigger = null; rig.lastAgent = t;
                FlushForAgent(rig, pending, cfg);
                var s = rig.brain.Snapshot();
                var req = new JObject
                {
                    ["robot_dir"] = Path.GetFullPath(rig.dir), ["t"] = t,
                    ["trace_path"] = Path.Combine(root, "agent_trace.jsonl"),
                    ["debug_dir"] = Path.Combine(root, "agent_debug"),
                    ["pose"] = JObject.FromObject(new { x = s.pose.position.x, y = s.pose.position.y, heading_deg = -s.pose.yawRad * Mathf.Rad2Deg }),
                    ["mode"] = s.mode, ["state"] = s.state, ["heat_budget"] = s.heatBudget, ["torch"] = rig.robot.TorchMode,
                    ["goal"] = s.goal.HasValue ? new JArray(s.goal.Value.x, s.goal.Value.y) : null,
                    ["event"] = ev,
                    ["frame_times"] = new JArray(rig.thermalTimes.Skip(Math.Max(0, rig.thermalTimes.Count - 45)).Select(x => (object)x)),
                    ["inbox"] = new JArray(rig.inbox),
                    ["survivors"] = new JArray((s.confirmedSurvivors ?? new Vector2[0]).Select(p => new JArray(p.x, p.y))),
                    ["search_area"] = scenario.search_area != null ? new JArray(scenario.search_area) : null,
                };
                rig.inbox.Clear();
                var reply = Post($"/robot/{rig.robot.RobotId}/step", req);
                w.steps++;
                if (reply == null) { w.errors++; continue; }
                rig.agentLog.Append(reply.ToString(Formatting.None)).Append('\n');
                w.refusals += (reply["refusals"] as JArray)?.Count ?? 0;
                foreach (JObject order in (reply["orders"] as JArray) ?? new JArray()) { w.orders++; Apply(w, rig, order, t); }
                if (reply.Value<string>("summary") is string summary && summary.Length > 0)
                    rig.brain.Log(t, "agent_decision", summary, "rounds", reply.Value<int?>("rounds") ?? 0, "latency_s", reply.Value<double?>("latency_s") ?? 0);
            }
            if (t - w.lastCommander >= CommanderPeriodS && rigs.Any(r => r.brain != null))
            {
                w.lastCommander = t;
                var reply = Post("/commander/step", new JObject
                {
                    ["t"] = t, ["trace_path"] = Path.Combine(root, "agent_trace.jsonl"),
                    ["reports"] = new JArray(w.postInbox), ["robots"] = new JArray(rigs.Select(r => r.robot.RobotId)),
                });
                w.postInbox.Clear();
                foreach (JObject o in (reply?["orders"] as JArray) ?? new JArray())
                    w.postOutbox.Add((o.Value<string>("robot") ?? "*", o));
            }
        }

        static void Apply(AgentWorld w, RobotRig rig, JObject o, double t)
        {
            var brain = rig.brain;
            string type = o.Value<string>("type"), ev = o.Value<string>("evidence_id") ?? "";
            Vector2 P() => new Vector2(o.Value<float>("x"), o.Value<float>("y"));
            switch (type)
            {
                case "go_to": brain.AgentGoTo(P(), t, OrderHoldS, o.Value<string>("reason")); break;
                case "confirm_survivor": brain.AgentConfirmSurvivor(P(), t, ev); break;
                case "reject_candidate": brain.AgentRejectCandidate(P(), t, ev); break;
                case "avoid_area": brain.AgentAvoidArea(P(), o.Value<float?>("radius_m") ?? 1.5f, t, o.Value<double?>("seconds") ?? 30, ev); break;
                case "set_torch":
                    try
                    {
                        rig.robot.SetTorch(o.Value<string>("mode"), o.Value<float?>("pitch_deg") ?? 0, o.Value<float?>("yaw_deg") ?? 0);
                        brain.LogAgentAction(t, $"torch set to {rig.robot.TorchMode}, OpenCV evidence {ev}.", "pitch_deg", rig.robot.TorchPitchDeg, "yaw_deg", rig.robot.TorchYawDeg);
                    }
                    catch (ArgumentException e) { brain.LogAgentAction(t, "torch order refused: " + e.Message); }
                    break;
                case "open_door":
                case "clear_debris":
                    bool begun = rig.actions.Begin(type);
                    brain.LogAgentAction(t, begun ? $"{type} started on {rig.actions.TargetId}, OpenCV evidence {ev}." : $"{type} refused: {rig.actions.FailureReason}");
                    break;
                case "send_team_message":
                    var msg = new JObject { ["from"] = rig.robot.RobotId, ["t"] = t, ["text"] = o.Value<string>("text"),
                        ["x"] = brain.Snapshot().pose.position.x, ["y"] = brain.Snapshot().pose.position.y };
                    rig.outbox.Add(msg);
                    if (rig.outbox.Count > 5) rig.outbox.RemoveAt(0);
                    brain.Log(t, "team_message_sent", o.Value<string>("text") ?? "");
                    break;
                case "request_human_approval":
                    var ans = Post("/approval", o);
                    brain.Log(t, "approval", ans == null ? "approval service unreachable; treated as refused"
                        : $"{(ans.Value<bool?>("approved") == true ? "approved" : "refused")} by {ans.Value<string>("by")}: {ans.Value<string>("reason")}",
                        "question", o.Value<string>("question") ?? "");
                    break;
            }
        }

        /// <summary>Agent texts and commander orders over the RadioModel, every broadcast period.</summary>
        static void AgentRadio(AgentWorld w, List<RobotRig> rigs, RadioModel radio, double t)
        {
            foreach (var from in rigs)
            {
                if (from.outbox.Count == 0) continue;
                bool delivered = false;
                foreach (var to in rigs.Where(r => r != from && r.brain != null))
                    if (radio.Evaluate(from.robot.transform.position, to.robot.transform.position).delivered)
                    {
                        foreach (var m in from.outbox) { to.inbox.Add(m); to.brain.Log(t, "team_message_received", $"{m["from"]}: {m["text"]}"); }
                        delivered = true;
                    }
                if (radio.Evaluate(from.robot.transform.position, w.post).delivered) { w.postInbox.AddRange(from.outbox); delivered = true; }
                if (delivered) from.outbox.Clear();   // otherwise the texts wait for a link
            }
            for (int i = w.postOutbox.Count - 1; i >= 0; i--)
            {
                var (to, order) = w.postOutbox[i];
                bool sent = false;
                foreach (var rig in rigs.Where(r => r.brain != null && (to == "*" || to == r.robot.RobotId)))
                {
                    if (!radio.Evaluate(w.post, rig.robot.transform.position).delivered) continue;
                    if (order.Value<string>("type") == "assign_sector" && order["rect"] is JArray r)
                        rig.brain.AssignSector(Rect.MinMaxRect(r[0].Value<float>(), r[1].Value<float>(), r[2].Value<float>(), r[3].Value<float>()),
                            t, "commander", order.Value<string>("reason"));
                    else
                    {
                        rig.inbox.Add(new JObject { ["from"] = "commander", ["t"] = t, ["text"] = order.Value<string>("text") });
                        rig.brain.Log(t, "team_message_received", "commander: " + order.Value<string>("text"));
                    }
                    sent = true;
                }
                if (sent) w.postOutbox.RemoveAt(i);
            }
        }

        static void FlushForAgent(RobotRig rig, Queue<Task> pending, SensorConfig cfg)
        {
            while (pending.Count > 0) pending.Dequeue().Wait();
            File.WriteAllText(Path.Combine(rig.dir, "rgb.txt"), rig.rgb.ToString());
            File.WriteAllText(Path.Combine(rig.dir, "depth.txt"), rig.depth.ToString());
            File.WriteAllText(Path.Combine(rig.dir, "associations.txt"), rig.assoc.ToString());
            File.WriteAllText(Path.Combine(rig.dir, "thermal.txt"), rig.thermalTxt.ToString());
            File.WriteAllText(Path.Combine(rig.dir, "camera.yaml"), CameraYaml(cfg));
            if (cfg.thermal != null) File.WriteAllText(Path.Combine(rig.dir, "thermal.yaml"), ThermalYaml(cfg));
            // Decision and brain-state logs too, so a run that dies keeps them and progress can be watched live.
            File.WriteAllText(Path.Combine(rig.dir, "decisions.jsonl"), rig.decisions.ToString());
            File.WriteAllText(Path.Combine(rig.dir, "brain_state.jsonl"), rig.brainState.ToString());
        }

        static JObject Post(string path, JObject body)
        {
            try
            {
                var content = new StringContent(body.ToString(Formatting.None), Encoding.UTF8, "application/json");
                var resp = Http.PostAsync(AgentUrl.TrimEnd('/') + path, content).Result;
                string text = resp.Content.ReadAsStringAsync().Result;
                if (!resp.IsSuccessStatusCode) { Debug.LogWarning($"[SmokeRescue] agent {path}: HTTP {(int)resp.StatusCode} {text}"); return null; }
                return JObject.Parse(text);
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[SmokeRescue] agent {path} failed: {e.GetBaseException().Message}");
                return null;
            }
        }

        static void WriteAgentLogs(AgentWorld w, List<RobotRig> rigs, string root)
        {
            foreach (var rig in rigs.Where(r => r.agentLog.Length > 0))
                File.WriteAllText(Path.Combine(rig.dir, "agent_steps.jsonl"), rig.agentLog.ToString());
            if (AgentUrl != null)
                File.WriteAllText(Path.Combine(root, "agent_summary.json"), JsonConvert.SerializeObject(new
                    { url = AgentUrl, w.steps, w.orders, w.refusals, w.errors }, Formatting.Indented));
        }
    }
}
