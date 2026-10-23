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
        /// <summary>Vision service for the live colour person detector (OpenCV 5 dnn, YOLO11m). It is perception, not
        /// the language model, so it runs with or without agents. Null skips it (logged at the start of a run).</summary>
        public static string PeopleUrl;
        public static float PeoplePeriodS = 1f;
        /// <summary>How often the brain takes the live 3D map's height layer, and how often the map is saved as a point
        /// cloud in the recording (for the replay and the video).</summary>
        const double Map3dLayersPeriodS = 5, Map3dExportPeriodS = 30;
        // The agent server bounds a step at about 75 s plus one 45 s call; past 120 s the reflex layer carries on alone.
        static float PlaceHttpTimeoutS;
        static readonly HttpClient Http = new HttpClient { Timeout = TimeSpan.FromSeconds(120) };
        static readonly HashSet<string> AgentTriggers = new HashSet<string>
            { "survivor_candidate", "survivor_view_ready", "sensor_switch", "avoid_fire", "retreat", "stuck",
              // The model chooses where to search next: ask it whenever an area is done or a route has failed.
              "area_searched", "no_progress", "goal_blacklisted", "collision_hold", "no_frontier", "escape_closing",
              // It asked to go somewhere to look, lift or measure: the arrival is its cue.
              "agent_goal_reached" };
        const double AgentPeriodS = 20, AgentMinGapS = 4, CommanderPeriodS = 30, OrderHoldS = 15, ExploreHoldS = 60;
        const int FrontierOptions = 8;

        /// <summary>Consecutive failed agent calls after which the reflex confirms people itself (Unity HTTP timeout 120 s, so
        /// three in a row is minutes of a silent agent, never one slow reply).</summary>
        const int AgentDownAfterFailures = 3;
        partial class RobotRig
        {
            public int agentFailures;
            public RobotActions actions;
            public double lastAgent = double.NegativeInfinity;
            public double lastBrainStep = double.NaN;
            public JObject trigger;
            public readonly List<JObject> inbox = new List<JObject>();     // delivered texts and orders, for the next step
            public readonly List<Outgoing> outbox = new List<Outgoing>();  // texts not yet heard by every listener
            public readonly List<double> thermalTimes = new List<double>();
            public readonly StringBuilder agentLog = new StringBuilder();
            public double lastPeople = double.NegativeInfinity, lastMap3dLayers = double.NegativeInfinity, lastMap3dExport = double.NegativeInfinity;
            public VisualPlaceWorker visualPlaces;
        }

        /// <summary>A team text and who has heard it. Delivery is tracked per listener: clearing the whole outbox when
        /// any one listener heard it lost texts for the others (agent run 9: robot A's survivor report reached robot B
        /// and was dropped before the command post heard it, so the commander never learned of the survivor).</summary>
        sealed class Outgoing
        {
            public JObject message;
            public readonly HashSet<string> heardBy = new HashSet<string>();
        }
        const string CommandPostId = "command_post";

        sealed class AgentWorld
        {
            public DoorRuntime doors;
            public ObstacleRuntime obstacles;
            public Vector3 post;
            public double lastCommander = double.NegativeInfinity;
            public readonly List<JObject> postInbox = new List<JObject>();
            public NozzleConfig nozzle;
            public RescueConfig rescue;
            /// <summary>Recorder log of people strapped onto a sled and released (rescues.jsonl), with where they ended up.</summary>
            public readonly StringBuilder rescueLog = new StringBuilder();
            /// <summary>Recorder log of every spray the world received (water.jsonl): who, when, where, how much.</summary>
            public readonly StringBuilder waterLog = new StringBuilder();
            /// <summary>Recorder-side audit of the command post's radio: each link check with a text waiting, and each text
            /// heard. Written beside agent_trace.jsonl; no robot reads it.</summary>
            public readonly StringBuilder postLog = new StringBuilder();
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
            w.nozzle = NozzleConfig.Load(); w.rescue = RescueConfig.Load();
            foreach (var rig in rigs) rig.actions = new RobotActions(rig.robot, w.doors, w.obstacles, w.nozzle, w.rescue);
            var at = scenario.robots[0].spawn.at;    // world side only: where the commander's radio stands, outside the first entry
            w.post = new Vector3(at[0], 0f, at[1] - 2f);
            return w;
        }

        static void AgentWorldStep(AgentWorld w, List<RobotRig> rigs, float dt)
        {
            foreach (var rig in rigs) rig.actions?.Tow();   // a person on the sled follows the robot every tick
            foreach (var rig in rigs) rig.actions?.StepVisuals(dt);   // water in flight lands after the nozzle closes
            w.doors?.Step(dt);
            w.obstacles?.Step(dt);
            foreach (var rig in rigs)
            {
                if (rig.actions == null || !rig.actions.IsRunning) continue;
                rig.actions.Step(dt);
                if (!rig.actions.IsRunning)
                    rig.brain?.LogAgentAction(LastT, $"{rig.actions.Action} {rig.actions.Status}" +
                        (rig.actions.FailureReason != null ? ": " + rig.actions.FailureReason : "") +
                        (rig.actions.TargetId != null ? $" (target {rig.actions.TargetId})" : ""));   // numbers only in the values
            }
        }

        static double LastT;

        /// <summary>Whether a robot's agent is asked now. Events wait AgentMinGapS after the last call, to bound model calls;
        /// an arrival the agent asked for, or the candidate view it set up, is decided at once, because the robot is
        /// standing still for exactly that decision and the simulation pauses while the agent answers. Measured on held-out
        /// seeds 915 to 917: 28 to 58 s per run of standing after agent turns, most of it waiting out this gap and the hold.</summary>
        public static bool AgentDue(string triggerKind, double sinceLastS)
        {
            if (sinceLastS >= AgentPeriodS) return true;
            if (triggerKind == null) return false;
            if (triggerKind == "agent_goal_reached" || triggerKind == "survivor_view_ready") return true;
            return sinceLastS >= AgentMinGapS;
        }

        /// <summary>The decision trigger after a radio message arrives: a safety or person event already waiting keeps its
        /// place; otherwise the message asks for a decision (after the usual gap, AgentDue). Messages only reached the inbox
        /// and waited for some other event or the 20 s period (Codex design review, 22 Oct).</summary>
        public static JObject MessageTrigger(JObject held, JObject message)
        {
            string kind = held?.Value<string>("kind");
            if (kind == "survivor_candidate" || kind == "survivor_view_ready" || kind == "avoid_fire" || kind == "retreat" ||
                kind == "escape_closing" || kind == "agent_goal_reached")
                return held;
            return JObject.FromObject(new { kind = "team_message",
                reason = $"Radio from {message.Value<string>("from")}: {message.Value<string>("kind") ?? "message"}.",
                numbers = new { } });
        }

        static void NoteAgentTrigger(RobotRig rig, Brain.BrainEvent e)
        {
            if (!AgentTriggers.Contains(e.kind)) return;
            string held = rig.trigger?.Value<string>("kind");
            if ((held == "survivor_candidate" || held == "survivor_view_ready") && e.kind != "survivor_view_ready") return;   // keep the most important
            rig.trigger = JObject.FromObject(new { e.kind, e.reason, e.numbers });
        }

        static void AgentDecisions(AgentWorld w, List<RobotRig> rigs, double t, Queue<Task> pending, SensorConfig cfg,
                                   Scenario scenario, string root)
        {
            LastT = t;
            var calls = new List<(RobotRig rig, JObject req)>();
            foreach (var rig in rigs)
            {
                if (rig.brain == null || rig.missionDone || (rig.actions != null && rig.actions.IsRunning)) continue;
                bool due = AgentDue(rig.trigger?.Value<string>("kind"), t - rig.lastAgent);
                if (!due || rig.thermalTimes.Count == 0) continue;
                var ev = rig.trigger ?? JObject.FromObject(new { kind = "periodic", reason = "Regular check-in.", numbers = new { } });
                rig.trigger = null; rig.lastAgent = t;
                FlushForAgent(rig, pending, cfg);
                var s = rig.brain.Snapshot();
                var frontiers = new JArray(rig.brain.FrontierOptions(t, FrontierOptions).Select(f =>
                {
                    var option = new JObject
                    {
                        ["id"] = f.id, ["x"] = Math.Round(f.position.x, 2), ["y"] = Math.Round(f.position.y, 2),
                        ["path_m"] = Math.Round(f.pathM, 1), ["unsearched_m2"] = Math.Round(f.unsearchedM2, 1), ["mapped_unsearched_m2"] = Math.Round(f.mappedUnsearchedM2, 1), ["proven_indoors"] = f.provenIndoors, ["toward_partner"] = f.towardPartner,
                    };
                    if (f.roomId >= 0) option["room_id"] = f.roomId;
                    return (object)option;
                }));
                var req = new JObject
                {
                    ["robot_dir"] = Path.GetFullPath(rig.dir), ["t"] = t, ["mission_time_left_s"] = Math.Round(rig.brain.MissionTimeLeft(t)),
                    ["trace_path"] = Path.Combine(root, "agent_trace.jsonl"),
                    ["debug_dir"] = Path.Combine(root, "agent_debug"),
                    ["pose"] = JObject.FromObject(new { x = s.pose.position.x, y = s.pose.position.y, heading_deg = -s.pose.yawRad * Mathf.Rad2Deg }),
                    ["mode"] = s.mode, ["state"] = s.state, ["heat_budget"] = s.heatBudget, ["torch"] = rig.robot.TorchMode,
                    ["goal"] = s.goal.HasValue ? new JArray(s.goal.Value.x, s.goal.Value.y) : null,
                    ["event"] = ev,
                    ["frame_times"] = new JArray(rig.thermalTimes.Skip(Math.Max(0, rig.thermalTimes.Count - 45)).Select(x => (object)x)),
                    ["inbox"] = new JArray(rig.inbox),
                    ["survivors"] = new JArray((s.confirmedSurvivors ?? new Vector2[0]).Select(p => new JArray(p.x, p.y))),
                    ["frontiers"] = frontiers,
                    ["fire_blocks_frontier"] = frontiers.Count > 0 ? null : rig.brain.FireBlockedFrontier(t) is Brain.BrainFireBlock fb ? new JObject
                    {
                        ["bearing_deg"] = Math.Round(fb.bearingDegRight, 1), ["distance_m"] = Math.Round(fb.distanceM, 2),
                        ["frontier_path_m"] = Math.Round(fb.frontierPathM, 1), ["unsearched_m2"] = Math.Round(fb.unsearchedM2, 1),
                        ["burning_m2"] = Math.Round(fb.burningM2, 2),
                    } : null,
                    ["fire_blocks_exit"] = rig.brain.FireBlockedExit(t) is Brain.BrainFireBlock fx ? new JObject
                    {
                        ["bearing_deg"] = Math.Round(fx.bearingDegRight, 1), ["distance_m"] = Math.Round(fx.distanceM, 2),
                        ["exit_path_m"] = Math.Round(fx.frontierPathM, 1), ["burning_m2"] = Math.Round(fx.burningM2, 2),
                    } : null,
                    ["fire_near_robot"] = rig.brain.FireNearRobot(t) is Brain.BrainFireBlock fn ? new JObject
                    {
                        ["bearing_deg"] = Math.Round(fn.bearingDegRight, 1), ["distance_m"] = Math.Round(fn.distanceM, 2),
                        ["burning_m2"] = Math.Round(fn.burningM2, 2),
                    } : null,
                    // The robot's own air sensor and thermal map (never the simulator's fire): issue list 23 Oct, item 4.
                    ["air_c"] = rig.brain.AirState(t) is Brain.BrainAirState air ? Math.Round(air.airC, 1) : (double?)null,
                    ["air_trend_c_per_min"] = rig.brain.AirState(t) is Brain.BrainAirState trend ? Math.Round(trend.trendCPerMin, 1) : (double?)null,
                    ["heat_seconds_left"] = rig.brain.AirState(t) is Brain.BrainAirState left && left.heatSecondsLeft.HasValue ? Math.Round(left.heatSecondsLeft.Value, 1) : (double?)null,
                    ["nearest_fire"] = rig.brain.NearestFire() is Brain.BrainFireBlock nf ? new JObject
                    {
                        ["bearing_deg"] = Math.Round(nf.bearingDegRight, 1), ["distance_m"] = Math.Round(nf.distanceM, 2),
                        ["burning_m2"] = Math.Round(nf.burningM2, 2),
                    } : null,
                    ["lidar"] = LidarSummary(rig.lastScan),
                    ["water_l"] = Math.Round(rig.actions?.WaterLitres ?? 0, 2),
                    ["carrying_person"] = rig.actions?.Carrying != null,
                    ["blocked_unsearched"] = rig.brain.BlockedUnsearched() is Brain.BrainBlockedFloor blocked ? new JObject
                    {
                        ["bearing_deg"] = Math.Round(blocked.bearingDegRight, 1), ["distance_m"] = Math.Round(blocked.distanceM, 2),
                        ["unsearched_m2"] = Math.Round(blocked.areaM2, 1),
                        ["approach"] = new JObject { ["x"] = Math.Round(blocked.approach.x, 2), ["y"] = Math.Round(blocked.approach.y, 2) },
                    } : null,
                    ["partners"] = new JArray(s.peers.Select(p => (object)new JObject { ["robot"] = p.robot, ["maps_linked"] = p.linked,
                        ["position"] = p.linked ? new JArray(Math.Round(p.pose.position.x, 2), Math.Round(p.pose.position.y, 2)) : null })),
                };
                var rooms = rig.brain.RoomOptions(t);
                if (rooms.Count > 0)
                    req["rooms"] = new JArray(rooms.Select(room => (object)new JObject
                    {
                        ["id"] = room.id, ["x"] = Math.Round(room.centre.x, 2), ["y"] = Math.Round(room.centre.y, 2),
                        ["covered_share"] = room.coveredShare, ["entered"] = room.entered,
                        ["dwell_s"] = Math.Round(room.dwellS, 1), ["indoor_floor_m2"] = Math.Round(room.indoorFloorM2, 1),
                    }));
                calls.Add((rig, req));
            }
            // Every due robot's step goes out at once, so the sim waits for the slowest model call, not their sum.
            // Replies are applied in robot order, so the outcome does not depend on which call returns first.
            var replies = PostAll(calls.Select(d => ($"/robot/{d.rig.robot.RobotId}/step", d.req)).ToList());
            for (int i = 0; i < calls.Count; i++)
            {
                var rig = calls[i].rig; var reply = replies[i];
                w.steps++;
                // The messages this step was shown leave the inbox only once the step answered; a failed call keeps them for
                // the next one (they were cleared before the call; Codex design review, 22 Oct). The sim is paused during
                // the call, so nothing arrived meanwhile and the first ones are exactly those shown.
                // Agent liveness: with the agent unreachable for agent_down_after_failures calls in a row, the reflex confirms
                // people itself again, so a dead server cannot leave a found person unconfirmed (issue list 23 Oct, item 2).
                if (reply == null)
                {
                    w.errors++;
                    if (++rig.agentFailures >= AgentDownAfterFailures && rig.brain.AgentConfirms)
                    { rig.brain.AgentConfirms = false; rig.brain.LogAgentAction(t, $"agent unreachable {rig.agentFailures} calls in a row: the robot confirms people itself"); }
                    continue;
                }
                if (rig.agentFailures >= AgentDownAfterFailures) rig.brain.LogAgentAction(t, "agent reachable again: confirmations go back to the agent");
                rig.agentFailures = 0; rig.brain.AgentConfirms = true;
                rig.inbox.RemoveRange(0, Math.Min(((JArray)calls[i].req["inbox"]).Count, rig.inbox.Count));
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
                case "go_to":
                    brain.AgentGoTo(P(), t, OrderHoldS, o.Value<string>("reason"),
                        o["face_x"] != null && o["face_y"] != null ? new Vector2(o.Value<float>("face_x"), o.Value<float>("face_y")) : (Vector2?)null);
                    break;
                case "explore_frontier":
                    brain.AgentExplore(P(), t, ExploreHoldS, o.Value<string>("reason"), o.Value<bool?>("unproven") == true ? ev : null);
                    break;
                case "confirm_survivor": brain.AgentConfirmSurvivor(P(), t, ev); break;
                case "reject_candidate": brain.AgentRejectCandidate(P(), t, ev); break;
                case "withdraw_survivor": brain.AgentWithdrawSurvivor(P(), t, ev); break;
                case "avoid_area": brain.AgentAvoidArea(P(), o.Value<float?>("radius_m") ?? 1.5f, t, o.Value<double?>("seconds") ?? 30, ev); break;
                case "set_torch":
                    try
                    {
                        rig.robot.SetTorch(o.Value<string>("mode"), o.Value<float?>("pitch_deg") ?? 0, o.Value<float?>("yaw_deg") ?? 0);
                        brain.LogAgentAction(t, $"torch set to {rig.robot.TorchMode}, OpenCV evidence {ev}.", "pitch_deg", rig.robot.TorchPitchDeg, "yaw_deg", rig.robot.TorchYawDeg);
                    }
                    catch (ArgumentException e) { brain.LogAgentAction(t, "torch order refused: " + e.Message); }
                    break;
                case "extract_survivor":
                {
                    // The agent names the person by a position in its robot's own frame; the arm reaches that way from the
                    // robot's real pose (relative bearing and range, as a physical arm would).
                    Vector3? target = null;
                    if (o["target_x"] != null && o["target_y"] != null)
                    {
                        var pose = brain.Snapshot().pose;
                        var rel = new Vector2(o.Value<float>("target_x"), o.Value<float>("target_y")) - pose.position;
                        float forward = Vector2.Dot(rel, pose.Forward), right = Vector2.Dot(rel, new Vector2(Mathf.Cos(pose.yawRad), Mathf.Sin(pose.yawRad)));
                        float yaw = rig.robot.Driver.YawDeg * Mathf.Deg2Rad;
                        target = rig.robot.transform.position + new Vector3(Mathf.Sin(yaw), 0f, Mathf.Cos(yaw)) * forward + new Vector3(Mathf.Cos(yaw), 0f, -Mathf.Sin(yaw)) * right;
                    }
                    bool gripping = rig.actions.BeginExtract(target);
                    brain.LogAgentAction(t, gripping ? $"extract_survivor: strapping {rig.actions.TargetId} onto the sled, OpenCV evidence {ev}."
                        : $"extract_survivor refused: {rig.actions.FailureReason}");
                    break;
                }
                case "spray_water":
                {
                    bool sprayed = rig.actions.BeginSpray(o.Value<float>("bearing_deg"), o.Value<float>("seconds"));
                    brain.LogAgentAction(t, sprayed ? $"spray_water at {o.Value<float>("bearing_deg"):0} deg for up to {o.Value<float>("seconds"):0.#} s, OpenCV evidence {ev}."
                        : $"spray_water refused: {rig.actions.FailureReason}", "water_l", rig.actions.WaterLitres);
                    break;
                }
                case "open_door":
                case "clear_debris":
                    bool begun = rig.actions.Begin(type);
                    brain.LogAgentAction(t, begun ? $"{type} started on {rig.actions.TargetId}, OpenCV evidence {ev}." : $"{type} refused: {rig.actions.FailureReason}");
                    break;
                case "send_team_message":
                    // Every field the agent server stamped (message_id, kind, proposal_id, reply_to, expires_t, my_part,
                    // your_part) travels with the text; the recorder adds sender, time and the sender's own-frame position.
                    var msg = new JObject();
                    foreach (var p in o.Properties()) if (p.Name != "type" && p.Name != "robot") msg[p.Name] = p.Value.DeepClone();
                    msg["from"] = rig.robot.RobotId; msg["t"] = t;
                    msg["x"] = brain.Snapshot().pose.position.x; msg["y"] = brain.Snapshot().pose.position.y;
                    rig.outbox.Add(new Outgoing { message = msg });
                    if (rig.outbox.Count > 5) rig.outbox.RemoveAt(0);
                    brain.Log(t, "team_message_sent", o.Value<string>("text") ?? "");
                    break;
                case "request_human_approval":
                    var ans = Post("/approval", o);
                    brain.Log(t, "approval", ans == null ? "approval service unreachable; treated as refused"
                        : $"{(ans.Value<bool?>("approved") == true ? "approved" : "refused")} by {ans.Value<string>("by")}: {ans.Value<string>("reason")}" +
                          $" (question: {o.Value<string>("question") ?? ""})");
                    break;
            }
        }

        /// <summary>Agent texts and commander orders over the RadioModel, every broadcast period.</summary>
        static void AgentRadio(AgentWorld w, List<RobotRig> rigs, RadioModel radio, double t)
        {
            foreach (var from in rigs)
            {
                if (from.outbox.Count == 0) continue;
                var partners = rigs.Where(r => r != from && r.brain != null).ToList();
                foreach (var to in partners)
                {
                    if (!radio.Evaluate(from.robot.transform.position, to.robot.transform.position).delivered) continue;
                    foreach (var m in from.outbox)
                    {
                        if (!m.heardBy.Add(to.robot.RobotId)) continue;
                        // The sender's position is in its own frame: convert it once the frames are linked, else drop it.
                        var copy = (JObject)m.message.DeepClone();
                        if (copy["x"] != null && to.brain.PartnerToOwn(from.robot.RobotId, new Vector2(copy.Value<float>("x"), copy.Value<float>("y")), out var own))
                        { copy["x"] = Math.Round(own.x, 2); copy["y"] = Math.Round(own.y, 2); copy["position"] = "converted into your frame"; }
                        else { copy.Remove("x"); copy.Remove("y"); copy["position"] = "unknown: your frames are not linked yet"; }
                        to.inbox.Add(copy); to.brain.Log(t, "team_message_received", $"{copy["from"]}: {copy["text"]}");
                        to.trigger = MessageTrigger(to.trigger, copy);
                    }
                }
                var toPost = radio.Evaluate(from.robot.transform.position, w.post);
                if (from.outbox.Any(m => !m.heardBy.Contains(CommandPostId)))
                    w.postLog.Append(new JObject { ["t"] = Math.Round(t, 3), ["from"] = from.robot.RobotId, ["kind"] = "link",
                        ["distance_m"] = Math.Round(toPost.distance, 2), ["walls"] = toPost.walls, ["reason"] = toPost.reason }.ToString(Formatting.None)).Append('\n');
                if (toPost.delivered)
                    foreach (var m in from.outbox)
                    {
                        if (!m.heardBy.Add(CommandPostId)) continue;
                        w.postLog.Append(new JObject { ["t"] = Math.Round(t, 3), ["from"] = from.robot.RobotId, ["kind"] = "heard",
                            ["text"] = m.message.Value<string>("text") }.ToString(Formatting.None)).Append('\n');
                        // The commander has no map: robot positions (each in its own frame) would mean nothing to it.
                        var copy = (JObject)m.message.DeepClone(); copy.Remove("x"); copy.Remove("y"); w.postInbox.Add(copy);
                    }
                // A text waits until every partner and the command post have heard it.
                from.outbox.RemoveAll(m => m.heardBy.Count >= partners.Count + 1);
            }
            for (int i = w.postOutbox.Count - 1; i >= 0; i--)
            {
                var (to, order) = w.postOutbox[i];
                foreach (var rig in rigs.Where(r => r.brain != null && (to == "*" || to == r.robot.RobotId)))
                {
                    if (!radio.Evaluate(w.post, rig.robot.transform.position).delivered) continue;
                    // The commander only talks: it has no map and assigns no areas (Tashon, 2026-10-20).
                    // Once per robot: a broadcast stays queued until every robot it is for has heard it (it was dropped
                    // after the first one, so the other never heard it; Codex design review, 22 Oct).
                    var heard = order["heard_by"] as JArray ?? (JArray)(order["heard_by"] = new JArray());
                    if (heard.Any(h => (string)h == rig.robot.RobotId)) continue;
                    heard.Add(rig.robot.RobotId);
                    var said = new JObject { ["from"] = "commander", ["t"] = t, ["text"] = order.Value<string>("text") };
                    rig.inbox.Add(said); rig.trigger = MessageTrigger(rig.trigger, said);
                    rig.brain.Log(t, "team_message_received", "commander: " + order.Value<string>("text"));
                }
                var meant = rigs.Where(r => r.brain != null && (to == "*" || to == r.robot.RobotId)).Select(r => r.robot.RobotId);
                var got = (order["heard_by"] as JArray)?.Select(h => (string)h) ?? Enumerable.Empty<string>();
                if (meant.All(id => got.Contains(id))) w.postOutbox.RemoveAt(i);
            }
        }

        /// <summary>Nearest LiDAR return in eight 45 degree sectors (bearing + right, 0 ahead), for the agent's context.</summary>
        static JObject LidarSummary(LidarScan scan)
        {
            if (scan == null) return null;
            var nearest = Enumerable.Repeat(float.PositiveInfinity, 8).ToArray(); int returns = 0;
            for (int i = 0; i < scan.RangesM.Length; i++)
            {
                float r = scan.RangesM[i]; if (r <= 0) continue; returns++;
                float bearingRight = -scan.Angle(i) * Mathf.Rad2Deg;
                int sector = Mathf.RoundToInt(Mathf.Repeat(bearingRight, 360f) / 45f) % 8;
                nearest[sector] = Mathf.Min(nearest[sector], r);
            }
            string[] names = { "ahead", "front_right", "right", "back_right", "behind", "back_left", "left", "front_left" };
            var o = new JObject { ["t"] = Math.Round(scan.Timestamp, 2), ["return_share"] = Math.Round(returns / (double)scan.RangesM.Length, 2) };
            for (int k = 0; k < 8; k++) o[names[k] + "_m"] = float.IsPositiveInfinity(nearest[k]) ? (JToken)null : Math.Round(nearest[k], 2);
            return o;
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

        /// <summary>Once per PeoplePeriodS per robot: run the colour person detector on the robot's latest frame and hand
        /// each sighting (bearing, range, confidence) to its brain.</summary>
        static void PeopleScan(List<RobotRig> rigs, double t, Queue<Task> pending, SensorConfig cfg)
        {
            if (PeopleUrl == null) return;
            foreach (var rig in rigs)
            {
                if (rig.brain == null || rig.missionDone || t - rig.lastPeople < PeoplePeriodS - 1e-9) continue;
                rig.lastPeople = t;
                FlushForAgent(rig, pending, cfg);
                var reply = Post("/perceive/people", new JObject { ["robot_dir"] = Path.GetFullPath(rig.dir), ["t"] = t, ["min_conf"] = 0.25 }, PeopleUrl);
                if (reply == null) continue;
                double seen = reply.Value<double?>("t") ?? t;
                foreach (JObject p in (reply["people"] as JArray) ?? new JArray())
                {
                    if (p["range_m"] == null || p["range_m"].Type == JTokenType.Null) continue;
                    rig.brain.OnPersonSighting(Math.Min(seen, t), p.Value<float>("bearing_deg"), p.Value<float>("range_m"), p.Value<float>("confidence"));
                }
                Map3d(rig, t);
            }
        }

        static void VisualKeyframe(RobotRig rig, double t, Queue<Task> pending, SensorConfig cfg)
        {
            if (PeopleUrl == null || rig.brain == null || rig.missionDone || !rig.brain.VisualPlacesEnabled) return;
            if (rig.visualPlaces == null) rig.visualPlaces = new VisualPlaceWorker(rig.brain, rig.robot.RobotId,
                Path.GetFullPath(rig.dir), new VisualPlaceClient(PeopleUrl, PlaceHttpTimeoutS));
            if (!rig.visualPlaces.CaptureDue(t)) return;
            FlushForAgent(rig, pending, cfg);
            rig.visualPlaces.TryCapture(t, rig.brain.Snapshot().pose);
        }

        /// <summary>The robot's live 3D map: its latest depth frame fused at its own estimated pose into an OpenCV 5 TSDF
        /// volume (agent service), the top-height layer handed back to the brain, and the map saved now and then.</summary>
        static void Map3d(RobotRig rig, double t)
        {
            var s = rig.brain.Snapshot();
            string id = rig.robot.RobotId;
            var pose = new JObject { ["x"] = s.pose.position.x, ["y"] = s.pose.position.y, ["heading_deg"] = -s.pose.yawRad * Mathf.Rad2Deg };
            if (Post($"/map3d/{id}/integrate", new JObject { ["robot_dir"] = Path.GetFullPath(rig.dir), ["t"] = t, ["pose"] = pose }, PeopleUrl) == null) return;
            if (t - rig.lastMap3dLayers >= Map3dLayersPeriodS - 1e-9)
            {
                rig.lastMap3dLayers = t;
                var layers = Post($"/map3d/{id}/layers", new JObject { ["cell_m"] = s.cellM, ["origin"] = new JArray(s.gridOrigin.x, s.gridOrigin.y), ["size"] = s.gridSize }, PeopleUrl);
                string top = layers?.Value<string>("top_cm");
                if (top != null)
                {
                    var bytes = Convert.FromBase64String(top);
                    if (bytes.Length == s.gridSize * s.gridSize * 2)
                    {
                        var cm = new ushort[s.gridSize * s.gridSize];
                        Buffer.BlockCopy(bytes, 0, cm, 0, bytes.Length);
                        rig.brain.OnTopHeights(cm);
                    }
                    else rig.brain.Log(t, "map3d", $"height layer size {bytes.Length} does not match the grid; ignored");
                }
            }
            if (t - rig.lastMap3dExport >= Map3dExportPeriodS - 1e-9)
            {
                rig.lastMap3dExport = t;
                Post($"/map3d/{id}/export", new JObject { ["path"] = Path.GetFullPath(Path.Combine(rig.dir, "map3d.ply")) }, PeopleUrl);
            }
        }

        static List<JObject> PostAll(List<(string path, JObject body)> calls, string baseUrl = null)
        {
            var tasks = calls.Select(c => Task.Run(() => Post(c.path, c.body, baseUrl))).ToArray();
            Task.WaitAll(tasks);
            return tasks.Select(task => task.Result).ToList();
        }

        static JObject Post(string path, JObject body, string baseUrl = null)
        {
            try
            {
                var content = new StringContent(body.ToString(Formatting.None), Encoding.UTF8, "application/json");
                var resp = Http.PostAsync((baseUrl ?? AgentUrl).TrimEnd('/') + path, content).Result;
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
            if (w.postLog.Length > 0) File.WriteAllText(Path.Combine(root, "command_post.jsonl"), w.postLog.ToString());
            if (w.waterLog.Length > 0) File.WriteAllText(Path.Combine(root, "water.jsonl"), w.waterLog.ToString());
            if (w.rescueLog.Length > 0) File.WriteAllText(Path.Combine(root, "rescues.jsonl"), w.rescueLog.ToString());
            if (AgentUrl != null)
                File.WriteAllText(Path.Combine(root, "agent_summary.json"), JsonConvert.SerializeObject(new
                    { url = AgentUrl, w.steps, w.orders, w.refusals, w.errors }, Formatting.Indented));
        }
    }
}
