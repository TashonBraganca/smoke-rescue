using System.Collections.Generic;
using System;
using System.IO;
using SmokeRescue.World;
using UnityEngine;

namespace SmokeRescue.Robots
{
    /// <summary>The robot's water nozzle and tank, from config/actuators.json (each value with its reason there).</summary>
    [Serializable]
    public sealed class NozzleConfig
    {
        public float tank_l, flow_lps, range_m, half_angle_deg, max_spray_s, aim_tolerance_deg;
        [Serializable] sealed class File { public NozzleConfig nozzle; }
        public static NozzleConfig Load() =>
            JsonUtility.FromJson<File>(System.IO.File.ReadAllText(Path.Combine(FloorPlan.ConfigDir, "actuators.json"))).nozzle;
    }

    /// <summary>Gripping a person onto a rescue sled and towing them, from config/actuators.json.</summary>
    [Serializable]
    public sealed class RescueConfig
    {
        public float reach_m, grip_s, tow_offset_m, carry_speed_mps, target_tolerance_m;
        [Serializable] sealed class File { public RescueConfig rescue; }
        public static RescueConfig Load() =>
            JsonUtility.FromJson<File>(System.IO.File.ReadAllText(Path.Combine(FloorPlan.ConfigDir, "actuators.json"))).rescue;
    }

    /// <summary>
    /// Physical actions an agent can order: "open_door", "clear_debris" and "spray_water". Each runs reach, act, stow on sim time and
    /// holds the robot still meanwhile. Call order per tick: robot.Step, doors.Step and obstacles.Step, then Step here,
    /// so the gripper follows the door handle or crate where the world just moved it.
    /// Status is idle, running, succeeded or failed; FailureReason says why a Begin was refused or an action failed.
    /// </summary>
    public sealed class RobotActions
    {
        public const float ReachSeconds = 1f, StowSeconds = 1f, PushSeconds = 1f, PushDistanceM = 0.6f, DoorSeconds = 2f;
        public const float ReachAheadM = 1.2f;

        readonly Robot robot;
        readonly DoorRuntime doors;
        readonly ObstacleRuntime obstacles;

        public string Action { get; private set; }
        public string TargetId { get; private set; }
        public string Status { get; private set; } = "idle";
        public string FailureReason { get; private set; }
        public bool IsRunning => Status == "running";
        /// <summary>Water left in the tank, litres. The robot cannot refill inside the building.</summary>
        public double WaterLitres { get; private set; }
        /// <summary>Raised once when water starts to flow: nozzle position (Unity world), heading (radians, Unity yaw:
        /// 0 = +z, positive toward +x), flow (L/s) and duration (s). The recorder hands it to the fire simulation.</summary>
        public event Action<Vector3, float, float, float> SprayStarted;
        readonly NozzleConfig nozzle;
        float sprayHeadingDeg, spraySeconds;
        readonly RescueConfig rescue;
        LineRenderer jet;   // the visible stream while water flows (cameras and the showcase view see it)
        Transform carried; Collider[] carriedColliders;
        /// <summary>Where the robot walked while towing, starting where the person lay: the sled follows this trail.</summary>
        readonly List<Vector3> trail = new List<Vector3>();
        /// <summary>Raised when a person is strapped onto the sled; the recorder tells the brain to carry them out.</summary>
        public event Action<string> GripCompleted;
        /// <summary>Scene name of the person on the sled, or null.</summary>
        public string Carrying => carried != null ? carried.name : null;

        enum Phase { Reach, Act, Stow }
        Phase phase;
        double phaseTime;
        Vector3 phaseStart, contact, objectStart;
        Vector2 direction;
        Transform pushed;

        public RobotActions(Robot robot, DoorRuntime doors, ObstacleRuntime obstacles, NozzleConfig nozzle = null, RescueConfig rescue = null)
        {
            this.robot = robot ?? throw new ArgumentNullException(nameof(robot));
            this.doors = doors;
            this.obstacles = obstacles;
            this.nozzle = nozzle;
            this.rescue = rescue;
            WaterLitres = nozzle?.tank_l ?? 0;
        }

        /// <summary>Strap the nearest person within rescue reach ahead onto the sled (reach, grip, stow). Afterwards the
        /// person is towed behind the robot (Tow, every tick) until Release.</summary>
        public bool BeginExtract() => BeginExtract(null);

        /// <summary>Strap on the person nearest the named world point (the agent's target, carried from its robot's own
        /// frame through the robot's pose), within target_tolerance_m of it; null names nobody and takes the nearest
        /// within reach (kept for tests and callers without a target).</summary>
        public bool BeginExtract(Vector3? target)
        {
            if (IsRunning) return Refuse($"busy with {Action}");
            if (rescue == null) return Refuse("no rescue sled fitted");
            if (carried != null) return Refuse("already towing someone");
            var root = GameObject.Find("Survivors");
            if (root == null) return Refuse("no people in this scene");
            var position = robot.transform.position; float yaw = robot.Driver.YawDeg * Mathf.Deg2Rad;
            var forward = new Vector3(Mathf.Sin(yaw), 0f, Mathf.Cos(yaw));
            Transform best = null; float bestDistance = float.PositiveInfinity; Vector3 bestPoint = default;
            foreach (Transform person in root.transform)
            {
                var col = person.GetComponent<Collider>(); if (col == null || !col.enabled) continue;
                var near = col.ClosestPoint(position + Vector3.up * .3f);
                var flat = new Vector3(near.x - position.x, 0f, near.z - position.z);
                if (flat.magnitude > .05f && Vector3.Angle(forward, flat) > 60f) continue;
                // Whether the strap can go on, from the arm's own link lengths: the nearest part of the body to the arm.
                if (!robot.Arm.CanReach(col.ClosestPoint(robot.Arm.Root.position))) continue;
                // Only the person the agent named (review: the arm used to lift whoever was nearest): the body closest to
                // the named spot, within target_tolerance_m of it.
                float score = flat.magnitude;
                if (target.HasValue)
                {
                    var onBody = col.ClosestPoint(target.Value);
                    score = new Vector2(onBody.x - target.Value.x, onBody.z - target.Value.z).magnitude;
                    if (score > rescue.target_tolerance_m) continue;
                }
                // The strap goes on the nearest part of the body (a shoulder or limb), the point closest to the arm. The
                // centre of a lying person's bounds is a metre past their near edge, beyond the arm (run 26: robot A's lift
                // of s1 failed with "contact point is out of the arm's reach" while their near edge was within reach_m).
                if (score < bestDistance) { bestDistance = score; best = person; bestPoint = col.ClosestPoint(robot.Arm.Root.position); }
            }
            if (best == null) return Refuse(target.HasValue ? "nobody at the named spot within the arm's reach ahead; move closer, facing them"
                                                          : "nobody within the arm's reach ahead; move closer, facing them");
            contact = bestPoint; TargetId = best.name; pushed = best;
            Action = "extract_survivor"; Status = "running"; FailureReason = null;
            Enter(Phase.Reach);
            robot.Driver.HoldPosition = true;
            return true;
        }

        /// <summary>Keep the towed person behind the robot; call every tick.</summary>
        public void Tow()
        {
            if (carried == null) return;
            // The sled follows the robot's own trail at the tow distance, so it goes where the robot went (review: placed
            // straight behind the robot, it cut through walls at corners).
            var here = robot.transform.position;
            if (trail.Count == 0 || (here - trail[trail.Count - 1]).sqrMagnitude > .0004f) trail.Add(here);
            float left = rescue.tow_offset_m; var at = trail[trail.Count - 1];
            for (int i = trail.Count - 1; i > 0 && left > 0; i--)
            {
                var a = trail[i]; var b = trail[i - 1]; float d = Vector2.Distance(new Vector2(a.x, a.z), new Vector2(b.x, b.z));
                if (d >= left) { at = Vector3.Lerp(a, b, left / Mathf.Max(d, 1e-6f)); left = 0; }
                else { left -= d; at = b; }
            }
            carried.position = new Vector3(at.x, carried.position.y, at.z);
        }

        /// <summary>Unstrap the towed person where they are; returns their scene name, or null.</summary>
        public string Release()
        {
            if (carried == null) return null;
            foreach (var col in carriedColliders) if (col != null) col.enabled = true;
            string name = carried.name; carried = null; carriedColliders = null;
            return name;
        }

        /// <summary>Spray toward bearingDegRight (right positive, from the robot's heading) for up to seconds, limited by
        /// max_spray_s and the water left. The arm points the nozzle; the robot holds still meanwhile.</summary>
        public bool BeginSpray(float bearingDegRight, float seconds)
        {
            if (IsRunning) return Refuse($"busy with {Action}");
            if (nozzle == null) return Refuse("no nozzle fitted");
            if (!(WaterLitres > 1e-6)) return Refuse("the water tank is empty");
            if (!float.IsFinite(bearingDegRight) || !float.IsFinite(seconds) || seconds <= 0) return Refuse("invalid spray order");
            spraySeconds = Mathf.Min(seconds, nozzle.max_spray_s, (float)(WaterLitres / nozzle.flow_lps));
            sprayHeadingDeg = robot.Driver.YawDeg + bearingDegRight;
            var aim = new Vector3(Mathf.Sin(sprayHeadingDeg * Mathf.Deg2Rad), 0f, Mathf.Cos(sprayHeadingDeg * Mathf.Deg2Rad));
            // The gripper holds the nozzle about a metre up, forward along the spray direction, within the arm's reach.
            contact = robot.transform.position + Vector3.up * 1.0f + aim * 0.5f;
            Action = "spray_water"; TargetId = $"{bearingDegRight:0}deg"; Status = "running"; FailureReason = null;
            Enter(Phase.Reach);
            robot.Driver.HoldPosition = true;
            return true;
        }

        public bool Begin(string action)
        {
            if (IsRunning) return Refuse($"busy with {Action}");
            var position = robot.transform.position;
            float heading = robot.Driver.YawDeg;
            direction = new Vector2(Mathf.Sin(heading * Mathf.Deg2Rad), Mathf.Cos(heading * Mathf.Deg2Rad));
            var forward = new Vector3(direction.x, 0f, direction.y);
            switch (action)
            {
                case "clear_debris":
                {
                    if (obstacles == null) return Refuse("no debris in this scene");
                    var body = obstacles.TryFindMovableAhead(position, heading, ReachAheadM);
                    if (body == null) return Refuse(string.IsNullOrEmpty(obstacles.FailureReason) ? "no movable debris within reach ahead" : obstacles.FailureReason);
                    var colliders = body.GetComponentsInChildren<Collider>();
                    if (colliders.Length == 0) return Refuse($"debris {body.Id} has no collider");
                    var b = colliders[0].bounds;
                    foreach (var col in colliders) b.Encapsulate(col.bounds);
                    float halfAlong = Mathf.Abs(direction.x) * b.extents.x + Mathf.Abs(direction.y) * b.extents.z;
                    contact = new Vector3(b.center.x, b.center.y, b.center.z) - forward * (halfAlong + 0.03f);
                    TargetId = body.Id; pushed = body.transform;
                    break;
                }
                case "open_door":
                {
                    if (doors == null) return Refuse("no doors in this scene");
                    var door = doors.TryFindDoorAhead(position, heading, ReachAheadM);
                    if (door == null) return Refuse(string.IsNullOrEmpty(doors.FailureReason) ? "no closed door within reach ahead" : doors.FailureReason);
                    contact = door.HandlePosition;
                    TargetId = door.Id; pushed = null;
                    break;
                }
                default:
                    return Refuse($"unknown action '{action}'");
            }
            Action = action; Status = "running"; FailureReason = null;
            Enter(Phase.Reach);
            robot.Driver.HoldPosition = true;
            return true;
        }

        public void Step(float dt)
        {
            if (!IsRunning) return;
            phaseTime += dt;
            switch (phase)
            {
                case Phase.Reach:
                    if (!robot.Arm.Reach(Vector3.Lerp(phaseStart, contact, Smooth(phaseTime / ReachSeconds))))
                    { Fail("contact point is out of the arm's reach"); return; }
                    if (phaseTime + 1e-6 < ReachSeconds) return;
                    if (Action == "extract_survivor")
                    {
                        // Strapped on: the body no longer blocks the robot or its LiDAR; it rides the sled behind.
                        carried = pushed; carriedColliders = carried.GetComponentsInChildren<Collider>();
                        trail.Clear(); trail.Add(carried.position); trail.Add(robot.transform.position);
                        foreach (var col in carriedColliders) col.enabled = false;
                    }
                    else if (Action == "spray_water")
                    {
                        float litres = Mathf.Min((float)WaterLitres, nozzle.flow_lps * spraySeconds);
                        WaterLitres -= litres; spraySeconds = litres / nozzle.flow_lps;
                        SprayStarted?.Invoke(robot.Arm.GripperPosition, sprayHeadingDeg * Mathf.Deg2Rad, nozzle.flow_lps, litres / nozzle.flow_lps);
                        ShowJet(true);
                    }
                    else if (Action == "clear_debris")
                    {
                        objectStart = pushed.position;
                        if (!obstacles.Push(TargetId, direction, PushDistanceM, PushSeconds)) { Fail(obstacles.FailureReason); return; }
                    }
                    else if (!doors.Open(TargetId, 175f, DoorSeconds)) { Fail(doors.FailureReason); return; }
                    Enter(Phase.Act);
                    return;
                case Phase.Act:
                    if (Action == "extract_survivor")
                    {
                        if (phaseTime + 1e-6 < rescue.grip_s) return;
                        GripCompleted?.Invoke(carried.name);
                    }
                    else if (Action == "spray_water")
                    {
                        if (phaseTime + 1e-6 < spraySeconds) return;
                        ShowJet(false);
                    }
                    else if (Action == "clear_debris")
                    {
                        robot.Arm.Reach(contact + (pushed.position - objectStart));
                        if (phaseTime + 1e-6 < PushSeconds) return;
                    }
                    else
                    {
                        // Follow the handle while it stays within reach; the door swings away from the arm as it opens.
                        robot.Arm.Reach(doors.Find(TargetId).HandlePosition);
                        if (phaseTime + 1e-6 < DoorSeconds || doors.Find(TargetId).IsMoving) return;
                    }
                    Enter(Phase.Stow);
                    return;
                case Phase.Stow:
                    robot.Arm.Reach(Vector3.Lerp(phaseStart, robot.Arm.StowTipPosition, Smooth(phaseTime / StowSeconds)));
                    if (phaseTime + 1e-6 < StowSeconds) return;
                    robot.Arm.Stow();
                    Status = "succeeded";
                    robot.Driver.HoldPosition = false;
                    return;
            }
        }

        void ShowJet(bool on)
        {
            if (jet == null)
            {
                var go = new GameObject("WaterJet"); go.transform.SetParent(robot.transform, false);
                jet = go.AddComponent<LineRenderer>(); jet.positionCount = 8; jet.useWorldSpace = true;
                jet.widthCurve = AnimationCurve.Linear(0f, .06f, 1f, .5f);
                var shader = Shader.Find("HDRP/Unlit") ?? Shader.Find("Unlit/Color");
                if (shader != null) { jet.material = new Material(shader); jet.material.color = new Color(.75f, .88f, 1f); }
            }
            jet.enabled = on;
            if (!on) return;
            // A shallow arc: the stream leaves the gripper level and drops about half a metre over its range.
            var from = robot.Arm.GripperPosition; float yaw = sprayHeadingDeg * Mathf.Deg2Rad;
            var dir = new Vector3(Mathf.Sin(yaw), 0f, Mathf.Cos(yaw));
            for (int k = 0; k < jet.positionCount; k++)
            {
                float s = k / (float)(jet.positionCount - 1);
                jet.SetPosition(k, from + dir * (nozzle.range_m * s) + Vector3.down * (.5f * s * s));
            }
        }

        void Enter(Phase next) { phase = next; phaseTime = 0; phaseStart = robot.Arm.GripperPosition; }

        static float Smooth(double s) { float x = Mathf.Clamp01((float)s); return x * x * (3f - 2f * x); }

        bool Refuse(string reason) { FailureReason = reason; return false; }

        void Fail(string reason)
        {
            FailureReason = reason; Status = "failed";
            robot.Arm.Stow();
            robot.Driver.HoldPosition = false;
        }
    }
}
