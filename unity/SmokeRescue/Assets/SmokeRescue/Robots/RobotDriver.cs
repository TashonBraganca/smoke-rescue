using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json.Linq;
using SmokeRescue.World;
using UnityEngine;

namespace SmokeRescue.Robots
{
    /// <summary>Motion limits from config/sensors.json "limits".</summary>
    public struct RobotLimits
    {
        public float maxForward, maxReverse, maxYawRateDeg, maxAccel;

        public static RobotLimits Load()
        {
            var j = JObject.Parse(File.ReadAllText(Path.Combine(FloorPlan.ConfigDir, "sensors.json")))["limits"];
            return new RobotLimits
            {
                maxForward = j.Value<float>("max_forward_mps"),
                maxReverse = j.Value<float>("max_reverse_mps"),
                maxYawRateDeg = j.Value<float>("max_yaw_rate_dps"),
                maxAccel = j.Value<float>("max_accel_mps2"),
            };
        }
    }

    /// <summary>
    /// Scripted waypoint driver for one robot: pure pursuit with turn-in-place, doorway slow-down and
    /// look-around sweeps. Pure kinematics stepped by the caller with a fixed dt, so a recording is
    /// identical every run. This is the simulator's driver, not robot-side autonomy: it may use the
    /// floor plan's door positions to slow down, exactly like a human teleoperator would.
    /// </summary>
    public class RobotDriver
    {
        enum Mode { Driving, Turning, LookingAround, Done }

        readonly Scenario.Robot spec;
        readonly Scenario.LookAround look;
        readonly RobotLimits limits;
        readonly List<Vector2> route = new List<Vector2>();
        readonly List<Vector2> doors = new List<Vector2>();
        readonly HashSet<int> lookIndices = new HashSet<int>();

        public Vector2 Position { get; private set; }     // x, z in Unity world metres
        public float YawDeg { get; private set; }         // about +y, 0 faces +z, 90 faces +x
        public float Speed { get; private set; }          // forward m/s
        public float YawRateDeg { get; private set; }     // deg/s
        public float Distance { get; private set; }       // metres travelled
        public bool Finished => mode == Mode.Done;
        public int TargetIndex => target;
        public float GaitPhase { get; private set; }      // radians, advances with distance

        Mode mode = Mode.Driving;
        int target;
        float lookTimer, lookStartYaw;

        const float Lookahead = 0.45f;
        const float WaypointTolerance = 0.12f;
        const float TurnInPlaceDeg = 35f;
        const float DoorSlowRadius = 0.9f;
        const float StrideLength = 0.25f;
        const float TurnInPlaceDps = 30f;                // metres per gait cycle, one trot step pair

        public RobotDriver(Scenario.Robot spec, Scenario.LookAround look, RobotLimits limits, FloorPlan plan)
        {
            this.spec = spec;
            this.look = look;
            this.limits = limits;
            Position = new Vector2(spec.spawn.at[0], spec.spawn.at[1]);
            YawDeg = spec.spawn.yaw;
            foreach (var p in spec.route) route.Add(new Vector2(p[0], p[1]));
            foreach (var d in plan.doors) doors.Add(new Vector2(d.at[0], d.at[1]));
            // A look-around point is either a waypoint or lies on a route segment (spawn included);
            // in the second case insert it as a waypoint so the robot stops there.
            foreach (var l in spec.look_around_at)
            {
                var lp = new Vector2(l[0], l[1]);
                if (route.Exists(w => Vector2.Distance(w, lp) < 0.05f)) continue;
                for (int i = 0; i < route.Count; i++)
                {
                    Vector2 a = i == 0 ? Position : route[i - 1], b = route[i];
                    if (DistanceToSegment(lp, a, b) < 0.05f) { route.Insert(i, lp); break; }
                }
            }
            for (int i = 0; i < route.Count; i++)
                foreach (var l in spec.look_around_at)
                    if (Vector2.Distance(route[i], new Vector2(l[0], l[1])) < 0.05f) lookIndices.Add(i);
            if (lookIndices.Count != spec.look_around_at.Count)
                throw new System.InvalidOperationException($"{spec.id}: {spec.look_around_at.Count - lookIndices.Count} look_around_at point(s) are not on the route");
        }

        public void Step(float dt)
        {
            float vCmd = 0f, wCmd = 0f;
            switch (mode)
            {
                case Mode.Done:
                    break;
                case Mode.LookingAround:
                    (vCmd, wCmd) = LookAroundCommand(dt);
                    break;
                default:
                    (vCmd, wCmd) = PursuitCommand();
                    break;
            }
            // Acceleration and rate limits.
            float dvMax = limits.maxAccel * dt;
            Speed = Mathf.Clamp(Mathf.Clamp(vCmd, Speed - dvMax, Speed + dvMax), -limits.maxReverse, limits.maxForward);
            YawRateDeg = Mathf.Clamp(wCmd, -limits.maxYawRateDeg, limits.maxYawRateDeg);
            YawDeg = Mathf.Repeat(YawDeg + YawRateDeg * dt, 360f);
            Position += Forward * (Speed * dt);
            Distance += Mathf.Abs(Speed * dt);
            GaitPhase = Distance / StrideLength * 2f * Mathf.PI + Mathf.Abs(YawRateDeg) * 0.004f;
        }

        static float DistanceToSegment(Vector2 p, Vector2 a, Vector2 b)
        {
            Vector2 ab = b - a;
            float t = ab.sqrMagnitude > 1e-9f ? Mathf.Clamp01(Vector2.Dot(p - a, ab) / ab.sqrMagnitude) : 0f;
            return Vector2.Distance(p, a + ab * t);
        }

        public Vector2 Forward => new Vector2(Mathf.Sin(YawDeg * Mathf.Deg2Rad), Mathf.Cos(YawDeg * Mathf.Deg2Rad));

        (float, float) PursuitCommand()
        {
            // Advance through reached waypoints.
            while (target < route.Count && Vector2.Distance(Position, route[target]) < WaypointTolerance)
            {
                if (lookIndices.Remove(target))
                {
                    mode = Mode.LookingAround;
                    lookTimer = 0f;
                    lookStartYaw = YawDeg;
                    target++;
                    return (0f, 0f);
                }
                target++;
            }
            if (target >= route.Count)
            {
                mode = Mode.Done;
                return (0f, 0f);
            }

            Vector2 goal = LookaheadPoint();
            Vector2 to = goal - Position;
            float headingErr = Mathf.DeltaAngle(YawDeg, Mathf.Atan2(to.x, to.y) * Mathf.Rad2Deg);
            float wMax = limits.maxYawRateDeg;

            if (mode == Mode.Turning || Mathf.Abs(headingErr) > TurnInPlaceDeg)
            {
                // Stop, then rotate in place until roughly aligned.
                mode = Mathf.Abs(headingErr) < 4f ? Mode.Driving : Mode.Turning;
                // Turn in place at half the yaw-rate limit: 2 degrees per camera frame keeps visual tracking stable.
                float wTurn = Mathf.Min(wMax, TurnInPlaceDps);
                float w = Mathf.Clamp(headingErr * 2.5f, -wTurn, wTurn);
                return (0f, Speed > 0.02f ? 0f : w);
            }

            float cruise = Mathf.Min(spec.cruise_speed, limits.maxForward);
            float nearestDoor = float.MaxValue;
            foreach (var d in doors) nearestDoor = Mathf.Min(nearestDoor, Vector2.Distance(Position, d));
            float v = nearestDoor < DoorSlowRadius ? Mathf.Min(cruise, spec.doorway_speed) : cruise;
            // Slow down before the final waypoint and before look-around stops.
            float toWaypoint = Vector2.Distance(Position, route[target]);
            if (target == route.Count - 1 || lookIndices.Contains(target))
                v = Mathf.Min(v, Mathf.Sqrt(2f * limits.maxAccel * 0.6f * Mathf.Max(0f, toWaypoint - WaypointTolerance * 0.5f)) + 0.05f);
            // Pure pursuit: curvature = 2 sin(alpha) / L, yaw rate = v * curvature.
            float alpha = headingErr * Mathf.Deg2Rad;
            float curvature = 2f * Mathf.Sin(alpha) / Mathf.Max(to.magnitude, 0.05f);
            float wDeg = v * curvature * Mathf.Rad2Deg;
            if (Mathf.Abs(wDeg) > wMax) v *= wMax / Mathf.Abs(wDeg);  // slow down for sharp curves rather than skid
            return (v, Mathf.Clamp(wDeg, -wMax, wMax));
        }

        Vector2 LookaheadPoint()
        {
            // Walk along the remaining polyline from the robot's projection, Lookahead metres ahead,
            // but never past the current target so corners are honoured.
            Vector2 a = target > 0 ? route[target - 1] : Position;
            Vector2 b = route[target];
            Vector2 ab = b - a;
            float t = ab.sqrMagnitude > 1e-6f ? Mathf.Clamp01(Vector2.Dot(Position - a, ab) / ab.sqrMagnitude) : 1f;
            Vector2 proj = a + ab * t;
            float remain = Vector2.Distance(proj, b);
            return remain <= Lookahead ? b : proj + ab.normalized * Lookahead;
        }

        (float, float) LookAroundCommand(float dt)
        {
            // Sweep left by half the sweep, right by the full sweep, back to centre.
            float half = look.sweep_deg / 2f, speed = Mathf.Min(look.speed_dps, limits.maxYawRateDeg);
            float total = (half + look.sweep_deg + half) / speed;
            lookTimer += dt;
            if (Speed > 0.02f) return (0f, 0f);
            if (lookTimer >= total)
            {
                mode = Mode.Driving;
                return (0f, 0f);
            }
            float t = lookTimer * speed;
            float dir = t < half ? -1f : t < half + look.sweep_deg ? 1f : -1f;
            return (0f, dir * speed);
        }
    }
}
