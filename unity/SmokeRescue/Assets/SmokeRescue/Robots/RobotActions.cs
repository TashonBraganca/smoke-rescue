using System;
using SmokeRescue.World;
using UnityEngine;

namespace SmokeRescue.Robots
{
    /// <summary>
    /// Physical actions an agent can order: "open_door" and "clear_debris". Each runs reach, act, stow on sim time and
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

        enum Phase { Reach, Act, Stow }
        Phase phase;
        double phaseTime;
        Vector3 phaseStart, contact, objectStart;
        Vector2 direction;
        Transform pushed;

        public RobotActions(Robot robot, DoorRuntime doors, ObstacleRuntime obstacles)
        {
            this.robot = robot ?? throw new ArgumentNullException(nameof(robot));
            this.doors = doors;
            this.obstacles = obstacles;
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
                    var b = body.GetComponent<Collider>().bounds;
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
                    if (Action == "clear_debris")
                    {
                        objectStart = pushed.position;
                        if (!obstacles.Push(TargetId, direction, PushDistanceM, PushSeconds)) { Fail(obstacles.FailureReason); return; }
                    }
                    else if (!doors.Open(TargetId, 175f, DoorSeconds)) { Fail(doors.FailureReason); return; }
                    Enter(Phase.Act);
                    return;
                case Phase.Act:
                    if (Action == "clear_debris")
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
