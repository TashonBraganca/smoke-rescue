using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace SmokeRescue.World
{
    /// <summary>World-side door actuator. The recorder owns one instance and calls Step once per tick.</summary>
    public sealed class DoorRuntime
    {
        public sealed class Door
        {
            public string Id { get; internal set; }
            public Transform Hinge { get; internal set; }
            public BoxCollider Leaf { get; internal set; }
            public bool IsEntrance { get; internal set; }
            public float AngleDeg { get; internal set; }
            public string State => AngleDeg < .01f ? "closed" : AngleDeg >= 174.99f ? "open" : "half";
            public bool IsMoving { get; internal set; }
            public float Progress { get; internal set; }
            // The handle is a fixed point on the leaf, 12 cm from its free edge, at 1 m height.
            public Vector3 HandlePosition => Hinge.TransformPoint(HandleLocal);
            internal Vector3 HandleLocal;
            internal float Sign, Start, Target, Duration;
            internal double Elapsed;
        }

        readonly List<Door> doors = new List<Door>();
        public IReadOnlyList<Door> Doors => doors;
        public string FailureReason { get; private set; }

        /// <summary>Call after HouseBuilder.Build. Scenario overrides change the built hinges, never the FloorPlan.</summary>
        public DoorRuntime(Transform house, FloorPlan plan, Scenario scenario = null)
        {
            if (house == null) throw new ArgumentNullException(nameof(house));
            foreach (var spec in plan.doors)
            {
                var hinge = house.Find($"Doors/Door_{spec.id}/Hinge");
                if (hinge == null) continue; // Removed doors have no actuator.
                var leaf = hinge.Find($"Leaf_{spec.id}")?.GetComponent<BoxCollider>();
                if (leaf == null) throw new InvalidOperationException($"Door {spec.id} has no leaf collider.");
                string roomId = spec.between[0] == "outside" ? spec.between[1] : spec.between[0];
                var room = plan.FindRoom(roomId).Rect;
                float side = Mathf.Sign((spec.RunsAlongX ? room.center.y : room.center.x) -
                    (spec.RunsAlongX ? spec.at[1] : spec.at[0]));
                float sign = spec.RunsAlongX ? -side : side;
                float yaw = Mathf.DeltaAngle(0, hinge.localEulerAngles.y);
                var door = new Door { Id = spec.id, Hinge = hinge, Leaf = leaf, Sign = sign,
                    IsEntrance = spec.between.Contains("outside"), AngleDeg = Mathf.Clamp(yaw * sign, 0, 175),
                    HandleLocal = spec.RunsAlongX ? new Vector3(spec.width - .14f, 1, 0) : new Vector3(0, 1, spec.width - .14f) };
                doors.Add(door);
            }
            if (scenario?.door_states != null)
            {
                // Validate the whole override before moving any hinge.
                foreach (var entry in scenario.door_states)
                {
                    if (Find(entry.Key) == null) throw new ArgumentException($"Unknown or removed door {entry.Key}.");
                    StateAngle(entry.Value);
                }
                foreach (var entry in scenario.door_states) SetAngle(Find(entry.Key), StateAngle(entry.Value));
            }
            Physics.SyncTransforms();
        }

        public Door Find(string doorId) => doors.Find(d => d.Id == doorId);

        public Door TryFindDoorAhead(Vector3 robotPosition, float headingDeg, float reachM)
        {
            ValidateQuery(robotPosition, headingDeg, reachM);
            Physics.SyncTransforms();
            var forward = Quaternion.Euler(0, headingDeg, 0) * Vector3.forward;
            Door nearest = null; float best = float.PositiveInfinity;
            foreach (var door in doors)
            {
                if (door.State == "open" || door.IsMoving) continue;
                var sample = new Vector3(robotPosition.x, door.Leaf.bounds.center.y, robotPosition.z);
                var delta = door.Leaf.ClosestPoint(sample) - sample; delta.y = 0;
                if (Vector3.Dot(delta, forward) <= 0 || delta.magnitude > reachM) continue;
                if (delta.sqrMagnitude < best) { best = delta.sqrMagnitude; nearest = door; }
            }
            return nearest;
        }

        public bool Open(string doorId, float toDeg = 175f, float seconds = 2f) => Begin(doorId, toDeg, seconds);
        public bool Close(string doorId, float toDeg = 0f, float seconds = 2f) => Begin(doorId, toDeg, seconds);

        bool Begin(string id, float target, float seconds)
        {
            FailureReason = null;
            var door = Find(id);
            if (door == null) { FailureReason = $"Unknown or removed door '{id}'."; return false; }
            if (!Finite(target) || target < 0 || target > 175 || !Finite(seconds) || seconds < 0)
            { FailureReason = "Door angle must be 0..175 degrees and duration must be finite and nonnegative."; return false; }
            if (door.IsMoving) { FailureReason = $"Door '{id}' is busy."; return false; }
            door.Start = door.AngleDeg; door.Target = target; door.Duration = seconds; door.Elapsed = 0;
            door.Progress = 0; door.IsMoving = true;
            if (seconds == 0) { SetAngle(door, target); door.Progress = 1; door.IsMoving = false; Physics.SyncTransforms(); }
            return true;
        }

        public void Step(float dt)
        {
            ValidateStep(dt);
            foreach (var door in doors)
            {
                if (!door.IsMoving) continue;
                door.Elapsed += dt;
                door.Progress = Mathf.Clamp01((float)(door.Elapsed / door.Duration));
                SetAngle(door, Mathf.Lerp(door.Start, door.Target, door.Progress));
                if (door.Progress >= 1) door.IsMoving = false;
            }
            Physics.SyncTransforms();
        }

        public void Stop(string doorId)
        {
            var door = Find(doorId);
            if (door != null) door.IsMoving = false;
        }

        static void SetAngle(Door door, float angle)
        {
            door.AngleDeg = angle;
            door.Hinge.localRotation = Quaternion.Euler(0, door.Sign * angle, 0);
        }
        static float StateAngle(string state) => state switch
        { "closed" => 0, "half" => 45, "open" => 175, _ => throw new ArgumentException($"Invalid door state '{state}'.") };
        internal static bool Finite(float value) => !float.IsNaN(value) && !float.IsInfinity(value);
        internal static void ValidateStep(float dt)
        {
            if (!Finite(dt) || dt < 0) throw new ArgumentOutOfRangeException(nameof(dt));
        }
        internal static void ValidateQuery(Vector3 position, float heading, float reach)
        {
            if (!Finite(position.x) || !Finite(position.y) || !Finite(position.z) || !Finite(heading) || !Finite(reach) || reach < 0)
                throw new ArgumentOutOfRangeException(nameof(reach));
        }
    }
}
