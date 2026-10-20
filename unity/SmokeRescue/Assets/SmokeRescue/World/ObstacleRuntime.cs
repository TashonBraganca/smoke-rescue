using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace SmokeRescue.World
{
    /// <summary>Kinematic floor pushes with overlap and swept-box checks. The recorder steps this once per tick.</summary>
    public sealed class ObstacleRuntime
    {
        sealed class Motion
        {
            public ObstacleBody Body;
            public Vector3 Start, End;
            public float Duration;
            public double Elapsed;
        }
        readonly List<ObstacleBody> obstacles;
        readonly List<Motion> motions = new List<Motion>();
        readonly int solid = LayerMask.GetMask("Walls", "Props", "Robots");
        public IReadOnlyList<ObstacleBody> Obstacles => obstacles;
        public string FailureReason { get; private set; }
        public static JObject LoadDefinitions() => (JObject)JObject.Parse(File.ReadAllText(
            Path.Combine(FloorPlan.ConfigDir, "props.json")))["obstacle_physics"];

        public ObstacleRuntime(Transform obstacleRoot)
        {
            if (obstacleRoot == null) throw new ArgumentNullException(nameof(obstacleRoot));
            obstacles = obstacleRoot.GetComponentsInChildren<ObstacleBody>().OrderBy(b => b.Id, StringComparer.Ordinal).ToList();
            if (obstacles.Select(b => b.Id).Distinct().Count() != obstacles.Count)
                throw new ArgumentException("Obstacle ids must be unique.");
        }
        public ObstacleBody Find(string obstacleId) => obstacles.Find(b => b.Id == obstacleId);

        public ObstacleBody TryFindMovableAhead(Vector3 robotPosition, float headingDeg, float reachM)
        {
            DoorRuntime.ValidateQuery(robotPosition, headingDeg, reachM);
            Physics.SyncTransforms();
            var forward = Quaternion.Euler(0, headingDeg, 0) * Vector3.forward;
            ObstacleBody nearest = null; float best = float.PositiveInfinity;
            foreach (var body in obstacles)
            {
                if (!body.Movable || body.IsMoving) continue;
                foreach (var collider in body.GetComponentsInChildren<Collider>())
                {
                    if (!collider.enabled || collider.isTrigger) continue;
                    var sample = new Vector3(robotPosition.x, collider.bounds.center.y, robotPosition.z);
                    var delta = collider.ClosestPoint(sample) - sample; delta.y = 0;
                    if (Vector3.Dot(delta, forward) <= 0 || delta.magnitude > reachM || delta.sqrMagnitude >= best) continue;
                    best = delta.sqrMagnitude; nearest = body;
                }
            }
            return nearest;
        }

        public bool Push(string obstacleId, Vector2 directionXZ, float distanceM = .6f, float seconds = 1.5f)
        {
            FailureReason = null;
            var body = Find(obstacleId);
            if (body == null) return Refuse(null, $"Unknown obstacle '{obstacleId}'.");
            if (!body.Movable) return Refuse(body, $"Obstacle '{obstacleId}' is heavy ({body.MassKg:0.#} kg estimate) and cannot be pushed.");
            if (body.IsMoving) { FailureReason = $"Obstacle '{obstacleId}' is busy."; return false; }
            if (!DoorRuntime.Finite(directionXZ.x) || !DoorRuntime.Finite(directionXZ.y) || directionXZ.sqrMagnitude < 1e-8f ||
                !DoorRuntime.Finite(distanceM) || distanceM <= 0 || !DoorRuntime.Finite(seconds) || seconds <= 0)
                return Refuse(body, "Push direction must be nonzero; distance and duration must be positive and finite.");
            var direction = directionXZ.normalized;
            var delta = new Vector3(direction.x, 0, direction.y) * distanceM;
            if (!PathClear(body, delta, out string reason)) return Refuse(body, reason);
            body.FailureReason = null; body.Progress = 0; body.IsMoving = true;
            motions.Add(new Motion { Body = body, Start = body.transform.position,
                End = body.transform.position + delta, Duration = seconds });
            return true;
        }

        public void Step(float dt)
        {
            DoorRuntime.ValidateStep(dt);
            // List order follows accepted commands, so simultaneous pushes have a stable order.
            for (int i = 0; i < motions.Count;)
            {
                var motion = motions[i]; var body = motion.Body;
                motion.Elapsed += dt;
                float t = Mathf.Clamp01((float)(motion.Elapsed / motion.Duration));
                var next = Vector3.Lerp(motion.Start, motion.End, t);
                if (!PathClear(body, next - body.transform.position, out string reason))
                {
                    Refuse(body, reason); body.IsMoving = false; motions.RemoveAt(i); continue;
                }
                body.transform.position = next; body.Progress = t;
                Physics.SyncTransforms();
                if (t >= 1) { body.IsMoving = false; motions.RemoveAt(i); } else i++;
            }
        }

        public void Stop(string obstacleId)
        {
            var body = Find(obstacleId);
            if (body == null) return;
            body.IsMoving = false;
            motions.RemoveAll(m => m.Body == body);
        }

        bool Refuse(ObstacleBody body, string reason)
        {
            FailureReason = reason;
            if (body != null) body.FailureReason = reason;
            return false;
        }

        bool PathClear(ObstacleBody body, Vector3 delta, out string reason)
        {
            Physics.SyncTransforms(); reason = null;
            var colliders = body.GetComponentsInChildren<Collider>().Where(c => c.enabled && !c.isTrigger).ToArray();
            if (colliders.Length == 0) { reason = $"Obstacle '{body.Id}' has no collider."; return false; }
            foreach (var collider in colliders)
            {
                Vector3 center, half; Quaternion rotation;
                if (collider is BoxCollider box)
                {
                    center = box.transform.TransformPoint(box.center);
                    half = Vector3.Scale(box.size, Abs(box.transform.lossyScale)) * .5f;
                    rotation = box.transform.rotation;
                }
                else { center = collider.bounds.center; half = collider.bounds.extents; rotation = Quaternion.identity; }
                // One millimetre tolerance avoids treating touching faces as penetration.
                half = Vector3.Max(half - Vector3.one * .001f, Vector3.one * .0001f);
                bool Other(Collider hit) => !hit.transform.IsChildOf(body.transform);
                var overlap = Physics.OverlapBox(center + delta, half, rotation, solid, QueryTriggerInteraction.Ignore).FirstOrDefault(Other);
                if (overlap == null && delta.sqrMagnitude > 1e-10f)
                    overlap = Physics.BoxCastAll(center, half, delta.normalized, rotation, delta.magnitude, solid,
                        QueryTriggerInteraction.Ignore).Select(h => h.collider).FirstOrDefault(Other);
                if (overlap == null) continue;
                reason = $"Push blocked by '{overlap.name}'."; return false;
            }
            return true;
        }
        static Vector3 Abs(Vector3 v) => new Vector3(Mathf.Abs(v.x), Mathf.Abs(v.y), Mathf.Abs(v.z));
    }
}
