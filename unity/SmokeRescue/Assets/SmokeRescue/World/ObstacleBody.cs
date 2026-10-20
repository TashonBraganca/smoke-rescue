using System;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace SmokeRescue.World
{
    /// <summary>Serialized identity and physical estimates on one placed obstacle group. No frame-clock motion.</summary>
    public sealed class ObstacleBody : MonoBehaviour
    {
        [SerializeField] string obstacleId, kind;
        [SerializeField] bool movable;
        [SerializeField] float massKg;
        public string Id => obstacleId;
        public string Kind => kind;
        public bool Movable => movable;
        public float MassKg => massKg;
        public bool IsMoving { get; internal set; }
        public float Progress { get; internal set; }
        public string FailureReason { get; internal set; }

        public void Initialize(string id, string obstacleKind, JObject definitions)
        {
            if (string.IsNullOrWhiteSpace(id)) throw new ArgumentException("An obstacle needs an id.");
            var definition = definitions[obstacleKind];
            if (definition == null) throw new ArgumentException($"No obstacle physics definition for '{obstacleKind}'.");
            float mass = definition.Value<float>("mass_kg");
            if (!DoorRuntime.Finite(mass) || mass <= 0) throw new ArgumentException("Obstacle mass must be positive and finite.");
            obstacleId = id; kind = obstacleKind;
            movable = definition.Value<bool?>("movable") ?? false; massKg = mass;
        }
    }
}
