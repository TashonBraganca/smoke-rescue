using System;
using System.Collections.Generic;
using UnityEngine;

namespace SmokeRescue.Brain
{
    /// <summary>Building floor frame: x is Unity x, y is Unity z, yaw is left-positive in radians.
    /// This is a navigation frame, not the pitched camera optical frame used in recordings.</summary>
    [Serializable]
    public struct BrainPose
    {
        public Vector2 position;
        public float yawRad;
        public Vector2 Forward => new Vector2(-Mathf.Sin(yawRad), Mathf.Cos(yawRad));
        public Vector2 Transform(float right, float forward) => position +
            new Vector2(Mathf.Cos(yawRad) * right - Mathf.Sin(yawRad) * forward,
                Mathf.Sin(yawRad) * right + Mathf.Cos(yawRad) * forward);
    }
    public readonly struct BrainCommand
    {
        public readonly float forwardMps, yawRateDeg;
        public BrainCommand(float forward, float yaw) { forwardMps = forward; yawRateDeg = yaw; }
    }
    public sealed class BrainEvent
    {
        public double t;
        public string robot, kind, reason;
        // External JSON writers must serialize this dictionary as the numbers object.
        public SortedDictionary<string, double> numbers;
    }
    public sealed class BrainSnapshot
    {
        public BrainPose pose;
        public int gridSize;
        public float cellM;
        public Vector2 gridOrigin;
        // Row-major [row * gridSize + column], increasing row is brain +y.
        // Occupancy: 0 unknown, 1 free, 255 occupied. Arrays are detached copies.
        public byte[] occupancy, peerExplored;
        // 1 where this robot's own camera, or a linked partner's, has searched the floor; row-major like occupancy.
        public byte[] searched;
        /// <summary>Floor this robot's own cameras searched (searched also holds floor received from partners).</summary>
        public byte[] ownSearched;
        public PeerSnapshot[] peers;
        public SurvivorSnapshot[] survivors;
        public float[] heat;
        public Vector2[] path, confirmedSurvivors, fireHotspots;
        public Vector2? goal;
        public string mode, state;
        public float heatBudget;
    }
    /// <summary>Unsearched floor the robot cannot reach (RobotBrain.BlockedUnsearched).</summary>
    public struct BrainBlockedFloor
    {
        public Vector2 point, approach;
        public float bearingDegRight, distanceM, areaM2;
    }
    /// <summary>Unsearched floor reachable only through the fire margin (RobotBrain.FireBlockedFrontier): where the route
    /// first enters the margin, as a bearing (degrees, right positive) and distance from the robot.</summary>
    public struct BrainFireBlock
    {
        public Vector2 blockAt, frontier;
        public float bearingDegRight, distanceM, frontierPathM, unsearchedM2;
        /// <summary>Floor this robot measured burning (heat at or above fire_c, not forecast) within burning_radius_m of
        /// blockAt: the fire water would have to knock down to open the way.</summary>
        public float burningM2;
    }
    /// <summary>A search target offered to the agent (RobotBrain.FrontierOptions).</summary>
    public struct BrainFrontier
    {
        public int id;
        public Vector2 position;
        public float pathM, unsearchedM2;
        /// <summary>False when the robot has no evidence of its own that this floor is indoors (no ceiling seen, walls
        /// not mapped around it). The agent may still choose it, citing an OpenCV check of its own camera frame.</summary>
        public bool provenIndoors;
        /// <summary>Not a frontier: the way in behind the partner (RobotBrain.FollowPartnerGoal), offered when nothing proven is left.</summary>
        public bool towardPartner;
        /// <summary>Indoor floor this robot mapped beyond the frontier that no camera has searched
        /// (OccupancyGrid.UnsearchedFloorBeyond).</summary>
        public float mappedUnsearchedM2;
        /// <summary>Local sensed-map room estimate, or -1 when the ledger is off or no room is known.</summary>
        public int roomId;
    }
    public readonly struct PersonDetection
    {
        public readonly Vector2 position;
        public readonly float areaM2;
        public PersonDetection(Vector2 p, float area) { position = p; areaM2 = area; }
    }
    internal static class BrainMath
    {
        internal static bool Finite(float v) => !float.IsNaN(v) && !float.IsInfinity(v);
        internal static void Time(double t)
        { if (double.IsNaN(t) || double.IsInfinity(t) || t < 0) throw new ArgumentOutOfRangeException(nameof(t)); }
        internal static float Heading(Vector2 delta) => Mathf.Atan2(-delta.x, delta.y);
        internal static float Angle(float a, float b) => Mathf.DeltaAngle(a * Mathf.Rad2Deg, b * Mathf.Rad2Deg);
    }
}
