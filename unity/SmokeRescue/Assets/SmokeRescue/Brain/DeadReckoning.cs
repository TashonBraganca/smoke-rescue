using System;
using System.Collections.Generic;
using SmokeRescue.Sensors;
using UnityEngine;

namespace SmokeRescue.Brain
{
    /// <summary>Integrates interval-ending sensor velocities and retains capture-time pose history.
    /// Only operator entry and explicit pose corrections set absolute pose. Expect 2% translation scale error plus 10-30% slip overshoot,
    /// and accumulated gyro bias. No simulator pose or odometry yaw is used.</summary>
    public sealed class DeadReckoning
    {
        struct Segment { public double start, end; public BrainPose pose; public float right, forward, yaw; public bool valid; }
        readonly List<OdometrySample> odometry = new List<OdometrySample>();
        readonly List<ImuSample> imu = new List<ImuSample>();
        readonly List<Segment> history = new List<Segment>();
        readonly float stale, retention;
        int oi, ii;
        double now;
        OdometrySample lastO;
        ImuSample lastI;
        bool hasO, hasI;
        // The tail beyond the newest complete sensor interval is a prediction only.
        // Rebuild it when the next interval mean arrives instead of committing it twice.
        struct Checkpoint
        {
            public double now; public BrainPose pose; public int oi, ii, segments;
            public OdometrySample o; public ImuSample i; public bool hasO, hasI;
        }
        Checkpoint complete;
        public BrainPose Pose { get; private set; }
        public float Speed { get; private set; }
        public float YawRate { get; private set; }
        public float ForwardSpeed { get; private set; }
        // Reject every interval with motion or a sensor gap, even if its final sample is zero.
        public bool ForwardStayedNearZero(double from, double to, float threshold)
        {
            double covered = from;
            foreach (var segment in history)
            {
                if (segment.end <= from || segment.start >= to) continue;
                if (segment.start > covered + 1e-8 || !segment.valid || Mathf.Abs(segment.forward) > threshold) return false;
                covered = Math.Max(covered, segment.end);
            }
            return to > from && covered >= to - 1e-8 && Healthy(to);
        }
        public bool Healthy(double t) => hasO && hasI && t - lastO.Timestamp <= stale && t - lastI.Timestamp <= stale;
        public DeadReckoning(float staleSeconds, float historySeconds, BrainPose entry)
        { stale = staleSeconds; retention = historySeconds; Pose = entry; complete.pose = entry; }
        // Left-compose a building-frame rigid delta. Grids keep their existing cells.
        public void ApplyCorrection(BrainPose delta)
        {
            Pose = Correct(Pose, delta);
            complete.pose = Correct(complete.pose, delta);
            for (int i = 0; i < history.Count; i++)
            { var s = history[i]; s.pose = Correct(s.pose, delta); history[i] = s; }
        }
        internal static BrainPose Correct(BrainPose pose, BrainPose delta) => new BrainPose
        { position = delta.Transform(pose.position.x, pose.position.y), yawRad = pose.yawRad + delta.yawRad };
        public void Add(OdometrySample s)
        {
            BrainMath.Time(s.Timestamp);
            if (!BrainMath.Finite(s.Vx) || !BrainMath.Finite(s.Vy)) throw new ArgumentException("Nonfinite odometry.");
            if (s.Timestamp < now || (hasO && s.Timestamp <= lastO.Timestamp)) throw new ArgumentException("Deliver motion before Step at its timestamp.");
            int index = odometry.BinarySearch(s, OdomComparer.Instance);
            if (index >= 0) throw new ArgumentException("Duplicate odometry timestamp.");
            odometry.Insert(~index, s);
        }
        public void Add(ImuSample s)
        {
            BrainMath.Time(s.Timestamp);
            if (!BrainMath.Finite(s.AngularRate.y)) throw new ArgumentException("Nonfinite gyro.");
            if (s.Timestamp < now || (hasI && s.Timestamp <= lastI.Timestamp)) throw new ArgumentException("Deliver motion before Step at its timestamp.");
            int index = imu.BinarySearch(s, ImuComparer.Instance);
            if (index >= 0) throw new ArgumentException("Duplicate IMU timestamp.");
            imu.Insert(~index, s);
        }
        public void Advance(double t)
        {
            BrainMath.Time(t);
            if (t < now) throw new ArgumentException("Motion time must increase.");
            // Discard only the extrapolated tail. Complete intervals keep their capture poses.
            if (history.Count > complete.segments) history.RemoveRange(complete.segments, history.Count - complete.segments);
            now = complete.now; Pose = complete.pose; oi = complete.oi; ii = complete.ii;
            lastO = complete.o; lastI = complete.i; hasO = complete.hasO; hasI = complete.hasI;
            if (oi > 1024) { odometry.RemoveRange(0, oi); oi = 0; }
            if (ii > 1024) { imu.RemoveRange(0, ii); ii = 0; }
            int old = 0;
            while (old < history.Count && history[old].end < t - retention) old++;
            if (old > 0) history.RemoveRange(0, old);
            double odomEnd = hasO ? lastO.Timestamp : 0, imuEnd = hasI ? lastI.Timestamp : 0;
            for (int j = oi; j < odometry.Count && odometry[j].Timestamp <= t; j++) odomEnd = odometry[j].Timestamp;
            for (int j = ii; j < imu.Count && imu[j].Timestamp <= t; j++) imuEnd = imu[j].Timestamp;
            // A long outage cannot be repaired by a late interval mean. Commit the stopped stream after its stale
            // limit, so a missing sensor cannot retain an entire run of the other sensor for replay every step.
            if (t - odomEnd > stale) odomEnd = t;
            if (t - imuEnd > stale) imuEnd = t;
            IntegrateTo(Math.Max(now, Math.Min(odomEnd, imuEnd)));
            complete = new Checkpoint { now = now, pose = Pose, oi = oi, ii = ii, segments = history.Count,
                o = lastO, i = lastI, hasO = hasO, hasI = hasI };
            IntegrateTo(t);
            ForwardSpeed = hasO && t - lastO.Timestamp <= stale ? lastO.Vx : 0;
            Speed = hasO && t - lastO.Timestamp <= stale ? Mathf.Sqrt(lastO.Vx * lastO.Vx + lastO.Vy * lastO.Vy) : 0;
            YawRate = hasI && t - lastI.Timestamp <= stale ? -lastI.AngularRate.y : 0;
        }
        void IntegrateTo(double t)
        {
            while (true)
            {
                while (oi < odometry.Count && odometry[oi].Timestamp <= now + 1e-9) { lastO = odometry[oi++]; hasO = true; }
                while (ii < imu.Count && imu[ii].Timestamp <= now + 1e-9) { lastI = imu[ii++]; hasI = true; }
                if (now >= t - 1e-9) break;
                bool nextO = oi < odometry.Count && odometry[oi].Timestamp <= t;
                bool nextI = ii < imu.Count && imu[ii].Timestamp <= t;
                var o = nextO ? odometry[oi] : lastO;
                var i = nextI ? imu[ii] : lastI;
                bool validO = nextO ? o.Timestamp - (hasO ? lastO.Timestamp : 0) <= stale : hasO && now < lastO.Timestamp + stale - 1e-9;
                bool validI = nextI ? i.Timestamp - (hasI ? lastI.Timestamp : 0) <= stale : hasI && now < lastI.Timestamp + stale - 1e-9;
                double end = t;
                if (nextO) end = Math.Min(end, o.Timestamp);
                else if (validO) end = Math.Min(end, lastO.Timestamp + stale);
                if (nextI) end = Math.Min(end, i.Timestamp);
                else if (validI) end = Math.Min(end, lastI.Timestamp + stale);
                float right = validO ? -o.Vy : 0, forward = validO ? o.Vx : 0;
                // Optical body gyro y is positive for a Unity right turn. Brain yaw is left positive.
                float yaw = validI ? -i.AngularRate.y : 0;
                var segment = new Segment { start = now, end = end, pose = Pose, right = right, forward = forward, yaw = yaw, valid = validO && validI };
                history.Add(segment);
                Pose = Integrate(segment, (float)(end - now)); now = end;
            }
        }
        /// <summary>False when t is older than the retained history (At would return a zero pose).</summary>
        public bool Covers(double t) => t >= now - 1e-9 || (history.Count > 0 && t >= history[0].start);
        public BrainPose At(double t)
        {
            int lo = 0, hi = history.Count - 1;
            while (lo <= hi)
            {
                int m = (lo + hi) / 2; var s = history[m];
                if (t < s.start) hi = m - 1;
                else if (t > s.end) lo = m + 1;
                else return Integrate(s, (float)(t - s.start));
            }
            return t >= now ? Pose : default;
        }
        static BrainPose Integrate(Segment s, float dt)
        {
            float half = s.yaw * dt / 2;
            float scale = Mathf.Abs(half) < 1e-5f ? dt : dt * Mathf.Sin(half) / half;
            var mid = s.pose; mid.yawRad += half;
            return new BrainPose { position = mid.Transform(s.right * scale, s.forward * scale),
                yawRad = s.pose.yawRad + s.yaw * dt };
        }
        sealed class OdomComparer : IComparer<OdometrySample>
        { public static OdomComparer Instance => new OdomComparer(); public int Compare(OdometrySample a, OdometrySample b) => a.Timestamp.CompareTo(b.Timestamp); }
        sealed class ImuComparer : IComparer<ImuSample>
        { public static ImuComparer Instance => new ImuComparer(); public int Compare(ImuSample a, ImuSample b) => a.Timestamp.CompareTo(b.Timestamp); }
    }
}
