using System;
using UnityEngine;

namespace SmokeRescue.Sensors
{
    internal sealed class SensorRandom
    {
        readonly System.Random random;
        public SensorRandom(int seed) { random = new System.Random(seed); }
        public double Uniform() => random.NextDouble();
        public float Gaussian() => (float)(Math.Sqrt(-2 * Math.Log(1 - random.NextDouble())) * Math.Cos(2 * Math.PI * random.NextDouble()));
        public Vector3 Gaussian3(float sigma) => new Vector3(Gaussian(), Gaussian(), Gaussian()) * sigma;
    }
    internal static class SensorMotion
    {
        public static Vector3 BodyAngularRate(Quaternion previous, Quaternion current, float dt)
        {
            var delta = Quaternion.Inverse(previous) * current;
            if (delta.w < 0) delta = new Quaternion(-delta.x, -delta.y, -delta.z, -delta.w);
            delta.ToAngleAxis(out float angle, out Vector3 axis);
            return angle < 0.00001f ? Vector3.zero : axis * (Mathf.DeltaAngle(0, angle) * Mathf.Deg2Rad / dt);
        }
        public static void CheckStep(float dt, float rate)
        {
            if (!(dt > 0) || float.IsInfinity(dt) || dt > 1f / rate + 1e-6f)
                throw new ArgumentOutOfRangeException(nameof(dt), "Step at or above the sensor rate; large steps would lose samples.");
        }
    }
}
