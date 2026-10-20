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
            // Angle from the vector part with atan2. ToAngleAxis uses acos(w), and for one 600 Hz tick at 30 deg/s
            // w is within two float steps of 1, which put 20 to 35 degree heading errors into the recorded gyro.
            var v = new Vector3(delta.x, delta.y, delta.z);
            float s = v.magnitude;
            if (s < 1e-12f) return Vector3.zero;
            return v / s * (2f * Mathf.Atan2(s, delta.w) / dt);
        }
        public static void CheckStep(float dt, float rate)
        {
            if (!(dt > 0) || float.IsInfinity(dt) || dt > 1f / rate + 1e-6f)
                throw new ArgumentOutOfRangeException(nameof(dt), "Step at or above the sensor rate; large steps would lose samples.");
        }
    }
}
