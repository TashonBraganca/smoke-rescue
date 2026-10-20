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
    /// <summary>
    /// Fast seeded stream for per-pixel image noise: xoshiro256** (Blackman and Vigna 2018) seeded by SplitMix64, and
    /// normal deviates by the Marsaglia polar method, which yields two per logarithm and keeps the spare. Same
    /// distributions as SensorRandom at a fraction of the cost; a struct, one per image row.
    /// </summary>
    internal struct PixelRandom
    {
        ulong s0, s1, s2, s3;
        double spare; bool hasSpare;
        public PixelRandom(ulong seed)
        {
            s0 = SplitMix(ref seed); s1 = SplitMix(ref seed); s2 = SplitMix(ref seed); s3 = SplitMix(ref seed);
            spare = 0; hasSpare = false;
        }
        static ulong SplitMix(ref ulong x)
        {
            ulong z = x += 0x9E3779B97F4A7C15UL;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
            return z ^ (z >> 31);
        }
        static ulong Rotl(ulong x, int k) => (x << k) | (x >> (64 - k));
        ulong Next()
        {
            ulong result = Rotl(s1 * 5, 7) * 9, t = s1 << 17;
            s2 ^= s0; s3 ^= s1; s1 ^= s2; s0 ^= s3; s2 ^= t; s3 = Rotl(s3, 45);
            return result;
        }
        /// <summary>Uniform in [0, 1) with 53 random bits.</summary>
        public double Uniform() => (Next() >> 11) * (1.0 / (1UL << 53));
        public float Gaussian()
        {
            if (hasSpare) { hasSpare = false; return (float)spare; }
            double u, v, s;
            do { u = Uniform() * 2 - 1; v = Uniform() * 2 - 1; s = u * u + v * v; } while (s >= 1 || s == 0);
            double m = Math.Sqrt(-2 * Math.Log(s) / s);
            spare = v * m; hasSpare = true;
            return (float)(u * m);
        }
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
