using System;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace SmokeRescue.Tests.EditMode
{
    /// <summary>Evaluator-only request errors. Bias and noise are left-positive gyro rates in degrees per second.
    /// Initial heading error is right-positive, like EntryPose. Omit motion_error to retain exact sensors.</summary>
    public sealed class HeadlessMotionErrors
    {
        readonly System.Random random;
        readonly float bias, noise;
        public float InitialHeadingDeg { get; }
        public HeadlessMotionErrors(JObject options, int seed)
        {
            bias = (float?)options?["gyro_bias_dps"] ?? 0; noise = (float?)options?["gyro_noise_dps"] ?? 0;
            InitialHeadingDeg = (float?)options?["initial_heading_error_deg"] ?? 0;
            if (float.IsNaN(bias) || float.IsInfinity(bias) || float.IsNaN(noise) || float.IsInfinity(noise) || noise < 0 ||
                float.IsNaN(InitialHeadingDeg) || float.IsInfinity(InitialHeadingDeg)) throw new ArgumentException("Invalid headless motion error.");
            random = new System.Random((int?)options?["seed"] ?? seed);
        }
        public float Gyro(float exactRate)
        {
            if (bias == 0 && noise == 0) return exactRate;
            float gaussian = noise == 0 ? 0 : (float)(Math.Sqrt(-2 * Math.Log(1 - random.NextDouble())) * Math.Cos(2 * Math.PI * random.NextDouble()));
            return exactRate + (bias + noise * gaussian) * Mathf.Deg2Rad;
        }
    }
}
