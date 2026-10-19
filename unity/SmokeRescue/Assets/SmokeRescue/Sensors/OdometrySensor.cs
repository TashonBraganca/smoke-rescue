using System;
using UnityEngine;

namespace SmokeRescue.Sensors
{
    public readonly struct OdometrySample
    {
        public readonly double Timestamp;
        public readonly float Vx, Vy, Wz;
        public OdometrySample(double timestamp, float vx, float vy, float wz) { Timestamp = timestamp; Vx = vx; Vy = vy; Wz = wz; }
    }
    /// <summary>Planar robot convention: vx forward, vy left, wz counterclockwise about up, in m/s and rad/s.
    /// Slip arrivals are Poisson. During a slip both velocity and yaw rate overshoot by the sampled fraction.</summary>
    public sealed class OdometrySensor
    {
        readonly SensorConfig.OdometryConfig config;
        readonly SensorRandom random;
        Vector3 position;
        Quaternion rotation;
        double elapsed, slipRemaining;
        float slipFactor = 1;
        bool primed;
        public bool NoiseEnabled { get; set; } = true;
        public float RateHz => config.rate_hz;
        public OdometrySensor(SensorConfig config, int seed) { config.Validate(); this.config = config.odometry; random = new SensorRandom(seed); }
        public void Reset(Vector3 truePosition, Quaternion trueRotation)
        { position = truePosition; rotation = trueRotation; primed = true; elapsed = 0; slipRemaining = 0; slipFactor = 1; }
        public OdometrySample? Step(double timestamp, Vector3 truePosition, Quaternion trueRotation, float dt)
        {
            SensorMotion.CheckStep(dt, RateHz);
            if (!primed) { Reset(truePosition, trueRotation); return null; }
            elapsed += dt;
            if (elapsed + 1e-7 < 1.0 / RateHz) return null;
            float sampleDt = (float)elapsed;
            var localVelocity = Quaternion.Inverse(trueRotation) * ((truePosition - position) / sampleDt);
            var localAngular = SensorMotion.BodyAngularRate(rotation, trueRotation, sampleDt);
            position = truePosition; rotation = trueRotation; elapsed = 0;
            float factor = 1;
            if (NoiseEnabled)
            {
                slipRemaining = Math.Max(0, slipRemaining - sampleDt);
                if (slipRemaining == 0 && random.Uniform() < 1 - Math.Exp(-config.slip.events_per_minute * sampleDt / 60))
                {
                    slipRemaining = config.slip.duration_s;
                    slipFactor = 1 + config.slip.overshoot_min + (float)random.Uniform() * (config.slip.overshoot_max - config.slip.overshoot_min);
                }
                factor = (1 + config.scale_error) * (slipRemaining > 0 ? slipFactor : 1);
            }
            return new OdometrySample(timestamp, localVelocity.z * factor, -localVelocity.x * factor, -localAngular.y * factor);
        }
    }
}
