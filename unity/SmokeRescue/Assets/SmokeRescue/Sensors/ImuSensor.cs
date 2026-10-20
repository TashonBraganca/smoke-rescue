using UnityEngine;

namespace SmokeRescue.Sensors
{
    public readonly struct ImuSample
    {
        public readonly double Timestamp;
        public readonly Vector3 SpecificForce, AngularRate;
        public ImuSample(double timestamp, Vector3 force, Vector3 angular) { Timestamp = timestamp; SpecificForce = force; AngularRate = angular; }
    }
    /// <summary>Caller supplies truth at fixed steps. Output is right-handed body x right, y down, z forward.
    /// Specific force is S R^-1 (a - gravity); gyro is -S times Unity's body angle-axis rate.
    /// Densities use continuous-time units, white sigma = density / sqrt(sample period), walk sigma = walk sqrt(dt).</summary>
    public sealed class ImuSensor
    {
        readonly SensorConfig.ImuConfig config;
        readonly SensorRandom random;
        Vector3 position, velocity, accelBias, gyroBias;
        Quaternion sampleStart;
        bool primed, velocityKnown;
        double elapsed;
        public bool NoiseEnabled { get; set; } = true;
        public float RateHz => config.rate_hz;
        public ImuSensor(SensorConfig config, int seed) { config.Validate(); this.config = config.imu; random = new SensorRandom(seed); }
        /// <summary>Prime with known initial velocity to get a sample on the first step. Otherwise two steps warm up derivatives.</summary>
        public void Reset(Vector3 truePosition, Quaternion trueRotation, Vector3 trueVelocity = default)
        {
            position = truePosition; sampleStart = trueRotation; velocity = trueVelocity;
            primed = velocityKnown = true; elapsed = 0; accelBias = gyroBias = Vector3.zero;
        }
        public ImuSample? Step(double timestamp, Vector3 truePosition, Quaternion trueRotation, float dt)
        {
            SensorMotion.CheckStep(dt, RateHz);
            if (!primed) { position = truePosition; sampleStart = trueRotation; primed = true; return null; }
            var nextVelocity = (truePosition - position) / dt;
            var acceleration = (nextVelocity - velocity) / dt;
            position = truePosition; velocity = nextVelocity;
            if (!velocityKnown) { velocityKnown = true; sampleStart = trueRotation; return null; }
            elapsed += dt;
            if (NoiseEnabled)
            {
                accelBias += random.Gaussian3(config.accel_bias_random_walk * Mathf.Sqrt(dt));
                gyroBias += random.Gaussian3(config.gyro_bias_random_walk * Mathf.Sqrt(dt));
            }
            if (elapsed + 1e-7 < 1.0 / RateHz) return null;
            float sampleDt = (float)elapsed; elapsed = 0;
            // A gyro reports the mean rate over its sample period, not the last simulation tick.
            var angular = FrameConvert.Angular(SensorMotion.BodyAngularRate(sampleStart, trueRotation, sampleDt));
            sampleStart = trueRotation;
            var force = FrameConvert.Point(Quaternion.Inverse(trueRotation) * (acceleration - new Vector3(0, -9.81f, 0)));
            if (NoiseEnabled)
            {
                force += accelBias + random.Gaussian3(config.accel_noise_density / Mathf.Sqrt(sampleDt));
                angular += gyroBias + random.Gaussian3(config.gyro_noise_density / Mathf.Sqrt(sampleDt));
            }
            return new ImuSample(timestamp, force, angular);
        }
    }
}
