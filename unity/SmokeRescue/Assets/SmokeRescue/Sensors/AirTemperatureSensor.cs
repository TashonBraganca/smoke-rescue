using System;
using UnityEngine;

namespace SmokeRescue.Sensors
{
    /// <summary>
    /// Body-mounted thermocouple: reads the air temperature at the robot with a first-order lag (time constant from
    /// sensors.json) and Gaussian noise. The world supplies the true air temperature through a function, the same way
    /// RgbdCamera gets smoke density, so this assembly never sees the fire model.
    /// </summary>
    public sealed class AirTemperatureSensor
    {
        readonly SensorConfig.AirTemperatureConfig cfg;
        readonly System.Random rng;
        double nextSample = double.NegativeInfinity, lastT = double.NaN;
        float reading = float.NaN;
        public bool NoiseEnabled { get; set; } = true;

        public AirTemperatureSensor(SensorConfig.AirTemperatureConfig config, int seed)
        {
            cfg = config ?? throw new ArgumentNullException(nameof(config));
            rng = new System.Random(seed);
        }

        /// <summary>Advance to t with the true air temperature now; returns a reading when one is due (rate_hz).</summary>
        public float? Step(double t, float trueC)
        {
            if (float.IsNaN(reading)) reading = trueC;
            else
            {
                double dt = t - lastT;
                float k = cfg.time_constant_s > 0 ? 1f - Mathf.Exp((float)(-dt / cfg.time_constant_s)) : 1f;
                reading += k * (trueC - reading);
            }
            lastT = t;
            if (t + 1e-9 < nextSample) return null;
            nextSample = (nextSample < 0 ? t : nextSample) + 1.0 / cfg.rate_hz;
            float noise = NoiseEnabled ? (float)(Gauss() * cfg.noise_c) : 0f;
            return reading + noise;
        }

        double Gauss()
        {
            double u1 = 1.0 - rng.NextDouble(), u2 = rng.NextDouble();
            return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        }
    }
}
