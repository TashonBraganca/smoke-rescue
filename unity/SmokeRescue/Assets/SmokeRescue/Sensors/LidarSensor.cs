using System;
using UnityEngine;

namespace SmokeRescue.Sensors
{
    /// <summary>One sweep of the planar LiDAR. Beam i points at AngleMinRad + i * AngleIncrementRad in the body frame:
    /// 0 is straight ahead, positive turns left (counter-clockwise seen from above). Ranges are metres along the beam
    /// from the sensor centre; 0 means no return (nothing within range, or the pulse was lost in smoke).</summary>
    public sealed class LidarScan
    {
        public readonly double Timestamp;
        public readonly float AngleMinRad, AngleIncrementRad, MinM, MaxM, MountForwardM, MountHeightM;
        public readonly float[] RangesM;
        public LidarScan(double timestamp, float angleMinRad, float angleIncrementRad, float minM, float maxM,
                         float mountForwardM, float mountHeightM, float[] rangesM)
        {
            Timestamp = timestamp; AngleMinRad = angleMinRad; AngleIncrementRad = angleIncrementRad; MinM = minM; MaxM = maxM;
            MountForwardM = mountForwardM; MountHeightM = mountHeightM; RangesM = rangesM;
        }
        public float Angle(int beam) => AngleMinRad + beam * AngleIncrementRad;
    }

    /// <summary>
    /// Planar 360 degree time-of-flight scanner on the robot's back (config/sensors.json "lidar"). It ray casts the
    /// physics colliders: walls, doors, furniture, debris and other robots, so it measures what the body can hit.
    /// Simplifications, stated so nobody mistakes them for the real thing: furniture colliders are boxes, so a table
    /// reads as a solid block rather than four legs; the whole sweep is taken at one instant (no rotation skew); the
    /// mount is levelled (gait sway does not tilt the plane). Range noise is Gaussian, sigma = range_sigma_m +
    /// range_sigma_per_m * range. In smoke a pulse is lost with probability per_unit_density times the mean density
    /// along the beam, capped at max, and a lost pulse reads 0 like a beam that hit nothing.
    /// </summary>
    public sealed class LidarSensor
    {
        readonly SensorConfig.LidarConfig cfg;
        readonly SensorRandom random;
        readonly RaycastHit[] hits = new RaycastHit[32];
        double next = double.NegativeInfinity;
        public bool NoiseEnabled { get; set; } = true;
        /// <summary>Smoke density at a world point, supplied by the world like RgbdCamera.LocalDensity.</summary>
        public Func<Vector3, float> LocalDensity { get; set; } = _ => 0;
        /// <summary>Layers the pulses can hit.</summary>
        public int Mask { get; set; } = ~0;
        /// <summary>This robot's root. Its own colliders never return a pulse.</summary>
        public Transform Self { get; set; }
        public SensorConfig.LidarConfig Config => cfg;

        /// <summary>Wall time in each part of a sweep across all LiDARs, read by the recorder's perf log.</summary>
        public static readonly System.Diagnostics.Stopwatch SyncTime = new System.Diagnostics.Stopwatch(),
            RayTime = new System.Diagnostics.Stopwatch(), SmokeTime = new System.Diagnostics.Stopwatch();

        public LidarSensor(SensorConfig.LidarConfig config, int seed)
        {
            cfg = config ?? throw new ArgumentNullException(nameof(config));
            random = new SensorRandom(seed);
        }

        public float AngleMinRad => -cfg.fov_deg * .5f * Mathf.Deg2Rad;
        public float AngleIncrementRad => cfg.fov_deg * Mathf.Deg2Rad / cfg.beams;

        /// <summary>Advance to t with the body's true pose; returns a sweep when one is due (rate_hz), else null.</summary>
        public LidarScan Step(double t, Vector3 rootPosition, Quaternion rootRotation)
        {
            if (t + 1e-9 < next) return null;
            next = (double.IsNegativeInfinity(next) ? t : next) + 1.0 / cfg.rate_hz;
            // Doors swing and robots move by transform; make the colliders agree with this instant.
            SyncTime.Start(); Physics.SyncTransforms(); SyncTime.Stop();
            float yaw = rootRotation.eulerAngles.y * Mathf.Deg2Rad;
            var forward = new Vector3(Mathf.Sin(yaw), 0f, Mathf.Cos(yaw));
            var right = new Vector3(forward.z, 0f, -forward.x);
            var origin = rootPosition + Vector3.up * cfg.mount.height_m + forward * cfg.mount.forward_offset_m;
            var ranges = new float[cfg.beams];
            for (int i = 0; i < cfg.beams; i++)
            {
                float a = AngleMinRad + i * AngleIncrementRad;
                var dir = forward * Mathf.Cos(a) - right * Mathf.Sin(a);   // positive angle turns left
                RayTime.Start(); float r = Nearest(origin, dir); RayTime.Stop();
                if (r < cfg.min_m || r > cfg.max_m) { ranges[i] = 0f; continue; }
                if (NoiseEnabled)
                {
                    SmokeTime.Start();
                    float density = (LocalDensity(origin + dir * (r * .25f)) + LocalDensity(origin + dir * (r * .5f)) +
                                     LocalDensity(origin + dir * (r * .75f))) / 3f;
                    SmokeTime.Stop();
                    float lost = Mathf.Min(cfg.smoke_dropout.max, cfg.smoke_dropout.per_unit_density * Mathf.Max(0f, density));
                    if (random.Uniform() < lost) { ranges[i] = 0f; continue; }
                    r += random.Gaussian() * (cfg.range_sigma_m + cfg.range_sigma_per_m * r);
                    if (r < cfg.min_m || r > cfg.max_m) { ranges[i] = 0f; continue; }
                }
                ranges[i] = Mathf.Round(r * 1000f) / 1000f;   // the recording stores millimetres
            }
            return new LidarScan(t, AngleMinRad, AngleIncrementRad, cfg.min_m, cfg.max_m, cfg.mount.forward_offset_m, cfg.mount.height_m, ranges);
        }

        float Nearest(Vector3 origin, Vector3 dir)
        {
            int n = Physics.RaycastNonAlloc(origin, dir, hits, cfg.max_m + 0.5f, Mask, QueryTriggerInteraction.Ignore);
            float best = float.PositiveInfinity;
            for (int i = 0; i < n; i++)
            {
                if (hits[i].distance >= best) continue;
                if (Self != null && hits[i].collider.transform.IsChildOf(Self)) continue;
                best = hits[i].distance;
            }
            return best;
        }
    }
}
