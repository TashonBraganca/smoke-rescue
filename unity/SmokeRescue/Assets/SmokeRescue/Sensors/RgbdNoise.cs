using System;
using UnityEngine;

namespace SmokeRescue.Sensors
{
    /// <summary>One seeded stream per camera, consumed in pixel order after both readbacks complete.</summary>
    public sealed class RgbdNoise
    {
        readonly SensorConfig config;
        readonly SensorRandom random;
        public RgbdNoise(SensorConfig config, int seed) { config.Validate(); this.config = config; random = new SensorRandom(seed); }
        public ushort[] Process(byte[] rgb, float[] depthMetres, Pose cameraPose, bool noiseEnabled,
            Func<Vector3, float> localDensity = null, float nearFire = 0)
        {
            var k = config.camera;
            if (rgb.Length != k.width * k.height * 3 || depthMetres.Length != k.width * k.height)
                throw new ArgumentException("RGB/depth dimensions do not match calibration.");
            var output = new ushort[depthMetres.Length];
            var invalid = new bool[output.Length];
            var d = config.depth;
            for (int i = 0; i < output.Length; i++) invalid[i] = !InRange(depthMetres[i]);
            if (noiseEnabled)
            {
                // A jump greater than 5 cm or three expected depth sigmas is a discontinuity.
                // Mark both sides, then expand by the configured pixel radius. No invalid-mask cascade.
                var edges = new bool[output.Length];
                for (int y = 0; y < k.height; y++)
                for (int x = 0; x < k.width; x++)
                {
                    int i = y * k.width + x;
                    if (x + 1 < k.width && Discontinuous(depthMetres[i], depthMetres[i + 1])) edges[i] = edges[i + 1] = true;
                    if (y + 1 < k.height && Discontinuous(depthMetres[i], depthMetres[i + k.width])) edges[i] = edges[i + k.width] = true;
                }
                if (d.edge_dropout_px > 0)
                for (int y = 0; y < k.height; y++)
                for (int x = 0; x < k.width; x++)
                {
                    if (!edges[y * k.width + x]) continue;
                    for (int dy = -d.edge_dropout_px; dy <= d.edge_dropout_px; dy++)
                    for (int dx = -d.edge_dropout_px; dx <= d.edge_dropout_px; dx++)
                        if (x + dx >= 0 && x + dx < k.width && y + dy >= 0 && y + dy < k.height)
                            invalid[(y + dy) * k.width + x + dx] = true;
                }
            }
            float flicker = noiseEnabled ? 1 + ((float)random.Uniform() * 2 - 1) * config.rgb_noise.exposure_flicker_near_fire * Mathf.Clamp01(nearFire) : 1;
            for (int y = 0; y < k.height; y++)
            for (int x = 0; x < k.width; x++)
            {
                int i = y * k.width + x;
                if (noiseEnabled)
                    for (int c = 0; c < 3; c++) rgb[i * 3 + c] = (byte)Mathf.Clamp(Mathf.RoundToInt(rgb[i * 3 + c] * flicker + random.Gaussian() * config.rgb_noise.gaussian_sigma_8bit), 0, 255);
                float z = depthMetres[i];
                if (invalid[i]) continue;
                var point = FrameConvert.BackProject(x, y, z, k);
                if (noiseEnabled)
                {
                    if (x > 0 && x < k.width - 1 && y > 0 && y < k.height - 1)
                    {
                        float l = depthMetres[i - 1], r = depthMetres[i + 1], u = depthMetres[i - k.width], b = depthMetres[i + k.width];
                        if (InRange(l) && InRange(r) && InRange(u) && InRange(b))
                        {
                            var horizontal = FrameConvert.BackProject(x + 1, y, r, k) - FrameConvert.BackProject(x - 1, y, l, k);
                            var vertical = FrameConvert.BackProject(x, y + 1, b, k) - FrameConvert.BackProject(x, y - 1, u, k);
                            var normal = Vector3.Cross(horizontal, vertical).normalized;
                            float cosine = Mathf.Abs(Vector3.Dot(normal, point.normalized));
                            if (cosine < Mathf.Cos(d.grazing_angle_invalid_deg * Mathf.Deg2Rad)) continue;
                        }
                    }
                    // Sample density halfway along the optical ray in Unity world metres.
                    // Geometry depth itself never includes fog; density affects only deliberate dropout.
                    var unityPoint = cameraPose.position + cameraPose.rotation * FrameConvert.Point(point * 0.5f);
                    float density = Mathf.Max(0, localDensity?.Invoke(unityPoint) ?? 0);
                    if (random.Uniform() < Mathf.Min(d.smoke_dropout.max, d.smoke_dropout.per_unit_density * density)) continue;
                    z += random.Gaussian() * d.sigma_coeff_mm_per_m2 * z * z * 0.001f;
                    if (!InRange(z)) continue;
                }
                output[i] = (ushort)Mathf.Clamp(Mathf.RoundToInt(z * 1000), 1, ushort.MaxValue);
            }
            return output;
        }
        bool InRange(float z) => !float.IsNaN(z) && z >= config.depth.min_m && z <= config.depth.max_m;
        bool Discontinuous(float a, float b) => InRange(a) != InRange(b) ||
            (InRange(a) && InRange(b) && Mathf.Abs(a - b) > Mathf.Max(0.05f, 3 * config.depth.sigma_coeff_mm_per_m2 * Mathf.Max(a * a, b * b) * 0.001f));
    }
}
