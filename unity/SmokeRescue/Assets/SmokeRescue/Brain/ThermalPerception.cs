using System;
using System.Collections.Generic;
using SmokeRescue.Sensors;
using UnityEngine;

namespace SmokeRescue.Brain
{
    public static class ThermalPerception
    {
        /// <summary>Parallel optical axes allow direction registration through the two intrinsics.
        /// The configured 3 cm camera separation is ignored. Pixels without matching depth are unusable.</summary>
        internal static bool Match(int x, int y, RgbdFrame rgb, SensorConfig s, out int u, out int v, out float z)
        {
            var a = s.thermal; var b = s.camera;
            u = Mathf.RoundToInt((x - a.cx) / a.fx * b.fx + b.cx);
            v = Mathf.RoundToInt((y - a.cy) / a.fy * b.fy + b.cy); z = 0;
            if (u < 0 || v < 0 || u >= rgb.Width || v >= rgb.Height) return false;
            z = rgb.Depth[v * rgb.Width + u] * .001f;
            return z >= s.depth.min_m && z <= s.depth.max_m;
        }
        public static List<PersonDetection> Detect(ThermalFrame thermal, RgbdFrame rgb, SensorConfig s, BrainConfig c, BrainPose pose)
        {
            int count = thermal.Width * thermal.Height;
            var depth = new float[count]; var seen = new bool[count]; var queue = new Queue<int>();
            for (int y = 0; y < thermal.Height; y++) for (int x = 0; x < thermal.Width; x++)
            {
                float temperature = thermal.Celsius(x, y);
                if (temperature >= c.person_min_c && temperature <= c.person_max_c && Match(x, y, rgb, s, out _, out _, out float z))
                    depth[y * thermal.Width + x] = z;
            }
            var result = new List<PersonDetection>(); var heights = new List<float>();
            for (int start = 0; start < count; start++)
            {
                if (seen[start] || depth[start] == 0) continue;
                seen[start] = true; queue.Enqueue(start); int pixels = 0; float area = 0; Vector2 sum = Vector2.zero; heights.Clear();
                while (queue.Count > 0)
                {
                    int index = queue.Dequeue(), x = index % thermal.Width, y = index / thermal.Width;
                    float z = depth[index]; Match(x, y, rgb, s, out int u, out int v, out _);
                    var p = OccupancyGrid.BackProject(u, v, z, s);
                    // Pixel area on a plane normal to the optical axis, not a world-floor area.
                    float pixelArea = z * z / (s.thermal.fx * s.thermal.fy);
                    sum += pose.Transform(p.x, p.y) * pixelArea; area += pixelArea; pixels++; heights.Add(p.z);
                    for (int direction = 0; direction < 4; direction++)
                    {
                        int nx = x + (direction == 0 ? -1 : direction == 1 ? 1 : 0);
                        int ny = y + (direction == 2 ? -1 : direction == 3 ? 1 : 0);
                        if (nx < 0 || ny < 0 || nx >= thermal.Width || ny >= thermal.Height) continue;
                        int next = ny * thermal.Width + nx;
                        if (!seen[next] && depth[next] > 0 && Mathf.Abs(depth[next] - z) <= c.blob_depth_jump_m)
                        { seen[next] = true; queue.Enqueue(next); }
                    }
                }
                // A body is off the floor and no taller than a standing person: heated floor near a fire and tall wall
                // patches were most of the false warm shapes (warm_eval: 361 false blobs, 0 real people in fire rooms).
                if (pixels >= c.person_min_pixels && area >= c.person_min_area_m2 && area <= c.person_max_area_m2 &&
                    Percentile(heights, .5f) >= c.person_floor_clearance_m && Percentile(heights, .95f) <= c.person_top_max_m)
                    result.Add(new PersonDetection(sum / area, area));
            }
            return result;
        }
        /// <summary>Linear-interpolated percentile of unweighted values (sorts the list in place).</summary>
        static float Percentile(List<float> values, float q)
        {
            if (values.Count == 0) return float.NaN;
            values.Sort(); float index = (values.Count - 1) * q; int lo = Mathf.FloorToInt(index), hi = Mathf.Min(values.Count - 1, lo + 1);
            return Mathf.Lerp(values[lo], values[hi], index - lo);
        }
        internal static void ProjectHeat(ThermalFrame thermal, RgbdFrame rgb, SensorConfig s, BrainConfig c, BrainPose pose, OccupancyGrid grid)
        {
            // Max per cell and per frame avoids estimating a temperature slope between pixels of one image.
            var maximum = new Dictionary<int, float>();
            for (int y = 0; y < thermal.Height; y++) for (int x = 0; x < thermal.Width; x++)
            {
                float temperature = thermal.Celsius(x, y); if (temperature <= c.heat_project_min_c) continue;
                if (!Match(x, y, rgb, s, out int u, out int v, out float z)) continue;
                var p = OccupancyGrid.BackProject(u, v, z, s); int cell = grid.Cell(pose.Transform(p.x, p.y));
                if (cell >= 0 && (!maximum.TryGetValue(cell, out float old) || temperature > old)) maximum[cell] = temperature;
            }
            // Sort keys so updates and any later extension to event logging have an explicit order.
            var keys = new List<int>(maximum.Keys); keys.Sort();
            foreach (int cell in keys) grid.ObserveHeat(cell, maximum[cell], thermal.Timestamp);
        }
    }

    internal sealed class CameraHealth
    {
        readonly BrainConfig c;
        double poorSince = double.NaN, goodSince = double.NaN, previous = double.NegativeInfinity;
        public bool Thermal { get; private set; }
        public float Score { get; private set; }
        public float Contrast { get; private set; }
        public float ValidDepth { get; private set; }
        public int Corners { get; private set; }
        public CameraHealth(BrainConfig settings) { c = settings; }
        public bool Observe(RgbdFrame f, SensorConfig calibration)
        {
            int stride = c.health_stride, w = (f.Width + stride - 1) / stride, h = (f.Height + stride - 1) / stride;
            var grey = new float[w * h]; double sum = 0, square = 0; int valid = 0;
            for (int y = 0; y < h; y++) for (int x = 0; x < w; x++)
            {
                int pixel = Math.Min(y * stride, f.Height - 1) * f.Width + Math.Min(x * stride, f.Width - 1), p = pixel * 3;
                float g = (.299f * f.Rgb[p] + .587f * f.Rgb[p + 1] + .114f * f.Rgb[p + 2]) / 255;
                grey[y * w + x] = g; sum += g; square += g * g;
                float z = f.Depth[pixel] * .001f; if (z >= calibration.depth.min_m && z <= calibration.depth.max_m) valid++;
            }
            Contrast = (float)Math.Sqrt(Math.Max(0, square / grey.Length - Math.Pow(sum / grey.Length, 2)));
            ValidDepth = valid / (float)grey.Length; Corners = 0;
            // A local structure tensor rejects a lone straight edge, unlike a two-gradient threshold.
            for (int y = 2; y < h - 2; y++) for (int x = 2; x < w - 2; x++)
            {
                float xx = 0, yy = 0, xy = 0;
                for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++)
                {
                    int p = (y + dy) * w + x + dx;
                    float gx = (grey[p + 1] - grey[p - 1]) / 2, gy = (grey[p + w] - grey[p - w]) / 2;
                    xx += gx * gx; yy += gy * gy; xy += gx * gy;
                }
                float minimumEigenvalue = (xx + yy - Mathf.Sqrt((xx - yy) * (xx - yy) + 4 * xy * xy)) / 18;
                if (minimumEigenvalue > c.corner_gradient * c.corner_gradient) Corners++;
            }
            Score = c.health_corner_weight * Mathf.Clamp01(Corners / (float)c.corner_target) +
                c.health_contrast_weight * Mathf.Clamp01(Contrast / c.contrast_target) + c.health_depth_weight * ValidDepth;
            double t = f.Timestamp;
            if (t - previous > c.health_max_gap_s) { poorSince = double.NaN; goodSince = double.NaN; }
            previous = t; bool poor = Score < c.health_threshold;
            if (poor)
            {
                goodSince = double.NaN; if (double.IsNaN(poorSince)) poorSince = t;
                if (!Thermal && t - poorSince + 1e-8 >= c.switch_to_thermal_s) { Thermal = true; return true; }
            }
            else
            {
                poorSince = double.NaN; if (double.IsNaN(goodSince)) goodSince = t;
                if (Thermal && t - goodSince + 1e-8 >= c.switch_to_rgb_s) { Thermal = false; return true; }
            }
            return false;
        }
    }
}
