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
            float nx = (x - a.cx) / a.fx, shift = a.mount_right_offset_m;
            u = Mathf.RoundToInt(nx * b.fx + b.cx);
            v = Mathf.RoundToInt((y - a.cy) / a.fy * b.fy + b.cy);
            if (!DepthAt(rgb, s, u, v, out z)) return false;
            if (shift == 0) return true;
            // The thermal camera sits mount_right_offset_m to the colour camera's right, so what a thermal pixel sees at
            // range z lies shift / z further right in the colour image than the intrinsics alone say. Move there and read
            // the range again until the pixel stays put; one that never settles straddles a depth edge. Without this, a far
            // fire's pixels beside a near door jamb took the jamb's range: one-door seed 924, three fire cells appeared on
            // robot B's way out at 51 s (the fire was 11 m away) and B lost its way out at 52 s.
            for (int k = 0; k < 3; k++)
            {
                int next = Mathf.RoundToInt((nx + shift / z) * b.fx + b.cx);
                if (next == u) return true;
                u = next; if (!DepthAt(rgb, s, u, v, out z)) return false;
            }
            return false;
        }
        static bool DepthAt(RgbdFrame rgb, SensorConfig s, int u, int v, out float z)
        {
            z = 0;
            if (u < 0 || v < 0 || u >= rgb.Width || v >= rgb.Height) return false;
            z = rgb.Depth[v * rgb.Width + u] * .001f;
            return z >= s.depth.min_m && z <= s.depth.max_m;
        }
        /// <summary>
        /// Range for a thermal pixel with no depth return, from the LiDAR beam at the same bearing. Dense smoke removes
        /// the depth camera's returns (hidden-corner run 2: 1 % valid depth at times), and without range a warm blob
        /// cannot be sized, so thermal confirmed nobody exactly when the colour camera was blind too. The planar LiDAR
        /// (0.45 m up) hits a sitting or slumped person; over someone lying flat it reads the wall behind, which makes the
        /// blob too large and it is rejected (no false person, but no detection either).
        /// </summary>
        internal static bool LidarDepth(int u, int v, SensorConfig s, LidarScan lidar, out float z)
        {
            z = 0;
            var p1 = OccupancyGrid.BackProject(u, v, 1f, s);
            var d = new Vector3(p1.x, p1.y - s.camera.mount.forward_offset_m, p1.z - s.camera.mount.height_m);
            float dh = Mathf.Sqrt(d.x * d.x + d.y * d.y); if (dh < 1e-4f) return false;
            float bearing = Mathf.Atan2(-d.x, d.y);   // left positive, as the LiDAR beams and the robot heading
            int n = lidar.RangesM.Length;
            int beam = ((Mathf.RoundToInt((bearing - lidar.AngleMinRad) / lidar.AngleIncrementRad) % n) + n) % n;
            float r = lidar.RangesM[beam];
            if (!(r >= lidar.MinM) || r > lidar.MaxM) return false;
            // The camera sits ahead of the LiDAR by the difference of their forward offsets.
            float horizontal = r - (s.camera.mount.forward_offset_m - lidar.MountForwardM) * Mathf.Cos(bearing);
            if (!(horizontal > 0)) return false;
            z = horizontal / dh;
            return true;
        }

        public static List<PersonDetection> Detect(ThermalFrame thermal, RgbdFrame rgb, SensorConfig s, BrainConfig c, BrainPose pose, LidarScan lidar = null)
        {
            int count = thermal.Width * thermal.Height;
            var depth = new float[count]; var seen = new bool[count]; var queue = new Queue<int>();
            for (int y = 0; y < thermal.Height; y++) for (int x = 0; x < thermal.Width; x++)
            {
                float temperature = thermal.Celsius(x, y);
                if (temperature < c.person_min_c || temperature > c.person_max_c) continue;
                if (Match(x, y, rgb, s, out int u, out int v, out float z)) depth[y * thermal.Width + x] = z;
                else if (lidar != null && u >= 0 && v >= 0 && u < rgb.Width && v < rgb.Height && LidarDepth(u, v, s, lidar, out float lz))
                    depth[y * thermal.Width + x] = lz;
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
            // Every pixel with a depth match is a look at its cell, whatever its temperature, so cool floor counts as seen
            // (RobotBrain route watch). This is BackProject then pose.Transform, with the angles' sines taken once per frame:
            // it runs for every pixel (82k at 15 Hz on the house camera), not only the few hot ones.
            var k = s.camera; float pitch = k.mount.pitch_down_deg * Mathf.Deg2Rad, cosPitch = Mathf.Cos(pitch), sinPitch = Mathf.Sin(pitch);
            float cosYaw = Mathf.Cos(pose.yawRad), sinYaw = Mathf.Sin(pose.yawRad);
            for (int y = 0; y < thermal.Height; y++) for (int x = 0; x < thermal.Width; x++)
            {
                if (!Match(x, y, rgb, s, out int u, out int v, out float z)) continue;
                float right = (u - k.cx) / k.fx * z, ahead = k.mount.forward_offset_m + z * cosPitch - (v - k.cy) / k.fy * z * sinPitch;
                int look = grid.Cell(pose.position + new Vector2(cosYaw * right - sinYaw * ahead, sinYaw * right + cosYaw * ahead));
                grid.MarkHeatLook(look, thermal.Timestamp);
                float temperature = thermal.Celsius(x, y);
                int cell;
                if (temperature <= c.heat_project_min_c)
                {
                    // A cool reading counts where the map still holds heat, so the "seen cool for heat_cool_after_s: the fire
                    // is out" rule in ObserveHeat can run. Skipping every pixel at or below heat_project_min_c kept a cell
                    // seen at 500 C and then at 20 C 3 s later "fire" at 499 C (Codex HeatCheck, 23 Oct); seed 934, robot A:
                    // 21 routes cut by fire margins while its own air never passed 44 C.
                    if (look < 0 || grid.Heat(look) <= c.heat_project_min_c) continue;
                    cell = look;
                }
                else { var p = OccupancyGrid.BackProject(u, v, z, s); cell = grid.Cell(pose.Transform(p.x, p.y)); }
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
