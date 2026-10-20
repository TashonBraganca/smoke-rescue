using System;
using SmokeRescue.Sensors;
using UnityEngine;

namespace SmokeRescue.Brain
{
    /// <summary>
    /// Live localization: each LiDAR sweep is matched against the robot's own LiDAR map before it is added, and the
    /// odometry pose is corrected by the result. This is the correction step of SLAM: odometry predicts, the scan
    /// against the map observes. Correlative matching (Olson 2009): every candidate pose in a small window around
    /// the odometry prediction is scored by how well the beam endpoints land on mapped walls (a Gaussian of the
    /// distance to the nearest LiDAR-occupied cell). The correction is the score-weighted mean of the candidates
    /// near the best one, not the single best: along a featureless corridor many candidates tie, and their mean
    /// stays at the prediction instead of jumping to one end of the window. No loop closure, no ground truth.
    /// </summary>
    internal sealed class ScanMatcher
    {
        readonly OccupancyGrid grid;
        readonly BrainConfig c;
        public ScanMatcher(OccupancyGrid map, BrainConfig config) { grid = map; c = config; }

        internal struct Result
        {
            public bool applied;
            public float fit, priorFit, dx, dy, dyawRad;
            public int beams;
            public BrainPose pose;
        }

        /// <summary>Window arguments override the per-sweep search (scan_window_*), for the wider search that links two
        /// robots' frames from a fingerprint match.</summary>
        public Result Match(LidarScan scan, BrainPose prior, float windowM = -1, float stepM = -1, float windowDeg = -1, float stepDeg = -1)
        {
            if (windowM < 0) windowM = c.scan_window_m; if (stepM <= 0) stepM = c.scan_step_m;
            if (windowDeg < 0) windowDeg = c.scan_window_deg; if (stepDeg <= 0) stepDeg = c.scan_step_deg;
            var result = new Result { pose = prior };
            // Beam endpoints in the body frame (x right, y forward), every scan_beam_stride-th valid return.
            int stride = Math.Max(1, c.scan_beam_stride), n = 0;
            var bx = new float[scan.RangesM.Length / stride + 1]; var by = new float[bx.Length];
            for (int i = 0; i < scan.RangesM.Length && n < bx.Length; i += stride)
            {
                float r = scan.RangesM[i];
                if (!(r >= scan.MinM) || r > Math.Min(scan.MaxM, c.scan_match_range_m)) continue;
                float a = scan.Angle(i);
                bx[n] = -Mathf.Sin(a) * r; by[n] = scan.MountForwardM + Mathf.Cos(a) * r; n++;
            }
            result.beams = n;
            if (n < c.scan_min_beams) return result;

            // Likelihood field in a window that holds every endpoint of every candidate.
            int centre = grid.Cell(prior.position); if (centre < 0) return result;
            float reach = c.scan_match_range_m + windowM + grid.CellM * 2;
            int half = Mathf.CeilToInt(reach / grid.CellM), side = 2 * half + 1;
            int x0 = centre % grid.Size - half, y0 = centre / grid.Size - half;
            var distance = grid.LidarDistanceWindow(x0, y0, side);
            var field = new float[distance.Length]; float k = 1f / (2 * c.scan_sigma_m * c.scan_sigma_m); bool any = false;
            for (int i = 0; i < field.Length; i++) { float d = distance[i]; field[i] = Mathf.Exp(-d * d * k); any |= d == 0; }
            if (!any) return result;
            Vector2 windowOrigin = grid.Origin + new Vector2(x0, y0) * grid.CellM;

            int steps = Mathf.RoundToInt(windowM / stepM), turns = Mathf.RoundToInt(windowDeg / stepDeg);
            int count = (2 * steps + 1) * (2 * steps + 1) * (2 * turns + 1);
            var scores = new float[count]; var poses = new Vector3[count]; float best = -1; int index = 0;
            for (int t = -turns; t <= turns; t++)
            {
                float yaw = prior.yawRad + t * stepDeg * Mathf.Deg2Rad, cos = Mathf.Cos(yaw), sin = Mathf.Sin(yaw);
                for (int iy = -steps; iy <= steps; iy++) for (int ix = -steps; ix <= steps; ix++)
                {
                    float px = prior.position.x + ix * stepM, py = prior.position.y + iy * stepM, sum = 0;
                    for (int b = 0; b < n; b++)
                        sum += Sample(field, side, windowOrigin, px + cos * bx[b] - sin * by[b], py + sin * bx[b] + cos * by[b]);
                    float score = sum / n; scores[index] = score; poses[index] = new Vector3(ix * stepM, iy * stepM, t * stepDeg * Mathf.Deg2Rad);
                    if (score > best) best = score;
                    if (t == 0 && ix == 0 && iy == 0) result.priorFit = score;
                    index++;
                }
            }
            result.fit = best;
            if (best < c.scan_min_fit) return result;
            Vector3 mean = Vector3.zero; float weights = 0;
            for (int i = 0; i < count; i++)
            {
                float w = Mathf.Exp((scores[i] - best) / c.scan_softness);
                mean += poses[i] * w; weights += w;
            }
            mean /= weights;
            result.applied = true; result.dx = mean.x; result.dy = mean.y; result.dyawRad = mean.z;
            result.pose = new BrainPose { position = prior.position + new Vector2(mean.x, mean.y), yawRad = prior.yawRad + mean.z };
            return result;
        }

        float Sample(float[] field, int side, Vector2 origin, float x, float y)
        {
            // Bilinear between cell centres, so sub-cell candidate steps change the score smoothly.
            float gx = (x - origin.x) / grid.CellM - .5f, gy = (y - origin.y) / grid.CellM - .5f;
            int ix = Mathf.FloorToInt(gx), iy = Mathf.FloorToInt(gy);
            if (ix < 0 || iy < 0 || ix + 1 >= side || iy + 1 >= side) return 0;
            float fx = gx - ix, fy = gy - iy; int i = iy * side + ix;
            return (field[i] * (1 - fx) + field[i + 1] * fx) * (1 - fy) + (field[i + side] * (1 - fx) + field[i + side + 1] * fx) * fy;
        }

        /// <summary>The building-frame rigid delta that moves the prior pose onto the matched one when left-composed
        /// (DeadReckoning.ApplyCorrection): rotate about the prior position, then translate.</summary>
        public static BrainPose Delta(BrainPose prior, BrainPose matched)
        {
            float dyaw = matched.yawRad - prior.yawRad, cos = Mathf.Cos(dyaw), sin = Mathf.Sin(dyaw);
            var rotated = new Vector2(cos * prior.position.x - sin * prior.position.y, sin * prior.position.x + cos * prior.position.y);
            return new BrainPose { position = matched.position - rotated, yawRad = dyaw };
        }
    }
}
