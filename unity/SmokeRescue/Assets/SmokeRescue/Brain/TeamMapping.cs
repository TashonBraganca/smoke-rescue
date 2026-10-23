using System;
using UnityEngine;

namespace SmokeRescue.Brain
{
    public sealed partial class RobotBrain
    {
        TeamMap teamMap;
        double uncertaintyAt;
        float uncertaintyTravel;
        void InitializeTeamMap()
        {
            teamMap = new TeamMap(id, SharedSize, grid.Origin, c);
            grid.TeamBelief = teamMap;
            UpdateTeamUncertainty(0);
        }
        void UpdateTeamUncertainty(double t)
        {
            uncertaintyTravel += motion.Speed * (float)Math.Max(0, t - uncertaintyAt); uncertaintyAt = t;
            float gyroNoise = sensors.imu?.gyro_noise_density ?? 0, walk = sensors.imu?.gyro_bias_random_walk ?? 0;
            float scale = Math.Abs(sensors.odometry?.scale_error ?? 0);
            // Entry budgets describe a nominal pad, not an exact physical placement. Integrate sensor noise without
            // reducing it after matching against the robot's own uncertain map. This budget has not been calibrated.
            float yawVariance = c.team_map_entry_yaw_std_deg * Mathf.Deg2Rad;
            yawVariance = yawVariance * yawVariance + gyroNoise * gyroNoise * (float)t + walk * walk * (float)(t * t * t / 3);
            teamMap.Uncertainty = new PoseUncertainty { positionStdM = c.team_map_entry_position_std_m + scale * uncertaintyTravel +
                uncertaintyTravel * Mathf.Sqrt(yawVariance), yawStdRad = Mathf.Sqrt(yawVariance) };
        }
        public TeamMapExport ExportTeamMap(double t) => teamMap.Export(t);
        public PoseUncertainty EstimatedPoseUncertainty => teamMap.Uncertainty;
    }
}
