using System;
using System.IO;
using Newtonsoft.Json;
using UnityEngine;

namespace SmokeRescue.Sensors
{
    /// <summary>Sensor calibration and noise values from config/sensors.json. No Oracle dependency.</summary>
    [Serializable]
    public sealed class SensorConfig
    {
        public CameraConfig camera;
        public RgbNoise rgb_noise;
        public DepthConfig depth;
        public ImuConfig imu;
        public OdometryConfig odometry;
        [Serializable] public sealed class CameraConfig
        {
            public int width, height;
            public float vertical_fov_deg, fx, fy, cx, cy, rate_hz;
            public Mount mount;
        }
        [Serializable] public sealed class Mount { public float height_m, forward_offset_m, pitch_down_deg; }
        [Serializable] public sealed class RgbNoise { public float gaussian_sigma_8bit, exposure_flicker_near_fire; }
        [Serializable] public sealed class DepthConfig
        {
            public float min_m, max_m, sigma_coeff_mm_per_m2, grazing_angle_invalid_deg;
            public int edge_dropout_px;
            public SmokeDropout smoke_dropout;
        }
        [Serializable] public sealed class SmokeDropout { public float per_unit_density, max; }
        [Serializable] public sealed class ImuConfig
        {
            public float rate_hz, accel_noise_density, accel_bias_random_walk, gyro_noise_density, gyro_bias_random_walk;
        }
        [Serializable] public sealed class OdometryConfig { public float rate_hz, scale_error; public Slip slip; }
        [Serializable] public sealed class Slip { public float events_per_minute, duration_s, overshoot_min, overshoot_max; }

        public static string ConfigDir => Path.GetFullPath(Path.Combine(Application.dataPath, "..", "..", "..", "config"));
        public static SensorConfig Load(string path = null)
        {
            var config = JsonConvert.DeserializeObject<SensorConfig>(File.ReadAllText(path ?? Path.Combine(ConfigDir, "sensors.json")),
                new JsonSerializerSettings { MissingMemberHandling = MissingMemberHandling.Ignore });
            if (config == null) throw new InvalidDataException("Empty sensor configuration.");
            config.Validate();
            return config;
        }
        public void Validate()
        {
            if (camera == null || rgb_noise == null || depth?.smoke_dropout == null || imu == null || odometry?.slip == null ||
                camera.width < 3 || camera.height < 3 || !(camera.fx > 0) || !(camera.fy > 0) || !(camera.rate_hz > 0) ||
                !(camera.vertical_fov_deg > 0 && camera.vertical_fov_deg < 180) || !(depth.min_m > 0) ||
                !(depth.max_m > depth.min_m && depth.max_m <= 65.535f) || !(imu.rate_hz > 0) || !(odometry.rate_hz > 0) ||
                rgb_noise.gaussian_sigma_8bit < 0 || depth.sigma_coeff_mm_per_m2 < 0 || depth.edge_dropout_px < 0 ||
                depth.grazing_angle_invalid_deg < 0 || depth.grazing_angle_invalid_deg > 90 ||
                depth.smoke_dropout.max < 0 || depth.smoke_dropout.max > 1 || depth.smoke_dropout.per_unit_density < 0 ||
                imu.accel_noise_density < 0 || imu.gyro_noise_density < 0 || imu.accel_bias_random_walk < 0 || imu.gyro_bias_random_walk < 0 ||
                odometry.slip.events_per_minute < 0 || odometry.slip.duration_s < 0 || odometry.slip.overshoot_min < 0 ||
                odometry.slip.overshoot_max < odometry.slip.overshoot_min)
                throw new InvalidDataException("Invalid or missing sensor calibration/noise parameters.");
        }
    }
}
