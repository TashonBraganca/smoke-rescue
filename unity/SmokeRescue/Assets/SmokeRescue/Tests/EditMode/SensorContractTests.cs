using System;
using System.Globalization;
using System.Linq;
using System.Threading.Tasks;
using NUnit.Framework;
using SmokeRescue.Sensors;
using UnityEngine;

namespace SmokeRescue.Tests.EditMode
{
    public class SensorContractTests
    {
        SensorConfig Config() => SensorConfig.Load();
        [Test]
        public void PoseConversionPreservesRotatedPointsAndTumUsesInvariantDecimals()
        {
            var position = new Vector3(1, 2, 3);
            var rotation = Quaternion.Euler(23, 47, -12);
            var local = new Vector3(0.4f, -0.7f, 2.2f);
            var pose = FrameConvert.Pose(position, rotation);
            Assert.That(Vector3.Distance(pose.position + pose.rotation * FrameConvert.Point(local),
                FrameConvert.Point(position + rotation * local)), Is.LessThan(1e-5));
            var culture = CultureInfo.CurrentCulture;
            try
            {
                CultureInfo.CurrentCulture = CultureInfo.GetCultureInfo("fr-FR");
                Assert.That(FrameConvert.TumLine(0.25, position, Quaternion.identity),
                    Is.EqualTo("0.250000 1.000000 -2.000000 3.000000 0.000000 0.000000 0.000000 1.000000"));
            }
            finally { CultureInfo.CurrentCulture = culture; }
        }
        [Test]
        public void StationaryImuMeasuresSpecificForceAndYawMatchesConvertedQuaternion()
        {
            var sensor = new ImuSensor(Config(), 9) { NoiseEnabled = false };
            sensor.Reset(Vector3.zero, Quaternion.identity);
            var sample = sensor.Step(0.005, Vector3.zero, Quaternion.identity, 0.005f).Value;
            Assert.That(Vector3.Distance(sample.SpecificForce, new Vector3(0, -9.81f, 0)), Is.LessThan(1e-5));
            var rotated = sensor.Step(0.01, Vector3.zero, Quaternion.AngleAxis(0.005f * Mathf.Rad2Deg, Vector3.up), 0.005f).Value;
            Assert.That(rotated.AngularRate.y, Is.EqualTo(1).Within(0.01));
            Assert.That(rotated.AngularRate.x, Is.EqualTo(0).Within(1e-5));
            var q = FrameConvert.Rotation(Quaternion.AngleAxis(10, Vector3.right));
            Assert.That(q.x, Is.LessThan(0));
            sensor.Reset(Vector3.zero, Quaternion.identity);
            var pitch = sensor.Step(0.005, Vector3.zero, Quaternion.AngleAxis(0.005f * Mathf.Rad2Deg, Vector3.right), 0.005f).Value;
            Assert.That(pitch.AngularRate.x, Is.EqualTo(-1).Within(0.01));
        }
        [Test]
        public void ImuRotatesGravityIntoTiltedBodyAndMeasuresAcceleration()
        {
            var sensor = new ImuSensor(Config(), 9) { NoiseEnabled = false };
            var q = Quaternion.Euler(0, 0, 90);
            sensor.Reset(Vector3.zero, q);
            var sample = sensor.Step(0.005, new Vector3(0, 0, 0.000025f), q, 0.005f).Value;
            Assert.That(sample.SpecificForce.x, Is.EqualTo(9.81f).Within(0.001));
            Assert.That(sample.SpecificForce.y, Is.EqualTo(0).Within(0.001));
            Assert.That(sample.SpecificForce.z, Is.EqualTo(1).Within(0.001));
        }
        [Test]
        public void ImuHonours200HzAndRejectsStepsThatWouldLoseSamples()
        {
            var sensor = new ImuSensor(Config(), 9) { NoiseEnabled = false };
            sensor.Reset(Vector3.zero, Quaternion.identity);
            int count = 0;
            for (int i = 1; i <= 1000; i++) if (sensor.Step(i * .001, Vector3.zero, Quaternion.identity, .001f).HasValue) count++;
            Assert.That(count, Is.EqualTo(200));
            Assert.Throws<ArgumentOutOfRangeException>(() => sensor.Step(2, Vector3.zero, Quaternion.identity, 0.02f));
        }
        [Test]
        public void ImuNoiseIsSeededAndHasConfiguredWhiteNoiseAndBiasWalk()
        {
            var c = Config();
            var a = new ImuSensor(c, 27); var b = new ImuSensor(c, 27); var other = new ImuSensor(c, 28);
            a.Reset(Vector3.zero, Quaternion.identity); b.Reset(Vector3.zero, Quaternion.identity); other.Reset(Vector3.zero, Quaternion.identity);
            double sum = 0, square = 0; int different = 0;
            for (int i = 1; i <= 4000; i++)
            {
                var x = a.Step(i * .005, Vector3.zero, Quaternion.identity, .005f).Value;
                var y = b.Step(i * .005, Vector3.zero, Quaternion.identity, .005f).Value;
                var z = other.Step(i * .005, Vector3.zero, Quaternion.identity, .005f).Value;
                Assert.That(x.SpecificForce, Is.EqualTo(y.SpecificForce)); Assert.That(x.AngularRate, Is.EqualTo(y.AngularRate));
                if (x.SpecificForce != z.SpecificForce) different++;
                sum += x.SpecificForce.x; square += x.SpecificForce.x * x.SpecificForce.x;
            }
            double sigma = Math.Sqrt(square / 4000 - Math.Pow(sum / 4000, 2));
            Assert.That(sigma, Is.EqualTo(c.imu.accel_noise_density / Math.Sqrt(.005)).Within(.003));
            Assert.That(different, Is.GreaterThan(3900));
            c.imu.accel_noise_density = c.imu.gyro_noise_density = 0;
            var walk = new ImuSensor(c, 19); walk.Reset(Vector3.zero, Quaternion.identity);
            Assert.That(walk.Step(.005, Vector3.zero, Quaternion.identity, .005f).Value.AngularRate.sqrMagnitude, Is.GreaterThan(0));
        }
        [Test]
        public void OdometryUsesPlanarBodyAxesAndConfiguredScaleAt50Hz()
        {
            var c = Config(); c.odometry.slip.events_per_minute = 0;
            var sensor = new OdometrySensor(c, 12);
            sensor.Reset(Vector3.zero, Quaternion.identity);
            Assert.That(sensor.Step(.01, new Vector3(.01f, 0, .02f), Quaternion.identity, .01f), Is.Null);
            var sample = sensor.Step(.02, new Vector3(.02f, 0, .04f), Quaternion.identity, .01f).Value;
            Assert.That(sample.Vx, Is.EqualTo(2.04f).Within(1e-5));
            Assert.That(sample.Vy, Is.EqualTo(-1.02f).Within(1e-5));
            sensor.NoiseEnabled = false;
            sensor.Reset(Vector3.zero, Quaternion.identity);
            var turn = sensor.Step(.02, Vector3.zero, Quaternion.AngleAxis(.02f * Mathf.Rad2Deg, Vector3.up), .02f).Value;
            Assert.That(turn.Wz, Is.EqualTo(-1).Within(.001));
        }
        [Test]
        public void OdometrySlipHasConfiguredOvershootAndDurationAndIsSeeded()
        {
            var c = Config(); c.odometry.scale_error = 0;
            c.odometry.slip.events_per_minute = 600000; c.odometry.slip.duration_s = .1f;
            c.odometry.slip.overshoot_min = c.odometry.slip.overshoot_max = .2f;
            var a = new OdometrySensor(c, 123); var b = new OdometrySensor(c, 123);
            a.Reset(Vector3.zero, Quaternion.identity); b.Reset(Vector3.zero, Quaternion.identity);
            for (int i = 1; i <= 12; i++)
            {
                var x = a.Step(i * .02, new Vector3(0, 0, i * .02f), Quaternion.identity, .02f).Value;
                var y = b.Step(i * .02, new Vector3(0, 0, i * .02f), Quaternion.identity, .02f).Value;
                if (i <= 5) Assert.That(x.Vx, Is.EqualTo(1.2f).Within(1e-4)); Assert.That(x.Vx, Is.EqualTo(y.Vx));
                if (i == 1) c.odometry.slip.events_per_minute = 0;
                if (i >= 7) Assert.That(x.Vx, Is.EqualTo(1).Within(1e-4));
            }
        }
        static SensorConfig SmallConfig()
        {
            var c = SensorConfig.Load(); c.camera.width = 80; c.camera.height = 60;
            c.camera.fx = c.camera.fy = 52; c.camera.cx = 39.5f; c.camera.cy = 29.5f;
            return c;
        }
        [Test]
        public void NoiseDisabledKeepsRgbAndLinearMillimetresButRejectsRange()
        {
            var c = SmallConfig(); int n = c.camera.width * c.camera.height;
            var rgb = Enumerable.Repeat((byte)120, n * 3).ToArray(); var depth = Enumerable.Repeat(2.345f, n).ToArray();
            depth[0] = .1f; depth[1] = 8.1f; depth[2] = float.NaN; depth[3] = float.PositiveInfinity;
            var output = new RgbdNoise(c, 1).Process(rgb, depth, new Pose(), false, _ => throw new Exception("Disabled noise sampled smoke."));
            Assert.That(output.Take(4), Is.All.EqualTo(0)); Assert.That(output.Skip(4), Is.All.EqualTo(2345));
            Assert.That(rgb, Is.All.EqualTo(120));
        }
        [Test]
        public void DepthAndRgbGaussianNoiseHaveConfiguredSigmaAndRepeatWithSeed()
        {
            var c = SmallConfig(); c.depth.edge_dropout_px = 0; c.depth.grazing_angle_invalid_deg = 90;
            int n = c.camera.width * c.camera.height;
            var depth = Enumerable.Repeat(2f, n).ToArray(); var rgbA = Enumerable.Repeat((byte)128, n * 3).ToArray(); var rgbB = (byte[])rgbA.Clone();
            var a = new RgbdNoise(c, 7).Process(rgbA, depth, new Pose(), true);
            var b = new RgbdNoise(c, 7).Process(rgbB, depth, new Pose(), true);
            Assert.That(a, Is.EqualTo(b)); Assert.That(rgbA, Is.EqualTo(rgbB));
            double depthSigma = Math.Sqrt(a.Average(v => Math.Pow(v - 2000, 2)));
            double rgbSigma = Math.Sqrt(rgbA.Average(v => Math.Pow(v - 128, 2)));
            Assert.That(depthSigma, Is.EqualTo(6).Within(.4)); Assert.That(rgbSigma, Is.EqualTo(2).Within(.15));
        }
        [Test]
        public void SmokeDropoutClampsToConfiguredMaximumAndSamplesUnityWorld()
        {
            var c = SmallConfig(); c.depth.edge_dropout_px = 0;
            int n = c.camera.width * c.camera.height;
            var pose = new Pose(new Vector3(10, 20, 30), Quaternion.identity); int calls = 0;
            var output = new RgbdNoise(c, 6).Process(new byte[n * 3], Enumerable.Repeat(2f, n).ToArray(), pose, true,
                point => { calls++; Assert.That(point.z, Is.EqualTo(31).Within(1e-5)); return 100; });
            double dropout = output.Count(v => v == 0) / (double)n;
            Assert.That(calls, Is.EqualTo(n)); Assert.That(dropout, Is.EqualTo(.6).Within(.03));
        }
        [Test]
        public void DiscontinuityDropoutMarksBothSidesWithoutCascadingAcrossImage()
        {
            var c = SmallConfig(); int w = c.camera.width, h = c.camera.height; c.depth.grazing_angle_invalid_deg = 90;
            var depth = new float[w * h];
            for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) depth[y * w + x] = x < w / 2 ? 2 : 4;
            var output = new RgbdNoise(c, 5).Process(new byte[w * h * 3], depth, new Pose(), true);
            Assert.That(output[30 * w + w / 2 - 2], Is.EqualTo(0)); Assert.That(output[30 * w + w / 2 + 1], Is.EqualTo(0));
            Assert.That(output[30 * w + 10], Is.GreaterThan(0)); Assert.That(output[30 * w + 70], Is.GreaterThan(0));
        }
        [Test]
        public void GrazingPlaneDropsOutUsingNeighbourDepthNormals()
        {
            var c = SmallConfig(); c.depth.edge_dropout_px = 0; c.depth.grazing_angle_invalid_deg = 80;
            int w = c.camera.width, h = c.camera.height; var depth = new float[w * h];
            // n dot p = .04, n=(1,0,.02). The normal is 88.85 degrees from the centre viewing ray.
            for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) depth[y * w + x] = .04f / ((x - c.camera.cx) / c.camera.fx + .02f);
            var output = new RgbdNoise(c, 8).Process(new byte[w * h * 3], depth, new Pose(), true);
            Assert.That(output[30 * w + 40], Is.EqualTo(0));
        }
        [Test]
        public async Task WorkerPngPreservesDepthAsSixteenBitGrayscale()
        {
            var png = await SensorPng.EncodeDepthAsync(new ushort[] {0, 1, 1000, 65535}, 2, 2);
            Assert.That(png.Take(8), Is.EqualTo(new byte[] {137,80,78,71,13,10,26,10}));
            Assert.That(png[24], Is.EqualTo(16), "Depth PNG lost 16-bit precision.");
            Assert.That(png[25], Is.EqualTo(0), "Depth PNG must be single-channel grayscale.");
        }
    }
}
