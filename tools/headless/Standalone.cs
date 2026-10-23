// Stand-ins for engine-only pieces, so the real brain and fire sources compile with plain .NET. Nothing here models
// behaviour: frame types are copied field for field from the Sensors files, and the two UnityEngine stand-ins read
// files the same way the engine versions do.
using System;
using System.IO;
namespace UnityEngine
{
    public static class Application
    {
        public static string dataPath => Path.Combine(Environment.GetEnvironmentVariable("SMOKE_REPO") ?? ".", "unity/SmokeRescue/Assets");
    }
    public static class JsonUtility
    {
        // JsonUtility fills public and [SerializeField] fields and skips [NonSerialized] ones; Newtonsoft does the same for
        // these plain field classes.
        public static T FromJson<T>(string json) => Newtonsoft.Json.JsonConvert.DeserializeObject<T>(json,
            new Newtonsoft.Json.JsonSerializerSettings { MissingMemberHandling = Newtonsoft.Json.MissingMemberHandling.Ignore });
    }
}
namespace SmokeRescue.Sensors
{
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
    public sealed class RgbdFrame
    {
        public readonly double Timestamp; public readonly int Width, Height; public readonly byte[] Rgb; public readonly ushort[] Depth;
        public RgbdFrame(double timestamp, int width, int height, byte[] rgb, ushort[] depth)
        { Timestamp = timestamp; Width = width; Height = height; Rgb = rgb; Depth = depth; }
    }
    public sealed class ThermalFrame
    {
        public readonly double Timestamp; public readonly int Width, Height; public readonly ushort[] DeciKelvin;
        public ThermalFrame(double timestamp, int width, int height, ushort[] deciKelvin)
        { Timestamp = timestamp; Width = width; Height = height; DeciKelvin = deciKelvin; }
        public float Celsius(int x, int y) => DeciKelvin[y * Width + x] / 10f - 273.15f;
    }
}
namespace SmokeRescue.World
{
    public sealed class ThermalField
    {
        public interface ISource
        {
            float AmbientC { get; }
            UnityEngine.Vector4 Sample(UnityEngine.Vector2 xz, float t);
            System.Collections.Generic.IReadOnlyList<(UnityEngine.Vector3 centre, float radius, float height, float tempC)> Flames(float t);
        }
    }
    internal class SmokeSolver
    {
        public void ScheduleWater(double t, UnityEngine.Vector2 p, float h, float r, float a, double f, double d)
            => throw new NotSupportedException("No fluid smoke in the headless simulator.");
    }
}
