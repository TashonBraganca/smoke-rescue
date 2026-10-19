using System.Globalization;
using UnityEngine;

namespace SmokeRescue.Sensors
{
    public static class FrameConvert
    {
        /// <summary>Polar vector, S = diag(1,-1,1). Body and optical axes are right, down, forward.</summary>
        public static Vector3 Point(Vector3 unity) => new Vector3(unity.x, -unity.y, unity.z);
        public static Quaternion Rotation(Quaternion unity) => new Quaternion(-unity.x, unity.y, -unity.z, unity.w);
        public static Pose Pose(Vector3 position, Quaternion rotation) => new Pose(Point(position), Rotation(rotation));
        /// <summary>Angular velocity is axial, so it uses det(S) S = -S, consistent with Rotation.</summary>
        public static Vector3 Angular(Vector3 unity) => new Vector3(-unity.x, unity.y, -unity.z);
        public static string TumLine(double timestamp, Vector3 unityPosition, Quaternion unityRotation)
        {
            var p = Point(unityPosition);
            var q = Rotation(unityRotation.normalized);
            return string.Format(CultureInfo.InvariantCulture, "{0:F6} {1:F6} {2:F6} {3:F6} {4:F6} {5:F6} {6:F6} {7:F6}",
                timestamp, p.x, p.y, p.z, q.x, q.y, q.z, q.w);
        }
        public static Vector2 Project(Vector3 optical, SensorConfig.CameraConfig k) =>
            new Vector2(k.fx * optical.x / optical.z + k.cx, k.fy * optical.y / optical.z + k.cy);
        public static Vector3 BackProject(float u, float v, float z, SensorConfig.CameraConfig k) =>
            new Vector3((u - k.cx) * z / k.fx, (v - k.cy) * z / k.fy, z);
    }
}
