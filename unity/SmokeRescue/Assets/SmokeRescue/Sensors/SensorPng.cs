using System;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Experimental.Rendering;

namespace SmokeRescue.Sensors
{
    /// <summary>Worker-thread encoding. Input arrays must remain unchanged until the returned task completes.</summary>
    public static class SensorPng
    {
        public static Task<byte[]> EncodeRgbAsync(byte[] topFirstRgb, int width, int height)
        {
            if (topFirstRgb.Length != width * height * 3) throw new ArgumentException("Invalid RGB length.");
            return Task.Run(() => ImageConversion.EncodeArrayToPNG(topFirstRgb, GraphicsFormat.R8G8B8_UNorm, (uint)width, (uint)height));
        }
        public static Task<byte[]> EncodeDepthAsync(ushort[] topFirstMillimetres, int width, int height)
        {
            if (topFirstMillimetres.Length != width * height) throw new ArgumentException("Invalid depth length.");
            return Task.Run(() => ImageConversion.EncodeArrayToPNG(topFirstMillimetres, GraphicsFormat.R16_UNorm, (uint)width, (uint)height));
        }
    }
}
