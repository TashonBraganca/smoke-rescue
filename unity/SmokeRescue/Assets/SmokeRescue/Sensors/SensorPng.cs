using System;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Experimental.Rendering;

namespace SmokeRescue.Sensors
{
    /// <summary>
    /// Worker-thread encoding. Input arrays must remain unchanged until the returned task completes.
    /// ImageConversion.EncodeArrayToPNG follows Unity's texture convention (row 0 is the bottom of the image),
    /// so the top-first sensor arrays are flipped here; without it every PNG came out upside down.
    /// </summary>
    public static class SensorPng
    {
        public static Task<byte[]> EncodeRgbAsync(byte[] topFirstRgb, int width, int height)
        {
            if (topFirstRgb.Length != width * height * 3) throw new ArgumentException("Invalid RGB length.");
            return Task.Run(() => ImageConversion.EncodeArrayToPNG(FlipRows(topFirstRgb, width * 3, height), GraphicsFormat.R8G8B8_UNorm, (uint)width, (uint)height));
        }
        public static Task<byte[]> EncodeDepthAsync(ushort[] topFirstMillimetres, int width, int height)
        {
            if (topFirstMillimetres.Length != width * height) throw new ArgumentException("Invalid depth length.");
            return Task.Run(() => ImageConversion.EncodeArrayToPNG(FlipRows(topFirstMillimetres, width, height), GraphicsFormat.R16_UNorm, (uint)width, (uint)height));
        }

        static T[] FlipRows<T>(T[] src, int rowLength, int rows)
        {
            var dst = new T[src.Length];
            for (int y = 0; y < rows; y++) Array.Copy(src, y * rowLength, dst, (rows - 1 - y) * rowLength, rowLength);
            return dst;
        }
    }
}
