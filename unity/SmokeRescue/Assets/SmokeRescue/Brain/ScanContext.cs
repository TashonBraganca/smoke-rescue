using System;
using System.Collections.Generic;
using UnityEngine;

namespace SmokeRescue.Brain
{
    /// <summary>
    /// Place fingerprint of one LiDAR sweep: its range profile, the median range in each of Sectors bearings around the
    /// sensor (a 2D polar signature in the spirit of Scan Context, Kim and Kim, IROS 2018, which uses heights in 3D).
    /// Two sweeps from about the same spot match at one circular sector shift, which also gives the turn between them,
    /// so the fingerprint needs no shared frame. Ranges change smoothly when the robot moves a little, unlike a binary
    /// polar occupancy picture, which jumped whole rings for half a metre of offset (measured: distances 0.68 to 0.84
    /// between nearby keyframes). The sorted profile does not change with heading and shortlists candidates cheaply.
    /// </summary>
    internal sealed class ScanContext
    {
        public const int Sectors = 60;
        public static float SectorRad => 2 * Mathf.PI / Sectors;
        public readonly float[] profile = new float[Sectors];   // median range per sector, metres; 0 = no return
        /// <summary>Share of bearings with a return. Out in the open most beams hit nothing, and two nearly empty
        /// sweeps look alike, so such keyframes are never used to link maps.</summary>
        public float ReturnShare { get; private set; }
        readonly float[] sorted;

        public ScanContext(float angleMinRad, float incrementRad, IReadOnlyList<float> rangesM)
        {
            var bins = new List<float>[Sectors];
            for (int s = 0; s < Sectors; s++) bins[s] = new List<float>();
            for (int i = 0; i < rangesM.Count; i++)
            {
                float r = rangesM[i]; if (!(r > 0)) continue;
                float a = Mathf.Repeat(angleMinRad + i * incrementRad, 2 * Mathf.PI);
                bins[Mathf.Min(Sectors - 1, Mathf.FloorToInt(a / SectorRad))].Add(r);
            }
            for (int s = 0; s < Sectors; s++)
            {
                if (bins[s].Count == 0) continue;
                bins[s].Sort(); profile[s] = bins[s][bins[s].Count / 2];
            }
            sorted = (float[])profile.Clone(); Array.Sort(sorted);
            int returned = 0; foreach (float r in profile) if (r > 0) returned++;
            ReturnShare = returned / (float)Sectors;
        }

        /// <summary>Heading-free shortlist key: mean absolute difference of the sorted profiles, metres.</summary>
        public float RingKeyDistance(ScanContext other)
        {
            float d = 0; for (int i = 0; i < Sectors; i++) d += Mathf.Abs(sorted[i] - other.sorted[i]);
            return d / Sectors;
        }

        /// <summary>Mean relative range difference over sectors where both sweeps have a return, at the best circular
        /// shift. shift is the number of sectors by which other's view is turned: sector j here matches sector j + shift
        /// there, so a bearing a here appears at a + shift * SectorRad there.</summary>
        public float Distance(ScanContext other, out int shift)
        {
            var all = Distances(other); float best = float.PositiveInfinity; shift = 0;
            for (int s = 0; s < Sectors; s++) if (all[s] < best) { best = all[s]; shift = s; }
            return best;
        }

        /// <summary>The few best shifts, at least minGap sectors apart. A rectangular room looks alike turned 180
        /// degrees (a square, 90), so the fingerprint alone cannot pick the turn; the map check decides among these.</summary>
        public List<(float distance, int shift)> BestShifts(ScanContext other, int count, int minGap)
        {
            var all = Distances(other); var order = new List<int>();
            for (int s = 0; s < Sectors; s++) order.Add(s);
            order.Sort((a, b) => { int k = all[a].CompareTo(all[b]); return k != 0 ? k : a.CompareTo(b); });
            var result = new List<(float, int)>();
            foreach (int s in order)
            {
                if (result.Count >= count) break;
                bool near = false;
                foreach (var r in result) { int gap = Math.Abs(r.Item2 - s); if (Math.Min(gap, Sectors - gap) < minGap) near = true; }
                if (!near) result.Add((all[s], s));
            }
            return result;
        }

        public float DistanceAt(ScanContext other, int shift) => Distances(other)[((shift % Sectors) + Sectors) % Sectors];

        float[] Distances(ScanContext other)
        {
            var result = new float[Sectors];
            for (int s = 0; s < Sectors; s++)
            {
                float sum = 0; int used = 0, missing = 0;
                for (int j = 0; j < Sectors; j++)
                {
                    float a = profile[j], b = other.profile[(j + s) % Sectors];
                    if (a <= 0 || b <= 0) { if (a > 0 || b > 0) missing++; continue; }
                    sum += Mathf.Abs(a - b) / Mathf.Max(.5f, (a + b) * .5f); used++;
                }
                // A return on one side only counts as a full mismatch.
                result[s] = used + missing == 0 ? 1 : (sum + missing) / (used + missing);
            }
            return result;
        }
    }
}
