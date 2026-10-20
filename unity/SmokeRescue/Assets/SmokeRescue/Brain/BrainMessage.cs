using System;
using System.IO;
using System.Text;
using UnityEngine;

namespace SmokeRescue.Brain
{
    /// <summary>Operator-supplied door pose. Unity heading is right-positive in degrees.
    /// Brain position is (Unity x, Unity z); internal left-positive yaw is -heading * Deg2Rad.</summary>
    public readonly struct EntryPose
    {
        public readonly BrainPose pose;
        public EntryPose(float x, float z, float headingDeg)
        {
            if (!BrainMath.Finite(x) || !BrainMath.Finite(z) || !BrainMath.Finite(headingDeg))
                throw new ArgumentException("Entry pose must be finite operator knowledge.");
            pose = new BrainPose { position = new Vector2(x, z), yawRad = -headingDeg * Mathf.Deg2Rad };
        }
    }

    [Serializable] public sealed class BrainClaim
    { public Vector2 centre; public float radiusM; public double expiresAt; }
    [Serializable] public struct BrainSurvivor
    { public Vector2 position; public double confirmedAt; public string sourceRobot; }
    [Serializable] public struct BrainHotspot
    { public int cell; public float temperatureC; public double observedAt; }
    [Serializable] public struct BrainAck
    { public string sender; public long sequence; }
    public sealed class PeerSnapshot
    { public string robot; public long sequence; public double sentAt; public BrainPose pose; public Vector2? goal; public BrainClaim claim; }
    [Serializable] public struct SurvivorSnapshot
    { public Vector2 position; public double confirmedAt; public string source, sourceRobot; }

    /// <summary>Version 1 binary radio state, using building coordinates. Masks are row-major,
    /// least-significant bit first. Explored is cumulative since BaseSequence; Occupied is always full.
    /// ToBytes is the transport format. JSON/base64 and transport headers are outside EstimatedBytes.
    /// Sequence numbers last for one run; create new brains when restarting a sender.</summary>
    public sealed class BrainMessage
    {
        public const float ResolutionM = .2f;
        public const int MaxBytes = 40000;
        public string Sender;
        public long Sequence, BaseSequence;
        public double SentAt;
        public bool FullSnapshot;
        public int GridSize;
        public Vector2 Origin;
        public BrainPose Pose;
        public Vector2? Goal;
        public BrainClaim Claim;
        public byte[] Explored = Array.Empty<byte>(), Occupied = Array.Empty<byte>();
        public BrainSurvivor[] Survivors = Array.Empty<BrainSurvivor>();
        public BrainHotspot[] Hotspots = Array.Empty<BrainHotspot>();
        public BrainAck[] Acknowledgements = Array.Empty<BrainAck>();
        public int EstimatedBytes => ToBytes().Length;
        internal static bool Bit(byte[] mask, int cell) => (mask[cell >> 3] & (1 << (cell & 7))) != 0;
        internal static void Set(byte[] mask, int cell) { mask[cell >> 3] |= (byte)(1 << (cell & 7)); }
        internal int Cell(Vector2 p)
        {
            int x = Mathf.FloorToInt((p.x - Origin.x) / ResolutionM), y = Mathf.FloorToInt((p.y - Origin.y) / ResolutionM);
            return x >= 0 && y >= 0 && x < GridSize && y < GridSize ? y * GridSize + x : -1;
        }
        internal Vector2 Centre(int cell) => Origin + new Vector2(cell % GridSize + .5f, cell / GridSize + .5f) * ResolutionM;
        public byte[] ToBytes()
        {
            using (var stream = new MemoryStream())
            using (var w = new BinaryWriter(stream, Encoding.UTF8, true))
            {
                w.Write((byte)1); WriteString(w, Sender); w.Write(Sequence); w.Write(BaseSequence); w.Write(SentAt); w.Write(FullSnapshot);
                w.Write(GridSize); Write(w, Origin); Write(w, Pose.position); w.Write(Pose.yawRad);
                w.Write(Goal.HasValue); if (Goal.HasValue) Write(w, Goal.Value);
                w.Write(Claim != null);
                if (Claim != null) { Write(w, Claim.centre); w.Write(Claim.radiusM); w.Write(Claim.expiresAt); }
                WriteMask(w, Explored, GridSize * GridSize); WriteMask(w, Occupied, GridSize * GridSize);
                w.Write(Survivors.Length);
                foreach (var s in Survivors) { Write(w, s.position); w.Write(s.confirmedAt); WriteString(w, s.sourceRobot); }
                WriteHeat(w);
                w.Write(Acknowledgements.Length);
                foreach (var a in Acknowledgements) { WriteString(w, a.sender); w.Write(a.sequence); }
                w.Flush();
                if (stream.Length >= MaxBytes) throw new InvalidDataException("Brain state exceeds the 40000-byte radio limit; increase compression before using this scene.");
                return stream.ToArray();
            }
        }
        public static BrainMessage FromBytes(byte[] bytes)
        {
            if (bytes == null || bytes.Length >= MaxBytes) throw new InvalidDataException("Invalid brain packet size.");
            using (var stream = new MemoryStream(bytes, false))
            using (var r = new BinaryReader(stream, Encoding.UTF8))
            {
                if (r.ReadByte() != 1) throw new InvalidDataException("Unsupported brain message version.");
                var m = new BrainMessage { Sender = ReadString(r), Sequence = r.ReadInt64(), BaseSequence = r.ReadInt64(), SentAt = r.ReadDouble(),
                    FullSnapshot = r.ReadBoolean(), GridSize = r.ReadInt32(), Origin = ReadPoint(r) };
                m.Pose = new BrainPose { position = ReadPoint(r), yawRad = r.ReadSingle() };
                if (r.ReadBoolean()) m.Goal = ReadPoint(r);
                if (r.ReadBoolean()) m.Claim = new BrainClaim { centre = ReadPoint(r), radiusM = r.ReadSingle(), expiresAt = r.ReadDouble() };
                if (m.GridSize < 1 || m.GridSize > 1000) throw new InvalidDataException("Invalid shared grid size.");
                m.Explored = ReadMask(r, m.GridSize * m.GridSize);
                m.Occupied = ReadMask(r, m.GridSize * m.GridSize);
                m.Survivors = new BrainSurvivor[ReadCount(r, 1024)];
                for (int i = 0; i < m.Survivors.Length; i++) m.Survivors[i] = new BrainSurvivor { position = ReadPoint(r), confirmedAt = r.ReadDouble(), sourceRobot = ReadString(r) };
                m.ReadHeat(r);
                m.Acknowledgements = new BrainAck[ReadCount(r, 64)];
                for (int i = 0; i < m.Acknowledgements.Length; i++) m.Acknowledgements[i] = new BrainAck { sender = ReadString(r), sequence = r.ReadInt64() };
                if (stream.Position != stream.Length) throw new InvalidDataException("Trailing brain packet bytes.");
                return m;
            }
        }
        // Sparse heat keeps float temperatures and double sample times. Dense heat uses a bitmap
        // followed by uint16 temperature and age in tenths. Temperature error is at most 0.05 C;
        // sample time rounds down by less than 0.1 s. No hot cells are dropped to meet a size limit.
        void WriteHeat(BinaryWriter w)
        {
            if (GridSize < 1 || GridSize > 1000) throw new InvalidDataException("Invalid shared grid size.");
            int cells = GridSize * GridSize, maskBytes = (cells + 7) / 8;
            bool dense = maskBytes + 4L * Hotspots.Length < 16L * Hotspots.Length;
            foreach (var h in Hotspots)
                if (!BrainMath.Finite(h.temperatureC) || h.temperatureC < 0 || h.temperatureC > 6553.5f ||
                    double.IsNaN(h.observedAt) || double.IsInfinity(h.observedAt) || h.observedAt > SentAt ||
                    SentAt - h.observedAt > 6553.5) dense = false;
            w.Write(dense); w.Write(Hotspots.Length);
            if (!dense)
            {
                foreach (var h in Hotspots) { w.Write(h.cell); w.Write(h.temperatureC); w.Write(h.observedAt); }
                return;
            }
            var mask = new byte[maskBytes]; var ordered = (BrainHotspot[])Hotspots.Clone();
            Array.Sort(ordered, (a, b) => a.cell.CompareTo(b.cell));
            foreach (var h in ordered)
            {
                if (h.cell < 0 || h.cell >= cells || Bit(mask, h.cell)) throw new InvalidDataException("Duplicate or invalid hot cell.");
                Set(mask, h.cell);
            }
            WriteMask(w, mask, cells);
            foreach (var h in ordered)
            {
                w.Write((ushort)Math.Round(h.temperatureC * 10.0, MidpointRounding.AwayFromZero));
                // The small tolerance makes a decoded tenth reencode to the same age.
                w.Write((ushort)Math.Max(0, Math.Ceiling((SentAt - h.observedAt) * 10 - 1e-7)));
            }
        }
        void ReadHeat(BinaryReader r)
        {
            bool dense = r.ReadBoolean(); int n = ReadCount(r, dense ? MaxBytes / 4 : MaxBytes / 16);
            Hotspots = new BrainHotspot[n];
            if (!dense)
            {
                for (int i = 0; i < n; i++) Hotspots[i] = new BrainHotspot { cell = r.ReadInt32(), temperatureC = r.ReadSingle(), observedAt = r.ReadDouble() };
                return;
            }
            if (GridSize < 1 || GridSize > 1000) throw new InvalidDataException("Invalid shared grid size.");
            int cells = GridSize * GridSize;
            var mask = ReadMask(r, cells);
            int index = 0;
            for (int cell = 0; cell < cells; cell++)
            {
                if (!Bit(mask, cell)) continue;
                if (index >= n) throw new InvalidDataException("Heat bitmap count exceeds header.");
                Hotspots[index++] = new BrainHotspot { cell = cell, temperatureC = r.ReadUInt16() / 10f,
                    observedAt = Math.Max(0, SentAt - r.ReadUInt16() / 10.0) };
            }
            if (index != n) throw new InvalidDataException("Heat bitmap count differs from header.");
        }
        static void Write(BinaryWriter w, Vector2 p) { w.Write(p.x); w.Write(p.y); }
        static Vector2 ReadPoint(BinaryReader r) => new Vector2(r.ReadSingle(), r.ReadSingle());
        static void WriteString(BinaryWriter w, string s)
        {
            var bytes = Encoding.UTF8.GetBytes(s ?? "");
            if (bytes.Length > 256) throw new InvalidDataException("Robot IDs must fit 256 UTF-8 bytes.");
            w.Write((ushort)bytes.Length); w.Write(bytes);
        }
        static string ReadString(BinaryReader r)
        {
            int n = r.ReadUInt16(); if (n > 256) throw new InvalidDataException("Robot ID is too long.");
            var bytes = r.ReadBytes(n); if (bytes.Length != n) throw new EndOfStreamException();
            return Encoding.UTF8.GetString(bytes);
        }
        static int ReadCount(BinaryReader r, int maximum)
        { int n = r.ReadInt32(); if (n < 0 || n > maximum) throw new InvalidDataException("Invalid brain array length."); return n; }
        // Each map layer uses the smaller of a raw bit mask and sorted one-runs.
        // A run has a 16-bit start and length; larger grids fall back to the raw mask.
        static void WriteMask(BinaryWriter w, byte[] mask, int cells)
        {
            if (mask == null || mask.Length != (cells + 7) / 8) throw new InvalidDataException("Invalid map mask length.");
            var runs = new System.Collections.Generic.List<(ushort start, ushort length)>();
            if (cells <= ushort.MaxValue)
            {
                int cell = 0;
                while (cell < cells)
                {
                    if (!Bit(mask, cell)) { cell++; continue; }
                    int start = cell++;
                    while (cell < cells && Bit(mask, cell)) cell++;
                    runs.Add(((ushort)start, (ushort)(cell - start)));
                }
            }
            if (cells <= ushort.MaxValue && runs.Count * 4 < mask.Length)
            {
                w.Write((byte)1); w.Write(runs.Count);
                foreach (var run in runs) { w.Write(run.start); w.Write(run.length); }
            }
            else { w.Write((byte)0); w.Write(mask.Length); w.Write(mask); }
        }
        static byte[] ReadMask(BinaryReader r, int cells)
        {
            int byteCount = (cells + 7) / 8;
            byte kind = r.ReadByte(); int count = ReadCount(r, kind == 0 ? byteCount : cells);
            var mask = new byte[byteCount];
            if (kind == 0)
            {
                if (count != byteCount) throw new InvalidDataException("Invalid raw map mask length.");
                mask = r.ReadBytes(byteCount); if (mask.Length != byteCount) throw new EndOfStreamException();
            }
            else if (kind == 1)
            {
                int next = 0;
                for (int i = 0; i < count; i++)
                {
                    int start = r.ReadUInt16(), length = r.ReadUInt16();
                    if (length == 0 || start < next || start + length > cells) throw new InvalidDataException("Invalid map mask run.");
                    for (int cell = start; cell < start + length; cell++) Set(mask, cell);
                    next = start + length;
                }
            }
            else throw new InvalidDataException("Unknown map mask encoding.");
            return mask;
        }
    }
}
