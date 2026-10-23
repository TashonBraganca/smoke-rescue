using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Text;
using UnityEngine;

namespace SmokeRescue.Brain
{
    [Serializable] public struct PoseUncertainty
    {
        // One-sigma budgets, not a calibrated SLAM covariance. Zero is allowed for exact test sensors.
        public float positionStdM, yawStdRad;
        internal bool Valid => BrainMath.Finite(positionStdM) && BrainMath.Finite(yawStdRad) && positionStdM >= 0 && yawStdRad >= 0;
    }
    [Serializable] public struct TeamEvidenceCell
    {
        public int cell;
        public float logOdds;
        public double observedAt, searchedAt;
        public uint revision;
    }

    /// <summary>Detached team belief in the receiving robot's map frame, south row first.</summary>
    public sealed class TeamMapExport
    {
        public int size;
        public float cellM;
        public Vector2 origin;
        public double at;
        public string[] sources;
        public float[] occupancyProbability, searchedConfidence, ageS, searchAgeS;
        public uint[] sourceMask, searchedSourceMask;

        public void Write(string path)
        {
            using (var file = File.Create(path))
            using (var zip = new GZipStream(file, System.IO.Compression.CompressionLevel.Fastest))
            using (var w = new BinaryWriter(zip, Encoding.UTF8))
            {
                w.Write(new byte[] { (byte)'T', (byte)'M', (byte)'A', (byte)'P' }); w.Write((byte)1);
                w.Write(size); w.Write(cellM); w.Write(origin.x); w.Write(origin.y); w.Write(at); w.Write((byte)sources.Length);
                foreach (var s in sources) { var b = Encoding.UTF8.GetBytes(s); w.Write((ushort)b.Length); w.Write(b); }
                foreach (float v in occupancyProbability) w.Write(v);
                foreach (float v in searchedConfidence) w.Write(v);
                foreach (uint v in sourceMask) w.Write(v);
                foreach (uint v in searchedSourceMask) w.Write(v);
                foreach (float v in ageS) w.Write(v);
                foreach (float v in searchAgeS) w.Write(v);
            }
        }
    }

    /// <summary>One absolute log-odds layer per originating robot. Radio sends own evidence only.
    /// Cell revisions replace a source's old belief; repeated packets never add its belief again.
    /// A rigid correction changes one source transform, so shared history moves without a grid pass.</summary>
    public sealed class TeamMap
    {
        sealed class Source
        {
            public string id; public BrainPose transform; public PoseUncertainty uncertainty;
            public long sequence;
            public readonly Dictionary<int, TeamEvidenceCell> cells = new Dictionary<int, TeamEvidenceCell>();
        }
        struct Observation { public bool wall, searched; }
        readonly List<Source> sources = new List<Source>();
        readonly List<int> ownCells = new List<int>();
        readonly Dictionary<int, Observation> observation = new Dictionary<int, Observation>();
        readonly BrainConfig config;
        readonly Source own;
        readonly int size;
        readonly Vector2 origin;
        uint revision;
        int sendCursor;
        double captureAt;
        bool observing;
        bool duplicateObservation;
        readonly double[] lastCapture = { double.NegativeInfinity, double.NegativeInfinity };
        int observationSensor;
        public string Robot => own.id;
        public bool HasOwnEvidence => ownCells.Count > 0;
        public PoseUncertainty Uncertainty { get => own.uncertainty; set { if (!value.Valid) throw new ArgumentException("Invalid uncertainty."); own.uncertainty = value; } }
        public TeamMap(string robot, int gridSize, Vector2 gridOrigin, BrainConfig settings)
        {
            own = new Source { id = robot }; sources.Add(own);
            size = gridSize; origin = gridOrigin; config = settings;
        }
        public int Cell(Vector2 p)
        {
            int x = Mathf.FloorToInt((p.x - origin.x) / BrainMessage.ResolutionM), y = Mathf.FloorToInt((p.y - origin.y) / BrainMessage.ResolutionM);
            return x >= 0 && y >= 0 && x < size && y < size ? y * size + x : -1;
        }
        Vector2 Centre(int cell) => origin + new Vector2(cell % size + .5f, cell / size + .5f) * BrainMessage.ResolutionM;
        static Vector2 Inverse(BrainPose pose, Vector2 p)
        {
            var d = p - pose.position; float c = Mathf.Cos(pose.yawRad), s = Mathf.Sin(pose.yawRad);
            return new Vector2(c * d.x + s * d.y, -s * d.x + c * d.y);
        }
        static BrainPose Compose(BrainPose a, BrainPose b) => new BrainPose
        { position = a.Transform(b.position.x, b.position.y), yawRad = a.yawRad + b.yawRad };

        public void BeginObservation(double t, int sensor = 0)
        {
            BrainMath.Time(t);
            if (observing) throw new InvalidOperationException("Finish the previous team-map observation.");
            if (sensor < 0 || sensor >= lastCapture.Length) throw new ArgumentOutOfRangeException(nameof(sensor));
            duplicateObservation = t <= lastCapture[sensor]; observationSensor = sensor;
            captureAt = t; observation.Clear(); observing = true;
        }
        public void Observe(Vector2 p, bool wall, bool searched)
        {
            if (!observing) throw new InvalidOperationException("Begin a team-map observation first.");
            if (duplicateObservation) return;
            int cell = Cell(Inverse(own.transform, p)); if (cell < 0) return;
            observation.TryGetValue(cell, out var o);
            o.wall |= wall; o.searched |= searched && !wall; observation[cell] = o;
        }
        public void EndObservation()
        {
            if (!observing) throw new InvalidOperationException("No team-map observation is open.");
            if (duplicateObservation) { observing = false; return; }
            lastCapture[observationSensor] = captureAt;
            uint next = ++revision;
            foreach (var pair in observation)
            {
                bool exists = own.cells.TryGetValue(pair.Key, out var old);
                if (exists && captureAt < old.observedAt) continue;
                // One contribution per 20 cm cell per capture. Occupied wins over free rays in that capture.
                float delta = pair.Value.wall ? config.occupied_log_odds : config.free_log_odds;
                var cell = new TeamEvidenceCell { cell = pair.Key, revision = next,
                    logOdds = Mathf.Clamp(old.logOdds + delta, -config.log_odds_cap, config.log_odds_cap),
                    observedAt = captureAt, searchedAt = exists ? old.searchedAt : -1 };
                if (pair.Value.searched && !pair.Value.wall) cell.searchedAt = captureAt;
                own.cells[pair.Key] = cell;
                if (!exists) ownCells.Add(pair.Key);
            }
            observing = false;
        }
        public void ApplyCorrection(BrainPose delta) => own.transform = Compose(delta, own.transform);

        /// <summary>A bounded rotating batch of absolute own beliefs. Lost batches return on the next circuit.
        /// The transform travels even when no cells changed. No peer-derived cells are forwarded.</summary>
        public void FillMessage(BrainMessage message)
        {
            message.PoseUncertainty = own.uncertainty; message.MapCorrection = own.transform;
            message.TeamCells = Array.Empty<TeamEvidenceCell>();
            int capacity = Math.Max(0, (BrainMessage.MaxBytes - 1 - message.ToBytes().Length) / 20);
            int n = Math.Min(capacity, Math.Min(BrainMessage.MaxTeamCells, ownCells.Count));
            message.TeamCells = new TeamEvidenceCell[n];
            for (int i = 0; i < n; i++)
            { message.TeamCells[i] = own.cells[ownCells[sendCursor++ % ownCells.Count]]; }
            if (ownCells.Count > 0) sendCursor %= ownCells.Count;
        }
        public void Receive(BrainMessage message, BrainPose link)
        {
            if (message.Sender == Robot) return;
            Source source = sources.Find(s => s.id == message.Sender);
            if (source == null)
            {
                if (sources.Count >= 32) return;
                source = new Source { id = message.Sender }; sources.Add(source);
            }
            if (message.Sequence <= source.sequence) return;
            source.sequence = message.Sequence; source.transform = Compose(link, message.MapCorrection); source.uncertainty = message.PoseUncertainty;
            foreach (var cell in message.TeamCells)
                if (!source.cells.TryGetValue(cell.cell, out var old) || cell.revision > old.revision) source.cells[cell.cell] = cell;
        }
        float Weight(Source source, Vector2 p)
        {
            if (source == own) return 1;
            float distance = Vector2.Distance(p, source.transform.position);
            float spread = source.uncertainty.positionStdM + distance * source.uncertainty.yawStdRad;
            return 1 / (1 + spread / BrainMessage.ResolutionM);
        }
        public bool TryOccupancy(Vector2 p, double t, out float logOdds)
        {
            logOdds = 0; bool found = false;
            own.cells.TryGetValue(Cell(Inverse(own.transform, p)), out var local);
            bool freshFree = local.revision > 0 && local.logOdds <= config.free_threshold && t - local.observedAt <= config.frame_max_age_s;
            foreach (var source in sources)
            {
                if (!source.cells.TryGetValue(Cell(Inverse(source.transform, p)), out var cell)) continue;
                float w = Weight(source, p); found = true;
                if (source != own && freshFree && cell.logOdds > 0 && w < 1) continue;
                logOdds += cell.logOdds * w;
            }
            return found;
        }

        /// <summary>Export only on a map snapshot or broadcast boundary, never in the 60 Hz step.</summary>
        public TeamMapExport Export(double t)
        {
            int count = size * size;
            var result = new TeamMapExport { size = size, cellM = BrainMessage.ResolutionM, origin = origin, at = t,
                sources = sources.ConvertAll(s => s.id).ToArray(), occupancyProbability = new float[count], searchedConfidence = new float[count],
                sourceMask = new uint[count], searchedSourceMask = new uint[count], ageS = new float[count], searchAgeS = new float[count] };
            // Rasterize each source once, taking its strongest belief if rotation puts two source cells in one cell.
            var layer = new Dictionary<int, TeamEvidenceCell>(); var local = new Dictionary<int, TeamEvidenceCell>();
            var sums = new float[count];
            for (int s = 0; s < sources.Count; s++)
            {
                var source = sources[s]; layer.Clear();
                foreach (var pair in source.cells)
                {
                    int i = Cell(source.transform.Transform(Centre(pair.Key).x, Centre(pair.Key).y)); if (i < 0) continue;
                    var cell = pair.Value;
                    if (layer.TryGetValue(i, out var old))
                    {
                        cell.logOdds = Mathf.Abs(old.logOdds) > Mathf.Abs(cell.logOdds) ? old.logOdds : cell.logOdds;
                        cell.observedAt = Math.Max(old.observedAt, cell.observedAt); cell.searchedAt = Math.Max(old.searchedAt, cell.searchedAt);
                    }
                    layer[i] = cell;
                }
                foreach (var pair in layer)
                {
                    int i = pair.Key; var cell = pair.Value; float weight = Weight(source, Centre(i));
                    if (s == 0) local[i] = cell;
                    bool freshFree = local.TryGetValue(i, out var ownCell) && ownCell.logOdds <= config.free_threshold && t - ownCell.observedAt <= config.frame_max_age_s;
                    if (!(s > 0 && freshFree && cell.logOdds > 0 && weight < 1)) sums[i] += cell.logOdds * weight;
                    float age = (float)Math.Max(0, t - cell.observedAt);
                    if (result.sourceMask[i] == 0 || age < result.ageS[i]) result.ageS[i] = age;
                    result.sourceMask[i] |= 1u << s;
                    if (cell.searchedAt < 0) continue;
                    float searchAge = (float)Math.Max(0, t - cell.searchedAt);
                    if (result.searchedSourceMask[i] == 0 || searchAge < result.searchAgeS[i]) result.searchAgeS[i] = searchAge;
                    result.searchedSourceMask[i] |= 1u << s;
                    // A search mark loses half its confidence every minute. It remains in the age and source fields.
                    result.searchedConfidence[i] = Mathf.Max(result.searchedConfidence[i], weight * Mathf.Pow(.5f, searchAge / 60));
                }
            }
            for (int i = 0; i < count; i++)
            {
                result.occupancyProbability[i] = 1 / (1 + Mathf.Exp(-sums[i]));
                if (result.sourceMask[i] == 0) result.ageS[i] = -1;
                if (result.searchedSourceMask[i] == 0) result.searchAgeS[i] = -1;
            }
            return result;
        }
    }
}
