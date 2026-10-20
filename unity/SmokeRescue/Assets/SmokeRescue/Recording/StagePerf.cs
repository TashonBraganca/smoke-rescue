using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Text;

namespace SmokeRescue.Recording
{
    /// <summary>
    /// Wall time the recorder spends in each stage, written to perf.txt every 10 sim seconds: one line per period with
    /// sim time, wall seconds for the period, and each stage's share. Measures the real run end to end, so a slow stage
    /// is found on real data instead of a synthetic probe.
    /// </summary>
    internal sealed class StagePerf
    {
        readonly Dictionary<string, Stopwatch> stages = new Dictionary<string, Stopwatch>(StringComparer.Ordinal);
        readonly List<string> order = new List<string>();
        readonly Stopwatch wall = Stopwatch.StartNew();
        readonly StringBuilder lines = new StringBuilder("# t_sim_s wall_s_in_period stage=seconds ...\n");
        double lastWall;
        readonly List<(string name, Func<double> read)> counters = new List<(string, Func<double>)>();
        /// <summary>A running count printed on every line, for example fire simulation rewinds.</summary>
        public void Counter(string name, Func<double> read) => counters.Add((name, read));

        public Scope Time(string stage)
        {
            if (!stages.TryGetValue(stage, out var watch)) { watch = new Stopwatch(); stages[stage] = watch; order.Add(stage); }
            return new Scope(watch);
        }

        public void Line(double t, params (string name, Stopwatch watch)[] inside)
        {
            double now = wall.Elapsed.TotalSeconds, period = now - lastWall, timed = 0; lastWall = now;
            lines.Append(t.ToString("F1", CultureInfo.InvariantCulture)).Append(' ').Append(period.ToString("F2", CultureInfo.InvariantCulture));
            foreach (var name in order)
            {
                double s = stages[name].Elapsed.TotalSeconds; timed += s; stages[name].Reset();
                lines.Append(' ').Append(name).Append('=').Append(s.ToString("F2", CultureInfo.InvariantCulture));
            }
            lines.Append(" untimed=").Append(Math.Max(0, period - timed).ToString("F2", CultureInfo.InvariantCulture));
            // Parts of a timed stage (not added to the total), for example the noise model inside rgbd_capture.
            foreach (var (name, watch) in inside)
            {
                lines.Append(" [").Append(name).Append('=').Append(watch.Elapsed.TotalSeconds.ToString("F2", CultureInfo.InvariantCulture)).Append(']');
                watch.Reset();
            }
            foreach (var (name, read) in counters)
                lines.Append(' ').Append(name).Append('=').Append(read().ToString("0.###", CultureInfo.InvariantCulture));
            lines.Append('\n');
        }

        public override string ToString() => lines.ToString();

        public readonly struct Scope : IDisposable
        {
            readonly Stopwatch watch;
            public Scope(Stopwatch w) { watch = w; watch.Start(); }
            public void Dispose() => watch.Stop();
        }
    }
}
