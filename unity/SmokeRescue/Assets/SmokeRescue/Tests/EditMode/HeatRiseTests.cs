using NUnit.Framework;
using SmokeRescue.Brain;
using UnityEngine;

namespace SmokeRescue.Tests.EditMode
{
    public sealed partial class BrainTests
    {
        /// <summary>One cell fed thermal readings at 15 Hz: (flicker) hot-gas radiance jumping between 70 and 95 C, as robot B's
        /// floor cells did at 212.5 to 215 s of seed 934 (24 Oct 04:53 run); (rise) a real sustained rise of 120 C/s from 30 C.
        /// Returns whether the cell was ever dangerous while every reading stayed below fire_c, and when the rising one first
        /// became dangerous.</summary>
        static (bool flickerFire, double riseFireAt) HeatRun(float windowS)
        {
            var c = Settings(); c.grid_size_m = 4; c.heat_rise_window_s = windowS;
            var g = new OccupancyGrid(c); int flick = g.Cell(new Vector2(-1, 0)), rising = g.Cell(new Vector2(1, 0));
            bool flickerFire = false; double riseFireAt = double.NaN;
            for (int k = 0; k <= 30; k++)
            {
                double t = k / 15.0; g.DecayTo(t);
                g.ObserveHeat(flick, k % 2 == 0 ? 70 : 95, t);
                flickerFire |= g.Dangerous(flick);
                float temp = 30 + 120 * (float)t;
                if (temp < c.fire_c) { g.ObserveHeat(rising, temp, t); if (double.IsNaN(riseFireAt) && g.Dangerous(rising)) riseFireAt = t; }
            }
            return (flickerFire, riseFireAt);
        }

        /// <summary>heat_rise_window_s. Seed 934 (24 Oct 04:53 run), robot B: 0 cells measured at fire_c and 441 forecast as fire
        /// from its own frames in 2.5 s, from 15-25 C flicker between frames 67 ms apart (210-285 C/s); 28 routes cut. With a 1 s
        /// window the flicker is no fire; a sustained 120 C/s rise is still forecast before it is measured (it would reach
        /// fire_c at 1.42 s).</summary>
        [Test]
        public void AFlickerBetweenFramesIsNoFireForecastButASustainedRiseStillIs()
        {
            var frame = HeatRun(0);
            Assert.That(frame.flickerFire, Is.True, "control: the frame-to-frame slope forecasts fire from a 70-95 C flicker");
            var window = HeatRun(1);
            Assert.That(window.flickerFire, Is.False, "a 1 s slope: the flicker is no fire");
            Assert.That(window.riseFireAt, Is.LessThan(1.42), "a sustained 120 C/s rise is forecast before it reaches fire_c");
        }
    }
}
