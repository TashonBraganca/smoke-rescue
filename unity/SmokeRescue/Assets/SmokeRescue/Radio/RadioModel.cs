using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using UnityEngine;

namespace SmokeRescue.Radio
{
    /// <summary>
    /// Link rule from docs/INTERFACES.md section 4.2: delivered when distance is under range_m and fewer than
    /// max_walls + 1 walls lie on the straight line between the antennas (raycast against the Walls layer only).
    /// Delay is 50 ms plus 1 ms per metre. In recording mode the fingerprints do not exist yet, so the recorder
    /// logs one tick per sender every broadcast period and the brain fills them in later.
    /// </summary>
    public sealed class RadioModel
    {
        public const float AntennaHeight = 0.6f;
        readonly float rangeM;
        readonly int maxWalls;
        readonly int wallMask;

        public RadioModel(float rangeM, int maxWalls, string wallsLayer = "Walls")
        {
            this.rangeM = rangeM;
            this.maxWalls = maxWalls;
            wallMask = LayerMask.GetMask(wallsLayer);
        }

        public struct Link
        {
            public bool delivered;
            public string reason;      // ok, distance, walls
            public float distance;
            public int walls;
            public double delay;
        }

        /// <summary>Walls crossed between two floor positions, counting each wall collider once.</summary>
        public int CountWalls(Vector3 a, Vector3 b)
        {
            a.y = b.y = AntennaHeight;
            Vector3 d = b - a;
            float len = d.magnitude;
            if (len < 1e-4f) return 0;
            var hits = Physics.RaycastAll(a, d / len, len, wallMask, QueryTriggerInteraction.Ignore);
            return hits.Select(h => h.collider).Distinct().Count();
        }

        public Link Evaluate(Vector3 from, Vector3 to)
        {
            var link = new Link { distance = Vector2.Distance(new Vector2(from.x, from.z), new Vector2(to.x, to.z)) };
            link.walls = CountWalls(from, to);
            link.delay = 0.050 + 0.001 * link.distance;
            if (link.distance >= rangeM) link.reason = "distance";
            else if (link.walls > maxWalls) link.reason = "walls";
            else { link.reason = "ok"; link.delivered = true; }
            return link;
        }
    }

    /// <summary>Builds radio.txt in the INTERFACES 4.3 format.</summary>
    public sealed class RadioLog
    {
        readonly StringBuilder sb = new StringBuilder();
        readonly Dictionary<string, int> ticks = new Dictionary<string, int>();

        public RadioLog()
        {
            sb.Append("# radio delivery log, one line per broadcast tick and receiver\n");
            sb.Append("# t_send t_deliver from to msg_id delivered reason\n");
        }

        public void Tick(double tSend, string from, IEnumerable<(string to, RadioModel.Link link)> receivers)
        {
            ticks.TryGetValue(from, out int n);
            ticks[from] = n + 1;
            string letter = from.StartsWith("robot_") ? from.Substring(6) : from;
            string msgId = $"{letter}-{n:000000}";
            foreach (var (to, link) in receivers)
            {
                string deliver = link.delivered ? (tSend + link.delay).ToString("F6", CultureInfo.InvariantCulture) : "-1";
                sb.Append(tSend.ToString("F6", CultureInfo.InvariantCulture)).Append(' ').Append(deliver).Append(' ')
                  .Append(from).Append(' ').Append(to).Append(' ').Append(msgId).Append(' ')
                  .Append(link.delivered ? 1 : 0).Append(' ').Append(link.reason).Append('\n');
            }
        }

        public override string ToString() => sb.ToString();
    }
}
