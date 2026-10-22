using System;
using System.Collections.Generic;
using UnityEngine;

namespace SmokeRescue.Brain
{
    /// <summary>Deterministic eight-neighbour A* and reachable frontier search. Unknown cells are blocked.</summary>
    public sealed class GridPlanner
    {
        readonly OccupancyGrid grid;
        readonly BrainConfig c;
        public Vector2 Home { get; set; }
        public GridPlanner(OccupancyGrid map, BrainConfig config) { grid = map; c = config; }
        public List<Vector2> Plan(Vector2 from, Vector2 to)
        {
            grid.PrepareNavigation();
            int end = grid.Cell(to);
            if (!Lead(from, out int start, out var lead) || !grid.Walkable(end)) return new List<Vector2>();
            Search(start, end, out var distance, out var parent);
            return float.IsPositiveInfinity(distance[end]) ? new List<Vector2>() : Join(lead, Reconstruct(parent, start, end));
        }
        /// <summary>Where a search starts. A robot can stand closer to something than the planning clearance (after a
        /// tight turn, a door leaf swinging in, or a wall appearing in the map beside it); every plan from such a cell
        /// failed, which left the robot turning on the spot. Step out to the nearest walkable cell through cells the
        /// body itself fits in (breadth first, at most lead_max_m), and plan from there. The lead cells are returned so
        /// the path starts where the robot stands.</summary>
        internal bool Lead(Vector2 from, out int start, out List<int> lead)
        {
            lead = new List<int>(); start = grid.Cell(from);
            if (start < 0) return false;
            if (grid.Walkable(start)) return true;
            var parent = new Dictionary<int, int> { [start] = -1 }; var queue = new Queue<int>(); queue.Enqueue(start);
            int reach = Mathf.CeilToInt(c.lead_max_m / grid.CellM), sx = start % grid.Size, sy = start / grid.Size;
            // Out of a fire margin the walk is as long as the margin is deep (agent run 9: 1.4 m and more), and every
            // step already moves away from the fire, so the lead_max_m box does not apply.
            bool hot = grid.FireDistance(start) <= c.fire_clearance_m;
            if (hot) reach = grid.Size;
            while (queue.Count > 0)
            {
                int cell = queue.Dequeue(), x = cell % grid.Size, y = cell / grid.Size;
                if (cell != start && grid.Walkable(cell))
                {
                    for (int i = cell; i != start; i = parent[i]) lead.Add(i);
                    lead.Reverse(); start = cell; return true;
                }
                for (int k = 0; k < 4; k++)
                {
                    int nx = x + (k == 0 ? 1 : k == 1 ? -1 : 0), ny = y + (k == 2 ? 1 : k == 3 ? -1 : 0);
                    if (!grid.Inside(nx, ny) || Math.Abs(nx - sx) > reach || Math.Abs(ny - sy) > reach) continue;
                    int n = ny * grid.Size + nx;
                    // Inside a fire margin, any free step that does not move closer to the fire is allowed: a robot must
                    // always be able to leave a hot spot, whoever marked it.
                    bool escaping = hot && grid.FireDistance(n) >= grid.FireDistance(cell) - 1e-4f &&
                                    grid.Occupancy(n) == 1 && grid.ObstacleDistance(n) > c.robot_radius_m * .8f;
                    if (parent.ContainsKey(n) || !(grid.Passable(n) || escaping)) continue;
                    parent[n] = cell; queue.Enqueue(n);
                }
            }
            return false;
        }
        /// <summary>A step on the way out of a tight or hot spot: passable, or free with room for the body and no closer
        /// to fire than the step before (leaving a fire margin).</summary>
        bool LeadStep(int cell, int previous) => grid.Passable(cell) ||
            (grid.Occupancy(cell) == 1 && grid.ObstacleDistance(cell) > c.robot_radius_m * .8f && grid.FireDistance(cell) >= grid.FireDistance(previous) - 1e-4f);
        List<Vector2> Join(List<int> lead, List<Vector2> path)
        {
            if (lead.Count == 0 || path.Count == 0) return path;
            var result = new List<Vector2>();
            foreach (int cell in lead) result.Add(grid.Centre(cell));
            for (int i = 1; i < path.Count; i++) result.Add(path[i]);
            return result;
        }
        public Vector2? SelectFrontier(Vector2 from, Func<Vector2, bool> excluded = null)
        {
            var result = Frontier(from, excluded); return result?.position;
        }
        internal sealed class FrontierChoice
        { public Vector2 position, claimCentre; public List<Vector2> path; public float cost, score, claimRadius; public int information; public bool towardPartner; }
        internal FrontierChoice Frontier(Vector2 from, Func<Vector2, bool> excluded = null, bool partnerSearchedToo = false)
        {
            var all = Frontiers(from, excluded, partnerSearchedToo); return all.Count > 0 ? all[0] : null;
        }
        public sealed class EvidenceLook
        { public Vector2 toward; public List<Vector2> path; public float cost; public int mappedCells; }

        /// <summary>Finish collecting local indoor evidence before leaving the camera's search horizon. The target is
        /// mapped unsearched floor, not an assumed room. The body follows proven own floor and stops on its side of
        /// the evidence boundary. No walkable cell beyond that boundary or complete door jambs are required.</summary>
        public EvidenceLook LocalEvidenceLook(Vector2 from, Func<Vector2, bool> proven, Func<Vector2, bool> excluded,
            Func<Vector2, Vector2, bool> tried)
        {
            grid.PrepareNavigation();
            if (!Lead(from, out int start, out var lead)) return null;
            var indoor = new Dictionary<int, bool>();
            bool Inside(int i)
            {
                if (!indoor.TryGetValue(i, out bool value)) indoor[i] = value = proven(grid.Centre(i));
                return value;
            }
            bool Allowed(int i) => grid.OwnOccupancy(i) == 1 && Inside(i);
            if (!Allowed(start) || lead.Exists(i => !Allowed(i))) return null;
            Search(start, -1, out var distance, out var parent, Allowed, c.search_max_range_m);
            var stands = new List<int>();
            for (int i = 0; i < grid.Count; i++)
                if (!float.IsPositiveInfinity(distance[i]) && (excluded == null || !excluded(grid.Centre(i)))) stands.Add(i);
            if (stands.Count == 0) return null;
            var labels = grid.UnsearchedRegions(); var counts = new Dictionary<int, int>();
            foreach (int label in labels) if (label >= 0) counts[label] = counts.TryGetValue(label, out int n) ? n + 1 : 1;
            EvidenceLook best = null; float bestGap = float.PositiveInfinity;
            int reach = Mathf.CeilToInt(2 * c.search_max_range_m / grid.CellM);
            int sx = start % grid.Size, sy = start / grid.Size;
            for (int y = Math.Max(0, sy - reach); y <= Math.Min(grid.Size - 1, sy + reach); y++)
                for (int x = Math.Max(0, sx - reach); x <= Math.Min(grid.Size - 1, sx + reach); x++)
                {
                    int cell = y * grid.Size + x, label = labels[cell]; var toward = grid.Centre(cell);
                    if (label < 0 || counts[label] * grid.CellM * grid.CellM < c.frontier_wide_min_m2 || grid.OpenGroundAt(toward) || Inside(cell) ||
                        (excluded != null && excluded(toward))) continue;
                    // Use the closest proven standpoint to this evidence boundary. A nearer-to-robot point can still
                    // be too far back to see a ceiling through the opening.
                    int stand = -1; float gap = float.PositiveInfinity;
                    foreach (int i in stands)
                    {
                        float d = Vector2.Distance(grid.Centre(i), toward);
                        if (d > c.search_max_range_m || d >= gap || (tried != null && tried(grid.Centre(i), toward)) ||
                            !grid.EvidenceSightLineClear(grid.Centre(i), toward)) continue;
                        gap = d; stand = i;
                    }
                    if (stand < 0 || (best != null && (distance[stand] > best.cost ||
                        (distance[stand] == best.cost && gap >= bestGap)))) continue;
                    bestGap = gap;
                    best = new EvidenceLook { toward = toward, path = Join(lead, Reconstruct(parent, start, stand)),
                        cost = distance[stand], mappedCells = counts[label] };
                }
            return best;
        }
        /// <summary>Every worthwhile reachable frontier cluster, best score first. partnerSearchedToo keeps frontier cells a
        /// partner's camera has searched (normally dropped), for following a partner through the one way on.</summary>
        internal List<FrontierChoice> Frontiers(Vector2 from, Func<Vector2, bool> excluded = null, bool partnerSearchedToo = false)
        {
            var choices = new List<FrontierChoice>();
            grid.PrepareNavigation();
            if (!Lead(from, out int start, out var lead)) return choices;
            Search(start, -1, out var distance, out var parent);
            var boundary = new bool[grid.Count];
            var peek = PeekCells();
            for (int i = 0; i < grid.Count; i++)
            {
                if (float.IsPositiveInfinity(distance[i]) || !grid.Walkable(i) || (!partnerSearchedToo && grid.PeerExplored(i))) continue;
                int x = i % grid.Size, y = i / grid.Size;
                for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++)
                    if (Math.Abs(dx) + Math.Abs(dy) == 1 && grid.Inside(x + dx, y + dy) && grid.Unsearched((y + dy) * grid.Size + x + dx))
                        boundary[i] = true;
                if (peek[i]) boundary[i] = true;
                // Filters run on boundary cells only, so a caller counting rejections counts frontier cells.
                if (boundary[i] && excluded != null && excluded(grid.Centre(i))) boundary[i] = false;
            }
            var queue = new Queue<int>();
            for (int i = 0; i < grid.Count; i++)
            {
                if (!boundary[i]) continue;
                boundary[i] = false; queue.Enqueue(i); Vector2 seed = grid.Centre(i);
                int target = -1; float targetCost = float.PositiveInfinity; var unknown = new HashSet<int>();
                var cluster = new List<int>();
                while (queue.Count > 0)
                {
                    int cell = queue.Dequeue(), x = cell % grid.Size, y = cell / grid.Size;
                    cluster.Add(cell);
                    if (Vector2.Distance(from, grid.Centre(cell)) >= c.frontier_min_distance_m && distance[cell] < targetCost)
                    { targetCost = distance[cell]; target = cell; }
                    // A peek cell's information lies past the tight strip, up to frontier_peek_m away.
                    int r = peek[cell] ? Math.Max(c.frontier_information_radius_cells, Mathf.CeilToInt(c.frontier_peek_m / grid.CellM)) : c.frontier_information_radius_cells;
                    for (int dy = -r; dy <= r; dy++) for (int dx = -r; dx <= r; dx++)
                        if (grid.Inside(x + dx, y + dy))
                        {
                            int n = (y + dy) * grid.Size + x + dx;
                            if (grid.Unsearched(n)) unknown.Add(n);
                        }
                    for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++)
                    {
                        if ((dx == 0 && dy == 0) || !grid.Inside(x + dx, y + dy)) continue;
                        int n = (y + dy) * grid.Size + x + dx;
                        // Bounded clusters: a long edge of unsearched floor becomes several targets to choose from.
                        if (boundary[n] && Vector2.Distance(grid.Centre(n), seed) <= c.frontier_cluster_max_m) { boundary[n] = false; queue.Enqueue(n); }
                    }
                }
                if (target < 0) continue;
                // The claim covers where this robot is going, not the whole cluster: with the LiDAR the boundary of
                // camera-searched floor is one cluster across the house, and a cluster-wide claim left the partner
                // nothing to search.
                Vector2 centre = grid.Centre(target); float radius = c.peer_claim_radius_m;
                // Nearest worthwhile frontier first, so an area is finished before the robot walks on: the size bonus
                // is capped, and slivers too small to hide a person (under furniture, wall seams) are ignored.
                if (unknown.Count < c.frontier_min_information_cells) continue;
                float score = targetCost - c.frontier_gain_weight * Math.Min(unknown.Count, c.frontier_gain_cap_cells);
                choices.Add(new FrontierChoice { position = grid.Centre(target), path = Join(lead, Reconstruct(parent, start, target)),
                    cost = targetCost, score = score, information = unknown.Count, claimCentre = centre, claimRadius = radius });
            }
            // Stable: equal scores keep discovery (row-major) order, so runs stay deterministic.
            var order = new List<int>(); for (int i = 0; i < choices.Count; i++) order.Add(i);
            order.Sort((x, y) => { int k = choices[x].score.CompareTo(choices[y].score); return k != 0 ? k : x.CompareTo(y); });
            var sorted = new List<FrontierChoice>(); foreach (int i in order) sorted.Add(choices[i]);
            return sorted;
        }
        /// <summary>
        /// Standable cells that look into unsearched floor through cells too tight for the body: a multi-source search
        /// from every unsearched cell, out through known non-wall cells that are not standable, up to frontier_peek_m;
        /// the standable cells it reaches are marked. Never through a wall or an obstacle cell. Hidden-corner run: a
        /// person behind a desk across a room corner had no standable cell beside her floor, so no frontier was offered
        /// and the robot finished with her unfound.
        /// </summary>
        bool[] PeekCells()
        {
            var peek = new bool[grid.Count];
            if (c.frontier_peek_m <= 0) return peek;
            var depth = new int[grid.Count]; for (int i = 0; i < grid.Count; i++) depth[i] = -1;
            int maxDepth = Mathf.FloorToInt(c.frontier_peek_m / grid.CellM);
            var queue = new Queue<int>();
            for (int i = 0; i < grid.Count; i++) if (grid.Unsearched(i)) { depth[i] = 0; queue.Enqueue(i); }
            while (queue.Count > 0)
            {
                int cell = queue.Dequeue(), x = cell % grid.Size, y = cell / grid.Size;
                if (depth[cell] >= maxDepth) continue;
                for (int k = 0; k < 4; k++)
                {
                    int nx = x + (k == 0 ? 1 : k == 1 ? -1 : 0), ny = y + (k == 2 ? 1 : k == 3 ? -1 : 0);
                    if (!grid.Inside(nx, ny)) continue;
                    int n = ny * grid.Size + nx;
                    // Known free floor, or an obstacle low enough to look over by the live 3D map (a desk, a bed); never a
                    // wall or anything taller than the camera's line of sight.
                    if (depth[n] >= 0 || (grid.Occupancy(n) != 1 && !(grid.Occupancy(n) == 255 && grid.SeeOver(n)))) continue;
                    depth[n] = depth[cell] + 1;
                    if (grid.Walkable(n)) { if (depth[cell] > 0) peek[n] = true; continue; }   // adjacent ones are frontiers already
                    queue.Enqueue(n);
                }
            }
            return peek;
        }
        /// <summary>Unsearched walkable floor in the search area that no known path reaches (behind debris, a closed door or
        /// fire): the nearest such cell, the reachable walkable cell nearest to it, and how many such cells there are.</summary>
        internal (Vector2 point, Vector2 approach, int cells)? NearestBlocked(Vector2 from, Func<Vector2, bool> inSearchArea)
        {
            grid.PrepareNavigation();
            if (!Lead(from, out int start, out _)) return null;
            Search(start, -1, out var distance, out _);
            int best = -1, count = 0; float bestM = float.PositiveInfinity;
            for (int i = 0; i < grid.Count; i++)
            {
                // Walkable floor the paths cannot reach: a separate pocket of floor. Free cells too close to a wall are left
                // out, since every wall has a strip of them.
                if (!float.IsPositiveInfinity(distance[i]) || !grid.Walkable(i) || !grid.Unsearched(i) || !inSearchArea(grid.Centre(i))) continue;
                count++; float m = Vector2.Distance(from, grid.Centre(i));
                if (m < bestM) { bestM = m; best = i; }
            }
            if (best < 0 || count < c.frontier_min_information_cells) return null;
            int approach = -1; float approachM = float.PositiveInfinity;
            for (int i = 0; i < grid.Count; i++)
            {
                if (float.IsPositiveInfinity(distance[i]) || !grid.Walkable(i)) continue;
                float m = Vector2.Distance(grid.Centre(i), grid.Centre(best));
                if (m < approachM) { approachM = m; approach = i; }
            }
            return approach < 0 ? ((Vector2, Vector2, int)?)null : (grid.Centre(best), grid.Centre(approach), count);
        }
        /// <summary>A standable viewing spot with a clear line to the person, nearest by path. The survivor_standoff_m ring
        /// is tried first, then the other radii of the viewing band (survivor_stop_min_m to survivor_stop_max_m), nearest
        /// to the standoff first. One ring alone failed whenever furniture, unmapped floor or a fire margin covered it: seed
        /// 929, robot B saw a person in a burning living room from 5.3 m and logged "No safe known survivor approach".</summary>
        public List<Vector2> ApproachPath(Vector2 from, Vector2 person)
        {
            grid.PrepareNavigation();
            if (!Lead(from, out int start, out var lead)) return new List<Vector2>();
            Search(start, -1, out var distance, out var parent);
            var radii = new List<float> { c.survivor_standoff_m };
            float step = 2 * grid.CellM;
            for (float r = c.survivor_stop_min_m; r <= c.survivor_stop_max_m + 1e-4f; r += step)
                if (Mathf.Abs(r - c.survivor_standoff_m) > grid.CellM * .5f) radii.Add(r);
            radii.Sort((x, y) => Mathf.Abs(x - c.survivor_standoff_m).CompareTo(Mathf.Abs(y - c.survivor_standoff_m)));
            foreach (float radius in radii)
            {
                int best = -1; float score = float.PositiveInfinity;
                for (int sample = 0; sample < c.standoff_samples; sample++)
                {
                    float a = sample * 2 * Mathf.PI / c.standoff_samples;
                    int cell = grid.Cell(person + new Vector2(Mathf.Cos(a), Mathf.Sin(a)) * radius);
                    if (cell < 0 || !grid.Walkable(cell) || distance[cell] >= score) continue;
                    float range = Vector2.Distance(grid.Centre(cell), person);
                    if (range < c.survivor_stop_min_m || range > c.survivor_stop_max_m || !ClearView(grid.Centre(cell), person)) continue;
                    score = distance[cell]; best = cell;
                }
                if (best >= 0) return Join(lead, Reconstruct(parent, start, best));
            }
            return new List<Vector2>();
        }
        /// <summary>No mapped wall on the straight line from a standpoint to a person, ignoring the last
        /// survivor_view_clear_m before the person (a body against a wall, or the body itself, may be mapped there). A
        /// standoff on the far side of a wall sees nothing of the person.</summary>
        bool ClearView(Vector2 from, Vector2 person)
        {
            float length = Vector2.Distance(from, person) - c.survivor_view_clear_m; if (length <= 0) return true;
            Vector2 dir = (person - from).normalized; int steps = Mathf.CeilToInt(length / (grid.CellM * .5f));
            for (int i = 1; i <= steps; i++) if (grid.OwnOccupancy(grid.Cell(from + dir * (length * i / steps))) == 255) return false;
            return true;
        }
        /// <summary>The coolest reachable known cell. allowed, when given, limits the candidates (stay_inside: floor proven
        /// indoors only), and preferStart false drops the pull toward the start, which is outside the house.</summary>
        internal List<Vector2> CoolestPath(Vector2 from, Func<Vector2, bool> allowed = null, bool preferStart = true, Func<int, float> airCost = null)
        {
            grid.PrepareNavigation();
            if (!Lead(from, out int start, out var lead)) return new List<Vector2>();
            Search(start, -1, out var distance, out var parent);
            int best = -1; float bestScore = float.PositiveInfinity;
            for (int i = 0; i < grid.Count; i++)
            {
                if (float.IsPositiveInfinity(distance[i]) || Vector2.Distance(from, grid.Centre(i)) < c.retreat_min_distance_m) continue;
                if (allowed != null && !allowed(grid.Centre(i))) continue;
                // Heat dominates, then prefer the start and a short escape. All candidates are known and reachable.
                float score = grid.Heat(i) + (airCost != null ? airCost(i) : 0) +
                    (preferStart ? Vector2.Distance(grid.Centre(i), Home) * c.retreat_start_weight : 0) + distance[i] * c.retreat_path_weight;
                if (score < bestScore) { bestScore = score; best = i; }
            }
            return best < 0 ? (allowed != null ? new List<Vector2>() : Plan(from, Home)) : Join(lead, Reconstruct(parent, start, best));
        }
        public bool PathSafe(IReadOnlyList<Vector2> path, int from = 0)
        {
            grid.PrepareNavigation(); bool leading = true;
            for (int i = Math.Max(0, from); i < path.Count; i++)
            {
                // A path may begin with a lead out of a tight spot (passable cells), then must stay walkable.
                int cell = grid.Cell(path[i]);
                if (grid.Walkable(cell)) leading = false;
                else if (!leading || !LeadStep(cell, i > from ? grid.Cell(path[i - 1]) : cell)) return false;
                if (i > from)
                {
                    int previous = grid.Cell(path[i - 1]);
                    int dx = cell % grid.Size - previous % grid.Size, dy = cell / grid.Size - previous / grid.Size;
                    if (!leading && dx != 0 && dy != 0 && (!grid.Walkable(previous + dx) || !grid.Walkable(previous + dy * grid.Size))) return false;
                }
            }
            return true;
        }
        // Check a pursuit chord too, since its lookahead can otherwise cut inside an inflated corner.
        public bool SegmentSafe(Vector2 a, Vector2 b)
        {
            grid.PrepareNavigation(); int steps = Math.Max(1, Mathf.CeilToInt(Vector2.Distance(a, b) / (grid.CellM * .25f)));
            bool leading = true; int previous = grid.Cell(a);
            for (int i = 0; i <= steps; i++)
            {
                // Same rule as PathSafe: leaving a tight spot is allowed through passable cells, entering one is not.
                int cell = grid.Cell(Vector2.Lerp(a, b, i / (float)steps));
                if (grid.Walkable(cell)) leading = false;
                else if (!leading || !LeadStep(cell, previous)) return false;
                previous = cell;
            }
            return true;
        }

        /// <summary>Check the accelerating capsule through this heading change. Each pose also needs a straight
        /// braking corridor, so the next control tick can abandon the arc safely. Unknown cells block the body,
        /// including its nose and tail. Quarter-cell sampling plus a half-sample pad covers motion between samples.</summary>
        public bool ArcSafe(BrainPose pose, float currentSpeed, float targetSpeed, float yawRateDeg, float turnDeg, float dt,
            Func<Vector2, bool> allowed = null)
        {
            if (currentSpeed < 0 || targetSpeed <= 0 || Mathf.Abs(yawRateDeg) <= 0 || dt <= 0) return false;
            grid.PrepareNavigation();
            if (!grid.Walkable(grid.Cell(pose.position))) return false;
            float omega = yawRateDeg * Mathf.Deg2Rad;
            float remaining = Mathf.Abs(turnDeg / yawRateDeg), speed = currentSpeed;
            float sampleM = grid.CellM * .25f, pad = sampleM * .5f;
            bool StopFits(BrainPose at, float v)
            {
                float stopping = v * v / (2 * c.limits.max_accel_mps2) + v * dt;
                var end = at.position + at.Forward * stopping;
                if (allowed != null)
                {
                    int n = Mathf.Max(1, Mathf.CeilToInt(stopping / sampleM));
                    for (int i = 0; i <= n; i++) if (!allowed(Vector2.Lerp(at.position, end, i / (float)n))) return false;
                }
                return SegmentSafe(at.position, end) && CapsuleKnownFree(at, stopping, pad);
            }
            if (!StopFits(pose, speed)) return false;
            while (remaining > 0)
            {
                float tick = Mathf.Min(dt, remaining);
                speed = Mathf.MoveTowards(speed, targetSpeed, c.limits.max_accel_mps2 * dt);
                if (!StopFits(pose, speed)) return false;
                int samples = Mathf.Max(1, Mathf.CeilToInt((speed + Mathf.Abs(omega) * c.body_half_length_m) * tick / sampleM));
                float h = tick / samples;
                for (int i = 0; i < samples; i++)
                {
                    var previous = pose.position; float yaw = pose.yawRad + omega * h;
                    pose.position += new Vector2(Mathf.Cos(yaw) - Mathf.Cos(pose.yawRad),
                        Mathf.Sin(yaw) - Mathf.Sin(pose.yawRad)) * (speed / omega);
                    pose.yawRad = yaw;
                    if (!SegmentSafe(previous, pose.position) || !StopFits(pose, speed)) return false;
                }
                remaining -= tick;
            }
            return true;
        }

        bool CapsuleKnownFree(BrainPose pose, float stopping, float pad)
        {
            float nose = Mathf.Max(0, c.body_half_length_m - c.body_radius_m);
            Vector2 a = pose.position - pose.Forward * nose, b = pose.position + pose.Forward * (nose + stopping), d = b - a;
            // A cell is a square. Its half diagonal bounds its distance from its centre in every orientation.
            float radius = c.body_radius_m + grid.CellM * Mathf.Sqrt(2) / 2 + pad;
            Vector2 lo = Vector2.Min(a, b) - Vector2.one * radius, hi = Vector2.Max(a, b) + Vector2.one * radius;
            int x0 = Mathf.FloorToInt((lo.x - grid.Origin.x) / grid.CellM), x1 = Mathf.FloorToInt((hi.x - grid.Origin.x) / grid.CellM);
            int y0 = Mathf.FloorToInt((lo.y - grid.Origin.y) / grid.CellM), y1 = Mathf.FloorToInt((hi.y - grid.Origin.y) / grid.CellM);
            for (int y = y0; y <= y1; y++) for (int x = x0; x <= x1; x++)
            {
                var p = grid.Origin + new Vector2(x + .5f, y + .5f) * grid.CellM;
                float u = d.sqrMagnitude > 0 ? Mathf.Clamp01(Vector2.Dot(p - a, d) / d.sqrMagnitude) : 0;
                if ((p - a - d * u).sqrMagnitude > radius * radius) continue;
                if (!grid.Inside(x, y) || grid.Occupancy(y * grid.Size + x) != 1) return false;
            }
            return true;
        }
        internal bool ReverseSafe(Vector2 a, Vector2 b)
        {
            if (!SegmentSafe(a, b)) return false;
            int steps = Math.Max(1, Mathf.CeilToInt(Vector2.Distance(a, b) / (grid.CellM * .25f)));
            float radius = c.robot_radius_m + grid.CellM * .707107f;
            int r = Mathf.CeilToInt(radius / grid.CellM);
            for (int step = 0; step <= steps; step++)
            {
                var p = Vector2.Lerp(a, b, step / (float)steps); int cell = grid.Cell(p);
                for (int dy = -r; dy <= r; dy++) for (int dx = -r; dx <= r; dx++)
                {
                    int x = cell % grid.Size + dx, y = cell / grid.Size + dy;
                    var centre = grid.Origin + new Vector2(x + .5f, y + .5f) * grid.CellM;
                    if (Vector2.Distance(centre, p) > radius) continue;
                    if (!grid.Inside(x, y) || grid.Occupancy(y * grid.Size + x) != 1) return false;
                }
            }
            return true;
        }
        void Search(int start, int goal, out float[] distance, out int[] parent, Func<int, bool> allowed = null,
            float maxCost = float.PositiveInfinity)
        {
            distance = new float[grid.Count]; parent = new int[grid.Count];
            for (int i = 0; i < distance.Length; i++) { distance[i] = float.PositiveInfinity; parent[i] = -1; }
            var heap = new CellHeap(); distance[start] = 0; heap.Push(start, Heuristic(start, goal), 0);
            while (heap.Count > 0)
            {
                var item = heap.Pop(); int cell = item.cell;
                if (item.cost > distance[cell]) continue;
                if (cell == goal) return;
                int x = cell % grid.Size, y = cell / grid.Size;
                for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++)
                {
                    if ((dx == 0 && dy == 0) || !grid.Inside(x + dx, y + dy)) continue;
                    int n = (y + dy) * grid.Size + x + dx;
                    if (!grid.Walkable(n) || (allowed != null && !allowed(n))) continue;
                    if (dx != 0 && dy != 0 && (!grid.Walkable(cell + dx) || !grid.Walkable(cell + dy * grid.Size))) continue;
                    if (allowed != null && dx != 0 && dy != 0 && (!allowed(cell + dx) || !allowed(cell + dy * grid.Size))) continue;
                    float step = grid.CellM * (dx != 0 && dy != 0 ? 1.41421356f : 1);
                    float cost = distance[cell] + step * (grid.Cost(cell) + grid.Cost(n)) / 2;
                    if (cost >= distance[n] || cost > maxCost) continue;
                    distance[n] = cost; parent[n] = cell; heap.Push(n, cost + Heuristic(n, goal), cost);
                }
            }
        }
        float Heuristic(int cell, int goal) => goal < 0 ? 0 : Vector2.Distance(grid.Centre(cell), grid.Centre(goal));
        List<Vector2> Reconstruct(int[] parents, int start, int goal)
        {
            var path = new List<Vector2>(); int i = goal;
            for (int guard = 0; guard < grid.Count; guard++)
            {
                path.Add(grid.Centre(i)); if (i == start) { path.Reverse(); return path; }
                i = parents[i]; if (i < 0) break;
            }
            return new List<Vector2>();
        }
        sealed class CellHeap
        {
            internal struct Item { public int cell; public float priority, cost; }
            readonly List<Item> items = new List<Item>();
            public int Count => items.Count;
            static bool Before(Item a, Item b) => a.priority < b.priority || (a.priority == b.priority && a.cell < b.cell);
            public void Push(int cell, float priority, float cost)
            {
                var value = new Item { cell = cell, priority = priority, cost = cost }; int index = items.Count; items.Add(value);
                while (index > 0)
                {
                    int parent = (index - 1) / 2; if (!Before(value, items[parent])) break;
                    items[index] = items[parent]; index = parent;
                }
                items[index] = value;
            }
            public Item Pop()
            {
                var root = items[0]; var tail = items[items.Count - 1]; items.RemoveAt(items.Count - 1);
                if (items.Count == 0) return root;
                int index = 0;
                while (index * 2 + 1 < items.Count)
                {
                    int child = index * 2 + 1;
                    if (child + 1 < items.Count && Before(items[child + 1], items[child])) child++;
                    if (!Before(items[child], tail)) break;
                    items[index] = items[child]; index = child;
                }
                items[index] = tail; return root;
            }
        }
    }
}
