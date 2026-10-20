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
            int start = grid.Cell(from), end = grid.Cell(to); grid.PrepareNavigation();
            if (!grid.Walkable(start) || !grid.Walkable(end)) return new List<Vector2>();
            Search(start, end, out var distance, out var parent);
            return float.IsPositiveInfinity(distance[end]) ? new List<Vector2>() : Reconstruct(parent, start, end);
        }
        public Vector2? SelectFrontier(Vector2 from, Func<Vector2, bool> excluded = null)
        {
            var result = Frontier(from, excluded); return result?.position;
        }
        internal sealed class FrontierChoice
        { public Vector2 position, claimCentre; public List<Vector2> path; public float cost, score, claimRadius; public int information; }
        internal FrontierChoice Frontier(Vector2 from, Func<Vector2, bool> excluded = null)
        {
            int start = grid.Cell(from); grid.PrepareNavigation();
            if (!grid.Walkable(start)) return null;
            Search(start, -1, out var distance, out var parent);
            var boundary = new bool[grid.Count];
            for (int i = 0; i < grid.Count; i++)
            {
                if (float.IsPositiveInfinity(distance[i]) || !grid.Walkable(i) || grid.PeerExplored(i) ||
                    (excluded != null && excluded(grid.Centre(i)))) continue;
                int x = i % grid.Size, y = i / grid.Size;
                for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++)
                    if (Math.Abs(dx) + Math.Abs(dy) == 1 && grid.Inside(x + dx, y + dy) && grid.Occupancy((y + dy) * grid.Size + x + dx) == 0)
                        boundary[i] = true;
            }
            FrontierChoice best = null; var queue = new Queue<int>();
            for (int i = 0; i < grid.Count; i++)
            {
                if (!boundary[i]) continue;
                boundary[i] = false; queue.Enqueue(i);
                int target = -1; float targetCost = float.PositiveInfinity; var unknown = new HashSet<int>();
                var cluster = new List<int>(); Vector2 sum = Vector2.zero;
                while (queue.Count > 0)
                {
                    int cell = queue.Dequeue(), x = cell % grid.Size, y = cell / grid.Size;
                    cluster.Add(cell); sum += grid.Centre(cell);
                    if (Vector2.Distance(from, grid.Centre(cell)) >= c.frontier_min_distance_m && distance[cell] < targetCost)
                    { targetCost = distance[cell]; target = cell; }
                    int r = c.frontier_information_radius_cells;
                    for (int dy = -r; dy <= r; dy++) for (int dx = -r; dx <= r; dx++)
                        if (grid.Inside(x + dx, y + dy))
                        {
                            int n = (y + dy) * grid.Size + x + dx;
                            if (grid.Occupancy(n) == 0) unknown.Add(n);
                        }
                    for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++)
                    {
                        if ((dx == 0 && dy == 0) || !grid.Inside(x + dx, y + dy)) continue;
                        int n = (y + dy) * grid.Size + x + dx;
                        if (boundary[n]) { boundary[n] = false; queue.Enqueue(n); }
                    }
                }
                if (target < 0) continue;
                Vector2 centre = sum / cluster.Count; float radius = c.peer_claim_radius_m;
                foreach (int cell in cluster) radius = Mathf.Max(radius, Vector2.Distance(centre, grid.Centre(cell)) + grid.CellM * .707107f);
                // Nearest worthwhile frontier first, so an area is finished before the robot walks on: the size bonus
                // is capped, and slivers too small to hide a person (under furniture, wall seams) are ignored.
                if (unknown.Count < c.frontier_min_information_cells) continue;
                float score = targetCost - c.frontier_gain_weight * Math.Min(unknown.Count, c.frontier_gain_cap_cells);
                if (best == null || score < best.score)
                    best = new FrontierChoice { position = grid.Centre(target), path = Reconstruct(parent, start, target),
                        cost = targetCost, score = score, information = unknown.Count, claimCentre = centre, claimRadius = radius };
            }
            return best;
        }
        internal List<Vector2> ApproachPath(Vector2 from, Vector2 person)
        {
            int start = grid.Cell(from); grid.PrepareNavigation();
            if (!grid.Walkable(start)) return new List<Vector2>();
            Search(start, -1, out var distance, out var parent);
            int best = -1; float score = float.PositiveInfinity;
            for (int sample = 0; sample < c.standoff_samples; sample++)
            {
                float a = sample * 2 * Mathf.PI / c.standoff_samples;
                int cell = grid.Cell(person + new Vector2(Mathf.Cos(a), Mathf.Sin(a)) * c.survivor_standoff_m);
                if (cell < 0 || !grid.Walkable(cell) || distance[cell] >= score) continue;
                float range = Vector2.Distance(grid.Centre(cell), person);
                if (range < c.survivor_stop_min_m || range > c.survivor_stop_max_m) continue;
                score = distance[cell]; best = cell;
            }
            return best < 0 ? new List<Vector2>() : Reconstruct(parent, start, best);
        }
        internal List<Vector2> CoolestPath(Vector2 from)
        {
            int start = grid.Cell(from); grid.PrepareNavigation();
            if (!grid.Walkable(start)) return new List<Vector2>();
            Search(start, -1, out var distance, out var parent);
            int best = -1; float bestScore = float.PositiveInfinity;
            for (int i = 0; i < grid.Count; i++)
            {
                if (float.IsPositiveInfinity(distance[i]) || Vector2.Distance(from, grid.Centre(i)) < c.retreat_min_distance_m) continue;
                // Heat dominates, then prefer the start and a short escape. All candidates are known and reachable.
                float score = grid.Heat(i) + Vector2.Distance(grid.Centre(i), Home) * c.retreat_start_weight + distance[i] * c.retreat_path_weight;
                if (score < bestScore) { bestScore = score; best = i; }
            }
            return best < 0 ? Plan(from, Home) : Reconstruct(parent, start, best);
        }
        public bool PathSafe(IReadOnlyList<Vector2> path, int from = 0)
        {
            grid.PrepareNavigation();
            for (int i = Math.Max(0, from); i < path.Count; i++)
            {
                int cell = grid.Cell(path[i]); if (!grid.Walkable(cell)) return false;
                if (i > from)
                {
                    int previous = grid.Cell(path[i - 1]);
                    int dx = cell % grid.Size - previous % grid.Size, dy = cell / grid.Size - previous / grid.Size;
                    if (dx != 0 && dy != 0 && (!grid.Walkable(previous + dx) || !grid.Walkable(previous + dy * grid.Size))) return false;
                }
            }
            return true;
        }
        // Check a pursuit chord too, since its lookahead can otherwise cut inside an inflated corner.
        internal bool SegmentSafe(Vector2 a, Vector2 b)
        {
            grid.PrepareNavigation(); int steps = Math.Max(1, Mathf.CeilToInt(Vector2.Distance(a, b) / (grid.CellM * .25f)));
            for (int i = 0; i <= steps; i++) if (!grid.Walkable(grid.Cell(Vector2.Lerp(a, b, i / (float)steps)))) return false;
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
        void Search(int start, int goal, out float[] distance, out int[] parent)
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
                    if (!grid.Walkable(n)) continue;
                    if (dx != 0 && dy != 0 && (!grid.Walkable(cell + dx) || !grid.Walkable(cell + dy * grid.Size))) continue;
                    float step = grid.CellM * (dx != 0 && dy != 0 ? 1.41421356f : 1);
                    float cost = distance[cell] + step * (grid.Cost(cell) + grid.Cost(n)) / 2;
                    if (cost >= distance[n]) continue;
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
