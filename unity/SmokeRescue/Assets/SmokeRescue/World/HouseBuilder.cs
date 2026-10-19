using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace SmokeRescue.World
{
    /// <summary>
    /// Builds the house from a FloorPlan. Pure function of its inputs: same plan, same hierarchy.
    ///
    /// Structure, all in Unity world metres (x east, z north, y up):
    ///   House/Ground     outdoor slab so robots can stand outside the doors
    ///   House/Floors     one slab per room, top at y = 0
    ///   House/Walls      structural wall boxes on the Walls layer (radio raycasts hit these), cut at doors
    ///   House/Finishes   per-room wall skins, because one wall faces two rooms with different finishes
    ///   House/Doors      casings and leaves
    ///   House/Ceiling    per-room ceilings and the roof, on the Ceiling layer so the showcase view can hide them
    /// </summary>
    public static class HouseBuilder
    {
        public const string RootName = "House";
        public const string WallsLayer = "Walls";
        public const string CeilingLayer = "Ceiling";
        public const string PropsLayer = "Props";
        public const string StructureMaterial = "paint_white_textured";
        public const string TrimMaterial = "prop_wood_cabinet";
        public const string GroundMaterial = "concrete_rough";
        const float Eps = 1e-4f;
        const float SkinOffset = 0.002f;
        const float CasingWidth = 0.07f;
        const float LeafThickness = 0.04f;

        public static GameObject Build(FloorPlan plan, Func<string, Material> materials)
        {
            int wallsLayer = RequireLayer(WallsLayer);
            int ceilingLayer = RequireLayer(CeilingLayer);
            var root = new GameObject(RootName);
            var ground = Child(root, "Ground");
            var floors = Child(root, "Floors");
            var walls = Child(root, "Walls");
            var finishes = Child(root, "Finishes");
            var doors = Child(root, "Doors");
            var ceiling = Child(root, "Ceiling");

            Rect fp = Rect.MinMaxRect(plan.footprint[0], plan.footprint[1], plan.footprint[2], plan.footprint[3]);
            float H = plan.wall_height;

            // Ground: 6 m apron around the house, top 2 mm below the floors.
            AddBox(ground, "Ground", new Vector3(fp.center.x, -0.052f, fp.center.y),
                new Vector3(fp.width + 12f, 0.1f, fp.height + 12f), materials(GroundMaterial), collider: true);

            foreach (var room in plan.rooms)
            {
                Rect r = room.Rect;
                AddBox(floors, $"Floor_{room.id}", new Vector3(r.center.x, -0.05f, r.center.y),
                    new Vector3(r.width, 0.1f, r.height), materials(room.floor), collider: true);
            }

            BuildWalls(plan, fp, walls, materials(StructureMaterial), wallsLayer);
            foreach (var room in plan.rooms) BuildFinishes(plan, fp, room, finishes, materials(room.walls));
            foreach (var door in plan.doors) BuildDoor(plan, fp, door, doors, materials(TrimMaterial));

            foreach (var room in plan.rooms)
            {
                Rect r = room.Rect;
                var go = AddQuad(ceiling, $"Ceiling_{room.id}", new Vector3(r.center.x, H - 0.001f, r.center.y),
                    Vector3.down, r.width, r.height, materials(room.ceiling ?? StructureMaterial));
                go.layer = ceilingLayer;
            }
            var roof = AddBox(ceiling, "Roof", new Vector3(fp.center.x, H + 0.075f, fp.center.y),
                new Vector3(fp.width, 0.15f, fp.height), materials(GroundMaterial), collider: false);
            roof.layer = ceilingLayer;
            return root;
        }

        // ------------------------------------------------------------------ walls

        /// <summary>A wall line: fixed coordinate `at` on axis, spanning [a, b] along the other axis.</summary>
        public struct WallLine
        {
            public bool alongX;     // true: wall runs along x at fixed z
            public float at;
            public float a, b;
            public float thickness;
            public float centre;    // the wall's centre coordinate (exterior walls sit inside the footprint)
        }

        public static List<WallLine> WallLines(FloorPlan plan, Rect fp)
        {
            var spans = new SortedDictionary<(bool, float), List<(float, float)>>();
            void Add(bool alongX, float at, float a, float b)
            {
                var key = (alongX, Mathf.Round(at * 1000f) / 1000f);
                if (!spans.TryGetValue(key, out var list)) spans[key] = list = new List<(float, float)>();
                list.Add((Mathf.Min(a, b), Mathf.Max(a, b)));
            }
            foreach (var room in plan.rooms)
            {
                Rect r = room.Rect;
                Add(true, r.yMin, r.xMin, r.xMax);
                Add(true, r.yMax, r.xMin, r.xMax);
                Add(false, r.xMin, r.yMin, r.yMax);
                Add(false, r.xMax, r.yMin, r.yMax);
            }
            var lines = new List<WallLine>();
            foreach (var kv in spans)
            {
                var (alongX, at) = kv.Key;
                foreach (var (a, b) in Union(kv.Value))
                {
                    bool exterior = alongX ? (Near(at, fp.yMin) || Near(at, fp.yMax)) : (Near(at, fp.xMin) || Near(at, fp.xMax));
                    float t = exterior ? plan.exterior_wall_thickness : plan.interior_wall_thickness;
                    float lo = alongX ? fp.yMin : fp.xMin;
                    float centre = !exterior ? at : (Near(at, lo) ? at + t / 2 : at - t / 2);
                    lines.Add(new WallLine { alongX = alongX, at = at, a = a, b = b, thickness = t, centre = centre });
                }
            }
            return lines;
        }

        static void BuildWalls(FloorPlan plan, Rect fp, GameObject parent, Material mat, int layer)
        {
            float H = plan.wall_height, D = plan.door_height;
            foreach (var line in WallLines(plan, fp))
            {
                var cuts = DoorCuts(plan, line.alongX, line.at, line.a, line.b);
                string id = $"{(line.alongX ? "Z" : "X")}{line.at:0.00}";
                int k = 0;
                foreach (var (a, b) in Subtract((line.a, line.b), cuts))
                    WallBox(parent, $"Wall_{id}_{k++}", line, a, b, 0f, H, mat, layer);
                foreach (var (a, b) in cuts)
                    WallBox(parent, $"Lintel_{id}_{k++}", line, a, b, D, H, mat, layer);
            }
        }

        static void WallBox(GameObject parent, string name, WallLine line, float a, float b, float y0, float y1, Material mat, int layer)
        {
            float len = b - a, mid = (a + b) / 2, h = y1 - y0;
            Vector3 c = line.alongX ? new Vector3(mid, y0 + h / 2, line.centre) : new Vector3(line.centre, y0 + h / 2, mid);
            Vector3 s = line.alongX ? new Vector3(len, h, line.thickness) : new Vector3(line.thickness, h, len);
            AddBox(parent, name, c, s, mat, collider: true).layer = layer;
        }

        static List<(float, float)> DoorCuts(FloorPlan plan, bool alongX, float at, float a, float b)
        {
            var cuts = new List<(float, float)>();
            foreach (var d in plan.doors)
            {
                if (d.RunsAlongX != alongX) continue;
                float dAt = alongX ? d.at[1] : d.at[0];
                float dPos = alongX ? d.at[0] : d.at[1];
                if (!Near(dAt, at) || dPos < a - Eps || dPos > b + Eps) continue;
                cuts.Add((dPos - d.width / 2, dPos + d.width / 2));
            }
            cuts.Sort();
            return cuts;
        }

        // ------------------------------------------------------------------ finishes

        static void BuildFinishes(FloorPlan plan, Rect fp, FloorPlan.Room room, GameObject parent, Material mat)
        {
            Rect r = room.Rect;
            float H = plan.wall_height, D = plan.door_height;
            float T(bool alongX, float at) =>
                (alongX ? (Near(at, fp.yMin) || Near(at, fp.yMax)) : (Near(at, fp.xMin) || Near(at, fp.xMax)))
                    ? plan.exterior_wall_thickness : plan.interior_wall_thickness;

            // (alongX, line coordinate, span start, span end, inward normal)
            var edges = new[]
            {
                (true, r.yMin, r.xMin, r.xMax, Vector3.forward, "S"),
                (true, r.yMax, r.xMin, r.xMax, Vector3.back, "N"),
                (false, r.xMin, r.yMin, r.yMax, Vector3.right, "W"),
                (false, r.xMax, r.yMin, r.yMax, Vector3.left, "E"),
            };
            foreach (var (alongX, at, a0, b0, n, side) in edges)
            {
                float t = T(alongX, at);
                // Shorten by the half thickness of the perpendicular walls at each end.
                float a = a0 + T(!alongX, a0) / 2, b = b0 - T(!alongX, b0) / 2;
                float plane = at + (alongX ? n.z : n.x) * (t / 2 + SkinOffset);
                var cuts = DoorCuts(plan, alongX, at, a0, b0);
                int k = 0;
                foreach (var (sa, sb) in Subtract((a, b), cuts))
                    Skin(parent, $"Finish_{room.id}_{side}_{k++}", alongX, plane, sa, sb, 0f, H, n, mat);
                foreach (var (sa, sb) in cuts)
                    Skin(parent, $"Finish_{room.id}_{side}_{k++}", alongX, plane, sa, sb, D, H, n, mat);
            }
        }

        static void Skin(GameObject parent, string name, bool alongX, float plane, float a, float b, float y0, float y1, Vector3 n, Material mat)
        {
            if (b - a < 0.01f) return;
            float mid = (a + b) / 2;
            Vector3 c = alongX ? new Vector3(mid, (y0 + y1) / 2, plane) : new Vector3(plane, (y0 + y1) / 2, mid);
            AddQuad(parent, name, c, n, b - a, y1 - y0, mat);
        }

        // ------------------------------------------------------------------ doors

        static void BuildDoor(FloorPlan plan, Rect fp, FloorPlan.Door d, GameObject parent, Material trim)
        {
            var go = Child(parent, $"Door_{d.id}");
            int propsLayer = LayerMask.NameToLayer(PropsLayer);
            bool alongX = d.RunsAlongX;
            float at = alongX ? d.at[1] : d.at[0];
            float pos = alongX ? d.at[0] : d.at[1];
            var line = WallLines(plan, fp).First(l => l.alongX == alongX && Near(l.at, at) && pos >= l.a - Eps && pos <= l.b + Eps);
            float depth = line.thickness + 0.03f, D = plan.door_height, w = d.width;

            Vector3 P(float along, float y, float across = 0f) =>
                alongX ? new Vector3(along, y, line.centre + across) : new Vector3(line.centre + across, y, along);
            Vector3 S(float along, float y, float across) =>
                alongX ? new Vector3(along, y, across) : new Vector3(across, y, along);

            AddBox(go, "Jamb_A", P(pos - w / 2 - CasingWidth / 2, D / 2), S(CasingWidth, D, depth), trim, collider: true);
            AddBox(go, "Jamb_B", P(pos + w / 2 + CasingWidth / 2, D / 2), S(CasingWidth, D, depth), trim, collider: true);
            AddBox(go, "Head", P(pos, D + CasingWidth / 2), S(w + 2 * CasingWidth, CasingWidth, depth), trim, collider: false);

            // "open" means fully open: swung back nearly flat against the wall, the way people leave a door so
            // it does not stick into the room. "half" stands at 45 degrees into the room.
            float angle = d.state switch { "open" => 175f, "half" => 45f, "closed" => 0f, _ => -1f };
            if (angle < 0f) { SetLayer(go, propsLayer); return; }  // removed: an open cased opening

            // Swing into the first room that is not outside.
            string swingRoom = d.between[0] == "outside" ? d.between[1] : d.between[0];
            var room = plan.FindRoom(swingRoom);
            float roomCentre = alongX ? room.Rect.center.y : room.Rect.center.x;
            float side = Mathf.Sign(roomCentre - at);

            var hinge = new GameObject("Hinge");
            hinge.transform.SetParent(go.transform, false);
            // Hinge just outside the wall face, so a fully open leaf lies in front of the wall, not inside it.
            hinge.transform.position = P(pos - w / 2 + 0.01f, 0f, side * (line.thickness / 2 + LeafThickness / 2 + 0.005f));
            // Closed, the leaf extends along +x (or +z). Euler(0, +90, 0) turns +x to -z, and +z to +x.
            float yaw = alongX ? -side * angle : side * angle;
            hinge.transform.rotation = Quaternion.Euler(0f, yaw, 0f);
            float leafLen = w - 0.02f;
            var leaf = AddBox(hinge, $"Leaf_{d.id}", Vector3.zero, Vector3.one, trim, collider: true);
            leaf.transform.localPosition = alongX ? new Vector3(leafLen / 2, (D - 0.02f) / 2, 0) : new Vector3(0, (D - 0.02f) / 2, leafLen / 2);
            leaf.GetComponent<MeshFilter>().sharedMesh = WorldMesh.Box(leaf.transform.position,
                alongX ? new Vector3(leafLen, D - 0.02f, LeafThickness) : new Vector3(LeafThickness, D - 0.02f, leafLen), $"Leaf_{d.id}");
            var col = leaf.GetComponent<BoxCollider>();
            col.center = Vector3.zero;
            col.size = leaf.GetComponent<MeshFilter>().sharedMesh.bounds.size;
            SetLayer(go, propsLayer);
        }

        static void SetLayer(GameObject go, int layer)
        {
            foreach (var t in go.GetComponentsInChildren<Transform>()) t.gameObject.layer = layer;
        }

        // ------------------------------------------------------------------ helpers

        static GameObject Child(GameObject parent, string name)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent.transform, false);
            return go;
        }

        static GameObject AddBox(GameObject parent, string name, Vector3 center, Vector3 size, Material mat, bool collider)
        {
            var go = Child(parent, name);
            go.transform.position = center;
            go.AddComponent<MeshFilter>().sharedMesh = WorldMesh.Box(center, size, name);
            go.AddComponent<MeshRenderer>().sharedMaterial = mat;
            if (collider) go.AddComponent<BoxCollider>().size = size;
            return go;
        }

        static GameObject AddQuad(GameObject parent, string name, Vector3 center, Vector3 normal, float width, float height, Material mat)
        {
            var go = Child(parent, name);
            go.transform.position = center;
            go.AddComponent<MeshFilter>().sharedMesh = WorldMesh.Quad(center, normal, width, height, name);
            go.AddComponent<MeshRenderer>().sharedMaterial = mat;
            return go;
        }

        static int RequireLayer(string name)
        {
            int layer = LayerMask.NameToLayer(name);
            if (layer < 0) throw new InvalidOperationException($"Layer '{name}' is missing. Run Tools > Smoke Rescue > Setup Layers.");
            return layer;
        }

        static bool Near(float a, float b) => Mathf.Abs(a - b) < 1e-3f;

        public static List<(float, float)> Union(List<(float, float)> spans)
        {
            var sorted = spans.OrderBy(s => s.Item1).ToList();
            var outList = new List<(float, float)>();
            foreach (var s in sorted)
            {
                if (outList.Count > 0 && s.Item1 <= outList[^1].Item2 + Eps)
                    outList[^1] = (outList[^1].Item1, Mathf.Max(outList[^1].Item2, s.Item2));
                else outList.Add(s);
            }
            return outList;
        }

        public static List<(float, float)> Subtract((float, float) span, List<(float, float)> cuts)
        {
            var result = new List<(float, float)>();
            float cur = span.Item1;
            foreach (var (a, b) in cuts.OrderBy(c => c.Item1))
            {
                if (a > cur + Eps) result.Add((cur, Mathf.Min(a, span.Item2)));
                cur = Mathf.Max(cur, b);
            }
            if (span.Item2 > cur + Eps) result.Add((cur, span.Item2));
            return result;
        }
    }
}
