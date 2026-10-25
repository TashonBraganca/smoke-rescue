using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Rendering.HighDefinition;

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
            BuildWindows(plan, fp, Child(root, "Windows"), materials(TrimMaterial), GlassMaterial());

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
                    if (plan.footprint_rects==null) {
                        bool exterior=alongX ? Near(at,fp.yMin)||Near(at,fp.yMax) : Near(at,fp.xMin)||Near(at,fp.xMax);
                        float t=exterior ? plan.exterior_wall_thickness : plan.interior_wall_thickness;
                        float low=alongX ? fp.yMin : fp.xMin;
                        lines.Add(new WallLine { alongX=alongX,at=at,a=a,b=b,thickness=t,
                            centre=exterior ? at+(Near(at,low) ? t/2 : -t/2) : at });
                        continue;
                    }
                    var points = new SortedSet<float> { a,b };
                    foreach (var room in plan.rooms)
                            foreach (float p in alongX ? new[] { room.rect[0],room.rect[2] } : new[] { room.rect[1],room.rect[3] })
                                if (p>a && p<b) points.Add(p);
                    var stops=points.ToArray();
                    for (int i=0;i<stops.Length-1;i++) {
                        float start=stops[i],end=stops[i+1];
                        Vector3 inward=ExteriorInward(plan,alongX,at,(start+end)/2);
                        bool exterior=inward.sqrMagnitude>0;
                        float t=exterior ? plan.exterior_wall_thickness : plan.interior_wall_thickness;
                        float centre=at+(alongX ? inward.z : inward.x)*t/2;
                        if (lines.Count>0 && lines[^1].alongX==alongX && Near(lines[^1].at,at) && Near(lines[^1].b,start)
                            && Near(lines[^1].thickness,t) && Near(lines[^1].centre,centre)) {
                            var previous=lines[^1];previous.b=end;lines[^1]=previous;
                        } else lines.Add(new WallLine { alongX=alongX,at=at,a=start,b=end,thickness=t,centre=centre });
                    }
                }
            }
            return lines;
        }

        // A facade has occupied floor on exactly one side, including a concave corner.
        public static Vector3 ExteriorInward(FloorPlan plan,bool alongX,float at,float pos)
        {
            bool Side(float across) => plan.rooms.Any(r => alongX
                ? pos>=r.rect[0] && pos<=r.rect[2] && across>r.rect[1] && across<r.rect[3]
                : across>r.rect[0] && across<r.rect[2] && pos>=r.rect[1] && pos<=r.rect[3]);
            bool low=Side(at-.01f), high=Side(at+.01f);
            return low==high ? Vector3.zero : (alongX ? Vector3.forward : Vector3.right)*(high ? 1 : -1);
        }

        static void BuildWalls(FloorPlan plan, Rect fp, GameObject parent, Material mat, int layer)
        {
            float H = plan.wall_height, D = plan.door_height;
            foreach (var line in WallLines(plan, fp))
            {
                var cuts = DoorCuts(plan, line.alongX, line.at, line.a, line.b);
                var wins = WindowCuts(plan, fp, line.alongX, line.at, line.a, line.b);
                string id = $"{(line.alongX ? "Z" : "X")}{line.at:0.00}";
                int k = 0;
                foreach (var (a, b) in Subtract((line.a, line.b), cuts.Concat(wins).OrderBy(c => c.Item1).ToList()))
                    WallBox(parent, $"Wall_{id}_{k++}", line, a, b, 0f, H, mat, layer);
                foreach (var (a, b) in cuts)
                    WallBox(parent, $"Lintel_{id}_{k++}", line, a, b, D, H, mat, layer);
                foreach (var (a, b) in wins)
                {
                    WallBox(parent, $"Sill_{id}_{k++}", line, a, b, 0f, WindowSill, mat, layer);
                    WallBox(parent, $"WindowHead_{id}_{k++}", line, a, b, WindowSill + WindowHeight, H, mat, layer);
                }
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
                bool exterior=ExteriorInward(plan,alongX,at,(a0+b0)/2).sqrMagnitude>0;
                float t=exterior ? 2*plan.exterior_wall_thickness : plan.interior_wall_thickness;
                // Shorten by the half thickness of the perpendicular walls at each end.
                float a = a0 + T(!alongX, a0) / 2, b = b0 - T(!alongX, b0) / 2;
                if (plan.footprint_rects==null) t=T(alongX,at);
                float plane = at + (alongX ? n.z : n.x) * (t / 2 + SkinOffset);
                var cuts = DoorCuts(plan, alongX, at, a0, b0);
                var wins = WindowCuts(plan, fp, alongX, at, a0, b0);
                int k = 0;
                foreach (var (sa, sb) in Subtract((a, b), cuts.Concat(wins).OrderBy(c => c.Item1).ToList()))
                    Skin(parent, $"Finish_{room.id}_{side}_{k++}", alongX, plane, sa, sb, 0f, H, n, mat);
                foreach (var (sa, sb) in cuts)
                    Skin(parent, $"Finish_{room.id}_{side}_{k++}", alongX, plane, sa, sb, D, H, n, mat);
                foreach (var (sa, sb) in wins)
                {
                    Skin(parent, $"Finish_{room.id}_{side}_{k++}", alongX, plane, sa, sb, 0f, WindowSill, n, mat);
                    Skin(parent, $"Finish_{room.id}_{side}_{k++}", alongX, plane, sa, sb, WindowSill + WindowHeight, H, n, mat);
                }
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

        // ------------------------------------------------------------------ windows

        public const float WindowWidth = 1.2f, WindowHeight = 1.3f, WindowSill = 0.85f;
        /// <summary>Rooms where the fire starts: their windows have burst, the way fire vents through windows.</summary>
        static readonly HashSet<string> BrokenWindowRooms = new HashSet<string> { "living", "kitchen" };

        public struct WindowSpan { public bool alongX; public float at, pos; public Vector3 inward; public string room; }

        /// <summary>
        /// One real window opening per exterior wall of each room longer than 2.2 m, at the first clear spot from the
        /// centre: at least 0.8 m from any door and 1.2 m from any landmark. Walls and wall finishes leave the opening
        /// open (sill below, head above), so robots and cameras see through it to the night outside.
        /// </summary>
        public static List<WindowSpan> WindowSpans(FloorPlan plan, Rect fp)
        {
            var list = new List<WindowSpan>();
            const float w = WindowWidth;
            foreach (var room in plan.rooms)
            {
                Rect r = room.Rect;
                var edges = new[]
                {
                    (true, r.yMin, r.xMin, r.xMax, Vector3.forward), (true, r.yMax, r.xMin, r.xMax, Vector3.back),
                    (false, r.xMin, r.yMin, r.yMax, Vector3.right), (false, r.xMax, r.yMin, r.yMax, Vector3.left),
                };
                foreach (var (alongX, at, a, b, n) in edges)
                {
                    bool exterior = alongX ? (Near(at, fp.yMin) || Near(at, fp.yMax)) : (Near(at, fp.xMin) || Near(at, fp.xMax));
                    if (plan.footprint_rects != null)
                        exterior=ExteriorInward(plan,alongX,at,(a+b)/2).sqrMagnitude>0;
                    if (!exterior || b - a < 2.2f) continue;
                    float mid = (a + b) / 2;
                    foreach (var pos in new[] { mid, mid - 1.2f, mid + 1.2f, mid - 2.2f, mid + 2.2f })
                    {
                        if (pos - w / 2 < a + 0.4f || pos + w / 2 > b - 0.4f) continue;
                        Vector2 p2 = alongX ? new Vector2(pos, at) : new Vector2(at, pos);
                        if (plan.doors.Any(d => Vector2.Distance(new Vector2(d.at[0], d.at[1]), p2) < w / 2 + d.width / 2 + 0.8f)) continue;
                        if (plan.landmarks.Any(l => Vector2.Distance(new Vector2(l.at[0], l.at[1]), p2) < 1.2f)) continue;
                        list.Add(new WindowSpan { alongX = alongX, at = at, pos = pos, inward = n, room = room.id });
                        break;
                    }
                }
            }
            return list;
        }

        static List<(float, float)> WindowCuts(FloorPlan plan, Rect fp, bool alongX, float at, float a, float b)
        {
            var cuts = new List<(float, float)>();
            foreach (var win in WindowSpans(plan, fp))
                if (win.alongX == alongX && Near(win.at, at) && win.pos >= a - Eps && win.pos <= b + Eps)
                    cuts.Add((win.pos - WindowWidth / 2, win.pos + WindowWidth / 2));
            cuts.Sort();
            return cuts;
        }

        /// <summary>Frame, mullion and glass in each opening. Windows of the rooms where the fire starts are broken:
        /// no pane, a few jagged shards left in the frame, soot streaks above the opening outside.</summary>
        static void BuildWindows(FloorPlan plan, Rect fp, GameObject parent, Material trim, Material glass)
        {
            const float w = WindowWidth, h = WindowHeight, sill = WindowSill, frame = 0.06f;
            int props = LayerMask.NameToLayer(PropsLayer);
            foreach (var span in WindowSpans(plan, fp))
            {
                var line = WallLines(plan, fp).First(l => l.alongX == span.alongX && Near(l.at, span.at) && span.pos >= l.a - Eps && span.pos <= l.b + Eps);
                bool alongX = span.alongX;
                float pos = span.pos, depth = line.thickness + 0.02f;
                var win = Child(parent, $"Window_{span.room}_{(alongX ? "Z" : "X")}{span.at:0}");
                Vector3 C(float along, float y, float off = 0f) => alongX ? new Vector3(along, y, line.centre + off) : new Vector3(line.centre + off, y, along);
                Vector3 Sz(float along, float y, float d) => alongX ? new Vector3(along, y, d) : new Vector3(d, y, along);
                AddBox(win, "FrameTop", C(pos, sill + h - frame / 2), Sz(w, frame, depth), trim, collider: false);
                AddBox(win, "FrameSill", C(pos, sill + frame / 2), Sz(w + 0.08f, frame, depth + 0.06f), trim, collider: true);
                AddBox(win, "FrameL", C(pos - w / 2 + frame / 2, sill + h / 2), Sz(frame, h, depth), trim, collider: false);
                AddBox(win, "FrameR", C(pos + w / 2 - frame / 2, sill + h / 2), Sz(frame, h, depth), trim, collider: false);
                AddBox(win, "Mullion", C(pos, sill + h / 2), Sz(0.04f, h - 2 * frame, 0.05f), trim, collider: false);
                AddBox(win, "Transom", C(pos, sill + h * 0.62f), Sz(w - 2 * frame, 0.04f, 0.05f), trim, collider: false);
                if (!BrokenWindowRooms.Contains(span.room))
                {
                    foreach (var side in new[] { -1f, 1f })
                        AddQuad(win, "Glass", C(pos, sill + h / 2, side * 0.004f), side * (alongX ? Vector3.forward : Vector3.right), w - 2 * frame, h - 2 * frame, glass);
                }
                else
                {
                    // Shards: small jagged triangles of glass stuck in two corners.
                    var rng = new System.Random(span.room.GetHashCode());
                    for (int k = 0; k < 3; k++)
                    {
                        float u = k == 0 ? -w / 2 + frame : k == 1 ? w / 2 - frame : -w / 2 + frame;
                        float v = k == 2 ? sill + h - frame : sill + frame;
                        float du = (k == 1 ? -1f : 1f) * (0.12f + 0.1f * (float)rng.NextDouble());
                        float dv = (k == 2 ? -1f : 1f) * (0.15f + 0.15f * (float)rng.NextDouble());
                        var shard = new GameObject("Shard");
                        shard.transform.SetParent(win.transform, false);
                        var mesh = new Mesh { name = "Shard" };
                        mesh.vertices = new[] { C(pos + u, v), C(pos + u + du, v), C(pos + u, v + dv) };
                        mesh.triangles = new[] { 0, 1, 2, 0, 2, 1 };
                        mesh.RecalculateNormals();
                        shard.AddComponent<MeshFilter>().sharedMesh = mesh;
                        shard.AddComponent<MeshRenderer>().sharedMaterial = glass;
                    }
                }
                foreach (var t in win.GetComponentsInChildren<Transform>()) t.gameObject.layer = props;
            }
        }

        static Material glassMat;
        static Material GlassMaterial()
        {
            if (glassMat != null) return glassMat;
            glassMat = new Material(Shader.Find("HDRP/Lit")) { name = "WindowGlass" };
            glassMat.SetFloat("_SurfaceType", 1f);          // transparent
            glassMat.SetFloat("_BlendMode", 0f);            // alpha
            glassMat.SetColor("_BaseColor", new Color(0.55f, 0.6f, 0.62f, 0.18f));
            glassMat.SetFloat("_Smoothness", 0.96f);
            glassMat.SetFloat("_Metallic", 0f);
            glassMat.SetFloat("_DoubleSidedEnable", 1f);
            HDMaterial.ValidateMaterial(glassMat);
            return glassMat;
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
