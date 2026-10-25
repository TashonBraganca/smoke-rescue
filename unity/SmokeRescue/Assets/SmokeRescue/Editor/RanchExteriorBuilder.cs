using System;
using System.Collections.Generic;
using System.Linq;
using SmokeRescue.World;
using UnityEngine;

namespace SmokeRescue.Editor
{
    /// <summary>Exterior generated from the footprint, openings and entrance. Decorative parts have no colliders.
    /// Ground-level parts stay outside the footprint. Roof parts sit above the ceiling on its cutaway layer.</summary>
    public static class RanchExteriorBuilder
    {
        // Presentation dimensions in metres, selected for a single-storey ranch.
        // A 0.45 m eave protects the siding; 22 degrees keeps the roof low.
        const float Overhang = .45f, Pitch = 22f, SidingCourse = .18f;
        const float CladdingDepth = .025f, FasciaHeight = .18f;
        const float YardMargin = 5.5f, EntranceClearance = 2.5f;

        public static void Build(FloorPlan plan, GameObject house)
        {
            var root = new GameObject("RanchExterior");
            root.transform.SetParent(house.transform, false);
            Rect fp = Rect.MinMaxRect(plan.footprint[0], plan.footprint[1], plan.footprint[2], plan.footprint[3]);
            var siding = MaterialLibrary.Get("brick_painted_white");
            var trim = MaterialLibrary.Get("paint_white_textured");
            var roof = MaterialLibrary.Get("concrete_rough");
            var paving = MaterialLibrary.Get("concrete_sealed");
            var foliage = MaterialLibrary.Get("wallpaper_deep_green");
            var timber = MaterialLibrary.Get("prop_wood_cabinet");
            int ceiling = LayerMask.NameToLayer(HouseBuilder.CeilingLayer);
            // Replace only the builder's flat roof. Per-room ceilings remain intact.
            var flat = house.transform.Find("Ceiling/Roof");
            if (flat != null) UnityEngine.Object.DestroyImmediate(flat.gameObject);
            float run = fp.height / 2 + Overhang;
            float rise = run * Mathf.Tan(Pitch * Mathf.Deg2Rad);
            float eaveY = plan.wall_height + .10f;
            foreach (float side in new[] { -1f, 1f })
            {
                var slope = Box(root, "RoofSlope", new Vector3(fp.center.x, eaveY + rise / 2, fp.center.y + side * run / 2),
                    new Vector3(fp.width + 2 * Overhang, .12f, run / Mathf.Cos(Pitch * Mathf.Deg2Rad)), roof, ceiling);
                slope.transform.rotation = Quaternion.Euler(side * Pitch, 0, 0);
                Box(root, "EaveFascia", new Vector3(fp.center.x, eaveY - FasciaHeight / 2, fp.center.y + side * run),
                    new Vector3(fp.width + 2 * Overhang, FasciaHeight, .08f), trim, ceiling);
            }
            foreach (float x in new[] { fp.xMin - Overhang, fp.xMax + Overhang })
            {
                Gable(root, x, fp.center.y, run, eaveY, rise, siding, ceiling);
                foreach (float side in new[] { -1f, 1f })
                {
                    var rake = Box(root, "RakeFascia", new Vector3(x, eaveY + rise / 2, fp.center.y + side * run / 2),
                        new Vector3(.08f, FasciaHeight, run / Mathf.Cos(Pitch * Mathf.Deg2Rad)), trim, ceiling);
                    rake.transform.rotation = Quaternion.Euler(side * Pitch, 0, 0);
                }
            }
            Box(root, "RidgeCap", new Vector3(fp.center.x, eaveY + rise + .06f, fp.center.y),
                new Vector3(fp.width + 2 * Overhang, .1f, .16f), roof, ceiling);
            Cladding(plan, fp, root, siding);
            // Existing HouseBuilder windows already have frames, mullions and glass.
            // Extend the sill outside the cladding so the windows read on camera.
            foreach (var win in HouseBuilder.WindowSpans(plan, fp))
            {
                Vector3 outward = -win.inward;
                var p = win.alongX ? new Vector3(win.pos, HouseBuilder.WindowSill, win.at) : new Vector3(win.at, HouseBuilder.WindowSill, win.pos);
                Box(root, "ExteriorWindowSill", p + outward * .11f,
                    win.alongX ? new Vector3(1.38f, .06f, .20f) : new Vector3(.20f, .06f, 1.38f), trim);
            }
            // Four strips leave the interior clear. Grass sits above the original
            // ground at -0.002 m. Paving is 5-6 mm above grass to prevent z-fighting.
            // The original ground supports robots; these finishes have no colliders.
            Slab(root, "SouthLawn", fp.xMin-YardMargin, fp.yMin-YardMargin, fp.xMax+YardMargin, fp.yMin, foliage, 0f);
            Slab(root, "NorthLawn", fp.xMin-YardMargin, fp.yMax, fp.xMax+YardMargin, fp.yMax+YardMargin, foliage, 0f);
            Slab(root, "WestLawn", fp.xMin-YardMargin, fp.yMin, fp.xMin, fp.yMax, foliage, 0f);
            Slab(root, "EastLawn", fp.xMax, fp.yMin, fp.xMax+YardMargin, fp.yMax, foliage, 0f);
            var entrance = plan.doors.Single(d => d.between.Contains("outside"));
            var at = new Vector2(entrance.at[0], entrance.at[1]);
            Vector2 inward = entrance.RunsAlongX ? new Vector2(0, Mathf.Abs(at.y-fp.yMin)<.01f ? 1 : -1) : new Vector2(Mathf.Abs(at.x-fp.xMin)<.01f ? 1 : -1, 0);
            Vector2 outwards = -inward, right = new Vector2(inward.y, -inward.x);
            Vector3 Point(float side, float distance, float y) { var p = at + right * side + outwards * distance; return new Vector3(p.x, y, p.y); }
            Vector3 Size(float width, float height, float depth) => entrance.RunsAlongX ? new Vector3(width,height,depth) : new Vector3(depth,height,width);
            Box(root, "FlushPorchSlab", Point(0,.85f,-.024f), Size(3.6f,.06f,1.7f), paving);
            Box(root, "Driveway", Point(1.3f,3.55f,-.020f), Size(3.8f,.05f,3.7f), paving);
            // Porch posts are 1.65 m either side of the gate, clear of both pads.
            foreach (float side in new[] { -1.65f, 1.65f })
                Box(root, "PorchPost", Point(side,1.45f,1.17f), new Vector3(.12f,2.34f,.12f), trim);
            Box(root, "PorchCanopy", Point(0,.9f,2.4f), Size(3.8f,.12f,1.8f), roof);
            Box(root, "MailboxPost", Point(-2.4f,4.9f,.55f), new Vector3(.10f,1.1f,.10f), timber);
            Box(root, "Mailbox", Point(-2.4f,4.9f,1.15f), Size(.35f,.30f,.55f), MaterialLibrary.Get("prop_steel_brushed"));
            // Every planting is outside the footprint and away from the entry corridor.
            var plantings = new[] { new Vector2(fp.xMin-2,fp.yMin-2), new Vector2(fp.xMax+2,fp.yMin-2),
                new Vector2(fp.xMin-2,fp.yMax+2), new Vector2(fp.xMax+2,fp.yMax+2) };
            foreach (var p in plantings)
            {
                if (Vector2.Distance(p,at)<EntranceClearance) continue;
                Primitive(root, "Shrub", PrimitiveType.Sphere, new Vector3(p.x,.45f,p.y), new Vector3(1.1f,.9f,1.1f), foliage);
            }
            var tree = new Vector2(fp.xMax+3.5f,fp.yMax+3.5f);
            Primitive(root, "TreeTrunk", PrimitiveType.Cylinder, new Vector3(tree.x,1.4f,tree.y), new Vector3(.24f,1.4f,.24f), timber);
            Primitive(root, "TreeCrown", PrimitiveType.Sphere, new Vector3(tree.x,3.1f,tree.y), new Vector3(2.4f,2.8f,2.4f), foliage);
        }

        static void Cladding(FloorPlan plan, Rect fp, GameObject root, Material mat)
        {
            var windows = HouseBuilder.WindowSpans(plan, fp);
            foreach (var line in HouseBuilder.WallLines(plan, fp))
            {
                bool low = Mathf.Abs(line.at - (line.alongX ? fp.yMin : fp.xMin)) < .001f;
                bool high = Mathf.Abs(line.at - (line.alongX ? fp.yMax : fp.xMax)) < .001f;
                if (!low && !high) continue;
                float plane = line.at + (low ? -1 : 1) * CladdingDepth / 2;
                for (float y = 0; y < plan.wall_height; y += SidingCourse)
                {
                    float top = Mathf.Min(y + SidingCourse, plan.wall_height);
                    var cuts = new List<(float,float)>();
                    // Cut a whole course when it meets an opening, preserving every door
                    // and window even when its sill falls between two siding courses.
                    foreach (var d in plan.doors)
                        if (d.RunsAlongX == line.alongX && Mathf.Abs((line.alongX ? d.at[1] : d.at[0])-line.at)<.001f && y<plan.door_height)
                        { float pos = line.alongX ? d.at[0] : d.at[1]; cuts.Add((pos-d.width/2,pos+d.width/2)); }
                    foreach (var w in windows)
                        if (w.alongX == line.alongX && Mathf.Abs(w.at-line.at)<.001f && top>HouseBuilder.WindowSill && y<HouseBuilder.WindowSill+HouseBuilder.WindowHeight)
                            cuts.Add((w.pos-HouseBuilder.WindowWidth/2,w.pos+HouseBuilder.WindowWidth/2));
                    foreach (var (a,b) in HouseBuilder.Subtract((line.a,line.b), HouseBuilder.Union(cuts)))
                        Box(root,"SidingCourse",line.alongX ? new Vector3((a+b)/2,(y+top)/2,plane) : new Vector3(plane,(y+top)/2,(a+b)/2),
                            line.alongX ? new Vector3(b-a,top-y-.008f,CladdingDepth) : new Vector3(CladdingDepth,top-y-.008f,b-a),mat);
                }
            }
        }

        static void Gable(GameObject root, float x, float z, float run, float y, float rise, Material mat, int layer)
        {
            var go = new GameObject("Gable"); go.transform.SetParent(root.transform,false); go.layer=layer;
            var mesh = new Mesh { name="RanchGable" };
            mesh.vertices = new[] { new Vector3(x,y,z-run),new Vector3(x,y+rise,z),new Vector3(x,y,z+run),
                new Vector3(x,y,z+run),new Vector3(x,y+rise,z),new Vector3(x,y,z-run) };
            mesh.triangles = new[] { 0,1,2,3,4,5 };
            mesh.uv = new[] { new Vector2(0,0),new Vector2(run,rise),new Vector2(run*2,0),
                new Vector2(run*2,0),new Vector2(run,rise),new Vector2(0,0) };
            mesh.RecalculateNormals(); mesh.RecalculateBounds();
            go.AddComponent<MeshFilter>().sharedMesh=mesh; go.AddComponent<MeshRenderer>().sharedMaterial=mat;
        }

        static void Slab(GameObject root, string name, float x0, float z0, float x1, float z1, Material mat, float top)
            => Box(root,name,new Vector3((x0+x1)/2,top-.01f,(z0+z1)/2),new Vector3(x1-x0,.02f,z1-z0),mat);

        static GameObject Box(GameObject root, string name, Vector3 at, Vector3 size, Material mat, int layer=0)
            => Primitive(root,name,PrimitiveType.Cube,at,size,mat,layer);

        static GameObject Primitive(GameObject root, string name, PrimitiveType kind, Vector3 at, Vector3 size, Material mat, int layer=0)
        {
            var go=GameObject.CreatePrimitive(kind); go.name=name; go.layer=layer;
            go.transform.SetParent(root.transform,false); go.transform.position=at; go.transform.localScale=size;
            UnityEngine.Object.DestroyImmediate(go.GetComponent<Collider>());
            go.GetComponent<MeshRenderer>().sharedMaterial=mat;
            return go;
        }
    }
}
