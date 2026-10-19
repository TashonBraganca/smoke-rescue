using UnityEngine;

namespace SmokeRescue.World
{
    /// <summary>
    /// Axis-aligned boxes and quads whose UVs are world coordinates in metres, so a material with
    /// tiling 1/tile_m shows its texture at real-world size on every surface, whatever the piece size.
    /// </summary>
    public static class WorldMesh
    {
        static readonly Vector3[] Normals =
        {
            Vector3.right, Vector3.left, Vector3.up, Vector3.down, Vector3.forward, Vector3.back,
        };

        /// <summary>A box mesh centred on `center` (world), vertices local to the centre.</summary>
        public static Mesh Box(Vector3 center, Vector3 size, string name)
        {
            var mesh = new Mesh { name = name };
            var verts = new Vector3[24];
            var norms = new Vector3[24];
            var uvs = new Vector2[24];
            var tris = new int[36];
            Vector3 h = size * 0.5f;
            for (int f = 0; f < 6; f++)
            {
                Vector3 n = Normals[f];
                // Two in-plane axes with u x v = n.
                Vector3 u = Mathf.Abs(n.y) > 0.5f ? Vector3.right : Vector3.Cross(Vector3.up, n);
                Vector3 v = Vector3.Cross(n, u);
                for (int i = 0; i < 4; i++)
                {
                    float su = (i == 1 || i == 2) ? 1 : -1;
                    float sv = (i >= 2) ? 1 : -1;
                    Vector3 local = Vector3.Scale(n, h) + Vector3.Scale(u * su, h) + Vector3.Scale(v * sv, h);
                    verts[f * 4 + i] = local;
                    norms[f * 4 + i] = n;
                    uvs[f * 4 + i] = WorldUV(center + local, n);
                }
                int b = f * 4;
                // (u, v) corners run counter-clockwise with u x v = n; in Unity's left-handed space that is
                // clockwise seen from outside, which is Unity's front face.
                tris[f * 6 + 0] = b; tris[f * 6 + 1] = b + 1; tris[f * 6 + 2] = b + 2;
                tris[f * 6 + 3] = b; tris[f * 6 + 4] = b + 2; tris[f * 6 + 5] = b + 3;
            }
            mesh.vertices = verts;
            mesh.normals = norms;
            mesh.uv = uvs;
            mesh.triangles = tris;
            mesh.RecalculateTangents();
            mesh.RecalculateBounds();
            return mesh;
        }

        /// <summary>A single-sided rectangle centred on `center`, facing `normal`, `width` along the in-plane horizontal axis.</summary>
        public static Mesh Quad(Vector3 center, Vector3 normal, float width, float height, string name)
        {
            var mesh = new Mesh { name = name };
            Vector3 u = Mathf.Abs(normal.y) > 0.5f ? Vector3.right : Vector3.Cross(Vector3.up, normal);
            Vector3 v = Vector3.Cross(normal, u);
            float hw = width * 0.5f, hh = height * 0.5f;
            var verts = new[] { -u * hw - v * hh, u * hw - v * hh, u * hw + v * hh, -u * hw + v * hh };
            var uvs = new Vector2[4];
            for (int i = 0; i < 4; i++) uvs[i] = WorldUV(center + verts[i], normal);
            mesh.vertices = verts;
            mesh.normals = new[] { normal, normal, normal, normal };
            mesh.uv = uvs;
            mesh.triangles = new[] { 0, 1, 2, 0, 2, 3 };
            mesh.RecalculateTangents();
            mesh.RecalculateBounds();
            return mesh;
        }

        static Vector2 WorldUV(Vector3 p, Vector3 n)
        {
            if (Mathf.Abs(n.y) > 0.5f) return new Vector2(p.x, p.z);
            if (Mathf.Abs(n.x) > 0.5f) return new Vector2(n.x > 0 ? -p.z : p.z, p.y);
            return new Vector2(n.z > 0 ? p.x : -p.x, p.y);
        }
    }
}
