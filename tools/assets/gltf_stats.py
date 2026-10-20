"""Parse a .glb/.gltf (JSON + accessor metadata only) and report triangle count and bounding box in the file's units (glTF = metres).

    python3 -I gltf_stats.py model.glb     -> prints JSON
Bounding box is the AABB of the transformed accessor min/max corners (exact for axis-aligned parts, slightly generous for rotated ones).
Run only with python3 -I; it never imports or executes anything from the model file.
"""
import json, struct, sys, os

def load(path):
    if path.endswith(".glb"):
        with open(path, "rb") as f:
            magic, ver, ln = struct.unpack("<4sII", f.read(12))
            assert magic == b"glTF", "not a GLB"
            clen, ctype = struct.unpack("<II", f.read(8))
            assert ctype == 0x4E4F534A
            return json.loads(f.read(clen))
    return json.load(open(path))

def mm(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(4)) for j in range(4)] for i in range(4)]

def node_matrix(n):
    if "matrix" in n:
        m = n["matrix"]  # column-major
        return [[m[c * 4 + r] for c in range(4)] for r in range(4)]
    t = n.get("translation", [0, 0, 0]); q = n.get("rotation", [0, 0, 0, 1]); s = n.get("scale", [1, 1, 1])
    x, y, z, w = q
    R = [[1 - 2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
         [2*(x*y + z*w), 1 - 2*(x*x+z*z), 2*(y*z - x*w)],
         [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x+y*y)]]
    return [[R[0][0]*s[0], R[0][1]*s[1], R[0][2]*s[2], t[0]], [R[1][0]*s[0], R[1][1]*s[1], R[1][2]*s[2], t[1]],
            [R[2][0]*s[0], R[2][1]*s[1], R[2][2]*s[2], t[2]], [0, 0, 0, 1]]

def stats(path):
    g = load(path)
    acc = g["accessors"]; meshes = g.get("meshes", [])
    def mesh_tris(m):
        t = 0
        for p in m["primitives"]:
            mode = p.get("mode", 4)
            if mode not in (4, 5, 6):
                continue
            cnt = acc[p["indices"]]["count"] if "indices" in p else acc[p["attributes"]["POSITION"]]["count"]
            t += cnt // 3 if mode == 4 else max(cnt - 2, 0)
        return t
    pts = []; tris = 0; nprims = 0
    def visit(i, parent):
        nonlocal tris, nprims
        n = g["nodes"][i]; M = mm(parent, node_matrix(n))
        if "mesh" in n:
            m = meshes[n["mesh"]]; tris += mesh_tris(m); nprims += len(m["primitives"])
            for p in m["primitives"]:
                a = acc[p["attributes"]["POSITION"]]
                if "min" in a and "max" in a:
                    lo, hi = a["min"], a["max"]
                    for cx in (lo[0], hi[0]):
                        for cy in (lo[1], hi[1]):
                            for cz in (lo[2], hi[2]):
                                pts.append([sum(M[r][k] * v for k, v in enumerate((cx, cy, cz, 1.0))) for r in range(3)])
        for c in n.get("children", []):
            visit(c, M)
    scene = g["scenes"][g.get("scene", 0)] if g.get("scenes") else {"nodes": list(range(len(g["nodes"])))}
    for r in scene["nodes"]:
        visit(r, [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    size = [max(p[i] for p in pts) - min(p[i] for p in pts) for i in range(3)] if pts else [0, 0, 0]
    imgs = g.get("images", [])
    return dict(tris=int(tris), bbox_m=[round(v, 4) for v in size], prims=nprims, materials=len(g.get("materials", [])), images=len(imgs),
                gen=g.get("asset", {}).get("generator", "")[:60], file_mb=round(os.path.getsize(path) / 1e6, 2))

if __name__ == "__main__":
    print(json.dumps(stats(sys.argv[1])))
