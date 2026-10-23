"""Refresh portable furniture envelopes from the local model cache, not Unity.

Reads glTF metadata and node transforms. POSITION accessor bounds enclose each
mesh, including unseen faces. No image or model payload is printed or exported.
Run with the brain environment, which supplies numpy.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import struct

import numpy as np


def mesh_bounds(path):
    with path.open("rb") as f:
        magic, version, _ = struct.unpack("<III", f.read(12))
        if magic != 0x46546C67 or version != 2:
            raise ValueError(f"Not glTF 2.0: {path}")
        size, kind = struct.unpack("<II", f.read(8))
        if kind != 0x4E4F534A:
            raise ValueError("Expected JSON chunk")
        doc = json.loads(f.read(size))
    points = []
    def walk(index, parent):
        node = doc["nodes"][index]
        if "matrix" in node:
            m = np.array(node["matrix"]).reshape((4,4),order="F")
        else:
            x,y,z,w = node.get("rotation", [0,0,0,1])
            rot = np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                            [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                            [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
            m = np.eye(4);m[:3,:3] = rot @ np.diag(node.get("scale",[1,1,1]));m[:3,3]=node.get("translation",[0,0,0])
        world = parent @ m
        if "mesh" in node:
            for primitive in doc["meshes"][node["mesh"]]["primitives"]:
                a = doc["accessors"][primitive["attributes"]["POSITION"]]
                for p in itertools.product(*zip(a["min"],a["max"])):
                    points.append((world @ [*p,1])[:3])
        for child in node.get("children",[]):walk(child,world)
    for root in doc["scenes"][doc.get("scene",0)]["nodes"]:walk(root,np.eye(4))
    return np.min(points,axis=0),np.max(points,axis=0)


def refresh(cache, repo):
    import random_layout as g
    props=json.loads((repo/"config/props.json").read_text())
    fire=json.loads((repo/"config/fire.json").read_text())
    out={"note":"Conservative centred horizontal envelopes. Union of transformed glTF accessor bounds and fire fuel estimates, plus 0.10 m on each side. fit_width and part offsets match FurnitureBuilder. No runtime mesh measurement is claimed.","kinds":{}}
    for kind in sorted({k for kinds in g.FURNITURE.values() for k in kinds}):
        hx,hz=np.array(fire["furniture"][kind]["size_m"])/2
        assets=[]
        for part in props["landmarks"][kind]:
            rel=part.get("downloaded") or "models/"+part["polyhaven"]+".glb"
            path=cache/rel;lo,hi=mesh_bounds(path)
            # glTFast flips x on import. Centred envelopes make handedness immaterial.
            hx0,hz0=max(abs(lo[0]),abs(hi[0])),max(abs(lo[2]),abs(hi[2]))
            if "fit_width" in part:
                factor=part["fit_width"]/max(hi[0]-lo[0],hi[2]-lo[2]);hx0*=factor;hz0*=factor
            angle=np.deg2rad(part.get("yaw",0));c,s=abs(np.cos(angle)),abs(np.sin(angle))
            ox,_,oz=part.get("offset",[0,0,0])
            hx=max(hx,abs(ox)+c*hx0+s*hz0);hz=max(hz,abs(oz)+s*hx0+c*hz0)
            assets.append(dict(path=rel,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),min_m=lo.tolist(),max_m=hi.tolist()))
        out["kinds"][kind]=dict(conservative_size_m=[round(float(hx)*2+.2,4),round(float(hz)*2+.2,4)],assets=assets)
    (repo/"tools/random_layout_assets.json").write_text(json.dumps(out,indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("cache",type=Path);args=p.parse_args()
    refresh(args.cache,Path(__file__).resolve().parents[1])
