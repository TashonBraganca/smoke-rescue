"""Send one camera frame to the Smoke Rescue API and print the people found. Standard library only.

    python robot_client.py frame.png [depth.png] [--url http://127.0.0.1:8095]
A ROS node does the same in its image callback: cv2.imencode('.png', bgr)[1].tobytes() in place of the file read.
"""
import argparse, base64, json, os, urllib.request

p = argparse.ArgumentParser()
p.add_argument("rgb", help="colour PNG or JPEG")
p.add_argument("depth", nargs="?", help="optional 16-bit depth PNG, millimetres, same size as rgb")
p.add_argument("--url", default=os.environ.get("SMOKE_API_URL", "http://127.0.0.1:8095"))
p.add_argument("--min-conf", type=float, default=0.25)
a = p.parse_args()

b64 = lambda path: base64.b64encode(open(path, "rb").read()).decode("ascii")
body = {"rgb_b64": b64(a.rgb), "depth_scale": 1000, "min_conf": a.min_conf}
if a.depth:
    body["depth_b64"] = b64(a.depth)
req = urllib.request.Request(a.url.rstrip("/") + "/perceive/people", data=json.dumps(body).encode(),
                             headers={"content-type": "application/json"}, method="POST")
with urllib.request.urlopen(req, timeout=75) as r:
    out = json.load(r)
print(f"{len(out['people'])} people ({out.get('ms')} ms, OpenCV {out.get('opencv')})")
for person in out["people"]:
    rng = "unknown" if person["range_m"] is None else f"{person['range_m']:.2f} m"
    print(f"  conf {person['confidence']:.2f}  bearing {person['bearing_deg']:+.1f} deg  range {rng}  "
          f"box {person['bbox_rgb']}")
