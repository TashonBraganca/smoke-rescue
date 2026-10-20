"""Parallel direct downloader: python3 -I dl_many.py jobs.json  (list of {"url":..., "out":...}). Skips existing files."""
import json, sys, os, urllib.request
from concurrent.futures import ThreadPoolExecutor
UA = {"User-Agent": "Mozilla/5.0 SmokeRescueSim/0.1 (private research simulation)"}

def get(job):
    out = job["out"]
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return out, "exists"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    try:
        req = urllib.request.Request(job["url"], headers=UA)
        with urllib.request.urlopen(req, timeout=120) as r, open(out + ".part", "wb") as f:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b)
        os.replace(out + ".part", out)
        return out, "ok %d" % os.path.getsize(out)
    except Exception as e:
        return out, "FAIL %s" % e

if __name__ == "__main__":
    jobs = json.load(open(sys.argv[1]))
    with ThreadPoolExecutor(8) as ex:
        for o, s in ex.map(get, jobs):
            if not s.startswith(("ok", "exists")):
                print(o, s)
    print("done", len(jobs))
