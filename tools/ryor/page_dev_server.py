#!/usr/bin/env python3
"""Local dev server for the "Run your own rescue" page (deploy/aws/rescue_page). Python stdlib only.

Serves the page at /rescue/ and fakes the API of tools/ryor/CONTRACT.txt section 4 from a data folder that holds
gallery.json and cache/<setup key>/{status.json,chunk_NNNN.json} (the real sample, or tools/ryor/page_fixture.py output).

    python3 tools/ryor/page_dev_server.py --data data/ryor/sample --port 8765 [--mode normal]

Modes (also switchable while running: GET /__dev/mode?m=<mode>&startup=<s>&speed=<x>):
  normal   live mission: "starting" for --startup s (default 4), then sim time grows at --speed x wall time (default 4)
  slow     like normal with startup 45 s and speed 1 (feels like a cold Fargate start)
  fail     live mission fails after 40 sim s with "fallback": "cache:<key>"
  busy     POST answers {"source":"recorded","reason":"busy"}
  down     POST answers 503 (page must fall back to the cache by itself)
  flaky    half of the GET polls answer 502
  stuck    live mission stays "starting" forever (page must give up and fall back)
  nocache  the 1-robot comparison and cache reads answer 404
Modes combine with commas, e.g. m=busy,nocache (recorded run missing: the page must show its error state).
"""
import argparse
import json
import random
import re
import threading
import time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "deploy" / "aws" / "rescue_page"
MIME = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
        ".json": "application/json", ".svg": "image/svg+xml", ".png": "image/png", ".ico": "image/x-icon"}
CFG = {"mode": "normal", "startup": 4.0, "speed": 4.0, "data": None}
MISSIONS = {}
LOCK = threading.Lock()
KEY_RE = re.compile(r"^[a-z0-9_]+__r[123]__(easy|medium|hard)$")


def cache_dir(key):
    return CFG["data"] / "cache" / key


def load_chunk(key, n):
    p = cache_dir(key) / f"chunk_{n:04d}.json"
    return json.loads(p.read_text()) if p.exists() else None


class H(BaseHTTPRequestHandler):
    def log_message(self, fmt, *a):
        print("[dev]", self.command, self.path, fmt % a if "%" in fmt else "")

    def send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/__dev/mode":
            if "m" in q:
                CFG["mode"] = q["m"][0]
                if CFG["mode"] == "slow":
                    CFG["startup"], CFG["speed"] = 45.0, 1.0
                elif "startup" not in q:
                    CFG["startup"], CFG["speed"] = 4.0, 4.0
            if "startup" in q:
                CFG["startup"] = float(q["startup"][0])
            if "speed" in q:
                CFG["speed"] = float(q["speed"][0])
            return self.send(200, {k: str(v) for k, v in CFG.items()})
        if u.path in ("/", "/rescue"):
            self.send_response(302)
            self.send_header("Location", "/rescue/")
            self.end_headers()
            return
        if u.path == "/rescue/gallery.json" and not (PAGE / "gallery.json").exists():
            return self.send(200, (CFG["data"] / "gallery.json").read_bytes())
        if u.path.startswith("/rescue/"):
            rel = u.path[len("/rescue/"):] or "index.html"
            p = (PAGE / rel).resolve()
            if PAGE.resolve() not in p.parents or not p.is_file():
                return self.send(404, {"error": "not found"})
            return self.send(200, p.read_bytes(), MIME.get(p.suffix, "application/octet-stream"))
        if u.path == "/api/mission/health":
            n = len(list((CFG["data"] / "cache").glob("*/status.json")))
            return self.send(200, {"ok": True, "running": sum(1 for m in MISSIONS.values() if not m.get("done")),
                                   "cap": 3, "cache_setups": n, "engine": "dev"})
        m = re.match(r"^/api/mission/([^/]+)$", u.path)
        if m:
            return self.poll(m.group(1).replace("%3A", ":").replace("%3a", ":"), int(q.get("from", ["0"])[0]))
        return self.send(404, {"error": "not found"})

    def poll(self, mid, start):
        mode = CFG["mode"]
        if "flaky" in mode.split(",") and random.random() < 0.5:
            return self.send(502, b"Bad Gateway", "text/plain")
        if mid.startswith("cache:"):
            key = mid[6:]
            if "nocache" in mode.split(",") or not (cache_dir(key) / "status.json").exists():
                return self.send(404, {"error": "no cached run for " + key})
            st = json.loads((cache_dir(key) / "status.json").read_text())
            chunks = [c for c in (load_chunk(key, n) for n in range(start, min(start + 6, st["chunks"]))) if c]
            return self.send(200, {"status": st, "chunks": chunks})
        with LOCK:
            ms = MISSIONS.get(mid)
        if not ms:
            return self.send(404, {"error": "unknown mission"})
        key = ms["setup"]
        full = json.loads((cache_dir(key) / "status.json").read_text())
        el = time.time() - ms["created"]
        sim_t = 0.0 if ("stuck" in ms["mode"].split(",") or el < ms["startup"]) else min(180.0, (el - ms["startup"]) * ms["speed"])
        st = {"v": 1, "mission": mid, "setup": key, "source": "live", "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
              "error": None, "score": None, "engine": {"git": "dev", "image": "dev"}}
        if "fail" in ms["mode"].split(",") and sim_t >= 40:
            st.update(state="failed", chunks=4, sim_t=40.0, error="robot_B pose left the house bounds (validation)",
                      fallback="cache:" + key)
            return self.send(200, {"status": st, "chunks": []})
        nch = int(sim_t // 10) if sim_t < 180 else full["chunks"]
        state = "starting" if el < ms["startup"] or "stuck" in ms["mode"].split(",") else ("done" if sim_t >= 180 else "running")
        st.update(state=state, chunks=nch, sim_t=round(sim_t, 1))
        if state == "done":
            st["score"] = full["score"]
            ms["done"] = True
        chunks = [c for c in (load_chunk(key, n) for n in range(start, min(start + 6, nch))) if c]
        return self.send(200, {"status": st, "chunks": chunks})

    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/api/mission":
            return self.send(404, {"error": "not found"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0") or 0)) or b"{}")
            key = f"{body['house']}__r{int(body['robots'])}__{body['difficulty']}"
        except Exception:
            return self.send(400, {"error": "bad setup"})
        if not KEY_RE.match(key) or not (cache_dir(key) / "status.json").exists():
            return self.send(400, {"error": "bad setup " + key})
        mode = CFG["mode"]
        if "down" in mode.split(","):
            return self.send(503, b"Service Unavailable", "text/plain")
        if "busy" in mode.split(","):
            return self.send(200, {"mission": "cache:" + key, "source": "recorded", "reason": "busy"})
        mid = "m%08x" % random.getrandbits(32)
        with LOCK:
            MISSIONS[mid] = {"setup": key, "created": time.time(), "mode": mode, "startup": CFG["startup"],
                             "speed": CFG["speed"]}
        return self.send(200, {"mission": mid, "source": "live", "poll": "/api/mission/" + mid})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "ryor" / "sample"))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--mode", default="normal")
    ap.add_argument("--startup", type=float, default=None)
    ap.add_argument("--speed", type=float, default=None)
    a = ap.parse_args()
    CFG["data"] = Path(a.data).resolve()
    CFG["mode"] = a.mode
    if a.mode == "slow":
        CFG["startup"], CFG["speed"] = 45.0, 1.0
    if a.startup is not None:
        CFG["startup"] = a.startup
    if a.speed is not None:
        CFG["speed"] = a.speed
    if not (CFG["data"] / "gallery.json").exists():
        raise SystemExit(f"no gallery.json in {CFG['data']}; run tools/ryor/page_fixture.py OUT first")
    print(f"serving http://127.0.0.1:{a.port}/rescue/ data={CFG['data']} mode={CFG['mode']}")
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
