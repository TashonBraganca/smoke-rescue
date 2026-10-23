"""Run one Unity EditMode test (or fixture) through the bridge once, with a long timeout and no retry.

    uv run --with websockets python tools/run_unity_test.py SmokeRescue.Tests.EditMode.HeadlessSim.RunHeadlessRequest [timeout_s]

A retrying caller re-sent long requests after its own 900 s timeout and started a second copy of a 14-minute
simulation inside the editor, which overwrote the first copy's trace files (23 Oct).
"""
import asyncio
import json
import sys

sys.path.insert(0, __import__("os").path.dirname(__file__))
from unity_bridge import call

name = sys.argv[1]
timeout = float(sys.argv[2]) if len(sys.argv) > 2 else 7200
resp = asyncio.run(call("run_tests", {"testMode": "EditMode", "testFilter": name, "returnOnlyFailures": False,
                                      "returnWithLogs": False}, timeout=timeout))
r = resp.get("result", resp)
print(r.get("message"))
for x in r.get("results", []):
    print(x.get("state"), x.get("name"), (x.get("message") or "")[:600].replace("\n", " | "))
    for line in (x.get("output") or "").splitlines()[-20:]:
        print("   ", line)
