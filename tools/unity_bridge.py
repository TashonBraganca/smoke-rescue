"""Talk to the running Unity editor through mcp-unity's WebSocket bridge, without an MCP client.

    uv run --with websockets python tools/unity_bridge.py menu "Tools/Smoke Rescue/Build House"
    uv run --with websockets python tools/unity_bridge.py logs [error|warning|info] [limit]
    uv run --with websockets python tools/unity_bridge.py call <method> '<json params>'

The editor must be open with the mcp-unity server running (it auto-starts on port 8090).
The auth token is the one mcp-unity writes to Library/McpUnity/bridge-token.
"""

import asyncio
import base64
import json
import sys
import uuid
from pathlib import Path

import websockets

PROJECT = Path(__file__).resolve().parents[1] / "unity" / "SmokeRescue"


def settings():
    cfg = json.loads((PROJECT / "ProjectSettings" / "McpUnitySettings.json").read_text())
    token = (PROJECT / "Library" / "McpUnity" / "bridge-token").read_text().strip()
    return cfg.get("Port", 8090), token


async def call(method: str, params: dict, timeout: float = 120.0) -> dict:
    port, token = settings()
    auth = base64.b64encode(f"mcp-unity:{token}".encode()).decode()
    async with websockets.connect(f"ws://localhost:{port}/McpUnity", additional_headers={"Authorization": f"Basic {auth}"},
                                  max_size=64 * 1024 * 1024) as ws:
        req_id = str(uuid.uuid4())
        await ws.send(json.dumps({"id": req_id, "method": method, "params": params}))
        while True:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout))
            if msg.get("id") == req_id:
                return msg


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    cmd = argv[0]
    if cmd == "menu":
        resp = asyncio.run(call("execute_menu_item", {"menuPath": argv[1]}, timeout=float(argv[2]) if len(argv) > 2 else 600))
    elif cmd == "logs":
        params = {"offset": 0, "limit": int(argv[2]) if len(argv) > 2 else 50, "includeStackTrace": False}
        if len(argv) > 1 and argv[1] != "all":
            params["logType"] = argv[1]
        resp = asyncio.run(call("get_console_logs", params))
    elif cmd == "call":
        resp = asyncio.run(call(argv[1], json.loads(argv[2]) if len(argv) > 2 else {}))
    else:
        print(__doc__)
        return 2
    print(json.dumps(resp, indent=1)[:20000])
    result = resp.get("result", resp)
    return 0 if (isinstance(result, dict) and result.get("success", True)) and "error" not in resp else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
