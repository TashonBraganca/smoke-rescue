import json
import os
import sys
from http.server import BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _vision import assess_visibility, read_image  # noqa: E402

MAX_BYTES = 4 * 1024 * 1024


class handler(BaseHTTPRequestHandler):
    def _send(self, code: int, data: dict):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > MAX_BYTES:
            return self._send(413 if n else 400, {"detail": "send one PNG or JPEG image up to 4 MB"})
        img = read_image(self.rfile.read(n), self.headers.get("Content-Type", ""))
        if img is None:
            return self._send(400, {"detail": "body must be a PNG or JPEG image"})
        return self._send(200, assess_visibility(img))

    def do_GET(self):
        self._send(405, {"detail": "POST an image"})
