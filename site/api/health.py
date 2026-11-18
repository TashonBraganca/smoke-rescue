import json
from http.server import BaseHTTPRequestHandler

import cv2


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"status": "ok", "opencv": cv2.__version__, "backend": "vercel",
                           "model": "recorded runs", "approval_mode": "n/a"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)
