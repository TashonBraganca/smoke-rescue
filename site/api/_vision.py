# The OpenCV 5 checks behind the "Try it" section. Same math as brain/agent/server.py /vision/{tool} on the old Fargate service.
import time
import cv2
import numpy as np


def read_image(body: bytes, ctype: str):
    if ctype.startswith("multipart/form-data"):
        # Browsers send multipart; take the first file part with the standard-library parser.
        from email.parser import BytesParser
        from email.policy import default
        msg = BytesParser(policy=default).parsebytes(b"Content-Type: " + ctype.encode() + b"\r\n\r\n" + body)
        parts = [p for p in msg.iter_parts() if p.get_filename() or p.get_content_type().startswith("image/")]
        body = parts[0].get_payload(decode=True) if parts else b""
    if not body:
        return None
    return cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_UNCHANGED)


def assess_visibility(img) -> dict:
    t0 = time.perf_counter()
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.astype(np.uint8)
    lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(gray.std())
    dark = float(cv2.erode(img.min(axis=2) if img.ndim == 3 else gray, np.ones((15, 15), np.uint8)).mean() / 255)
    return {"laplacian_var": round(lap, 1), "rms_contrast": round(contrast, 1), "dark_channel": round(dark, 3),
            "tool": "assess_visibility", "opencv": cv2.__version__, "ms": round((time.perf_counter() - t0) * 1000, 1)}
