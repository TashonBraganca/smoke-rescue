"""Build-time YOLO11m export. The runtime uses only OpenCV DNN."""
from pathlib import Path
import urllib.request

from ultralytics import YOLO

# Use a versioned upstream release, rather than Ultralytics' moving download default.
url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt"
weights = Path("/tmp/yolo11m.pt")
with urllib.request.urlopen(url, timeout=120) as response, weights.open("wb") as target:
    while chunk := response.read(1024 * 1024):
        target.write(chunk)
exported = YOLO(str(weights)).export(format="onnx", imgsz=640, batch=1, opset=12,
                                     dynamic=False, simplify=False, nms=False, device="cpu")
Path("/model").mkdir(exist_ok=True)
Path(exported).replace("/model/yolo11m.onnx")
