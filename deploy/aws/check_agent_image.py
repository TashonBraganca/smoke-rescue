"""Fail the build if server assets or required native OpenCV APIs are missing."""
import json

import cv2
import numpy as np

from brain.agent import server, vision, place
from brain.agent.map3d import CONFIG_DIR

assert cv2.__version__.split('.')[0] == '5', cv2.__version__
for name in ('brain', 'actuators', 'perception', 'sensors', 'map3d'):
    json.loads((CONFIG_DIR / f'{name}.json').read_text())
place.load_settings()
assert hasattr(cv2, 'Volume') and hasattr(cv2, 'VolumeType_HashTSDF')
assert hasattr(cv2, 'ORB_create') and hasattr(cv2, 'depthTo3d')
assert vision.PERSON_ONNX.is_file(), vision.PERSON_ONNX
net = cv2.dnn.readNetFromONNX(str(vision.PERSON_ONNX))
net.setInput(np.zeros((1, 3, 640, 640), np.float32))
output = net.forward()
assert output.shape == (1, 84, 8400), output.shape
assert np.isfinite(output).all(), 'nonfinite YOLO output'
print(f'PASS server import, config JSON, OpenCV {cv2.__version__}, HashTSDF/ORB/depthTo3d, YOLO {output.shape}')
print('team.json included' if (CONFIG_DIR / 'team.json').exists() else 'team.json absent in this checkout; no server reader')
print('PASS routes:', ', '.join(sorted(r.path for r in server.app.routes)))
