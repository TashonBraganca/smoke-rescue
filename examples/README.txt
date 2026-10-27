Smoke Rescue robot API
======================

Any robot or AI agent can use the perception our simulated robots run: send a camera frame, get people back; ask
whether the colour camera still sees through the smoke; check whether two views show the same place; find fire and
body-warm regions in a thermal frame. The API is HTTP (FastAPI, brain/brain/agent/server.py), for robots and ROS
nodes. examples/robot_client.py is the client.

It runs the same OpenCV 5 code as the robots: brain.agent.vision (YOLO11m person detector through cv2.dnn, visibility,
thermal scan) and brain.agent.place (ORB place keyframes and rigid RANSAC).


Install
-------

Needs macOS or Linux, Python 3.11 and uv (https://docs.astral.sh/uv/).

    cd smoke-rescue/brain
    uv sync                      # installs opencv-python 5, fastapi and the rest from uv.lock

The person detector needs data/models/onnx/yolo11m.onnx (config/perception.json person_onnx). Without that file
detect_people returns an empty list, so check it exists. deploy/aws/export_yolo.py shows the export (ultralytics,
640x640, opset 12; it writes /model/yolo11m.onnx inside the AWS image); copy the result to data/models/onnx/.


Run the HTTP API
----------------

    cd smoke-rescue/brain
    .venv/bin/uvicorn brain.agent.server:app --host 127.0.0.1 --port 8095

Docs while it runs: http://127.0.0.1:8095/docs (Swagger UI), /redoc, and the schema at /openapi.json. The description
at the top of /docs lists which routes are public on AWS.

Send a frame from a robot (standard library only, about 30 lines):

        --url http://127.0.0.1:8095
    1 people (2267.0 ms, OpenCV 5.0.0)
      conf 0.93  bearing +17.5 deg  range 2.06 m  box [407, 383, 142, 95]

The request is POST /perceive/people with JSON:

    {"rgb_b64": "<PNG or JPEG, base64>",
     "depth_b64": "<optional 16-bit depth PNG, base64, same size as rgb>",
     "depth_scale": 1000,                       depth units per metre (1000 = millimetres)
     "intrinsics": {"fx": .., "fy": .., "cx": .., "cy": ..},   optional, pixels
     "min_conf": 0.25}

and the answer is {"people": [{"confidence", "bbox_rgb": [x, y, w, h], "bearing_deg", "range_m", "range_source",
"floor_gap_m"}], "ms", "opencv", "intrinsics_source", ...}. bearing_deg is positive to the right of the camera axis.
range_m is the horizontal distance from the robot body centre, null without depth. floor_gap_m is about 0 for a person
lying on the floor. In a ROS node, encode the image message with cv2.imencode('.png', bgr) and post the same body.

Other routes a robot can use: POST /vision/assess_visibility and POST /vision/scan_thermal take a PNG body (raw or
multipart) and return numbers; /place/{robot}/keyframe and /place/{robot}/match, /map3d/... and /robot/{robot}/step
work on recordings uploaded through /runs (see /docs).


Public routes on AWS
--------------------

HTTPS front (AWS Lambda function URL): https://ajfjm7evl33gu6ptprty6hfthq0ukygx.lambda-url.us-east-1.on.aws/api
It forwards only these, with the /api prefix:

    GET  /api/health
    POST /api/vision/<tool>          assess_visibility (colour PNG), scan_thermal (16-bit thermal PNG)
    POST /api/perceive/people

The load balancer behind it also forwards POST /place/<robot>/match without a header. Every other route answers 403
without the deployment's X-Origin-Verify header, so no outside caller can spend the model budget or upload data.
The Lambda front does not forward /docs or /openapi.json; read those on a local instance.

Known gap, 25 Oct 2026: the deployed service predates inline frames on /perceive/people and answers HTTP 500 to a
rgb_b64 body. It needs the current brain/brain/agent/server.py redeployed (deploy/aws/deploy.sh). Until then use a
local server for people; visibility and thermal work on AWS today.
