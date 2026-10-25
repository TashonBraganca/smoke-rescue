=====================================

Any robot or AI agent can use the perception our simulated robots run: send a camera frame, get people back; ask
whether the colour camera still sees through the smoke; check whether two views show the same place; find fire and
body-warm regions in a thermal frame. Two ways in:

  1. HTTP (FastAPI, brain/brain/agent/server.py), for robots and ROS nodes. examples/robot_client.py is the client.

Both run the same OpenCV 5 code as the robots: brain.agent.vision (YOLO11m person detector through cv2.dnn, visibility,
thermal scan) and brain.agent.place (ORB place keyframes and rigid RANSAC).


Install
-------

Needs macOS or Linux, Python 3.11 and uv (https://docs.astral.sh/uv/).

    cd smoke-rescue/brain

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


------------------

    cd smoke-rescue/brain

Environment:

                    remote: send frames to the HTTP API at SMOKE_API_URL through its public routes.
    SMOKE_API_URL   default https://ajfjm7evl33gu6ptprty6hfthq0ukygx.lambda-url.us-east-1.on.aws/api
                    (or a local server, e.g. http://127.0.0.1:8095)

Every tool also takes backend="local" or "remote" per call.

The HTTP transport binds to 127.0.0.1 by default and can read any image file the server's user can read; do not
expose it on a network as is.


Connect from a coding agent
------------------------


or, with the HTTP server already running:




---------------------------


    {
      "mcpServers": {
        "smoke-rescue": {
          "command": "/Volumes/Tashon/opencv/smoke-rescue/brain/.venv/bin/python",
          "env": {"SMOKE_BACKEND": "local"}
        }
      }
    }

Replace the path with your checkout. The same entry through uv, if you prefer it:

          "command": "uv",

Both forms were started from / over stdio and answered tools/list and a tool call on 25 Oct 2026.


Tools
-----

detect_people(image, depth?, depth_scale=1000, intrinsics?, min_conf=0.25, backend?)
    YOLO11m person boxes with confidence, bearing, range (from depth) and floor gap. min_conf 0.25 is the robots' live
    detector default; the robots confirm a person only at 0.5 (config/brain.json person_rgb_min_confidence).

assess_visibility(image, depth?, depth_scale=1000, backend?)
    Local: laplacian_variance (sharpness), rms_contrast, dark_channel_haze (median of a 15x15 minimum filter, a
    brightness proxy, not physical extinction), valid_depth_fraction (null without depth) and usable_rgb (variance >= 20,
    contrast >= 12, haze <= 0.8). Remote: laplacian_var, rms_contrast, dark_channel (a mean, not a median), no verdict.

match_place(keyframe_a, keyframe_b, return_payloads=false)
    Each keyframe is {"rgb": .., "depth": .., "intrinsics"?: .., "depth_scale"?: ..} or a radio payload (base64, the
    b'VP' format the robots send each other, 38 bytes per feature, 7605 bytes for 200 features). Returns matched and,
    if true, keyframe_b's pose in keyframe_a's body frame: dx_m right, dy_m forward, dyaw_deg counter-clockwise, with
    inliers and rmse_m. Always local: no public AWS route takes two keyframes.

describe_scene_heat(thermal, rgb?, depth?, depth_scale=1000, backend?)
    thermal is a 16-bit single-channel PNG holding kelvin x 10. Local: hotspots (>= 200 C regions with bearing, mean,
    max and a range: from depth, or an upper bound from where the lowest hot pixels meet the floor, because flames
    return no depth), people (28 to 40 C warm regions, unconfirmed), rgb_people (colour person boxes, only when rgb is
    sent), max_c, median_c, hot_fraction, human_band_fraction. Remote: hot_regions and human_band_regions in thermal
    pixels.


    detect_people(person_rgb.png, person_depth.png)  1 person, conf 0.93, bearing +17.5 deg, range 2.06 m
    detect_people(fire_rgb.png, fire_depth.png)      no people
    assess_visibility(person_rgb.png)                 laplacian 526.3, contrast 41.3, haze 0.27, usable
    match_place(turn_*, person_*)                     matched, 68 inliers, dx 0.004 m, dy 0.062 m, dyaw -10.9 deg
                                                      (the robot's leg odometry: 0.004, 0.064 m, -10.3 deg)
    describe_scene_heat(fire_thermal.png)             1 hotspot, 1990 px, max 322.8 C, bearing -23.2 deg


Limits
------

- Ranges, floor gaps and place keyframes assume this project's camera mast (config/sensors.json camera.mount: 1.6 m
  high, 0.25 m ahead of the body centre, pitched 10 deg down). Another robot's boxes and bearings are right; its ranges
  are biased by the difference in mount.
- Without intrinsics, the project camera's are scaled to the image size, which is approximate for other cameras.
- The person detector costs about 1.3 s per frame on an idle M5 CPU (config/perception.json person_onnx_reason) and
  measured 1.8 to 7 s on 25 Oct while other jobs shared the machine.
- describe_scene_heat without rgb still runs the colour detector on a black image inside scan_thermal (about 2 s,
  always empty); its rgb_people is reported as null.
