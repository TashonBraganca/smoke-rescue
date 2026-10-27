<h1 align="center">Smoke Rescue</h1>

<p align="center">
  <b>Two robot dogs search a burning house with no map, find the people inside, and walk back out.</b><br>
  OpenCV AI Competition 2026 · OpenCV 5 · AWS
</p>

<p align="center">
  <a href="https://main.dyhvyjxo1bz4g.amplifyapp.com"><b>Watch a mission</b></a> ·
  <a href="https://main.dyhvyjxo1bz4g.amplifyapp.com/rescue/"><b>Start a live mission on AWS</b></a> ·
  <a href="https://ajfjm7evl33gu6ptprty6hfthq0ukygx.lambda-url.us-east-1.on.aws/"><b>System and results</b></a> ·
  <a href="#run-it"><b>Run it</b></a>
</p>

![Robot A's camera faces the fire while Robot B confirms a survivor. The shared map and team radio sit underneath.](docs/media/mission_dashboard.jpg)

In a burning house the first crew inside can see about an arm's length ahead. Every room searched twice costs time a trapped person does not have. Smoke Rescue sends two quadruped robots in first. They get no floor plan. Each robot maps the house with its own sensors, confirms people with OpenCV 5, and shares one map with its partner over a lossy radio. When the search ends, both walk out along their own trail.

## Results

| | Two robots, shared map | One robot |
|---|---|---|
| People found, 39 real floor plans the brain never saw | **99.0 %** | 92.3 % (p = 0.002) |
| Mean time to confirm a person | **67.2 s** | 94.5 s |
| False confirmations | **0** | 0 |

The three filmed Unity missions found **5 of 5, 4 of 5 and 4 of 5** people with no false finds. Both robots walked out every time. The [technical report](report/) has the method, the confidence intervals and the limits.

## How it works

![Architecture: the same robot brain runs in Unity for the films and in a headless simulator on AWS Fargate for live missions.](docs/media/architecture.png)

- **One brain, two homes.** Each robot runs the same C# brain at 60 Hz. It runs in Unity 6 HDRP for the filmed houses and in a headless simulator on AWS Fargate for live missions.
- **OpenCV 5 decides what is a person.** `cv2.dnn` runs YOLO11m on the robot's own camera, and the robot's depth image gives the range. A warm thermal shape only sends the robot over to look.
- **OpenCV 5 joins the two maps.** ORB keyframes with depth, a Hamming kNN match and a RANSAC rigid fit tell a robot where its partner's map sits. Each match must also pass a LiDAR check before the link moves.
- **The radio is the only link.** Map layers, room claims, people found and 7.6 KB keyframes travel over a radio that drops messages and carries 8 KB at most.
- **The agent advises.** Terra, a GPT-5.6 agent, suggests the next room. The brain checks every order and can refuse it.

## What is in this repo

| Folder | What it holds |
|---|---|
| [`unity/SmokeRescue`](unity/SmokeRescue) | Unity 6 HDRP project: houses, sensors, fire, and the C# robot brain |
| [`brain`](brain) | Python: OpenCV 5 perception, agent tools and server, evaluation |
| [`config`](config) | Every brain, sensor and fire constant, each with the reason behind it |
| [`tools/headless`](tools/headless) | The brain without Unity: logic tests and a headless mission runner |
| [`tools/ryor`](tools/ryor) | The live engine that AWS runs for each mission |
| [`deploy/aws`](deploy/aws) | CloudFormation stacks and deploy scripts |
| [`site`](site) | The Next.js judge website |
| [`report`](report) | Technical report |

## Run it

Pinned versions: Python 3.11 with `uv` (`brain/uv.lock` pins `opencv-python` 5.0.0.93), .NET 8, Unity 6000.6.6f1 HDRP, Node 20 or later (`site/package-lock.json` pins Next.js 16.4.3). Large files come through Git LFS, so run `git lfs install` before you clone.

```bash
# OpenCV 5 perception, agent server and Python tests
cd brain && uv sync && uv run pytest -q
uv run uvicorn brain.agent.server:app --port 8080

# The C# brain without Unity: 305 logic tests, then one scored mission
./tools/headless/test.sh
./tools/headless/run.sh <request.json>        # request format: tools/ryor/CONTRACT.txt

# The judge website (static export in site/out)
cd site && npm ci && npm run build
```

**Record a film.** Open `unity/SmokeRescue` in Unity 6000.6.6f1, then pick *Tools > Smoke Rescue > Record > House one-door seed 934 full mission*. Render the recording with `brain/.venv/bin/python tools/showcase.py data/recordings/<run> --speed 8 --out film.mp4`.

**Deploy to AWS** (us-east-1, needs AWS CLI v2 and Docker):

```bash
tools/ryor/build_image.sh <tag>                        # linux/arm64 engine image
docker tag smoke-rescue/ryor-engine:<tag> <account>.dkr.ecr.us-east-1.amazonaws.com/smoke-rescue/ryor-engine:<tag>
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/smoke-rescue/ryor-engine:<tag>
IMAGE_TAG=<tag> ./deploy/aws/deploy_ryor.sh            # Fargate engine, S3 runs bucket, DynamoDB limits
./deploy/aws/deploy_site_lambda.sh                     # mission API and report page
```

## Ground rules

- **Robots never read ground truth.** Unity assembly definitions and a pytest check keep it away from robot code. Only the evaluator scores a run.
- **No constant names a house.** Every threshold in `config/` has a written reason, and the same brain runs on every floor plan.
- **Every change gets a recording audit.** A fix ships with a test that fails without it.

## Credits

OpenCV 5 (Apache 2.0), YOLO11m (AGPL-3.0), HouseExpo floor plans (MIT), and CC0 or CC-BY textures and furniture. The full list with licences is in [CREDITS.txt](CREDITS.txt).
