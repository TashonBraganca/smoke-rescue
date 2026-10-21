# Smoke Rescue

Two quadruped robots search a burning, smoke-filled building for trapped people. They start with no map, no floor plan, no outline and no shared coordinate frame. Each robot maps with its own 360 degree LiDAR and RGB-D camera, localises against its own map, recognises places its partner has seen from LiDAR fingerprints, and links the two maps when two independent matches agree. An LLM agent on each robot decides where to search and whether a warm shape is a person, using OpenCV 5 tools on the robot's own images. An incident commander outside hears their radio reports and relays them.

Entry to the OpenCV AI Competition 2026 (AWS). Simulation in Unity 6 HDRP; robot brain in C#; agents and OpenCV perception in Python.

## How it works

- **Sensors** (Unity, `unity/SmokeRescue/Assets/SmokeRescue/Sensors`): RGB-D camera 640x480 at 15 Hz with a depth noise model, 320x256 thermal camera, planar LiDAR (720 beams, 10 Hz, 0.15 to 12 m, smoke drops pulses), IMU at 200 Hz, leg odometry, air thermocouple. Every sensor is seeded and deterministic.
- **Robot brain** (`Brain/`, C#, 60 Hz): occupancy grid from LiDAR and depth; heading from the gyro with a small share of each LiDAR scan-to-map correction; frontier exploration; A* with wall and fire clearance; swept-body speed limits; stuck recovery. Reflexes for heat: back off from rising heat, keep a way out (leave when fire nears the route to an exit), emergency exit when the heat budget is spent, re-enter when cool again.
- **No prior knowledge**: each robot starts at the origin of its own frame. Floor counts as indoors only from the robot's own evidence (a ceiling seen by its camera, or walls on its own map in nearly every direction). Nothing about the building is given to robots, agents or the commander.
- **Map linking** (`Brain/FrameLink.cs`, `ScanContext.cs`, `ScanMatcher.cs`): LiDAR range-profile fingerprints are sent over a lossy radio; a candidate place is verified by scan matching against the robot's own map, and frames link only when two matches at least 1 m apart agree on one transform. Partner maps, claims and finds convert into each robot's frame after that.
- **Agents** (`brain/brain/agent/`): one GPT agent per robot and a commander. Tools run OpenCV 5 on the robot's own frames: `scan_thermal`, `inspect_candidate`, `assess_visibility`, `track_fire_spread`, `measure_opening`, and a YOLO11m person detector run through `cv2.dnn`. Every action (confirm a survivor, avoid an area, open a door, clear debris, set the torch) is refused in code unless it cites a matching OpenCV result.
- **Evaluation** (`brain/brain/nav_eval.py`, `compare_runs.py`, `rgb_people_eval.py`): scores every run against hidden ground truth that the robots never read (pose error, map precision and recall, searched floor, survivors found and false, heat exposure, link error, stalls).

## Run it

```bash
cd brain && uv sync && uv run pytest                                   # Python tests
AGENT_BACKEND=azure uv run uvicorn brain.agent.server:app --port 8080  # agents and the OpenCV person detector
```

In Unity 6000.6 open `unity/SmokeRescue`, then Tools > Smoke Rescue > Record > House agents run (or Haze agents full run for the 40 x 28 m building). Unity EditMode tests: Window > General > Test Runner. Score a recording:

```bash
cd brain
uv run python -m brain.nav_eval ../data/recordings/<run>
uv run python -m brain.compare_runs ../data/recordings/<run>
python ../tools/showcase.py ../data/recordings/<run> --run ../data/runs/<run> --out showcase.mp4
```

Every threshold the robots use is in `config/brain.json`, `config/sensors.json` and `config/fire.json`, each with the reason or measurement behind it.

## Credits and licences

| Asset or library | Source | License | Used for |
|---|---|---|---|
| OpenCV 5.0.0 | opencv.org | Apache 2.0 | perception tools, dnn person detector, odometry |
| YOLO11m (exported to ONNX) | github.com/ultralytics/ultralytics | AGPL-3.0 | person detector run by OpenCV dnn |
| Qwen2-VL-2B-Instruct | huggingface.co/Qwen/Qwen2-VL-2B-Instruct | Apache 2.0 | decision narrator (mlx-community 4-bit) |
| WLRI HRC thermal sample | data/external/wlri_hrc_thermal_sample | CC-BY 4.0 | thermal detector checks |
| ORB-SLAM3 | github.com/UZ-SLAMLab/ORB_SLAM3 | GPLv3 | per-robot SLAM |
| Poly Haven assets | polyhaven.com | CC0 | textures and furniture |
| ambientCG assets | ambientcg.com | CC0 | textures |
| mcp-unity | github.com/CoderGamester/mcp-unity | MIT | a coding agent control of the Unity editor |
| Floor Globe by Chris Torres | https://sketchfab.com/3d-models/0318ad51a7294885b452ff4f408353ba | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | antique_globe candidate review; modified exports pending |
| Celestial globe by Virtual Museums of Małopolska | https://sketchfab.com/3d-models/341fa8a777e94883841409438756f747 | [CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/) | antique_globe candidate review; modified exports pending |
| Antique Globe by Matthew Collings | https://sketchfab.com/3d-models/78b47132c6be42bcb664545a69b380d9 | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | antique_globe candidate review; modified exports pending |
| Simple Bunk Bed by Blender3D | https://sketchfab.com/3d-models/9fb7c54715dd41b98b3dc67b367cf81d | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | bunk_bed candidate review; modified exports pending |
| simple wooden bunk bed by 3DWorkbench | https://sketchfab.com/3d-models/a7685bd7b2354157b4c1a8264892acf0 | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | bunk_bed candidate review; modified exports pending |
| Bunk Bed by Amy | https://sketchfab.com/3d-models/e3f81d29bf234b728a154b41f50bbb9b | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | bunk_bed candidate review; modified exports pending |
| Bathtub by 3ddominator | https://sketchfab.com/3d-models/3350f81025974c53808a2efd2b31d4fd | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | clawfoot_bathtub candidate review; modified exports pending |
| Bathtub Game Ready Asset by NapoJax | https://sketchfab.com/3d-models/536dbfc2fdbe4dec9203ab390b9a6eda | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | clawfoot_bathtub candidate review; modified exports pending |
| Classic Bathtub by RenderArteStudio | https://sketchfab.com/3d-models/7427e2c0c32c44b89deb058f6583742a | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | clawfoot_bathtub candidate review; modified exports pending |
| Washer Dryer SBS by maimo | https://sketchfab.com/3d-models/79e4cdde40034be5be40eb4d9c9c92a2 | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | clothes_dryer candidate review; modified exports pending |
| Washer Dryer Machine by rhcreations | https://sketchfab.com/3d-models/8358b47f171d43e98f473083ce55b36e | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | clothes_dryer candidate review; modified exports pending |
| Modern Clothes Dryer by Glowbox 3D | https://sketchfab.com/3d-models/ea0080f2433b41578145944a3d0b350c | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | clothes_dryer candidate review; modified exports pending |
| clothing-rack by Sflex | https://sketchfab.com/3d-models/775375ae59be45dc9d7dcde1bc87a07f | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | clothes_rack candidate review; modified exports pending |
| Coat rail 2 by Andrew | https://sketchfab.com/3d-models/e36505afd27d4b71bfb1e12744672ce2 | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | clothes_rack candidate review; modified exports pending |
| Clothes Rack by sara3dbear | https://sketchfab.com/3d-models/f797e187b19c45e89c87ce8826421f90 | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | clothes_rack candidate review; modified exports pending |
| Coat Rack by Whostea | https://sketchfab.com/3d-models/72e821fc75d9473183a63d548959693b | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | coat_rack candidate review; modified exports pending |
| Coat Rack & Bowler Hat by raeganmaddox | https://sketchfab.com/3d-models/7818b79c9c87488287c9a1cf62bf1548 | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | coat_rack candidate review; modified exports pending |
| Hoop Coat Rack By Covo by Toss90 | https://sketchfab.com/3d-models/c460c415c8ad4d46a0f866b2b66bdd8d | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | coat_rack candidate review; modified exports pending |
| Sketchup Fireplace by Narutonic | https://sketchfab.com/3d-models/c95276929eda4cfb842f73d29b071893 | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | fireplace candidate review; modified exports pending |
| Oatlands Park Fireplace by glasgowmuseums | https://sketchfab.com/3d-models/d74fe57bb6e04a218280415d40ad34d8 | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | fireplace candidate review; modified exports pending |
| Fireplace by Mark Bai | https://sketchfab.com/3d-models/e9854400c0c1410db01c8db897ef5359 | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | fireplace candidate review; modified exports pending |
| Refrigerator by Francesco Coldesina | https://sketchfab.com/3d-models/0f5cd303517e4d4080834c0f6cece5b4 | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | fridge candidate review; modified exports pending |
| Refrigerator - Grey Polished Metal by Glowbox 3D | https://sketchfab.com/3d-models/2071bda681b642218b6829b82e4fd93b | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | fridge candidate review; modified exports pending |
| Refrigerator by Francesco Coldesina | https://sketchfab.com/3d-models/35793d480f784aa5ab3549eca2160e65 | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | fridge candidate review; modified exports pending |
| French Door Refrigerator - Stainless by Smokahontas | https://sketchfab.com/3d-models/ebe62ed6dd9b446a9c9b7d7d6a8086e7 | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | fridge candidate review; modified exports pending |
| Modular kitchen cabinets by sauti | https://sketchfab.com/3d-models/4b765920f7c34676acda91c0bfd87abd | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | kitchen_island candidate review; modified exports pending |
| Shaker kitchen island by Faith Teel | https://sketchfab.com/3d-models/524fb34efd68497e96471cf5bd67b17b | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | kitchen_island candidate review; modified exports pending |
| Kitchen Cabinet 1 by ganic30 | https://sketchfab.com/3d-models/bc547563aace4159bb24a5ddb30823d9 | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | kitchen_island candidate review; modified exports pending |
| Office Kitchen Island by jtressle | https://sketchfab.com/3d-models/d5a4e03f45da445a9d84e77d8a9f93a4 | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | kitchen_island candidate review; modified exports pending |
| Kitchen Hood Model 3D fbx by GLOBALO | https://sketchfab.com/3d-models/44b527d7d10d428b9930732553f479a2 | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | range_hood candidate review; modified exports pending |
| FREE Rectangular Cooker Hood 60 X 40 cm by marcin_malcherek | https://sketchfab.com/3d-models/a3ec10632d83426aaa5dad54422a6aec | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | range_hood candidate review; modified exports pending |
| Kitchen Hood Model 3D fbx by GLOBALO | https://sketchfab.com/3d-models/a8c9ef47afd541b0a13a1df1e0dfe0b8 | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | range_hood candidate review; modified exports pending |
| Kitchen Hood Model 3D fbx by GLOBALO | https://sketchfab.com/3d-models/f2de7240cfe64ebebff8f6d78f7f7879 | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | range_hood candidate review; modified exports pending |
| Sofa double Velvet red by lauti.ramos | https://sketchfab.com/3d-models/48525ce6c6e0495b9be36c8e420a1567 | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | red_sofa candidate review; modified exports pending |
| Antique Sofa by Heliona | https://sketchfab.com/3d-models/5384d703c1094899a45ae0a9a8dee909 | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | red_sofa candidate review; modified exports pending |
| Red leather sofa by Lucas Wild | https://sketchfab.com/3d-models/9bf97c25da904a60a24a44f6ff4d0885 | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | red_sofa candidate review; modified exports pending |
| Red sofa by paul.poderat | https://sketchfab.com/3d-models/ac7eaf38f20645fb90ed71c8aae60a79 | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | red_sofa candidate review; modified exports pending |
| SPOT MINI "rigged" Boston dynamics low poly by Okopchi | https://sketchfab.com/3d-models/22a0941a83bf4adeaa8ab44050d3c58e | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | robot candidate review; modified exports pending |
| Spot robot low poly model with rig by CG Material | https://sketchfab.com/3d-models/5d114e1b5c224ae39b36fdbb3fae5a1e | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | robot candidate review; modified exports pending |
| **OLD** Spot Mini (Rigged) by Greg McKechnie | https://sketchfab.com/3d-models/5dcbee77730640269cef5bd2587e328a | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | robot candidate review; modified exports pending |
| Boston Dynamics robot spot by ASHISH | https://sketchfab.com/3d-models/71354fd599e34db898a7d083851b792a | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | robot candidate review; modified exports pending |
| Persian Tabriz Pictorial Carpet by mfb64 | https://sketchfab.com/3d-models/0ce2fc1e54724cb79cbb08ab2182760a | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | rug candidate review; modified exports pending |
| Persian Malayer Carpet by mfb64 | https://sketchfab.com/3d-models/1120ca810d0c46289d3b7071103067ac | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | rug candidate review; modified exports pending |
| Persian Qashqai Rug by mfb64 | https://sketchfab.com/3d-models/fa968bf5d57a4265bedb233293870dc2 | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | rug candidate review; modified exports pending |
| Rug by zeg.ai | https://sketchfab.com/3d-models/fb0a0301abf6431485b70e497aa3f5bc | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | rug candidate review; modified exports pending |
| Sink and vanity by Timothy Ahene | https://sketchfab.com/3d-models/6e20541ed8f543b28d2e71aef79b7b5e | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | sink_with_mirror candidate review; modified exports pending |
| Sink and mirror by davihoff | https://sketchfab.com/3d-models/c976dd7dddab48a3a2848ff560a3c7b6 | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | sink_with_mirror candidate review; modified exports pending |
| Bathroom Sink Cabinet by jimbogies | https://sketchfab.com/3d-models/e5cf10304ffb4c11bc7841b584ccd706 | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | sink_with_mirror candidate review; modified exports pending |
| Debra - Detective Woman (Game Model) by Zukomazi | https://sketchfab.com/3d-models/208e5cf748404521894bf30c95f25c12 | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | survivor candidate review; modified exports pending |
| Woman In Red by reizibarrientos | https://sketchfab.com/3d-models/450e29de3c1541d78943e3265012ee48 | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | survivor candidate review; modified exports pending |
| man 4k by Hamdoon-Animation | https://sketchfab.com/3d-models/47ebe15d4c0545b88d2846bff9918208 | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | survivor candidate review; modified exports pending |
| Jamal - LOD Man character by Egunoff | https://sketchfab.com/3d-models/a169a23fa24f4df39daace6c55b24194 | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | survivor candidate review; modified exports pending |
| Toy Chest - Photogrammetry by JHXC | https://sketchfab.com/3d-models/27ad3793a9f54fed89a3fb17202551f0 | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | toy_chest candidate review; modified exports pending |
| Toy Box by Hayley Jacka | https://sketchfab.com/3d-models/8ce01f03c4c54c1bb69f6ba47a590586 | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | toy_chest candidate review; modified exports pending |
| Toy box by Wyme | https://sketchfab.com/3d-models/e12493a7481548129179d926b9a15b35 | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | toy_chest candidate review; modified exports pending |
| Washing Machine by Francesco Coldesina | https://sketchfab.com/3d-models/67549616d13246cfb1a540f9d6969078 | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | washing_machine candidate review; modified exports pending |
| Washing Machine (new) by nikita.bulgakov | https://sketchfab.com/3d-models/82c91049303f4664a56cc0eaffa8d662 | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | washing_machine candidate review; modified exports pending |
| Washing Machine by Zlat | https://sketchfab.com/3d-models/a27aa928a6664473a0ebfbe2c7f72b23 | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | washing_machine candidate review; modified exports pending |
| Water Heater by berdibekovadilet | https://sketchfab.com/3d-models/064f46139ef747e7b92a28bd2f5b2536 | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | water_heater candidate review; modified exports pending |
| Waterheater by Clément Pasquier | https://sketchfab.com/3d-models/20d4386b22d44ade878e5551c55d8565 | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | water_heater candidate review; modified exports pending |
| Water Heater Obj by Figgleton | https://sketchfab.com/3d-models/4d2647ed28d646e38637060e3d2000f2 | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | water_heater candidate review; modified exports pending |

Fire, smoke and debris:

- Flame and smoke flipbooks (Flame02, Flame03-hollow, Explosion01-nofire, DiscSmoke01, Cloud01): Unity Labs Paris free VFX image sequences, CC0. https://unity.com/blog/engine-platform/free-vfx-image-sequences-flipbooks
- "PBR Debris trash wood planks Pile Factory Scan" by Per's Scan Collection, CC-BY 4.0 (Sketchfab, via Objaverse). Hallway wood pile.
- "broken wooden board" by TheDrone, CC-BY 4.0 (Sketchfab, via Objaverse). Boards on the wood pile.
- "Pile of construction waste [3D SCAN; 8K; PBR]" by MrUnity, CC-BY 4.0 (Sketchfab, via Objaverse). Living-room debris.
- "Burnt Crate" by Blender3D, CC-BY 4.0 (Sketchfab, via Objaverse). Foyer obstacle.
- "Roof Rubble Pile Debris Tiles Scan PBR" by Per's Scan Collection, CC-BY 4.0 (Sketchfab, via Objaverse). Fallen ceiling rubble.
- Dartboard, vintage telephone wall clock, picture frames, wall clock: Poly Haven, CC0.
- Decision narrator model: Qwen2-VL-2B-Instruct (Alibaba Qwen team), Apache 2.0, run as the mlx-community 4-bit MLX conversion.
