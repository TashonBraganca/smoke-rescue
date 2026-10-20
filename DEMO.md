# Demo runbook (10 minutes)

One command opens everything:

```sh
cd /Volumes/Tashon/opencv/smoke-rescue && tools/demo_now.sh
```

It prints the scored report, opens the results page in the browser, plays the robots' front-camera video in QuickTime, and opens the Rerun replay of the latest complete run (`unity_clear_prev_art2`). The replay is 256 MB and takes 10 to 20 seconds to load.

## What to show, in order

1. **The problem (30 s).** Two quadruped robots enter a burning, smoky house through different doors. Neither has a map, GPS or the floor plan. Each sees only through its own RGB-D camera at human eye height.

2. **What the robots see (2 min): QuickTime video.** Robot A and robot B side by side. Top: colour camera. Bottom left: depth; the speckle is smoke dropout. Bottom right: the ORB corners that SLAM tracks on that frame, with the count. Point out the fire in the living room, the smoke, the survivors on the floor. This is the exact data SLAM reads.

3. **Mapping and the merge (4 min): Rerun replay.** Press play on the timeline at the bottom.
   - Each robot builds its own map in its own frame. The robots do not know where the other one is.
   - Every 2 s each robot radios compact place fingerprints (512 numbers from DINOv2). When robot B's fingerprint matches a place robot A has seen, the match is checked with ORB features plus depth, then ICP. Accepted matches show as green lines, rejected candidates flash red for a second, and the log panel says why each one was rejected.
   - At about 110 s the maps merge: robot B's whole map snaps into robot A's frame. Scrub back and forth over that moment.
   - The grey outline is the true house, shown only for scoring. The robots never read it.

4. **Numbers (2 min): results page.** Point at the table row "Clear air, latest full run".
   - Merge error at the matched place: 0.6 cm and 0.06 degrees. False merges: 0.
   - Robot B's SLAM error over 137 s: 2.1 cm. Robot A's: 32.9 cm (be upfront, see below).
   - Merged map within 8.2 cm (median) of the true walls. Survivors: 2 of 4 found, 0 false; the other 2 are in rooms the scripted routes never enter.

5. **What is honest about it (1 min).** All numbers are measured against hidden ground truth the robots never see. Robot A's 32.9 cm comes from heading errors at doorways, caused by repeating wallpaper and glossy floors reflecting its own floodlight. Both are fixed in the house; the re-recorded run is processing.

## If someone asks

- **Is it real SLAM?** Yes: ORB-SLAM3 (RGB-D mode) in Docker, unmodified apart from a headless shutdown patch.
- **Is the place recognition real?** Yes: DINOv2 ViT-L/14 features, VLAD-aggregated, 94 ms per frame on the M5 GPU.
- **Why simulation?** Ground truth. Every pose, wall and survivor position is known exactly, so every error number is measured, not estimated.
- **What's next?** Robot A fix confirmed on the new recording, then the smoke run, the showcase video, and phase 2 (live robot-to-robot bridge, autonomous exploration, a third robot).

## Backup if something fails to open

- Results page: 
- Video file: `data/evidence/walkthrough_unity_haze_4x.mp4` (34 s at 4x speed) or `walkthrough_unity_haze.mp4` (real time, 137 s).
- Replay by hand: `cd brain && uv run rerun ../data/runs/unity_clear_prev_art2/replay.rrd`
- The TUM public-data run also works: `tools/demo_now.sh tum_fr3_long_office_2halves`
