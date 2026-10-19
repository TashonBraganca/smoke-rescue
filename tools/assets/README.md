# Smoke Rescue asset candidates

The research contract is to find three to five downloaded alternatives per prop slot, plus quadruped robot bodies and adult survivors, using explicit CC0 or CC-BY 3.0/4.0 licences. Realistic materials and geometry matter more than popularity. The metadata search covers all 160 Objaverse 1.0 shards. The current Poly Haven catalogue was also checked. Its available models did not fill most missing domestic prop slots.

The 2026-10-19 shortlist contains 61 GLBs across 18 slots. Another 31 downloaded models are retained under `assets_cache/rejected/` with rejection notes. `CREDITS.md` attributes all 92 retained downloads. Every shortlisted asset has a cached source API response, its original annotation, licence URL, author, source URL, download URL and SHA-256 digest. Downloads with missing current licence information, restricted licences or missing source records were excluded. Search failures and fetch logs remain under `assets_cache/research/`.

Open `assets_cache/candidates/gallery.html` for the visual choices. `index.json` holds the structured catalogue and provisional recommendations. `report.txt` gives each candidate's licence, source face count and processing state. These recommendations come from inspected source thumbnails, not Blender renders. No candidate is production-ready yet. Normalized sizes and triangle counts are null until Blender processes the model.

The specified Blender 5.2.2 binary crashed twice before executing Python in this session. The minimal reproduction was `/Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup --threads 2 --python-expr 'print("HEADLESS_OK")'`. The crash stack reaches `supports_barycentric_whitelist`, `MTLBackend::metal_is_supported` and `_platform_strstr`. The logs are `assets_cache/research/sofa_test.log`, `blender_startup.log` and `blender.crash.txt`. CPU Cycles cannot avoid a crash during Blender's startup GPU detection. The running Blender GUI was not used or closed.

Run the following commands from the repository root in a shell where that Blender binary starts. The Python tools use an isolated uv environment and a temporary uv cache, without the brain environment.

```sh
UV_CACHE_DIR=/private/tmp/smoke-assets-uv uv run --no-project --with requests python tools/assets/discover.py
UV_CACHE_DIR=/private/tmp/smoke-assets-uv uv run --no-project --with requests python tools/assets/pipeline.py fetch
UV_CACHE_DIR=/private/tmp/smoke-assets-uv uv run --no-project --with requests python tools/assets/pipeline.py process
UV_CACHE_DIR=/private/tmp/smoke-assets-uv uv run --no-project --with requests --with pillow python tools/assets/pipeline.py verify --require-ready
```

`choices.json` records shortlist IDs, target scales and provisional picks. `reviews.json` records the source-thumbnail review. `fetch` caches HTTP responses and downloads, validates GLB headers and requires an explicit allowed licence URL from the source API. The original GLB is `source.glb`. `process` invokes the specified Blender binary with `--background --factory-startup`, writes `model.glb` and `normalization.json`, and renders `preview.png` at 640 by 480 with CPU Cycles. A Blender failure stops the batch before repeating the same startup failure. Use `process --slots robot survivor` to process a subset first.

The normalizer retains armatures, skins and separate meshes. Static mesh transforms are baked after clearing parents. Uniform scaling uses the configured target axis, and the origin is the bounding-box bottom centre. Rigged assets keep their hierarchy transforms so joint and animation data survive. Triangle counts above 150,000 trigger decimation. Rig deformation, animation retargeting and Unity import still require review. Front-facing orientation is unverified for every candidate; edit the per-UID yaw settings after checking the Blender renders.

Mesh filters can remove presentation geometry before measuring and exporting. The Toy Box filter retains the box and toys while excluding its room. The red velvet sofa filter removes its ottoman. These filters have not run in Blender yet. Kitchen cabinet sets need isolation and scale review, since their overall height can include wall cabinets or objects on the worktop. The washer-dryer pair is a fallback that needs the dryer separated from its single combined mesh. Do not promote these candidates before checking the exported geometry.

The garment-rail slot has no suitable realistic candidate with clothes. Two retained rails are untextured fallbacks. Only two textured domestic tank water heaters were found; the third candidate is untextured. Two of the Spot alternatives are low-poly fallbacks. The textured, rigged Spot Mini by Greg McKechnie is the provisional robot pick because the source GLB contains named joints for all four legs. Survivor candidates are clothed adults with source skins, but none has a verified lying or slumped pose. The fireplaces include stone and ornate mantels; no suitable brick fireplace with a mantel was found.

After inspecting a normalized model and its Blender preview, select a prop with the following command. It refuses models without completed normalization and preview output.

```sh
UV_CACHE_DIR=/private/tmp/smoke-assets-uv uv run --no-project --with requests python tools/assets/promote.py --slot fridge --uid ebe62ed6dd9b446a9c9b7d7d6a8086e7
```

Promotion preserves existing landmark offsets, adds a `downloaded` cache path to its `blender` part, and copies the chosen GLB into `assets_cache/models/blender_<prop>.glb`. The existing `blender` naming convention remains usable by the lead's importer. `build_props.py` reuses the downloaded model instead of regenerating that prop from basic geometry. Dryer selection handles the existing `washer` part with `variant` set to `dryer`. Robot and survivor selections remain in the catalogue because integration belongs to the lead. No promotion ran in this session, and `config/props.json` was not changed.

Rebuild the gallery with `pipeline.py catalogue`. Generate labelled contact sheets with `contact_sheet.py` after Blender processing, or with `contact_sheet.py --source-thumbnails` for the source images. Append attribution for newly retained downloads with `pipeline.py credits`. That command only appends rows. `pipeline.py archive` moves excluded downloads within `assets_cache/` according to the shortlist and records them in the catalogue's rejected section.

Primary source records are available at [Objaverse 1.0 on Hugging Face](https://huggingface.co/datasets/allenai/objaverse), [the Objaverse metadata directory](https://huggingface.co/datasets/allenai/objaverse/tree/main/metadata), [the Poly Haven model API](https://api.polyhaven.com/assets?t=models), and each candidate's Sketchfab URL in its metadata. The official [Unitree ROS repository](https://github.com/unitreerobotics/unitree_ros) is a further robot lead with a BSD-3-Clause repository licence and Go1/Go2 description folders. Its meshes were not downloaded or converted into candidates here.
