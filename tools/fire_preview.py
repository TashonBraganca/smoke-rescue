#!/usr/bin/env python3
"""Render FireSim JSON snapshots, without calculating or advancing fire state.

Schema v1 is produced by sim.ToJson(), or by FireSim.DumpJson(timeSeconds).
The static overload DumpJson(plan, firePath, seed, timeSeconds) accepts a scenario.
An Editor menu can use File.WriteAllText(path, FireSim.DumpJson(t)); no editor
menu or Unity process is started here. Save one snapshot per UTF-8 JSON file.

Required root fields:
  version: 1
  time_s: simulated fixed-step time, seconds
  seed: integer
  width, height: grid counts
  cell_size_m: cell edge length in metres
  origin: [x_min, z_min], Unity world metres
  wall_height_m: ceiling height, metres
  ambient_c, max_temperature_c: heat-map bounds, Celsius
  state_hash: SHA-256 of internal solver state
  cells: width*height entries in z-major order, index = z*width+x
    room: string or null for cells outside all rooms
    fuel_kg, initial_fuel_kg: accessible combustible mass in the cell
    temperature_c: equivalent burning/preheating surface temperature, Celsius
    flame, burnt: fractions in [0,1]
    collapsed: boolean
  rooms: [{room, upper_c, lower_c, layer_height_m, smoke,
           flashover_time_s, hot_exposure_s}, ...]
    smoke is a dimensionless visual density. flashover_time_s is -1 until reached.
  collapses: [{time_s, room, rect:[xmin,zmin,xmax,zmax], hot_exposure_s}, ...]
  walls: [[x1,z1,x2,z2], ...], metre segments with door gaps already removed

No interpolation, ignition, spread, gas transport, or wall inference occurs here.
The first JSON row is the southern row and is flipped for north-up PNG output.
Fuel consumption darkens the heat map; yellow/white dots mark active flames;
purple outlines and crosses mark ceiling-panel events. Grey lines are walls.

Run `python3 tools/fire_preview.py` after placing dumps in data/fire_dumps/.
Or run `python3 tools/fire_preview.py path/to/060.json path/to/240.json`.
Directories and quoted glob patterns are accepted. Use --output to set the PNG
directory, or --stdin to read one snapshot from standard input. The default
output directory is data/fire_preview/. numpy and Pillow are the only packages.
"""

import argparse
import glob
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw


def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def validate(snapshot):
    """Check the dump contract before allocating a potentially large raster."""
    if not isinstance(snapshot, dict) or snapshot.get("version") != 1:
        raise ValueError("Expected one FireSim snapshot with version 1")
    for name in ("width", "height"):
        value = snapshot.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    width, height = snapshot["width"], snapshot["height"]
    if width * height > 1_000_000:
        raise ValueError("Grid exceeds one million cells")
    cell = finite(snapshot.get("cell_size_m"), "cell_size_m")
    if cell <= 0:
        raise ValueError("cell_size_m must be positive")
    ambient = finite(snapshot.get("ambient_c"), "ambient_c")
    maximum = finite(snapshot.get("max_temperature_c"), "max_temperature_c")
    if maximum <= ambient:
        raise ValueError("max_temperature_c must exceed ambient_c")
    finite(snapshot.get("time_s"), "time_s")
    if not isinstance(snapshot.get("origin"), list) or len(snapshot["origin"]) != 2:
        raise ValueError("origin must contain x_min and z_min")
    for v in snapshot["origin"]:
        finite(v, "origin")
    cells = snapshot.get("cells")
    if not isinstance(cells, list) or len(cells) != width * height:
        raise ValueError("cells length must equal width*height")
    for index, state in enumerate(cells):
        if not isinstance(state, dict):
            raise ValueError(f"cells[{index}] must be an object")
        if state.get("room") is not None and not isinstance(state["room"], str):
            raise ValueError(f"cells[{index}].room must be a string or null")
        temperature = finite(state.get("temperature_c"), "temperature_c")
        if not ambient <= temperature <= maximum:
            raise ValueError(f"cells[{index}] temperature exceeds dump bounds")
        for name in ("flame", "burnt"):
            if not 0 <= finite(state.get(name), name) <= 1:
                raise ValueError(f"cells[{index}].{name} must be in [0,1]")
        for name in ("fuel_kg", "initial_fuel_kg"):
            if finite(state.get(name), name) < 0:
                raise ValueError(f"cells[{index}].{name} must be nonnegative")
    for segment in snapshot.get("walls", []):
        if not isinstance(segment, list) or len(segment) != 4:
            raise ValueError("Each wall needs [x1,z1,x2,z2]")
        for v in segment:
            finite(v, "wall coordinate")
    for event in snapshot.get("collapses", []):
        if not isinstance(event.get("rect"), list) or len(event["rect"]) != 4:
            raise ValueError("Each collapse needs rect [xmin,zmin,xmax,zmax]")
        for v in event["rect"]:
            finite(v, "collapse coordinate")
        if event["rect"][2] <= event["rect"][0] or event["rect"][3] <= event["rect"][1]:
            raise ValueError("Collapse rectangle must have positive area")
    return snapshot


def render(snapshot, pixels_per_cell=12):
    """Return a PIL image. Rendering uses only values present in the snapshot."""
    validate(snapshot)
    if not 4 <= pixels_per_cell <= 40:
        raise ValueError("pixels_per_cell must be between 4 and 40")
    width, height = snapshot["width"], snapshot["height"]
    if width * height * pixels_per_cell**2 > 80_000_000:
        raise ValueError("Output exceeds 80 million pixels; reduce --scale")
    cells = snapshot["cells"]
    temperatures = np.array([c["temperature_c"] for c in cells], dtype=np.float64).reshape(height, width)
    burnt = np.array([c["burnt"] for c in cells], dtype=np.float64).reshape(height, width)
    # Fixed Celsius colour stops make different snapshots comparable.
    stops = np.array([20, 150, 300, 450, 600, 800, 1100], dtype=np.float64)
    colours = np.array([[20, 27, 40], [35, 70, 130], [155, 45, 45], [215, 65, 22],
                        [250, 125, 25], [255, 220, 80], [255, 250, 235]], dtype=np.float64)
    rgb = np.stack([np.interp(temperatures, stops, colours[:, i]) for i in range(3)], axis=-1)
    rgb = rgb * (1 - .72 * burnt[..., None]) + np.array([23, 20, 18]) * (.72 * burnt[..., None])
    outside = np.array([c.get("room") is None for c in cells]).reshape(height, width)
    rgb[outside] = [7, 9, 13]
    top = 58
    margin = 16
    bottom = 74
    grid = Image.fromarray(np.flipud(rgb.astype(np.uint8))).resize(
        (width * pixels_per_cell, height * pixels_per_cell), Image.Resampling.NEAREST)
    image = Image.new("RGB", (grid.width + 2 * margin, grid.height + top + bottom), (12, 15, 22))
    image.paste(grid, (margin, top))
    draw = ImageDraw.Draw(image)
    size = snapshot["cell_size_m"]
    ox, oz = snapshot["origin"]

    def point(x, z):
        return (margin + (x - ox) / size * pixels_per_cell,
                top + (height - (z - oz) / size) * pixels_per_cell)

    for i, state in enumerate(cells):
        intensity = state["flame"]
        if intensity <= 0:
            continue
        z, x = divmod(i, width)
        px, py = point(ox + (x + .5) * size, oz + (z + .5) * size)
        radius = max(1, pixels_per_cell * .36 * math.sqrt(intensity))
        draw.ellipse((px - radius, py - radius, px + radius, py + radius),
                     fill=(255, int(150 + 105 * intensity), int(70 + 130 * intensity)))
    for event in snapshot.get("collapses", []):
        x0, z0, x1, z1 = event["rect"]
        left, lower = point(x0, z0)
        right, upper = point(x1, z1)
        draw.rectangle((left, upper, right, lower), outline=(235, 125, 255), width=2)
        draw.line((left, upper, right, lower), fill=(235, 125, 255), width=2)
        draw.line((left, lower, right, upper), fill=(235, 125, 255), width=2)
    for x1, z1, x2, z2 in snapshot.get("walls", []):
        draw.line((*point(x1, z1), *point(x2, z2)), fill=(200, 208, 215), width=3)
    for room in snapshot.get("rooms", []):
        indices = [i for i, c in enumerate(cells) if c.get("room") == room["room"]]
        if not indices:
            continue
        xs = [i % width for i in indices]
        zs = [i // width for i in indices]
        px, py = point(ox + (sum(xs) / len(xs) + .5) * size, oz + (sum(zs) / len(zs) + .5) * size)
        text = f"{room['room']} {room['upper_c']:.0f} C\nh={room['layer_height_m']:.2f} m smoke={room['smoke']:.2f}"
        box = draw.multiline_textbbox((px, py), text, anchor="mm")
        draw.rectangle((box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2), fill=(12, 15, 22))
        draw.multiline_text((px, py), text, fill=(235, 239, 243), anchor="mm", align="center")
    active = sum(c["flame"] > 0 for c in cells)
    title = f"Smoke Rescue  t={snapshot['time_s']:.1f} s  seed={snapshot['seed']}  burning cells={active}"
    draw.text((margin, 10), title, fill=(240, 243, 247))
    draw.text((margin, 30), f"North up | state {snapshot.get('state_hash', 'unknown')[:20]}", fill=(165, 181, 198))
    legend_y = top + grid.height + 12
    draw.text((margin, legend_y), "Surface C: 20   150   300   450   600   800   1100", fill=(210, 219, 228))
    draw.text((margin, legend_y + 19), "Yellow dots: flames. Dark cells: burnt fuel. Purple crosses: ceiling panels.", fill=(210, 219, 228))
    draw.text((margin, legend_y + 38), "Room labels show upper gas C, layer height and dimensionless smoke.", fill=(165, 181, 198))
    return image


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="*", help="Snapshot JSON paths, directories or globs")
    parser.add_argument("--output", type=Path, default=Path("data/fire_preview"))
    parser.add_argument("--scale", type=int, default=12, help="Pixels per cell, 4 to 40")
    parser.add_argument("--stdin", action="store_true", help="Read one snapshot from standard input")
    args = parser.parse_args(argv)
    try:
        if args.stdin:
            if args.inputs:
                parser.error("--stdin cannot be combined with input paths")
            snapshots = [("stdin", json.load(sys.stdin))]
        else:
            paths = []
            for pattern in args.inputs or ["data/fire_dumps/*.json"]:
                path = Path(pattern)
                matches = sorted(path.glob("*.json")) if path.is_dir() else [Path(p) for p in sorted(glob.glob(pattern))]
                if not matches:
                    raise ValueError(f"No dumps found at {pattern}. Save FireSim.DumpJson(t) output first.")
                paths.extend(matches)
            snapshots = [(path.stem, json.loads(path.read_text(encoding="utf-8"))) for path in dict.fromkeys(paths)]
        for index, (name, snapshot) in enumerate(snapshots):
            image = render(snapshot, args.scale)
            args.output.mkdir(parents=True, exist_ok=True)
            output = args.output / f"{index:03d}_{name}_{snapshot['time_s']:07.1f}s.png"
            image.save(output)
            print(output)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(2, f"fire_preview: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
