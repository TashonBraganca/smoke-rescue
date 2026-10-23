"""Collect the M1 offline evidence after running the requests in data/runs/headless/m1."""
from pathlib import Path
import argparse
import importlib.util
import json
import shutil
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'brain'))
sys.path.insert(0, str(ROOT / 'tools'))
from brain.nav_eval import body_trajectory, localization_errors, error_summary, wrap_angle
from brain.nav_io import read_json_lines, load_scenario, recording_frame, placed_states
from brain.team_map import load_team_map
from headless_compare import main as compare


def replay_errors(recordings, runs):
    rows = {}
    for run, robot, names in [
        ('unity_house_onedoor_seed934', 'robot_B', ['recorded', 'before', 'after', 'gyro_before', 'gyro_after', 'gain0', 'gain20', 'gain100']),
        ('unity_house_onedoor_seed934', 'robot_A', ['A934_gain0.05', 'A934_gain1']),
        ('unity_house_agents_run13', 'robot_A', ['A13_gain0.05', 'A13_gain1']),
        ('unity_house_agents_run13', 'robot_B', ['B13_gain0.05', 'B13_gain1']),
    ]:
        rec = recordings / run
        scenario = load_scenario(json.loads((rec / 'meta.json').read_text()))
        truth_t, truth_pose = body_trajectory(rec, robot)
        transform = recording_frame(read_json_lines(rec / robot / 'brain_state.jsonl'), scenario, robot)[0]
        for name in names:
            path = rec / robot / 'brain_state.jsonl' if name == 'recorded' else runs / f'replay_{name}.jsonl'
            states = read_json_lines(path)
            report, *_ = localization_errors(placed_states(states, transform), truth_t, truth_pose)
            rows[name] = report | dict(run=run, robot=robot, path=str(path))
    return rows


def fault_errors(runs):
    result = {}
    for variant in ['noise_final_off', 'noise_final_on']:
        rows = read_json_lines(runs / ('5001_' + variant) / 'trace.jsonl')
        result[variant] = {}
        for robot in ['robot_A', 'robot_B']:
            poses = [r for r in rows if r.get('kind') == 'pose' and r['robot'] == robot]
            pos = np.array([np.hypot(r['bx'] - r['step_x'], r['by'] - r['step_y']) for r in poses])
            yaw = np.array([abs(np.degrees(wrap_angle(r['byaw_rad'] - r['step_yaw_rad']))) for r in poses])
            result[variant][robot] = dict(position_m=error_summary(pos), yaw_deg=error_summary(yaw))
    return result


def render_beliefs(runs, out):
    spec = importlib.util.spec_from_file_location('showcase', ROOT / 'tools/showcase.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    grids = [load_team_map(runs / '5001_final' / (robot + '.team.bin.gz')) for robot in ['robot_A', 'robot_B']]
    known = (grids[0].source_mask != 0) | (grids[1].source_mask != 0)
    yy, xx = np.nonzero(known)
    box = (slice(max(0, yy.min() - 3), min(grids[0].size, yy.max() + 4)),
           slice(max(0, xx.min() - 3), min(grids[0].size, xx.max() + 4)))
    image = Image.new('RGB', (1240, 780), module.BG)
    draw = ImageDraw.Draw(image)
    draw.text((20, 16), 'House 5001, robot exports at 240 s', font=module.F['title'], fill=module.TEXT)
    draw.text((20, 58), 'No truth geometry in these panels. Each receiver keeps only the partner evidence it received.', font=module.F['sub'], fill=module.DIM)
    for k, grid in enumerate(grids):
        crop = Image.fromarray(np.flipud(module.team_map_rgb(grid)[box]))
        scale = min(590 / crop.width, 560 / crop.height)
        crop = crop.resize((round(crop.width * scale), round(crop.height * scale)), Image.Resampling.NEAREST)
        x = 20 + 620 * k
        image.paste(crop, (x + (590 - crop.width) // 2, 140))
        draw.text((x, 105), f"robot_{'AB'[k]}'s team belief", font=module.F['h'], fill=module.TEXT)
        draw.text((x, 725), 'Walls brighten with occupancy confidence.', font=module.F['small'], fill=module.DIM)
        draw.text((x, 747), 'Floor colour follows camera search confidence and age.', font=module.F['small'], fill=module.DIM)
    image.save(out / 'team_maps.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recordings', type=Path, default=ROOT.parent / 'smoke-rescue/data/recordings')
    parser.add_argument('--runs', type=Path, default=ROOT / 'data/runs/headless/m1')
    parser.add_argument('--out', type=Path, default=ROOT / 'm1_evidence')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    comparison = args.runs / 'comparison'
    comparison.mkdir(exist_ok=True)
    for seed in range(5001, 5005):
        for variant in ['before', 'final']:
            run = args.runs / f'{seed}_{variant}'
            if not (run / 'result.json').exists():
                raise RuntimeError(f'incomplete house measurement {run}')
            link = comparison / run.name
            if not link.exists():
                link.symlink_to(run.resolve(), target_is_directory=True)
    compare(comparison)
    data = dict(replays=replay_errors(args.recordings, args.runs), motion_error=fault_errors(args.runs),
                houses=json.loads((comparison / 'compare.json').read_text()),
                revert_controls=json.loads((args.runs / 'reverts/summary.json').read_text()))
    data['team_maps_by_receiver'] = {}
    data['runtime'] = {}
    for variant in ['before', 'final']:
        data['team_maps_by_receiver'][variant] = {}
        data['runtime'][variant] = {}
        for seed in map(str, range(5001, 5005)):
            run = args.runs / f'{seed}_{variant}'
            report = json.loads((run / 'report.json').read_text())
            result = json.loads((run / 'result.json').read_text())
            data['team_maps_by_receiver'][variant][seed] = report['team_map_by_receiver']
            data['runtime'][variant][seed] = {key: result[key] for key in ['wall_s', 'brain_s', 'render_s']}
    data['planner_trial'] = {seed: json.loads((args.runs / f'{seed}_planner/report.json').read_text())
                             for seed in map(str, range(5001, 5005))}
    # Drop large room and pose details from the tracked evidence. Full reports stay in the run folders.
    from headless_compare import summary
    data['planner_trial'] = {seed: summary(report) for seed, report in data['planner_trial'].items()}
    (args.out / 'measurements.json').write_text(json.dumps(data, indent=2) + '\n')
    for name in ['logic_final.txt', 'pytest_final.txt', 'team_final.txt', 'reverts.log', 'before.log', 'final.log', 'noise_final.log']:
        shutil.copyfile(args.runs / name, args.out / name)
    shutil.copyfile(comparison / 'compare.json', args.out / 'headless_compare.json')
    for path in (args.runs / 'reverts').glob('*.txt'):
        (args.out / 'reverts').mkdir(exist_ok=True)
        shutil.copyfile(path, args.out / 'reverts' / path.name)
    render_beliefs(args.runs, args.out)


if __name__ == '__main__':
    main()
