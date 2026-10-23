"""Run M1 regression controls, restoring every edited byte in a finally block."""
from pathlib import Path
import json
import os
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BRAIN = 'unity/SmokeRescue/Assets/SmokeRescue/Brain/'
TESTS = 'unity/SmokeRescue/Assets/SmokeRescue/Tests/EditMode/'
OUT = ROOT / 'data/runs/headless/m1/reverts'


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'expected one occurrence of {old!r}')
    return text.replace(old, new)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    env = os.environ | dict(PYTHONDONTWRITEBYTECODE='1', UV_CACHE_DIR='/private/tmp/m1-uv-cache',
                            UV_PROJECT_ENVIRONMENT='/Volumes/Tashon/opencv/smoke-rescue/brain/.venv')
    python = ['uv', 'run', '--no-sync', 'pytest', '-q']
    cases = [
        ('gyro', {BRAIN + 'DeadReckoning.cs': lambda _: subprocess.check_output(['git', 'show', 'b03cced:' + BRAIN + 'DeadReckoning.cs'], cwd=ROOT).decode()},
         ['tools/headless/test.sh', 'GyroIntervals'], ROOT, 'FAIL GyroIntervalsDoNotDependOnBrainStepPhase'),
        ('source_once', {BRAIN + 'TeamMap.cs': lambda s: replace_once(s, 'duplicateObservation = t <= lastCapture[sensor];', 'duplicateObservation = false;')},
         ['tools/headless/test.sh', 'TeamMapFusesEachSourceOnce'], ROOT, 'FAIL TeamMapFusesEachSourceOnce'),
        ('correction', {
            BRAIN + 'RobotBrain.cs': lambda s: replace_once(s, 'motion.ApplyCorrection(correction); teamMap.ApplyCorrection(correction);', 'motion.ApplyCorrection(correction);'),
            BRAIN + 'DistributedContext.cs': lambda s: replace_once(s, '            teamMap.ApplyCorrection(delta);', '            // Regression control omits shared evidence correction.')},
         ['tools/headless/test.sh', 'TeamMapRobotCorrections'], ROOT, 'FAIL TeamMapRobotCorrections'),
        ('compression', {BRAIN + 'BrainMessage.cs': lambda s: replace_once(s, 'System.IO.Compression.CompressionLevel.Fastest', 'System.IO.Compression.CompressionLevel.NoCompression')},
         ['tools/headless/test.sh', 'TeamMapLostBatches'], ROOT, 'FAIL TeamMapLostBatches'),
        ('motion_error', {TESTS + 'HeadlessMotionErrors.cs': lambda s: replace_once(s, 'return exactRate + (bias + noise * gaussian) * Mathf.Deg2Rad;', 'return exactRate;')},
         ['tools/headless/test.sh', 'HeadlessMotionErrors'], ROOT, 'FAIL HeadlessMotionErrors'),
        ('viewer', {'tools/showcase.py': lambda s: replace_once(s, '        if belief is not None:', '        if False and belief is not None:')},
         python + ['tests/test_team_map.py::test_live_maps_uses_one_receivers_export_without_reading_binary_pngs'], ROOT / 'brain', 'FAILED'),
        ('shared_frame', {'brain/brain/nav_eval.py': lambda s: replace_once(s, '    if receiver in pads and any(r != receiver for r in pads):', '    if False and receiver in pads and any(r != receiver for r in pads):')},
         python + ['tests/test_team_map.py::test_shared_pad_frame_is_reported_without_a_discovery_event'], ROOT / 'brain', 'FAILED'),
        ('map_metric', {'brain/brain/nav_eval.py': lambda s: replace_once(s, 'wall_share = float(np.mean(distances <= .2 + 1e-7))', 'wall_share = float(np.mean(distances <= 2.0))')},
         python + ['tests/test_team_map.py::test_team_accuracy_measures_wall_distance_and_camera_searched_floor'], ROOT / 'brain', 'FAILED'),
    ]
    summary = {}
    for name, edits, command, cwd, expected in cases:
        saved = {path: (ROOT / path).read_bytes() for path in edits}
        try:
            for path, edit in edits.items():
                (ROOT / path).write_text(edit(saved[path].decode()))
            result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True)
            output = result.stdout + result.stderr
            (OUT / (name + '.txt')).write_text(output)
            correct = result.returncode != 0 and expected in output and 'Build FAILED' not in output
            summary[name] = dict(expected_failure=correct, returncode=result.returncode)
            print(name, summary[name], flush=True)
            if not correct:
                raise RuntimeError(f'{name} did not produce its expected regression failure')
        finally:
            for path, content in saved.items():
                (ROOT / path).write_bytes(content)
            (OUT / 'summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
