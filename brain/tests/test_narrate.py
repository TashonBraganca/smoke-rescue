"""Narration contracts, with inference checks enabled only on a prepared Mac."""

import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

from brain import narrate


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "data/fixtures/tiny_2robots"


@pytest.mark.parametrize("key", ["oracle", "OraclePath", "true_pose", "true_x", "groundtruth", "ground_truth", "floorplan"])
@pytest.mark.parametrize("nested", [False, True])
def test_prompt_rejects_untrusted_context_keys(key, nested):
    context = {"event": "periodic", key: "hidden data"}
    if nested:
        context = {"event": "periodic", "reason": [context]}
    with pytest.raises(ValueError, match="context key"):
        narrate.build_prompt(context)


def test_prompt_contains_event_and_camera_only_instructions():
    prompt = narrate.build_prompt({"robot": "robot_A", "t": 48.2, "event": "sensor_switch",
                                   "from": "rgb", "to": "thermal", "reason": "61 corners"})
    for text in ["sensor_switch", "61 corners", "25 words", "present tense", "visible"]:
        assert text in prompt


@pytest.mark.parametrize("raw", ["A wall is ahead.\nA chair is nearby.",
                                     "The camera sees " + "objects " * 60,
                                     '"A door is open! A person might be inside."'])
def test_output_is_one_sentence_under_word_limit(raw):
    text = narrate.clean_sentence(raw)
    assert text and "\n" not in text and "\r" not in text
    assert len(text.split()) <= 25
    assert text[-1] in ".!?"
    assert not any(c in text[:-1] for c in ".!?")


def test_empty_model_output_is_an_error():
    with pytest.raises(ValueError, match="empty"):
        narrate.clean_sentence(" \n")


def test_describe_normalizes_generated_result(monkeypatch):
    narrator = narrate.Narrator.__new__(narrate.Narrator)
    narrator.supports_multiple_images = True

    def generate(images, prompt):
        assert len(images) == 1 and images[0].mode == "RGB"
        assert "periodic" in prompt
        return "A wall is ahead.\nThe robot sees another wall."

    monkeypatch.setattr(narrator, "_generate", generate)
    assert narrator.describe(np.zeros((8, 8, 3), np.uint8), {"event": "periodic"}) == "A wall is ahead."


@pytest.mark.parametrize("supports_two", [False, True])
def test_thermal_is_a_second_image_only_for_supported_models(monkeypatch, supports_two):
    narrator = narrate.Narrator.__new__(narrate.Narrator)
    narrator.supports_multiple_images = supports_two

    def generate(images, prompt):
        assert len(images) == (2 if supports_two else 1)
        if supports_two:
            assert images[1].mode == "RGB"
            assert "false-colour thermal" in prompt
        return "A wall fills the view."

    monkeypatch.setattr(narrator, "_generate", generate)
    thermal = np.full((8, 8), 3100, np.uint16)
    assert narrator.describe(np.zeros((8, 8, 3), np.uint8), {"event": "periodic"}, thermal=thermal)


def test_thermal_has_fixed_temperature_scale_and_black_invalid_pixels():
    rgb = narrate.false_colour(np.array([[0, 2732, 3100, 3732]], np.uint16))
    assert rgb.shape == (1, 4, 3) and rgb.dtype == np.uint8
    assert np.all(rgb[0, 0] == 0)
    assert not np.array_equal(rgb[0, 1], rgb[0, 3])
    with pytest.raises(ValueError):
        narrate.false_colour(np.zeros((2, 2), np.uint8))


def test_cli_writes_fixture_jsonl_with_all_events_and_no_future_frame(tmp_path, monkeypatch, capsys):
    output = tmp_path / "run"
    a = output / "robot_A"
    a.mkdir(parents=True)
    (a / "decisions.jsonl").write_text(json.dumps({"t": 0.1, "event": "sensor_switch", "from": "rgb", "to": "thermal"}) + "\n")
    (a / "survivors.json").write_text(json.dumps([{"id": "A0", "first_seen_t": 0.2,
                                                 "position_map": [1, 2, 3], "n_frames": 3}]))
    narrator = narrate.Narrator.__new__(narrate.Narrator)
    narrator.supports_multiple_images = True
    monkeypatch.setattr(narrator, "_generate", lambda images, prompt: "A room is visible.")
    monkeypatch.setattr(narrate, "Narrator", lambda model_id, device: narrator)
    narrate.main([str(FIXTURE), str(output), "--every", "0.2"])
    records = {r: [json.loads(line) for line in (output / r / "narration.jsonl").read_text().splitlines()]
               for r in ["robot_A", "robot_B"]}
    assert [r["event"] for r in records["robot_A"]] == ["periodic", "sensor_switch", "periodic", "survivor_seen", "periodic"]
    assert [r["t"] for r in records["robot_B"]] == pytest.approx([34.651745, 34.851745, 35.051745])
    assert "warning" not in capsys.readouterr().err.lower()
    for robot, rows in records.items():
        for row in rows:
            assert set(row) == {"t", "robot", "event", "text", "latency_s"}
            assert row["robot"] == robot and row["latency_s"] >= 0
    reader = narrate.read_associations(FIXTURE / "robot_A/associations.txt")
    jobs = narrate.narration_jobs(a, "robot_A", reader, 0.2)
    assert all(reader[index].rgb_timestamp <= context["t"] for index, context in jobs)
    assert any(context.get("frame_t") == 0.09999 for _, context in jobs if context["event"] == "sensor_switch")


@pytest.mark.parametrize("interval", [0, -1, float("nan"), float("inf")])
def test_bad_intervals_fail_before_loading_model(tmp_path, interval):
    with pytest.raises(ValueError, match="positive"):
        narrate.run(FIXTURE, tmp_path, every=interval)


def test_event_from_another_robot_is_rejected(tmp_path):
    (tmp_path / "decisions.jsonl").write_text(json.dumps({"t": 0.1, "robot": "robot_B", "event": "replan"}) + "\n")
    associations = narrate.read_associations(FIXTURE / "robot_A/associations.txt")
    with pytest.raises(ValueError, match="robot"):
        narrate.narration_jobs(tmp_path, "robot_A", associations, 10)


def test_missing_model_fails_without_replacing_previous_narration(tmp_path, monkeypatch):
    output = tmp_path / "run/robot_A/narration.jsonl"
    output.parent.mkdir(parents=True)
    output.write_text("previous completed run\n")
    monkeypatch.setattr(narrate, "HF_HOME", narrate.MODEL_ROOT / "absent-test-cache")
    with pytest.raises(FileNotFoundError, match="local VLM files absent"):
        narrate.run(FIXTURE, tmp_path / "run")
    assert output.read_text() == "previous completed run\n"


def test_path_cannot_leave_robot_camera_directory(tmp_path):
    (tmp_path / "rgb").mkdir()
    (tmp_path / "rgb/link.png").symlink_to(FIXTURE / "robot_A/rgb/0.000000.png")
    with pytest.raises(ValueError, match="camera directory"):
        narrate.camera_path(tmp_path, Path("rgb/link.png"), "rgb")
    with pytest.raises(ValueError, match="camera directory"):
        narrate.camera_path(tmp_path, Path("../other/rgb.png"), "rgb")


@pytest.fixture(scope="module")
def local_model():
    if os.environ.get("SMOKE_RESCUE_VLM") != "1":
        pytest.skip("set SMOKE_RESCUE_VLM=1 with local weights and Metal available")
    try:
        path = narrate.cached_model_path(narrate.DEFAULT_MODEL)
    except FileNotFoundError:
        pytest.skip("local VLM weights are absent; tests never download models")
    # A failed native MLX import can leave the process unable to import it again.
    # Probe in a child process so an unavailable GPU cannot abort pytest.
    try:
        probe = subprocess.run([sys.executable, "-c", "import mlx.core as mx; assert mx.metal.is_available()"],
                               capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        pytest.skip("MLX/Metal availability probe timed out")
    if probe.returncode:
        reason = probe.stderr.strip().splitlines()
        pytest.skip("MLX/Metal unavailable: " + (reason[-1] if reason else f"probe exited {probe.returncode}"))
    return path


def test_model_repeatable_short_description(local_model):
    narrator = narrate.Narrator(str(local_model), "mps")
    rgb = cv2.cvtColor(cv2.imread(str(FIXTURE / "robot_A/rgb/0.000000.png")), cv2.COLOR_BGR2RGB)
    context = {"event": "periodic", "robot": "robot_A", "t": 0.0}
    first = narrator.describe(rgb, context)
    assert narrator.describe(rgb, context) == first
    assert first and len(first.split()) <= 25 and "\n" not in first
    assert narrator.describe(rgb, context, thermal=np.full(rgb.shape[:2], 3000, np.uint16))


def test_model_cli_on_fixture(local_model, tmp_path):
    result = subprocess.run([sys.executable, "-m", "brain.narrate", str(FIXTURE), str(tmp_path), "--every", "10"],
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr
    for robot in ["robot_A", "robot_B"]:
        rows = [json.loads(line) for line in (tmp_path / robot / "narration.jsonl").read_text().splitlines()]
        assert len(rows) == 1
        assert rows[0]["t"] == (0.0 if robot == "robot_A" else 34.651745)
        assert rows[0]["text"] and rows[0]["latency_s"] > 0
