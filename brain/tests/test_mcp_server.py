"""The MCP server, run in-process through the official SDK client, on committed frames from a recorded run.

data/fixtures/mcp_frames/SOURCE.txt says where each frame comes from and what it shows (checked by eye)."""
import asyncio
import base64
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from mcp import Client

from brain import mcp_server

FIX = Path(__file__).resolve().parents[2] / "data" / "fixtures" / "mcp_frames"
PERSON_RGB, PERSON_DEPTH, PERSON_THERMAL = (str(FIX / f"person_{k}.png") for k in ("rgb", "depth", "thermal"))
TURN_RGB, TURN_DEPTH = str(FIX / "turn_rgb.png"), str(FIX / "turn_depth.png")
FIRE_RGB, FIRE_DEPTH, FIRE_THERMAL = (str(FIX / f"fire_{k}.png") for k in ("rgb", "depth", "thermal"))
# The robot's own leg odometry from turn_* (31.0 s) to person_* (31.27 s): SOURCE.txt.
ODOM_FORWARD_M, ODOM_RIGHT_M, ODOM_YAW_DEG = 0.064, 0.004, -10.3


def call(name, args, server=None):
    async def run():
        async with Client(server or mcp_server.mcp) as client:
            return await client.call_tool(name, args)
    result = asyncio.run(run())
    assert not result.is_error, result.content
    return result.structured_content


@pytest.fixture(autouse=True)
def local_backend(monkeypatch):
    monkeypatch.setenv("SMOKE_BACKEND", "local")


def test_the_server_lists_its_four_tools():
    async def run():
        async with Client(mcp_server.mcp) as client:
            return await client.list_tools()
    tools = {t.name: t for t in asyncio.run(run()).tools}
    assert set(tools) == {"detect_people", "assess_visibility", "match_place", "describe_scene_heat"}
    assert tools["detect_people"].input_schema["required"] == ["image"]


def test_detect_people_finds_the_person_on_the_floor():
    out = call("detect_people", {"image": PERSON_RGB, "depth": PERSON_DEPTH})
    assert out["count"] >= 1
    best = max(out["people"], key=lambda p: p["confidence"])
    x, y, w, h = best["bbox_rgb"]
    # The person lies at the bottom right of the frame, about 2 m away (the robot's own run measured 2.06 m).
    assert best["confidence"] >= 0.5
    assert 380 <= x + w / 2 <= 600 and 360 <= y + h / 2 <= 480
    assert best["range_source"] == "depth" and 1.5 <= best["range_m"] <= 2.6
    assert best["floor_gap_m"] is not None and best["floor_gap_m"] < 0.3


def test_detect_people_finds_no_one_in_an_empty_corridor():
    assert call("detect_people", {"image": FIRE_RGB, "depth": FIRE_DEPTH})["people"] == []


def test_assess_visibility_gives_real_numbers_and_rejects_a_blank_frame():
    out = call("assess_visibility", {"image": PERSON_RGB, "depth": PERSON_DEPTH})
    for key in ("laplacian_variance", "rms_contrast", "dark_channel_haze", "valid_depth_fraction"):
        assert out[key] is not None and math.isfinite(out[key]), key
    assert out["laplacian_variance"] > 20 and out["rms_contrast"] > 12 and out["usable_rgb"] is True
    assert 0.5 < out["valid_depth_fraction"] <= 1.0
    grey = base64.b64encode(cv2.imencode(".png", np.full((480, 640, 3), 128, np.uint8))[1].tobytes()).decode()
    blank = call("assess_visibility", {"image": grey})
    assert blank["laplacian_variance"] == 0 and blank["usable_rgb"] is False
    assert blank["valid_depth_fraction"] is None


def test_match_place_recovers_the_turn_between_two_views():
    out = call("match_place", {"keyframe_a": {"rgb": TURN_RGB, "depth": TURN_DEPTH},
                               "keyframe_b": {"rgb": PERSON_RGB, "depth": PERSON_DEPTH}, "return_payloads": True})
    assert out["matched"] is True and out["inliers"] >= 10
    assert abs(out["dyaw_deg"] - ODOM_YAW_DEG) < 2.0
    assert abs(out["dy_m"] - ODOM_FORWARD_M) < 0.05 and abs(out["dx_m"] - ODOM_RIGHT_M) < 0.05
    # The same match from the radio payloads a robot would send instead of images.
    again = call("match_place", {"keyframe_a": out["payload_b64"]["a"], "keyframe_b": out["payload_b64"]["b"]})
    assert again["matched"] is True and again["inliers"] == out["inliers"]
    assert abs(again["dyaw_deg"] - out["dyaw_deg"]) < 1e-6


def test_match_place_refuses_two_different_places():
    out = call("match_place", {"keyframe_a": {"rgb": TURN_RGB, "depth": TURN_DEPTH},
                               "keyframe_b": {"rgb": FIRE_RGB, "depth": FIRE_DEPTH}})
    assert out["matched"] is False and "inliers" in out["reason"]


def test_describe_scene_heat_finds_the_fire_and_nothing_hot_near_the_person():
    fire = call("describe_scene_heat", {"thermal": FIRE_THERMAL})
    assert len(fire["hotspots"]) >= 1 and fire["max_c"] > 300
    hottest = max(fire["hotspots"], key=lambda r: r["max_c"])
    assert hottest["bbox"][0] < 40 and hottest["pixels"] > 1000      # the doorway fire on the left of the view
    calm = call("describe_scene_heat", {"thermal": PERSON_THERMAL})
    assert calm["hotspots"] == [] and calm["max_c"] < 40


def test_remote_backend_calls_the_http_routes_with_the_same_frames(monkeypatch):
    from fastapi.testclient import TestClient
    from brain.agent.server import app
    monkeypatch.setenv("SMOKE_API_URL", "http://testserver")
    monkeypatch.setattr(mcp_server, "_client", lambda: TestClient(app, base_url="http://testserver"))
    people = call("detect_people", {"image": PERSON_RGB, "depth": PERSON_DEPTH, "backend": "remote"})
    local = call("detect_people", {"image": PERSON_RGB, "depth": PERSON_DEPTH})
    assert people["backend"] == "remote" and people["url"] == "http://testserver/perceive/people"
    assert [p["bbox_rgb"] for p in people["people"]] == [p["bbox_rgb"] for p in local["people"]]
    vis = call("assess_visibility", {"image": PERSON_RGB, "backend": "remote"})
    assert vis["laplacian_var"] > 20 and vis["rms_contrast"] > 12
    heat = call("describe_scene_heat", {"thermal": FIRE_THERMAL, "backend": "remote"})
    assert len(heat["hot_regions"]) >= 1
