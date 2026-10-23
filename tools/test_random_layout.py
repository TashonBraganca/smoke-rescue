"""World config tests. No Unity or model downloads are needed."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

MODULE = Path(__file__).with_name("random_layout.py")


def generator():
    assert MODULE.exists(), "The world-side floor-plan generator is missing"
    spec = importlib.util.spec_from_file_location("random_layout", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_seed_reproduces_all_three_configs():
    g = generator()
    assert g.generate(5001) == g.generate(5001)
    assert g.generate(5001) != g.generate(5002)


@pytest.mark.parametrize("seed", range(5001, 5051))
def test_fifty_unseen_houses(seed):
    g = generator()
    plan, fire, scenario = g.generate(seed)
    g.validate(plan, fire, scenario)
    assert 7 <= len(plan["rooms"]) <= 12
    assert 12 <= plan["footprint"][2] <= 24
    assert 10 <= plan["footprint"][3] <= 18
    assert len(scenario["survivors"]) == 5
    assert all(r["id"].startswith("r") for r in plan["rooms"])
    assert scenario["survivor_placement"] == "random"
    assert all(r["control"] == "autonomous" and not r["route"] for r in scenario["robots"])


@pytest.mark.parametrize("damage", ["overlap", "disconnected", "door", "furniture", "person", "fire", "hole"])
def test_rejects_broken_world(damage):
    g = generator()
    plan, fire, sc = copy.deepcopy(g.generate(5001))
    if damage == "overlap":
        plan["rooms"][1]["rect"] = plan["rooms"][0]["rect"][:]
    elif damage == "disconnected":
        plan["doors"] = [d for d in plan["doors"] if "outside" in d["between"]]
    elif damage == "door":
        plan["doors"][0]["width"] = 0.4
    elif damage == "furniture":
        lm = plan["landmarks"][0]
        room = next(r for r in plan["rooms"] if r["id"] == lm["room"])
        lm["at"] = [(room["rect"][0] + room["rect"][2]) / 2,
                    (room["rect"][1] + room["rect"][3]) / 2]
    elif damage == "person":
        sc["survivors"][0]["at"] = [-100, -100]
    elif damage == "fire":
        fire["ignitions"][0]["room"] = "missing"
    else:
        plan["rooms"][0]["rect"][2] -= 0.3
    with pytest.raises(ValueError):
        g.validate(plan, fire, sc)


def test_corridor_and_open_living_families_occur():
    g = generator()
    plans = [g.generate(s)[0] for s in range(5001, 5051)]
    hall = [p for p in plans if any(r["kind"] == "corridor" for r in p["rooms"])]
    assert 25 < len(hall) < 50
    for p in hall:
        rid = next(r["id"] for r in p["rooms"] if r["kind"] == "corridor")
        assert sum(rid in d["between"] and "outside" not in d["between"] for d in p["doors"]) >= 3


def test_cli_writes_scenario_selected_files(tmp_path):
    g = generator()
    g.write(5001, tmp_path)
    sc = json.loads((tmp_path / "scenario_random_5001.json").read_text())
    assert (tmp_path / sc["floorplan_file"]).is_file()
    assert (tmp_path / sc["fire"]["config_file"]).is_file()
