"""Paired controls for deterministic camera tools."""
import json
from pathlib import Path
import numpy as np
import pytest
from brain.agent import vision as v
ROOT = Path(__file__).resolve().parents[2]

def scene(kind="body", t=0.):
    k = np.array([[120.,0,79.5],[0,120.,59.5],[0,0,1.]])
    depth = np.full((120,160),3.,np.float32)
    thermal = np.full((120,160),2932,np.uint16)
    yy,xx = np.mgrid[30:90,69:91]
    thermal[30:90,69:91] = np.round((303.15+5*(yy-30)/60)*10).astype(np.uint16)
    if kind != "wall":
        depth[30:90,69:91] -= .25*np.sin((xx-68)/24*np.pi)*np.sin((yy-29)/62*np.pi)
    if kind == "uniform": thermal[30:90,69:91] = 3072
    if kind == "window": depth[30:90,69:91] += 2.
    rgb = np.full((120,160,3),70,np.uint8)
    rgb[30:90,69:91] = 160
    return v.Frame(t=t,rgb=rgb,depth_m=depth,thermal=thermal,K=k,thermal_K=k,
                   rgb_to_thermal=np.eye(4),stamps={"rgb":t,"depth":t,"thermal":t})

def inspect(f,tmp_path):
    c = v.scan_thermal(f)["people"][0]
    return v.inspect_candidate(f,c["bearing_deg"],c["range_m"] or 3.,output_dir=tmp_path)

def test_planarity_controls_person_verdict(tmp_path):
    body,wall = inspect(scene(),tmp_path),inspect(scene("wall"),tmp_path)
    assert body["verdict"] == "person"
    assert wall["verdict"] == "not_person"
    assert wall["cues"]["plane"]["large_plane"]
    assert not body["cues"]["plane"]["large_plane"]
    assert Path(body["debug_crop"]).is_file()

def test_temperature_and_window_controls(tmp_path):
    assert inspect(scene(),tmp_path)["verdict"] == "person"
    assert inspect(scene("uniform"),tmp_path)["verdict"] != "person"
    window = inspect(scene("window"),tmp_path)
    assert window["verdict"] == "not_person"
    assert window["cues"]["window"]["opening"]

def test_size_control(tmp_path):
    assert inspect(scene(),tmp_path)["verdict"] == "person"
    f = scene(); f.thermal_K[:2,:2] *= 5
    assert inspect(f,tmp_path)["verdict"] != "person"

def test_expanding_fire_and_static_control():
    a,b,c = scene(t=0),scene(t=1),scene(t=2)
    a.thermal[:] = b.thermal[:] = c.thermal[:] = 2932
    a.thermal[45:65,65:85] = 5500
    b.thermal[40:70,60:90] = c.thermal[40:70,60:90] = 5500
    assert v.track_fire_spread([a,b])["growth_m2_s"] > 0
    assert v.track_fire_spread([b,c])["growth_m2_s"] == 0
    with pytest.raises(ValueError): v.track_fire_spread([b,a])

def test_opening_width_and_blocked_control():
    f = scene(); f.depth_m[:] = 2.; f.depth_m[:,50:110] = 5.
    result = v.measure_opening(f,0.,body_width_m=.6)
    assert result["clear_width_m"] == pytest.approx(1.,abs=.05)
    assert result["passable"]
    assert not v.measure_opening(f,0.,body_width_m=1.2)["passable"]
    f.depth_m[:] = 2.
    assert not v.measure_opening(f,0.)["passable"]
    f.depth_m[:] = 0
    assert not v.measure_opening(f,0.)["passable"]

def test_visibility_and_json_determinism(tmp_path):
    f = scene(); r = inspect(f,tmp_path)
    assert json.dumps(r,allow_nan=False,sort_keys=True) == json.dumps(inspect(f,tmp_path),allow_nan=False,sort_keys=True)
    for result in (f,v.scan_thermal(f),v.assess_visibility(f),v.measure_opening(f,0)):
        assert {"id","tool","t","evidence"} <= result.keys()
        json.dumps(result,allow_nan=False)
    f.depth_m[:] = 0; f.rgb[:] = 0
    assert not v.assess_visibility(f)["usable_rgb"]
    assert inspect(f,tmp_path)["verdict"] == "uncertain"

@pytest.mark.parametrize("t",[0.,10.,30.])
def test_real_frames(t,tmp_path):
    folder = ROOT / "data/recordings/unity_haze/robot_A"
    if not folder.exists(): pytest.skip("local recording absent")
    f = v.load_frame(folder,t)
    assert f is v.load_frame(folder,t)
    assert f.registered_depth_m.shape == f.thermal.shape
    assert f.rgb.shape == (480,640,3)
    for c in v.scan_thermal(f)["people"][:2]:
        r = v.inspect_candidate(f,c["bearing_deg"],c["range_m"] or 3.,output_dir=tmp_path)
        assert r["verdict"] in {"person","not_person","uncertain"}
    assert v.assess_visibility(f)["valid_depth_fraction"] > 0

def test_registration_offset_and_missing_depth():
    f = scene("wall")
    f.depth_m[:] = 0
    f.depth_m[60,90] = 3
    f.rgb_to_thermal[0,3] = -.1
    registered = f.registered_depth_m
    assert np.isfinite(registered[60,86])
    assert not np.isfinite(registered[60,90])
    points = __import__('cv2').depthTo3d(np.full((120,160),3,np.float32),f.K)
    assert points[60,90,0] > 0


def test_loader_rejects_sensor_path_escape(tmp_path):
    (tmp_path / 'rgb.txt').write_text('0 ../outside.png\n')
    with pytest.raises(ValueError): v._index(tmp_path,'rgb')


def test_evaluation_counts_misses_and_bootstraps():
    from brain.agent.eval_vision import summarize
    rows = [{'group':'a','label':True,'raw':True,'inspect':True},
            {'group':'b','label':True,'raw':False,'inspect':False},
            {'group':'c','label':False,'raw':True,'inspect':False}]
    r = summarize(rows)
    assert r['raw']['precision'] == .5
    assert r['raw']['recall'] == .5
    assert r['inspect']['precision'] == 1.
    assert r['inspect']['recall'] == .5
    assert r == summarize(rows)


@pytest.mark.parametrize("bbox,verdict", [([0, 50, 22, 64], "uncertain"), ([138, 50, 22, 64], "uncertain"),
                                          ([60, 30, 30, 70], "person"), ([60, 60, 30, 60], "person")])
def test_colour_detection_cut_by_the_image_side_cannot_confirm(bbox, verdict, tmp_path, monkeypatch):
    # Held-out seed 914 run 3: with no warm body at the bearing, a 22 px colour box cut by the left border (confidence
    # 0.58) confirmed a person 2.6 m from anyone. Cut by either side it stays uncertain; whole, or cut only at the
    # bottom (a person lying close), it still confirms.
    f = scene()
    f.thermal[:] = 2932                                   # 20 C everywhere: no warm component
    hit = {"confidence": 0.58, "bbox_rgb": bbox, "bearing_deg": 0.0, "range_m": 3.0}
    monkeypatch.setattr(v, "detect_people_rgb", lambda frame, **k: [hit])
    r = v.inspect_candidate(f, 0.0, 3.0, output_dir=tmp_path)
    assert r["verdict"] == verdict
    assert r["cues"]["rgb_dnn"]["clipped_by_side"] == (1 if verdict == "uncertain" else 0)



def test_a_colour_person_beside_the_selected_warm_body_confirms_it(tmp_path, monkeypatch):
    # Held-out seed 915, robot A (Codex replay): asked at 0 deg, the warm component lay at -6.9 deg and the colour
    # detector saw a person at -8.9 deg, just outside 8 deg of the request, so the verdict stayed uncertain. Here the
    # warm body (uniform temperature, so thermal alone is uncertain) sits near 0 deg, the request is at +6 deg and the
    # colour person at -7 deg: 13 deg from the request, 7 deg from the body. Control: no warm body to associate with.
    f = scene("uniform")
    monkeypatch.setattr(v, "detect_people_rgb", lambda frame, **k: [{"confidence": 0.83, "bbox_rgb": [60, 30, 30, 70], "bearing_deg": -7.0, "range_m": 3.0}])
    assert v.inspect_candidate(f, 6.0, 3.0, output_dir=tmp_path)["verdict"] == "person"
    g = scene("uniform"); g.thermal[:] = 2932
    assert v.inspect_candidate(g, 6.0, 3.0, output_dir=tmp_path)["verdict"] == "uncertain", "control: nothing at -7 deg to tie it to"


def test_a_colour_only_confirmation_carries_the_measured_position(tmp_path, monkeypatch):
    # Codex review: with no warm body matched, a colour-only person verdict left candidate empty, so the tool layer used
    # the requested bearing and range as the person's position. It now carries the detector's measured geometry.
    f = scene(); f.thermal[:] = 2932
    monkeypatch.setattr(v, "detect_people_rgb", lambda frame, **k: [{"confidence": 0.9, "bbox_rgb": [60, 30, 30, 70], "bearing_deg": 5.0, "range_m": 3.4}])
    r = v.inspect_candidate(f, 3.0, 3.0, output_dir=tmp_path)
    assert r["verdict"] == "person"
    assert (r["candidate"]["bearing_deg"], r["candidate"]["range_m"]) == (5.0, 3.4)
