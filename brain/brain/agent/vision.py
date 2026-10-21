"""Deterministic OpenCV 5 camera tools.

Frame is a JSON dictionary of metadata with image arrays held as attributes.
Lengths are metres, depth is optical z, and bearings are positive to the right.
Area is projected pixel footprint, not reconstructed curved surface area.
Cached recordings are treated as immutable; clear load_frame.cache_clear() after
replacing files. No tool infers world coordinates or reads scene metadata.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
import threading
from pathlib import Path

import cv2
import numpy as np

from brain.io import load_opencv_yaml, read_camera, required_number

cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)


def _perception():
    return json.loads((Path(__file__).resolve().parents[3] / "config/perception.json").read_text())


@dataclass(frozen=True)
class LidarScan:
    """One body-frame sweep. Angles turn left; ranges are metres from the scanner."""
    t: float
    angle_min_rad: float
    increment_rad: float
    mount_forward_m: float
    min_m: float
    max_m: float
    ranges_m: np.ndarray


def _number(x):
    return round(float(x), 6) if np.isfinite(x) else None


def _result(tool, frame_stamps, params, **values):
    payload = json.dumps([tool, frame_stamps, params], sort_keys=True, allow_nan=False, separators=(",", ":"))
    return {"id": hashlib.sha256(payload.encode()).hexdigest()[:16], "tool": tool,
            "t": frame_stamps[-1]["thermal"], "evidence": [], **values}


def _base(tool, frame, params, **values):
    return _result(tool, [frame["stamps"]], {"frame_id": frame["id"], **params}, **values)


class Frame(dict):
    """JSON metadata plus RGB uint8, depth float32 metres and thermal uint16 arrays.

    Synthetic callers may construct Frame directly. RGB uses RGB channel order.
    rgb_to_thermal maps RGB optical points into thermal optical coordinates.
    The default identity calibration is useful for synthetic registered frames.
    lidar optionally holds a timestamped sweep; inspection leaves these sensor arrays unchanged.
    """
    def __init__(self, *, t, rgb, depth_m, thermal, K, thermal_K,
                 rgb_to_thermal, stamps, source="synthetic", requested_t=None, lidar=None):
        if not math.isfinite(t) or not all(math.isfinite(x) for x in stamps.values()):
            raise ValueError("timestamps must be finite")
        self.rgb = np.asarray(rgb)
        self.depth_m = np.asarray(depth_m, dtype=np.float32)
        self.thermal = np.asarray(thermal)
        self.K = np.array(K, dtype=np.float64, copy=True)
        self.thermal_K = np.array(thermal_K, dtype=np.float64, copy=True)
        self.rgb_to_thermal = np.asarray(rgb_to_thermal, dtype=np.float64)
        self.lidar = lidar
        if (self.rgb.dtype != np.uint8 or self.rgb.ndim != 3 or self.rgb.shape[2] != 3
                or self.depth_m.shape != self.rgb.shape[:2]
                or self.thermal.dtype != np.uint16 or self.thermal.ndim != 2):
            raise ValueError("expected RGB uint8 HxWx3, matching depth, and thermal uint16 HxW")
        for k in (self.K, self.thermal_K):
            if k.shape != (3,3) or not np.isfinite(k).all() or k[0,0] <= 0 or k[1,1] <= 0:
                raise ValueError("invalid intrinsics")
        if self.rgb_to_thermal.shape != (4,4) or not np.isfinite(self.rgb_to_thermal).all():
            raise ValueError("invalid camera transform")
        # Including bytes prevents collisions between different synthetic scenes.
        digest = hashlib.sha256()
        for a in (self.rgb,self.depth_m,self.thermal,self.K,self.thermal_K,self.rgb_to_thermal):
            digest.update(a.tobytes())
        if lidar is not None:
            digest.update(lidar.ranges_m.tobytes())
            digest.update(str((lidar.t, lidar.angle_min_rad, lidar.increment_rad,
                               lidar.mount_forward_m, lidar.min_m, lidar.max_m)).encode())
        super().__init__(_result("load_frame", [stamps], {"source":source,"content":digest.hexdigest(),
            "requested_t":requested_t}, stamps=dict(stamps), source=source,
            intrinsics={"rgb":self.K.tolist(),"thermal":self.thermal_K.tolist()},
            rgb_to_thermal=self.rgb_to_thermal.tolist(),
            evidence=["Depth registered from RGB-D into thermal optical coordinates; invalid depth remains missing."]))

    @property
    def registered_depth_m(self):
        # Recompute for synthetic callers that alter their arrays; recording Frames are immutable.
        # cv2.registerDepth in opencv-python 5.0.0.93 dies with SIGBUS on any input (NaN or zero invalid depth,
        # measured 2026-10-20), so the same pinhole registration is done here: back-project, transform, z-buffer.
        return register_depth(self.depth_m, self.K, self.thermal_K, self.rgb_to_thermal,
                              (self.thermal.shape[0], self.thermal.shape[1]))


def register_depth(depth_m, K, K_out, transform, out_shape):
    """Depth (metres, NaN or <=0 invalid) from the RGB-D camera into another pinhole camera; nearest surface wins."""
    h, w = out_shape
    v, u = np.nonzero(np.isfinite(depth_m) & (depth_m > 0))
    z = depth_m[v, u].astype(np.float64)
    pts = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z, np.ones_like(z)])
    p = (transform @ pts)[:3]
    front = p[2] > 1e-6
    p = p[:, front]
    uo = np.floor(K_out[0, 0] * p[0] / p[2] + K_out[0, 2] + .5).astype(np.int64)
    vo = np.floor(K_out[1, 1] * p[1] / p[2] + K_out[1, 2] + .5).astype(np.int64)
    inside = (uo >= 0) & (uo < w) & (vo >= 0) & (vo < h)
    flat = np.full(h * w, np.inf)
    np.minimum.at(flat, vo[inside] * w + uo[inside], p[2, inside])
    out = flat.reshape(h, w).astype(np.float32)
    out[~np.isfinite(out)] = np.nan
    return out


def _safe_file(folder, relative, sensor=None):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("recording path must stay inside the robot folder")
    if sensor is not None and (len(relative.parts) != 2 or relative.parts[0] != sensor or relative.suffix != ".png"):
        raise ValueError("image must belong to the requested sensor")
    path = folder / relative
    if not path.resolve().is_relative_to(folder.resolve()):
        raise ValueError("recording symlink leaves the robot folder")
    return path


def _index(folder, sensor):
    # Key the cache on the index file's size and mtime: during a live recording the recorder rewrites it before every
    # agent step, and a plain cache served the first index forever, so every later step saw the frame at t = 0.
    st = _safe_file(folder, sensor + ".txt").stat()
    return _index_cached(folder, sensor, st.st_mtime_ns, st.st_size)


@lru_cache(maxsize=24)
def _index_cached(folder, sensor, _mtime_ns, _size):
    rows = []
    for line in _safe_file(folder,sensor+".txt").read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"): continue
        stamp,relative = line.split()
        stamp = float(stamp)
        if not math.isfinite(stamp): raise ValueError("nonfinite timestamp")
        rows.append((stamp,_safe_file(folder,relative,sensor)))
    if not rows or any(b[0] <= a[0] for a,b in zip(rows,rows[1:])):
        raise ValueError("sensor timestamps must increase")
    return rows


@lru_cache(maxsize=24)
def _lidar_index(folder, _mtime_ns, _size):
    header, rows = None, []
    for line in _safe_file(folder, "lidar.txt").read_text().splitlines():
        if line.startswith("# beams "):
            fields = line[2:].split()
            header = dict(zip(fields[::2], map(float, fields[1::2])))
            if (not all(math.isfinite(value) for value in header.values())
                    or header["beams"] < 1 or not header["beams"].is_integer()
                    or header["increment_rad"] <= 0
                    or not 0 < header["min_m"] < header["max_m"]):
                raise ValueError("invalid LiDAR calibration")
        elif line.strip() and not line.lstrip().startswith("#"):
            if header is None:
                raise ValueError("missing LiDAR calibration header")
            values = np.asarray(line.split(), dtype=np.float64)
            if (values.size != int(header["beams"]) + 1 or not np.isfinite(values).all()
                    or (values[1:] < 0).any()):
                raise ValueError("invalid LiDAR sweep")
            ranges = values[1:] / 1000
            ranges.flags.writeable = False
            rows.append(LidarScan(float(values[0]), header["angle_min_rad"], header["increment_rad"],
                                  header["mount_forward_m"], header["min_m"], header["max_m"], ranges))
    if any(b.t <= a.t for a, b in zip(rows, rows[1:])):
        raise ValueError("LiDAR timestamps must increase")
    return rows


def _load_lidar(folder, t, tolerance):
    try:
        stat = _safe_file(folder, "lidar.txt").stat()
    except FileNotFoundError:
        return None
    scans = _lidar_index(folder, stat.st_mtime_ns, stat.st_size)
    scan = min(scans, key=lambda scan: (abs(scan.t - t), scan.t), default=None)
    return scan if scan is not None and abs(scan.t - t) <= tolerance else None


@lru_cache(maxsize=16)
def load_frame(robot_dir: str | Path, t: float) -> Frame:
    """Load nearest stamps, ties earlier. Cache 16 frames and 24 sensor indexes.

    T_rgb_thermal.translation_m is the thermal camera centre in RGB coordinates.
    A camera mounted right by +0.03 m needs point transform x_thermal=x_rgb-0.03.
    No dilation is used since it would invent depth across opening boundaries.
    """
    if not math.isfinite(t): raise ValueError("t must be finite")
    folder = Path(robot_dir).resolve()
    tolerance = _perception()["frame_tolerance_s"]
    cam = read_camera(_safe_file(folder,"camera.yaml"))
    settings = load_opencv_yaml(_safe_file(folder,"thermal.yaml"))
    kt = np.array([[required_number(settings,"Camera1.fx","Camera.fx","fx"),0,required_number(settings,"Camera1.cx","Camera.cx","cx")],
                   [0,required_number(settings,"Camera1.fy","Camera.fy","fy"),required_number(settings,"Camera1.cy","Camera.cy","cy")],[0,0,1.]])
    k = np.array([[cam.fx,0,cam.cx],[0,cam.fy,cam.cy],[0,0,1.]])
    translation = settings.get("T_rgb_thermal.translation_m")
    if translation is None:
        translation = [required_number(settings,"mount_right_offset_m"),0.,0.]
    translation = np.asarray(translation,float)
    if translation.shape != (3,) or not np.isfinite(translation).all(): raise ValueError("invalid camera translation")
    transform = np.eye(4); transform[:3,3] = -translation
    stamps,images = {},{}
    for sensor in ("rgb","depth","thermal"):
        stamp,path = min(_index(folder,sensor),key=lambda row:(abs(row[0]-t),row[0]))
        if abs(stamp - t) > tolerance:
            raise ValueError(f"no {sensor} frame within {tolerance} s of t={t:.3f} (nearest {stamp:.3f}); the index is stale or incomplete")
        image = cv2.imread(str(path),cv2.IMREAD_COLOR if sensor == "rgb" else cv2.IMREAD_UNCHANGED)
        if image is None: raise ValueError(f"cannot read {path}")
        stamps[sensor],images[sensor] = stamp,image
    if images["depth"].dtype != np.uint16: raise ValueError("depth must be uint16")
    if images["depth"].shape != (cam.height,cam.width): raise ValueError("depth size differs from calibration")
    if images["thermal"].shape != (int(required_number(settings,"Camera.height","height")),int(required_number(settings,"Camera.width","width"))):
        raise ValueError("thermal size differs from calibration")
    lidar = _load_lidar(folder, stamps["thermal"], tolerance)
    if lidar is not None:
        stamps["lidar"] = lidar.t
    frame = Frame(t=stamps["thermal"],rgb=cv2.cvtColor(images["rgb"],cv2.COLOR_BGR2RGB),
        depth_m=images["depth"].astype(np.float32)/cam.depth_factor,thermal=images["thermal"],
        K=k,thermal_K=kt,rgb_to_thermal=transform,stamps=stamps,source=str(folder),requested_t=t,lidar=lidar)
    # Force registration during load and retain it for immutable cached recordings.
    frame._registered = frame.registered_depth_m
    for a in (frame.rgb,frame.depth_m,frame.thermal,frame.K,frame.thermal_K,frame._registered): a.flags.writeable = False
    return frame


def _depth(f):
    return f._registered if hasattr(f,"_registered") else f.registered_depth_m


def _lidar_component_depth(frame, mask, depth, tolerance):
    """Fill only missing warm pixels, using the pitched ray's horizontal bearing."""
    scan = frame.lidar
    if scan is None or abs(scan.t - frame["stamps"]["thermal"]) > tolerance:
        return depth, None
    yy, xx = np.nonzero(mask & ~(np.isfinite(depth) & (depth > 0)))
    if not len(xx):
        return depth, None
    sensors = json.loads((Path(__file__).resolve().parents[3] / "config/sensors.json").read_text())
    mount = sensors["camera"]["mount"]
    pitch = math.radians(mount["pitch_down_deg"])
    right = (xx - frame.thermal_K[0, 2]) / frame.thermal_K[0, 0]
    down = (yy - frame.thermal_K[1, 2]) / frame.thermal_K[1, 1]
    forward = math.cos(pitch) - down * math.sin(pitch)
    horizontal_length = np.hypot(right, forward)
    bearing = np.arctan2(-right, forward)
    beams = np.rint((bearing - scan.angle_min_rad) / scan.increment_rad).astype(int) % len(scan.ranges_m)
    ranges = scan.ranges_m[beams]
    horizontal = ranges - (mount["forward_offset_m"] - scan.mount_forward_m) * np.cos(bearing)
    valid = (np.isfinite(ranges) & (ranges >= scan.min_m) & (ranges <= scan.max_m)
             & (horizontal > 0) & (horizontal_length > 0))
    if not valid.any():
        return depth, None
    filled = depth.copy()
    filled[yy[valid], xx[valid]] = horizontal[valid] / horizontal_length[valid]
    # Keep each beam's actual return in the trace; the median alone could conceal a wall return.
    used = np.unique(beams[valid])
    return filled, {"t": scan.t, "beam_range_m": _number(np.median(ranges[valid])),
                    "beams": used.tolist(), "beam_ranges_m": scan.ranges_m[used].tolist(),
                    "filled_pixels": int(valid.sum())}


def _components(mask, min_pixels):
    n,labels,stats,_ = cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
    return [(labels == i,stats[i]) for i in range(1,n) if stats[i,cv2.CC_STAT_AREA] >= min_pixels]


def _region(f, mask, stats, depth):
    x,y,w,h,pixels = map(int,stats)
    z = depth[mask]; z = z[np.isfinite(z) & (z>0)]
    median = float(np.median(z)) if z.size else None
    ys,xs = np.nonzero(mask)
    u = float(np.mean(xs)); v = float(np.mean(ys))
    kt = f.thermal_K
    c = f.thermal[mask].astype(float)/10-273.15
    area = float(np.sum(z*z)/(kt[0,0]*kt[1,1])) if z.size else None
    return {"bbox":[x,y,w,h],"pixels":pixels,"depth_fraction":_number(z.size/pixels),
        "area_m2":_number(area) if area is not None else None,
        "range_m":_number(median) if median is not None else None,
        "bearing_deg":_number(np.degrees(np.arctan2(u-kt[0,2],kt[0,0]))),
        "height_m":_number(h*median/kt[1,1]) if median else None,
        "width_m":_number(w*median/kt[0,0]) if median else None,
        "aspect":_number(h/w),"mean_c":_number(c.mean()),"std_c":_number(c.std()),"max_c":_number(c.max())}


def _size_plausible(width, height):
    """Body-size gate shared by scan_thermal and inspect_candidate (metres, either orientation)."""
    return bool(width and height and .45 <= max(width, height) <= 2.2 and .15 <= min(width, height) <= 1.0
                and max(width, height) / min(width, height) <= 7)


def scan_thermal(frame: Frame, *, human_band_c=(28.,40.), hot_min_c=200., min_pixels=8) -> dict:
    """Return unconfirmed warm components and >=200 C projected hot footprints."""
    lo,hi = map(float,human_band_c)
    if not np.isfinite([lo,hi,hot_min_c]).all() or lo >= hi or min_pixels < 1: raise ValueError("invalid thermal thresholds")
    c = frame.thermal.astype(float)/10-273.15; valid = frame.thermal>0; depth = _depth(frame)
    people = [_region(frame,m,s,depth) for m,s in _components(valid & (c>=lo) & (c<=hi),min_pixels)]
    for p in people:
        p["score"] = _number(min(1.,p["pixels"]/100)*p["depth_fraction"])
        p["size_plausible"] = _size_plausible(p["width_m"],p["height_m"])
    hot = [_region(frame,m,s,depth) for m,s in _components(valid & (c>=hot_min_c),min_pixels)]
    return _base("scan_thermal",frame,{"human_band_c":[lo,hi],"hot_min_c":hot_min_c,"min_pixels":min_pixels},
        hotspots=hot,people=people,rgb_people=detect_people_rgb(frame),
        evidence=[f"{len(hot)} hot and {len(people)} warm connected components; warm candidates are not person confirmations.",
                  "rgb_people: OpenCV dnn YOLO11m person boxes on the colour image, a second sensor when thermal saturates."])


def _plane_cue(f, mask, depth, sigma_m=0.):
    """Planarity of the depth under a mask. sigma_m is the expected depth noise at the mask range."""
    points = cv2.depthTo3d(depth,f.thermal_K)
    normals = cv2.RgbdNormals.create(*depth.shape,cv2.CV_32F,f.thermal_K,5).apply(points)
    labels,coeff = cv2.findPlanes(points,normals,block_size=10,min_size=80,threshold=float(np.clip(2.5*sigma_m,.018,.15)))
    valid = mask & np.isfinite(depth)
    ids,counts = np.unique(labels[valid],return_counts=True)
    best = max(((int(count),int(i)) for i,count in zip(ids,counts) if i != 255),default=(0,255))
    overlap = best[0]/max(1,int(valid.sum()))
    plane_pixels = int(np.count_nonzero(labels == best[1])) if best[1]!=255 else 0
    xyz = points[valid,:3]
    residual = None
    if len(xyz)>=20:
        # Deterministic least-squares residual supplements the segmented plane overlap.
        xyz = xyz[::max(1,len(xyz)//2000)]
        centered = xyz-xyz.mean(axis=0)
        _,s,_ = np.linalg.svd(centered,full_matrices=False)
        residual = float(s[-1]/np.sqrt(len(xyz)))
    normal_values = normals[valid,:3]
    normal_values = normal_values[np.isfinite(normal_values).all(axis=1)]
    dispersion = float(np.linalg.norm(normal_values.std(axis=0))) if len(normal_values) else None
    return {"overlap_fraction":_number(overlap),"plane_pixels":plane_pixels,
        "large_plane":bool(overlap>=.90 and plane_pixels>=2*int(mask.sum())),
        "residual_m":_number(residual) if residual is not None else None,"noise_floor_m":_number(sigma_m),
        "normal_dispersion":_number(dispersion) if dispersion is not None else None,
        "planes":len(coeff) if coeff is not None else 0}


def _rgb_cue(f, mask, depth):
    # Project thermal region into RGB using measured depth, never a resized overlay.
    yy,xx = np.nonzero(mask & np.isfinite(depth))
    aligned = np.zeros(f.rgb.shape[:2],np.uint8)
    if len(xx):
        z = depth[yy,xx]
        points = np.column_stack(((xx-f.thermal_K[0,2])*z/f.thermal_K[0,0],(yy-f.thermal_K[1,2])*z/f.thermal_K[1,1],z))
        inv = np.linalg.inv(f.rgb_to_thermal)
        points = points@inv[:3,:3].T+inv[:3,3]
        uv = points@f.K.T
        u = np.rint(uv[:,0]/uv[:,2]).astype(int); v = np.rint(uv[:,1]/uv[:,2]).astype(int)
        ok = (u>=0)&(u<aligned.shape[1])&(v>=0)&(v<aligned.shape[0])
        aligned[v[ok],u[ok]] = 255
        aligned = cv2.dilate(aligned,np.ones((3,3),np.uint8))
    gray = cv2.createCLAHE(2.,(8,8)).apply(cv2.cvtColor(f.rgb,cv2.COLOR_RGB2GRAY))
    edges = cv2.Canny(gray,40,100)
    edges[aligned==0] = 0
    contours,_ = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    best = max(contours,key=cv2.contourArea,default=None)
    solidity = None
    if best is not None:
        hull = cv2.contourArea(cv2.convexHull(best))
        solidity = cv2.contourArea(best)/hull if hull>0 else None
    return {"solidity":_number(solidity) if solidity is not None else None,
            "edge_pixels":int(np.count_nonzero(edges)),"weak_cue":True},aligned


def _debug(f,mask,aligned,path):
    scaled = np.clip((f.thermal.astype(float)/10-273.15)*2.55,0,255).astype(np.uint8)
    thermal = cv2.applyColorMap(scaled,cv2.COLORMAP_INFERNO)
    contours,_ = cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(thermal,contours,-1,(0,255,0),1)
    rgb = cv2.cvtColor(f.rgb,cv2.COLOR_RGB2BGR)
    contours,_ = cv2.findContours(aligned,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rgb,contours,-1,(0,255,0),1)
    def crop(image,m):
        ys,xs = np.nonzero(m)
        if len(xs): image = image[max(0,ys.min()-12):ys.max()+13,max(0,xs.min()-12):xs.max()+13]
        return cv2.resize(image,(256,256),interpolation=cv2.INTER_NEAREST)
    path.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(path),np.hstack((crop(thermal,mask),crop(rgb,aligned)))):
        raise OSError("cannot write debug crop")


def inspect_candidate(frame: Frame, bearing_deg: float, range_m: float, *, output_dir: str | Path,
                      human_band_c=(28.,40.), bearing_tolerance_deg=8., range_tolerance_m=1.5,
                      depth_noise_mm_per_m2=1.5) -> dict:
    """Inspect nearest warm component. Output directory is required for evidence.

    Person needs body dimensions, temperature structure and nonplanar depth.
    Below the configured depth coverage, a matching LiDAR sweep fills missing
    warm pixels before these checks. Measured depth always takes precedence.
    A large plane or >=0.8 m background recess is a negative cue. Uniform
    temperature alone is uncertain because clothing can hide skin structure.
    Body size accepts standing or recumbent components, 0.45 to 2.2 m long.
    Depth noise grows as depth_noise_mm_per_m2 * z**2 (sensor spec, 1.5 mm per m squared). A region counts as
    nonplanar only when its plane-fit residual exceeds both 12 mm and 1.5 times that noise at its range,
    otherwise sensor noise at distance would make every wall patch look like a body.
    """
    if (not np.isfinite([bearing_deg,range_m,bearing_tolerance_deg,range_tolerance_m,depth_noise_mm_per_m2]).all()
            or range_m<=0 or bearing_tolerance_deg<=0 or range_tolerance_m<=0 or depth_noise_mm_per_m2<0):
        raise ValueError("invalid candidate location")
    params = {"bearing_deg":bearing_deg,"range_m":range_m,"human_band_c":list(human_band_c),
              "bearing_tolerance_deg":bearing_tolerance_deg,"range_tolerance_m":range_tolerance_m,
              "depth_noise_mm_per_m2":depth_noise_mm_per_m2,
              "output_dir":str(Path(output_dir))}
    result = _base("inspect_candidate",frame,params,verdict="uncertain",reasons=[],cues={},candidate=None,debug_crop=None)
    config = _perception()
    min_depth_fraction = config["inspection_min_depth_fraction"]
    depth = _depth(frame); c = frame.thermal.astype(float)/10-273.15
    options = []
    for mask,stats in _components((frame.thermal>0)&(c>=human_band_c[0])&(c<=human_band_c[1]),8):
        region = _region(frame,mask,stats,depth)
        da = abs(region["bearing_deg"]-bearing_deg)
        dr = abs(region["range_m"]-range_m) if region["range_m"] is not None else 0
        if da<=bearing_tolerance_deg and dr<=range_tolerance_m:
            options.append((da/bearing_tolerance_deg+dr/range_tolerance_m,mask,region))
    if not options:
        result["reasons"] = ["No warm component matches the requested bearing and depth."]
        mask = np.zeros(frame.thermal.shape,bool); aligned = np.zeros(frame.rgb.shape[:2],np.uint8)
    else:
        _,mask,region = min(options,key=lambda o:o[0])
        registered_fraction = region["depth_fraction"]
        lidar_cue = None
        if registered_fraction < min_depth_fraction:
            depth, lidar_cue = _lidar_component_depth(frame, mask, depth, config["frame_tolerance_s"])
            if lidar_cue is not None:
                x, y, w, h = region["bbox"]
                region = _region(frame, mask, (x, y, w, h, region["pixels"]), depth)
        result["candidate"] = region
        width,height = region["width_m"],region["height_m"]
        size_ok = _size_plausible(width,height)
        x0,y0,bw,bh = region["bbox"]
        truncated = bool(x0<=0 or y0<=0 or x0+bw>=frame.thermal.shape[1] or y0+bh>=frame.thermal.shape[0])
        # A mask clipped by the image border measures only part of the object, so it cannot confirm a body.
        # It can still be too big to be one: clipping only makes the true object larger.
        too_big = bool(width and height and (max(width,height)>2.2 or min(width,height)>1.0))
        contours,_ = cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        rect = cv2.minAreaRect(max(contours,key=cv2.contourArea))
        vals = c[mask]; spread = float(np.percentile(vals,90)-np.percentile(vals,10))
        temp_ok = bool(.35<=region["std_c"]<=4.5 and spread>=.8)
        sigma = depth_noise_mm_per_m2*1e-3*(region["range_m"] or 0.)**2
        plane = _plane_cue(frame,mask,depth,sigma)
        # A component whose rim touches both hotter and cooler pixels is a slope crossing the band, not an object.
        rim = (cv2.dilate(mask.astype(np.uint8),np.ones((5,5),np.uint8))>0)&~mask&(frame.thermal>0)
        rim_hot = float(np.mean(c[rim]>human_band_c[1])) if rim.sum()>=10 else 0.
        rim_cold = float(np.mean(c[rim]<human_band_c[0])) if rim.sum()>=10 else 0.
        ramp = bool(rim_hot>=.25 and rim_cold>=.25)
        zs = depth[mask & np.isfinite(depth)]
        depth_span = float(np.percentile(zs,95)-np.percentile(zs,5)) if len(zs)>=10 else None
        # One body sits at one range. A warm region spanning several surfaces is background, not a person.
        single_surface = bool(depth_span is not None and depth_span<=1.5)
        ring = (cv2.dilate(mask.astype(np.uint8),np.ones((21,21),np.uint8))>0)&~mask
        ring_z = depth[ring & np.isfinite(depth)]
        surround = float(np.median(ring_z)) if len(ring_z)>=20 else None
        recess = region["range_m"]-surround if region["range_m"] is not None and surround is not None else None
        window = bool(recess is not None and recess>=.8)
        rgb,aligned = _rgb_cue(frame,mask,depth)
        nonplanar = bool(plane["residual_m"] is not None and plane["residual_m"]>=max(.012,1.5*sigma))
        result["cues"] = {"range_source": "lidar" if lidar_cue is not None else "depth",
            "size":{"plausible":size_ok,"height_m":height,"width_m":width,"aspect":region["aspect"],
            "rotated_size_px":[_number(x) for x in rect[1]],"truncated_by_image_edge":truncated},
            "temperature":{"structured":temp_ok,"mean_c":region["mean_c"],"std_c":region["std_c"],"p90_p10_c":_number(spread),
                "rim_hot_fraction":_number(rim_hot),"rim_cold_fraction":_number(rim_cold),"band_crossing_ramp":ramp},
            "plane":plane,"window":{"opening":window,"surrounding_z_m":_number(surround) if surround is not None else None,
                "recess_m":_number(recess) if recess is not None else None},"rgb":rgb,
            "depth":{"valid_fraction":region["depth_fraction"],"registered_fraction":registered_fraction,"nonplanar":nonplanar,
                "span_m":_number(depth_span) if depth_span is not None else None,"single_surface":single_surface}}
        reasons = []
        if lidar_cue is not None:
            result["cues"]["lidar"] = lidar_cue
            reasons.append("Inspection uses LiDAR range for missing registered depth; the scan measures the horizontal plane, not vertical body shape.")
        size_negative = (not size_ok) and (not truncated or too_big)
        if region["depth_fraction"]<min_depth_fraction: reasons.append("Too little registered depth to judge shape.")
        elif plane["large_plane"] or window or size_negative:
            result["verdict"] = "not_person"
            if plane["large_plane"]: reasons.append("Warm pixels lie on a plane larger than the warm component.")
            if window: reasons.append("Warm region is at least 0.8 m behind its surrounding depth, consistent with an opening.")
            if size_negative: reasons.append("Metric dimensions fall outside the body-size bounds.")
        elif truncated: reasons.append("Warm component touches the image edge, so its size is clipped and cannot confirm a body.")
        elif ramp: reasons.append("Warm component is a slope between hotter and cooler surfaces, not a distinct body.")
        elif not single_surface: reasons.append("Warm component spans more than 1.5 m of depth, so it covers several surfaces rather than one body.")
        elif temp_ok and nonplanar:
            result["verdict"] = "person"
            reasons.append("Body-sized warm component has temperature structure and nonplanar depth without a background recess.")
        else: reasons.append("Temperature structure or nonplanar body shape is insufficient for confirmation.")
        result["reasons"] = reasons
    if result["verdict"] == "uncertain":
        # Second sensor: a confident RGB person detection at the same bearing and range. It never overrides a
        # not_person verdict from planarity, a window recess or impossible size.
        hits = [p for p in detect_people_rgb(frame) if abs(p["bearing_deg"] - bearing_deg) <= bearing_tolerance_deg
                and (p["range_m"] is None or abs(p["range_m"] - range_m) <= range_tolerance_m)]
        result["cues"]["rgb_dnn"] = {"model": PERSON_ONNX.name, "hits": hits}
        if hits and max(h["confidence"] for h in hits) >= 0.5:
            best = max(hits, key=lambda h: h["confidence"])
            result["verdict"] = "person"
            result["reasons"] = result["reasons"] + [f"OpenCV dnn person detector (YOLO11m) confidence {best['confidence']:.2f} "
                                                     f"at bearing {best['bearing_deg']:.1f} deg, range {best['range_m']} m."]
    path = Path(output_dir)/f"{result['id']}.png"
    _debug(frame,mask,aligned,path)
    result["debug_crop"] = str(path)
    result["evidence"] = result["reasons"]+[f"Thermal crop left, projected RGB crop right: {path}"]
    return result


PERSON_ONNX = Path(__file__).resolve().parents[3] / "data" / "models" / "onnx" / "yolo11m.onnx"
_net = None
_net_lock = threading.Lock()   # one cv2.dnn.Net is not safe to run from two server threads at once


def detect_people_rgb(frame: Frame, *, min_conf: float = 0.35, model: Path | None = None) -> list[dict]:
    """People in the RGB image with OpenCV 5's dnn module running YOLO11m (ONNX, letterboxed 640x640, class 0).

    Independent of thermal, so it still sees a person when the thermal camera saturates in a hot smoke layer, and it
    fails where RGB fails (thick smoke, darkness). Returns boxes with confidence, bearing and median box depth.
    """
    global _net
    path = Path(model) if model else PERSON_ONNX
    if not path.exists():
        return []
    img = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    s = 640 / max(h, w); nh, nw = int(round(h * s)), int(round(w * s)); top, left = (640 - nh) // 2, (640 - nw) // 2
    pad = np.full((640, 640, 3), 114, np.uint8)
    pad[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh))
    with _net_lock:
        if _net is None:
            _net = cv2.dnn.readNetFromONNX(str(path))
        _net.setInput(cv2.dnn.blobFromImage(pad, 1 / 255.0, (640, 640), swapRB=True))
        out = _net.forward()[0].T                               # 8400 x 84: cx cy w h, then 80 class scores
    conf = out[:, 4]
    keep = conf >= min_conf
    if not keep.any():
        return []
    b = out[keep, :4]; conf = conf[keep]
    boxes = [[float((cx - bw / 2 - left) / s), float((cy - bh / 2 - top) / s), float(bw / s), float(bh / s)] for cx, cy, bw, bh in b]
    idx = cv2.dnn.NMSBoxes(boxes, conf.tolist(), min_conf, 0.45)
    people = []
    for i in np.array(idx).reshape(-1):
        x, y, bw, bh = boxes[i]
        x0, y0, x1, y1 = max(0, int(x)), max(0, int(y)), min(w, int(x + bw)), min(h, int(y + bh))
        z = frame.depth_m[y0 + (y1 - y0) // 4: y1 - (y1 - y0) // 4, x0 + (x1 - x0) // 4: x1 - (x1 - x0) // 4]
        z = z[np.isfinite(z) & (z > 0)]
        u = (x0 + x1) / 2
        people.append({"confidence": _number(float(conf[i])), "bbox_rgb": [x0, y0, x1 - x0, y1 - y0],
                       "bearing_deg": _number(math.degrees(math.atan2(u - frame.K[0, 2], frame.K[0, 0]))),
                       "range_m": _number(float(np.median(z))) if z.size >= 10 else None})
    return people


def check_indoors(frame: Frame, bearing_deg: float, *, output_dir: str | Path,
                  config: dict | None = None, sensors: dict | None = None) -> dict:
    """Measure the viewed direction, not the camera's location, using only its depth.

    Optical depth is z, x is right and y is down. Rotate by the mast pitch
    before computing height above the floor. Missing upward depth alone is
    insufficient evidence of outdoors because smoke also removes returns.
    """
    if not math.isfinite(bearing_deg):
        raise ValueError("bearing must be finite")
    root = Path(__file__).resolve().parents[3] / "config"
    config = dict(config) if config is not None else json.loads((root / "perception.json").read_text())
    sensors = sensors if sensors is not None else json.loads((root / "sensors.json").read_text())
    mount, depth_sensor = sensors["camera"]["mount"], sensors["depth"]
    tolerance = float(config["bearing_tolerance_deg"])
    step = int(config["normal_offset_px"])
    if tolerance <= 0 or step < 1:
        raise ValueError("bearing tolerance and normal baseline must be positive")
    pitch = math.radians(float(mount["pitch_down_deg"]))
    sin_pitch, cos_pitch = math.sin(pitch), math.cos(pitch)
    depth = frame.depth_m.copy()
    valid = np.isfinite(depth) & (depth >= depth_sensor["min_m"]) & (depth < depth_sensor["max_m"])
    depth[~valid] = np.nan
    points = cv2.depthTo3d(depth, frame.K)[:, :, :3]
    heights = float(mount["height_m"]) - sin_pitch * points[:, :, 2] - cos_pitch * points[:, :, 1]
    h, w = depth.shape
    up = -sin_pitch - (np.arange(h)[:, None] - frame.K[1, 2]) / frame.K[1, 1] * cos_pitch
    columns = np.degrees(np.arctan2(np.arange(w) - frame.K[0, 2], frame.K[0, 0]))[None, :]
    normals = np.full_like(points, np.nan)
    if h > 2 * step and w > 2 * step:
        horizontal = points[step:-step, 2*step:] - points[step:-step, :-2*step]
        vertical = points[2*step:, step:-step] - points[:-2*step, step:-step]
        cross = np.cross(horizontal, vertical)
        length = np.linalg.norm(cross, axis=2, keepdims=True)
        np.divide(cross, length, out=normals[step:-step, step:-step], where=length > 0)
    normal_up = np.abs(-sin_pitch * normals[:, :, 2] - cos_pitch * normals[:, :, 1])
    tilt = math.radians(float(config["surface_tilt_tolerance_deg"]))
    horizontal_surface = normal_up >= math.cos(tilt)
    vertical_surface = normal_up <= math.sin(tilt)
    band = np.broadcast_to(np.abs(columns - bearing_deg) <= tolerance, depth.shape)
    upward, lower = band & (up > 0), band & (up < 0)
    ceiling = (upward & valid & horizontal_surface &
               (heights >= float(mount["height_m"]) + config["overhead_clearance_m"]))

    def fraction(mask, region):
        count = int(region.sum())
        return float(np.count_nonzero(mask & region) / count) if count else 0.

    def median(array, mask):
        return _number(np.median(array[mask])) if mask.any() else None

    walls = []
    for offset in (-config["wall_side_offset_deg"], 0., config["wall_side_offset_deg"]):
        wall_band = np.broadcast_to(np.abs(columns - bearing_deg - offset) <= tolerance, depth.shape)
        wall = wall_band & valid & vertical_surface
        walls.append({"bearing_deg": float(bearing_deg + offset), "hit_fraction": fraction(wall, wall_band),
                      "range_m": median(depth, wall)})
    ceiling_share = fraction(ceiling, upward)
    void_share = fraction(~valid, upward)
    lower_share = fraction(valid, lower)
    verdict, reasons = "unknown", []
    if not upward.any() or not lower.any():
        reasons.append("The bearing has no upward and lower rays in this frame.")
    elif (ceiling_share >= config["min_ceiling_upward_fraction"] and
          lower_share >= config["min_lower_return_fraction"] and
          any(wall["hit_fraction"] >= config["min_wall_fraction"] for wall in walls)):
        verdict = "indoors"
        reasons.append("Horizontal overhead returns and vertical wall returns bound this direction; lower rays also return depth.")
    elif (void_share >= config["min_upward_void_fraction"] and
          lower_share >= config["min_lower_return_fraction"] and
          ceiling_share < config["min_ceiling_upward_fraction"] and
          all(wall["hit_fraction"] < config["min_wall_fraction"] for wall in walls)):
        verdict = "outdoors"
        reasons.append("Most upward rays have no return within sensor range while lower rays return depth, without enclosing wall evidence.")
    else:
        reasons.append("The frame does not establish an overhead enclosure or open sky; walls alone cannot distinguish a facade from an indoor wall.")
    params = {"bearing_deg": float(bearing_deg), "bearing_tolerance_deg": tolerance,
              "config": config, "mount": mount, "depth_sensor": depth_sensor}
    result = _base("check_indoors", frame, params, bearing_deg=float(bearing_deg),
                   bearing_tolerance_deg=tolerance, verdict=verdict, reasons=reasons,
                   ceiling={"height_above_floor_m": median(heights, ceiling),
                            "upward_hit_fraction": ceiling_share, "range_m": median(depth, ceiling)},
                   walls=walls, void={"upward_no_return_fraction": void_share,
                                     "lower_return_fraction": lower_share,
                                     "max_range_m": float(depth_sensor["max_m"])},
                   evidence=["OpenCV depthTo3d on this robot's RGB-D frame, with gravity-aligned normals from the configured mast pitch.",
                             "Ceiling height is relative to the configured camera mount's floor datum; no ceiling height band or map is used."],
                   debug_crop=None)
    debug = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
    debug[upward & ~valid] = (255, 80, 0)
    debug[band & valid & vertical_surface] = (0, 200, 255)
    debug[ceiling] = (0, 255, 0)
    contours, _ = cv2.findContours(band.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(debug, contours, -1, (255, 255, 255), 1)
    cv2.putText(debug, f"{verdict}: green ceiling, yellow wall, blue void", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
    path = Path(output_dir) / f"{result['id']}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), debug):
        raise OSError("cannot write indoor evidence image")
    result["debug_crop"] = str(path)
    return result


def assess_visibility(frame: Frame) -> dict:
    """RGB usable if Laplacian variance>=20, RMS contrast>=12 and haze<=0.8.

    Dark-channel haze is median local 15x15 minimum RGB /255. It is a
    brightness proxy, not physical extinction. Depth fraction is independent.
    RMS contrast is grayscale standard deviation on the 0 to 255 scale.
    """
    gray = cv2.cvtColor(frame.rgb,cv2.COLOR_RGB2GRAY)
    lap = float(cv2.Laplacian(gray,cv2.CV_64F).var()); rms = float(gray.std())
    dark = cv2.erode(frame.rgb.min(axis=2),np.ones((15,15),np.uint8))
    haze = float(np.median(dark)/255)
    valid = np.isfinite(frame.depth_m)&(frame.depth_m>0)
    return _base("assess_visibility",frame,{},laplacian_variance=_number(lap),rms_contrast=_number(rms),
        dark_channel_haze=_number(haze),valid_depth_fraction=_number(valid.mean()),
        usable_rgb=bool(lap>=20 and rms>=12 and haze<=.8),
        thresholds={"laplacian_min":20.,"rms_min":12.,"haze_max":.8},
        evidence=["RGB usability uses sharpness, grayscale contrast and local dark-channel brightness; haze is a proxy."])


def track_fire_spread(frames: list[Frame], *, hot_min_c=200.) -> dict:
    """2 to 10 increasing thermal timestamps; apparent flow without ego-motion correction.

    growth_m2_s is projected hot footprint change per second (needs depth under at least 10 percent of hot
    pixels in the first and last frame, else None; no hot pixels is area 0); growth_px_s is the same change in thermal pixels and is
    always available. Neither separates fire growth from camera approach.

    Direction is image-plane degrees, right=0, down=90. Speed is projected m/s.
    Toward-camera means hot median optical depth decreases >0.1 m/s with growth.
    These measurements cannot separate camera motion from fire motion.
    """
    if not 2<=len(frames)<=10: raise ValueError("expected 2 to 10 frames")
    times = np.array([f["stamps"]["thermal"] for f in frames])
    if not np.all(np.diff(times)>0): raise ValueError("thermal frames must be chronological")
    if any(f.thermal.shape!=frames[0].thermal.shape or f.get("source")!=frames[0].get("source") or not np.array_equal(f.thermal_K,frames[0].thermal_K) for f in frames):
        raise ValueError("fire tracking requires one camera and fixed calibration")
    areas,zs,masks,images,pixels,coverage = [],[],[],[],[],[]
    for f in frames:
        c = f.thermal.astype(float)/10-273.15; mask = (f.thermal>0)&(c>=hot_min_c)
        d = _depth(f); z = d[mask & np.isfinite(d)]
        # Flames are visible to the thermal camera but often return no depth, and the share that does changes
        # from frame to frame. Summing only pixels with depth would turn that share into fake growth, so
        # pixels without depth take the median range of those that have it. Too little depth gives no area.
        n = int(mask.sum()); missing = n-len(z)
        enough = len(z)>=max(20,.1*n)
        areas.append(0. if n==0 else float((np.sum(z*z)+missing*np.median(z)**2)/(f.thermal_K[0,0]*f.thermal_K[1,1])) if enough else None)
        pixels.append(n); coverage.append(_number(len(z)/n) if n else None)
        zs.append(float(np.median(z)) if len(z) else None); masks.append(mask)
        images.append(np.clip((c-20)*255/630,0,255).astype(np.uint8))
    flow_rows = []
    for i,(a,b) in enumerate(zip(images,images[1:])):
        flow = cv2.calcOpticalFlowFarneback(a,b,None,.5,3,21,4,7,1.5,0)
        front = cv2.morphologyEx(masks[i].astype(np.uint8),cv2.MORPH_GRADIENT,np.ones((5,5),np.uint8))>0
        if np.any(front) and zs[i] is not None:
            vec = flow[front]
            moving = np.linalg.norm(vec,axis=1)>.02
            if moving.any():
                vec = vec[moving]; dt = times[i+1]-times[i]
                flow_rows.append((np.median(vec,axis=0)/dt, float(np.median(np.linalg.norm(vec,axis=1)))*zs[i]/frames[i].thermal_K[0,0]/dt))
    growth = (areas[-1]-areas[0])/(times[-1]-times[0]) if areas[0] is not None and areas[-1] is not None else None
    rate = (zs[-1]-zs[0])/(times[-1]-times[0]) if zs[0] is not None and zs[-1] is not None else None
    vec = np.median([v for v,s in flow_rows],axis=0) if flow_rows else np.zeros(2)
    direction = float(np.degrees(np.arctan2(vec[1],vec[0]))) if np.linalg.norm(vec)>.02 else None
    return _result("track_fire_spread",[f["stamps"] for f in frames],{"hot_min_c":hot_min_c,"frame_ids":[f["id"] for f in frames]},
        areas_m2=[_number(a) if a is not None else None for a in areas],hot_pixels=pixels,hot_depth_fraction=coverage,
        growth_px_s=_number((pixels[-1]-pixels[0])/(times[-1]-times[0])),growth_m2_s=_number(growth) if growth is not None else None,
        direction_deg=_number(direction) if direction is not None else None,
        speed_m_s=_number(np.median([s for v,s in flow_rows])) if flow_rows else 0.,
        hot_depth_rate_m_s=_number(rate) if rate is not None else None,
        spreading_toward_camera=bool(rate is not None and rate<-.1 and growth is not None and growth>0),
        evidence=["Farneback tracks the thermal hot boundary; all motion is apparent and includes camera motion."])


def measure_opening(frame: Frame, bearing_deg: float, *, body_width_m=.6, clearance_m=.1,
                    body_height_m=0., band_half_height_m=.1) -> dict:
    """Measure a bounded depth recess at optical y=body_height_m, positive down.

    Takes the nearest valid depth in a band of rows (placed at the wall's range) for every column, then walks
    outward from the requested bearing to the two jambs. A jamb is the first valid column after a run of
    columns with no depth return, or a column at least 0.5 m nearer than the gap interior. A gap into a dark
    or far room often returns no depth at all, so missing depth inside the gap is expected and is reported.
    Both jambs need valid depth within 0.25 m of each other in z, and the interior must be missing or at
    least 0.5 m deeper than the jambs, otherwise this is a flat wall with a nearer object on each side.
    Missing or clipped jambs give unknown width and passable=False. This is not a collision check.
    Camera-relative body height must be supplied by the caller's mount model.
    """
    if not np.isfinite([bearing_deg,body_width_m,clearance_m,body_height_m,band_half_height_m]).all() or body_width_m<=0 or clearance_m<0 or band_half_height_m<=0:
        raise ValueError("invalid opening parameters")
    params = dict(bearing_deg=bearing_deg,body_width_m=body_width_m,clearance_m=clearance_m,
                  body_height_m=body_height_m,band_half_height_m=band_half_height_m)
    r = _base("measure_opening",frame,params,clear_width_m=None,passable=False,jamb_pixels=None,
        wall_z_m=None,gap_depth_missing=None,evidence=["No bounded depth gap with two measured jambs at the requested bearing."])
    d = frame.depth_m; k=frame.K
    u = int(round(k[0,2]+k[0,0]*math.tan(math.radians(bearing_deg))))
    if not 0<=u<d.shape[1]: return r
    positive = d[np.isfinite(d)&(d>0)]
    if not len(positive): return r
    ref = float(np.median(positive))
    jambs = None
    for _ in range(2):
        # The band row depends on range (row = cy + fy*y/z). First pass uses the scene median, second the jamb range.
        cy = int(round(k[1,2]+body_height_m*k[1,1]/ref)); half = max(1,int(math.ceil(band_half_height_m*k[1,1]/ref)))
        y0,y1 = max(0,cy-half),min(d.shape[0],cy+half+1)
        if y0>=y1:return r
        band = d[y0:y1]
        # Incomplete depth in the body band is not evidence of clearance.
        valid = np.isfinite(band)&(band>0)
        profile = np.min(np.where(valid,band,np.inf),axis=0)
        fraction = valid.mean(axis=0)
        missing = fraction<.2
        profile[fraction<.8] = np.nan
        found = _find_jambs(profile,missing,u)
        if found is None: return r
        left,right = found
        z = (profile[left]+profile[right])/2
        if abs(z-ref)<.05: break
        ref = float(z)
    zl,zr = float(profile[left]),float(profile[right])
    if abs(zl-zr)>.25:return r
    inside = slice(left+1,right)
    finite = profile[inside][np.isfinite(profile[inside])]
    no_return = float(np.mean(missing[inside]))
    if no_return<.5 and not (len(finite) and np.median(finite)>=max(zl,zr)+.5):return r
    z=(zl+zr)/2; width=(right-left-1)*z/k[0,0]
    r.update(clear_width_m=_number(width),passable=bool(width>=body_width_m+2*clearance_m),
        jamb_pixels=[left+.5,right-.5],wall_z_m=_number(z),gap_depth_missing=_number(no_return),
        evidence=[f"Measured two jambs at z={z:.3f} m; gap {width:.3f} m, required {body_width_m+2*clearance_m:.3f} m including margins."
                  +(f" {no_return:.0%} of gap columns have no depth return." if no_return>0 else "")])
    return r


def _find_jambs(profile, missing, u):
    """Columns of the left and right jamb around column u, or None when either side runs off the image."""
    def walk(step):
        i,ref = u,(float(profile[u]) if np.isfinite(profile[u]) else None)
        while 0<=i+step<len(profile):
            i += step
            if not np.isfinite(profile[i]): continue
            z = float(profile[i])
            if ref is None or z<=ref-.5: return i
            ref = z
        return None
    left,right = walk(-1),walk(1)
    return None if left is None or right is None else (left,right)
