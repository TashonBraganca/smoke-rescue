"""Sensor-only reference for ThermalPerception.Detect and proposed geometry gates.

The baseline deliberately ignores the thermal camera's 3 cm translation, like C#.
Coordinates are robot right, forward, up. No building coordinates enter this module.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

# Experimental values, not changes to the running brain's configuration.
PROPOSED = {
    'person_floor_clearance_m': .12,
    'person_top_max_m': 1.25,
    'person_background_ring_px': 8,
    'person_min_background_pixels': 20,
    'person_plane_min_tolerance_m': .025,
    'person_plane_noise_multiplier': 2.5,
    'person_plane_support_fraction': .65,
    'person_contrast_min_c': 1.0,
    'person_max_recess_m': .8,
    'person_local_contrast_window_px': 61,
}


PROPOSED_REASONS = {
    'person_floor_clearance_m': 'Require median observed height above floor noise; 0.12 m is a provisional minimum for body thickness, not a proven bound for lying people.',
    'person_top_max_m': 'Allow slumped and supported bodies up to 1.1 m plus 0.15 m calibration margin; reject tall walls and frames. Standing people are outside this scenario assumption.',
    'person_background_ring_px': 'Measure an 8-pixel neighborhood around the actual component, not its bounding rectangle. This is about 7 cm at 3 m with the recorded thermal focal length.',
    'person_min_background_pixels': 'Require 20 valid observations before judging background geometry or temperature; missing depth cannot establish an isolated object.',
    'person_plane_min_tolerance_m': 'Allow 25 mm for depth quantization, the ignored camera offset and plane-fit error at close range.',
    'person_plane_noise_multiplier': 'Allow 2.5 times the configured z-squared depth-noise sigma so distant planar walls are not called curved bodies because of noise.',
    'person_plane_support_fraction': 'Call a fitted plane extended background when at least 65 percent of valid ring depths also lie on it. This value is a provisional in-sample tolerance.',
    'person_contrast_min_c': 'Require a 1 C median temperature excess over the ring, above sensor NETD and fixed pattern noise. This can reject real people in hot rooms and remains experimental.',
    'person_max_recess_m': 'Reject warm pixels at least 0.8 m behind their neighborhood, consistent with an opening. Reuses the agent vision opening criterion.',
    'person_local_contrast_window_px': 'Experimental 61-pixel square mean background, about 0.53 m across at 3 m. Absolute deviation also permits a cooler body in a hotter room; measured recall did not justify adoption.',
}


def matched_geometry(frame, mount, depth_limits=(.2, 8.)):
    """C# Match and OccupancyGrid.BackProject, including ties-to-even rounding."""
    y, x = np.indices(frame.thermal.shape)
    a, b = frame.thermal_K.astype(np.float32), frame.K.astype(np.float32)
    u = np.rint((x.astype(np.float32)-a[0, 2])/a[0, 0]*b[0, 0]+b[0, 2]).astype(int)
    v = np.rint((y.astype(np.float32)-a[1, 2])/a[1, 1]*b[1, 1]+b[1, 2]).astype(int)
    valid = (u >= 0) & (v >= 0) & (u < frame.depth_m.shape[1]) & (v < frame.depth_m.shape[0])
    z = frame.depth_m[np.clip(v, 0, frame.depth_m.shape[0]-1), np.clip(u, 0, frame.depth_m.shape[1]-1)].copy()
    valid &= np.isfinite(z) & (z >= depth_limits[0]) & (z <= depth_limits[1])
    z[~valid] = 0
    pitch = np.float32(np.radians(mount['pitch_down_deg']))
    down = (v-b[1, 2])/b[1, 1]*z
    points = np.stack(((u-b[0, 2])/b[0, 0]*z,
                       mount['forward_offset_m']+z*np.cos(pitch)-down*np.sin(pitch),
                       mount['height_m']-z*np.sin(pitch)-down*np.cos(pitch)), axis=-1)
    return z, valid, points, u, v


def detect(frame, config, mount, pose=(0., 0., 0.), *, features=True, settings=None,
           depth_limits=(.2, 8.), noise_coeff=.0015, segmentation='absolute'):
    """Faithful four-connected, edge-depth-constrained components in raster order.

    Sparse graph connectivity equals the C# queue traversal. Summation uses double
    precision, so centroid/area parity is numerical rather than bit-for-bit float parity.
    Each return includes mask, metric area, robot-frame centroid and diagnostic cues.
    """
    settings = {k: config[k] for k in PROPOSED if k in config} | (settings or {})
    z, valid, points, u, v = matched_geometry(frame, mount, depth_limits)
    celsius = frame.thermal.astype(np.float32)*np.float32(.1)-np.float32(273.15)
    if segmentation == 'absolute':
        mask = valid & (celsius >= config['person_min_c']) & (celsius <= config['person_max_c'])
    elif segmentation == 'contrast':
        s = PROPOSED | (settings or {})
        width = s['person_local_contrast_window_px']
        thermal_valid = frame.thermal > 0
        n = cv2.boxFilter(thermal_valid.astype(np.float32), -1, (width, width), normalize=False, borderType=cv2.BORDER_CONSTANT)
        total = cv2.boxFilter(np.where(thermal_valid, celsius, 0), -1, (width, width), normalize=False, borderType=cv2.BORDER_CONSTANT)
        background = total/np.maximum(n, 1)
        mask = valid & thermal_valid & (n >= s['person_min_background_pixels']) & (np.abs(celsius-background) >= s['person_contrast_min_c'])
    else:
        raise ValueError('unknown thermal segmentation')
    ids = np.full(mask.shape, -1, int); ids[mask] = np.arange(mask.sum())
    if not mask.any():
        return []
    edges = []
    for a, b in ((np.s_[:, :-1], np.s_[:, 1:]), (np.s_[:-1, :], np.s_[1:, :])):
        ok = mask[a] & mask[b] & (np.abs(z[a]-z[b]) <= config['blob_depth_jump_m'])
        edges.append((ids[a][ok], ids[b][ok]))
    src = np.concatenate([e[0] for e in edges]); dst = np.concatenate([e[1] for e in edges])
    graph = coo_matrix((np.ones(len(src)), (src, dst)), shape=(int(mask.sum()),)*2)
    n, labels = connected_components(graph, directed=False)
    indices = np.flatnonzero(mask)
    # Stable sorting keeps raster order inside each component.
    order = np.argsort(labels, kind='stable'); counts = np.bincount(labels, minlength=n)
    groups = np.split(indices[order], np.cumsum(counts)[:-1])
    result = []
    area_px = z.astype(float)**2 / (float(frame.thermal_K[0, 0])*float(frame.thermal_K[1, 1]))
    cosine, sine = np.cos(pose[2]), np.sin(pose[2])
    for group in sorted(groups, key=lambda g: g[0]):
        if len(group) < config['person_min_pixels']:
            continue
        area = float(area_px.flat[group].sum())
        if not config['person_min_area_m2'] <= area <= config['person_max_area_m2']:
            continue
        m = np.zeros(mask.shape, bool); m.flat[group] = True
        p = points[m]; xy = np.average(p[:, :2], axis=0, weights=area_px[m])
        xy = np.array([pose[0]+cosine*xy[0]-sine*xy[1], pose[1]+sine*xy[0]+cosine*xy[1]])
        yy, xx = np.nonzero(m)
        comp = dict(mask=m, xy=xy, area_m2=area, pixels=len(group),
                    bbox=[int(xx.min()), int(yy.min()), int(np.ptp(xx)+1), int(np.ptp(yy)+1)],
                    range_m=float(np.median(z[m])))
        if features:
            comp.update(blob_features(m, z, points, celsius, valid, settings, noise_coeff))
        result.append(comp)
    return result


def blob_features(mask, z, points, celsius, valid, settings=None, noise_coeff=.0015):
    s = PROPOSED | (settings or {})
    radius = s['person_background_ring_px']
    ring = (cv2.dilate(mask.astype(np.uint8), np.ones((2*radius+1,)*2, np.uint8)) > 0) & ~mask
    p = points[mask]; mean = p.mean(axis=0)
    covariance = (p-mean).T @ (p-mean)/len(p)
    eig, axes = np.linalg.eigh(covariance); normal = axes[:, 0]
    residual = float(np.sqrt(max(0., eig[0])))
    tolerance = max(s['person_plane_min_tolerance_m'], s['person_plane_noise_multiplier']*noise_coeff*float(np.median(z[mask]))**2)
    ring_points = points[ring & valid]
    support = float(np.mean(np.abs((ring_points-mean) @ normal) <= tolerance)) if len(ring_points) >= s['person_min_background_pixels'] else None
    background = celsius[ring & (celsius > -273.15)]
    contrast = float(np.median(celsius[mask])-np.median(background)) if len(background) >= s['person_min_background_pixels'] else None
    heights = np.percentile(p[:, 2], [5, 50, 95])
    extent = np.percentile(p, 95, axis=0)-np.percentile(p, 5, axis=0)
    return dict(temperature_c=np.percentile(celsius[mask], [5, 50, 95]).tolist(),
                temperature_std_c=float(np.std(celsius[mask])), height_m=heights.tolist(),
                extent_m=extent.tolist(), fill_fraction=float(mask.sum()/np.prod(np.ptp(np.argwhere(mask), axis=0)+1)),
                plane_rms_m=residual, plane_tolerance_m=tolerance,
                plane_up_normal=float(abs(normal[2])), ring_plane_support=support,
                ring_valid_pixels=len(ring_points), contrast_c=contrast,
                ring_depth_delta_m=float(np.median(z[ring & valid])-np.median(z[mask])) if len(ring_points) else None)


def gates(comp, settings=None):
    """Each rule can be measured alone. Unknown background fails closed."""
    s = PROPOSED | (settings or {})
    height = comp['height_m']
    return dict(height=bool(height[1] >= s['person_floor_clearance_m'] and height[2] <= s['person_top_max_m']),
                separate=bool(comp['ring_plane_support'] is not None and
                              comp['ring_depth_delta_m'] is not None and
                              comp['ring_depth_delta_m'] > -s['person_max_recess_m'] and
                              not (comp['plane_rms_m'] <= comp['plane_tolerance_m'] and
                                   comp['ring_plane_support'] >= s['person_plane_support_fraction'])),
                contrast=bool(comp['contrast_c'] is not None and comp['contrast_c'] >= s['person_contrast_min_c']))


def detect_filtered(frame, config, mount, pose=(0., 0., 0.), *, rules=('height', 'separate', 'contrast'),
                    settings=None, **kwargs):
    """Proposed C# insertion point before returning PersonDetection to ObservePeople.

    This filters baseline candidates; it cannot recover a body absent from the
    absolute-temperature segmentation. Use an empty rules tuple for an ablation.
    """
    settings = {k: config[k] for k in PROPOSED if k in config} | (settings or {})
    if set(rules)-{'height', 'separate', 'contrast'}:
        raise ValueError('unknown warm-blob rule')
    return [c for c in detect(frame, config, mount, pose, settings=settings, **kwargs)
            if all(gates(c, settings)[rule] for rule in rules)]


class CandidateTracker:
    """ObservePeople reference, without navigation or confirmation decisions.

    Fixed-trajectory replay can expose new false candidates after filtering. It
    cannot predict a route changed by the new detector or simulate close-view dwell.
    """
    def __init__(self, config):
        self.config = config
        self.tracks = []
        self.confirmed_positions = []

    def observe(self, t, detections):
        events = []
        c = self.config
        for detection in detections:
            xy = np.asarray(detection['xy'])
            if any(np.linalg.norm(xy-p) <= c['survivor_merge_m'] for p in self.confirmed_positions):
                continue
            nearest, best = None, c['survivor_merge_m']
            for track in self.tracks:
                distance = np.linalg.norm(track['xy']-xy)
                if not track['confirmed'] and distance <= best:
                    best, nearest = distance, track
            if nearest is None:
                nearest = dict(xy=xy, last_seen=-np.inf, observations=[], candidate=False, confirmed=False)
                self.tracks.append(nearest)
            if t <= nearest['last_seen']:
                continue
            nearest['xy'], nearest['last_seen'] = xy.copy(), t
            nearest['observations'] = [stamp for stamp in nearest['observations'] if t-stamp <= c['candidate_window_s']]
            nearest['observations'].append(t)
            if not nearest['candidate'] and len(nearest['observations']) >= c['candidate_frames']:
                nearest['candidate'] = True
                events.append(dict(t=t, xy=xy.tolist(), area_m2=detection['area_m2'], frames=len(nearest['observations'])))
        return events
