"""Read a robot's exported team belief. This module never loads ground truth."""
from __future__ import annotations

from dataclasses import dataclass
import gzip
from pathlib import Path
import struct

import numpy as np


@dataclass(frozen=True)
class TeamGrid:
    size: int
    cell_m: float
    origin: np.ndarray
    at: float
    sources: tuple[str, ...]
    occupancy: np.ndarray
    searched_confidence: np.ndarray
    source_mask: np.ndarray
    searched_source_mask: np.ndarray
    age_s: np.ndarray
    search_age_s: np.ndarray


def load_team_map(path: Path) -> TeamGrid:
    data = gzip.decompress(Path(path).read_bytes())
    header = struct.Struct('<4sBifffdB')
    if len(data) < header.size:
        raise ValueError(f'truncated team-map header in {path}')
    magic, version, size, cell, x, y, at, count = header.unpack_from(data)
    if magic != b'TMAP' or version != 1 or not 1 <= size <= 1000 or not 1 <= count <= 32:
        raise ValueError(f'unsupported team-map header in {path}')
    if not np.isfinite([cell, x, y, at]).all() or cell <= 0 or at < 0:
        raise ValueError(f'invalid team-map geometry in {path}')
    offset, sources = header.size, []
    for _ in range(count):
        if len(data) < offset + 2:
            raise ValueError(f'truncated team-map source in {path}')
        length = struct.unpack_from('<H', data, offset)[0]
        offset += 2
        if not 1 <= length <= 256 or len(data) < offset + length:
            raise ValueError(f'invalid team-map source in {path}')
        sources.append(data[offset:offset + length].decode('utf-8'))
        offset += length
    if len(set(sources)) != count or len(data) != offset + size * size * 24:
        raise ValueError(f'invalid team-map array length or sources in {path}')
    arrays = []
    for dtype in ('<f4', '<f4', '<u4', '<u4', '<f4', '<f4'):
        array = np.frombuffer(data, dtype=dtype, count=size * size, offset=offset).reshape(size, size)
        arrays.append(array)
        offset += size * size * 4
    occupancy, searched, mask, search_mask, age, search_age = arrays
    if (not all(np.isfinite(a).all() for a in (occupancy, searched, age, search_age)) or
            np.any((occupancy < 0) | (occupancy > 1)) or np.any((searched < 0) | (searched > 1)) or
            np.any((mask.astype(np.uint64) >> count) != 0) or np.any((search_mask.astype(np.uint64) >> count) != 0)):
        raise ValueError(f'invalid team-map belief in {path}')
    return TeamGrid(size, cell, np.array([x, y]), at, tuple(sources), *arrays)


def latest_team_map(robot_dir: Path, at: float = float('inf')) -> TeamGrid | None:
    candidates = []
    for path in (Path(robot_dir) / 'brain_map').glob('*.team.bin.gz'):
        stamp = float(path.name.removesuffix('.team.bin.gz'))
        if stamp <= at:
            candidates.append((stamp, path))
    for _, path in sorted(candidates, reverse=True):
        try:
            grid = load_team_map(path)
        except (OSError, EOFError, ValueError):
            continue
        if grid.at <= at:
            return grid
    return None
