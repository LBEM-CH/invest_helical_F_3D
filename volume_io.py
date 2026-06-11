#!/usr/bin/env python3
"""
Reference-map reading for invest_helical_F_3D's 3D view.

author: Wen-Lu Chung

Reads the refined average density for placement in the 3D scene: RELION/CCP-EM
`.mrc`/`.map`/`.rec` via the `mrcfile` package, and Dynamo's native `.em`
(512-byte header + raw data) with a small built-in reader. No OpenGL dependency,
so the CLI can preload a map without pulling in PyOpenGL.
"""

from __future__ import annotations

import numpy as np

# EM data-type code (header byte 3) -> numpy dtype
_EM_DTYPE = {1: "i1", 2: "i2", 4: "i4", 5: "f4", 9: "f8"}


def read_em(path: str):
    """Read a Dynamo/TOM `.em` volume. Returns (data float32 (nz,ny,nx), None)."""
    with open(path, "rb") as fh:
        header = fh.read(512)
        little = header[0] in (6,)            # byte0=6 -> PC little-endian; else assume LE
        end = "<" if little or header[0] not in (3, 5) else ">"
        dt = _EM_DTYPE.get(header[3], "f4")
        nx, ny, nz = np.frombuffer(header, dtype=end + "i4", count=3, offset=4)
        data = np.frombuffer(fh.read(), dtype=end + dt)
    # EM stores fastest axis first (x); reshape to (nz, ny, nx) for marching cubes
    return np.ascontiguousarray(data.reshape(int(nz), int(ny), int(nx)), "float32"), None


def read_volume(path: str):
    """Read a density map. Returns (data float32 (nz,ny,nx), voxel_size_A or None)."""
    if path.lower().endswith(".em"):
        return read_em(path)
    import mrcfile
    with mrcfile.open(path, permissive=True) as m:
        data = np.ascontiguousarray(m.data, dtype="float32")
        vx = float(m.voxel_size.x) if m.voxel_size.x else None
    return data, vx
