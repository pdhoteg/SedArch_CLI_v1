"""paths.py – Flow-path geometry utilities ported from paths.cxx

All 2-D positions use Python ``complex`` numbers (Dot):
  real(dot) → x (column direction)
  imag(dot) → y (row direction)

Grid matrices have shape ``(ncols, nrows)`` and are indexed ``matrix[icol, irow]``.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from sedarch.sequence import (
    SedVector, Sequence, interpolate_grid, interpolate_value,
    get_adjacent4, limit_bottom,
)

# ---------------------------------------------------------------------------
# Basic 2-D vector helpers (Dot = complex)
# ---------------------------------------------------------------------------

def vec_pro(a: complex, b: complex) -> float:
    """2-D 'cross product' (scalar z-component): a.x*b.y - a.y*b.x."""
    return a.real * b.imag - a.imag * b.real


def dot_abs(d: complex) -> float:
    return abs(d)


def unit_vec(a: complex, b: complex = None) -> complex:
    """Return unit vector of *a*, or of (b-a) when *b* is given."""
    if b is not None:
        a = b - a
    mag = abs(a)
    return a / mag if mag > 1e-12 else 0j


def rotate(d: complex, angle_rad: float) -> complex:
    """Rotate complex number by angle_rad."""
    return d * complex(math.cos(angle_rad), math.sin(angle_rad))


def proj_pt(a: complex, b: complex, p: complex) -> float:
    """Fraction of projection of p onto segment a→b."""
    ab = b - a
    mag2 = ab.real ** 2 + ab.imag ** 2
    if mag2 < 1e-24:
        return 0.0
    return ((p.real - a.real) * ab.real + (p.imag - a.imag) * ab.imag) / mag2


def dist_pt_seg(s1: complex, s2: complex, p: complex) -> float:
    """Signed distance from point p to segment s1→s2."""
    seg = s2 - s1
    mag = abs(seg)
    if mag < 1e-12:
        return abs(p - s1)
    # Perpendicular distance (signed via cross product)
    return vec_pro(seg / mag, p - s1)


def ang_dif(a1: complex, a2: complex, b1: complex, b2: complex) -> float:
    """Angle difference (radians, CCW) from segment a1→a2 to b1→b2."""
    da = a2 - a1
    db = b2 - b1
    return math.atan2(vec_pro(da, db), da.real * db.real + da.imag * db.imag)


# ---------------------------------------------------------------------------
# Cell-corner / slope helpers
# ---------------------------------------------------------------------------

def _get_cell_corners(matrix: np.ndarray, x: float, y: float):
    """
    Return corners z[0..3] (CCW from SW) and residuals dx, dy.
    SW=z[0], SE=z[1], NE=z[2], NW=z[3].
    """
    ncols, nrows = matrix.shape
    icell = int(math.floor(x))
    jcell = int(math.floor(y))
    dx = x - icell
    dy = y - jcell

    # Clamp cell indices
    if icell < 0:
        icell, dx = 0, 0.0
    elif icell > ncols - 2:
        icell, dx = ncols - 2, 1.0
    if jcell < 0:
        jcell, dy = 0, 0.0
    elif jcell > nrows - 2:
        jcell, dy = nrows - 2, 1.0

    z = [
        float(matrix[icell,     jcell]),     # SW
        float(matrix[icell + 1, jcell]),     # SE
        float(matrix[icell + 1, jcell + 1]),  # NE
        float(matrix[icell,     jcell + 1]), # NW
    ]
    return z, dx, dy


def value_in_cell(matrix: np.ndarray, x: float, y: float) -> float:
    """Bilinear interpolation within a cell."""
    z, dx, dy = _get_cell_corners(matrix, x, y)
    return ((z[0] * (1 - dx) + z[1] * dx) * (1 - dy) +
            (z[3] * (1 - dx) + z[2] * dx) * dy)


def slope_in_cell(matrix: np.ndarray, x: float, y: float) -> Tuple[float, float]:
    """Return (sx, sy) slope at position (x,y). Positive → downhill in that direction."""
    z, dx, dy = _get_cell_corners(matrix, x, y)
    sx = (z[0] - z[1]) * (1 - dy) + (z[3] - z[2]) * dy
    sy = (z[0] - z[3]) * (1 - dx) + (z[1] - z[2]) * dx
    return sx, sy


def slope_in_cell_center(matrix: np.ndarray, icell: int, jcell: int) -> Tuple[float, float]:
    return slope_in_cell(matrix, icell + 0.5, jcell + 0.5)


# ---------------------------------------------------------------------------
# Path interpolation
# ---------------------------------------------------------------------------

def _linear(t: float, p0: complex, p1: complex) -> complex:
    return p0 * (1.0 - t) + p1 * t


def _bezier(t: float, p0: complex, p1: complex, p2: complex, p3: complex) -> complex:
    u = 1.0 - t
    return p0 * u**3 + p1 * (u**2 * t * 3) + p2 * (u * t**2 * 3) + p3 * t**3


def _inter_bezier(t: float, b0: complex, b1: complex, b2: complex, b3: complex) -> complex:
    """Cubic Bézier interpolation between b1 and b2 with context b0, b3."""
    dis = abs(b2 - b1)
    factor = 0.333333
    q0 = unit_vec(unit_vec(b0, b1) + unit_vec(b1, b2))
    q3 = unit_vec(unit_vec(b1, b2) + unit_vec(b2, b3))
    r0 = b1 + q0 * dis * factor
    r3 = b2 - q3 * dis * factor
    return _bezier(t, b1, r0, r3, b2)


def path_length(path: List[complex]) -> float:
    return sum(abs(path[i + 1] - path[i]) for i in range(len(path) - 1))


def resample_path(path: List[complex], step: float,
                  numrows: int = sys.maxsize, numcols: int = sys.maxsize,
                  do_bezier: bool = True) -> Tuple[bool, List[complex]]:
    """Resample path at regular intervals of *step* cell-units."""
    if step <= 0 or len(path) < 2:
        return False, path

    total = path_length(path)
    num = int(math.floor(total / step)) + 1
    step_adj = total / num * 1.0001

    # Ghost points before first and after last
    n = len(path)
    if n >= 3:
        angrad = ang_dif(path[0], path[1], path[1], path[2])
        p1o = path[2] - path[1]
        p1o = rotate(p1o, -1.5 * angrad)
        prior = path[0] - p1o

        angrad2 = ang_dif(path[-1], path[-2], path[-2], path[-3])
        p1o2 = path[-3] - path[-2]
        p1o2 = rotate(p1o2, -1.5 * angrad2)
        after = path[-1] - p1o2
    else:
        prior = path[0] - (path[1] - path[0])
        after = path[-1] + (path[-1] - path[-2])

    newpath = [path[0]]
    segment = 0.0
    newdot = path[0]

    def _check(d: complex) -> bool:
        if numcols != sys.maxsize and (d.real < 0 or d.real > numcols - 1):
            return False
        if numrows != sys.maxsize and (d.imag < 0 or d.imag > numrows - 1):
            return False
        return True

    for i in range(len(path) - 1):
        b1, b2 = path[i], path[i + 1]
        dis = abs(b2 - b1)
        segment += dis
        remainder = segment
        b0 = prior if i == 0 else path[i - 1]
        b3 = after if i >= len(path) - 2 else path[i + 2]
        while remainder > step_adj:
            remainder -= step_adj
            t = 1.0 - remainder / dis if dis > 0 else 0.0
            if do_bezier:
                newdot = _inter_bezier(t, b0, b1, b2, b3)
            else:
                newdot = _linear(t, b1, b2)
            if not _check(newdot):
                return False, newpath
            newpath.append(newdot)
        segment = remainder

    last = path[-1]
    if _check(last):
        newpath.append(last)
    return True, newpath


def smooth_path(path: List[complex]) -> List[complex]:
    """Smooth path by moving interior points toward chord."""
    factor = 0.5
    if len(path) < 3:
        return path
    newpath = [path[0]]
    for i in range(1, len(path) - 1):
        b0, b1, b2 = path[i - 1], path[i], path[i + 1]
        frac = proj_pt(b0, b2, b1)
        frac = max(0.0, min(1.0, frac))
        dotproj = _linear(frac, b0, b2)
        newpath.append(_linear(factor, b1, dotproj))
    newpath.append(path[-1])
    return newpath


def round_path(path: List[complex]) -> List[complex]:
    """Eliminate tiny floating-point residuals from path coordinates."""
    eps = 0.00001
    result = []
    for d in path:
        rx = round(d.real) if abs(round(d.real) - d.real) < eps * 2 else d.real
        ry = round(d.imag) if abs(round(d.imag) - d.imag) < eps * 2 else d.imag
        result.append(complex(rx, ry))
    return result


# ---------------------------------------------------------------------------
# Cell-edge / boundary geometry
# ---------------------------------------------------------------------------

def side_of_cell(icellx: int, jcelly: int, iside: int) -> Tuple[complex, complex]:
    """Return (s1, s2) endpoints of a cell side (CCW: 0=bot, 1=right, 2=top, 3=left)."""
    iside = iside % 4
    if iside == 0:
        return complex(icellx, jcelly), complex(icellx + 1, jcelly)
    elif iside == 1:
        return complex(icellx + 1, jcelly), complex(icellx + 1, jcelly + 1)
    elif iside == 2:
        return complex(icellx + 1, jcelly + 1), complex(icellx, jcelly + 1)
    else:  # 3
        return complex(icellx, jcelly + 1), complex(icellx, jcelly)


def cross_segs(p1: complex, p2: complex, q1: complex, q2: complex
               ) -> Tuple[bool, float, float, bool]:
    """Find intersection of two segments. Returns (intersects, fracp, fracq, toleft)."""
    p12 = p2 - p1
    q12 = q2 - q1
    tails = q1 - p1
    den = vec_pro(p12, q12)
    toleft = den > 0
    if den == 0:
        return False, 0.0, 0.0, toleft
    fracp = vec_pro(tails, q12) / den
    fracq = vec_pro(tails, p12) / den
    intersects = (0.0 <= fracp <= 1.0) and (0.0 <= fracq <= 1.0)
    return intersects, fracp, fracq, toleft


def move_to_edge(dot: complex, sx: float, sy: float) -> Tuple[bool, complex]:
    """Move dot to cell edge in direction (sx, sy). Returns (moved, new_dot)."""
    eps = 1e-5
    if abs(sx) < eps and abs(sy) < eps:
        return False, dot

    icell = int(math.floor(dot.real))
    jcell = int(math.floor(dot.imag))
    q2 = dot + complex(sx, sy)

    for iside in range(4):
        p1, p2 = side_of_cell(icell, jcell, iside)
        intersects, fracs, fracq, toleft = cross_segs(p1, p2, dot, q2)
        if not toleft and fracq > 0 and 0 <= fracs <= 1:
            crossdot = p1 + (p2 - p1) * fracs
            rx = crossdot.real
            ry = crossdot.imag
            # Clamp to cell
            if math.floor(rx) < icell:
                rx = float(icell)
            if math.floor(rx) > icell:
                rx = float(icell) + 1.0 - eps * 0.5
            if math.floor(ry) < jcell:
                ry = float(jcell)
            if math.floor(ry) > jcell:
                ry = float(jcell) + 1.0 - eps * 0.5
            return True, complex(rx, ry)
    return False, dot


def is_on_edge(dot: complex) -> Tuple[bool, int, bool]:
    """
    Check if dot is on a cell edge or corner.
    Returns (on_edge, iwhich, is_corner).
    """
    eps = 0.001
    icell = int(math.floor(dot.real))
    jcell = int(math.floor(dot.imag))
    dx = dot.real - icell
    dy = dot.imag - jcell

    ix = -1 if dx < eps else (1 if dx > 1.0 - eps else 0)
    iy = -1 if dy < eps else (1 if dy > 1.0 - eps else 0)

    is_corner = (ix != 0 and iy != 0)
    iwhich = -1
    if is_corner:
        iwhich = (ix + 1) // 2
        if iy > 0:
            iwhich = 3 - iwhich
    else:
        if ix == 0:
            iwhich = iy + 1
        else:
            iwhich = 2 - ix
    return iwhich >= 0, iwhich, is_corner


def edge_beside(dot: complex) -> Tuple[bool, complex]:
    """Express a dot on cell edge as a dot in the adjacent cell."""
    eps = 0.001
    icell = int(math.floor(dot.real))
    jcell = int(math.floor(dot.imag))
    dx = dot.real - icell
    dy = dot.imag - jcell
    if dx < eps:
        return True, complex(icell - eps, dot.imag)
    elif dx > 1.0 - eps:
        return True, complex(float(icell + 1), dot.imag)
    elif dy < eps:
        return True, complex(dot.real, jcell - eps)
    elif dy > 1.0 - eps:
        return True, complex(dot.real, float(jcell + 1))
    return False, dot


def next_corner(dot: complex) -> Tuple[bool, complex]:
    """Move a corner dot to the next CCW corner."""
    eps = 0.001
    icell = int(math.floor(dot.real))
    jcell = int(math.floor(dot.imag))
    dx = dot.real - icell
    dy = dot.imag - jcell
    if dx <= 2 * eps:
        if dy <= 2 * eps:
            return True, complex(float(icell) - eps, dot.imag)
        elif dy >= 1.0 - 2 * eps:
            return True, complex(dot.real, float(jcell + 1))
    elif dx >= 1.0 - 2 * eps:
        if dy >= 1.0 - 2 * eps:
            return True, complex(float(icell + 1), dot.imag)
        elif dy <= 2 * eps:
            return True, complex(dot.real, float(jcell) - eps)
    return False, dot


def nearest_node(dot: complex, numcols: int, numrows: int) -> Tuple[int, int]:
    """Return (icol, irow) of nearest grid node."""
    irow = max(0, min(int(round(dot.imag)), numrows - 1))
    icol = max(0, min(int(round(dot.real)), numcols - 1))
    return icol, irow


def is_node_hole(icol: int, irow: int, matrix: np.ndarray, nstrict: int = 1) -> bool:
    """Return True if (icol, irow) is a local minimum."""
    f4 = get_adjacent4(matrix, icol, irow)
    zc = matrix[icol, irow]
    if zc <= f4[0] and zc <= f4[1] and zc <= f4[2] and zc <= f4[3]:
        ns = sum(1 for f in f4 if zc < f)
        return ns >= nstrict
    return False


def inward_slope(dot: complex, matrix: np.ndarray) -> Tuple[bool, float, float]:
    """Return (is_inward, sx, sy): whether cell slope points into the cell from dot."""
    icellx = int(math.floor(dot.real))
    jcelly = int(math.floor(dot.imag))
    dx = dot.real - icellx
    dy = dot.imag - jcelly
    sx, sy = slope_in_cell_center(matrix, icellx, jcelly)
    inx = (dx < 0.5 and sx >= 0) or (dx >= 0.5 and sx <= 0)
    iny = (dy < 0.5 and sy >= 0) or (dy >= 0.5 and sy <= 0)
    return inx and iny, sx, sy


def get_adjacent8(matrix: np.ndarray, icol: int, irow: int) -> np.ndarray:
    """Return 8 neighbors: W, S, E, N, SW, SE, NE, NW."""
    ncols, nrows = matrix.shape
    f = np.empty(8, dtype=matrix.dtype)
    cl = max(0, icol - 1); cr = min(ncols - 1, icol + 1)
    rb = max(0, irow - 1); rt = min(nrows - 1, irow + 1)
    f[0] = matrix[cl, irow]   # W
    f[1] = matrix[icol, rb]   # S
    f[2] = matrix[cr, irow]   # E
    f[3] = matrix[icol, rt]   # N
    f[4] = matrix[cl, rb]     # SW
    f[5] = matrix[cr, rb]     # SE
    f[6] = matrix[cr, rt]     # NE
    f[7] = matrix[cl, rt]     # NW
    return f


def get_adjacent_nodes4(icol: int, irow: int) -> List[Tuple[int, int]]:
    """Return 4 adjacent (icol, irow) node tuples."""
    return [(icol - 1, irow), (icol, irow - 1),
            (icol + 1, irow), (icol, irow + 1)]


# ---------------------------------------------------------------------------
# Downslope path tracing
# ---------------------------------------------------------------------------

def advance_down(z: np.ndarray, dot: complex) -> Tuple[bool, complex, bool, bool]:
    """
    Advance dot one step downhill along cell edges.
    Returns (moved, new_dot, on_boundary, local_hole).
    """
    eps = 1e-5
    ncols, nrows = z.shape

    def _is_bound(d: complex) -> bool:
        return (d.real < eps or d.real > ncols - 1 - eps or
                d.imag < eps or d.imag > nrows - 1 - eps)

    on_bound = _is_bound(dot)
    on_edge, iwhich, is_corner = is_on_edge(dot)
    local_hole = False
    if is_corner:
        icol, irow = nearest_node(dot, ncols, nrows)
        local_hole = is_node_hole(icol, irow, z, 4)

    sx, sy = slope_in_cell(z, dot.real, dot.imag)
    _sxlast, _sylast = (sx, sy) if (sx or sy) else (2e-5, 0.0)

    if is_corner:
        icol, irow = nearest_node(dot, ncols, nrows)
        smax2 = -1.0
        best_sx, best_sy = _sxlast, _sylast
        best_dot = dot
        cur = dot
        for _ in range(4):
            ok, cur = next_corner(cur)
            if not ok:
                break
            if not (0 <= cur.real <= ncols - 1 and 0 <= cur.imag <= nrows - 1):
                continue
            in_slope, stx, sty = inward_slope(cur, z)
            if not in_slope:
                continue
            s2 = stx ** 2 + sty ** 2
            if s2 > smax2:
                smax2 = s2
                best_sx, best_sy = stx, sty
                best_dot = cur

        if smax2 >= 0:
            moved, newdot = move_to_edge(best_dot, best_sx, best_sy)
            if not moved:
                return False, dot, on_bound, True
            dot = newdot
        else:
            # Fall back to steepest neighboring node
            f4 = get_adjacent4(z, icol, irow)
            zc = z[icol, irow]
            smax = -1.0
            imax = -1
            for i, fv in enumerate(f4):
                s = zc - fv
                if s >= 0 and s > smax:
                    smax, imax = s, i
            if imax >= 0:
                offsets = [(-1, 0), (0, -1), (1, 0), (0, 1)]
                dx, dy = offsets[imax]
                dot = complex(float(icol + dx), float(irow + dy))
    else:
        _, dot = edge_beside(dot)
        in_slope, sx, sy = inward_slope(dot, z)
        if not (sx or sy):
            sx, sy = _sxlast, _sylast
        if in_slope:
            moved, newdot = move_to_edge(dot, sx, sy)
            if not moved:
                return False, dot, on_bound, True
            dot = newdot
        else:
            # Valley: move toward lower endpoint of this edge
            on_edge2, iwhich2, _ = is_on_edge(dot)
            icell = min(ncols - 1, max(0, int(math.floor(dot.real))))
            jcell = min(nrows - 1, max(0, int(math.floor(dot.imag))))
            p1, p2 = side_of_cell(icell, jcell, iwhich2)
            icol1, irow1 = nearest_node(p1, ncols, nrows)
            icol2, irow2 = nearest_node(p2, ncols, nrows)
            dot = p1 if z[icol1, irow1] < z[icol2, irow2] else p2

    on_bound = _is_bound(dot)
    _, _, is_corner2 = is_on_edge(dot)
    local_hole = False
    if is_corner2:
        ic, ir = nearest_node(dot, ncols, nrows)
        ic = max(0, min(ic, ncols - 1))
        ir = max(0, min(ir, nrows - 1))
        local_hole = is_node_hole(ic, ir, z, 0)

    return True, dot, on_bound, local_hole


def dist_to_path(path: List[complex], ibeg: int, px: complex) -> Tuple[float, int]:
    """Return (min_distance, segment_index) from px to path starting at ibeg."""
    dismin = math.inf
    iseg = ibeg
    for i in range(ibeg, len(path) - 1):
        d = abs(dist_pt_seg(path[i], path[i + 1], px))
        if d < dismin:
            dismin, iseg = d, i
    return dismin, iseg


def down_slope(z: np.ndarray, source: complex, sealevel: float
               ) -> Tuple[bool, List[complex], bool]:
    """
    Trace downslope path from source to boundary or local minimum.
    Returns (ok, path, ended_at_low).
    """
    eps = 1e-5
    ncols, nrows = z.shape
    path = [source]

    sx, sy = slope_in_cell(z, source.real, source.imag)
    if not (sx or sy):
        sx, sy = 2 * eps, 0.0

    moved, dot = move_to_edge(source, sx, sy)
    if moved:
        path.append(dot)

    def _is_bound(d: complex) -> bool:
        return (d.real < eps or d.real > ncols - 1 - eps or
                d.imag < eps or d.imag > nrows - 1 - eps)

    on_bound = _is_bound(dot)
    on_edge, iwhich, is_corner = is_on_edge(dot)
    local_hole = False
    if is_corner:
        ic, ir = nearest_node(dot, ncols, nrows)
        local_hole = is_node_hole(ic, ir, z)

    max_steps = (nrows + ncols) * 10
    while not on_bound and not local_hole:
        if len(path) > max_steps:
            return False, path, False

        ok, dot, on_bound, local_hole = advance_down(z, dot)
        if not ok:
            return False, path, local_hole and not on_bound

        # Detect looping
        if len(path) >= 2 and (dot == path[-1] or dot == path[-2]):
            return False, path, True

        path.append(dot)
        on_bound = _is_bound(dot)

    ended_at_low = local_hole and not on_bound
    return True, path, ended_at_low


# ---------------------------------------------------------------------------
# Node-path conversion
# ---------------------------------------------------------------------------

def path_to_node_path(path: List[complex]) -> List[Tuple[int, int]]:
    """Convert a float path (on cell edges) to a sequence of integer nodes."""
    if not path:
        return []

    # Estimate ncols/nrows from path extent (generous upper bound)
    max_x = max(int(round(d.real)) for d in path) + 2
    max_y = max(int(round(d.imag)) for d in path) + 2

    icur, jcur = nearest_node(path[0], max_x, max_y)
    nodepath = [(icur, jcur)]
    irec = [2, 3, 0, 1]  # reciprocal sides
    isidelast = -1
    ibeg = 0

    while ibeg < len(path) - 2:
        dismin = 1.0
        isidemin = -1
        inext, jnext = icur, jcur

        for iside in range(4):
            if isidelast >= 0 and irec[isidelast] == iside:
                continue
            offsets = [(-1, 0), (0, -1), (1, 0), (0, 1)]
            inode = icur + offsets[iside][0]
            jnode = jcur + offsets[iside][1]
            px = complex(float(inode), float(jnode))
            dis, iseg = dist_to_path(path, ibeg, px)
            if dis < dismin:
                dismin, ibeg, isidemin = dis, iseg, iside
                inext, jnext = inode, jnode

        if dismin < 1.0 and isidemin >= 0:
            isidelast = isidemin
            icur, jcur = inext, jnext
            newnode = (icur, jcur)
            if newnode in nodepath:
                break
            nodepath.append(newnode)
        else:
            break
    return nodepath


def nodes_nearest_path(path: List[complex], numrows: int, numcols: int,
                       dist2max: float) -> List[List[Tuple[Tuple[int, int], float]]]:
    """
    For each path point, find all grid nodes within dist2max of it.
    Returns a list (per path point) of [(node, dist2), ...] sorted by dist2.
    """
    ipath: List[List[Tuple[Tuple[int, int], float]]] = [[] for _ in path]

    for irow in range(numrows):
        for icol in range(numcols):
            fnode = complex(float(icol), float(irow))
            best_d2 = math.inf
            best_i = 0
            for i, dot in enumerate(path):
                dx = dot.real - icol
                dy = dot.imag - irow
                d2 = dx * dx + dy * dy
                if d2 < best_d2:
                    best_d2, best_i = d2, i
            if best_d2 <= dist2max:
                ipath[best_i].append(((icol, irow), best_d2))

    for lst in ipath:
        lst.sort(key=lambda x: x[1])
    return ipath


# ---------------------------------------------------------------------------
# Channel cross-section helpers
# ---------------------------------------------------------------------------

def section_depth(width: float, depth: float, x2: float) -> float:
    """Channel depth at squared distance x2 from centreline (cosine profile)."""
    x = math.sqrt(max(0.0, x2))
    if abs(x / width) < 1.0:
        return depth * (math.cos(math.pi * x / width) + 1.0) / 2.0
    return 0.0


def section_width(flow: float) -> float:
    """Estimate channel width from discharge [m³/s]."""
    return math.sqrt(max(0.0, flow)) * 10.0


def section_depth_equiv(width: float) -> float:
    """Equivalent depth to erode from channel width."""
    return width / 30.0


# ---------------------------------------------------------------------------
# Build elevation profile along node-path
# ---------------------------------------------------------------------------

def build_profile(z: np.ndarray, nodepath: List[Tuple[int, int]]) -> np.ndarray:
    return np.array([z[ix, iy] for ix, iy in nodepath], dtype=np.float32)


# ---------------------------------------------------------------------------
# Advance channel: erode/deposit along profile and disperse
# ---------------------------------------------------------------------------

def advance_channel(top: np.ndarray, seq: Sequence,
                    path: List[complex],
                    ipath: List[List[Tuple[Tuple[int, int], float]]],
                    sealevel: float, cellside: float,
                    channel_factor: float, timeinc: float,
                    flowin: float, sedin: np.ndarray,
                    is_src_above_sl: bool, is_src_steady: bool):
    """Erode/deposit along channel and disperse fluvially."""
    import sedarch.sequence as _seq

    if flowin <= 0:
        return

    ncols, nrows = top.shape
    cellarea = cellside * cellside

    coefflinsec = (flowin ** 1.2) * 0.5
    coefflin = coefflinsec * 3.1e10        # m³/ka
    coeff = coefflin / cellside            # m²/ka

    # Temporarily scale transportability
    factor = 0.01
    _seq._trans = _seq._trans * factor if _seq._trans is not None else None
    old_trans = None
    if _seq._trans is not None:
        old_trans = _seq._trans.copy()
        _seq._trans *= factor

    width = section_width(flowin)   # m

    timeinc_std = 0.01  # ka
    deltat = 0.5 * cellarea / coeff if coeff > 0 else timeinc_std
    niter = max(1, int(timeinc_std / deltat))

    # Initial load vector (m height / ka)
    nsed = _seq._nsed
    load = np.zeros(nsed, dtype=np.float64)
    for is_ in range(nsed):
        load[is_] = sedin[is_] * 3.1e10 * timeinc_std / cellarea

    nodepath = path_to_node_path(path)
    if not nodepath:
        if old_trans is not None:
            _seq._trans = old_trans / factor
        return

    profile = build_profile(top, nodepath).astype(np.float64)
    profnew = profile.copy()

    fact_joint = (timeinc_std / niter) * coeff * 0.5 / cellarea

    for _ in range(niter):
        load_total = float(np.sum(load))
        # First node: add incoming load
        profnew[0] += load_total * timeinc_std / niter
        # Diffuse along profile
        for i in range(len(profile) - 1):
            ytrans = fact_joint * 0.5 * (profile[i] - profile[i + 1])
            profnew[i] -= ytrans
            profnew[i + 1] += ytrans
        profile = profnew.copy()

    # Apply elevation changes
    total_net = sum(profile[i] - top[nodepath[i][0], nodepath[i][1]]
                    for i in range(len(nodepath)))
    load_total = float(np.sum(load))
    loaditer = load.copy()
    if load_total > 0:
        loaditer *= total_net / load_total

    for i, (ix, iy) in enumerate(nodepath):
        hdiff = profile[i] - top[ix, iy]
        sv = seq(ix, iy)
        if hdiff < 0:
            sv.erode_thickness(loaditer, -hdiff)
        else:
            sv.deposit_thickness(loaditer, hdiff, coarse_first=True)  # matches C++ default
        top[ix, iy] = profile[i]

    # Fluvial dispersion around channel
    scoeff = 10000.0
    timediff = timeinc_std * 100.0
    if coeff > 0:
        numinct = max(1, int(8 * timediff * scoeff * _seq.max_trans()
                             / (cellside ** 2)) + 1)
    else:
        numinct = 1
    coeff_diff = timediff * scoeff / (cellside ** 2 * numinct)

    # Build distance-weighted coefficient map
    width_dif = width * 5.0 / cellside
    coeffs_map = np.zeros((ncols, nrows), dtype=np.float32)
    for ix in range(ncols):
        for iy in range(nrows):
            distmin2 = math.inf
            for (nx, ny) in nodepath:
                d2 = (nx - ix) ** 2 + (ny - iy) ** 2
                if d2 < distmin2:
                    distmin2 = d2
            if distmin2 < width_dif ** 2:
                ang = math.sqrt(distmin2) * math.pi / width_dif
                coeffs_map[ix, iy] = (math.cos(ang) + 1.0) / 2.0

    # Perform diffusion
    from sedarch.geology import _increment
    topnew = top.copy()
    for _ in range(numinct):
        _increment(topnew, seq, coeffs_map, coeff_diff, False)
    top[:] = topnew

    # Restore transportability
    if old_trans is not None:
        _seq._trans = old_trans / factor


# ---------------------------------------------------------------------------
# Channel lateral migration
# ---------------------------------------------------------------------------

def shift_path(path: List[complex], top: np.ndarray, deltxy: float,
               sealevel: float, dfact: float, afact: float, sfact: float,
               keep_last: bool) -> List[complex]:
    """Meander the channel path based on curvature and slope."""
    if len(path) < 2:
        return path

    _cumcurv = [0.0]
    _icall = getattr(shift_path, '_icall', 0)
    pi = math.pi
    perturbation = 0.2
    devinit = perturbation * math.sin(2.0 * pi * _icall / 16.0)
    _cumcurv[0] = devinit
    cumsfact = 0.9

    newpath = list(path)
    newpath[0] = path[0]
    if len(path) > 1:
        newpath[1] = path[1]

    sx0, sy0 = slope_in_cell(top, path[0].real, path[0].imag)
    cumslope = complex(sx0 / deltxy, sy0 / deltxy)

    belowsea = False
    ito = len(path) - 1
    if keep_last:
        ito -= 1

    for i in range(1, ito):
        if i < len(path) - 1:
            diff0 = path[i] - path[i - 1]
            diff1 = path[i + 1] - path[i]
            curv = vec_pro(diff0, diff1)
        else:
            curv = 0.0
        _cumcurv[0] = _cumcurv[0] * dfact + curv * (1.0 - dfact)
        dshift = -_cumcurv[0] * afact

        if i < len(path) - 1:
            diff = (path[i + 1] - path[i - 1]) / 2.0
        else:
            diff = path[i] - path[i - 1]
        shift = complex(-diff.imag, diff.real) * dshift

        sx, sy = slope_in_cell(top, path[i].real, path[i].imag)
        sx /= deltxy; sy /= deltxy
        cumslope = cumslope * cumsfact + complex(sx, sy) * (1.0 - cumsfact)

        mag_slope = abs(cumslope)
        unitvec = unit_vec(cumslope) if mag_slope > 1e-12 else 0j
        dis = mag_slope * sfact
        mag = max(1.0, -math.log10(dis)) if dis > 0 else 1.0
        weight = 0.5 / mag

        if i + 1 < len(path):
            newpath[i + 1] = (path[i + 1] + shift) * (1 - weight) + \
                              (path[i] + unitvec) * weight
            elev = value_in_cell(top, newpath[i + 1].real, newpath[i + 1].imag)
            if elev <= sealevel:
                belowsea = True
            elif belowsea:
                newpath[i + 1] = path[i + 1]

    if keep_last:
        newpath[-1] = path[-1]

    shift_path._icall = _icall + 1
    return newpath


def avulse_path(path: List[complex], width: float) -> List[complex]:
    """Cut out loops where path crosses itself within *width* cell-units."""
    width2 = width * width
    i = 0
    while i < len(path) - 2:
        away = False
        j = i + 1
        while j < len(path):
            d = path[i] - path[j]
            d2 = d.real ** 2 + d.imag ** 2
            if d2 < width2 and away:
                del path[i + 1:j]
                away = False
            else:
                away = True
                j += 1
        i += 1
    return path
