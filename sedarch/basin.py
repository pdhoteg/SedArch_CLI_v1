"""basin.py – Subbasin analysis ported from basin.cxx

Used by the CHANNEL module to route flow and fill depressions with sediment.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

import numpy as np

from sedarch.sequence import (
    SedVector, Sequence, interpolate_value,
)
from sedarch.paths import (
    advance_down, nearest_node, is_node_hole,
    slope_in_cell, move_to_edge, is_on_edge,
    get_adjacent_nodes4,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Bas:
    """One subbasin descriptor."""
    isill: Tuple[int, int] = (0, 0)    # Sill node (icol, irow)
    zsill: float = 0.0                  # Elevation of sill
    sill_on_boundary: bool = False
    idfm: int = -1                      # Index of parent (overflow-into) basin (-1 = none)


@dataclass
class SubBasins:
    """Collection of subbasins for a drainage path."""
    baslist: List[Bas] = field(default_factory=list)
    ibasmap: Optional[np.ndarray] = None   # (ncols, nrows) int32, -1 = unlabelled


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_bound(icol: int, irow: int, ncols: int, nrows: int) -> bool:
    return icol == 0 or icol == ncols - 1 or irow == 0 or irow == nrows - 1


def move_dot_to_low(dot: complex, z: np.ndarray) -> Tuple[bool, complex]:
    """
    Move dot downslope to local minimum. Returns (found_low, dot).
    Returns False if reached boundary.
    """
    eps = 1e-5
    ncols, nrows = z.shape

    sx, sy = slope_in_cell(z, dot.real, dot.imag)
    if not (sx or sy):
        sx, sy = 2 * eps, 0.0
    _, dot = move_to_edge(dot, sx, sy)

    dotprev = dotprevprev = dot
    max_count = (nrows + ncols) * 10

    for _ in range(max_count):
        ok, dot, on_bound, local_hole = advance_down(z, dot)
        if dot == dotprev or dot == dotprevprev:
            local_hole = True
        dotprevprev = dotprev
        dotprev = dot
        if on_bound:
            return False, dot
        if local_hole:
            break

    return True, dot   # local_hole must be True here


# ---------------------------------------------------------------------------
# Spread upward (BFS)
# ---------------------------------------------------------------------------

def _spread_up(idotstart: Tuple[int, int], z: np.ndarray, ceil: float,
               ibas: int, ibasfm: int, ibasmap: np.ndarray
               ) -> Tuple[Tuple[int, int], int]:
    """
    BFS flood-fill upward from idotstart, labelling nodes with ibas.
    Returns (sill_node, ibascov).
    """
    ncols, nrows = z.shape

    def _add_frontier(ibasfm_: int, ibasto_: int, frontline_: set):
        if ibasfm_ < 0 or ibasto_ < 0:
            return
        for iy in range(nrows):
            for ix in range(ncols):
                if ibasmap[ix, iy] >= 0:
                    continue
                for nx, ny in get_adjacent_nodes4(ix, iy):
                    if 0 <= nx < ncols and 0 <= ny < nrows:
                        if ibasfm_ <= ibasmap[nx, ny] <= ibasto_:
                            frontline_.add((ix, iy))
                            break

    frontline: Set[Tuple[int, int]] = set()
    _add_frontier(ibasfm, ibas, frontline)
    frontline.add(idotstart)

    ibascov = -1

    while frontline:
        frontline_new: Set[Tuple[int, int]] = set()
        for (ix, iy) in frontline:
            if not (0 <= ix < ncols and 0 <= iy < nrows):
                continue
            if z[ix, iy] >= ceil:
                continue
            if ibasmap[ix, iy] < 0:
                ibasmap[ix, iy] = ibas
            for nx, ny in get_adjacent_nodes4(ix, iy):
                if not (0 <= nx < ncols and 0 <= ny < nrows):
                    continue
                if (nx, ny) in frontline or (nx, ny) in frontline_new:
                    continue
                if ibasmap[nx, ny] >= 0:
                    if ibasmap[nx, ny] != ibas:
                        touched = ibasmap[nx, ny]
                        if ibascov < 0 or touched < ibascov:
                            ibascov = touched
                            _add_frontier(touched, ibas, frontline_new)
                    continue
                if z[nx, ny] > ceil:
                    continue
                if z[nx, ny] < z[ix, iy]:
                    continue   # Goes downhill from (ix,iy): skip
                frontline_new.add((nx, ny))
        frontline = frontline_new

    # Find sill: lowest in-basin node on model or basin boundary
    zmin_bound = math.inf
    idotsill = idotstart
    for iy in range(nrows):
        for ix in range(ncols):
            if ibasmap[ix, iy] != ibas:
                continue
            is_bound = _is_bound(ix, iy, ncols, nrows)
            if not is_bound:
                for nx, ny in get_adjacent_nodes4(ix, iy):
                    if 0 <= nx < ncols and 0 <= ny < nrows:
                        if ibasmap[nx, ny] < 0 and z[nx, ny] < z[ix, iy]:
                            is_bound = True
                            break
            if is_bound and z[ix, iy] < zmin_bound:
                zmin_bound = z[ix, iy]
                idotsill = (ix, iy)

    # Unlabel nodes above sill
    for iy in range(nrows):
        for ix in range(ncols):
            if ibasmap[ix, iy] == ibas and z[ix, iy] > zmin_bound:
                ibasmap[ix, iy] = -1

    return idotsill, ibascov


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def find_sub_basins(z: np.ndarray, source: complex) -> SubBasins:
    """
    Analyze surface z starting from source. Return SubBasins descriptor.
    """
    ncols, nrows = z.shape
    subbas = SubBasins()
    subbas.ibasmap = np.full((ncols, nrows), -1, dtype=np.int32)

    dot = source
    ibas = 0
    ibasfm = 0
    ceil = math.inf

    for _ in range(51):
        found, dot = move_dot_to_low(dot, z)
        if not found:
            break

        icol, irow = nearest_node(dot, ncols, nrows)
        idotsill, ibascov = _spread_up((icol, irow), z, ceil, ibas,
                                       ibasfm, subbas.ibasmap)
        zminbasin = z[icol, irow]
        zminbound = z[idotsill[0], idotsill[1]]

        bas = Bas()
        bas.isill = idotsill
        bas.zsill = zminbound
        bas.sill_on_boundary = _is_bound(idotsill[0], idotsill[1], ncols, nrows)
        if ibascov >= 0 and zminbound > subbas.baslist[ibascov].zsill:
            bas.idfm = ibascov
        else:
            bas.idfm = -1

        if bas.sill_on_boundary:
            subbas.baslist.append(bas)
            break

        # Find next start: neighbor of sill with lower elevation, not in basin
        idotsill_next = idotsill
        found_next = False
        for nx, ny in get_adjacent_nodes4(idotsill[0], idotsill[1]):
            if 0 <= nx < ncols and 0 <= ny < nrows:
                if z[nx, ny] < zminbound and subbas.ibasmap[nx, ny] < 0:
                    idotsill_next = (nx, ny)
                    found_next = True
                    break

        if not found_next:
            subbas.baslist.append(bas)
            break

        eps = 1e-5
        if abs(zminbound - zminbasin) < eps:
            dot = complex(float(idotsill_next[0]), float(idotsill_next[1]))
            ceil = math.inf
            continue

        dot = complex(float(idotsill_next[0]), float(idotsill_next[1]))
        ceil = math.inf
        ibas += 1
        ibasfm = ibas
        subbas.baslist.append(bas)

    return subbas


def find_volume(z: np.ndarray, subbas: SubBasins, cellside: float, ibas: int) -> float:
    """Compute fill volume of subbasin ibas (m³)."""
    ncols, nrows = z.shape
    ibasmap = subbas.ibasmap
    zsill = subbas.baslist[ibas].zsill

    ibasfm = ibas
    while ibasfm > 0 and subbas.baslist[ibasfm - 1].zsill < zsill:
        ibasfm -= 1
    zsill_prev = (-math.inf if ibasfm == ibas
                  else subbas.baslist[ibas - 1].zsill)

    volume = 0.0
    for iy in range(nrows):
        for ix in range(ncols):
            bl = ibasmap[ix, iy]
            if bl == ibas:
                volume += zsill - z[ix, iy]
            elif ibasfm <= bl < ibas:
                volume += zsill - zsill_prev
    return volume * cellside * cellside


def vol_curve(z: np.ndarray, subbas: SubBasins, cellside: float,
              ibas: int) -> np.ndarray:
    """Return elevation-volume curve (complex array) for subbasin ibas."""
    ncols, nrows = z.shape
    ibasmap = subbas.ibasmap
    zsill = subbas.baslist[ibas].zsill

    ibasfm = ibas
    while ibasfm > 0 and subbas.baslist[ibasfm - 1].zsill < zsill:
        ibasfm -= 1
    zsill_prev = (-math.inf if ibasfm == ibas
                  else subbas.baslist[ibas - 1].zsill)

    depths = []
    for iy in range(nrows):
        for ix in range(ncols):
            bl = ibasmap[ix, iy]
            if bl == ibas:
                depths.append(zsill - z[ix, iy])
            elif ibasfm <= bl < ibas:
                depths.append(zsill - zsill_prev)

    if not depths:
        return np.array([], dtype=complex)

    depths.sort(reverse=True)  # deepest first
    depmax = depths[0]
    depths = [depmax - d for d in depths]  # elevation above deepest

    cellarea = cellside * cellside
    pairs = []
    prev_vol = 0.0
    pairs.append(complex(0.0, zsill - depmax))
    for i in range(1, len(depths)):
        vol = prev_vol + float(i) * (depths[i] - depths[i - 1]) * cellarea
        p = complex(vol, zsill - depmax + depths[i])
        prev_vol = vol
        if p != pairs[-1]:
            pairs.append(p)

    return np.array(pairs, dtype=complex)


def fill_sub_basins(z: np.ndarray, seq: Sequence, subbas: SubBasins,
                    cellside: float, timeinc: float, sedin: np.ndarray):
    """Fill subbasins with inflowing sediment."""
    import sedarch.sequence as _seq

    if timeinc <= 0 or cellside <= 0:
        return

    nsed = _seq._nsed
    volsedin = sedin.astype(np.float64) * 3.1e10 * timeinc  # m³

    volsed_total = float(np.sum(volsedin))
    ncols, nrows = z.shape

    for ibas, bas in enumerate(subbas.baslist):
        volume = find_volume(z, subbas, cellside, ibas)
        if volume < volsed_total:
            # Fill entire subbasin to sill
            for iy in range(nrows):
                for ix in range(ncols):
                    bl = subbas.ibasmap[ix, iy]
                    if bl < 0:
                        continue
                    if bl == ibas or (bl <= ibas and
                                      subbas.baslist[bl].zsill < bas.zsill):
                        thick = bas.zsill - z[ix, iy]
                        if thick <= 0:
                            continue
                        added = seq(ix, iy).deposit_thickness(volsedin, thick,
                                                               proportional=False)
                        z[ix, iy] += added
            volsed_total -= volume
        else:
            # Partial fill
            curve = vol_curve(z, subbas, cellside, ibas)
            elev = interpolate_value(curve, volsed_total)
            for iy in range(nrows):
                for ix in range(ncols):
                    bl = subbas.ibasmap[ix, iy]
                    if bl < 0:
                        continue
                    if bl == ibas or (bl <= ibas and
                                      subbas.baslist[bl].zsill < bas.zsill):
                        thick = elev - z[ix, iy]
                        if thick <= 0:
                            continue
                        added = seq(ix, iy).deposit_thickness(volsedin, thick,
                                                               proportional=False)
                        z[ix, iy] += added
            volsed_total = 0.0
            break
