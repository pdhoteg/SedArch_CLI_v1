"""
geology.py – Geologic process implementations for SedArch.

Implements: dispersion (slope diffusion), growth (carbonate/aggradation),
wave energy, wave-induced sediment transport, vertical tectonics,
and the I/O helpers (read_input / write_input).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import sedarch.sequence as _seq
from sedarch.sequence import (
    Sequence, SedVector,
    interpolate_value, map_matrix_from_curve, slopes,
    get_adjacent4, get_adjacent_delt4, count_node_types,
    max_curve_val, max_curve_rate, is_all_zero,
    new_top_seq, remove_top_seq, merge_top, trim_all_tops,
    new_base_uniform, new_base_volume, write_sequence,
    set_num_sediments, set_sed_trans_from_diam, max_trans,
    limit_bottom,
)
from sedarch import vvm as vvmio


# ---------------------------------------------------------------------------
# Vars – all model variables
# ---------------------------------------------------------------------------

class Vars:
    """Container for all model parameters (mirrors C++ Vars class)."""

    def __init__(self):
        self.title: str = ''
        self.outfile: str = ''
        self.postprocess: str = ''
        self.cellside: float = 100.0
        self.xorigin: float = 0.0
        self.yorigin: float = 0.0
        self.rotation: float = 0.0

        # Sediment types
        self.sed_diameters: np.ndarray = np.array([], dtype=np.float32)
        self.sed_base_fracs: np.ndarray = np.array([], dtype=np.float32)

        # Time
        self.time: float = 0.0
        self.timeprev: float = -1e38
        self.sealevel: float = 0.0

        # Maps
        self.top: np.ndarray = np.array([[]], dtype=np.float32)
        self.tectonics: np.ndarray = np.array([], dtype=np.float32)
        self.sed: np.ndarray = np.array([], dtype=np.float32)   # Volume

        # Time control
        self.step_duration: float = 1.0
        self.step_count: float = 1.0

        # Sea level
        self.time_sealevel_curve: Optional[np.ndarray] = None

        # Growth
        self.growth_factor: np.ndarray = np.array([], dtype=np.float32)
        self.growth_sed_id: np.ndarray = np.array([], dtype=np.float32)
        self.level_growth_curve: List[Optional[np.ndarray]] = []
        self.growth_map: np.ndarray = np.array([], dtype=np.float32)
        self.time_growth_curve: List[Optional[np.ndarray]] = []
        self.erg_growth_curve: List[Optional[np.ndarray]] = []
        self.clarity_growth_curve: List[Optional[np.ndarray]] = []

        # Dispersion
        self.disp_factor: np.ndarray = np.array([], dtype=np.float32)
        self.level_disp_curve: List[Optional[np.ndarray]] = []
        self.time_disp_curve: List[Optional[np.ndarray]] = []
        self.disp_map: np.ndarray = np.array([], dtype=np.float32)

        # Channels
        self.channel_factor: np.ndarray = np.array([], dtype=np.float32)
        self.channel_rework_factor: np.ndarray = np.array([], dtype=np.float32)
        self.channel_event_period: np.ndarray = np.array([], dtype=np.float32)
        self.channel_event_duration: np.ndarray = np.array([], dtype=np.float32)
        self.channel_sources: List[np.ndarray] = []
        self.time_channel_curve: List[Optional[np.ndarray]] = []
        # SFLOW
        self.sflow_factor: np.ndarray = np.array([], dtype=np.float32)
        self.sflow_sed_factor: np.ndarray = np.array([], dtype=np.float32)
        self.time_sflow_curve: List[Optional[np.ndarray]] = []
        self.sflow_source_map: np.ndarray = np.array([], dtype=np.float32)
        self.sflow_depth_map: np.ndarray = np.array([], dtype=np.float32)
        self.sflow_velx_map: np.ndarray = np.array([], dtype=np.float32)
        self.sflow_vely_map: np.ndarray = np.array([], dtype=np.float32)

        # Waves
        self.wave_factor: np.ndarray = np.array([], dtype=np.float32)
        self.wave_sed_factor: float = 0.0
        self.wave_sources: List[np.ndarray] = []
        self.time_wave_curve: List[Optional[np.ndarray]] = []
        self.wave_cur_factor: np.ndarray = np.array([], dtype=np.float32)
        self.wave_cur_sed_factor: float = 0.0

        # Tectonics
        self.tect_factor: np.ndarray = np.array([], dtype=np.float32)
        self.tect_map: np.ndarray = np.array([], dtype=np.float32)
        self.time_tect_curve: List[Optional[np.ndarray]] = []

    def check_map_sizes(self) -> bool:
        nxt, nyt = self.top.shape
        if self.tectonics.size > 0:
            if self.tectonics.shape != self.top.shape:
                print(f'ERROR: Incompatible TECTONICS dimensions {self.tectonics.shape} vs TOP {self.top.shape}',
                      file=sys.stderr)
                return False
        if self.sed.size > 0:
            if self.sed.shape[:2] != (nxt, nyt):
                print(f'ERROR: Incompatible SED dimensions {self.sed.shape[:2]} vs TOP {(nxt,nyt)}',
                      file=sys.stderr)
                return False
        return True


# ---------------------------------------------------------------------------
# Register and read variables from a parsed VVM dict
# ---------------------------------------------------------------------------

def _to_curve(val) -> Optional[np.ndarray]:
    """Convert a parsed value to a 2-column curve array."""
    if val is None:
        return None
    arr = np.asarray(val, dtype=np.float32)
    if arr.ndim == 1 and len(arr) % 2 == 0:
        return arr.reshape(-1, 2)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return arr
    if arr.ndim == 2 and arr.shape[0] == 2:
        return arr.T
    return arr.reshape(-1, 2) if arr.size % 2 == 0 else None


def _to_vec(val) -> np.ndarray:
    if val is None:
        return np.array([], dtype=np.float32)
    return np.atleast_1d(np.asarray(val, dtype=np.float32)).flatten()


def _scalar(d: dict, key: str, default: float = 0.0) -> float:
    v = d.get(key)
    if v is None:
        return default
    return float(np.asarray(v).flat[0])


def _vec(d: dict, key: str) -> np.ndarray:
    return _to_vec(d.get(key))


def _layer(d: dict, key: str) -> np.ndarray:
    v = d.get(key)
    if v is None:
        return np.array([], dtype=np.float32)
    return np.asarray(v, dtype=np.float32)


def _curve(d: dict, key: str) -> Optional[np.ndarray]:
    return _to_curve(d.get(key))


def read_input(path: str, v: Vars, seq: Sequence) -> bool:
    """
    Parse a VVM input file, populate Vars ``v`` and Sequence ``seq``.
    Mirrors the C++ readInput() function.
    Returns True on success.
    """
    try:
        d = vvmio.parse_file(path)
    except Exception as e:
        print(f'ERROR reading {path}: {e}', file=sys.stderr)
        return False

    # Basic metadata
    v.title = str(d.get('TITLE', ''))
    v.outfile = str(d.get('OUTFILE', ''))
    v.postprocess = str(d.get('POSTPROCESS', ''))
    v.cellside = _scalar(d, 'CELLSIDE', 100.0)
    v.xorigin = _scalar(d, 'XORIGIN', 0.0)
    v.yorigin = _scalar(d, 'YORIGIN', 0.0)
    v.rotation = _scalar(d, 'ROTATION', 0.0)

    # Sediment diameters
    v.sed_diameters = _vec(d, 'SED_DIAMETERS')
    nsed = len(v.sed_diameters)
    if nsed == 0:
        print('ERROR: SED_DIAMETERS not found or empty', file=sys.stderr)
        return False
    set_num_sediments(nsed)
    set_sed_trans_from_diam(v.sed_diameters)

    # Time
    v.time = _scalar(d, 'TIME', 0.0)
    v.sealevel = _scalar(d, 'SEALEVEL', 0.0)
    v.step_duration = _scalar(d, 'STEP_DURATION', 1.0)
    v.step_count = _scalar(d, 'STEP_COUNT', 1.0)

    # Topography
    top_val = d.get('TOP')
    if top_val is None:
        print('ERROR: TOP map not found', file=sys.stderr)
        return False
    v.top = np.asarray(top_val, dtype=np.float32)
    if v.top.ndim == 1:
        v.top = v.top.reshape(-1, 1)

    # Tectonics
    tect_val = d.get('TECTONICS')
    if tect_val is not None:
        v.tectonics = np.asarray(tect_val, dtype=np.float32)

    # Sediment base fractions from SED1, SED2, ...
    nx, ny = v.top.shape
    sed_slices = []
    for is_ in range(nsed):
        key = f'SED{is_ + 1}'
        sv = d.get(key)
        if sv is not None:
            arr = np.asarray(sv, dtype=np.float32)
            if arr.ndim == 0:
                arr = np.full((nx, ny), float(arr), dtype=np.float32)
            elif arr.ndim == 1 and arr.size == 1:
                arr = np.full((nx, ny), float(arr[0]), dtype=np.float32)
            sed_slices.append(arr)
        else:
            sed_slices.append(np.full((nx, ny), 1.0 / nsed, dtype=np.float32))

    # Build SED volume (nx, ny, nsed)
    v.sed = np.stack(sed_slices, axis=2).astype(np.float32)
    # Normalise fractions per node
    total = v.sed.sum(axis=2, keepdims=True)
    total = np.where(total > 0, total, 1.0)
    v.sed /= total

    # Sea level curve
    v.time_sealevel_curve = _curve(d, 'TIME_SEALEVEL_CURVE')
    if v.time_sealevel_curve is None:
        # Build a trivial 1-point curve at current sea level
        v.time_sealevel_curve = np.array([[v.time, v.sealevel]], dtype=np.float32)

    # Dispersion
    v.disp_factor = _vec(d, 'DISP_FACTOR')
    _pad_groups(v, d, 'DISP')

    # Growth
    v.growth_factor = _vec(d, 'GROWTH_FACTOR')
    v.growth_sed_id = _vec(d, 'GROWTH_SED_ID')
    _pad_groups(v, d, 'GROWTH')

    # Channels
    v.channel_factor = _vec(d, 'CHANNEL_FACTOR')
    v.channel_rework_factor = _vec(d, 'CHANNEL_REWORK_FACTOR')
    v.channel_event_period = _vec(d, 'CHANNEL_EVENT_PERIOD')
    v.channel_event_duration = _vec(d, 'CHANNEL_EVENT_DURATION')
    _pad_vec_to(v.channel_rework_factor, len(v.channel_factor))
    _pad_vec_to(v.channel_event_period, len(v.channel_factor))
    _pad_vec_to(v.channel_event_duration, len(v.channel_factor))
    ngrp_chan = len(v.channel_factor)
    v.channel_sources = []
    v.mult = [1] * ngrp_chan
    for ig in range(ngrp_chan):
        key = f"CHANNEL_SOURCES{ig+1}" if ig > 0 else "CHANNEL_SOURCES"
        src = d.get(key, None)
        if src is not None:
            arr = np.array(src, dtype=np.float32)
            # Ensure shape (3+nsed, numsources):
            # If 1-D vector → single source, reshape to column
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)  # (n_props, 1)
            # If 2-D but wrong orientation (numsources rows, n_props cols) → transpose
            # We check: rows should be >= 3 for valid sources
            if arr.ndim == 2 and arr.shape[0] < 3 and arr.shape[1] >= 3:
                arr = arr.T
            v.channel_sources.append(arr)
        else:
            v.channel_sources.append(np.array([], dtype=np.float32).reshape(0, 0))

    # SFLOW
    v.sflow_factor = _vec(d, 'SFLOW_FACTOR')
    v.time_sflow_curve = [_curve(d, 'TIME_SFLOW_CURVE')] * len(v.sflow_factor)
    raw_smap = d.get('SFLOW_SOURCE_MAP', None)
    if raw_smap is not None:
        v.sflow_source_map = np.array(raw_smap, dtype=np.float32)
    else:
        v.sflow_source_map = np.array([], dtype=np.float32)

    # Waves
    v.wave_factor = _vec(d, 'WAVE_FACTOR')
    v.wave_sed_factor = _scalar(d, 'WAVE_SED_FACTOR', 0.0)
    v.wave_cur_factor = _vec(d, 'WAVE_CUR_FACTOR')
    v.wave_cur_sed_factor = _scalar(d, 'WAVE_CUR_SED_FACTOR', 0.0)

    # Tectonics groups
    v.tect_factor = _vec(d, 'TECT_FACTOR')
    v.tect_map = _layer(d, 'TECT_MAP')
    tect_tc = _curve(d, 'TIME_TECT_CURVE')
    ngrp_tect = len(v.tect_factor)
    v.time_tect_curve = [tect_tc] * ngrp_tect

    # Build initial sediment sequence
    new_base_volume(v.top, v.sed, seq)

    return True


def _pad_groups(v: Vars, d: dict, prefix: str):
    """Populate per-group curves for DISP or GROWTH based on VVM dict."""
    if prefix == 'DISP':
        ngroups = len(v.disp_factor)
        lc_key = 'LEVEL_DISP_CURVE'
        tc_key = 'TIME_DISP_CURVE'
        lc = _curve(d, lc_key)
        tc = _curve(d, tc_key)
        v.level_disp_curve = [lc] * ngroups
        v.time_disp_curve = [tc] * ngroups
        dm = d.get('DISP_MAP')
        if dm is not None:
            arr = np.asarray(dm, dtype=np.float32)
            if arr.ndim == 2:
                arr = arr[:, :, np.newaxis]
            v.disp_map = arr
        else:
            nx, ny = v.top.shape
            v.disp_map = np.ones((nx, ny, ngroups), dtype=np.float32)

    elif prefix == 'GROWTH':
        ngroups = len(v.growth_factor)
        lc = _curve(d, 'LEVEL_GROWTH_CURVE') or _curve(d, 'CARB_LEVEL_GROWTH')
        tc = _curve(d, 'TIME_GROWTH_CURVE')
        ec = _curve(d, 'ERG_GROWTH_CURVE')
        cc = _curve(d, 'CLARITY_GROWTH_CURVE')
        v.level_growth_curve = [lc] * ngroups
        v.time_growth_curve = [tc] * ngroups
        v.erg_growth_curve = [ec or np.array([], dtype=np.float32)] * ngroups
        v.clarity_growth_curve = [cc or np.array([], dtype=np.float32)] * ngroups
        gm = d.get('GROWTH_MAP')
        if gm is not None:
            arr = np.asarray(gm, dtype=np.float32)
            if arr.ndim == 2:
                arr = arr[:, :, np.newaxis]
            v.growth_map = arr
        else:
            nx, ny = v.top.shape
            v.growth_map = np.ones((nx, ny, ngroups), dtype=np.float32)


def _pad_vec_to(vec: np.ndarray, n: int) -> np.ndarray:
    if len(vec) < n:
        return np.zeros(n, dtype=np.float32)
    return vec


def write_initial_state(v: Vars, seq: Sequence):
    """Write the initial state (before any cycles) to the output file."""
    if not v.outfile:
        return
    vvmio.clear_file(v.outfile)
    entries = [
        ('TITLE', v.title),
        ('TIME', float(v.time)),
        ('SEALEVEL', float(v.sealevel)),
        ('TOP', v.top),
    ]
    vvmio.write_file(v.outfile, entries)
    write_sequence(v.outfile, 'SED', seq)


def write_step(v: Vars, seq: Sequence,
               wave_erg: Optional[np.ndarray] = None,
               tectonics: Optional[np.ndarray] = None):
    """Append one time-step output to the output file."""
    if not v.outfile:
        return
    entries: list = [
        ('', ''),
        ('TIME', float(v.time)),
        ('SEALEVEL', interpolate_value(v.time_sealevel_curve, v.time, v.sealevel)),
        ('TOP', v.top),
    ]
    vvmio.write_file(v.outfile, entries)
    write_sequence(v.outfile, 'SED', seq)
    if wave_erg is not None:
        vvmio.write_file(v.outfile, [('WAVE_ERG', wave_erg)])
    if tectonics is not None:
        vvmio.write_file(v.outfile, [('TECTONICS', tectonics)])


# ---------------------------------------------------------------------------
# Dispersion (slope diffusion)
# ---------------------------------------------------------------------------

def move_sed_disp(top: np.ndarray, seq: Sequence,
                  cellside: float, timebeg: float, timeend: float,
                  time_sealevel_curve: Optional[np.ndarray],
                  disp_factor: np.ndarray,
                  level_disp_curve: list,
                  time_disp_curve: list,
                  disp_map: np.ndarray) -> bool:
    """Slope-driven sediment diffusion (mirrors C++ moveSedDisp)."""
    ngroups = len(disp_factor)
    if is_all_zero(disp_factor):
        return True

    timeinc = timeend - timebeg
    time = (timebeg + timeend) / 2.0
    sealev_start = interpolate_value(time_sealevel_curve, timebeg)
    sealev_end = interpolate_value(time_sealevel_curve, timeend)

    scoeff = 100.0

    # Determine number of sub-increments for stability
    max_disp_sum = 0.0
    for ig in range(ngroups):
        max_lev = max_curve_val(level_disp_curve[ig]) if level_disp_curve[ig] is not None else 1.0
        if max_lev <= 0:
            max_lev = 1.0
        dm = disp_map[:, :, ig].max() if disp_map.size > 0 else 1.0
        if dm <= 0:
            dm = 1.0
        factor = float(disp_factor[ig]) * interpolate_value(time_disp_curve[ig], time, 1.0)
        max_disp_sum += factor * max_lev * dm

    numinct = max(5, int(4 * timeinc * scoeff * max_trans() * max_disp_sum
                         / (cellside * cellside)))
    coeff = timeinc * scoeff / (cellside * cellside * numinct)

    # Add auxiliary top layer
    new_top_seq(seq)
    topnew = top.copy()

    for inct in range(numinct):
        sl = sealev_start + (sealev_end - sealev_start) / numinct * (inct + 0.5)

        # Build combined coefficient map from the ORIGINAL top (mirrors C++ `group_map = top`)
        # C++ never updates `top` inside the loop; only `topnew` evolves.
        coeffs_map = np.zeros_like(top)
        for ig in range(ngroups):
            group_map = np.ones_like(top)
            if level_disp_curve[ig] is not None and len(level_disp_curve[ig]) > 0:
                group_map = top - sl  # relative to sea level, using original top
                map_matrix_from_curve(level_disp_curve[ig], group_map)
            if disp_map.size > 0 and disp_map.ndim >= 3:
                group_map *= disp_map[:, :, ig]
            if time_disp_curve[ig] is not None:
                tval = interpolate_value(time_disp_curve[ig], time, 1.0)
                group_map *= tval
            group_map *= float(disp_factor[ig])
            coeffs_map += group_map

        _increment(topnew, seq, coeffs_map, coeff)

    remove_top_seq(seq)
    top[:] = topnew  # in-place update
    return True


def _increment(topused: np.ndarray, seq: Sequence,
               coeffs_map: np.ndarray, coeff: float, merge: bool = True):
    """
    Single diffusion sub-step (mirrors C++ increment()).
    Topological sort ensures upstream-to-downstream ordering.
    """
    nx, ny = topused.shape
    topnew = topused.copy()

    numunin, numunou, nodestodo = count_node_types(topused)

    load = np.zeros(_seq._nsed, dtype=np.float32)
    partialload = np.zeros(_seq._nsed, dtype=np.float32)
    dxdy = [(-1, 0), (0, -1), (1, 0), (0, 1)]

    while nodestodo:
        nodesnext = []
        for (ix, iy) in nodestodo:
            coeff_c = float(coeffs_map[ix, iy]) if coeffs_map.size > 0 else 1.0
            diff4 = get_adjacent_delt4(topused, ix, iy)  # neighbor - center
            # diff4 < 0 means neighbor is lower → outflow from center
            outflow4 = np.where(diff4 < 0, -diff4 * coeff * coeff_c, 0.0)
            equivtot = float(outflow4.sum())

            load[:] = 0
            if equivtot > 0:
                topnew[ix, iy] += seq(ix, iy).erode(load, equivtot, True)

            for it, (dx, dy) in enumerate(dxdy):
                nx2, ny2 = ix + dx, iy + dy
                if 0 <= nx2 < nx and 0 <= ny2 < ny and outflow4[it] > 0:
                    partialload[:] = load * (outflow4[it] / equivtot)
                    topnew[nx2, ny2] += seq(nx2, ny2).deposit_all(partialload)
                    if numunin[nx2, ny2] > 0:
                        numunin[nx2, ny2] -= 1
                    if numunin[nx2, ny2] == 0 and numunou[nx2, ny2] > 0:
                        nodesnext.append((nx2, ny2))

            numunou[ix, iy] = 0

        nodestodo = nodesnext

    if merge:
        merge_top(seq)
    topused[:] = topnew


# ---------------------------------------------------------------------------
# Growth (carbonate / aggradation)
# ---------------------------------------------------------------------------

def growth(top: np.ndarray, seq: Sequence,
           timebeg: float, timeend: float,
           time_sealevel_curve: Optional[np.ndarray],
           growth_factor: np.ndarray,
           growth_sed_id: np.ndarray,
           level_growth_curve: list,
           growth_map: np.ndarray,
           time_growth_curve: list,
           erg_growth_curve: list,
           erg_map: Optional[np.ndarray],
           clarity_growth_curve: list,
           clarity_map: Optional[np.ndarray]) -> bool:
    """In-situ sediment growth (carbonates, etc.)."""
    ngroups = len(growth_factor)
    timeinc = timeend - timebeg
    timeavg = (timebeg + timeend) / 2.0
    nx, ny = top.shape

    factors = []
    for ig in range(ngroups):
        f = float(growth_factor[ig]) * timeinc * interpolate_value(time_growth_curve[ig], timeavg, 1.0)
        factors.append(f)

    # Number of sub-increments
    toprange = float(top.max() - top.min())
    numinct = 100
    if toprange > 0:
        ncells = max(nx, ny)
        searate = max_curve_rate(time_sealevel_curve)
        numinct = int(ncells * searate * timeinc / toprange * 2)
    numinct = max(1, min(numinct, 1000))

    load = np.zeros(_seq._nsed, dtype=np.float32)

    for inct in range(numinct):
        time = timebeg + timeinc / numinct * (inct + 0.5)
        sea = interpolate_value(time_sealevel_curve, time)

        for iy in range(ny):
            for ix in range(nx):
                rel_level = top[ix, iy] - sea
                load[:] = 0
                for ig in range(ngroups):
                    id1 = int(round(float(growth_sed_id[ig]))) - 1  # 0-based
                    id1 = max(0, min(id1, _seq._nsed - 1))
                    lg = factors[ig] / numinct
                    # Depth factor
                    lg *= interpolate_value(level_growth_curve[ig], rel_level, 1.0)
                    # Map factor
                    if (growth_map.size > 0 and growth_map.ndim >= 3
                            and growth_map.shape[0] == nx and growth_map.shape[1] == ny):
                        lg *= float(growth_map[ix, iy, ig])
                    # Wave energy factor
                    if (erg_map is not None and erg_map.size > 0
                            and erg_map.shape == (nx, ny)):
                        erg = float(erg_map[ix, iy])
                        lg *= interpolate_value(erg_growth_curve[ig], erg, 1.0)
                    load[id1] += lg

                total_load = float(load.sum())
                if total_load > 0:
                    seq(ix, iy).aggrade(load)
                    top[ix, iy] += total_load

    return True


# ---------------------------------------------------------------------------
# Wave energy
# ---------------------------------------------------------------------------

def wave(top: np.ndarray, seq: Sequence,
         cellside: float, timebeg: float, timeend: float,
         time_sealevel_curve: Optional[np.ndarray],
         wave_factor: np.ndarray,
         wave_sources: list,
         time_wave_curve: list,
         wave_erg_map: np.ndarray,
         wave_ergvx_map: np.ndarray,
         wave_ergvy_map: np.ndarray) -> bool:
    """Compute wave energy field (simplified plane wave propagation)."""
    ngroups = len(wave_factor)
    sealev_start = interpolate_value(time_sealevel_curve, timebeg)
    sealev_end = interpolate_value(time_sealevel_curve, timeend)
    sealev = (sealev_start + sealev_end) / 2.0
    time = (timebeg + timeend) / 2.0

    wave_erg_map[:] = 0
    wave_ergvx_map[:] = 0
    wave_ergvy_map[:] = 0

    for ig in range(ngroups):
        factor = float(wave_factor[ig]) * interpolate_value(time_wave_curve[ig], time, 1.0)
        if factor == 0 or wave_sources[ig].size == 0:
            continue
        src = wave_sources[ig]  # shape (3, nsrc): row 0=azim, row 1=amplit, row 2=period
        nsrc = src.shape[1] if src.ndim > 1 else 1
        for isrc in range(nsrc):
            if src.ndim > 1:
                azim = float(src[0, isrc])
                amplit = float(src[1, isrc])
                period = float(src[2, isrc])
            else:
                azim = float(src[0])
                amplit = float(src[1])
                period = float(src[2])
            _wave_source(top, cellside, sealev, factor * amplit, azim, period,
                         wave_erg_map, wave_ergvx_map, wave_ergvy_map)

    return True


def _wave_source(top: np.ndarray, cellside: float, sealev: float,
                 amplit: float, azim_deg: float, period: float,
                 erg: np.ndarray, ergvx: np.ndarray, ergvy: np.ndarray):
    """Add wave energy from a single source (plane wave, simplified)."""
    nx, ny = top.shape
    # Direction of wave propagation (azimuth from North, clockwise)
    azim_rad = np.radians(azim_deg)
    dx = np.sin(azim_rad)
    dy = np.cos(azim_rad)

    for iy in range(ny):
        for ix in range(nx):
            depth = sealev - top[ix, iy]
            if depth <= 0:
                continue
            # Simple depth-shoaling: energy ∝ amplit^2 / depth^0.5
            base_erg = amplit ** 2 / max(depth ** 0.5, 0.01) * 0.01
            erg[ix, iy] += base_erg
            ergvx[ix, iy] += base_erg * dx
            ergvy[ix, iy] += base_erg * dy


def move_sed_waves(top: np.ndarray, seq: Sequence,
                   cellside: float, timebeg: float, timeend: float,
                   time_sealevel_curve: Optional[np.ndarray],
                   wave_sed_factor: float,
                   wave_erg_map: np.ndarray) -> bool:
    """Move sediments driven by wave energy gradient."""
    if wave_erg_map is None or wave_erg_map.size == 0:
        return False
    if wave_sed_factor == 0:
        return True

    timeinc = timeend - timebeg
    scoeff = 100.0
    max_d = float(wave_erg_map.max()) * wave_sed_factor
    numinct = max(5, int(4 * timeinc * scoeff * max_trans() * max_d / (cellside * cellside)))
    coeff = timeinc * scoeff / (cellside * cellside * numinct) * wave_sed_factor

    new_top_seq(seq)
    topused = wave_erg_map.copy()
    topnew = topused.copy()

    for _ in range(numinct):
        _increment(topnew, seq, wave_erg_map, coeff)

    remove_top_seq(seq)
    top += topnew - topused
    return True


# ---------------------------------------------------------------------------
# Vertical tectonics
# ---------------------------------------------------------------------------

def vertical_tectonics(tectonics: np.ndarray,
                       timebeg: float, timeend: float,
                       tect_factor: np.ndarray,
                       tect_map: np.ndarray,
                       time_tect_curve: list) -> bool:
    """Accumulate vertical tectonic displacement (uplift/subsidence)."""
    tectonics[:] = 0
    ngroups = len(tect_factor)
    time = (timebeg + timeend) / 2.0
    timeinc = timeend - timebeg

    for ig in range(ngroups):
        factor = float(tect_factor[ig]) * timeinc
        if len(time_tect_curve) > ig and time_tect_curve[ig] is not None:
            factor *= interpolate_value(time_tect_curve[ig], time, 1.0)
        if tect_map.size > 0 and tect_map.ndim >= 3:
            sl = tect_map[:, :, ig] if tect_map.shape[2] > ig else tect_map[:, :, 0]
            tectonics += sl * factor
        elif tect_map.size > 0 and tect_map.ndim == 2:
            tectonics += tect_map * factor
        else:
            tectonics += factor

    return True


# ---------------------------------------------------------------------------
# Channel module
# ---------------------------------------------------------------------------

def channels(top: np.ndarray, seq: Sequence,
             cellside: float,
             timebeg: float, timeend: float,
             time_sealevel_curve,
             channel_factor: np.ndarray,
             channel_event_period: np.ndarray,
             channel_event_duration: np.ndarray,
             mult: list,
             channel_sources: list,
             ipath_state: list,
             load_held: list) -> bool:
    """
    Process all channel groups for one time step.

    Parameters
    ----------
    channel_sources : list of np.ndarray, shape (3+nsed, numsources)
        Each column: [x_m, y_m, flow_m3s, sed0_m3s, ...]
    ipath_state : list (modified in place)
        Cached path data for re-use between steps.
    load_held : list (modified in place)
        Held sediment load from previous step.
    """
    from sedarch.paths import (
        down_slope, round_path, smooth_path, resample_path,
        nodes_nearest_path, advance_channel, section_width,
    )
    from sedarch.basin import find_sub_basins, fill_sub_basins
    import sedarch.sequence as _seq

    timeinc = timeend - timebeg
    time = (timebeg + timeend) / 2.0
    sealev_start = interpolate_value(time_sealevel_curve, timebeg, 0.0)
    sealev_end   = interpolate_value(time_sealevel_curve, timeend,  0.0)
    sealevavg = (sealev_start + sealev_end) * 0.5

    ncols, nrows = top.shape
    numgroups = len(channel_factor)

    for igroup in range(numgroups):
        ok = _channel_group(
            top, seq, cellside, timeinc,
            sealev_start, sealev_end,
            float(channel_factor[igroup]),
            float(channel_event_period[igroup]),
            float(channel_event_duration[igroup]),
            mult[igroup],
            channel_sources[igroup],
        )
        if not ok:
            print(f"WARNING: CHANNEL module: error in group {igroup + 1}",
                  file=__import__('sys').stderr)
    return True


def _channel_group(top: np.ndarray, seq: Sequence,
                   cellside: float, timeinc: float,
                   sealev_start: float, sealev_end: float,
                   channel_factor: float,
                   channel_event_period: float,
                   channel_event_duration: float,
                   mult: int,
                   sources: np.ndarray) -> bool:
    """Process one channel group."""
    import math
    from sedarch.paths import (
        down_slope, round_path, smooth_path, resample_path,
        nodes_nearest_path, advance_channel, section_width,
        value_in_cell,
    )
    from sedarch.basin import find_sub_basins, fill_sub_basins
    import sedarch.sequence as _seq

    if mult <= 0:
        return True

    sealevavg = (sealev_start + sealev_end) * 0.5
    ncols, nrows = top.shape

    # sources shape: (3+nsed, numsources)
    if sources is None or sources.size == 0:
        return True
    numsources = sources.shape[1] if sources.ndim == 2 else 1
    nsed = _seq._nsed

    niter = max(1, min(100, int(timeinc / 5.0)))

    # Parse source positions, flows, sed inputs
    source_pos = []  # list of complex
    flowin = []
    sedin_all = []   # list of np.ndarray

    for isrc in range(numsources):
        col = sources[:, isrc]
        xpos = float(col[0]) / cellside
        ypos = float(col[1]) / cellside
        if xpos < 0 or xpos > ncols or ypos < 0 or ypos > nrows:
            print(f"WARNING: CHANNEL source {isrc} out of bounds", file=__import__('sys').stderr)
            return False
        source_pos.append(complex(xpos, ypos))
        flowin.append(float(col[2]))
        sed_in = np.zeros(nsed, dtype=np.float64)
        for is_ in range(nsed):
            if col.size > is_ + 3:
                sed_in[is_] = float(col[is_ + 3]) / niter
        sedin_all.append(sed_in)

    # Flow event timing
    flowtime_sec = float(channel_event_duration)
    flowfact = 1.0
    if flowtime_sec <= 0:
        flowtime_sec = timeinc / mult * 3.1e10
    if flowtime_sec < 1e4:
        flowfact = flowtime_sec / 1e4
        flowtime_sec = 1e4
    flowtime = flowtime_sec / 3.1e10  # ka

    for _iter in range(niter):
        for isrc in range(numsources):
            src = source_pos[isrc]
            flow = flowin[isrc] * flowfact * channel_factor

            topused = top.copy()
            is_src_above_sl = (value_in_cell(top, src.real, src.imag) > sealevavg)
            is_src_steady = (channel_event_period == 0 and channel_event_duration == 0)

            sealevused = -1e30
            if is_src_steady:
                limit_bottom(topused, sealevavg, 0.1)
            if is_src_above_sl or is_src_steady:
                sealevused = sealevavg

            # Fill depressions
            subbas = find_sub_basins(topused, src)
            fill_time = max(0.0, flowtime - 0.01)
            fill_sub_basins(topused, seq, subbas, cellside, fill_time,
                            sedin_all[isrc])

            width = section_width(flow)
            dfact = 1.0 - 1.0 / (1.0 + width / cellside)
            afact = 1.0
            sfact = 0.1 if is_src_above_sl else 0.01

            for imult in range(mult):
                # Trace downslope path
                ok, path, ended_at_low = down_slope(topused, src, sealevused)
                if not path:
                    continue
                path = round_path(path)
                path = smooth_path(path)
                _, path = resample_path(path, 1.0, nrows, ncols, False)
                if not path:
                    continue

                # Lateral migration
                factnshift = (1.0 if dfact < 0.5
                              else 0.5 / (1.0 - dfact))
                numshift = max(1, int(factnshift * len(path)))
                from sedarch.paths import shift_path, avulse_path
                for _s in range(numshift):
                    path = shift_path(path, topused, cellside, sealevavg,
                                      dfact, afact, sfact, ended_at_low)
                    path = avulse_path(path, width / cellside)
                    _, path = resample_path(path, 1.0, nrows, ncols, False)
                    if not path:
                        break

                if not path:
                    continue

                ipath = nodes_nearest_path(path, nrows, ncols, 100.0)

                advance_channel(top, seq, path, ipath,
                                sealevused, cellside, channel_factor,
                                flowtime, flow, sedin_all[isrc],
                                is_src_above_sl, is_src_steady)

    return True


# ---------------------------------------------------------------------------
# Sheet flow (SFLOW) module
# ---------------------------------------------------------------------------

def sflow(top: np.ndarray, cellside: float,
          timebeg: float, timeend: float,
          time_sealevel_curve,
          sflow_factor: np.ndarray,
          time_sflow_curve: list,
          sflow_source_map: np.ndarray,
          sflow_depth_map: np.ndarray,
          sflow_velx_map: np.ndarray,
          sflow_vely_map: np.ndarray) -> bool:
    """
    Compute steady-state sheet-flow depth and velocity using Manning's equation.

    Parameters
    ----------
    sflow_source_map : ndarray, shape (ncols, nrows, ngroups)
        Source discharge per cell [m³/s / cellside].
    sflow_depth_map : ndarray (ncols, nrows), modified in-place
        Water depth [m].
    sflow_velx_map, sflow_vely_map : ndarray (ncols, nrows), modified in-place
        Depth-averaged velocity components [m/s].
    """
    numgroups = len(sflow_factor)
    if numgroups == 0 or np.all(sflow_factor == 0):
        return True
    if sflow_source_map is None or sflow_source_map.size == 0:
        return True

    timeinc = timeend - timebeg
    time = (timebeg + timeend) / 2.0
    sealevel = interpolate_value(time_sealevel_curve, time, 0.0)

    ncols, nrows = top.shape

    # Build combined source map
    src = np.zeros((ncols, nrows), dtype=np.float64)
    for ig in range(numgroups):
        if sflow_factor[ig] <= 0:
            continue
        curve_val = 1.0
        if time_sflow_curve and ig < len(time_sflow_curve):
            curve_val = interpolate_value(time_sflow_curve[ig], time, 1.0)
        if sflow_source_map.ndim == 3 and ig < sflow_source_map.shape[2]:
            src += sflow_source_map[:, :, ig] * float(sflow_factor[ig]) * curve_val
        elif sflow_source_map.ndim == 2:
            src += sflow_source_map * float(sflow_factor[ig]) * curve_val

    ok = _calc_sflow(top, cellside, sealevel, src,
                     sflow_depth_map, sflow_velx_map, sflow_vely_map)
    return ok


def _calc_sflow(top: np.ndarray, cellside: float, sealevel: float,
                src: np.ndarray,
                dep: np.ndarray,
                velx: np.ndarray,
                vely: np.ndarray) -> bool:
    """Iterative Manning's-equation sheet-flow solver."""
    import math

    ncols, nrows = top.shape

    topused = top.copy()
    limit_bottom(topused, sealevel, 0.0)

    if dep.shape != top.shape:
        dep.resize((ncols, nrows))
        dep[:] = 0.0
    if velx.shape != top.shape:
        velx.resize((ncols, nrows))
        velx[:] = 0.0
    if vely.shape != top.shape:
        vely.resize((ncols, nrows))
        vely[:] = 0.0

    dep2 = dep.copy()
    manning = 0.03
    depdiff_eps = 0.001
    max_itersup = 4
    max_iter = 1000
    factor = 1.0

    for _itersup in range(max_itersup):
        depdiffmax = float('inf')
        for _iter in range(max_iter):
            if depdiffmax <= depdiff_eps:
                break
            depdiffmax = 0.0
            for irow in range(1, nrows - 1):
                for icol in range(1, ncols - 1):
                    sc = src[icol, irow] / cellside
                    zc = float(topused[icol, irow])
                    z4 = np.array([topused[icol-1, irow], topused[icol, irow-1],
                                   topused[icol+1, irow], topused[icol, irow+1]],
                                  dtype=np.float64)
                    d4 = np.array([dep[icol-1, irow], dep[icol, irow-1],
                                   dep[icol+1, irow], dep[icol, irow+1]],
                                  dtype=np.float64)
                    depnew = _flow_dep4(manning, cellside, sc, zc,
                                        float(dep[icol, irow]), z4, d4)
                    if not math.isfinite(depnew):
                        return False
                    depnew = dep[icol, irow] + (depnew - dep[icol, irow]) * factor
                    depdiffmax = max(depdiffmax, abs(depnew - dep[icol, irow]))
                    dep2[icol, irow] = depnew
            dep[:] = dep2
        factor *= 0.1

    # Compute x velocities (between adjacent column nodes)
    for irow in range(nrows):
        zprev = float(topused[0, irow])
        dprev = float(dep[0, irow])
        for icol in range(1, ncols):
            z = float(topused[icol, irow])
            d = float(dep[icol, irow])
            velx[icol - 1, irow] = _vel_tie(manning, cellside, zprev, z, dprev, d, 0.0)
            zprev, dprev = z, d
    # Average ties to nodes
    _tie_to_node(velx, True)
    # Zero out where depth is zero
    velx[dep == 0] = 0.0

    # Compute y velocities
    for icol in range(ncols):
        zprev = float(topused[icol, 0])
        dprev = float(dep[icol, 0])
        for irow in range(1, nrows):
            z = float(topused[icol, irow])
            d = float(dep[icol, irow])
            vely[icol, irow - 1] = _vel_tie(manning, cellside, zprev, z, dprev, d, 0.0)
            zprev, dprev = z, d
    _tie_to_node(vely, False)
    vely[dep == 0] = 0.0

    return True


def _flow_dep4(manning: float, cellside: float, sc: float, zc: float,
               dhint: float, z4: np.ndarray, d4: np.ndarray) -> float:
    """Find water depth at center node using Manning's equation (binary search)."""
    import math

    if np.all(d4 == 0) and sc <= 0:
        return 0.0

    h4 = z4 + d4   # water levels at neighbors

    eps = 0.001
    if sc == 0:
        hmin = max(float(h4.min()), zc)
        hmax = max(float(h4.max()), zc)
    elif sc > 0:
        hmin = max(float(h4.min()), zc)
        hmax = math.inf
    else:
        hmin = zc
        hmax = float(h4.max())

    if abs(hmax - hmin) < eps:
        return (hmax + hmin) / 2.0 - zc

    # Find hmax if unbounded
    hused = hmin + 1.0 if math.isinf(hmax) else hmax
    in_bounds = not math.isinf(hmax)
    while not in_bounds:
        dc = hused - zc
        flowin = _flow4o(manning, cellside, sc, zc, dc, z4, d4)
        if flowin < 0:
            hmax = hused
            in_bounds = True
        else:
            hused += hused - hmin

    # Binary search
    for _ in range(50):
        if abs(hmax - hmin) <= eps:
            break
        hused = (hmin + hmax) / 2.0
        dc = hused - zc
        flowin = _flow4o(manning, cellside, sc, zc, dc, z4, d4)
        if flowin < 0:
            hmax = hused
        else:
            hmin = hused

    return (hmax + hmin) / 2.0 - zc


def _flow4o(manning: float, cellside: float, sc: float, zc: float,
            dc: float, z4: np.ndarray, d4: np.ndarray) -> float:
    """Net inflow using origin-node depth (Manning's equation)."""
    import math
    hc = zc + dc
    h4 = z4 + d4
    s4 = (h4 - hc) / cellside   # slopes toward center
    td4 = np.where(s4 > 0, d4, dc)  # upstream depth
    f4 = np.copysign(
        np.sqrt(np.abs(s4)) * np.power(np.maximum(td4, 0), 1.666667) / manning,
        s4
    )
    return float(f4.sum()) + sc


def _vel_tie(manning: float, cellside: float,
             zc: float, z1: float, dc: float, d1: float, aslope: float) -> float:
    """Velocity at a tie between two nodes."""
    import math
    depmean = (dc + d1) / 2.0
    if depmean <= 0:
        return 0.0
    slope = ((zc + dc) - (z1 + d1)) / cellside + aslope
    return math.copysign(
        math.sqrt(abs(slope)) * (depmean ** 0.6666667) / manning, slope
    )


def _tie_to_node(field: np.ndarray, is_x: bool):
    """Average tie values to nodes (in-place)."""
    # For x-ties: field has ncols-1 valid values per row → average with neighbor
    # Simplified: average adjacent ties
    temp = field.copy()
    ncols, nrows = field.shape
    if is_x:
        for icol in range(ncols):
            left  = temp[icol - 1, :] if icol > 0 else temp[icol, :]
            right = temp[icol,     :] if icol < ncols - 1 else temp[icol, :]
            field[icol, :] = (left + right) / 2.0
    else:
        for irow in range(nrows):
            below = temp[:, irow - 1] if irow > 0 else temp[:, irow]
            above = temp[:, irow]     if irow < nrows - 1 else temp[:, irow]
            field[:, irow] = (below + above) / 2.0
