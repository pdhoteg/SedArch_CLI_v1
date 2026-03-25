"""
__main__.py – SedArch command-line entry point.

Usage:
    python -m sedarch input.vvm
    sedarch input.vvm           # if installed as a script
"""

from __future__ import annotations

import sys
import time as _time
from pathlib import Path
from typing import Optional

import numpy as np

from sedarch.vvm import write_file, clear_file, write_value
from sedarch.sequence import (
    Sequence, _nsed, interpolate_value, new_top_seq, write_sequence,
    set_num_sediments, set_sed_trans_from_diam,
)
from sedarch.geology import (
    channels, sflow,
    Vars, read_input, write_initial_state, write_step,
    move_sed_disp, growth, wave, move_sed_waves, vertical_tectonics,
)


def run(input_file: str, step_callback=None):
    print(f'SedArch Python, revision 1.0', file=sys.stderr)
    print(f'Input file = {input_file}', file=sys.stderr)
    print('=' * 60, file=sys.stderr)

    t_start = _time.time()

    # ------------------------------------------------------------------ #
    # 1. Read input
    # ------------------------------------------------------------------ #
    v = Vars()
    seq = Sequence()

    ok = read_input(input_file, v, seq)
    if not ok:
        print('ERROR: Failed to read input.', file=sys.stderr)
        sys.exit(1)

    if not v.check_map_sizes():
        sys.exit(1)

    nx, ny = v.top.shape

    print(f'Grid: {nx} (E-W) x {ny} (S-N) nodes, cellside = {v.cellside} m',
          file=sys.stderr)
    print(f'Total size: {(nx-1)*v.cellside:.0f} m x {(ny-1)*v.cellside:.0f} m',
          file=sys.stderr)
    print(f'Sea-level curve entries: '
          f'{len(v.time_sealevel_curve) if v.time_sealevel_curve is not None else 0}',
          file=sys.stderr)

    # Apply initial tectonics if present
    if v.tectonics.size > 0 and v.tectonics.shape == v.top.shape:
        v.top += v.tectonics

    # ------------------------------------------------------------------ #
    # 2. Initialise output file
    # ------------------------------------------------------------------ #
    if v.outfile:
        clear_file(v.outfile)
        write_initial_state(v, seq)

    # ------------------------------------------------------------------ #
    # 3. Determine which modules to run
    # ------------------------------------------------------------------ #
    n_disp = int(np.count_nonzero(v.disp_factor))
    n_growth = int(np.count_nonzero(v.growth_factor))
    n_channels = int(np.count_nonzero(v.channel_factor))
    n_waves = int(np.count_nonzero(v.wave_factor))
    n_tect = int(np.count_nonzero(v.tect_factor))

    print(f'\nModules enabled:', file=sys.stderr)
    if n_waves:
        print(f'  WAVES (groups={n_waves})', file=sys.stderr)
    if n_disp:
        print(f'  DISPERSION (groups={n_disp})', file=sys.stderr)
    if n_channels:
        print(f'  CHANNELS (groups={n_channels})', file=sys.stderr)
    if n_growth:
        print(f'  GROWTH (groups={n_growth})', file=sys.stderr)
    if n_tect:
        print(f'  TECTONICS (groups={n_tect})', file=sys.stderr)

    # Work-space maps (lazy allocation)
    wave_erg_map: Optional[np.ndarray] = None
    wave_ergvx_map: Optional[np.ndarray] = None
    wave_ergvy_map: Optional[np.ndarray] = None
    tectonics_map: Optional[np.ndarray] = None
    sflow_depth_map: Optional[np.ndarray] = None
    sflow_velx_map: Optional[np.ndarray] = None
    sflow_vely_map: Optional[np.ndarray] = None

    n_sflow = int(np.count_nonzero(v.sflow_factor)) if hasattr(v, "sflow_factor") else 0
    if n_sflow:
        print(f"  SFLOW (groups={n_sflow})", file=sys.stderr)

    # Channel path/load state (persisted across time steps)
    channel_ipath: list = []
    channel_load: list = []

    num_cycles = int(round(v.step_count))

    print(f'\nInitial time   = {v.time} ka', file=sys.stderr)
    print(f'Step duration  = {v.step_duration} ka', file=sys.stderr)
    print(f'Steps          = {num_cycles}', file=sys.stderr)
    print(f'Total duration = {num_cycles * v.step_duration} ka', file=sys.stderr)
    print(f'End time       = {v.time + num_cycles * v.step_duration} ka\n',
          file=sys.stderr)
    print('Running...', file=sys.stderr)

    # ------------------------------------------------------------------ #
    # 4. Main time loop
    # ------------------------------------------------------------------ #
    for icycle in range(num_cycles):
        timebeg = v.time
        timeend = v.time + v.step_duration

        print(f'  step {icycle+1}/{num_cycles}  t={timebeg:.1f}→{timeend:.1f} ka  \r',
              end='', file=sys.stderr, flush=True)

        sealev_start = interpolate_value(v.time_sealevel_curve, timebeg, v.sealevel)
        sealev_end = interpolate_value(v.time_sealevel_curve, timeend, v.sealevel)

        # Add empty working layer at top
        new_top_seq(seq)

        # -- WAVES --
        if n_waves > 0:
            if wave_erg_map is None:
                wave_erg_map = np.zeros((nx, ny), dtype=np.float32)
                wave_ergvx_map = np.zeros((nx, ny), dtype=np.float32)
                wave_ergvy_map = np.zeros((nx, ny), dtype=np.float32)
            else:
                wave_erg_map[:] = 0
                wave_ergvx_map[:] = 0
                wave_ergvy_map[:] = 0

            wave(v.top, seq,
                 v.cellside, timebeg, timeend,
                 v.time_sealevel_curve,
                 v.wave_factor, v.wave_sources, v.time_wave_curve,
                 wave_erg_map, wave_ergvx_map, wave_ergvy_map)

            move_sed_waves(v.top, seq, v.cellside, timebeg, timeend,
                           v.time_sealevel_curve, v.wave_sed_factor,
                           wave_erg_map)

        # -- DISPERSION --
        if n_disp > 0:
            move_sed_disp(v.top, seq, v.cellside, timebeg, timeend,
                          v.time_sealevel_curve,
                          v.disp_factor, v.level_disp_curve,
                          v.time_disp_curve, v.disp_map)

        # -- CHANNELS --
        if n_channels > 0 and hasattr(v, 'channel_factor') and v.channel_sources:
            channels(v.top, seq,
                     v.cellside, timebeg, timeend,
                     v.time_sealevel_curve,
                     v.channel_factor,
                     v.channel_event_period,
                     v.channel_event_duration,
                     v.mult,
                     v.channel_sources,
                     channel_ipath,
                     channel_load)

        # -- SFLOW --
        if n_sflow > 0 and hasattr(v, 'sflow_factor'):
            if sflow_depth_map is None:
                sflow_depth_map = np.zeros((nx, ny), dtype=np.float32)
                sflow_velx_map  = np.zeros((nx, ny), dtype=np.float32)
                sflow_vely_map  = np.zeros((nx, ny), dtype=np.float32)
            sflow(v.top, v.cellside, timebeg, timeend,
                  v.time_sealevel_curve,
                  v.sflow_factor,
                  v.time_sflow_curve,
                  v.sflow_source_map,
                  sflow_depth_map, sflow_velx_map, sflow_vely_map)

        # -- GROWTH --
        if n_growth > 0:
            growth(v.top, seq, timebeg, timeend,
                   v.time_sealevel_curve,
                   v.growth_factor, v.growth_sed_id,
                   v.level_growth_curve, v.growth_map,
                   v.time_growth_curve, v.erg_growth_curve,
                   wave_erg_map,
                   v.clarity_growth_curve, None)

        # -- TECTONICS --
        if n_tect > 0:
            if tectonics_map is None:
                tectonics_map = np.zeros((nx, ny), dtype=np.float32)
            # Subtract old tectonics before computing new
            v.top -= tectonics_map
            vertical_tectonics(tectonics_map, timebeg, timeend,
                                v.tect_factor, v.tect_map, v.time_tect_curve)

        # Advance time
        v.time = timeend

        # ------------------------------------------------------------------ #
        # 5. Write output
        # ------------------------------------------------------------------ #
        if v.outfile:
            write_step(v, seq,
                       wave_erg=wave_erg_map,
                       tectonics=tectonics_map)

        # Add tectonics to top for next cycle
        if n_tect > 0 and tectonics_map is not None:
            v.top += tectonics_map

        # GUI progress callback — returns True to cancel
        if step_callback is not None:
            cancel = step_callback(icycle + 1, num_cycles, v.time,
                                   v.top.copy(), sealev_end, seq)
            if cancel:
                print('\nCancelled by user.', file=sys.stderr)
                return

    print('\n', file=sys.stderr)

    # ------------------------------------------------------------------ #
    # 6. Postprocessing
    # ------------------------------------------------------------------ #
    if v.postprocess:
        import subprocess
        print(f'Postprocessing: {v.postprocess}', file=sys.stderr)
        result = subprocess.run(v.postprocess, shell=True)
        if result.returncode != 0:
            print(f'WARNING: Postprocessing returned code {result.returncode}',
                  file=sys.stderr)

    elapsed = _time.time() - t_start
    print(f'Run completed normally.', file=sys.stderr)
    print(f'Elapsed time = {elapsed:.2f} s', file=sys.stderr)


def main():
    if len(sys.argv) < 2:
        print('SedArch – Sedimentary and Stratigraphic Forward Modelling', file=sys.stderr)
        print('Usage: sedarch <inputfile.vvm>', file=sys.stderr)
        print('       python -m sedarch <inputfile.vvm>', file=sys.stderr)
        sys.exit(1)

    for input_file in sys.argv[1:]:
        run(input_file)


if __name__ == '__main__':
    main()
