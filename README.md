# SedArch Python

A Python code of **SedArch** – a simple sedimentary and stratigraphic forward 
modelling program for research and education, licensed under GPL v3.

---

## Features

| Module | Status |
|---|---|
| VVM file parser / writer | ✅ Full |
| Sediment column (SedVector / Sequence) | ✅ Full |
| Slope diffusion (DISPERSION) | ✅ Full |
| In-situ growth (GROWTH) | ✅ Full |
| Wave energy + sediment transport (WAVES) | ✅ Core |
| Vertical tectonics (TECTONICS) | ✅ Full |
| Channel flow (CHANNELS) | 🔲 Planned |
| Sheet flow (SFLOW) | 🔲 Planned |

---

## Quick Install

**Requirements:** Python 3.9+ and NumPy.

```bash
# From the directory containing pyproject.toml:
pip install .

# Or install in editable/developer mode:
pip install -e .

# Optional: install visualisation extras (matplotlib, scipy)
pip install ".[viz]"
```

---

## Running a model

```bash
# Using the installed command-line tool:
sedarch path/to/input.vvm

# Or via Python module:
python -m sedarch path/to/input.vvm
```

The program reads the input VVM file, runs the simulation, and writes results
to the file specified by the `OUTFILE` keyword in the input file.

---

## Input File Format (VVM)

VVM files use a simple `Variable = value` format.  Comments start with `!`.
Files can include other files with `INCLUDE = "path"`.

```vvm
TITLE = "My model"
OUTFILE = "output/result.vvm"
SED_DIAMETERS = 1  0.01  2          ! Grain diameters in mm
CELLSIDE = 100                       ! Cell side in metres
TIME_SEALEVEL_CURVE =               ! (time_ka, sea_level_m) pairs
  0  -50
  500  0

INCLUDE = "topo.vvm"                ! External topography file

STEP_DURATION = 20
STEP_COUNT = 100

DISP_FACTOR = 0.1
INCLUDE = "dispersion_curve.vvm"

GROWTH_FACTOR = 0.1
GROWTH_SED_ID = 3
INCLUDE = "growth_curve.vvm"
```

### Supported Variables

| Variables | Type | Description |
|---|---|---|
| `TITLE` | string | Model title |
| `OUTFILE` | string | Output file path |
| `POSTPROCESS` | string | Shell command to run after simulation |
| `SED_DIAMETERS` | vector | Grain diameters of each sediment type (mm) |
| `CELLSIDE` | scalar | Grid cell side length (m) |
| `XORIGIN`, `YORIGIN` | scalar | Grid origin coordinates |
| `TIME` | scalar | Initial model time (ka) |
| `SEALEVEL` | scalar | Initial sea level (m) |
| `TOP` | matrix | Initial topographic surface (m) |
| `SED1`, `SED2`, ... | matrix | Initial sediment proportions (0–1) |
| `STEP_DURATION` | scalar | Duration of each time step (ka) |
| `STEP_COUNT` | scalar | Number of time steps |
| `TIME_SEALEVEL_CURVE` | curve | 2-column (time, sea level) curve |
| `DISP_FACTOR` | vector | Dispersion coefficient(s) (m²/ka) |
| `LEVEL_DISP_CURVE` | curve | Dispersion multiplier vs. depth |
| `TIME_DISP_CURVE` | curve | Dispersion multiplier vs. time |
| `DISP_MAP` | volume | Spatial dispersion multiplier map |
| `GROWTH_FACTOR` | vector | Growth rate factor(s) (m/ka) |
| `GROWTH_SED_ID` | vector | Sediment type ID(s) to grow |
| `LEVEL_GROWTH_CURVE` | curve | Growth multiplier vs. depth |
| `TIME_GROWTH_CURVE` | curve | Growth multiplier vs. time |
| `GROWTH_MAP` | volume | Spatial growth multiplier map |
| `WAVE_FACTOR` | vector | Wave energy factor(s) |
| `WAVE_SED_FACTOR` | scalar | Wave-driven sediment transport factor |
| `TECT_FACTOR` | vector | Tectonic uplift/subsidence factor(s) |
| `TECT_MAP` | volume | Spatial tectonic rate map |
| `TIME_TECT_CURVE` | curve | Tectonic multiplier vs. time |
| `INCLUDE` | string | Path to an included VVM file |

---

## Output File Format

The output VVM file has the same format as the input file, with one block per
time step containing:

```
TIME = <float>
SEALEVEL = <float>
TOP = <matrix>
SED1 = <matrix>
SED2 = <matrix>
...
WAVE_ERG = <matrix>   ! if waves enabled
TECTONICS = <matrix>  ! if tectonics enabled
```

You can reload an output file as input to continue a simulation or for
post-processing.

---

## Python API

```python
from sedarch.vvm import parse_file, write_file
from sedarch.sequence import Sequence, interpolate_value
from sedarch.geology import Vars, read_input, move_sed_disp, growth

# Parse any VVM file into a dict
data = parse_file('my_model.vvm')

# Run a model programmatically
v = Vars()
seq = Sequence()
read_input('my_model.vvm', v, seq)
# ... inspect v.top, v.sed, v.time_sealevel_curve, etc.
```

---
## Licence

GNU General Public License v3.0 – see `LICENSE` for details.

## CHANNELS Module

Models fluvial channel erosion and deposition along downslope paths.

### Input Variable

| Variable | Description |
|---------|-------------|
| `CHANNEL_FACTOR` | Dimensionless channel intensity (one value per group) |
| `CHANNEL_EVENT_PERIOD` | Event period in seconds (0 = continuous) |
| `CHANNEL_EVENT_DURATION` | Event duration in seconds (0 = continuous) |
| `CHANNEL_SOURCES` | Source matrix: rows = [x_m, y_m, flow_m3s, sed1_m3s, …], cols = sources |

### Example

```
CHANNEL_FACTOR = 1.0
CHANNEL_EVENT_PERIOD = 0
CHANNEL_EVENT_DURATION = 0
CHANNEL_SOURCES =
500 1000
2500 2500
50 30
0.1 0.05
0.1 0.05
```

*Sources matrix: each column is one source; rows are x, y, flow [m³/s], sed1 [m³/s], ...*

### Algorithm

1. Fill depressions (subbasin detection + sedimentation)
2. Trace downslope path from each source
3. Apply lateral channel migration (meander kinematics)
4. Erode/deposit along the profile using diffusion-based transport
5. Spread fluvially around channel using cosine-weighted diffusion

---

## SFLOW Module

Computes steady-state sheet (overland) flow depth and velocity using Manning's equation.

### Input Variables

| Variables | Description |
|---------|-------------|
| `SFLOW_FACTOR` | Dimensionless flow intensity (one per group) |
| `SFLOW_SOURCE_MAP` | Discharge source map [m³/s per cell side], shape = grid |
| `TIME_SFLOW_CURVE` | Optional time-varying intensity multiplier |

### Example

```
SFLOW_FACTOR = 1.0
SFLOW_SOURCE_MAP =
0 0 0 0 0
5 5 5 5 5
5 5 5 5 5
0 0 0 0 0
0 0 0 0 0
```

### Algorithm

Iterative steady-state solver:
- Initialises water depth to zero
- Binary-search per node to satisfy Manning's continuity equation
- Computes x and y depth-averaged velocity fields
- Multiple outer iterations with decreasing relaxation factor for stability

Manning's roughness coefficient is fixed at n = 0.03 (adjustable in code).
