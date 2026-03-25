"""
vvm.py – VVM (Keyword = Value Matrix) file parser and writer for SedArch.

VVM format rules
----------------
- Each entry has the form:   KEYWORD = <values>
- Values can be scalars, 1-D vectors, 2-D matrices (rows separated by blank
  lines), 3-D volumes (groups separated by double blank lines), or quoted
  strings.
- Comments begin with '!' and extend to end-of-line.
- INCLUDE = "path"  recursively includes another file.
- Paths in INCLUDE directives are relative to the directory of the file that
  contains the directive.

Public API
----------
VVMReader  – low-level token / line reader used by parse_file()
parse_file(path) -> dict[str, Any]
write_value(stream, keyword, value) – append one keyword=value block
write_file(path, entries)           – write/append a list of (keyword, value)
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_comment(line: str) -> str:
    """Remove trailing comment (starts with '!') respecting quoted strings."""
    in_quote = False
    for i, ch in enumerate(line):
        if ch == '"':
            in_quote = not in_quote
        elif ch == '!' and not in_quote:
            return line[:i]
    return line


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Reader (low-level)
# ---------------------------------------------------------------------------

class VVMReader:
    """Iterate over logical tokens/lines from a VVM file with INCLUDE support."""

    def __init__(self, path: Union[str, Path], parent_dir: Optional[Path] = None):
        self.path = Path(path)
        if not self.path.is_absolute() and parent_dir is not None:
            self.path = parent_dir / self.path
        self.path = self.path.resolve()
        self._parent_dir = self.path.parent
        self._lines: List[str] = []
        self._pos = 0
        self._load()

    def _load(self):
        with open(self.path, encoding='utf-8', errors='replace') as fh:
            raw_lines = fh.readlines()

        expanded: List[str] = []
        for raw in raw_lines:
            line = _strip_comment(raw).rstrip()
            # Check for INCLUDE
            m = re.match(r'^\s*INCLUDE\s*=\s*"([^"]+)"', line, re.IGNORECASE)
            if m:
                inc_path = Path(m.group(1))
                if not inc_path.is_absolute():
                    inc_path = self._parent_dir / inc_path
                inc_reader = VVMReader(inc_path)
                expanded.extend(inc_reader._lines)
            else:
                expanded.append(line)
        self._lines = expanded

    def peek(self) -> Optional[str]:
        if self._pos < len(self._lines):
            return self._lines[self._pos]
        return None

    def next_line(self) -> Optional[str]:
        if self._pos < len(self._lines):
            line = self._lines[self._pos]
            self._pos += 1
            return line
        return None

    def has_more(self) -> bool:
        return self._pos < len(self._lines)

    def backup(self):
        if self._pos > 0:
            self._pos -= 1


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _read_values(reader: VVMReader) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Read numeric values or a quoted string that follow an '=' sign.

    Numeric data layout:
      - Values on the same line form one row (x-direction / columns).
      - Each new line is a new row (y-direction / rows).
      - A single blank line separates z-layers within one group.
      - A double blank line separates w-groups.

    Returns (ndarray, None) or (None, str) or (None, None).
    """
    # Skip leading blanks; detect quoted string
    while reader.has_more():
        line = reader.peek()
        if line is None:
            return None, None
        stripped = line.strip()
        if not stripped:
            reader.next_line()
            continue
        if stripped.startswith('"'):
            reader.next_line()
            # Remove surrounding quotes
            val = stripped.strip()
            if val.startswith('"') and val.endswith('"'):
                val = val[1:-1]
            return None, val
        break
    else:
        return None, None

    # groups[iw][iz] = list of rows;  each row = list of floats
    groups: List[List[List[List[float]]]] = [[[]]]   # 1 group, 1 layer, 0 rows
    blank_count = 0

    while reader.has_more():
        line = reader.next_line()
        if line is None:
            break
        stripped = line.strip()

        # New keyword → stop
        if '=' in stripped:
            eq_pos = stripped.find('=')
            quote_pos = stripped.find('"')
            if quote_pos == -1 or eq_pos < quote_pos:
                reader.backup()
                break

        if not stripped:
            blank_count += 1
            continue

        # Try to parse as numbers
        tokens = stripped.split()
        numbers: List[float] = []
        all_numeric = True
        for tok in tokens:
            if _is_number(tok):
                numbers.append(float(tok))
            else:
                reader.backup()
                all_numeric = False
                break
        if not all_numeric:
            break

        # Decide what the blanks mean
        if blank_count >= 2:
            groups.append([[]])     # new group
        elif blank_count == 1:
            groups[-1].append([])   # new layer within group
        blank_count = 0

        groups[-1][-1].append(numbers)  # append row

    # Nothing read?
    if not groups or not groups[0] or not groups[0][0]:
        return None, None

    # Build ndarray from groups structure
    nw = len(groups)
    nz = max(len(g) for g in groups)
    ny = max(len(layer) for g in groups for layer in g)
    nx = max((len(row) for g in groups for layer in g for row in layer), default=0)

    if nx == 0:
        return None, None

    if nw == 1 and nz == 1 and ny == 1 and nx == 1:
        return np.float32(groups[0][0][0][0]), None

    if nw == 1 and nz == 1 and ny == 1:
        # 1-D vector: a single row
        return np.array(groups[0][0][0], dtype=np.float32), None

    if nw == 1 and nz == 1:
        # 2-D matrix: shape (nx, ny)
        arr = np.zeros((nx, ny), dtype=np.float32)
        for iy, row in enumerate(groups[0][0]):
            for ix, val in enumerate(row):
                arr[ix, iy] = val
        return arr, None

    if nw == 1:
        # 3-D volume: shape (nx, ny, nz)
        arr = np.zeros((nx, ny, nz), dtype=np.float32)
        for iz, layer in enumerate(groups[0]):
            for iy, row in enumerate(layer):
                for ix, val in enumerate(row):
                    arr[ix, iy, iz] = val
        return arr, None

    # 4-D: shape (nx, ny, nz, nw)
    arr = np.zeros((nx, ny, nz, nw), dtype=np.float32)
    for iw, group in enumerate(groups):
        for iz, layer in enumerate(group):
            for iy, row in enumerate(layer):
                for ix, val in enumerate(row):
                    arr[ix, iy, iz, iw] = val
    return arr, None


def parse_file(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Parse a VVM file and return a dict mapping keyword -> value.

    Values are:
      - np.float32 scalar
      - np.ndarray of float32 (1-D, 2-D, 3-D, or 4-D)
      - str  (quoted string)

    When a keyword appears multiple times the values are collected into a
    list under that keyword (e.g. multiple SED1 blocks).
    """
    reader = VVMReader(path)
    result: Dict[str, Any] = {}

    while reader.has_more():
        line = reader.next_line()
        if line is None:
            break
        line_stripped = line.strip()
        if not line_stripped:
            continue

        # Look for 'KEYWORD ='
        if '=' not in line_stripped:
            continue
        eq_pos = line_stripped.find('=')
        quote_pos = line_stripped.find('"')
        if quote_pos != -1 and quote_pos < eq_pos:
            continue  # '=' is inside a string

        keyword = line_stripped[:eq_pos].strip().upper()
        if not keyword:
            continue

        # Put remaining text after '=' back for value reading
        remainder = line_stripped[eq_pos + 1:].strip()
        if remainder:
            # Insert as next line
            reader._lines.insert(reader._pos, remainder)

        arr, sval = _read_values(reader)
        value: Any = sval if arr is None else arr

        if keyword in result:
            existing = result[keyword]
            if not isinstance(existing, list):
                result[keyword] = [existing]
            result[keyword].append(value)
        else:
            result[keyword] = value

    return result


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def _fmt_float(v: float) -> str:
    """Format a float compactly."""
    s = f'{v:.6g}'
    return s


def write_value(stream, keyword: str, value: Any):
    """Append one 'KEYWORD = ...' block to an open text stream."""
    if not keyword:
        stream.write('\n')
        return

    stream.write(f'\n{keyword} = ')

    if isinstance(value, str):
        stream.write(f'"{value}"\n')
        return

    if isinstance(value, (int, float, np.floating, np.integer)):
        stream.write(f'{_fmt_float(float(value))}\n')
        return

    arr = np.asarray(value, dtype=np.float32)

    if arr.ndim == 0:
        stream.write(f'{_fmt_float(float(arr))}\n')
    elif arr.ndim == 1:
        stream.write(' '.join(_fmt_float(v) for v in arr) + '\n')
    elif arr.ndim == 2:
        nx, ny = arr.shape
        stream.write('\n')
        for iy in range(ny):
            row = ' '.join(_fmt_float(arr[ix, iy]) for ix in range(nx))
            stream.write(f'  {row}\n')
    elif arr.ndim == 3:
        nx, ny, nz = arr.shape
        stream.write('\n')
        for iz in range(nz):
            if iz > 0:
                stream.write('\n')  # blank line between layers
            for iy in range(ny):
                row = ' '.join(_fmt_float(arr[ix, iy, iz]) for ix in range(nx))
                stream.write(f'  {row}\n')
    elif arr.ndim == 4:
        nx, ny, nz, nw = arr.shape
        for iw in range(nw):
            if iw > 0:
                stream.write('\n\n')  # double blank line between groups
            for iz in range(nz):
                if iz > 0:
                    stream.write('\n')
                for iy in range(ny):
                    row = ' '.join(_fmt_float(arr[ix, iy, iz, iw]) for ix in range(nx))
                    stream.write(f'  {row}\n')
    else:
        raise ValueError(f'Cannot write array with ndim={arr.ndim}')


def write_file(path: Union[str, Path], entries: List[Tuple[str, Any]],
               append: bool = True):
    """Write (or append) a list of (keyword, value) pairs to a VVM file."""
    mode = 'a' if append else 'w'
    with open(path, mode, encoding='utf-8') as fh:
        for keyword, value in entries:
            write_value(fh, keyword, value)


def clear_file(path: Union[str, Path]):
    """Clear (truncate) a VVM output file."""
    open(path, 'w').close()
