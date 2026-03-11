import numpy as np
import xarray as xr
from datetime import datetime, timedelta
import re
from pathlib import Path
import gc # garbage collector

ctl_file = "cm1out_s.ctl"
datadir = Path(".")
byte_order = "<"  # likely little-endian

# === Parse .ctl file ===
with open(ctl_file) as f:
    lines = f.readlines()

def parse_levels(lines, key):
    for i, line in enumerate(lines):
        if line.lower().startswith(key):
            parts = line.split()
            if parts[2].lower() == "levels":
                count = int(parts[1])
                levels = []
                for j in range(i+1, i+1+count):
                    levels.append(float(lines[j].strip()))
                return np.array(levels)
            elif parts[2].lower() == "linear":
                count = int(parts[1])
                start = float(parts[3])
                step = float(parts[4])
                return np.linspace(start, start + (count - 1) * step, count)
    raise ValueError(f"{key} not found")

def parse_time_def(line):
    parts = line.strip().split()
    count = int(parts[1])
    start_str = parts[3]
    interval = parts[4].lower()
    base_time = datetime.strptime(start_str, "%H:%MZ%d%b%Y")
    delta = {
        "10mn": timedelta(minutes=10),
        "1hr": timedelta(hours=1),
        "1dy": timedelta(days=1),
        "1mo": timedelta(days=30),
        "1yr": timedelta(days=365)
    }[interval]
    return [base_time + i * delta for i in range(count)]

# Extract metadata
x = parse_levels(lines, "xdef")
y = parse_levels(lines, "ydef")
z = parse_levels(lines, "zdef")
times = []
for line in lines:
    if line.lower().startswith("tdef"):
        times = parse_time_def(line)
        break

# Dataset pattern
dset_pattern = None
for line in lines:
    if line.lower().startswith("dset"):
        dset_pattern = line.split("^")[1].strip()
        break
assert dset_pattern is not None, "No dset pattern found"

# Parse variables
in_vars = False
varnames = []
var_levels = []
for line in lines:
    if line.strip().lower().startswith("vars"):
        in_vars = True
        continue
    if line.strip().lower().startswith("endvars"):
        break
    if in_vars:
        parts = line.strip().split()
        varnames.append(parts[0])
        var_levels.append(int(parts[1]))

nz, ny, nx = len(z), len(y), len(x)
coords = {"xh": x, "yh": y, "zh": z}

# === Process each timestep
for t_index, t in enumerate(times, start=1):
    datfile = datadir / dset_pattern.replace("%6t", f"{t_index:06d}")
    print(f"Reading {datfile}")

    data = np.fromfile(datfile, dtype=byte_order + 'f4')

    expected = sum([(ny * nx if lev == 0 else nz * ny * nx) for lev in var_levels])
    if data.size != expected:
        raise RuntimeError(f"File size mismatch: {datfile} has {data.size}, expected {expected}")

    ds = xr.Dataset(coords={k: coords[k] for k in coords if k in ["xh", "yh", "zh"]})
    ds = ds.expand_dims(time=[t])
    i = 0
    for name, lev in zip(varnames, var_levels):
        if lev == 0:
            size = ny * nx
            arr = data[i:i+size].reshape((ny, nx))
            ds[name] = (("yh", "xh"), arr)
        else:
            size = nz * ny * nx
            arr = data[i:i+size].reshape((nz, ny, nx))
            ds[name] = (("zh", "yh", "xh"), arr)
        i += size

    outname = f"cm1_t{t_index:06d}.nc"
    ds.to_netcdf(outname)
    print(f"✔ Saved {outname}")

    # Free up memory
    del ds
    del data
    gc.collect()
