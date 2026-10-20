"""Run dCREST for any basin with the global parameter files.

  # 1. parameters (GitHub release assets)
  gh release download v1.0.0 -R Skyan1002/crest-global-params -p '*.nc' -D data
  # 2. cells of the basin from a polygon (GeoJSON, lon/lat)
  python scripts/run_basin.py cells basin.geojson basin_cells.npz
  # 3. forcing at those cells: IMERG V07B 3B-DAY files + ERA5 daily-mean t2m (netCDF, 0.25 degree)
  python scripts/run_basin.py forcing basin_cells.npz --imerg-dir IMERG/ --era5 'ERA5/t2m_*.nc' \
         --start 2001-01-01 --end 2020-12-31 forcing.npz
  # 4. simulate (spin up at least two years; the parameters were trained from 2001 with two warm-up years)
  python scripts/run_basin.py run basin_cells.npz forcing.npz data/crest_global_params_imerg_final_v1.nc q.csv

Precipitation must match the parameter set: IMERG Final forcing with the imerg_final file, MSWEP with mswep.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from crest_global import basin, forcing, model, params  # noqa: E402


def cmd_cells(a):
    from shapely.geometry import shape
    gj = json.load(open(a.polygon))
    feats = gj["features"] if gj.get("type") == "FeatureCollection" else [gj]
    from shapely.ops import unary_union
    geom = unary_union([shape(f["geometry"] if "geometry" in f else f) for f in feats])
    R, C, W = basin.cells_from_polygon(geom)
    import netCDF4
    with netCDF4.Dataset(a.static) as d:
        elev = np.asarray(d.variables["elev"][:], np.float32)[R, C]
        dz = np.asarray(d.variables["dz"][:], np.float32)[R, C]
    lat, lon = forcing.cell_latlon(R, C)
    area = float((basin.cell_area_km2(R) * W).sum())
    np.savez_compressed(a.out, R=R, C=C, w=W.astype(np.float32), lat=lat.astype(np.float32), lon=lon.astype(np.float32),
                        elev=elev, dz=dz, area_km2=np.float32(area))
    print(f"{len(R)} cells, area {area:.0f} km2 -> {a.out}")


def cmd_forcing(a):
    import netCDF4
    c = np.load(a.cells)
    dates = np.arange(np.datetime64(a.start), np.datetime64(a.end) + 1)
    P = np.zeros((len(dates), len(c["R"])), np.float32)
    for i, d in enumerate(dates):
        s = str(d).replace("-", "")
        f = glob.glob(f"{a.imerg_dir}/**/3B-DAY.MS.MRG.3IMERG.{s}-*.nc4", recursive=True)
        if not f:
            sys.exit(f"missing IMERG day {d}")
        P[i] = forcing.read_imerg_day(f[0], c["R"], c["C"])
    T = np.full((len(dates), len(c["R"])), np.nan, np.float32)
    for fn in sorted(glob.glob(a.era5)):
        with netCDF4.Dataset(fn) as ds:
            tv = ds.variables.get("time") or ds.variables.get("valid_time")
            t = netCDF4.num2date(tv[:], tv.units)
            var = ds.variables["t2m"]
            for j, tt in enumerate(t):
                d = np.datetime64(f"{tt.year:04d}-{tt.month:02d}-{tt.day:02d}")
                k = int((d - dates[0]).astype(int))
                if 0 <= k < len(dates):
                    T[k] = forcing.era5_t2m_at_cells(var[j], c["R"], c["C"], c["dz"])
    if np.isnan(T).any():
        sys.exit(f"ERA5 t2m missing on {int(np.isnan(T).any(1).sum())} days")
    np.savez_compressed(a.out, dates=dates.astype(str), precip=P, t2m=T)
    print(f"{len(dates)} days x {len(c['R'])} cells -> {a.out}")


def cmd_run(a):
    c = np.load(a.cells); f = np.load(a.forcing)
    dates = f["dates"].astype("datetime64[D]")
    fields, attrs = params.open_paramset(a.params)
    p = params.sample(fields, c["R"], c["C"])
    melt, pet = forcing.model_inputs(f["precip"], f["t2m"], c["elev"], c["lat"], dates)
    q = model.simulate(p, melt, pet, c["w"], float(c["area_km2"]))
    with open(a.out, "w") as fh:
        fh.write("date,q_mm_day,q_m3s\n")
        for d, v, w in zip(dates.astype(str), q, model.to_m3s(q, float(c["area_km2"]))):
            fh.write(f"{d},{v:.4f},{w:.3f}\n")
    print(f"{attrs.get('title', '')}\nmean Q after two warm-up years {q[730:].mean():.3f} mm/day -> {a.out}")


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
sub = ap.add_subparsers(dest="cmd", required=True)
s = sub.add_parser("cells"); s.add_argument("polygon"); s.add_argument("out")
s.add_argument("--static", default="data/crest_global_static_v1.nc")
s = sub.add_parser("forcing"); s.add_argument("cells"); s.add_argument("--imerg-dir", required=True)
s.add_argument("--era5", required=True); s.add_argument("--start", required=True); s.add_argument("--end", required=True)
s.add_argument("out")
s = sub.add_parser("run"); s.add_argument("cells"); s.add_argument("forcing"); s.add_argument("params"); s.add_argument("out")
a = ap.parse_args()
{"cells": cmd_cells, "forcing": cmd_forcing, "run": cmd_run}[a.cmd](a)
