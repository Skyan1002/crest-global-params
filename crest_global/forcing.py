"""Forcing for dCREST: precipitation + 2 m temperature -> Snow-17 liquid input and PET on 0.1-degree cells.

Identical to the global training pipeline (CREST_Global_v260629: pub10/rebuild_melt_v2.py MELTV=3 and
verify/build_gg_cache.py), so parameters trained there apply unchanged:

  Snow-17     default global parameters (DEFAULT_SNOW17), 24 h step, pressure from cell elevation,
              Anderson's seasonal melt factor with the southern hemisphere shifted by half a year,
              single dt/6 on the heat-exchange term (the two EF5 defects fixed; MELTV=3)
  PET         temperature-based (Oudin-type): max(0.408 Ra (T + 5) / 100, 0), Ra extraterrestrial
              radiation from latitude and day of year; returned in mm/h
  temperature ERA5 t2m bilinear to the cell centre, minus 6.5 K/km times (cell elevation - ERA5-grid
              elevation proxy); the per-cell elevation difference `dz` is part of each basin file

Readers for the daily products used in training are included (IMERG V07B Final 3B-DAY, MSWEP V2.8 daily,
ERA5 daily-mean t2m on the 0.25-degree grid). Cells are given as (R, C) on the canonical IMERG 0.1-degree
grid: row R from the north (lat = 89.95 - 0.1 R), column C from the date line (lon = -179.95 + 0.1 C).
"""
import math

import numpy as np
import torch

from . import snow17 as S17

DEFAULT_SNOW17 = {"uadj": 0.05, "mbase": 0.0, "mfmax": 1.05, "mfmin": 0.30, "tipm": 0.1,
                  "nmf": 0.15, "plwhc": 0.04, "scf": 1.0, "pxtemp": 1.0}


def cell_latlon(R, C):
    return 89.95 - 0.1 * np.asarray(R), -179.95 + 0.1 * np.asarray(C)


def _doy(dates):
    d = np.asarray(dates).astype("datetime64[D]")
    return (d - d.astype("datetime64[Y]").astype("datetime64[D]")).astype(int) + 1


def snow_melt(precip_mm_day, t2m_c, elev_m, lat, dates):
    """Snow-17 liquid water input (rain + melt), mm/h, shape (T, Nc)."""
    P = np.asarray(precip_mm_day, np.float32); TM = np.asarray(t2m_c, np.float32)
    T, n = P.shape
    jday = _doy(dates)
    p = {k: torch.full((n,), v, dtype=torch.float64) for k, v in DEFAULT_SNOW17.items()}
    p_atm = S17.p_atm_from_elevation(torch.tensor(np.asarray(elev_m, np.float64)))
    st = S17.init_states(n, torch.float64)
    shift = torch.tensor((np.asarray(lat) < 0) * 183.0, dtype=torch.float64)
    ati, wq, wi, dfc = st["ati"], st["wq"], st["wi"], st["deficit"]
    Pt, Tt = torch.from_numpy(P), torch.from_numpy(TM)
    out = np.zeros((T, n), np.float32)
    with torch.no_grad():
        for t in range(T):
            stp = {"ati": ati, "wq": wq, "wi": wi, "deficit": dfc}
            m, _, stp = S17.step(p, stp, shift + float(jday[t]), Pt[t].to(torch.float64) / 24.0,
                                 Tt[t].to(torch.float64), 24.0, p_atm, fix_sv=True, fix_dt6=True)
            ati, wq, wi, dfc = stp["ati"], stp["wq"], stp["wi"], stp["deficit"]
            out[t] = m.numpy()
    return out


def pet_oudin(t2m_c, lat, dates):
    """Temperature-based PET, mm/h, shape (T, Nc)."""
    doy = _doy(dates).astype(float)
    latr = np.radians(np.asarray(lat, float))
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365.0)
    dec = 0.409 * np.sin(2 * np.pi * doy / 365.0 - 1.39)
    ws = np.arccos(np.clip(-np.tan(latr)[None, :] * np.tan(dec)[:, None], -1, 1))
    Ra = (24 * 60 / np.pi) * 0.0820 * dr[:, None] * (
        ws * np.sin(latr)[None, :] * np.sin(dec)[:, None]
        + np.cos(latr)[None, :] * np.cos(dec)[:, None] * np.sin(ws))
    return (np.maximum(0.408 * Ra * (np.asarray(t2m_c, np.float32) + 5.0) / 100.0, 0.0) / 24.0).astype(np.float32)


# ---------------------------------------------------------------- readers
def read_imerg_day(path, R, C):
    """IMERG V07 3B-DAY (Final or Late) daily precipitation at the cells, mm/day."""
    import h5py
    with h5py.File(path, "r") as h:
        a = h["precipitation"][0]                     # (3600 lon, 1800 lat), latitude ascending
    v = a[np.asarray(C), 1799 - np.asarray(R)].astype(np.float32)
    v[~np.isfinite(v) | (v < -100)] = 0.0
    return v


def read_mswep_day(path, R, C):
    """MSWEP V2.8 daily file (0.1 degree) at the cells, mm/day."""
    import netCDF4
    with netCDF4.Dataset(path) as d:
        name = [n for n in ("precipitation", "precip", "pr") if n in d.variables][0]
        a = np.asarray(d.variables[name][0] if d.variables[name].ndim == 3 else d.variables[name][:], np.float32)
        lat = d.variables["lat"][:] if "lat" in d.variables else d.variables["latitude"][:]
    if lat[0] < lat[-1]:
        a = a[::-1]
    return np.nan_to_num(a[np.asarray(R), np.asarray(C)], nan=0.0)


def era5_t2m_at_cells(t2m_k_025, R, C, dz):
    """One day of ERA5 t2m (K) on the 0.25-degree grid (721 lat from 90N, 1440 lon from 0E) -> deg C at
    the cells, bilinear plus a 6.5 K/km lapse over dz (cell elevation minus ERA5-grid elevation proxy, m)."""
    lat, lon = cell_latlon(R, C)
    lat25 = np.linspace(90, -90, 721); lon25 = np.linspace(-180, 179.75, 1440)
    yi = np.clip(np.searchsorted(-lat25, -lat) - 1, 0, 719)
    xi = np.clip(np.searchsorted(lon25, lon) - 1, 0, 1438)
    wy = (lat25[yi] - lat) / 0.25; wx = (lon - lon25[xi]) / 0.25
    t25 = np.roll(np.asarray(t2m_k_025, np.float32), 720, axis=1)   # 0..360E -> -180..180
    v = ((1 - wy) * (1 - wx) * t25[yi, xi] + (1 - wy) * wx * t25[yi, xi + 1]
         + wy * (1 - wx) * t25[yi + 1, xi] + wy * wx * t25[yi + 1, xi + 1])
    return (v - 273.15 - 0.0065 * np.asarray(dz)).astype(np.float32)


def model_inputs(precip_mm_day, t2m_c, elev_m, lat, dates):
    """(melt, pet) in mm/h for model.simulate."""
    return snow_melt(precip_mm_day, t2m_c, elev_m, lat, dates), pet_oudin(t2m_c, lat, dates)
