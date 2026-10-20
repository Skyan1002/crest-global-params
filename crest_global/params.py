"""Read the global parameter files and sample them at basin cells."""
import numpy as np

from .model import ALL_KEYS

PRODUCTS = ("imerg_final", "mswep")


def open_paramset(path):
    """Global parameter file -> dict name -> (1800, 3600) float32 array (NaN off land), plus attributes."""
    import netCDF4
    with netCDF4.Dataset(path) as d:
        fields = {k: np.asarray(d.variables[k][:], np.float32) for k in d.variables
                  if k not in ("lat", "lon")}
        attrs = {k: d.getncattr(k) for k in d.ncattrs()}
    return fields, attrs


def sample(fields, R, C):
    """Per-cell model parameters for cells (R, C); raises if a cell has no parameters (ocean, ice)."""
    R, C = np.asarray(R), np.asarray(C)
    out = {k: fields[k][R, C].astype(np.float32) for k in ALL_KEYS}
    bad = ~np.isfinite(out["wm"])
    if bad.any():
        raise ValueError(f"{int(bad.sum())} of {len(R)} cells have no parameters (off the land mask)")
    return out


def save_cell_params(path, params):
    np.savez_compressed(path, **{k: np.asarray(params[k], np.float32) for k in ALL_KEYS})


def load_cell_params(path):
    z = np.load(path)
    return {k: z[k] for k in ALL_KEYS}
