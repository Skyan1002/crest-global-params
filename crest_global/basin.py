"""Basins on the 0.1-degree grid: cells (R, C) with fractional-area weights."""
import json

import numpy as np


def cells_from_polygon(geom):
    """Cells overlapping a shapely (Multi)Polygon in lon/lat, weighted by the fraction of each cell inside
    the polygon (0-1, in lon/lat degrees). The training masks use the same rule on the official GRDC basin
    polygons; as there, cells are averaged with these fractions, without a cos(latitude) area factor."""
    from shapely.geometry import box
    lon0, lat0, lon1, lat1 = geom.bounds
    c0, c1 = int(np.floor((lon0 + 180) / 0.1)), int(np.ceil((lon1 + 180) / 0.1))
    r0, r1 = int(np.floor((90 - lat1) / 0.1)), int(np.ceil((90 - lat0) / 0.1))
    R, C, W = [], [], []
    for r in range(max(r0, 0), min(r1, 1800)):
        latn = 90 - 0.1 * r; lats = latn - 0.1
        for c in range(max(c0, 0), min(c1, 3600)):
            lonw = -180 + 0.1 * c
            cell = box(lonw, lats, lonw + 0.1, latn)
            if not cell.intersects(geom):
                continue
            f = cell.intersection(geom).area / cell.area
            if f > 0:
                R.append(r); C.append(c); W.append(f)
    return np.array(R), np.array(C), np.array(W)


def cell_area_km2(R):
    lat = 89.95 - 0.1 * np.asarray(R)
    return (111.195 * 0.1) ** 2 * np.cos(np.radians(lat))


def load_case(case_dir):
    """A packaged example case: metadata, cells and IMERG Final + ERA5 forcing."""
    meta = json.load(open(f"{case_dir}/case.json"))
    cells = dict(np.load(f"{case_dir}/cells.npz"))
    f = np.load(f"{case_dir}/forcing_imerg_final_era5.npz")
    forcing = {"dates": f["dates"].astype("datetime64[D]"), "precip": f["precip"].astype(np.float32),
               "t2m": f["t2m"].astype(np.float32)}
    return meta, cells, forcing
