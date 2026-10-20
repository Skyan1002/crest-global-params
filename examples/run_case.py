"""Run one packaged case with the released parameters.

  python examples/run_case.py GRDC_6342970                 # IMERG Final parameters, shipped forcing
  python examples/run_case.py GRDC_6342970 --heldout       # the fold network that never saw this basin
  python examples/run_case.py GRDC_6342970 --product mswep --mswep-dir /path/to/MSWEP/Past/Daily

Writes examples/output/<id>_<product>.csv (date, Q in mm/day and m3/s) and prints the stored skill.
MSWEP forcing is not redistributed: point --mswep-dir at your own MSWEP V2.8 daily files (YYYYDDD.nc).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from crest_global import basin, forcing, metrics, model, params  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("case")
ap.add_argument("--product", choices=params.PRODUCTS, default="imerg_final")
ap.add_argument("--heldout", action="store_true", help="use the held-out fold network's parameters")
ap.add_argument("--mswep-dir", help="folder of MSWEP V2.8 daily files YYYYDDD.nc (needed for --product mswep)")
a = ap.parse_args()

cdir = f"{HERE}/cases/{a.case}"
meta, cells, f = basin.load_case(cdir)
precip = f["precip"]
if a.product == "mswep":
    if not a.mswep_dir:
        sys.exit("--product mswep needs --mswep-dir (MSWEP is not redistributed with this repository)")
    days = f["dates"]
    precip = np.stack([forcing.read_mswep_day(
        f"{a.mswep_dir}/{d.astype(object).year}{(d - d.astype('datetime64[Y]').astype('datetime64[D]')).astype(int) + 1:03d}.nc",
        cells["R"], cells["C"]) for d in days])
melt, pet = forcing.model_inputs(precip, f["t2m"], cells["elev"], cells["lat"], f["dates"])
pfile = f"{cdir}/params_{a.product}{'_heldout' if a.heldout else ''}.npz"
q = model.simulate(params.load_cell_params(pfile), melt, pet, cells["w"], float(cells["area_km2"]))

os.makedirs(f"{HERE}/output", exist_ok=True)
out = f"{HERE}/output/{a.case}_{a.product}{'_heldout' if a.heldout else ''}.csv"
with open(out, "w") as fh:
    fh.write("date,q_mm_day,q_m3s\n")
    for d, v, w in zip(f["dates"].astype(str), q, model.to_m3s(q, float(cells["area_km2"]))):
        fh.write(f"{d},{v:.4f},{w:.3f}\n")
s = meta["skill"][a.product]
key = "heldout_network_test_2003_2011" if a.heldout else "released_params_test_2003_2011"
print(f"{a.case} {meta['label']} | {a.product} | mean Q {q[730:].mean():.3f} mm/day | wrote {out}")
print(f"stored daily KGE 2003-2011 ({key}): {s[key]}")
obs = np.load(f"{cdir}/obs_grdc_caravan.npz")["q_mm_day"].astype(float)   # GRDC-Caravan, CC BY 4.0
test = (f["dates"] >= np.datetime64("2003-01-01")) & (f["dates"] <= np.datetime64("2011-12-31"))
print(f"this run vs observed (GRDC-Caravan), 2003-2011: {metrics.kge(q, obs, test)}")
