"""The package reproduces the training harness on every example case.

From the shipped IMERG Final precipitation and ERA5 temperature, run Snow-17, PET and dCREST with the
parameters of the fold network that never saw the basin, and compare with the training run's own
simulation (stored in float16, so differences of a few 1e-3 mm/day are rounding).

Run: python -m pytest -q tests   (about 2 minutes on a CPU)
"""
import glob
import os

import numpy as np
import pytest

from crest_global import basin, forcing, model, params

CASES = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "examples", "cases", "GRDC_*")))


assert len(CASES) == 8, CASES


@pytest.mark.parametrize("case_dir", CASES, ids=[os.path.basename(c) for c in CASES])
def test_reproduces_training_run(case_dir):
    meta, cells, f = basin.load_case(case_dir)
    melt, pet = forcing.model_inputs(f["precip"], f["t2m"], cells["elev"], cells["lat"], f["dates"])
    h = lambda a: a.astype(np.float16).astype(np.float32)        # the training caches are float16
    p = params.load_cell_params(f"{case_dir}/params_imerg_final_heldout.npz")
    q = model.simulate(p, h(melt), h(pet), cells["w"], float(cells["area_km2"]))
    ref = np.load(f"{case_dir}/reference_training.npz")["imerg_final"].astype(np.float64)
    assert q.shape == ref.shape
    assert np.abs(q - ref).max() < 0.05 + 0.002 * ref.max()
    assert np.corrcoef(q, ref)[0, 1] > 0.99999
