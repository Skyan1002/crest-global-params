"""dCREST forward model for one basin, identical to the global training harness.

Reproduces the forward pass of CREST_Global_v260629/verify/calibrate_gg.py (forward() and seg_run())
in float32 with the P0 settings the published parameters were trained with:
CRESTPART=ef5, FCLEARN=1, UHDAYS=30, MELTV=3 (Snow-17 is applied beforehand, see forcing.py).

Per cell and day:
  liquid input L = melt * pm                                  (pm: precipitation multiplier)
  canopy/wetland store wt (capacity ft * WM) fills first, evaporates first at KE * PET
  CREST-Phys water balance (variable infiltration curve, aquifer, exponential baseflow) on the rest
  fast flow passes a linear store with outflow fraction qc; total flow is reduced by the loss factor lf
Per basin:
  area-weighted sum over cells -> gamma unit hydrograph (shape nshape, scale kmult * geomorphic lag)
"""
import numpy as np
import torch

from . import crestphys as CP
from . import routing as RT

DT = torch.float32
UH_DAYS = 30
CELL_KEYS = ("wm", "b", "im", "ke", "fc", "ksoil", "hmaxaq", "gwc", "gwe")
ALL_KEYS = CELL_KEYS + ("ft", "qc", "lf", "pm", "nshape", "kmult")


def _t(a):
    return torch.as_tensor(np.asarray(a, dtype=np.float32), dtype=DT)


def simulate(params, melt, pet, cellw, area_km2, uh_days=UH_DAYS, return_cells=False):
    """Daily discharge of one basin.

    params   dict of per-cell physical parameters (Nc,), keys ALL_KEYS (parameter files store them so)
    melt     (T, Nc) liquid water input after Snow-17, mm/h (daily mean rate)
    pet      (T, Nc) potential evapotranspiration, mm/h
    cellw    (Nc,) cell weights, normalised to sum 1 inside this function
    area_km2 basin area, sets the geomorphic lag of the unit hydrograph
    returns  (T,) routed discharge as basin-mean runoff depth, mm/day
             (multiply by area_km2 / 86.4 for m3/s)
    """
    P = {k: _t(params[k]) for k in ALL_KEYS}
    w = _t(cellw); w = w / w.sum()
    M, E = _t(melt), _t(pet)
    T, Nc = M.shape
    p = {k: P[k] for k in CELL_KEYS}
    ftc, qcv, lfv, pmv = P["ft"], P["qc"], P["lf"], P["pm"]
    sm = 0.5 * p["wm"]; gw = 0.3 * p["hmaxaq"]
    wt = torch.zeros(Nc, dtype=DT); qs = torch.zeros(Nc, dtype=DT)
    qb = torch.zeros(T, dtype=DT)
    qcell = torch.zeros(T, Nc, dtype=DT) if return_cells else None
    with torch.no_grad():
        for t in range(T):
            L = M[t] * pmv * 24.0
            wtcap = ftc * p["wm"]
            fill = torch.minimum(L, torch.clamp(wtcap - wt, min=0.0))
            wt = wt + fill; L = L - fill
            D = p["ke"] * E[t] * 24.0
            et_t = torch.minimum(D, wt)
            wt = torch.minimum(torch.clamp(wt - et_t, min=0.0), wtcap)
            pk = torch.clamp(D - et_t, min=0.0) / torch.clamp(p["ke"], min=1e-3)
            ff, itf, bf, sm, gw = CP.step(p, sm, gw, L / 24.0, pk / 24.0, 24.0)
            fast = torch.nan_to_num(ff * 86400.0)
            rest = torch.nan_to_num((itf + bf) * 86400.0)
            qs = qs + fast; fast = qcv * qs; qs = qs - fast
            q = (fast + rest) * (1.0 - lfv)
            qb[t] = (q * w).sum()
            if return_cells:
                qcell[t] = q
            sm = torch.clamp(torch.nan_to_num(sm), min=0.0)
            gw = torch.clamp(torch.nan_to_num(gw), min=0.0)
            wt = torch.clamp(torch.nan_to_num(wt), min=0.0)
        nsh = (P["nshape"] * w).sum().reshape(1)
        kml = (P["kmult"] * w).sum().reshape(1)
        kgeo = RT.geomorphic_k(torch.tensor([float(area_km2)]), dtype=DT)
        uh = RT.gamma_uh(nsh, kml * kgeo, uh_days, dt_h=24.0)
        Q = RT.uh_convolve(qb[:, None], uh)[:, 0]
    out = Q.numpy().astype(np.float64)
    return (out, qcell.numpy()) if return_cells else out


def to_m3s(q_mm_day, area_km2):
    """mm/day over the basin -> m3/s."""
    return np.asarray(q_mm_day) * float(area_km2) / 86.4
