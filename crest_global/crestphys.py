# Vendored from CREST_Global_v260629/pycrest/pycrest/crestphys_torch.py (2026-10-20).
# Change: the runoff partition is fixed to CRESTPART=ef5, the setting all published parameters were trained with.
"""Differentiable CRESTPhys water balance (PyTorch).

Faithful port of CRESTPHYSModel::WaterBalanceInt from the mchen15ouedu/EF5
development fork (commit c15755e lineage). Extends classic CREST with an
explicit aquifer:

  - soil drainage to groundwater:      soil2gw = ksoil * SM        (per step)
  - GW-supported residual ET (dry days): unmet ET draws from GW scaled by
      GW/hmaxaq (riparian / deep-rooted evaporation)
  - fill-and-spill aquifer of capacity hmaxaq (spill joins baseflow)
  - exponential baseflow: base = gwc * (exp(gwe * GW/hmaxaq) - 1)
      -> smoothly -> 0 as GW -> 0: a DRY-ABLE aquifer with gradients
      everywhere (no dead-threshold pathology)

The classic soil part (VIC curve, ET-first, fc interflow split) mirrors
crest_torch.step exactly. Branches via torch.where; power bases guarded by
_safe_pow. step() returns (fast, inter, base) in mm/s.

Differences vs the C++ (documented, lumped-mode simplifications):
  - the gridded routing feedback (GW += routed-in baseFlow) has no lumped
    equivalent and is omitted;
  - iwu/igw initial states are fixed by spin-up, not calibrated.

Runoff partition (CRESTPART, audit 2026-10-04): this port originally treated the
LAYER_BASEFLOW excess as routing feedback and dropped it, splitting the soil
excess R the classic-CREST way (min(R, temX) -> interflow, R - min + impervious
-> overland, above-capacity water -> interflow). EF5_mchen's CRESTPHYS
(src/CRESTPhysModel.cpp:238-266) instead sends min(R, temX) to the AQUIFER,
R - min to interflow, and impervious + above-capacity water to overland flow;
that LAYER_BASEFLOW term is recharge, not routing. CRESTPART=ef5 reproduces the
C++ (bench log 2026-10-05); "classic" stays the default so results built
before then remain reproducible.
"""

import os

import torch

from . import _safepow as CT

BOUNDS = {
    "wm":     (1.0, 1000.0),
    "b":      (0.01, 3.0),
    "im":     (0.0, 0.5),
    "ke":     (0.1, 2.0),
    "fc":     (0.0, 150.0),
    "ksoil":  (0.0, 0.5),      # soil->GW drainage fraction [1/day]
    "hmaxaq": (50.0, 2000.0),  # aquifer capacity [mm]
    "gwc":    (1e-4, 5.0),     # baseflow coefficient [mm/day]
    "gwe":    (0.5, 6.0),      # baseflow exponent [-]
}
PARAM_ORDER = list(BOUNDS.keys())
PARTITION = "ef5"
assert PARTITION in ("classic", "ef5"), f"CRESTPART must be classic|ef5, got {PARTITION}"


def raw_from_physical(phys, dtype=torch.float64, device="cpu"):
    raw = {}
    for k, v in phys.items():
        lo, hi = BOUNDS[k]
        v = torch.as_tensor(v, dtype=dtype, device=device)
        u = ((v - lo) / (hi - lo)).clamp(1e-6, 1 - 1e-6)
        raw[k] = torch.log(u / (1 - u))
    return raw


def physical(raw):
    return {k: BOUNDS[k][0] + (BOUNDS[k][1] - BOUNDS[k][0]) * torch.sigmoid(r)
            for k, r in raw.items()}


_safe_pow = CT._safe_pow


def step(p, sm, gw, precip_mmhr, pet_mmhr, step_hours, return_et=False):
    """One step. Returns (fast_mms, inter_mms, base_mms, sm_new, gw_new);
    with return_et=True appends actual ET in mm/step (wet branch: adjPET is
    fully met; dry branch: all precip + soil ExcessET + GW-supported resET)."""
    sh = step_hours
    WM, B, IM, KE, FC = p["wm"], p["b"], p["im"], p["ke"], p["fc"]
    KSOIL, HMAX, GWC, GWE = p["ksoil"], p["hmaxaq"], p["gwc"], p["gwe"]
    kd = sh / 24.0                                   # per-day rates -> per step
    precip = precip_mmhr * sh
    pet = pet_mmhr * sh
    adjPET = pet * KE
    SM0 = sm

    # ---- wet branch (precip > adjPET): identical to classic CREST ----
    precipSoil = (precip - adjPET) * (1.0 - IM)
    precipImperv = precip - adjPET - precipSoil
    interflowExcess_w = torch.clamp(SM0 - WM, min=0.0)
    SM_c = torch.minimum(SM0, WM)
    sm_lt = SM_c < WM

    Wmaxm = WM * (1.0 + B)
    base_A = torch.clamp(1.0 - SM_c / WM, min=0.0)
    A = Wmaxm * (1.0 - _safe_pow(base_A, 1.0 / (1.0 + B)))

    saturated = (precipSoil + A) >= Wmaxm
    R_sat = torch.clamp(precipSoil - (WM - SM_c), min=0.0)
    b1 = torch.clamp(1.0 - A / Wmaxm, min=0.0)
    b2 = torch.clamp(1.0 - (A + precipSoil) / Wmaxm, min=0.0)
    infiltration = WM * (_safe_pow(b1, 1.0 + B) - _safe_pow(b2, 1.0 + B))
    infiltration = torch.minimum(infiltration, precipSoil)
    R_inf = torch.clamp(precipSoil - infiltration, min=0.0)
    Wo_inf = SM_c + infiltration

    R_lt = torch.where(saturated, R_sat, R_inf)
    Wo_lt = torch.where(saturated, torch.broadcast_to(WM, R_sat.shape), Wo_inf)
    R_w = torch.where(sm_lt, R_lt, precipSoil)
    Wo_w = torch.where(sm_lt, Wo_lt, torch.broadcast_to(WM, R_w.shape))

    temX = (SM_c + Wo_w) / WM / 2.0 * (FC * sh)
    rech_w = torch.minimum(R_w, temX)
    if PARTITION == "ef5":            # EF5_mchen: min -> aquifer, rest -> interflow
        inter_w = R_w - rech_w
        over_w = precipImperv + interflowExcess_w
    else:                             # classic CREST: min -> interflow, rest -> overland
        inter_w = rech_w + interflowExcess_w
        over_w = R_w - rech_w + precipImperv
        rech_w = torch.zeros_like(R_w)
    resET_w = torch.zeros_like(SM0)

    # ---- dry branch (precip <= adjPET): soil ET then GW-supported residual ----
    interflowExcess_d = torch.clamp(SM0 - WM, min=0.0)
    SM_cd = torch.minimum(SM0, WM)
    ExcessET = (adjPET - precip) * SM_cd / WM
    ExcessET_act = torch.minimum(ExcessET, SM_cd)
    Wo_d = SM_cd - ExcessET_act
    unmet = torch.clamp(adjPET - precip - ExcessET_act, min=0.0)
    resET_d = torch.minimum(unmet * gw / HMAX, gw)   # riparian/deep ET from GW
    over_d = torch.zeros_like(SM0)

    wet = precip > adjPET
    if PARTITION == "ef5":            # EF5 dry branch: above-capacity water is overland
        over = torch.where(wet, over_w, interflowExcess_d)
        inter = torch.where(wet, inter_w, torch.zeros_like(inter_w))
    else:
        over = torch.where(wet, over_w, over_d)
        inter = torch.where(wet, inter_w, interflowExcess_d)
    Wo = torch.where(wet, Wo_w, Wo_d)
    gw = gw - torch.where(wet, resET_w, resET_d) + torch.where(wet, rech_w, torch.zeros_like(rech_w))

    # ---- soil -> GW drainage (rate KSOIL per day) ----
    soil2gw = torch.minimum(KSOIL * kd * Wo, Wo)
    Wo = Wo - soil2gw
    gw = gw + soil2gw

    # ---- fill-and-spill aquifer + exponential baseflow ----
    spill = torch.clamp(gw - HMAX, min=0.0)
    gw = torch.minimum(gw, HMAX)
    base = GWC * kd * torch.expm1(GWE * torch.clamp(gw, min=0.0) / HMAX)
    base = torch.minimum(base, gw)
    gw = gw - base

    fast = over / (sh * 3600.0)
    inter_f = inter / (sh * 3600.0)
    base_f = (base + spill) / (sh * 3600.0)
    if return_et:
        et_act = torch.where(wet, adjPET,
                             precip + ExcessET_act + resET_d)
        return fast, inter_f, base_f, Wo, gw, et_act
    return fast, inter_f, base_f, Wo, gw


def init_states(nb, dtype=torch.float64, device="cpu"):
    return {"sm": torch.full((nb,), 50.0, dtype=dtype, device=device),
            "gw": torch.full((nb,), 50.0, dtype=dtype, device=device)}


def run(raw_params, states, precip_mmhr, pet_mmhr, step_hours):
    p = physical(raw_params)
    sm, gw = states["sm"], states["gw"]
    ff, itf, bf = [], [], []
    for t in range(precip_mmhr.shape[0]):
        f, i_, b, sm, gw = step(p, sm, gw, precip_mmhr[t], pet_mmhr[t], step_hours)
        ff.append(f); itf.append(i_); bf.append(b)
    return torch.stack(ff), torch.stack(itf), torch.stack(bf), {"sm": sm, "gw": gw}
