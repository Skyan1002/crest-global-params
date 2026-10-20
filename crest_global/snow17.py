# Vendored from CREST_Global_v260629/pycrest/pycrest/snow17_torch.py (2026-10-20), unchanged.
"""Differentiable Snow-17 (PyTorch) for gradient-based calibration.

Same algorithm as ``snow17_np`` (faithful to EF5's SnowBalanceInt), made
autodiff-friendly per the locked Stage-1 spec:

* Rain/snow partition softened:  f_snow = sigmoid((PXTEMP - Ta)/tau)  (no hard
  threshold). EF5 hard-codes PXTEMP=1 deg C; here PXTEMP is a *learnable*
  parameter, and tau a small fixed width.
* clamp / max kept as torch ops (subdifferentiable -- fine for autodiff).
* All parameters bounded by sigmoid reparam:  theta = lo + (hi-lo)*sigmoid(raw)
  so unconstrained gradients stay in-range (Anderson 2006 ranges).
* State recursion is an explicit time loop (BPTT). For long horizons wrap the
  per-step call in torch.utils.checkpoint to bound memory.

Parameters are carried as raw (unconstrained) tensors; call ``physical()`` to
get the bounded physical values. Everything runs in float64 by default for
clean gradient checks; switch to float32 for large grids.
"""

import math
import torch

# Physical ranges (lo, hi). PXTEMP added beyond EF5; DAYGM/ADC handled elsewhere.
BOUNDS = {
    "uadj":  (0.03, 0.19),
    "mbase": (0.0, 1.0),
    "mfmax": (0.5, 2.5),
    "mfmin": (0.05, 0.5),
    "tipm":  (0.1, 1.0),
    "nmf":   (0.05, 0.5),
    "plwhc": (0.02, 0.30),
    "scf":   (0.7, 1.6),
    "pxtemp": (-1.0, 3.0),
}
PARAM_ORDER = list(BOUNDS.keys())


def raw_from_physical(phys, dtype=torch.float64, device="cpu"):
    """Invert the sigmoid reparam to initialise raw tensors at given physical values."""
    raw = {}
    for k, v in phys.items():
        lo, hi = BOUNDS[k]
        v = torch.as_tensor(v, dtype=dtype, device=device)
        u = ((v - lo) / (hi - lo)).clamp(1e-6, 1 - 1e-6)
        raw[k] = torch.log(u / (1 - u))   # logit
    return raw


def physical(raw):
    """Map raw (unconstrained) params to bounded physical values."""
    phys = {}
    for k, r in raw.items():
        lo, hi = BOUNDS[k]
        phys[k] = lo + (hi - lo) * torch.sigmoid(r)
    return phys


def p_atm_from_elevation(elev_m):
    """Atmospheric pressure (mb) from elevation in METRES.

    Anderson's formula takes elevation in HUNDREDS of metres; EF5 divides its
    metre DEM by 100 (Snow17Model.cpp:33). The previous version used the
    argument directly: pressure ~18x too high at 500 m, ~2300x at 3 km, ~5300x
    at 4 km (and too low below ~200 m), and the rain-on-snow term with it
    (bench log 2026-10-04). Below sea level is held at sea-level pressure,
    since the 2.4 power of a negative number is NaN.
    """
    e = (torch.as_tensor(elev_m, dtype=torch.float64) / 100.0).clamp(min=0.0)
    return 33.86 * (29.9 - 0.335 * e + 0.00022 * e.pow(2.4))


def init_states(n, dtype=torch.float64, device="cpu"):
    z = lambda: torch.zeros(n, dtype=dtype, device=device)
    return {"ati": z(), "wq": z(), "wi": z(), "deficit": z()}


def step(p, st, jday, precip_mmhr, temp, step_hours, p_atm, tau=0.5,
         fix_sv=False, fix_dt6=False):
    """One differentiable Snow-17 step. p = physical params dict.

    Defaults reproduce EF5 (Snow17Model.cpp) exactly, including two defects
    found in EF5 itself during the 2026-10-04 audit; the flags opt into
    Anderson's intended forms (bench log 2026-10-04):
      fix_sv   EF5 writes sin((jday - 81*2*pi)/366): an operator-precedence
               slip that confines the seasonal melt-factor weight Sv to
               0.008..0.31, ramping monotonically to its peak on 31 Dec.
               Anderson's form is sin((jday-81)*2*pi/366), 0..1 with the peak
               at the June solstice. jday may be a per-cell tensor, so callers
               can shift the southern hemisphere by half a year.
      fix_dt6  EF5 applies (dt/6) to the heat-exchange term twice -- once
               explicitly and once inside mf -- so at a 24 h step that term is
               4x too large. At 6 h, the step every parity test used, the two
               forms coincide, which is why the tests could not see it.
    """
    sh = step_hours
    stefan = 6.12e-10
    PXTEMP = p["pxtemp"]

    ATI, WQ, WI, DEFICIT = st["ati"], st["wq"], st["wi"], st["deficit"]
    TIPM_dtt = 1.0 - (1.0 - p["tipm"]).pow(sh / 6.0)
    precip = precip_mmhr * sh
    Ta = temp

    if fix_sv:
        jd = torch.as_tensor(jday, dtype=Ta.dtype, device=Ta.device)
        Sv = 0.5 * torch.sin((jd - 81.0) * 2.0 * math.pi / 366.0) + 0.5
    else:   # EF5 verbatim (precedence slip kept for parity)
        Sv = 0.5 * math.sin((jday - 81 * 2 * math.pi) / 366.0) + 0.5
    mf = (sh / 6.0) * (Sv * (p["mfmax"] - p["mfmin"]) + p["mfmin"])

    # --- soft accumulation -------------------------------------------------
    f_snow = torch.sigmoid((PXTEMP - Ta) / tau)
    Pn = precip * f_snow * p["scf"]
    WI = WI + Pn
    RAIN = (1.0 - f_snow) * precip

    cold = (Ta < 0.0)
    T_snow_new = torch.where(cold, Ta, torch.zeros_like(Ta))
    delta_HD_snow = torch.where(cold, -(T_snow_new * Pn) / (80.0 / 0.5),
                                torch.zeros_like(Ta))
    T_rain = torch.where(cold, torch.full_like(Ta, 1.0), Ta)  # PXTEMP=1 for T_rain term (per EF5)

    heavy = Pn > (1.5 * sh)
    ATI = torch.where(heavy, T_snow_new, ATI + TIPM_dtt * (Ta - ATI))
    ATI = torch.clamp(ATI, max=0.0)

    # mf already carries (sh/6); EF5 multiplies by it again (fix_dt6 removes it)
    delta_HD_T = (p["nmf"] * (1.0 if fix_dt6 else sh / 6.0)
                  * (mf / p["mfmax"]) * (ATI - T_snow_new))

    e_sat = 2.7489e8 * torch.exp(-4278.63 / (Ta + 242.792))
    ros = RAIN > (0.25 * sh)
    M1 = torch.clamp(stefan * sh * ((Ta + 273.0).pow(4.0) - 273.0 ** 4), min=0.0)
    M2 = torch.clamp(0.0125 * RAIN * T_rain, min=0.0)
    M3 = torch.clamp(8.5 * p["uadj"] * (sh / 6.0) *
                     ((0.9 * e_sat - 6.11) + 0.00057 * p_atm * Ta), min=0.0)
    M_RoS = torch.where(ros, M1 + M2 + M3, torch.zeros_like(Ta))

    nonrain = (RAIN <= (0.25 * sh)) & (Ta > p["mbase"])
    M_NR = torch.where(nonrain, mf * (Ta - p["mbase"]) + 0.0125 * RAIN * T_rain,
                       torch.zeros_like(Ta))

    Melt = torch.clamp(M_RoS + M_NR, min=0.0)
    melt_lt = Melt < WI
    Melt = torch.where(melt_lt, Melt, WI + WQ)
    WI = torch.where(melt_lt, WI - Melt, torch.zeros_like(WI))

    Qw = Melt + RAIN
    W_qx = p["plwhc"] * WI
    DEFICIT = DEFICIT + delta_HD_snow + delta_HD_T
    DEFICIT = torch.clamp(DEFICIT, min=0.0)
    DEFICIT = torch.minimum(DEFICIT, 0.33 * WI)

    has_snow = WI > 0.0
    ripe = (Qw + WQ) > (DEFICIT * (1.0 + p["plwhc"]) + W_qx)
    melting = Qw >= DEFICIT

    E_ripe = Qw + WQ - W_qx - DEFICIT * (1.0 + p["plwhc"])
    WQ_s = torch.where(ripe, W_qx, torch.where(melting, WQ + Qw - DEFICIT, WQ))
    WI_s = torch.where(ripe, WI + DEFICIT,
                       torch.where(melting, WI + DEFICIT, WI + Qw))
    DEF_s = torch.where(ripe, torch.zeros_like(WI),
                        torch.where(melting, torch.zeros_like(WI), DEFICIT - Qw))
    E_s = torch.where(ripe, E_ripe, torch.zeros_like(Qw))
    SWE_s = WI_s + WQ_s

    E = torch.where(has_snow, E_s, Qw)
    WQ = torch.where(has_snow, WQ_s, torch.zeros_like(WQ))
    WI = torch.where(has_snow, WI_s, WI)
    DEFICIT = torch.where(has_snow, DEF_s, DEFICIT)
    SWE = torch.where(has_snow, SWE_s, torch.zeros_like(Qw))
    ATI = torch.where(DEFICIT == 0.0, torch.zeros_like(ATI), ATI)

    return E / sh, SWE, {"ati": ATI, "wq": WQ, "wi": WI, "deficit": DEFICIT}


def run(raw_params, states, jday, precip_mmhr, temp, step_hours, p_atm, tau=0.5):
    """Differentiable rollout. precip/temp shape (T,N); jday shape (T,).
    Returns (melt (T,N), swe (T,N)) with the graph intact for backprop."""
    p = physical(raw_params)
    T = precip_mmhr.shape[0]
    st = dict(states)
    melts, swes = [], []
    for t in range(T):
        m, s, st = step(p, st, float(jday[t]), precip_mmhr[t], temp[t],
                        step_hours, p_atm, tau)
        melts.append(m); swes.append(s)
    return torch.stack(melts), torch.stack(swes)
