# Vendored from CREST_Global_v260629/pycrest/pycrest/routing30_torch.py (2026-10-20), unchanged.
"""Differentiable sub-daily routing for CREST-AI-30.

At the scale this product is trained on (median 480 km2, ~9 cells of 0.1 deg,
observed rise times 9-11 h) explicit cell-to-cell channel routing is not the
dominant timing control — every cell reaches the outlet within one or two
30-min steps of channel travel. What actually sets the shape of the 30-min
hydrograph is the *hillslope/sub-grid* response: how fast a cell's generated
runoff arrives at the basin outlet.

We therefore route with a **learnable Gamma unit hydrograph** rather than a
fixed geomorphic triangle. The UH is fully differentiable in its two shape
parameters, so the parameter network can learn timing from the 15-min
observations — which is the entire point of going sub-daily.

    UH(t) ~ Gamma(shape=n, scale=k),   mean lag = n*k  [hours]

Both n and k are emitted per basin by the parameter net (as multipliers on a
geomorphic prior, so the cold start reproduces the classic behaviour):

    k = k_geom * kmult,   k_geom = 0.35*(A/1000)^0.38 days -> hours
    n = nshape                      (1 -> exponential, 3-5 -> classic S-curve)

Separately, `route_linres` provides a differentiable linear-reservoir cascade
for the rare large basins in the evaluation set, where channel storage does
matter. Both operate on (T, Nbasin) tensors and are autodiff-safe.
"""
import torch


def gamma_uh(nshape, kscale, nstep, dt_h=0.5, eps=1e-6):
    """Discrete Gamma unit hydrograph.

    nshape : (Nb,)   shape parameter, >0 (1 = exponential decay)
    kscale : (Nb,)   scale parameter in HOURS
    returns: (nstep, Nb) weights summing to 1 along dim 0

    Uses the log-density so gradients stay finite for small/large arguments,
    and normalises numerically so mass is conserved exactly whatever the
    truncation length.
    """
    n = torch.clamp(nshape, min=0.5, max=12.0)
    k = torch.clamp(kscale, min=dt_h * 0.5)
    t = (torch.arange(nstep, dtype=n.dtype, device=n.device) + 0.5) * dt_h
    tt = t[:, None] / k[None, :]                       # (nstep, Nb)
    logw = (n[None, :] - 1.0) * torch.log(tt + eps) - tt - torch.lgamma(n[None, :])
    w = torch.exp(logw)
    return w / torch.clamp(w.sum(0, keepdim=True), min=eps)


def uh_convolve(q, w):
    """Convolve runoff with a per-basin UH.

    q : (T, Nb) runoff [mm/step]
    w : (L, Nb) unit hydrograph weights
    -> (T, Nb) routed discharge [mm/step]

    Implemented as an FFT-free explicit sum over the (short) UH length: L is
    typically <= 96 steps (2 days at 30 min), so this is cheaper than an FFT
    and keeps the graph shallow for autodiff.
    """
    T, Nb = q.shape
    L = w.shape[0]
    out = torch.zeros_like(q)
    for l in range(L):
        out[l:] = out[l:] + q[: T - l] * w[l][None, :]
    return out


def geomorphic_k(area_km2, dtype=torch.float32, device="cpu"):
    """Prior mean lag [hours] from drainage area (the classic tc relation)."""
    a = torch.as_tensor(area_km2, dtype=dtype, device=device)
    return 0.35 * torch.pow(torch.clamp(a, min=1.0) / 1000.0, 0.38) * 24.0


def route_basin(q_cell, cellw, seg, nb, nshape, kscale, nstep=96, dt_h=0.5):
    """Aggregate cell runoff to basins, then route with the learnable UH.

    q_cell : (T, Ncell) runoff [mm/step]
    cellw  : (Ncell,)   fractional area weights, summing to 1 within each basin
    seg    : (Ncell,)   basin index per cell
    """
    T = q_cell.shape[0]
    qb = torch.zeros(T, nb, dtype=q_cell.dtype, device=q_cell.device)
    qb = qb.index_add(1, seg, q_cell * cellw[None, :])
    return uh_convolve(qb, gamma_uh(nshape, kscale, nstep, dt_h))


def route_linres(q_cell, cellw, seg, nb, kres, dt_h=0.5):
    """Differentiable single linear reservoir per basin (channel storage).

    Used for the large-basin evaluation set where channel storage, not
    hillslope response, dominates. kres is the residence time in hours.
    """
    T = q_cell.shape[0]
    qb = torch.zeros(T, nb, dtype=q_cell.dtype, device=q_cell.device)
    qb = qb.index_add(1, seg, q_cell * cellw[None, :])
    a = torch.exp(-dt_h / torch.clamp(kres, min=dt_h))
    s = torch.zeros(nb, dtype=q_cell.dtype, device=q_cell.device)
    out = []
    for t in range(T):
        s = s * a + qb[t]
        o = s * (1.0 - a)
        s = s - o
        out.append(o)
    return torch.stack(out)
