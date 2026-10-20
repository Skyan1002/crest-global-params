"""Vendored _safe_pow from CREST_Global_v260629/pycrest/pycrest/crest_torch.py (2026-10-20)."""
import torch


def _safe_pow(base, exp, gmin=1e-3):
    """base**exp with an EXACT forward value (0 for base<=0, matching the numpy
    port) but a BOUNDED gradient. For fractional exp the derivative exp*base**(exp-1)
    blows up as base->0+ (inf grad -> NaN via 0*inf downstream). We keep the forward
    exact via a detached correction, but route the gradient through base clamped at
    gmin, so exp*base**(exp-1) stays finite. base<=0 branch is a constant 0."""
    pos = base > 0
    bp = torch.where(pos, base, torch.ones_like(base))
    bg = bp.clamp(min=gmin)                      # gradient-only floor
    y = bg.pow(exp) + (bp.pow(exp) - bg.pow(exp)).detach()   # forward == bp**exp, grad via bg
    return torch.where(pos, y, torch.zeros_like(base))
