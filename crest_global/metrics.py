"""Daily skill scores (same definitions as the training harness)."""
import numpy as np


def kge(sim, obs, mask=None, min_n=730):
    """Kling-Gupta efficiency (Gupta et al. 2009) and its parts; NaN if fewer than min_n valid days."""
    s, o = np.asarray(sim, float), np.asarray(obs, float)
    m = np.isfinite(s) & np.isfinite(o)
    if mask is not None:
        m &= np.asarray(mask)
    if m.sum() < min_n:
        return dict(kge=np.nan, r=np.nan, alpha=np.nan, beta=np.nan, n=int(m.sum()))
    s, o = s[m], o[m]
    r = np.corrcoef(s, o)[0, 1]; a = s.std() / o.std(); b = s.mean() / o.mean()
    return dict(kge=1 - np.sqrt((r - 1) ** 2 + (a - 1) ** 2 + (b - 1) ** 2), r=r, alpha=a, beta=b, n=int(m.sum()))


def nse(sim, obs):
    s, o = np.asarray(sim, float), np.asarray(obs, float)
    m = np.isfinite(s) & np.isfinite(o)
    return 1 - np.sum((s[m] - o[m]) ** 2) / np.sum((o[m] - o[m].mean()) ** 2)
