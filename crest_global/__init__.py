"""crest_global: run dCREST with the global 0.1-degree parameter sets (IMERG Final, MSWEP)."""
from . import basin, forcing, metrics, model, params  # noqa: F401
from .model import simulate, to_m3s  # noqa: F401

__version__ = "1.0.0"
