"""BioMiner local-runner connector.

Pairs with :mod:`bm_runner`: this package downloads data from a BioMiner
deployment, runs the analysis locally with ``bm_runner``, and registers the
result back. It bundles no credentials and computes no science itself.
"""
from .client import BioMinerClient, BioMinerError, DEFAULT_BASE_URL
from . import workflow

__all__ = ["BioMinerClient", "BioMinerError", "DEFAULT_BASE_URL", "workflow"]
__version__ = "0.1.0"
