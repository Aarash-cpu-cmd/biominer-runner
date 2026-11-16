"""test_local_runner runs REAL pydeseq2 in-process.

tests/test_workspace_full_mcp/conftest.py installs process-global
tripwires on DeseqDataSet.deseq2 / DeseqStats.summary so its synthetic
suite can never reach a real engine, and execsvc/worker_main.py wraps
the same attrs with budget counters when imported in-process. Wrap
order is not guaranteed: either layer may sit on top. Every test here
unwinds the installed attribute back to the real engine entry first
(a no-op when nothing was wrapped):

- tripwires stash the wrapped callable as ``_orig`` (test-only attr)
- execsvc counter wrappers keep their inner callable in ``__closure__``
  cells, which may be the real function or another wrapper, so each
  function-valued cell is unwrapped recursively; production code needs
  no test-specific markers
"""
from __future__ import annotations

import inspect

import pytest

_ENGINE_MODULE = "pydeseq2"


def _unwrap_engine(fn, depth: int = 0):
    if fn is None or depth > 10:
        return None
    if str(getattr(fn, "__module__", "")).startswith(_ENGINE_MODULE):
        return fn
    nxt = getattr(fn, "_orig", None)
    if nxt is not None and nxt is not fn:
        return _unwrap_engine(nxt, depth + 1)
    for cell in getattr(fn, "__closure__", None) or ():
        value = cell.cell_contents
        if inspect.isfunction(value):
            real = _unwrap_engine(value, depth + 1)
            if real is not None:
                return real
    return None


@pytest.fixture(autouse=True)
def _restore_engine_from_workspace_tripwires():
    try:
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.ds import DeseqStats
    except ImportError:
        yield
        return
    for cls, name in ((DeseqDataSet, "deseq2"), (DeseqStats, "summary")):
        fn = _unwrap_engine(getattr(cls, name, None))
        if fn is not None and fn is not getattr(cls, name, None):
            setattr(cls, name, fn)
    yield
