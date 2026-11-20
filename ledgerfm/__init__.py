"""LedgerFM: open foundation model for transaction sequences (fraud / AML).

    from ledgerfm import load, embed, Blend
"""
__version__ = "0.1.2"

from .blend import Blend


def load(*a, **k):
    from .hub import load as _load
    return _load(*a, **k)


def embed(*a, **k):
    from .hub import embed as _embed
    return _embed(*a, **k)


__all__ = ["load", "embed", "Blend", "__version__"]
