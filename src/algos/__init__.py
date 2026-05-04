from .base import Classifier
from .cca import CCA
from .fbcca import FBCCA
from .psda import PSDA
from .trca import TRCA


def make_classifier(name: str, freqs, fs, cfg: dict | None = None):
    """Factory used by apps and benchmark."""
    cfg = cfg or {}
    name = name.lower()
    if name == "psda":
        return PSDA(freqs, fs, **cfg.get("psda", {}))
    if name == "cca":
        return CCA(freqs, fs, **cfg.get("cca", {}))
    if name == "fbcca":
        return FBCCA(freqs, fs, **cfg.get("fbcca", {}))
    if name == "trca":
        return TRCA(freqs, fs, **cfg.get("trca", {}))
    raise ValueError(f"unknown algorithm: {name}")


__all__ = ["Classifier", "PSDA", "CCA", "FBCCA", "TRCA", "make_classifier"]
