from .base import EEGSource, LSLPublisher, lsl_marker_outlet
from .mock import MockSource


def make_source(kind: str, cfg: dict, freqs=None):
    kind = kind.lower()
    fs = cfg["acquisition"]["fs"]
    channels = cfg["acquisition"]["channels"]
    if kind == "cyton":
        from .cyton import CytonSource
        return CytonSource(
            serial_port=cfg["acquisition"]["cyton"]["serial_port"],
            channels=channels, fs=fs,
        )
    if kind == "mock":
        m = cfg["acquisition"]["mock"]
        return MockSource(freqs=freqs or cfg["stimulus"]["freqs_hz"],
                          channels=channels, fs=fs, snr_db=m["snr_db"], seed=m["seed"])
    raise ValueError(f"unknown source: {kind}")


__all__ = ["EEGSource", "LSLPublisher", "MockSource", "make_source", "lsl_marker_outlet"]
