"""Voice activity detection package for the ``audio-vad`` bug-fix exercise."""

from .detector import (
    VadError,
    decide,
    detect_batch,
    frame_energy,
    hangover,
    segments,
    smooth,
    spectral_flatness,
    zero_crossing_rate,
)

__all__ = [
    "VadError",
    "decide",
    "detect_batch",
    "frame_energy",
    "hangover",
    "segments",
    "smooth",
    "spectral_flatness",
    "zero_crossing_rate",
]
