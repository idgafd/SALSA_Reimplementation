"""SALSA features for first-order ambisonics.

A reimplementation of the feature extraction described in
"SALSA: Spatial Cue-Augmented Log-Spectrogram Features for Polyphonic Sound
Event Localization and Detection" (Nguyen et al., IEEE/ACM TASLP 2022).
"""

from .encoding import FoaConverter
from .salsa import FoaSalsa, intensity_vector
from .stft import StftConfig, istft, stft

__version__ = "0.1.0"

__all__ = [
    "FoaConverter",
    "FoaSalsa",
    "StftConfig",
    "intensity_vector",
    "stft",
    "istft",
]
