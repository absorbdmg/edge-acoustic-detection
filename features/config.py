"""
Configuration for PyJetsonSTM features.

Default values match Bellur & Elhilali (2017).
"""

from dataclasses import dataclass, field
import numpy as np

from .constants import STANDARD_RATES, STANDARD_SCALES


@dataclass
class Config:
    """Configuration for the streaming feature pipeline."""

    # General
    sample_rate: int = 16000

    # Spectrogram
    n_filters: int = 128
    f_min: float = 180.0
    octaves: float = 5.3
    tau_ms: float = 8.0
    frmlen_ms: float = 10.0

    # RSF / Gabor
    rates: np.ndarray = field(default_factory=lambda: STANDARD_RATES.copy())
    scales: np.ndarray = field(default_factory=lambda: STANDARD_SCALES.copy())
    resolution: str = "low"
    rsf_frame_size_ms: int = 500
    rsf_frame_shift_ms: int = 10