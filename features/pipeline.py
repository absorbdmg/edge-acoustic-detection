"""
Streaming pipeline.

Chains AuditorySpectrogram and GaborFilterbank with no host round-trip
between stages. compute_device(audio) is the hot path; compute(audio)
wraps for offline / dataclass use.
"""

import cupy as cp
import numpy as np

from .backend import to_numpy
from .config import Config
from .gabor import GaborFilterbank
from .spectrogram import AuditorySpectrogram
from .structs import RSF, Spectrogram


class PyJetsonSTM:
    """
    Streaming spectro-temporal modulation pipeline.

    Args:
        config: Config object (uses defaults if None).
        chunk_samples: Number of audio samples per call. Must divide evenly
            by the spectrogram frame size.

    Example:
        pipeline = PyJetsonSTM(Config(), chunk_samples=16000)
        rsf_dev = pipeline.compute_device(audio)   # cp.ndarray, on device
        rsf = pipeline.compute(audio)              # RSF dataclass, on host
        spec = pipeline.spectrogram(audio)         # Spectrogram dataclass
        rsf  = pipeline.rsf(spec)                  # RSF dataclass from spec
    """

    def __init__(self, config: Config | None = None, *, chunk_samples: int):
        self.config = config or Config()
        self._chunk_samples = chunk_samples

        self.spec_model = AuditorySpectrogram(
            self.config, chunk_samples=chunk_samples
        )
        n_time, n_freq = self.spec_model.output_shape
        self.gabor_model = GaborFilterbank(
            self.config, n_time=n_time, n_freq=n_freq
        )

    @property
    def output_shape(self) -> tuple[int, int, int, int]:
        """Shape of compute_device() output: (n_frames, n_rates, n_scales, n_freq)."""
        return self.gabor_model.output_shape

    # Test compatibility helpers (one stage at a time)
    def spectrogram(self, audio: np.ndarray) -> Spectrogram:
        """Run only the spectrogram stage; return Spectrogram dataclass."""
        return self.spec_model.compute(audio)

    def rsf(self, spec: Spectrogram | cp.ndarray | np.ndarray) -> RSF:
        """Run only the Gabor stage; return RSF dataclass."""
        return self.gabor_model.compute(spec)

    # Chained pipeline (both stages on device)
    def compute_device(self, audio: np.ndarray) -> cp.ndarray:
        """Hot path. audio (chunk_samples,) → RSF on device."""
        spec_dev = self.spec_model.compute_device(audio)
        return self.gabor_model.compute_device(spec_dev)

    def compute(self, audio: np.ndarray) -> RSF:
        """Run the pipeline and return an RSF dataclass on host."""
        rsf_dev = self.compute_device(audio)
        host = to_numpy(rsf_dev)
        frame_period = self.config.rsf_frame_shift_ms / 1000.0
        times = np.arange(host.shape[0]) * frame_period
        return RSF(
            data=host,
            times=times,
            rates=self.gabor_model.rates,
            scales=self.gabor_model.scales,
            freqs=self.spec_model.center_freqs,
        )