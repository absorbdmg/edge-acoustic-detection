from dataclasses import dataclass
import numpy as np


@dataclass
class Spectrogram:
    """
    Auditory spectrogram representation.

    Attributes:
        data: Spectrogram array [n_freq × n_time]
        times: Time axis in seconds [n_time]
        freqs: Center frequencies in Hz [n_freq]
        sr: Sample rate of original audio
    """

    data: np.ndarray
    times: np.ndarray
    freqs: np.ndarray
    sr: int

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def n_freqs(self) -> int:
        return self.data.shape[0]

    @property
    def n_times(self) -> int:
        return self.data.shape[1]

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self.times[-1] - self.times[0] if len(self.times) > 1 else 0.0

    def to_numpy(self) -> np.ndarray:
        """Return raw data array."""
        return self.data


@dataclass
class RSF:
    """
    Rate-Scale-Frequency representation.

    Attributes:
        data: RSF array [n_frames × n_rates × n_scales × n_freq]
        times: Frame times in seconds [n_frames]
        rates: Temporal modulation rates in Hz [n_rates]
        scales: Spectral modulation scales in cycles/octave [n_scales]
        freqs: Center frequencies in Hz [n_freq]
    """

    data: np.ndarray
    times: np.ndarray
    rates: np.ndarray
    scales: np.ndarray
    freqs: np.ndarray

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def n_frames(self) -> int:
        return self.data.shape[0]

    @property
    def n_rates(self) -> int:
        return self.data.shape[1]

    @property
    def n_scales(self) -> int:
        return self.data.shape[2]

    @property
    def n_freqs(self) -> int:
        return self.data.shape[3]

    def to_numpy(self) -> np.ndarray:
        """Return raw data array."""
        return self.data

    def mean_over_time(self) -> np.ndarray:
        """
        Collapse time dimension.

        Returns:
            Array [n_rates × n_scales × n_freq]
        """
        return self.data.mean(axis=0)

    def mean_over_freq(self) -> np.ndarray:
        """
        Collapse frequency dimension.

        Returns:
            Array [n_frames × n_rates × n_scales]
        """
        return self.data.mean(axis=3)

    def _split_by_direction(self) -> tuple[np.ndarray, np.ndarray]:
        """Split data into upward (negative) and downward (positive) rates."""
        mid = self.n_rates // 2
        return self.data[:, :mid, :, :], self.data[:, mid:, :, :]

    def upward_rates(self) -> np.ndarray:
        """Get negative (upward) rate values."""
        return self.rates[: self.n_rates // 2]

    def downward_rates(self) -> np.ndarray:
        """Get positive (downward) rate values."""
        return self.rates[self.n_rates // 2 :]

    def rate_scale_matrix(self, fold: bool = False) -> np.ndarray:
        """
        Get 2D rate-scale representation (averaged over time and frequency).

        Args:
            fold: If True, fold positive/negative rates for symmetric visualization

        Returns:
            Array [n_scales × n_rates]
        """
        rs = self.data.mean(axis=(0, 3)).T  # [n_scales × n_rates]

        if not fold:
            return rs

        return self._fold_rates_scales()

    def rate_scale_matrix_split(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get separate rate-scale matrices for upward and downward modulation.

        Returns:
            (upward_rs, downward_rs) each [n_scales, n_rates/2]
        """
        up_data, down_data = self._split_by_direction()
        return up_data.mean(axis=(0, 3)).T, down_data.mean(axis=(0, 3)).T

    def _fold_rates_scales(self) -> np.ndarray:
        """
        Fold RSF by averaging positive and negative rates.

        Returns:
            Folded matrix [n_scales × n_rates] (symmetric)
        """
        upward_rs, downward_rs = self.rate_scale_matrix_split()

        # Flip upward so magnitudes align with downward
        rs_folded = (np.flip(upward_rs, axis=1) + downward_rs) / 2

        # Mirror back for symmetric visualization
        return np.concatenate([np.flip(rs_folded, axis=1), rs_folded], axis=1)