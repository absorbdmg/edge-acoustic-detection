"""
Gabor Filterbank and RSF Extraction — streaming version.

Implements 2D Gabor filters for spectro-temporal modulation analysis,
following Bellur & Elhilali (2017). Fixed input shape known at init;
all kernels and their FFTs are precomputed.

Pipeline:
    1. Cached: 2D Gabor kernels tuned to (rate, scale) pairs, plus their FFTs
    2. Hot path: FFT input spectrogram → broadcast multiply with cached kernel FFTs
       → batched IFFT → magnitude → frame integration
"""

from typing import Tuple

import cupy as cp
import numpy as np

from .backend import (
    COMPLEX_DTYPE,
    FLOAT_DTYPE,
    get_available_memory,
    next_fast_len,
    to_numpy,
)
from .config import Config
from .structs import RSF, Spectrogram

# Default Gabor parameter options (for adaptive tuning, GA workflow).
PARAM_OPTIONS = {
    "sigma_t": np.array(
        [1 / 1.4, 1 / 1.6, 1 / 1.8, 1 / 2.0, 1 / 2.2, 1 / 2.4, 1 / 2.6]
    ),
    "sigma_f": np.array(
        [1 / 1.4, 1 / 1.6, 1 / 1.8, 1 / 2.0, 1 / 2.2, 1 / 2.4, 1 / 2.6]
    ),
    "theta": np.radians(np.array([-4.5, -3.0, -1.5, 0.0, 1.5, 3.0, 4.5])),
    "alpha": np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]),
}
DEFAULT_PARAM_IDX = 3


class GaborFilterbank:
    """
    Streaming Gabor filterbank. Fixed input shape required at init.

    Args:
        config: Config object (uses defaults if None).
        n_time: Number of time frames in the input spectrogram.
        n_freq: Number of frequency channels in the input spectrogram.

    Example:
        gabor = GaborFilterbank(Config(), n_time=62, n_freq=128)
        device_out = gabor.compute_device(spec_device)   # cp.ndarray
        rsf_dataclass = gabor.compute(spec)              # RSF object
    """

    RESOLUTION_MULTIPLIERS = {
        "low": 1,
        "medium": 2,
        "high": 4,
    }

    # Memory budget for the kernel FFT cache, as a fraction of free VRAM.
    # 0.25 leaves room on Orin Nano for ROS2/PX4/other processes.
    _CACHE_MEM_FRAC = 0.25

    def __init__(
        self,
        config: Config | None = None,
        *,
        n_time: int,
        n_freq: int,
        params: np.ndarray | None = None,
    ):
        cfg = config or Config()

        self.sample_rate = cfg.sample_rate
        self.n_filters = cfg.n_filters
        self.rsf_frame_size_ms = cfg.rsf_frame_size_ms
        self.rsf_frame_shift_ms = cfg.rsf_frame_shift_ms

        self.frmlen_ms = cfg.frmlen_ms
        self.bandwidth_oct = cfg.octaves
        self.time_per_frame = self.frmlen_ms / 1000.0

        self.rates, self.scales = self._get_rates_scales(cfg)
        self._n_rates = len(self.rates)
        self._n_scales = len(self.scales)
        self._n_kernels = self._n_rates * self._n_scales

        self._n_time = n_time
        self._n_freq = n_freq

        # Frame integration parameters.
        self._window_size, self._frame_shift, self._n_frames = (
            self._compute_frame_params(n_time)
        )

        # FFT pad shape (linear convolution via zero-padding).
        self._pad_shape = (
            next_fast_len(2 * n_time - 1),
            next_fast_len(2 * n_freq - 1),
        )
        self._crop_t = (n_time - 1) // 2
        self._crop_f = (n_freq - 1) // 2

        # Cached frequency axis for the dataclass output.
        self._freq_axis = np.arange(n_freq)

        # Build and cache: kernels → kernel FFTs → frame indices → batch size → output buffer.
        self._init_kernels(params)
        self._init_frame_indices()
        self._init_output_buffer()
        self._batch_size = self._auto_batch_size()

        # Sanity check on memory headroom.
        self._verify_kernel_cache_fits()

    # ----- init helpers ------------------------------------------------------

    def _get_rates_scales(self, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
        """Generate rate and scale arrays from config + resolution multiplier."""
        cfg_rates = np.asarray(cfg.rates, dtype=np.float64)
        cfg_scales = np.asarray(cfg.scales, dtype=np.float64)

        if cfg.resolution not in self.RESOLUTION_MULTIPLIERS:
            raise ValueError(
                f"Invalid resolution '{cfg.resolution}'. "
                f"Choose from {list(self.RESOLUTION_MULTIPLIERS.keys())}"
            )

        multiplier = self.RESOLUTION_MULTIPLIERS[cfg.resolution]

        if multiplier == 1:
            return cfg_rates, cfg_scales

        pos_rates = cfg_rates[cfg_rates > 0]
        rate_min, rate_max = pos_rates.min(), pos_rates.max()
        scale_min, scale_max = cfg_scales.min(), cfg_scales.max()

        n_rates_pos = len(pos_rates) * multiplier
        n_scales = len(cfg_scales) * multiplier

        rates_pos = np.logspace(
            np.log2(rate_min), np.log2(rate_max), n_rates_pos, base=2
        )
        rates = np.concatenate([-rates_pos[::-1], rates_pos])
        scales = np.logspace(
            np.log2(scale_min), np.log2(scale_max), n_scales, base=2
        )
        return rates, scales

    def _compute_frame_params(self, n_time: int) -> Tuple[int, int, int]:
        """RSF windowing parameters (window_size, frame_shift, n_frames)."""
        window_size = int(self.rsf_frame_size_ms / 1000.0 / self.time_per_frame)
        frame_shift = max(
            1, int(self.rsf_frame_shift_ms / 1000.0 / self.time_per_frame)
        )
        n_frames = max(1, (n_time - window_size) // frame_shift + 1)
        if n_frames == 1:
            window_size = n_time
        return window_size, frame_shift, n_frames

    def _init_kernels(self, params: np.ndarray | None) -> None:
        """Build all Gabor kernels, FFT them, cache the FFTs. Drop kernels."""
        T, F = self._build_meshgrid(self._n_time, self._n_freq)

        if params is None:
            params = self._get_default_params()
        decoded = self._decode_params(params)
        self._decoded_params = decoded   # kept for diagnostics

        kernels = self._build_all_kernels(T, F, decoded)         # (K, n_time, n_freq) complex64
        self._kernel_ffts = cp.fft.fft2(
            kernels, s=self._pad_shape, axes=(-2, -1)
        ).astype(COMPLEX_DTYPE)
        del kernels

    def _init_frame_indices(self) -> None:
        """Precompute frame integration indices once."""
        starts = cp.arange(self._n_frames) * self._frame_shift
        offsets = cp.arange(self._window_size)
        idx = starts[:, None] + offsets[None, :]
        self._frame_indices = cp.clip(idx, 0, self._n_time - 1)

    def _init_output_buffer(self) -> None:
        """Preallocate (n_frames, n_rates, n_scales, n_freq) RSF output."""
        self._buf_rsf = cp.empty(
            (self._n_frames, self._n_rates, self._n_scales, self._n_freq),
            dtype=FLOAT_DTYPE,
        )

    def _build_meshgrid(self, n_time: int, n_freq: int) -> Tuple[cp.ndarray, cp.ndarray]:
        octaves_per_bin = self.bandwidth_oct / n_freq
        t_grid = (
            cp.arange(n_time, dtype=FLOAT_DTYPE) - n_time / 2
        ) * self.time_per_frame
        f_grid = (
            cp.arange(n_freq, dtype=FLOAT_DTYPE) - n_freq / 2
        ) * octaves_per_bin
        return cp.meshgrid(t_grid, f_grid, indexing="ij")

    def _build_all_kernels(
        self,
        T: cp.ndarray,
        F: cp.ndarray,
        decoded_params: np.ndarray,
    ) -> cp.ndarray:
        """Build all kernels and stack to (n_kernels, n_time, n_freq) complex64."""
        kernels = []
        for i, omega in enumerate(self.rates):
            for j, Omega in enumerate(self.scales):
                k_idx = i * self._n_scales + j
                sigma_t_mult, sigma_f_mult, theta, alpha = decoded_params[k_idx]
                kernel = self._create_gabor_kernel(
                    omega, Omega, T, F, sigma_t_mult, sigma_f_mult, theta, alpha
                )
                kernels.append(kernel)
        return cp.stack(kernels)

    def _create_gabor_kernel(
        self,
        omega: float,
        Omega: float,
        T: cp.ndarray,
        F: cp.ndarray,
        sigma_t_mult: float = 0.5,
        sigma_f_mult: float = 0.5,
        theta: float = 0.0,
        alpha: float = 1.0,
    ) -> cp.ndarray:
        """Single 2D Gabor kernel:
        F(ω, Ω, t, f) = α/(2πσ_t σ_f) · exp(-½(t₁²/σ_t² + f₁²/σ_f²)) · exp(2πj(ωt + Ωf))
        """
        omega_abs = max(abs(omega), 0.5)
        sigma_t = sigma_t_mult / omega_abs
        sigma_f = sigma_f_mult / Omega

        t1 = T * cp.cos(theta) + F * cp.sin(theta)
        f1 = -T * cp.sin(theta) + F * cp.cos(theta)

        gaussian = (alpha / (2 * cp.pi * sigma_t * sigma_f)) * cp.exp(
            -0.5 * (t1**2 / sigma_t**2 + f1**2 / sigma_f**2)
        )
        carrier = cp.exp(2j * cp.pi * (omega * T + Omega * F))
        return (gaussian * carrier).astype(COMPLEX_DTYPE)

    def _get_default_params(self) -> np.ndarray:
        return np.full((self._n_kernels, 4), DEFAULT_PARAM_IDX, dtype=np.int32)

    def _decode_params(self, indices: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [
                PARAM_OPTIONS["sigma_t"][indices[:, 0]],
                PARAM_OPTIONS["sigma_f"][indices[:, 1]],
                PARAM_OPTIONS["theta"][indices[:, 2]],
                PARAM_OPTIONS["alpha"][indices[:, 3]],
            ]
        )

    def _auto_batch_size(self) -> int:
        """Pick a batch size that fits the working set in VRAM headroom."""
        bytes_per_complex = np.dtype(COMPLEX_DTYPE).itemsize
        # Peak: filtered_fft + filtered_full per filter in batch (kernel FFTs already cached).
        mem_per_filter = 2 * self._pad_shape[0] * self._pad_shape[1] * bytes_per_complex
        available = get_available_memory()
        bs = max(1, int(available * self._CACHE_MEM_FRAC / mem_per_filter))
        return min(bs, self._n_kernels)

    def _verify_kernel_cache_fits(self) -> None:
        """Sanity check: kernel FFT cache should leave room for working set."""
        cache_bytes = self._kernel_ffts.nbytes
        free_after = get_available_memory()
        
        if free_after < cache_bytes:
            import warnings
            warnings.warn(
                f"Gabor kernel cache is {cache_bytes / 1e6:.0f}MB, only "
                f"{free_after / 1e6:.0f}MB free after init. Hot path may "
                f"hit memory pressure. Consider lower resolution.",
                ResourceWarning,
            )

    # ----- public API --------------------------------------------------------

    @property
    def output_shape(self) -> tuple[int, int, int, int]:
        """Shape of compute_device() output: (n_frames, n_rates, n_scales, n_freq)."""
        return (self._n_frames, self._n_rates, self._n_scales, self._n_freq)

    def compute_device(self, spec_device: cp.ndarray) -> cp.ndarray:
        """
        Hot path. Process one fixed-shape spectrogram; return device RSF.

        Args:
            spec_device: cp.ndarray of shape (n_time, n_freq), float32, on device.

        Returns:
            cp.ndarray of shape (n_frames, n_rates, n_scales, n_freq), float32.
            View into the internal buffer — copy if you need to retain.
        """
        if spec_device.shape != (self._n_time, self._n_freq):
            raise ValueError(
                f"expected spectrogram shape ({self._n_time}, {self._n_freq}), "
                f"got {spec_device.shape}"
            )

        # The one genuine per-call FFT — input shape changes, kernel shape doesn't.
        spec_fft = cp.fft.fft2(spec_device, s=self._pad_shape)

        ct, cf = self._crop_t, self._crop_f
        nt, nf = self._n_time, self._n_freq
        K = self._n_kernels

        # Reshape output buffer to (n_frames, K, n_freq) flat-kernel view for writing.
        # This is a view as long as _buf_rsf is C-contiguous (which cp.empty gives us).
        rsf_flat = self._buf_rsf.reshape(self._n_frames, K, nf)

        for start in range(0, K, self._batch_size):
            end = min(start + self._batch_size, K)
            # Multiply cached kernel FFTs by current spec FFT (broadcast over batch).
            filtered_fft = self._kernel_ffts[start:end] * spec_fft[None, :, :]
            filtered_full = cp.fft.ifft2(
                filtered_fft, s=self._pad_shape, axes=(-2, -1)
            )
            filtered = cp.abs(filtered_full[:, ct : ct + nt, cf : cf + nf])
            # filtered: (B, n_time, n_freq) → frame-integrate → (B, n_frames, n_freq)
            windowed = filtered[:, self._frame_indices, :].mean(axis=2)
            # Write into output buffer with axes reordered to (n_frames, B, n_freq).
            rsf_flat[:, start:end, :] = windowed.transpose(1, 0, 2)

        return self._buf_rsf

    def compute(
        self,
        spectrogram: Spectrogram | cp.ndarray | np.ndarray,
    ) -> RSF:
        """
        Wrapper: runs compute_device, copies to host, returns RSF dataclass.

        Accepts either a Spectrogram dataclass (uses .data.T to get n_time × n_freq),
        a cupy device array (already in correct orientation), or a numpy array.
        """
        if isinstance(spectrogram, Spectrogram):
            spec_dev = cp.asarray(spectrogram.data.T, dtype=FLOAT_DTYPE)
            freqs = spectrogram.freqs
        elif isinstance(spectrogram, cp.ndarray):
            spec_dev = spectrogram
            freqs = self._freq_axis
        else:
            spec_dev = cp.asarray(spectrogram, dtype=FLOAT_DTYPE)
            freqs = self._freq_axis

        device_out = self.compute_device(spec_dev)
        host = to_numpy(device_out)
        frame_period = self.rsf_frame_shift_ms / 1000.0
        times = np.arange(self._n_frames) * frame_period
        return RSF(
            data=host,
            times=times,
            rates=self.rates,
            scales=self.scales,
            freqs=freqs,
        )