"""
Auditory Spectrogram (Cochlear Model) — streaming version.

Implements the auditory spectrogram from Chi, Ru & Shamma (2005) for fixed
chunk size known at init. Everything shape-dependent is precomputed; the
hot path is pure data-dependent compute.

Pipeline:
    y1: Cochlear filtering (gammatone filterbank, IIR per channel)
    y2: Hair cell transduction (diff + tanh)
    y3: Lateral inhibition
    y4: Half-wave rectification
    y5: Leaky temporal integration (FFT-based convolution)
    final: cube root + downsample to frame rate
"""

import cupy as cp
import numpy as np
from cupyx.scipy.signal import resample_poly, sosfilt

from .backend import FLOAT_DTYPE, next_fast_len, to_numpy
from .config import Config
from .structs import Spectrogram


class AuditorySpectrogram:
    """
    Streaming auditory spectrogram. Fixed chunk size required at init.

    Output orientation note:
        compute_device() returns shape (n_time, n_freq) — the orientation
        Gabor consumes directly.
        compute() wraps the result in a Spectrogram dataclass which uses
        the legacy (n_freq, n_time) orientation for backward compat.

    Args:
        config: Config object (uses defaults if None).
        chunk_samples: Number of audio samples per call to compute_device().
            Must be a multiple of the frame size in samples
            (sample_rate * frmlen_ms / 1000), otherwise output frame count
            won't divide evenly.

    Example:
        spec = AuditorySpectrogram(Config(), chunk_samples=16000)
        device_out = spec.compute_device(audio_np)   # (n_time, n_freq) cp.ndarray
        dataclass = spec.compute(audio_np)           # Spectrogram object
    """

    def __init__(self, config: Config | None = None, *, chunk_samples: int):
        cfg = config or Config()

        self.sample_rate = cfg.sample_rate
        self.n_filters = cfg.n_filters
        self.f_min = cfg.f_min
        self.octaves = cfg.octaves
        self.tau_ms = cfg.tau_ms
        self.frmlen_ms = cfg.frmlen_ms

        self.f_max = self.f_min * (2 ** self.octaves)
        self.center_freqs = self._create_frequency_scale()

        # Validate chunk_samples vs frame size (must divide evenly).
        self._L_frm = int((self.frmlen_ms / 1000.0) * self.sample_rate)
        if chunk_samples % self._L_frm != 0:
            raise ValueError(
                f"chunk_samples={chunk_samples} must be a multiple of frame size "
                f"L_frm={self._L_frm} (frmlen_ms={self.frmlen_ms} @ "
                f"sample_rate={self.sample_rate}). Suggestion: "
                f"{(chunk_samples // self._L_frm) * self._L_frm} or "
                f"{(chunk_samples // self._L_frm + 1) * self._L_frm}."
            )
        self._chunk_samples = chunk_samples
        self._n_frames_out = chunk_samples // self._L_frm

        # Static precomputation.
        self._init_gammatone_filters()       # builds host SOS + stacks on device
        self._init_y5_kernel()               # leaky integration kernel + its FFT
        self._init_buffers()                 # preallocated device tensors

    # ----- init helpers ------------------------------------------------------

    def _create_frequency_scale(self) -> np.ndarray:
        return np.logspace(
            np.log2(self.f_min), np.log2(self.f_max), self.n_filters, base=2.0
        )

    def _init_gammatone_filters(self) -> None:
        """Build SOS coefficients and stack them on device (no per-call transfer)."""
        filter_order = 4
        erb_scale = 0.6
        T = 1.0 / self.sample_rate

        ERB = 24.7 * (4.37 * self.center_freqs / 1000.0 + 1.0) * erb_scale
        B = 1.019 * 2 * np.pi * ERB

        sos_list = []
        for fc, bw in zip(self.center_freqs, B):
            omega = 2 * np.pi * fc
            r = np.exp(-bw * T)
            theta = omega * T

            a0, a1, a2 = 1.0, -2.0 * r * np.cos(theta), r * r
            b0, b1, b2 = 1.0, 0.0, 0.0
            sos = np.array([[b0, b1, b2, a0, a1, a2]] * filter_order)

            # Normalize gain at center frequency
            w = 2 * np.pi * fc / self.sample_rate
            z = np.exp(1j * w)
            H_section = (b0 + b1 * z**-1 + b2 * z**-2) / (a0 + a1 * z**-1 + a2 * z**-2)
            gain = np.abs(H_section ** filter_order)
            if gain > 0:
                sos[0, 0] = b0 / gain

            sos_list.append(sos)

        # Stack to (n_filters, filter_order, 6) and put on device once.
        sos_stack = np.stack(sos_list, axis=0).astype(np.float64)
        self._sos_device = cp.asarray(sos_stack)
        # cupyx sosfilt expects float64 SOS; data can be float32. Confirmed via testing.

    def _init_y5_kernel(self) -> None:
        """Precompute leaky integration kernel and its FFT for fixed chunk size."""
        tau_sec = self.tau_ms / 1000.0
        tau_samples = int(tau_sec * self.sample_rate)
        t = cp.arange(tau_samples, dtype=FLOAT_DTYPE) / self.sample_rate
        kernel = cp.exp(-t / tau_sec)
        kernel = (kernel / kernel.sum()).astype(FLOAT_DTYPE)

        n_conv = self._chunk_samples + tau_samples - 1
        n_fft = next_fast_len(n_conv)
        self._y5_n_fft = n_fft
        self._y5_pad = (tau_samples - 1) // 2
        self._y5_kernel_fft = cp.fft.rfft(kernel, n=n_fft)
        # broadcast to row vector for the multiply downstream
        self._y5_kernel_fft = self._y5_kernel_fft[None, :]

    def _init_buffers(self) -> None:
        """Preallocate device buffers reused across compute_device calls."""
        n = self._chunk_samples
        self._buf_audio = cp.empty(n, dtype=FLOAT_DTYPE)
        # y3 needs a separate buffer because lateral inhibition writes
        # selectively (top n-1 rows differ from bottom row).
        self._buf_y3 = cp.empty((self.n_filters, n), dtype=FLOAT_DTYPE)
        # Spectrogram output in (n_time, n_freq) Gabor-ready orientation.
        self._buf_spec = cp.empty(
            (self._n_frames_out, self.n_filters), dtype=FLOAT_DTYPE
        )

    # ----- public API --------------------------------------------------------

    @property
    def output_shape(self) -> tuple[int, int]:
        """Shape of compute_device() output: (n_time, n_freq)."""
        return (self._n_frames_out, self.n_filters)

    def compute_device(self, audio: np.ndarray) -> cp.ndarray:
        """
        Hot path. Process one fixed-size audio chunk; return device array.

        Args:
            audio: 1D numpy array of length chunk_samples.

        Returns:
            cp.ndarray of shape (n_time, n_freq), float32, on device.
            View into the internal buffer — copy if you need to retain
            across calls (next compute_device overwrites it).
        """
        if audio.ndim > 1:
            audio = audio.ravel()
        if audio.shape[0] != self._chunk_samples:
            raise ValueError(
                f"expected {self._chunk_samples} samples, got {audio.shape[0]}"
            )

        # Transfer + preprocess in one shot. The reassignment to a name
        # makes the data flow readable; CuPy elides the temporaries it can.
        cp.copyto(self._buf_audio, cp.asarray(audio, dtype=FLOAT_DTYPE))
        a = self._buf_audio
        a -= a.mean()
        a /= cp.abs(a).max() + 1e-10

        y1 = self._y1_cochlear_filter(a)            # (n_filters, n)
        y2 = self._y2_transduction(y1)              # (n_filters, n)
        y3 = self._y3_lateral_inhibition(y2)        # (n_filters, n) — buf_y3
        y4 = self._y4_rectification(y3)             # (n_filters, n) view
        y5 = self._y5_integration(y4)               # (n_filters, n)

        cp.cbrt(y5, out=y5)
        spec = self._downsample(y5)                 # (n_filters, n_frames_out)

        # Write transposed into preallocated output buffer.
        # cp.copyto handles the stride change; result is C-contiguous.
        cp.copyto(self._buf_spec, spec.T)
        return self._buf_spec

    def compute(self, audio: np.ndarray) -> Spectrogram:
        """
        Wrapper: runs compute_device, copies to host, returns Spectrogram dataclass.

        Spectrogram.data uses the legacy (n_freq, n_time) orientation.
        """
        device_out = self.compute_device(audio)        # (n_time, n_freq)
        host = to_numpy(device_out).T                  # (n_freq, n_time)
        frame_period = self.frmlen_ms / 1000.0
        times = np.arange(host.shape[1]) * frame_period
        return Spectrogram(
            data=host,
            times=times,
            freqs=self.center_freqs,
            sr=self.sample_rate,
        )

    # ----- pipeline stages ---------------------------------------------------

    def _y1_cochlear_filter(self, audio: cp.ndarray) -> cp.ndarray:
        """Apply gammatone filterbank. Loops in Python over channels;
        each sosfilt call is a single CUDA launch."""
        n = audio.shape[0]
        out = cp.empty((self.n_filters, n), dtype=FLOAT_DTYPE)
        for i in range(self.n_filters):
            out[i, :] = 2.0 * sosfilt(self._sos_device[i], audio)
        return out

    def _y2_transduction(self, y1: cp.ndarray) -> cp.ndarray:
        """diff + tanh."""
        diff = cp.diff(y1, axis=1, prepend=y1[:, 0:1])
        return cp.tanh(diff * 0.5)

    def _y3_lateral_inhibition(self, y2: cp.ndarray) -> cp.ndarray:
        """Lateral inhibitory network. Writes into preallocated _buf_y3."""
        cp.subtract(y2[:-1, :], y2[1:, :], out=self._buf_y3[:-1, :])
        cp.copyto(self._buf_y3[-1:, :], y2[-1:, :])
        return self._buf_y3

    def _y4_rectification(self, y3: cp.ndarray) -> cp.ndarray:
        """Half-wave rectification. Returns view; downstream FFT reads it."""
        return cp.maximum(y3, 0)

    def _y5_integration(self, y4: cp.ndarray) -> cp.ndarray:
        """Leaky temporal integration via cached FFT-of-kernel."""
        n_fft = self._y5_n_fft
        Y4 = cp.fft.rfft(y4, n=n_fft, axis=1)
        Y4 *= self._y5_kernel_fft                       # in-place multiply
        y5_full = cp.fft.irfft(Y4, n=n_fft, axis=1)
        pad = self._y5_pad
        n = self._chunk_samples
        return y5_full[:, pad : pad + n]

    def _downsample(self, spectrogram: cp.ndarray) -> cp.ndarray:
        """GPU polyphase resampling. No CPU round-trip."""
        return resample_poly(spectrogram, up=1, down=self._L_frm, axis=1)