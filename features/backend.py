"""GPU-only utilities for the streaming pipeline. CuPy is required."""
import cupy as cp
from cupyx.scipy import signal as cp_signal
import numpy as np

# Convenience alias for places where the offline code reads `xp.fft.fft2(...)` —
# new streaming code should just `import cupy as cp` directly.
xp = cp
signal = cp_signal

FLOAT_DTYPE = cp.float32
COMPLEX_DTYPE = cp.complex64


def next_fast_len(n: int) -> int:
    """Round up to next power of 2. cuFFT-friendly, slightly wasteful vs 5-smooth."""
    return int(2 ** np.ceil(np.log2(n)))


def get_available_memory() -> int:
    """Free VRAM on the current device, in bytes."""
    free, _ = cp.cuda.Device().mem_info
    return free


def to_numpy(array) -> np.ndarray:
    """Copy a cupy array to host. No-op for numpy arrays."""
    if isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return np.asarray(array)