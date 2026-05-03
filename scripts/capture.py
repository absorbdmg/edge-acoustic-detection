"""
Stereo audio capture from ESP32-S3 over USB serial.

Firmware sends interleaved L/R 16-bit PCM at 16 kHz, no framing.
"""

import threading
import time
from dataclasses import dataclass

import numpy as np
import serial


@dataclass
class CaptureConfig:
    port: str = "/dev/ttyACM0"      # Linux S3 native USB; macOS: /dev/cu.usbmodem*
    baudrate: int = 921600          # ignored by USB CDC but pyserial wants it
    sample_rate: int = 16000
    channels: int = 2
    sample_dtype: type = np.int16
    read_chunk_bytes: int = 4096    # bytes per serial.read() call
    ringbuf_seconds: float = 4.0    # how much audio to keep in memory


class StereoCapture:
    """
    Background thread reads bytes from serial, deinterleaves to L/R,
    writes into a preallocated ring buffer. Foreground pulls fixed-size frames.

    Ring buffer layout:
        - shape (capacity, channels) int16, C-contiguous
        - write_pos: next index to write to (advances on producer side)
        - read_pos:  next index to read from (advances on consumer side)
        - count:     samples currently available (write_pos - read_pos, modular)
        - capacity:  max samples the buffer can hold

    Reads and writes both wrap around capacity. Lock protects the three indices;
    actual numpy slicing happens outside the lock for minimal hold time.

    Usage:
        cap = StereoCapture(CaptureConfig(port="/dev/ttyACM0"))
        cap.start()
        frame = cap.get_frame(n_samples=16128)   # blocks until ready
        # frame is shape (16128, 2) int16  -- columns are L, R
        cap.stop()
    """

    def __init__(self, config: CaptureConfig | None = None):
        self.cfg = config or CaptureConfig()
        self._ser: serial.Serial | None = None

        self._bytes_per_sample = np.dtype(self.cfg.sample_dtype).itemsize
        self._bytes_per_pair = self._bytes_per_sample * self.cfg.channels  # 4

        self._capacity = int(self.cfg.sample_rate * self.cfg.ringbuf_seconds)
        self._buf = np.zeros(
            (self._capacity, self.cfg.channels), dtype=self.cfg.sample_dtype
        )
        self._write_pos = 0
        self._read_pos = 0
        self._count = 0  # samples currently available

        self._lock = threading.Lock()
        self._data_available = threading.Condition(self._lock)

        self._reader_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # leftover bytes from a partial sample-pair across reads
        self._leftover = b""

        # diagnostics
        self.bytes_read = 0
        self.samples_dropped = 0  # ring overruns
        self.start_time: float | None = None

    def start(self) -> None:
        if self._reader_thread is not None:
            raise RuntimeError("already started")
        self._ser = serial.Serial(
            port=self.cfg.port,
            baudrate=self.cfg.baudrate,
            timeout=0.1,
        )
        self._ser.reset_input_buffer()
        self._stop_event.clear()
        self.start_time = time.monotonic()
        self._reader_thread = threading.Thread(
            target=self._reader_loop, name="stereo-capture", daemon=True
        )
        self._reader_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2.0)
            self._reader_thread = None
        if self._ser is not None:
            self._ser.close()
            self._ser = None

    def _reader_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                raw = self._ser.read(self.cfg.read_chunk_bytes)
            except serial.SerialException:
                break
            if not raw:
                continue

            self.bytes_read += len(raw)

            # align to L/R sample-pair boundary
            buf = self._leftover + raw
            n_pairs = len(buf) // self._bytes_per_pair
            consumed = n_pairs * self._bytes_per_pair
            self._leftover = buf[consumed:]
            if n_pairs == 0:
                continue

            samples = np.frombuffer(
                buf[:consumed], dtype=self.cfg.sample_dtype
            ).reshape(-1, self.cfg.channels)

            self._write(samples)

    def _write(self, samples: np.ndarray) -> None:
        """Append samples to ring buffer, advancing write_pos. Wraps around capacity."""
        n = samples.shape[0]
        cap = self._capacity

        with self._data_available:
            # If we'd overrun, drop oldest by advancing read_pos.
            # (Producer wins; consumer falls behind only if it's already starving.)
            free = cap - self._count
            if n > free:
                drop = n - free
                self._read_pos = (self._read_pos + drop) % cap
                self._count -= drop
                self.samples_dropped += drop

            # Write in up to two slices to handle wrap.
            wp = self._write_pos
            first = min(n, cap - wp)
            self._buf[wp:wp + first] = samples[:first]
            if first < n:
                self._buf[:n - first] = samples[first:]
            self._write_pos = (wp + n) % cap
            self._count += n

            self._data_available.notify_all()

    def get_frame(self, n_samples: int, timeout: float | None = None) -> np.ndarray:
        """
        Block until n_samples per channel are available, return (n_samples, 2) int16.

        Each call CONSUMES the samples from the buffer (FIFO).
        """
        if n_samples > self._capacity:
            raise ValueError(
                f"requested {n_samples} > capacity {self._capacity}; "
                f"increase ringbuf_seconds"
            )

        deadline = None if timeout is None else time.monotonic() + timeout
        with self._data_available:
            while self._count < n_samples:
                remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                if remaining == 0.0:
                    raise TimeoutError(f"only {self._count} samples after timeout")
                self._data_available.wait(timeout=remaining)

            # Read in up to two slices to handle wrap.
            out = np.empty((n_samples, self.cfg.channels), dtype=self.cfg.sample_dtype)
            rp = self._read_pos
            first = min(n_samples, self._capacity - rp)
            out[:first] = self._buf[rp:rp + first]
            if first < n_samples:
                out[first:] = self._buf[:n_samples - first]
            self._read_pos = (rp + n_samples) % self._capacity
            self._count -= n_samples
            return out

    def buffer_fill(self) -> tuple[int, int]:
        """Returns (current samples, capacity) for monitoring."""
        with self._lock:
            return self._count, self._capacity

    def stats(self) -> dict:
        elapsed = time.monotonic() - self.start_time if self.start_time else 0.0
        expected_bytes = elapsed * self.cfg.sample_rate * self._bytes_per_pair
        return {
            "elapsed_sec": round(elapsed, 2),
            "bytes_read": self.bytes_read,
            "expected_bytes": int(expected_bytes),
            "drift_pct": round((self.bytes_read / expected_bytes - 1) * 100, 2)
                          if expected_bytes > 0 else 0.0,
            "samples_dropped": self.samples_dropped,
            "buffer_fill": self.buffer_fill(),
        }