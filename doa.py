from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt


SPEED_OF_SOUND = 343.0  # m/s at ~20°C, sea level

# Drone propeller energy is concentrated in this band:
#   - blade-pass fundamental: ~100-300 Hz depending on RPM
#   - harmonics extend up to a few kHz
#   - <80 Hz is mostly wind, HVAC, traffic rumble — REJECT
#   - >4000 Hz is mostly hiss, electronic noise — REJECT
# Tuning this band is the single biggest lever for indoor DoA reliability.
DRONE_BAND_LOW_HZ  = 80.0
DRONE_BAND_HIGH_HZ = 4000.0


@dataclass
class DoaResult:
    angle_deg: float        # estimated angle of arrival, [-90, +90]
    delay_samples: float    # estimated time delay in fractional samples
    delay_seconds: float    # same in seconds
    confidence: float       # peak height of the GCC; higher = more confident
    max_delay_samples: int  # geometric maximum given baseline + sample rate


# Pre-compute the bandpass filter coefficients once. The filter design depends
# only on the sample rate and band edges, both fixed for our use case.
_FILTER_CACHE: dict = {}

def _get_bandpass_sos(sr: int, low_hz: float, high_hz: float) -> np.ndarray:
    """Cached bandpass filter design. 4th-order Butterworth, sos format."""
    key = (sr, low_hz, high_hz)
    if key not in _FILTER_CACHE:
        _FILTER_CACHE[key] = butter(
            4, [low_hz, high_hz], btype="band", fs=sr, output="sos"
        )
    return _FILTER_CACHE[key]


def _bandpass(signal: np.ndarray, sr: int,
              low_hz: float = DRONE_BAND_LOW_HZ,
              high_hz: float = DRONE_BAND_HIGH_HZ) -> np.ndarray:
    """Zero-phase bandpass filter to drone-relevant frequencies."""
    sos = _get_bandpass_sos(sr, low_hz, high_hz)
    return sosfiltfilt(sos, signal).astype(np.float32)


def gcc_phat(
    left: np.ndarray,
    right: np.ndarray,
    sr: int,
    baseline_m: float,
    interp: bool = True,
    bandpass: bool = True,
) -> DoaResult:
    """
    Estimate direction of arrival via GCC-PHAT.

    Args:
        left:        (N,) mono audio from the left mic, float32 or int16
        right:       (N,) mono audio from the right mic, same length
        sr:          sample rate (Hz)
        baseline_m:  distance between the two mics in meters
        interp:      sub-sample parabolic interpolation around the peak
        bandpass:    apply drone-band bandpass before correlation. Big indoor
                     win — set False to disable for debugging.

    Returns:
        DoaResult with the estimated angle.
    """
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: left {left.shape}, right {right.shape}")
    if left.ndim != 1:
        raise ValueError(f"expected 1D mono audio, got {left.shape}")

    left = left.astype(np.float32)
    right = right.astype(np.float32)

    if bandpass:
        left  = _bandpass(left,  sr)
        right = _bandpass(right, sr)

    # Geometric maximum delay given baseline (samples cannot exceed this)
    max_delay_seconds = baseline_m / SPEED_OF_SOUND
    max_delay_samples = int(np.ceil(max_delay_seconds * sr))

    # ── Cross-correlation via FFT with PHAT weighting ────────────────────────
    n = len(left)
    n_fft = 1 << (2 * n - 1).bit_length()

    L = np.fft.rfft(left,  n=n_fft)
    R = np.fft.rfft(right, n=n_fft)

    cross = L * np.conj(R)

    # PHAT weighting: divide by magnitude → only phase information remains.
    # Whitens the spectrum, sharpens the correlation peak for broadband
    # signals, improves robustness to room reverb.
    eps = 1e-10
    cross_phat = cross / (np.abs(cross) + eps)

    gcc = np.fft.irfft(cross_phat, n=n_fft)
    gcc = np.fft.fftshift(gcc)
    center = n_fft // 2
    search = gcc[center - max_delay_samples : center + max_delay_samples + 1]

    peak_idx = int(np.argmax(search))
    delay_samples = peak_idx - max_delay_samples

    # ── Sub-sample interpolation ─────────────────────────────────────────────
    if interp and 0 < peak_idx < len(search) - 1:
        y0, y1, y2 = search[peak_idx - 1], search[peak_idx], search[peak_idx + 1]
        denom = y0 - 2 * y1 + y2
        if abs(denom) > 1e-12:
            offset = 0.5 * (y0 - y2) / denom
            delay_samples = (peak_idx - max_delay_samples) + offset
            confidence = float(y1 - 0.25 * (y0 - y2) * offset)
        else:
            confidence = float(y1)
    else:
        confidence = float(search[peak_idx])

    # ── Convert delay → angle ────────────────────────────────────────────────
    delay_seconds = delay_samples / sr
    sin_arg = delay_seconds * SPEED_OF_SOUND / baseline_m
    sin_arg = np.clip(sin_arg, -1.0, 1.0)
    angle_rad = np.arcsin(sin_arg)
    angle_deg = float(np.degrees(angle_rad))

    return DoaResult(
        angle_deg=angle_deg,
        delay_samples=float(delay_samples),
        delay_seconds=float(delay_seconds),
        confidence=confidence,
        max_delay_samples=max_delay_samples,
    )


# ── Temporal smoothing ────────────────────────────────────────────────────────

class SmoothedDoa:
    """Confidence-weighted exponential moving average over angle estimates.

    Each new estimate updates the held angle proportional to its confidence.
    Low-confidence estimates barely move the held angle; high-confidence
    estimates pull it harder. Below `min_confidence` the held angle isn't
    updated at all (so noise spikes don't corrupt a good lock).

    Usage:
        smoother = SmoothedDoa()
        for chunk in stream:
            raw = gcc_phat(...)
            smoothed_angle, smoothed_conf = smoother.update(raw.angle_deg, raw.confidence)
    """

    def __init__(self, alpha: float = 0.4, min_confidence: float = 0.15):
        """
        Args:
            alpha: base learning rate. Higher = more responsive, more wobble.
                   0.4 is a good demo default (reaches new estimate in ~3 samples
                   of high-confidence input).
            min_confidence: ignore estimates below this raw confidence threshold.
        """
        self.alpha = alpha
        self.min_confidence = min_confidence
        self.angle: float | None = None
        self.confidence: float = 0.0

    def update(self, raw_angle_deg: float, raw_confidence: float) -> tuple[float, float]:
        if raw_confidence < self.min_confidence:
            # Hold the previous estimate; don't pollute with low-conf noise
            return (self.angle if self.angle is not None else 0.0,
                    self.confidence)

        if self.angle is None:
            # First good sample — initialize directly
            self.angle = raw_angle_deg
            self.confidence = raw_confidence
        else:
            w = self.alpha * raw_confidence
            self.angle      = (1 - w) * self.angle      + w * raw_angle_deg
            self.confidence = (1 - self.alpha) * self.confidence + self.alpha * raw_confidence

        return self.angle, self.confidence

    def reset(self) -> None:
        self.angle = None
        self.confidence = 0.0


# ── Self-test ────────────────────────────────────────────────────────────────

def _self_test() -> None:
    """Synthesize stereo audio with a known time delay, verify recovery.
    Tests both raw GCC-PHAT and the smoothed version."""
    sr = 16000
    duration = 1.0
    n = int(sr * duration)
    baseline = 0.3198   # 1 ft + 1.5 cm

    np.random.seed(42)
    source = np.random.randn(n).astype(np.float32)

    test_cases = [
        ("broadside",       0),
        ("left  side (4)", -4),
        ("right side (3)",  3),
    ]

    print(f"Self-test: sr={sr}, baseline={baseline}m")
    print(f"  Max delay: {baseline / SPEED_OF_SOUND * sr:.1f} samples")
    print(f"  Bandpass:  {DRONE_BAND_LOW_HZ:.0f}-{DRONE_BAND_HIGH_HZ:.0f} Hz")
    print()
    print(f"  {'case':<20s}  {'true':>5s}  {'est':>7s}  {'angle':>7s}  {'conf':>6s}")
    print(f"  {'-' * 20}  {'-' * 5}  {'-' * 7}  {'-' * 7}  {'-' * 6}")

    for label, true_delay in test_cases:
        left = source
        if true_delay >= 0:
            right = np.concatenate([np.zeros(true_delay), source[:-true_delay]
                                    if true_delay > 0 else source])
        else:
            d = -true_delay
            right = np.concatenate([source[d:], np.zeros(d)])
        right = right + 0.01 * np.random.randn(n).astype(np.float32)

        result = gcc_phat(left, right, sr=sr, baseline_m=baseline)
        print(f"  {label:<20s}  {true_delay:>5d}  {result.delay_samples:>+7.2f}  "
              f"{result.angle_deg:>+6.1f}°  {result.confidence:>6.4f}")

    # Smoother demo: noisy estimates with intermittent high-confidence hits
    print()
    print("Smoother demo: 10 noisy estimates of a true +30° source")
    smoother = SmoothedDoa()
    np.random.seed(0)
    for i in range(10):
        # Mostly low-conf noise around 0°, two clean hits at +30°
        if i in (3, 7):
            raw_angle = 30.0 + np.random.randn() * 2.0
            raw_conf  = 0.6
        else:
            raw_angle = np.random.randn() * 40.0
            raw_conf  = 0.05 + np.random.rand() * 0.1
        smooth_angle, smooth_conf = smoother.update(raw_angle, raw_conf)
        marker = " <- high-conf" if raw_conf >= 0.15 else ""
        print(f"  step {i:2d}  raw {raw_angle:+6.1f}° (c={raw_conf:.2f})  "
              f"smoothed {smooth_angle:+6.1f}° (c={smooth_conf:.2f}){marker}")


if __name__ == "__main__":
    _self_test()