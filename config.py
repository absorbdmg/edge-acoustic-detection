"""
config.py — single source of truth for all shared runtime constants.

Anything that's used by more than one entry point (realtime.py, server.py,
record_inference.py, etc.) lives here. Change a value once, every consumer
picks it up.

Imports allowed at top level: only stdlib + numpy. No GPU libs, no project
modules — keep this tree-leaf-clean so any script can import it without
side effects.
"""

from __future__ import annotations


# ── Audio / capture ──────────────────────────────────────────────────────────

SAMPLE_RATE = 16000

# 1-second analysis window. Model expects this exact size; do not change without
# retraining.
WINDOW_SAMPLES = 16000

# Legacy alias — older code still uses CHUNK_SAMPLES. Keep both pointing at the
# same value so renaming can be incremental.
CHUNK_SAMPLES = WINDOW_SAMPLES

# RMS-normalize each window to this target level before classification.
# Matches the dataset builder's normalization.
TARGET_RMS = 0.1


# ── Microphone array ─────────────────────────────────────────────────────────

# Center-to-center distance between mics, meters.
# 1 ft + 1.5 cm = 30.48 + 1.5 = 31.98 cm  →  0.3198 m
MIC_BASELINE_M = 0.3198

# Hardware was wired with channels physically swapped, so we present:
#   frame[:, MIC_LEFT_CH]  → "left"  mic
#   frame[:, MIC_RIGHT_CH] → "right" mic
# If you ever rewire the breadboard, update these two and ONLY these two.
MIC_LEFT_CH = 1
MIC_RIGHT_CH = 0


# ── Classifier ───────────────────────────────────────────────────────────────

# LLR threshold for drone decision. Set during training at the 1% FPR
# operating point.
DEFAULT_THRESHOLD = 1.455

# Discard this many initial frames at startup (kernel JIT, mic settle).
DEFAULT_WARMUP_FRAMES = 2


# ── RSF / feature pipeline (must match `Config()` in features module) ────────

# Used by frontend for axis labels + midline placement. If features.Config
# changes, update these too.
RATES = [-32.0, -16.0, -8.0, -4.0, -2.0, 2.0, 4.0, 8.0, 16.0, 32.0]
SCALES = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
N_RATES = len(RATES)            # = 10
N_SCALES = len(SCALES)           # = 6
MIDLINE_COL = N_RATES // 2       # = 5; vertical line between col 4 and 5


# ── DoA preprocessing toggles ────────────────────────────────────────────────

# Master switches for the two DoA preprocessing stages. Set to False to
# debug raw GCC-PHAT behavior (no bandpass, no temporal smoothing).
DOA_BANDPASS_ENABLED = False
DOA_SMOOTHING_ENABLED = True


# ── DoA smoothing ────────────────────────────────────────────────────────────

# Exponential moving average on the angle estimate. Higher alpha = more
# responsive, more wobble. 0.4 is a good default at 1 Hz update rate;
# scale it down for faster update rates.
DOA_SMOOTHER_ALPHA = 0.4
DOA_MIN_CONFIDENCE = 0.15


# ── Networking ───────────────────────────────────────────────────────────────

DEFAULT_SERVER_HOST = "0.0.0.0"
DEFAULT_HTTP_PORT = 8000
DEFAULT_WS_PATH = "/ws/state"

# Default ESP32 USB CDC port on Linux. macOS would be /dev/cu.usbmodem*
DEFAULT_SERIAL_PORT = "/dev/ttyACM0"


# ── Update rate / sliding window ─────────────────────────────────────────────

# Updates per second. 2 Hz is the demo target on A1000:
#   1 Hz: 7x RT factor (safe, low responsiveness)
#   2 Hz: 3.5x RT factor (safe, smooth)
#   4 Hz: 1.8x RT factor (tight, snappy)
DEFAULT_HOP_HZ = 2


def hop_samples(hop_hz: int = DEFAULT_HOP_HZ) -> int:
    """Window-slide step in samples for a given update rate."""
    return WINDOW_SAMPLES // hop_hz


def doa_alpha_for_hop(hop_hz: int = DEFAULT_HOP_HZ) -> float:
    """Scale the DoA smoother's alpha to keep its time-constant ~constant
    regardless of update rate."""
    return DOA_SMOOTHER_ALPHA / hop_hz
