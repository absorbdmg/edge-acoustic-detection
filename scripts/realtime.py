"""
realtime.py — live drone detection + DoA from ESP32 stereo capture.

Sliding-window inference loop:
    1. Maintain a 1-sec stereo window
    2. Slide by HOP_SAMPLES = WINDOW_SAMPLES / hop_hz (1, 2, or 4 Hz)
    3. Bandpassed GCC-PHAT for DoA, with temporal smoothing
    4. Mono mix → RMS-normalize → features → tSVD → cohort GMM
    5. Print LLR + decision + smoothed angle + buffer stats

All shared constants come from config.py — change once there, every entry
point picks it up. Ctrl+C for clean shutdown.
"""

from __future__ import annotations

import argparse
import pickle
import signal
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capture import CaptureConfig, StereoCapture
from config import (
    CHUNK_SAMPLES,
    DEFAULT_HOP_HZ,
    DEFAULT_SERIAL_PORT,
    DEFAULT_THRESHOLD,
    DEFAULT_WARMUP_FRAMES,
    DOA_BANDPASS_ENABLED,
    DOA_MIN_CONFIDENCE,
    DOA_SMOOTHING_ENABLED,
    MIC_BASELINE_M,
    MIC_LEFT_CH,
    MIC_RIGHT_CH,
    SAMPLE_RATE,
    TARGET_RMS,
    WINDOW_SAMPLES,
    doa_alpha_for_hop,
    hop_samples,
)
from doa import SmoothedDoa, gcc_phat
from features import Config, PyJetsonSTM
from training import CohortGMM, TSVDReducer


def normalize_chunk(chunk: np.ndarray) -> np.ndarray:
    """Mean-center, RMS-normalize, peak-limit. Matches dataset builder."""
    chunk = chunk.astype(np.float32)
    chunk = chunk - chunk.mean()
    rms = float(np.sqrt(np.mean(chunk ** 2)))
    if rms < 1e-8:
        return chunk
    chunk = chunk * (TARGET_RMS / rms)
    peak = float(np.abs(chunk).max())
    if peak > 0.95:
        chunk = chunk * (0.95 / peak)
    return chunk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",      default=DEFAULT_SERIAL_PORT)
    parser.add_argument("--models",    type=Path, default=Path("models"))
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"LLR > threshold → DRONE (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--baseline",  type=float, default=MIC_BASELINE_M,
                        help=f"mic baseline in meters (default: {MIC_BASELINE_M:.4f})")
    parser.add_argument("--warmup-frames", type=int, default=DEFAULT_WARMUP_FRAMES,
                        help="discard this many frames at start (kernel JIT, mic settle)")
    parser.add_argument("--hop-hz", type=int, default=DEFAULT_HOP_HZ, choices=[1, 2, 4],
                        help=f"inference updates per second (default: {DEFAULT_HOP_HZ})")
    args = parser.parse_args()

    HOP = hop_samples(args.hop_hz)

    # ── Load models ───────────────────────────────────────────────────────────
    if not (args.models / "tsvd.pkl").exists():
        sys.exit(f"ERROR: models not found in {args.models}/")

    print(f"Loading models from {args.models}/...")
    with open(args.models / "tsvd.pkl", "rb") as f:
        tsvd = pickle.load(f)
    with open(args.models / "gmm.pkl", "rb") as f:
        gmm = pickle.load(f)

    # ── Build feature pipeline (warm up CuPy) ────────────────────────────────
    print("Building feature pipeline...")
    pipeline = PyJetsonSTM(Config(), chunk_samples=WINDOW_SAMPLES)

    print("Warming up GPU (compiling kernels)...")
    t0 = time.monotonic()
    dummy = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    _ = pipeline.compute_device(dummy)
    cp.cuda.Stream.null.synchronize()
    print(f"  Warmup done in {time.monotonic() - t0:.1f}s")

    # ── Start capture ─────────────────────────────────────────────────────────
    print(f"\nOpening {args.port}...")
    cap = StereoCapture(CaptureConfig(port=args.port))

    try:
        cap.start()
    except Exception as e:
        sys.exit(f"ERROR: failed to open {args.port}: {e}")

    # ── Clean shutdown on Ctrl+C ──────────────────────────────────────────────
    stop_requested = False
    def handle_sigint(signum, frame):
        nonlocal stop_requested
        stop_requested = True
    signal.signal(signal.SIGINT, handle_sigint)

    # ── DoA smoother (alpha scales with hop rate; can be disabled in config) ─
    if DOA_SMOOTHING_ENABLED:
        smoother = SmoothedDoa(
            alpha=doa_alpha_for_hop(args.hop_hz),
            min_confidence=DOA_MIN_CONFIDENCE,
        )
    else:
        smoother = None
        print("  [DoA smoothing DISABLED via config]")

    if not DOA_BANDPASS_ENABLED:
        print("  [DoA bandpass DISABLED via config]")

    print(f"Threshold:    LLR > {args.threshold} → DRONE")
    print(f"Mic baseline: {args.baseline:.4f} m ({args.baseline * 39.37:.1f}\")")
    print(f"Update rate:  {args.hop_hz} Hz  (hop = {HOP} samples = "
          f"{HOP / SAMPLE_RATE:.3f} sec)")
    print(f"Press Ctrl+C to stop.\n")
    print(f"  {'idx':>4s}  {'LLR':>8s}  decision   {'angle':>7s}   {'conf':>5s}   buffer    drift")
    print(f"  {'-' * 4}  {'-' * 8}  --------   {'-' * 7}   {'-' * 5}   --------  -----")

    # ── Prime buffer with one full window ─────────────────────────────────────
    try:
        buffer = cap.get_frame(WINDOW_SAMPLES, timeout=2.0)
    except TimeoutError as e:
        sys.exit(f"ERROR: initial buffer timeout ({e}); is ESP32 streaming?")

    # ── Main loop ─────────────────────────────────────────────────────────────
    idx = 0
    detections = 0
    chunk_times: list[float] = []

    try:
        while not stop_requested:
            # Slide window: pull HOP new samples, shift buffer
            try:
                new_samples = cap.get_frame(HOP, timeout=2.0)
            except TimeoutError as e:
                print(f"  WARN: capture timeout ({e}); is the ESP32 streaming?")
                continue

            buffer = np.concatenate([buffer[HOP:], new_samples], axis=0)

            # Split channels (mics physically swapped — see config)
            left  = buffer[:, MIC_LEFT_CH ].astype(np.float32)
            right = buffer[:, MIC_RIGHT_CH].astype(np.float32)

            # Process
            t0 = time.monotonic()

            doa_raw = gcc_phat(left, right, sr=SAMPLE_RATE,
                               baseline_m=args.baseline,
                               bandpass=DOA_BANDPASS_ENABLED)
            if smoother is not None:
                smooth_angle, smooth_conf = smoother.update(
                    doa_raw.angle_deg, doa_raw.confidence
                )
            else:
                smooth_angle, smooth_conf = doa_raw.angle_deg, doa_raw.confidence

            mono = (left + right) * 0.5
            audio_n = normalize_chunk(mono)
            rsf_dev = pipeline.compute_device(audio_n)
            rsf = cp.asnumpy(rsf_dev.mean(axis=0)).astype(np.float32)
            V = tsvd.transform(rsf)
            llr = float(gmm.llr(V[None, :])[0])

            chunk_time = time.monotonic() - t0

            # Skip output for warmup frames
            if idx < args.warmup_frames:
                idx += 1
                continue

            # Decision
            is_drone = llr > args.threshold
            mark = "★ DRONE" if is_drone else "  -    "
            if is_drone:
                detections += 1

            # Buffer + drift stats every (hop_hz × 5) chunks (~5 sec)
            buffer_str = ""
            drift_str = ""
            if idx % (args.hop_hz * 5) == 0:
                stats = cap.stats()
                fill, cap_total = stats["buffer_fill"]
                buffer_str = f"{fill:>5d}/{cap_total}"
                drift_str = f"{stats['drift_pct']:+.1f}%"

            print(f"  {idx:>4d}  {llr:>+8.3f}  {mark}   "
                  f"{smooth_angle:>+6.1f}°   "
                  f"{smooth_conf:>5.3f}   "
                  f"{buffer_str:>9s}  {drift_str:>5s}")

            chunk_times.append(chunk_time)
            idx += 1

    finally:
        # ── Shutdown ──────────────────────────────────────────────────────────
        print(f"\nStopping...")
        cap.stop()

        if chunk_times:
            avg = sum(chunk_times) / len(chunk_times)
            audio_per_step = HOP / SAMPLE_RATE
            rt = audio_per_step / avg
            print(f"\n── Summary ──────────────────────────────────────")
            print(f"  Update rate:       {args.hop_hz} Hz")
            print(f"  Frames processed:  {len(chunk_times)}")
            print(f"  Drone detections:  {detections}")
            print(f"  Avg chunk time:    {avg * 1000:.0f}ms")
            print(f"  Real-time factor:  {rt:.2f}x "
                  f"({'OK' if rt > 1.0 else 'TOO SLOW'})")

        # Final capture stats
        stats = cap.stats()
        print(f"\n  Capture stats:")
        print(f"    Bytes read:       {stats['bytes_read']:,}")
        print(f"    Drift:            {stats['drift_pct']:+.2f}%")
        print(f"    Samples dropped:  {stats['samples_dropped']:,}")


if __name__ == "__main__":
    main()