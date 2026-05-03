"""
realtime.py — live drone detection + DoA from ESP32 stereo capture.

Loop:
    1. Maintain a sliding 1-sec window of stereo audio
    2. Slide by HOP_SAMPLES = WINDOW_SAMPLES / hop_hz (1, 2, or 4 Hz)
    3. Compute DoA (GCC-PHAT) on both channels — CPU, ~1-3ms
    4. Mix to mono for classification
    5. Mean-center + RMS-normalize (matches dataset builder)
    6. Run features → tSVD → cohort GMM
    7. Print LLR + decision + angle + buffer stats

Updates per second is controlled by --hop-hz {1,2,4}. The 1-sec WINDOW
size never changes (model expects 16,000 samples), only the hop between
windows. With hop_hz=2, each second of audio is classified twice in
overlapping 1-sec windows.

Real-time-factor budget on A1000:
    1 Hz  →  ~140ms work per 1.0 sec audio   = 7.0x RT factor (safe)
    2 Hz  →  ~280ms work per 1.0 sec audio   = 3.5x RT factor (safe)
    4 Hz  →  ~560ms work per 1.0 sec audio   = 1.8x RT factor (tight)

Ctrl+C for clean shutdown. Capture thread is daemon.
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
from doa import gcc_phat
from features import Config, PyJetsonSTM
from training import CohortGMM, TSVDReducer


CHUNK_SAMPLES = 16000             # 1 sec @ 16 kHz — model expects this
TARGET_RMS = 0.1
MIC_BASELINE_M = 0.1524           # 6 inches

MIC_LEFT_CH  = 1                  # mics physically swapped — pick channels here
MIC_RIGHT_CH = 0


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
    parser.add_argument("--port",      default="/dev/ttyACM0")
    parser.add_argument("--models",    type=Path, default=Path("models"))
    parser.add_argument("--threshold", type=float, default=1.455,
                        help="LLR > threshold → DRONE")
    parser.add_argument("--baseline",  type=float, default=MIC_BASELINE_M)
    parser.add_argument("--warmup-frames", type=int, default=2,
                        help="discard this many frames at start")
    parser.add_argument("--hop-hz", type=int, default=2, choices=[1, 2, 4],
                        help="inference updates per second (default: 2)")
    args = parser.parse_args()

    WINDOW_SAMPLES = CHUNK_SAMPLES
    HOP_SAMPLES = WINDOW_SAMPLES // args.hop_hz   # 16000, 8000, or 4000

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

    print(f"Threshold:    LLR > {args.threshold} → DRONE")
    print(f"Mic baseline: {args.baseline:.4f} m ({args.baseline * 39.37:.1f}\")")
    print(f"Update rate:  {args.hop_hz} Hz  (hop = {HOP_SAMPLES} samples = "
          f"{HOP_SAMPLES / 16000:.3f} sec)")
    print(f"Press Ctrl+C to stop.\n")
    print(f"  {'idx':>4s}  {'LLR':>8s}  decision   {'angle':>7s}   {'conf':>5s}   buffer    drift")
    print(f"  {'-' * 4}  {'-' * 8}  --------   {'-' * 7}   {'-' * 5}   --------  -----")

    # ── Prime the buffer with one full window ─────────────────────────────────
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
            # Slide window: pull HOP_SAMPLES new samples, shift buffer
            try:
                new_samples = cap.get_frame(HOP_SAMPLES, timeout=2.0)
            except TimeoutError as e:
                print(f"  WARN: capture timeout ({e}); is the ESP32 streaming?")
                continue

            buffer = np.concatenate(
                [buffer[HOP_SAMPLES:], new_samples], axis=0
            )

            # Split channels (mics physically swapped — see MIC_*_CH constants)
            left  = buffer[:, MIC_LEFT_CH ].astype(np.float32)
            right = buffer[:, MIC_RIGHT_CH].astype(np.float32)

            # Process
            t0 = time.monotonic()

            doa = gcc_phat(left, right, sr=16000, baseline_m=args.baseline)

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
                  f"{doa.angle_deg:>+6.1f}°   "
                  f"{doa.confidence:>5.3f}   "
                  f"{buffer_str:>9s}  {drift_str:>5s}")

            chunk_times.append(chunk_time)
            idx += 1

    finally:
        # ── Shutdown ──────────────────────────────────────────────────────────
        print(f"\nStopping...")
        cap.stop()

        if chunk_times:
            avg = sum(chunk_times) / len(chunk_times)
            audio_per_step = HOP_SAMPLES / 16000.0
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