"""
infer.py — classify drone vs nodrone from a wav file.

Loads tsvd.pkl + gmm.pkl from models/, slices the input wav into 1-sec chunks,
runs each chunk through features → tSVD → cohort GMM → LLR, prints per-chunk
results.

For v1: wav file in, terminal output. ESP32 streaming swap is a follow-up.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features import Config, PyJetsonSTM
from training import CohortGMM, TSVDReducer


CHUNK_SAMPLES = 16000
SAMPLE_RATE   = 16000


def load_models(models_dir: Path) -> tuple[TSVDReducer, CohortGMM]:
    with open(models_dir / "tsvd.pkl", "rb") as f:
        tsvd = pickle.load(f)
    with open(models_dir / "gmm.pkl", "rb") as f:
        gmm = pickle.load(f)
    return tsvd, gmm


def normalize_chunk(chunk: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
    """Match the training pipeline's mean-center + RMS-normalize."""
    chunk = chunk - chunk.mean()
    rms = float(np.sqrt(np.mean(chunk ** 2)))
    if rms < 1e-8:
        return chunk
    chunk = chunk * (target_rms / rms)
    # Peak limiter (matches dataset builder)
    peak = float(np.abs(chunk).max())
    if peak > 0.95:
        chunk = chunk * (0.95 / peak)
    return chunk.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wav", type=Path, help="input wav file")
    parser.add_argument("--models",    type=Path, default=Path("models"))
    parser.add_argument("--threshold", type=float, default=0.152,
                        help="LLR threshold for drone decision (train.py reported 0.152 at 1%% FPR)")
    parser.add_argument("--stride",    type=float, default=1.0,
                        help="seconds between chunks; 1.0 = non-overlapping, 0.5 = 50%% overlap")
    parser.add_argument("--quiet",     action="store_true",
                        help="only print summary, not per-chunk LLR")
    args = parser.parse_args()

    if not args.wav.exists():
        sys.exit(f"ERROR: wav file not found at {args.wav}")
    if not (args.models / "tsvd.pkl").exists():
        sys.exit(f"ERROR: models not found in {args.models}/. Run train.py first.")

    # ── Load models + pipeline ────────────────────────────────────────────────
    print(f"Loading models from {args.models}/...")
    tsvd, gmm = load_models(args.models)
    print(f"  tSVD ranks: {tsvd.ranks}")
    print(f"  GMM cohorts: {sorted(gmm.gmms.keys())} (target='{gmm.target}')")

    print("Building feature pipeline...")
    pipeline = PyJetsonSTM(Config(), chunk_samples=CHUNK_SAMPLES)

    # ── Load + preprocess audio ───────────────────────────────────────────────
    print(f"Loading {args.wav}...")
    audio, sr = sf.read(str(args.wav), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # downmix to mono
    if sr != SAMPLE_RATE:
        sys.exit(f"ERROR: expected sr={SAMPLE_RATE}, got {sr}. Resample first.")

    print(f"  Duration: {len(audio) / sr:.1f}s ({len(audio)} samples)")

    # ── Slice into chunks (configurable stride) ───────────────────────────────
    stride_samples = int(args.stride * SAMPLE_RATE)
    n_chunks = max(0, (len(audio) - CHUNK_SAMPLES) // stride_samples + 1)
    if n_chunks == 0:
        sys.exit(f"ERROR: audio too short ({len(audio)} samples < {CHUNK_SAMPLES})")

    print(f"  {n_chunks} chunks @ {args.stride}s stride")
    print(f"  Decision threshold: LLR > {args.threshold}\n")

    # ── Score each chunk ──────────────────────────────────────────────────────
    if not args.quiet:
        print(f"  {'time':>6s}  {'LLR':>8s}  decision")
        print(f"  {'-' * 6}  {'-' * 8}  --------")

    llrs = np.empty(n_chunks, dtype=np.float32)
    detections = 0
    t_start = time.monotonic()

    for i in range(n_chunks):
        start = i * stride_samples
        chunk = audio[start : start + CHUNK_SAMPLES]
        chunk = normalize_chunk(chunk)

        # features → tSVD → GMM
        rsf_dev = pipeline.compute_device(chunk)
        rsf = cp.asnumpy(rsf_dev.mean(axis=0)).astype(np.float32)
        V = tsvd.transform(rsf)
        llr = float(gmm.llr(V[None, :])[0])

        llrs[i] = llr
        is_drone = llr > args.threshold
        if is_drone:
            detections += 1

        if not args.quiet:
            t = i * args.stride
            mark = "★ DRONE" if is_drone else ""
            print(f"  {t:>5.1f}s  {llr:>+8.3f}  {mark}")

    elapsed = time.monotonic() - t_start

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n── Summary ──────────────────────────────────────")
    print(f"  Chunks processed:  {n_chunks}")
    print(f"  Drone detections:  {detections} ({detections / n_chunks * 100:.1f}%)")
    print(f"  LLR mean:          {llrs.mean():+.3f}")
    print(f"  LLR median:        {np.median(llrs):+.3f}")
    print(f"  LLR max:           {llrs.max():+.3f}")
    print(f"  LLR min:           {llrs.min():+.3f}")
    print(f"  Processing time:   {elapsed:.1f}s ({n_chunks / elapsed:.1f} chunks/sec)")
    real_time_factor = (n_chunks * args.stride) / elapsed
    print(f"  Real-time factor:  {real_time_factor:.1f}x "
          f"({'real-time capable' if real_time_factor >= 1.0 else 'slower than real-time'})")


if __name__ == "__main__":
    main()