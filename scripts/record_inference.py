"""
record_inference.py — run inference on a wav file, save canonical JSON.

Output: a JSON list of canonical messages (same shape as websocket frames).
Use with: server.py --playback FILE.json [--loop]
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    DEFAULT_THRESHOLD,
    MIC_BASELINE_M,
    MIC_LEFT_CH,
    MIC_RIGHT_CH,
    SAMPLE_RATE,
    WINDOW_SAMPLES,
)
from server import (
    InferenceFrame,
    build_message,
    normalize_chunk,
    reduce_rsf,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("wav", type=Path)
    p.add_argument("--models", type=Path, default=Path("models"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--baseline", type=float, default=MIC_BASELINE_M)
    args = p.parse_args()

    import cupy as cp
    import soundfile as sf
    from doa import gcc_phat
    from features import Config, PyJetsonSTM

    print(f"loading {args.wav}...")
    audio, sr = sf.read(args.wav)
    if sr != SAMPLE_RATE:
        sys.exit(f"expected {SAMPLE_RATE} Hz, got {sr} Hz; resample first")

    if audio.ndim == 1:
        print("  mono input — DoA will be zero")
        left = right = audio.astype(np.float32)
    else:
        left  = audio[:, MIC_LEFT_CH ].astype(np.float32)
        right = audio[:, MIC_RIGHT_CH].astype(np.float32)

    n_chunks = len(left) // WINDOW_SAMPLES
    print(f"  {n_chunks} chunks ({n_chunks} sec)")

    print("loading models...")
    with open(args.models / "tsvd.pkl", "rb") as f:
        tsvd = pickle.load(f)
    with open(args.models / "gmm.pkl", "rb") as f:
        gmm = pickle.load(f)

    print("building feature pipeline...")
    pipeline = PyJetsonSTM(Config(), chunk_samples=WINDOW_SAMPLES)
    dummy = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    _ = pipeline.compute_device(dummy)
    cp.cuda.Stream.null.synchronize()

    records = []
    t_start = time.monotonic()

    for i in range(n_chunks):
        s = i * WINDOW_SAMPLES
        e = s + WINDOW_SAMPLES
        l = left[s:e]
        r = right[s:e]

        t0 = time.monotonic()
        doa = gcc_phat(l, r, sr=SAMPLE_RATE, baseline_m=args.baseline)
        mono = (l + r) * 0.5
        audio_n = normalize_chunk(mono)
        rsf_dev = pipeline.compute_device(audio_n)
        rsf_full = cp.asnumpy(rsf_dev.mean(axis=0)).astype(np.float32)
        V = tsvd.transform(rsf_full)
        llr = float(gmm.llr(V[None, :])[0])
        chunk_time = time.monotonic() - t0

        inf = InferenceFrame(
            idx=i,
            llr=llr,
            threshold=args.threshold,
            angle_deg=doa.angle_deg,
            doa_confidence=doa.confidence,
            delay_samples=getattr(doa, "delay_samples", 0.0),
            rate_scale=reduce_rsf(rsf_full),
            chunk_ms=chunk_time * 1000.0,
            rt_factor=1.0 / chunk_time if chunk_time > 0 else 0.0,
            drift_pct=0.02,
            samples_dropped=0,
            buffer_fill_pct=4.0,
        )
        records.append(build_message(inf))

        decision = "DRONE" if llr > args.threshold else "clear"
        print(f"  [{i+1:3d}/{n_chunks}] LLR {llr:+.2f}  "
              f"theta {doa.angle_deg:+.1f}  ({decision})")

    elapsed = time.monotonic() - t_start
    print(f"\ndone in {elapsed:.1f}s ({n_chunks/elapsed:.1f} chunks/sec)")
    print(f"writing {args.out}...")
    with open(args.out, "w") as f:
        json.dump(records, f)
    size_mb = args.out.stat().st_size / (1024 ** 2)
    print(f"  {len(records)} records, {size_mb:.1f} MB")


if __name__ == "__main__":
    main()