"""
compute_features.py — extract RSF features from every clip in the dataset.

Walks data/manifest.csv, runs each wav through PyJetsonSTM, mean-collapses
the time axis, and saves all features into a single .npz cache.

Run once. Re-run only if dataset or feature pipeline config changes.

Output (cache/features.npz): X, ids, labels, cohorts, splits
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features import Config, PyJetsonSTM


DATA_ROOT     = Path("data/output")
MANIFEST_PATH = DATA_ROOT / "manifest.csv"
CACHE_PATH    = Path("cache/features.npz")
CHUNK_SAMPLES = 16000


def wav_path(row: pd.Series) -> Path:
    """Mirror dataset builder's directory layout."""
    if row["label"] == "drone":
        return DATA_ROOT / row["split"] / "drone" / f"{row['id']}.wav"
    return DATA_ROOT / row["split"] / "nodrone" / row["cohort"] / f"{row['id']}.wav"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--cache",    type=Path, default=CACHE_PATH)
    parser.add_argument("--limit",    type=int,  default=None,
                        help="process only first N rows (smoke test)")
    args = parser.parse_args()

    if not args.manifest.exists():
        sys.exit(f"ERROR: manifest not found at {args.manifest}")

    df = pd.read_csv(args.manifest, dtype={"id": str})
    if args.limit:
        df = df.head(args.limit)
        print(f"[LIMIT] processing first {args.limit} rows")
    print(f"Loaded {len(df):,} clips from {args.manifest}")

    pipeline = PyJetsonSTM(Config(), chunk_samples=CHUNK_SAMPLES)
    _, n_rates, n_scales, n_freq = pipeline.output_shape
    print(f"Output shape per clip: ({n_rates}, {n_scales}, {n_freq}) after time-collapse")

    X = np.empty((len(df), n_rates, n_scales, n_freq), dtype=np.float32)
    failed: list[tuple[str, str]] = []

    t0 = time.monotonic()
    for i, row in enumerate(tqdm(df.itertuples(index=False), total=len(df), ncols=72)):
        try:
            audio, _ = sf.read(str(wav_path(row._asdict())), dtype="float32")
            rsf_dev = pipeline.compute_device(audio)
            X[i] = cp.asnumpy(rsf_dev.mean(axis=0)).astype(np.float32)
        except Exception as e:
            failed.append((row.id, str(e)))
            X[i] = 0.0

    elapsed = time.monotonic() - t0
    print(f"\nExtracted {len(df) - len(failed):,}/{len(df):,} features in {elapsed:.0f}s "
          f"({(len(df) - len(failed)) / elapsed:.1f} clips/sec)")
    if failed:
        print(f"Failures: {len(failed)}")
        for fid, err in failed[:5]:
            print(f"  {fid}: {err}")

    args.cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.cache,
        X=X,
        ids=df["id"].to_numpy().astype("U10"),
        labels=df["label"].to_numpy().astype("U10"),
        cohorts=df["cohort"].to_numpy().astype("U30"),
        splits=df["split"].to_numpy().astype("U10"),
    )
    size_mb = args.cache.stat().st_size / (1024 ** 2)
    print(f"Cached → {args.cache} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()