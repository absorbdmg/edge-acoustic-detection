"""
train.py — fit tSVD + cohort GMMs from cached features, evaluate on test set.

Input:  cache/features.npz (built by compute_features.py)
Output: models/tsvd.pkl, models/gmm.pkl
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training import CohortGMM, TSVDReducer


CACHE_PATH = Path("cache/features.npz")
MODELS_DIR = Path("models")
TARGET     = "drone"   # the positive class cohort name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache",      type=Path, default=CACHE_PATH)
    parser.add_argument("--out",        type=Path, default=MODELS_DIR)
    parser.add_argument("--ranks",      type=int, nargs=3, default=[4, 4, 6])
    parser.add_argument("--components", type=int, default=16)
    parser.add_argument("--target-fpr", type=float, default=0.01,
                        help="report TPR at this false-positive rate")
    args = parser.parse_args()

    if not args.cache.exists():
        sys.exit(f"ERROR: feature cache not found at {args.cache}.\n"
                 f"       Run compute_features.py first.")

    # ── Load features ─────────────────────────────────────────────────────────
    print(f"Loading {args.cache}...")
    npz = np.load(args.cache, allow_pickle=True)
    X = npz["X"]
    df = pd.DataFrame({
        "id":      npz["ids"],
        "label":   npz["labels"],
        "cohort":  npz["cohorts"],
        "split":   npz["splits"],
        "idx":     np.arange(len(npz["ids"])),
    })
    print(f"  X: {X.shape} {X.dtype}")
    print(f"  Splits: {dict(df['split'].value_counts())}")

    # Train and test partitions (we don't actually use val in this script —
    # threshold sweep on test is the operating-point selector for v1).
    train = df[df["split"] == "train"]
    test  = df[df["split"] == "test"]

    # ── Fit tSVD on training tensors only ─────────────────────────────────────
    print(f"\nFitting tSVD with ranks {tuple(args.ranks)}...")
    t0 = time.monotonic()
    train_tensors_last = X[train["idx"].values].transpose(1, 2, 3, 0)  # samples-last
    tsvd = TSVDReducer(ranks=tuple(args.ranks)).fit(train_tensors_last)
    feature_dim = int(np.prod(args.ranks))
    print(f"  Done in {time.monotonic() - t0:.1f}s, output dim = {feature_dim}")

    # ── Project all samples ───────────────────────────────────────────────────
    print("Projecting all samples...")
    t0 = time.monotonic()
    V = np.stack([tsvd.transform(X[i]) for i in range(len(X))])
    print(f"  Done in {time.monotonic() - t0:.1f}s, V: {V.shape}")

    # ── Group training features by cohort, treating drone as a cohort ─────────
    # For drone clips: every clip's cohort is also "drone" (as set in the manifest).
    print(f"\nFitting GMMs ({args.components} components)...")
    V_by_cohort: dict[str, np.ndarray] = {
        cohort: V[group["idx"].values]
        for cohort, group in train.groupby("cohort")
    }
    for c, v in sorted(V_by_cohort.items(), key=lambda kv: -len(kv[1])):
        print(f"  {c:<25s} {len(v):>6,} samples")

    t0 = time.monotonic()
    gmm = CohortGMM(n_components=args.components).fit(V_by_cohort, target=TARGET)
    print(f"  Done in {time.monotonic() - t0:.1f}s")

    # ── Evaluate on test ──────────────────────────────────────────────────────
    print("\n── Evaluation (test set) ────────────────────────────────────────")
    V_test = V[test["idx"].values]
    y_test = (test["label"].values == "drone").astype(int)
    llr = gmm.llr(V_test)

    y_pred = (llr > 0).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Accuracy (LLR > 0):  {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(f"  ROC AUC:             {roc_auc_score(y_test, llr):.4f}")
    print(f"\n  Confusion matrix (LLR > 0):")
    print(f"    pred:        nodrone  drone")
    print(f"    true nodrone   {cm[0, 0]:>7}  {cm[0, 1]:>5}")
    print(f"    true drone     {cm[1, 0]:>7}  {cm[1, 1]:>5}")

    fpr, tpr, thresholds = roc_curve(y_test, llr)
    idx = int(np.argmin(np.abs(fpr - args.target_fpr)))
    print(f"\n  At FPR={fpr[idx]:.3f}: TPR={tpr[idx]:.3f}, threshold={thresholds[idx]:.3f}")

    # Per-cohort accuracy on test set
    print("\n  Per-cohort accuracy on test set:")
    test_with_pred = test.assign(pred=y_pred, y=y_test)
    for cohort, group in test_with_pred.groupby("cohort"):
        n = len(group)
        correct = (group["y"] == group["pred"]).sum()
        true_label = group["label"].iloc[0]
        print(f"    {cohort:<25s} {correct / n * 100:>6.2f}%  "
              f"({correct}/{n} correct, true={true_label})")

    # ── Save ──────────────────────────────────────────────────────────────────
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "tsvd.pkl", "wb") as f:
        pickle.dump(tsvd, f)
    with open(args.out / "gmm.pkl", "wb") as f:
        pickle.dump(gmm, f)
    print(f"\nSaved tsvd.pkl, gmm.pkl → {args.out}/")


if __name__ == "__main__":
    main()