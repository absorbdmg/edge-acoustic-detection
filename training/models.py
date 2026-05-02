from __future__ import annotations

import numpy as np
from sklearn.mixture import GaussianMixture


class TSVDReducer:
    """Tucker decomposition via mode-wise PCA. Default ranks (4,4,6) → 96-dim."""

    def __init__(self, ranks: tuple[int, int, int] = (4, 4, 6)):
        self.ranks = ranks
        self.factors: list[np.ndarray] | None = None
        self.mean: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "TSVDReducer":
        """X: (n_rates, n_scales, n_freq, n_samples). Samples are the LAST axis."""
        if X.ndim != 4:
            raise ValueError(f"expected 4D samples-last tensor, got {X.shape}")

        self.mean = X.mean(axis=-1)
        centered = X - self.mean[..., np.newaxis]

        self.factors = []
        for mode in range(X.ndim - 1):  # all axes except the sample axis
            unfold = np.reshape(np.moveaxis(centered, mode, 0), (X.shape[mode], -1))
            U, _, _ = np.linalg.svd(np.cov(unfold))
            self.factors.append(U[:, : self.ranks[mode]])

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """X: (n_rates, n_scales, n_freq) → flat (rank0*rank1*rank2,)."""
        if self.factors is None:
            raise RuntimeError("call fit() before transform()")
        G = X - self.mean
        for f in self.factors:
            G = np.tensordot(G, f, axes=([0], [0]))
        return G.flatten()