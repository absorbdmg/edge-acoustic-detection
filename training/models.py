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
    

class CohortGMM:
    """
    One GMM per cohort. At inference time, one cohort is the "target" (drone)
    and the rest are competitors:

        LLR = ll_target - max(ll_competitors)
    """

    def __init__(self, n_components: int = 16):
        self.n_components = n_components
        self.gmms: dict[str, GaussianMixture] = {}
        self.target: str | None = None

    def fit(self, V_by_cohort: dict[str, np.ndarray], target: str) -> "CohortGMM":
        """
        Fit one GMM per cohort.

        Args:
            V_by_cohort: {cohort_name: (n_samples, n_features)}
            target: which cohort is the positive class (e.g. "drone")
        """
        if target not in V_by_cohort:
            raise ValueError(f"target cohort '{target}' not in {list(V_by_cohort)}")

        for cohort, V in V_by_cohort.items():
            if len(V) < self.n_components * 2:
                print(f"  WARN: cohort '{cohort}' has {len(V)} samples "
                      f"(< {self.n_components * 2} for {self.n_components}-comp GMM)")
            self.gmms[cohort] = GaussianMixture(
                n_components=self.n_components,
                covariance_type="diag",
                max_iter=200,
                random_state=42,
                reg_covar=1e-6,
            ).fit(V)

        self.target = target
        return self

    def llr(self, V: np.ndarray) -> np.ndarray:
        """LLR per sample. Positive → target cohort, negative → some competitor."""
        if self.target is None:
            raise RuntimeError("call fit() first")

        ll_target = self.gmms[self.target].score_samples(V)
        ll_competitors = np.stack([
            g.score_samples(V) for c, g in self.gmms.items() if c != self.target
        ], axis=0)
        return ll_target - ll_competitors.max(axis=0)