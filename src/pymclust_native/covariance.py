from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np

from .models import ModelName


VAR_FLOOR = 1e-8
FULL_JITTER = 1e-6


def _safe_det1_diagonal(vec: np.ndarray, floor: float = VAR_FLOOR) -> np.ndarray:
    """Normalize a diagonal vector to have determinant 1 (product = 1).
    Avoid zero by clipping.
    """
    v = np.clip(vec, floor, None)
    geo = np.exp(np.mean(np.log(v)))
    return v / geo


def _floor_diag(v: np.ndarray, floor: float = VAR_FLOOR) -> np.ndarray:
    return np.clip(v, floor, None)


@dataclass
class CovarianceModel:
    name: ModelName
    var_floor: float = VAR_FLOOR

    def init_params(self, G: int, d: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return initial (means, covariances) shapes suitable for the model.
        Placeholder implementation: allocate identity structures.
        """
        means = np.zeros((G, d))
        if self.name in (ModelName.EII, ModelName.VII):
            # spherical: store scalars per component
            if self.name == ModelName.EII:
                covs = np.ones((1,))  # shared scalar
            else:
                covs = np.ones((G,))  # per-component scalar
        elif self.name in (ModelName.EEI, ModelName.VVI):
            # diagonal: store diag vectors
            if self.name == ModelName.EEI:
                covs = np.ones((1, d))
            else:  # VVI
                covs = np.ones((G, d))
        else:
            # general
            if self.name == ModelName.EEE:
                covs = np.tile(np.eye(d)[None, :, :], (1, 1, 1))
            else:  # VVV
                covs = np.tile(np.eye(d)[None, :, :], (G, 1, 1))
        return means, covs

    # --- Updates for constrained covariance models ---
    def update_EII(self, X: np.ndarray, means: np.ndarray, resp: np.ndarray) -> np.ndarray:
        n, d = X.shape
        G = means.shape[0]
        Nk = resp.sum(axis=0) + 1e-16
        total = 0.0
        for g in range(G):
            diff = X - means[g]
            total += np.sum(resp[:, g] * np.sum(diff * diff, axis=1))
        lam = total / (np.sum(Nk) * d)
        lam = float(max(lam, self.var_floor))
        return np.array([lam])

    def update_VII(self, X: np.ndarray, means: np.ndarray, resp: np.ndarray) -> np.ndarray:
        n, d = X.shape
        G = means.shape[0]
        Nk = resp.sum(axis=0) + 1e-16
        lam = np.zeros(G)
        for g in range(G):
            diff = X - means[g]
            lam[g] = np.sum(resp[:, g] * np.sum(diff * diff, axis=1)) / (Nk[g] * d)
        lam = np.clip(lam, self.var_floor, None)
        return lam

    def update_EEI(self, X: np.ndarray, means: np.ndarray, resp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (lambda_scalar, diag_shape) where diag_shape has det=1 and is shared.
        Σ_g = λ A where A is diagonal, det(A)=1, shared across g.
        """
        n, d = X.shape
        G = means.shape[0]
        Nk = resp.sum(axis=0) + 1e-16
        # pooled weighted diagonal covariance
        S = np.zeros(d)
        total_N = np.sum(Nk)
        for g in range(G):
            diff = X - means[g]
            S += np.sum(resp[:, g][:, None] * diff * diff, axis=0)
        S = _floor_diag(S / total_N, self.var_floor)
        A = _safe_det1_diagonal(S, self.var_floor)
        lam = float(np.mean(S / A))
        lam = max(lam, self.var_floor)
        return np.array([lam]), A

    def update_VVI(self, X: np.ndarray, means: np.ndarray, resp: np.ndarray) -> np.ndarray:
        """Return per-component diagonal variances (G, d)."""
        n, d = X.shape
        G = means.shape[0]
        Nk = resp.sum(axis=0) + 1e-16
        diag_vars = np.zeros((G, d))
        for g in range(G):
            diff = X - means[g]
            diag_vars[g] = np.sum(resp[:, g][:, None] * diff * diff, axis=0) / Nk[g]
        diag_vars = _floor_diag(diag_vars, self.var_floor)
        return diag_vars

    def update_VEI(self, X: np.ndarray, means: np.ndarray, resp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """VEI: diagonal with variable volume (lambda_g) and equal shape (A shared, det(A)=1).
        Return (lambda_g vector shape (G,), A_diag shape (d,)).
        Σ_g = λ_g A.
        """
        n, d = X.shape
        G = means.shape[0]
        Nk = resp.sum(axis=0) + 1e-16
        Sg = np.zeros((G, d))
        for g in range(G):
            diff = X - means[g]
            Sg[g] = np.sum(resp[:, g][:, None] * diff * diff, axis=0) / Nk[g]
        A = np.zeros(d)
        total_N = np.sum(Nk)
        for g in range(G):
            s = _floor_diag(Sg[g], self.var_floor)
            geo = np.exp(np.mean(np.log(s)))
            A += Nk[g] * (s / geo)
        A = _safe_det1_diagonal(A / total_N, self.var_floor)
        lam = np.zeros(G)
        for g in range(G):
            lam[g] = float(np.mean(Sg[g] / A))
        lam = _floor_diag(lam, self.var_floor)
        return lam, A

    def update_EVI(self, X: np.ndarray, means: np.ndarray, resp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """EVI: diagonal with equal volume (lambda shared) and variable shape per component with det(A_g)=1.
        Return (lambda_scalar array([λ]), A_g matrix (G, d)). Σ_g = λ A_g.
        """
        n, d = X.shape
        G = means.shape[0]
        Nk = resp.sum(axis=0) + 1e-16
        Sg = np.zeros((G, d))
        for g in range(G):
            diff = X - means[g]
            Sg[g] = np.sum(resp[:, g][:, None] * diff * diff, axis=0) / Nk[g]
        Ag = np.zeros((G, d))
        for g in range(G):
            Ag[g] = _safe_det1_diagonal(_floor_diag(Sg[g], self.var_floor), self.var_floor)
        lam_vals = []
        for g in range(G):
            lam_vals.append(np.mean(Sg[g] / Ag[g]))
        lam = float(np.mean(lam_vals))
        lam = max(lam, self.var_floor)
        return np.array([lam]), Ag
