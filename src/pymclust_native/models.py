from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import numpy as np
from scipy.special import logsumexp

# Local reimplementation to avoid circular import with em.py
# Keep in sync with em.py implementations

def _log_gaussian_full(X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    d = X.shape[1]
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        cov = cov + 1e-6 * np.eye(d)
        L = np.linalg.cholesky(cov)
    diff = X - mean
    solve = np.linalg.solve(L, diff.T)
    maha = np.sum(solve**2, axis=0)
    log_det = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * (d * np.log(2 * np.pi) + log_det + maha)


def _log_gaussian_diag(X: np.ndarray, mean: np.ndarray, diag_var: np.ndarray) -> np.ndarray:
    d = X.shape[1]
    diff = X - mean
    inv = 1.0 / diag_var
    maha = np.sum(diff * diff * inv, axis=1)
    log_det = np.sum(np.log(diag_var))
    return -0.5 * (d * np.log(2 * np.pi) + log_det + maha)


def _log_gaussian_spherical(X: np.ndarray, mean: np.ndarray, var_scalar: float) -> np.ndarray:
    d = X.shape[1]
    diff = X - mean
    maha = np.sum(diff * diff, axis=1) / var_scalar
    log_det = d * np.log(var_scalar)
    return -0.5 * (d * np.log(2 * np.pi) + log_det + maha)


class ModelName(str, Enum):
    # Subset for initial implementation; expand later
    EII = "EII"  # spherical, equal volume
    VII = "VII"  # spherical, variable volume
    EEI = "EEI"  # diagonal, equal volume and shape
    VEI = "VEI"  # diagonal, variable volume, equal shape
    EVI = "EVI"  # diagonal, equal volume, variable shape
    VVI = "VVI"  # diagonal, variable volume and shape
    EEE = "EEE"  # general, equal volume, shape, orientation
    VVV = "VVV"  # general, variable all
    VEE = "VEE"  # variable volume, equal shape, equal orientation
    EVE = "EVE"  # equal volume, variable shape, equal orientation
    VVE = "VVE"  # variable volume, variable shape, equal orientation
    EVV = "EVV"  # equal volume, variable shape, variable orientation
    EEV = "EEV"  # equal volume, equal shape, variable orientation
    VEV = "VEV"  # variable volume, equal shape, variable orientation


@dataclass
class GMMResult:
    model_name: ModelName
    means: np.ndarray           # shape (G, d)
    covariances: np.ndarray     # representation depends on model
    weights: np.ndarray         # shape (G,)
    loglik: float
    bic: float | None = None
    n_iter: int | None = None
    converged: bool | None = None

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Posterior responsibilities given fitted parameters.
        Uses model_name and covariances representation to compute per-component log-densities.
        """
        X = np.asarray(X, dtype=float)
        G = self.weights.shape[0]
        d = X.shape[1]
        if self.model_name in (ModelName.VEE, ModelName.EVE, ModelName.VVE):
            # Accelerated path: rotate to shared D space and score with diagonal variances
            if self.model_name == ModelName.VEE:
                lam_g, A, D = self.covariances
                diag_all = lam_g[:, None] * A[None, :]
            elif self.model_name == ModelName.EVE:
                lam, A_g, D = self.covariances
                diag_all = float(lam[0]) * A_g
            else:  # VVE
                lam_g, A_g, D = self.covariances
                diag_all = lam_g[:, None] * A_g
            X_rot = X @ D
            means_rot = self.means @ D
            diff = X_rot[:, None, :] - means_rot[None, :, :]
            inv = 1.0 / diag_all[None, :, :]
            maha = np.sum(diff * diff * inv, axis=2)
            log_det = np.sum(np.log(diag_all), axis=1)[None, :]
            log_prob = -0.5 * (d * np.log(2 * np.pi) + log_det + maha) + np.log(self.weights)[None, :]
        elif self.model_name in (ModelName.VVV, ModelName.EVV, ModelName.EEV, ModelName.VEV):
            # Accelerated path: cache Cholesky per component and compute batched Mahalanobis
            # Build full covs per component
            if self.model_name == ModelName.VVV:
                covs = self.covariances
            elif self.model_name == ModelName.EVV:
                lam, A_g, D_g = self.covariances
                covs = np.array([float(lam[0]) * (D_g[g] @ np.diag(A_g[g]) @ D_g[g].T) for g in range(G)])
            elif self.model_name == ModelName.EEV:
                lam, A, D_g = self.covariances
                covs = np.array([float(lam[0]) * (D_g[g] @ np.diag(A) @ D_g[g].T) for g in range(G)])
            else:  # VEV
                lam_g, A, D_g = self.covariances
                covs = np.array([lam_g[g] * (D_g[g] @ np.diag(A) @ D_g[g].T) for g in range(G)])
            L_list = []
            logdet_vec = np.zeros(G)
            for g in range(G):
                Cg = covs[g]
                try:
                    L = np.linalg.cholesky(Cg)
                except np.linalg.LinAlgError:
                    Cg = Cg + 1e-6 * np.eye(d)
                    L = np.linalg.cholesky(Cg)
                L_list.append(L)
                logdet_vec[g] = 2.0 * np.sum(np.log(np.diag(L)))
            diff = X[:, None, :] - self.means[None, :, :]
            maha = np.zeros((X.shape[0], G))
            for g in range(G):
                y = np.linalg.solve(L_list[g], diff[:, g, :].T)
                maha[:, g] = np.sum(y * y, axis=0)
            log_prob = -0.5 * (d * np.log(2 * np.pi) + logdet_vec[None, :] + maha) + np.log(self.weights)[None, :]
        elif self.model_name == ModelName.VVV:
            log_prob = np.stack([
                _log_gaussian_full(X, self.means[g], self.covariances[g]) + np.log(self.weights[g])
                for g in range(G)
            ], axis=1)
        elif self.model_name == ModelName.EII:
            lam = float(self.covariances[0]) if np.ndim(self.covariances) == 1 else float(self.covariances)
            log_prob = np.stack([
                _log_gaussian_spherical(X, self.means[g], lam) + np.log(self.weights[g])
                for g in range(G)
            ], axis=1)
        elif self.model_name == ModelName.VII:
            log_prob = np.stack([
                _log_gaussian_spherical(X, self.means[g], float(self.covariances[g])) + np.log(self.weights[g])
                for g in range(G)
            ], axis=1)
        elif self.model_name == ModelName.EEI:
            lam, A = self.covariances
            diag = float(lam[0]) * A
            log_prob = np.stack([
                _log_gaussian_diag(X, self.means[g], diag) + np.log(self.weights[g])
                for g in range(G)
            ], axis=1)
        elif self.model_name == ModelName.VVI:
            log_prob = np.stack([
                _log_gaussian_diag(X, self.means[g], self.covariances[g]) + np.log(self.weights[g])
                for g in range(G)
            ], axis=1)
        elif self.model_name == ModelName.VEI:
            lam, A = self.covariances
            diag_all = lam[:, None] * A[None, :]
            log_prob = np.stack([
                _log_gaussian_diag(X, self.means[g], diag_all[g]) + np.log(self.weights[g])
                for g in range(G)
            ], axis=1)
        elif self.model_name == ModelName.EVI:
            lam, Ag = self.covariances
            log_prob = np.stack([
                _log_gaussian_diag(X, self.means[g], float(lam[0]) * Ag[g]) + np.log(self.weights[g])
                for g in range(G)
            ], axis=1)
        elif self.model_name == ModelName.EEE:
            log_prob = np.stack([
                _log_gaussian_full(X, self.means[g], self.covariances) + np.log(self.weights[g])
                for g in range(G)
            ], axis=1)
        elif self.model_name in (ModelName.VEE, ModelName.EVE, ModelName.VVE, ModelName.EVV, ModelName.EEV, ModelName.VEV):
            # Fallback for decomposition families (used for EE V/VEV class only here since others handled above)
            if self.model_name in (ModelName.VEE, ModelName.EVE, ModelName.VVE):
                # already handled above
                pass
            else:
                if self.model_name == ModelName.EVV:
                    lam, A_g, D_g = self.covariances
                    for_covs = np.array([float(lam[0]) * (D_g[g] @ (np.diag(A_g[g])) @ D_g[g].T) for g in range(G)])
                elif self.model_name == ModelName.EEV:
                    lam, A, D_g = self.covariances
                    for_covs = np.array([float(lam[0]) * (D_g[g] @ (np.diag(A)) @ D_g[g].T) for g in range(G)])
                else:  # VEV
                    lam_g, A, D_g = self.covariances
                    for_covs = np.array([lam_g[g] * (D_g[g] @ (np.diag(A)) @ D_g[g].T) for g in range(G)])
                log_prob = np.stack([
                    _log_gaussian_full(X, self.means[g], for_covs[g]) + np.log(self.weights[g])
                    for g in range(G)
                ], axis=1)
        else:
            log_prob = np.stack([
                _log_gaussian_full(X, self.means[g], self.covariances[g]) + np.log(self.weights[g])
                for g in range(G)
            ], axis=1)
        log_norm = logsumexp(log_prob, axis=1)
        return np.exp(log_prob - log_norm[:, None])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Hard classification from posterior responsibilities.
        """
        resp = self.predict_proba(X)
        return resp.argmax(axis=1)
