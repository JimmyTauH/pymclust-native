from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np
from numpy.typing import ArrayLike
from scipy.special import logsumexp

from .models import GMMResult, ModelName
from .covariance import CovarianceModel, FULL_JITTER


@dataclass
class EMOptions:
    tol: float = 1e-5
    max_iter: int = 1000
    init_method: str = "kmeans"  # or "random"
    n_init: int = 5
    random_state: Optional[int] = None
    var_floor: float = 1e-8
    full_jitter: float = 1e-6


def _log_gaussian_full(X: np.ndarray, mean: np.ndarray, cov: np.ndarray, jitter: float = FULL_JITTER) -> np.ndarray:
    d = X.shape[1]
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        cov = cov + jitter * np.eye(d)
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


class GaussianMixtureEM:
    def __init__(self, G: int, model_name: ModelName = ModelName.VVV, options: Optional[EMOptions] = None):
        self.G = G
        self.model_name = model_name
        self.options = options or EMOptions()
        self.cov_model = CovarianceModel(model_name, var_floor=self.options.var_floor)
        # Cache for E-step acceleration (refreshed after M-step for shared-D families)
        self._cache: Dict[str, Any] | None = None

    def _kmeans_init(self, X: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        n, d = X.shape
        # k-means++ like seeding (simplified): random first center, then farthest points
        centers = np.empty((self.G, d))
        idx0 = rng.integers(0, n)
        centers[0] = X[idx0]
        dists = np.sum((X - centers[0])**2, axis=1)
        for g in range(1, self.G):
            idx = int(np.argmax(dists))
            centers[g] = X[idx]
            dists = np.minimum(dists, np.sum((X - centers[g])**2, axis=1))
        # Lloyd iterations
        prev = centers.copy()
        for _ in range(100):
            labels = np.argmin(((X[:, None, :] - centers[None, :, :])**2).sum(axis=2), axis=1)
            for g in range(self.G):
                mask = labels == g
                if np.any(mask):
                    centers[g] = X[mask].mean(axis=0)
            if np.linalg.norm(centers - prev) < 1e-4:
                break
            prev = centers.copy()
        # responsibilities one-hot
        labels = np.argmin(((X[:, None, :] - centers[None, :, :])**2).sum(axis=2), axis=1)
        resp0 = np.zeros((n, self.G))
        resp0[np.arange(n), labels] = 1.0
        return centers, resp0

    def _init_params(self, X: np.ndarray, rng: np.random.Generator):
        n, d = X.shape
        means, covs = self.cov_model.init_params(self.G, d)
        if self.options.init_method == "kmeans":
            means_km, resp0 = self._kmeans_init(X, rng)
            means = means_km
            weights = resp0.mean(axis=0)
            weights = np.where(weights > 0, weights, 1.0 / self.G)
        else:
            idx = rng.choice(n, size=self.G, replace=False)
            means = X[idx]
            weights = np.full(self.G, 1.0 / self.G)
        # Expand cov representation to computation form
        if self.model_name == ModelName.EEE:
            covs_full = np.array([np.eye(d) for _ in range(self.G)])
            cov_repr = np.eye(d)
        elif self.model_name in (ModelName.EII, ModelName.VII):
            if self.model_name == ModelName.EII:
                covs_full = np.array([np.eye(d) for _ in range(self.G)])
                cov_repr = np.array([1.0])  # lambda
            else:
                covs_full = np.array([np.eye(d) for _ in range(self.G)])
                cov_repr = np.ones(self.G)  # lambda_g
        elif self.model_name in (ModelName.EEI, ModelName.VVI, ModelName.VEI, ModelName.EVI, ModelName.VEE, ModelName.EVE, ModelName.VVE, ModelName.EVV, ModelName.EEV, ModelName.VEV):
            if self.model_name == ModelName.EEI:
                covs_full = np.array([np.eye(d) for _ in range(self.G)])
                # Initialize as a valid tuple (lambda, A) to avoid first E-step issues
                cov_repr = (np.array([1.0]), np.ones(d))  # (lambda, A with det=1 placeholder)
            elif self.model_name == ModelName.VEI:
                covs_full = np.array([np.eye(d) for _ in range(self.G)])
                cov_repr = (np.ones(self.G), np.ones(d))  # (lambda_g, A)
            elif self.model_name == ModelName.EVI:
                covs_full = np.array([np.eye(d) for _ in range(self.G)])
                cov_repr = (np.ones(1), np.ones((self.G, d)))  # (lambda, A_g)
            elif self.model_name == ModelName.VEE:
                covs_full = np.array([np.eye(d) for _ in range(self.G)])
                cov_repr = (np.ones(self.G), np.ones(d), np.eye(d))  # (lambda_g, A, D)
            elif self.model_name == ModelName.EVE:
                covs_full = np.array([np.eye(d) for _ in range(self.G)])
                cov_repr = (np.ones(1), np.ones((self.G, d)), np.eye(d))  # (lambda, A_g, D)
            elif self.model_name == ModelName.VVE:
                covs_full = np.array([np.eye(d) for _ in range(self.G)])
                cov_repr = (np.ones(self.G), np.ones((self.G, d)), np.eye(d))  # (lambda_g, A_g, D)
            elif self.model_name == ModelName.EVV:
                covs_full = np.array([np.eye(d) for _ in range(self.G)])
                cov_repr = (np.ones(1), np.ones((self.G, d)), np.array([np.eye(d) for _ in range(self.G)]))  # (lambda, A_g, D_g)
            elif self.model_name == ModelName.EEV:
                covs_full = np.array([np.eye(d) for _ in range(self.G)])
                cov_repr = (np.ones(1), np.ones(d), np.array([np.eye(d) for _ in range(self.G)]))  # (lambda, A, D_g)
            elif self.model_name == ModelName.VEV:
                covs_full = np.array([np.eye(d) for _ in range(self.G)])
                cov_repr = (np.ones(self.G), np.ones(d), np.array([np.eye(d) for _ in range(self.G)]))  # (lambda_g, A, D_g)
            else:
                covs_full = np.array([np.eye(d) for _ in range(self.G)])
                cov_repr = np.ones((self.G, d))
        else:
            covs_full = np.array([np.eye(d) for _ in range(self.G)])
            cov_repr = covs if covs.ndim == 3 else np.array([np.eye(d) for _ in range(self.G)])
        return means, covs_full, cov_repr, weights

    def _log_prob_components(self, X: np.ndarray, means: np.ndarray, cov_repr) -> np.ndarray:
        G = means.shape[0]
        d = X.shape[1]
        # Fast path for shared-orientation families using rotation and diagonal log-density
        if self.model_name in (ModelName.VEE, ModelName.EVE, ModelName.VVE):
            # Ensure cache exists
            if self._cache is None or any(k not in self._cache for k in ("D", "means_rot", "diag_all")):
                if self.model_name == ModelName.VEE:
                    lam_g, A, D = cov_repr
                    diag_all = lam_g[:, None] * A[None, :]
                elif self.model_name == ModelName.EVE:
                    lam, A_g, D = cov_repr
                    diag_all = float(lam[0]) * A_g
                else:  # VVE
                    lam_g, A_g, D = cov_repr
                    diag_all = lam_g[:, None] * A_g
                means_rot = (means @ D)
                self._cache = {"D": D, "means_rot": means_rot, "diag_all": diag_all}
            D = self._cache["D"]
            means_rot = self._cache["means_rot"]
            diag_all = self._cache["diag_all"]
            X_rot = X @ D
            diff = X_rot[:, None, :] - means_rot[None, :, :]
            inv = 1.0 / diag_all[None, :, :]
            maha = np.sum(diff * diff * inv, axis=2)  # (n,G)
            log_det = np.sum(np.log(diag_all), axis=1)[None, :]
            log_prob = -0.5 * (d * np.log(2 * np.pi) + log_det + maha)
            return log_prob
        # Fast(er) path for full-covariance families using cached Cholesky and log-determinants
        if self.model_name in (ModelName.VVV, ModelName.EVV, ModelName.EEV, ModelName.VEV):
            # Build cache if missing
            if self._cache is None or any(k not in self._cache for k in ("L_list", "logdet_vec")):
                L_list = []
                logdet_vec = np.zeros(G)
                # Build full covariances per component from representation
                if self.model_name == ModelName.VVV:
                    covs = cov_repr  # shape (G,d,d)
                elif self.model_name == ModelName.EVV:
                    lam, A_g, D_g = cov_repr
                    covs = np.array([float(lam[0]) * (D_g[g] @ np.diag(A_g[g]) @ D_g[g].T) for g in range(G)])
                elif self.model_name == ModelName.EEV:
                    lam, A, D_g = cov_repr
                    covs = np.array([float(lam[0]) * (D_g[g] @ np.diag(A) @ D_g[g].T) for g in range(G)])
                else:  # VEV
                    lam_g, A, D_g = cov_repr
                    covs = np.array([lam_g[g] * (D_g[g] @ np.diag(A) @ D_g[g].T) for g in range(G)])
                for g in range(G):
                    Cg = covs[g]
                    try:
                        L = np.linalg.cholesky(Cg)
                    except np.linalg.LinAlgError:
                        Cg = Cg + self.options.full_jitter * np.eye(d)
                        L = np.linalg.cholesky(Cg)
                    L_list.append(L)
                    logdet_vec[g] = 2.0 * np.sum(np.log(np.diag(L)))
                self._cache = {"L_list": L_list, "logdet_vec": logdet_vec}
            L_list = self._cache["L_list"]
            logdet_vec = self._cache["logdet_vec"]
            # Vectorized diff for all components
            diff = X[:, None, :] - means[None, :, :]
            # Compute maha per component with G solves
            maha = np.zeros((X.shape[0], G))
            for g in range(G):
                L = L_list[g]
                # solve L y = diff^T -> y has shape (d,n)
                y = np.linalg.solve(L, diff[:, g, :].T)
                maha[:, g] = np.sum(y * y, axis=0)
            log_prob = -0.5 * (d * np.log(2 * np.pi) + logdet_vec[None, :] + maha)
            return log_prob
        if self.model_name == ModelName.VVV:
            return np.stack([
                _log_gaussian_full(X, means[g], cov_repr[g], self.options.full_jitter) for g in range(G)
            ], axis=1)
        elif self.model_name == ModelName.EEE:
            return np.stack([
                _log_gaussian_full(X, means[g], cov_repr, self.options.full_jitter) for g in range(G)
            ], axis=1)
        elif self.model_name == ModelName.EII:
            lam = float(cov_repr[0])
            return np.stack([
                _log_gaussian_spherical(X, means[g], lam) for g in range(G)
            ], axis=1)
        elif self.model_name == ModelName.VII:
            return np.stack([
                _log_gaussian_spherical(X, means[g], float(cov_repr[g])) for g in range(G)
            ], axis=1)
        elif self.model_name == ModelName.EEI:
            # Support both proper tuple (lam, A) and legacy/placeholder array forms
            if isinstance(cov_repr, tuple) and len(cov_repr) == 2:
                lam_arr, A = cov_repr
                lam = float(np.asarray(lam_arr).reshape(-1)[0])
                A = np.asarray(A).reshape(-1)
            else:
                # Placeholder encountered (e.g., shape (1, d)); use lambda=1.0 and interpret as A
                arr = np.asarray(cov_repr)
                if arr.ndim == 2 and arr.shape[0] == 1:
                    A = arr.reshape(-1)
                    lam = 1.0
                else:
                    # Fallback: treat as diagonal vector with lambda=1
                    A = arr.reshape(-1)
                    lam = 1.0
            diag = lam * A
            return np.stack([
                _log_gaussian_diag(X, means[g], diag) for g in range(G)
            ], axis=1)
        elif self.model_name == ModelName.VVI:
            return np.stack([
                _log_gaussian_diag(X, means[g], cov_repr[g]) for g in range(G)
            ], axis=1)
        elif self.model_name == ModelName.VEI:
            lam, A = cov_repr
            diag = lam[:, None] * A[None, :]
            return np.stack([
                _log_gaussian_diag(X, means[g], diag[g]) for g in range(G)
            ], axis=1)
        elif self.model_name == ModelName.EVI:
            lam, Ag = cov_repr
            return np.stack([
                _log_gaussian_diag(X, means[g], float(lam[0]) * Ag[g]) for g in range(G)
            ], axis=1)
        elif self.model_name in (ModelName.VEE, ModelName.EVE, ModelName.VVE, ModelName.EVV, ModelName.EEV, ModelName.VEV):
            # assemble full covariances from decomposition
            if self.model_name == ModelName.VEE:
                lam_g, A, D = cov_repr
                covs_full = np.array([lam_g[g] * (D @ np.diag(A) @ D.T) for g in range(G)])
            elif self.model_name == ModelName.EVE:
                lam, A_g, D = cov_repr
                covs_full = np.array([float(lam[0]) * (D @ np.diag(A_g[g]) @ D.T) for g in range(G)])
            elif self.model_name == ModelName.VVE:
                lam_g, A_g, D = cov_repr
                covs_full = np.array([lam_g[g] * (D @ np.diag(A_g[g]) @ D.T) for g in range(G)])
            elif self.model_name == ModelName.EVV:
                lam, A_g, D_g = cov_repr
                covs_full = np.array([float(lam[0]) * (D_g[g] @ np.diag(A_g[g]) @ D_g[g].T) for g in range(G)])
            elif self.model_name == ModelName.EEV:
                lam, A, D_g = cov_repr
                covs_full = np.array([float(lam[0]) * (D_g[g] @ np.diag(A) @ D_g[g].T) for g in range(G)])
            else:  # VEV
                lam_g, A, D_g = cov_repr
                covs_full = np.array([lam_g[g] * (D_g[g] @ np.diag(A) @ D_g[g].T) for g in range(G)])
            return np.stack([
                _log_gaussian_full(X, means[g], covs_full[g], self.options.full_jitter) for g in range(G)
            ], axis=1)
        else:
            # fallback full
            return np.stack([
                _log_gaussian_full(X, means[g], cov_repr[g]) for g in range(G)
            ], axis=1)

    def fit(self, X: ArrayLike) -> GMMResult:
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        rng_master = np.random.default_rng(self.options.random_state)

        best_res: Optional[GMMResult] = None
        best_ll = -np.inf

        for trial in range(max(1, self.options.n_init)):
            rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
            means, covs_full, cov_repr, weights = self._init_params(X, rng)

            prev_ll = -np.inf
            converged = False
            for it in range(1, self.options.max_iter + 1):
                # E-step
                log_prob = self._log_prob_components(X, means, cov_repr) + np.log(weights)[None, :]
                log_norm = logsumexp(log_prob, axis=1)
                ll = float(np.sum(log_norm))
                resp = np.exp(log_prob - log_norm[:, None])

                # M-step
                Nk = resp.sum(axis=0) + 1e-16
                weights = Nk / n
                means = (resp.T @ X) / Nk[:, None]

                if self.model_name == ModelName.VVV:
                    covs_full = np.zeros((self.G, d, d))
                    for g in range(self.G):
                        diff = X - means[g]
                        covs_full[g] = (diff * resp[:, [g]]).T @ diff / Nk[g]
                        covs_full[g].flat[:: d + 1] += self.options.full_jitter
                    cov_repr = covs_full
                elif self.model_name == ModelName.EEE:
                    Sigma = np.zeros((d, d))
                    for g in range(self.G):
                        diff = X - means[g]
                        Sigma += (diff * resp[:, [g]]).T @ diff
                    Sigma = Sigma / np.sum(Nk)
                    Sigma.flat[:: d + 1] += self.options.full_jitter
                    cov_repr = Sigma
                elif self.model_name == ModelName.EII:
                    lam = self.cov_model.update_EII(X, means, resp)
                    cov_repr = lam
                elif self.model_name == ModelName.VII:
                    lam = self.cov_model.update_VII(X, means, resp)
                    cov_repr = lam
                elif self.model_name == ModelName.EEI:
                    lam, A = self.cov_model.update_EEI(X, means, resp)
                    lam = np.asarray(lam).reshape(-1)[:1]
                    A = np.asarray(A).reshape(-1)
                    assert lam.size == 1, f"EEI lambda shape invalid: {lam.shape}"
                    assert A.size == d, f"EEI A shape invalid: {A.shape}, expected ({d},)"
                    cov_repr = (lam, A)
                elif self.model_name == ModelName.VVI:
                    diag_vars = self.cov_model.update_VVI(X, means, resp)
                    cov_repr = diag_vars
                elif self.model_name == ModelName.VEI:
                    lam, A = self.cov_model.update_VEI(X, means, resp)
                    cov_repr = (lam, A)
                elif self.model_name == ModelName.EVI:
                    lam, Ag = self.cov_model.update_EVI(X, means, resp)
                    cov_repr = (lam, Ag)

                elif self.model_name in (ModelName.VEE, ModelName.EVE, ModelName.VVE, ModelName.EVV, ModelName.EEV, ModelName.VEV):
                    # Minimal working M-step: estimate full covariances per component, then project to decomposition form.
                    # 1) Compute class covariances S_g
                    Sg = np.zeros((self.G, d, d))
                    for g in range(self.G):
                        diff = X - means[g]
                        Sg[g] = (diff * resp[:, [g]]).T @ diff / Nk[g]
                        Sg[g].flat[:: d + 1] += self.options.full_jitter
                    # 2) Estimate orientation D (shared vs per-component)
                    if self.model_name in (ModelName.VEE, ModelName.EVE, ModelName.VVE):
                        # shared D from pooled covariance (vectorized, avoid deprecated np.sum on generator)
                        Nk_sum = float(np.sum(Nk))
                        Sp = (Nk[:, None, None] * Sg).sum(axis=0) / Nk_sum
                        eigvals, D = np.linalg.eigh(Sp)
                        # enforce ordering
                        idx = np.argsort(eigvals)[::-1]
                        D = D[:, idx]
                        # 3) Rotate Sg
                        Sg_prime = np.array([D.T @ Sg[g] @ D for g in range(self.G)])
                        diag_Sg = np.array([np.diag(Sg_prime[g]) for g in range(self.G)])
                        if self.model_name == ModelName.VEE:
                            # shared A, per-component lambda_g
                            A_raw = np.average(diag_Sg, axis=0, weights=Nk)
                            A = A_raw / np.exp(np.mean(np.log(np.clip(A_raw, self.options.var_floor, None))))
                            lam_g = np.array([np.mean(diag_Sg[g] / A) for g in range(self.G)])
                            lam_g = np.clip(lam_g, self.options.var_floor, None)
                            cov_repr = (lam_g, A, D)
                        elif self.model_name == ModelName.EVE:
                            # shared lambda, per-component A_g
                            A_g = np.zeros((self.G, d))
                            for g in range(self.G):
                                a = np.clip(diag_Sg[g], self.options.var_floor, None)
                                A_g[g] = a / np.exp(np.mean(np.log(a)))
                            lam = np.array([np.mean([np.mean(diag_Sg[g] / A_g[g]) for g in range(self.G)])])
                            lam[0] = max(lam[0], self.options.var_floor)
                            cov_repr = (lam, A_g, D)
                        else:  # VVE
                            A_g = np.zeros((self.G, d))
                            lam_g = np.zeros(self.G)
                            for g in range(self.G):
                                a = np.clip(diag_Sg[g], self.options.var_floor, None)
                                A_g[g] = a / np.exp(np.mean(np.log(a)))
                                lam_g[g] = np.mean(diag_Sg[g] / A_g[g])
                            lam_g = np.clip(lam_g, self.options.var_floor, None)
                            cov_repr = (lam_g, A_g, D)
                        # update cache for shared-D families for fast E-step
                        if self.model_name == ModelName.VEE:
                            diag_all = lam_g[:, None] * A[None, :]
                        elif self.model_name == ModelName.EVE:
                            diag_all = float(lam[0]) * A_g
                        else:
                            diag_all = lam_g[:, None] * A_g
                        means_rot = (means @ D)
                        self._cache = {"D": D, "means_rot": means_rot, "diag_all": diag_all}
                        # clear any full-cov cache since representation changed
                        # and shared-D path will be used
                    else:
                        # EVV/EEV/VEV: per-component D_g
                        D_g = np.zeros((self.G, d, d))
                        diag_Sg_rot = np.zeros((self.G, d))
                        for g in range(self.G):
                            eigvals, Dg = np.linalg.eigh(Sg[g])
                            idx = np.argsort(eigvals)[::-1]
                            Dg = Dg[:, idx]
                            D_g[g] = Dg
                            a = np.diag(Dg.T @ Sg[g] @ Dg)
                            a = np.clip(a, self.options.var_floor, None)
                            diag_Sg_rot[g] = a
                        if self.model_name == ModelName.EVV:
                            A_g = np.zeros((self.G, d))
                            for g in range(self.G):
                                a = diag_Sg_rot[g]
                                A_g[g] = a / np.exp(np.mean(np.log(a)))
                            lam = np.array([np.mean([np.mean(diag_Sg_rot[g] / A_g[g]) for g in range(self.G)])])
                            lam[0] = max(lam[0], self.options.var_floor)
                            cov_repr = (lam, A_g, D_g)
                        elif self.model_name == ModelName.EEV:
                            # shared A, per-component D_g, shared lambda
                            A_raw = np.average(diag_Sg_rot, axis=0, weights=Nk)
                            A = np.clip(A_raw, self.options.var_floor, None)
                            A = A / np.exp(np.mean(np.log(A)))
                            lam = np.array([np.mean([np.mean(diag_Sg_rot[g] / A) for g in range(self.G)])])
                            lam[0] = max(lam[0], self.options.var_floor)
                            cov_repr = (lam, A, D_g)
                        else:
                            # VEV: shared A, per-component D_g, variable lambda_g
                            A_raw = np.average(diag_Sg_rot, axis=0, weights=Nk)
                            A = np.clip(A_raw, self.options.var_floor, None)
                            A = A / np.exp(np.mean(np.log(A)))
                            lam_g = np.array([np.mean(diag_Sg_rot[g] / A) for g in range(self.G)])
                            lam_g = np.clip(lam_g, self.options.var_floor, None)
                            cov_repr = (lam_g, A, D_g)
                else:
                    covs_full = np.zeros((self.G, d, d))
                    for g in range(self.G):
                        diff = X - means[g]
                        covs_full[g] = (diff * resp[:, [g]]).T @ diff / Nk[g]
                        covs_full[g].flat[:: d + 1] += 1e-6
                    cov_repr = covs_full
                # After M-step, refresh caches appropriately
                if self.model_name in (ModelName.VEE, ModelName.EVE, ModelName.VVE):
                    # shared-D cache set above in that branch
                    pass
                elif self.model_name in (ModelName.VVV, ModelName.EVV, ModelName.EEV, ModelName.VEV):
                    # build full-cov cache for E-step
                    L_list = []
                    logdet_vec = np.zeros(self.G)
                    if self.model_name == ModelName.VVV:
                        covs = cov_repr
                    elif self.model_name == ModelName.EVV:
                        lam, A_g, D_g = cov_repr
                        covs = np.array([float(lam[0]) * (D_g[g] @ np.diag(A_g[g]) @ D_g[g].T) for g in range(self.G)])
                    elif self.model_name == ModelName.EEV:
                        lam, A, D_g = cov_repr
                        covs = np.array([float(lam[0]) * (D_g[g] @ np.diag(A) @ D_g[g].T) for g in range(self.G)])
                    else:  # VEV
                        lam_g, A, D_g = cov_repr
                        covs = np.array([lam_g[g] * (D_g[g] @ np.diag(A) @ D_g[g].T) for g in range(self.G)])
                    for g in range(self.G):
                        Cg = covs[g]
                        try:
                            L = np.linalg.cholesky(Cg)
                        except np.linalg.LinAlgError:
                            Cg = Cg + self.options.full_jitter * np.eye(d)
                            L = np.linalg.cholesky(Cg)
                        L_list.append(L)
                        logdet_vec[g] = 2.0 * np.sum(np.log(np.diag(L)))
                    self._cache = {"L_list": L_list, "logdet_vec": logdet_vec}
                else:
                    # other families don't need heavy cache
                    self._cache = None

                if ll - prev_ll < self.options.tol:
                    converged = True
                    break
                prev_ll = ll

            # Normalize covariance representation shapes before packaging result
            covariances_out = cov_repr if self.model_name != ModelName.VVV else covs_full
            if self.model_name == ModelName.EEI:
                try:
                    lam, A = covariances_out
                    lam = np.asarray(lam).reshape(-1)[:1]
                    A = np.asarray(A).reshape(-1)
                    covariances_out = (lam, A)
                except Exception:
                    pass

            res = GMMResult(
                model_name=self.model_name,
                means=means,
                covariances=covariances_out,
                weights=weights,
                loglik=ll if converged else prev_ll,
                bic=None,
                n_iter=it if converged else self.options.max_iter,
                converged=converged,
            )
            if res.loglik > best_ll:
                best_ll = res.loglik
                best_res = res

        assert best_res is not None
        return best_res
