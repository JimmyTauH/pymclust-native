from __future__ import annotations

from typing import Iterable
import numpy as np
from scipy.special import logsumexp

from .models import GMMResult, ModelName
from .em import GaussianMixtureEM, EMOptions, _log_gaussian_full, _log_gaussian_diag, _log_gaussian_spherical


def _bic(loglik: float, n_params: int, n_samples: int) -> float:
    # mclust uses BIC = 2*loglik - m*log(n), where m is #free params
    return 2.0 * loglik - n_params * np.log(n_samples)


def _num_params(model: ModelName, G: int, d: int) -> int:
    # weights (G-1) + means (G*d) + covariance params
    if model == ModelName.VVV:
        cov_params = int(G * d * (d + 1) / 2)
    elif model == ModelName.EII:
        cov_params = 1
    elif model == ModelName.VII:
        cov_params = G
    elif model == ModelName.EEI:
        cov_params = 1 + (d - 1)  # lambda + diag shape with det=1
    elif model == ModelName.VVI:
        cov_params = G * d
    elif model == ModelName.VEI:
        cov_params = G + (d - 1)  # lambda_g + shared shape
    elif model == ModelName.EVI:
        cov_params = 1 + G * (d - 1)  # shared lambda + per-component shape
    elif model == ModelName.EEE:
        cov_params = int(d * (d + 1) / 2)
    elif model == ModelName.VEE:
        # variable volume (G), equal shape (d-1), equal orientation (d(d-1)/2)
        cov_params = G + (d - 1) + int(d * (d - 1) / 2)
    elif model == ModelName.EVE:
        # equal volume (1), variable shape (G*(d-1)), equal orientation (d(d-1)/2)
        cov_params = 1 + G * (d - 1) + int(d * (d - 1) / 2)
    elif model == ModelName.VVE:
        # variable volume (G), variable shape (G*(d-1)), equal orientation (d(d-1)/2)
        cov_params = G + G * (d - 1) + int(d * (d - 1) / 2)
    elif model == ModelName.EVV:
        # equal volume (1), variable shape (G*(d-1)), variable orientation (G*d(d-1)/2)
        cov_params = 1 + G * (d - 1) + G * int(d * (d - 1) / 2)
    elif model == ModelName.EEV:
        # equal volume (1), equal shape (d-1), variable orientation (G*d(d-1)/2)
        cov_params = 1 + (d - 1) + G * int(d * (d - 1) / 2)
    elif model == ModelName.VEV:
        # variable volume (G), equal shape (d-1), variable orientation (G*d(d-1)/2)
        cov_params = G + (d - 1) + G * int(d * (d - 1) / 2)
    else:
        cov_params = int(G * d * (d + 1) / 2)
    return (G - 1) + (G * d) + cov_params


def fit_mclust(
    X: np.ndarray,
    G_list: Iterable[int] | None = None,
    models: Iterable[ModelName] | None = None,
    random_state: int | None = 0,
    max_iter: int = 1000,
    tol: float = 1e-5,
    init_method: str = "kmeans",
    n_init: int = 5,
    selection_criterion: str = "BIC",
    var_floor: float | None = None,
    full_jitter: float | None = None,
) -> GMMResult:
    """Fit a set of mixture models and select the best, inspired by mclust.

    Supports models: VVV, EII, VII, EEI, VVI (incremental set).
    selection_criterion: "BIC" (default) or "ICL".
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape

    if G_list is None:
        G_list = [1, 2, 3]
    if models is None:
        models = [ModelName.VVV]

    if selection_criterion.upper() not in ("BIC", "ICL"):
        raise ValueError("selection_criterion must be 'BIC' or 'ICL'")

    best: GMMResult | None = None
    best_score: float = -np.inf

    def _entropy_penalty(z: np.ndarray) -> float:
        # sum_i sum_g z_ig log z_ig, with convention 0 log 0 = 0
        z_safe = np.clip(z, 1e-16, 1.0)
        return float(np.sum(z_safe * np.log(z_safe)))

    for G in G_list:
        for model in models:
            opts = EMOptions(
                tol=tol,
                max_iter=max_iter,
                random_state=random_state,
                init_method=init_method,
                n_init=n_init,
                var_floor=(var_floor if var_floor is not None else EMOptions().var_floor),
                full_jitter=(full_jitter if full_jitter is not None else EMOptions().full_jitter),
            )
            em = GaussianMixtureEM(G=G, model_name=model, options=opts)
            res = em.fit(X)
            k = _num_params(model, G, d)
            if selection_criterion.upper() == "BIC":
                score = _bic(res.loglik, k, n)
                res.bic = score
            else:
                # ICL = BIC + 2 * sum_i sum_g z_ig log z_ig
                # compute posterior z under fitted params
                def _post(X_new: np.ndarray, means=res.means, covs=res.covariances, weights=res.weights):
                    X_new = np.asarray(X_new, dtype=float)
                    if model == ModelName.VVV:
                        log_prob = np.stack([
                            _log_gaussian_full(X_new, means[g], covs[g], opts.full_jitter) + np.log(weights[g])
                            for g in range(len(weights))
                        ], axis=1)
                    elif model == ModelName.EII:
                        lam = float(res.covariances[0]) if np.ndim(res.covariances) == 1 else float(res.covariances)
                        log_prob = np.stack([
                            _log_gaussian_spherical(X_new, means[g], lam) + np.log(weights[g])
                            for g in range(len(weights))
                        ], axis=1)
                    elif model == ModelName.VII:
                        log_prob = np.stack([
                            _log_gaussian_spherical(X_new, means[g], float(res.covariances[g])) + np.log(weights[g])
                            for g in range(len(weights))
                        ], axis=1)
                    elif model == ModelName.EEI:
                        lam, A = res.covariances
                        diag = float(lam[0]) * A
                        log_prob = np.stack([
                            _log_gaussian_diag(X_new, means[g], diag) + np.log(weights[g])
                            for g in range(len(weights))
                        ], axis=1)
                    elif model == ModelName.VVI:
                        log_prob = np.stack([
                            _log_gaussian_diag(X_new, means[g], res.covariances[g]) + np.log(weights[g])
                            for g in range(len(weights))
                        ], axis=1)
                    elif model == ModelName.VEI:
                        lam, A = res.covariances
                        diag = lam[:, None] * A[None, :]
                        log_prob = np.stack([
                            _log_gaussian_diag(X_new, means[g], diag[g]) + np.log(weights[g])
                            for g in range(len(weights))
                        ], axis=1)
                    elif model == ModelName.EVI:
                        lam, Ag = res.covariances
                        log_prob = np.stack([
                            _log_gaussian_diag(X_new, means[g], float(lam[0]) * Ag[g]) + np.log(weights[g])
                            for g in range(len(weights))
                        ], axis=1)
                    else:
                        log_prob = np.stack([
                            _log_gaussian_full(X_new, means[g], res.covariances[g], opts.full_jitter) + np.log(weights[g])
                            for g in range(len(weights))
                        ], axis=1)
                    log_norm = logsumexp(log_prob, axis=1)
                    return np.exp(log_prob - log_norm[:, None])
                z = _post(X)
                bic = _bic(res.loglik, k, n)
                score = bic + 2.0 * _entropy_penalty(z)
                res.bic = bic

            # predict_proba now implemented on GMMResult; no closure injection needed

            if score > best_score:
                best_score = score
                best = res

    assert best is not None
    return best
