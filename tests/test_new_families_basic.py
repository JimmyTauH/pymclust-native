import numpy as np
from pymclust_native.api import fit_mclust
from pymclust_native.models import ModelName


def _two_rotated_gaussians(rng, n1=120, n2=130, theta=0.5):
    R = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    D1 = np.diag([1.0, 0.3])
    D2 = np.diag([0.8, 0.2])
    cov1 = R @ D1 @ R.T
    cov2 = R @ D2 @ R.T
    X1 = rng.multivariate_normal([0.0, 0.0], cov1, size=n1)
    X2 = rng.multivariate_normal([3.0, 3.0], cov2, size=n2)
    return np.vstack([X1, X2])


def _two_diff_orientations(rng, n1=120, n2=130, t1=0.2, t2=-0.7):
    R1 = np.array([[np.cos(t1), -np.sin(t1)],[np.sin(t1), np.cos(t1)]])
    R2 = np.array([[np.cos(t2), -np.sin(t2)],[np.sin(t2), np.cos(t2)]])
    D1 = np.diag([1.0, 0.3])
    D2 = np.diag([0.8, 0.2])
    cov1 = R1 @ D1 @ R1.T
    cov2 = R2 @ D2 @ R2.T
    X1 = rng.multivariate_normal([0.0, 0.0], cov1, size=n1)
    X2 = rng.multivariate_normal([3.0, 3.0], cov2, size=n2)
    return np.vstack([X1, X2])


def _check_posterior(P):
    assert P.ndim == 2 and P.shape[1] == 2
    row_sums = P.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6)


def test_vee_runs_and_posterior():
    rng = np.random.default_rng(0)
    X = _two_rotated_gaussians(rng)
    res = fit_mclust(X, G_list=[2], models=[ModelName.VEE], random_state=1, max_iter=300)
    assert res.converged is True
    P = res.predict_proba(X[:10])
    _check_posterior(P)


def test_eve_runs_and_posterior():
    rng = np.random.default_rng(1)
    X = _two_rotated_gaussians(rng)
    res = fit_mclust(X, G_list=[2], models=[ModelName.EVE], random_state=2, max_iter=300)
    assert res.converged is True
    P = res.predict_proba(X[:10])
    _check_posterior(P)


def test_vve_runs_and_posterior():
    rng = np.random.default_rng(2)
    X = _two_rotated_gaussians(rng)
    res = fit_mclust(X, G_list=[2], models=[ModelName.VVE], random_state=3, max_iter=300)
    assert res.converged is True
    P = res.predict_proba(X[:10])
    _check_posterior(P)


def test_evv_runs_and_posterior():
    rng = np.random.default_rng(3)
    X = _two_diff_orientations(rng)
    res = fit_mclust(X, G_list=[2], models=[ModelName.EVV], random_state=4, max_iter=300)
    assert res.converged is True
    P = res.predict_proba(X[:10])
    _check_posterior(P)


def test_eev_runs_and_posterior():
    rng = np.random.default_rng(4)
    X = _two_diff_orientations(rng)
    res = fit_mclust(X, G_list=[2], models=[ModelName.EEV], random_state=5, max_iter=300)
    assert res.converged is True
    P = res.predict_proba(X[:10])
    _check_posterior(P)


def test_vev_runs_and_posterior():
    rng = np.random.default_rng(5)
    X = _two_diff_orientations(rng)
    res = fit_mclust(X, G_list=[2], models=[ModelName.VEV], random_state=6, max_iter=300)
    assert res.converged is True
    P = res.predict_proba(X[:10])
    _check_posterior(P)
