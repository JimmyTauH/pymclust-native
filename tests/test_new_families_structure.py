import numpy as np
from numpy.linalg import det
from pymclust_native.api import fit_mclust
from pymclust_native.models import ModelName


def _gen_rotated_data(rng, n1=150, n2=160, theta=0.6):
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    D1 = np.diag([1.2, 0.4])
    D2 = np.diag([0.7, 0.25])
    cov1 = R @ D1 @ R.T
    cov2 = R @ D2 @ R.T
    X1 = rng.multivariate_normal([0.0, 0.0], cov1, size=n1)
    X2 = rng.multivariate_normal([3.0, 3.0], cov2, size=n2)
    return np.vstack([X1, X2])


def _gen_diff_orientation(rng, n1=140, n2=150, t1=0.2, t2=-0.8):
    R1 = np.array([[np.cos(t1), -np.sin(t1)], [np.sin(t1), np.cos(t1)]])
    R2 = np.array([[np.cos(t2), -np.sin(t2)], [np.sin(t2), np.cos(t2)]])
    D1 = np.diag([1.1, 0.35])
    D2 = np.diag([0.9, 0.22])
    cov1 = R1 @ D1 @ R1.T
    cov2 = R2 @ D2 @ R2.T
    X1 = rng.multivariate_normal([0.0, 0.0], cov1, size=n1)
    X2 = rng.multivariate_normal([3.0, 3.0], cov2, size=n2)
    return np.vstack([X1, X2])


def _check_det1(vec, tol=1e-6):
    # vec is diagonal elements with det constraint => product ~ 1.0
    prod = float(np.prod(np.clip(vec, 1e-16, None)))
    assert np.isfinite(prod)
    assert abs(np.log(prod)) < 1e-3  # allow small numerical error


def test_vee_structure_shared_D_and_A_det1():
    rng = np.random.default_rng(10)
    X = _gen_rotated_data(rng)
    res = fit_mclust(X, G_list=[2], models=[ModelName.VEE], random_state=1, max_iter=300)
    assert res.converged is True
    lam_g, A, D = res.covariances
    # shapes
    assert lam_g.shape == (2,)
    assert A.shape == (2,)
    assert D.shape == (2, 2)
    # det(A) == 1 (approximately)
    _check_det1(A)
    # D is orthonormal
    I = D.T @ D
    assert np.allclose(I, np.eye(2), atol=1e-5)


def test_eve_structure_shared_D_and_lambda_det1_Ag():
    rng = np.random.default_rng(11)
    X = _gen_rotated_data(rng)
    res = fit_mclust(X, G_list=[2], models=[ModelName.EVE], random_state=2, max_iter=300)
    assert res.converged is True
    lam, A_g, D = res.covariances
    assert lam.shape == (1,)
    assert A_g.shape == (2, 2)
    assert D.shape == (2, 2)
    for g in range(2):
        _check_det1(A_g[g])
    assert np.allclose(D.T @ D, np.eye(2), atol=1e-5)


def test_vve_structure_shared_D_det1_Ag():
    rng = np.random.default_rng(12)
    X = _gen_rotated_data(rng)
    res = fit_mclust(X, G_list=[2], models=[ModelName.VVE], random_state=3, max_iter=300)
    assert res.converged is True
    lam_g, A_g, D = res.covariances
    assert lam_g.shape == (2,)
    assert A_g.shape == (2, 2)
    assert D.shape == (2, 2)
    for g in range(2):
        _check_det1(A_g[g])
    assert np.allclose(D.T @ D, np.eye(2), atol=1e-5)


def test_evv_structure_per_component_Dg_det1_Ag():
    rng = np.random.default_rng(13)
    X = _gen_diff_orientation(rng)
    res = fit_mclust(X, G_list=[2], models=[ModelName.EVV], random_state=4, max_iter=300)
    assert res.converged is True
    lam, A_g, D_g = res.covariances
    assert lam.shape == (1,)
    assert A_g.shape == (2, 2)
    assert D_g.shape == (2, 2, 2)
    for g in range(2):
        _check_det1(A_g[g])
        assert np.allclose(D_g[g].T @ D_g[g], np.eye(2), atol=1e-5)


def test_eev_structure_per_component_Dg_shared_A_det1():
    rng = np.random.default_rng(14)
    X = _gen_diff_orientation(rng)
    res = fit_mclust(X, G_list=[2], models=[ModelName.EEV], random_state=7, max_iter=300)
    assert res.converged is True
    lam, A, D_g = res.covariances
    assert lam.shape == (1,)
    assert A.shape == (2,)
    assert D_g.shape == (2, 2, 2)
    _check_det1(A)
    for g in range(2):
        assert np.allclose(D_g[g].T @ D_g[g], np.eye(2), atol=1e-5)


def test_vev_structure_per_component_Dg_shared_A_det1_lambda_g():
    rng = np.random.default_rng(15)
    X = _gen_diff_orientation(rng)
    res = fit_mclust(X, G_list=[2], models=[ModelName.VEV], random_state=8, max_iter=300)
    assert res.converged is True
    lam_g, A, D_g = res.covariances
    assert lam_g.shape == (2,)
    assert A.shape == (2,)
    assert D_g.shape == (2, 2, 2)
    _check_det1(A)
    for g in range(2):
        assert np.allclose(D_g[g].T @ D_g[g], np.eye(2), atol=1e-5)
