import numpy as np
from pymclust_native.api import fit_mclust
from pymclust_native.models import ModelName


def test_eee_runs_and_posterior_normalizes():
    rng = np.random.default_rng(123)
    # Generate data with nearly shared covariance
    cov_shared = np.array([[1.0, 0.2],[0.2, 0.8]])
    X1 = rng.multivariate_normal([0.0, 0.0], cov_shared, size=120)
    X2 = rng.multivariate_normal([3.0, 3.0], cov_shared, size=130)
    X = np.vstack([X1, X2])

    res = fit_mclust(X, G_list=[2], models=[ModelName.EEE], random_state=0, max_iter=300)
    assert res.converged is True
    P = res.predict_proba(X[:10])
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-6)
