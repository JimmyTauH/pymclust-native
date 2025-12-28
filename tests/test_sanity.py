import numpy as np
from pymclust_native import GaussianMixtureEM, ModelName


def test_import_and_fit():
    rng = np.random.default_rng(0)
    X1 = rng.normal(loc=[0.0, 0.0], scale=0.3, size=(100, 2))
    X2 = rng.normal(loc=[3.0, 3.0], scale=0.4, size=(120, 2))
    X = np.vstack([X1, X2])

    gmm = GaussianMixtureEM(G=2, model_name=ModelName.VVV)
    result = gmm.fit(X)

    assert result.converged is True
    assert result.means.shape == (2, 2)
    assert result.covariances.shape == (2, 2, 2)
    assert np.isfinite(result.loglik)
