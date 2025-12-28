import numpy as np
from pymclust_native.api import fit_mclust
from pymclust_native.models import ModelName


def test_spherical_models_eii_vii():
    rng = np.random.default_rng(0)
    # spherical clusters
    X1 = rng.normal(loc=[0.0, 0.0], scale=0.2, size=(150, 2))
    X2 = rng.normal(loc=[3.0, 3.0], scale=0.4, size=(150, 2))
    X = np.vstack([X1, X2])

    for model in (ModelName.EII, ModelName.VII):
        res = fit_mclust(X, G_list=[1,2,3], models=[model], random_state=1, max_iter=200)
        assert res.converged is True
        assert res.weights.shape[0] in (1,2,3)
        # posterior sums to 1
        P = res.predict_proba(X[:10])
        assert np.allclose(P.sum(axis=1), 1.0, atol=1e-6)


def test_diagonal_vvi():
    rng = np.random.default_rng(1)
    # axis-aligned anisotropic
    X1 = np.column_stack([
        rng.normal(0.0, 0.1, 200),
        rng.normal(0.0, 0.5, 200),
    ])
    X2 = np.column_stack([
        rng.normal(3.0, 0.4, 180),
        rng.normal(3.0, 0.2, 180),
    ])
    X = np.vstack([X1, X2])

    res = fit_mclust(X, G_list=[1,2,3], models=[ModelName.VVI], random_state=2, max_iter=200)
    assert res.converged is True
    P = res.predict_proba(X[:10])
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-6)
