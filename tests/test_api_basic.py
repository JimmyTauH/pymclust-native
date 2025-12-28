import numpy as np
from pymclust_native.api import fit_mclust
from pymclust_native.models import ModelName


def test_fit_mclust_selects_two_components():
    rng = np.random.default_rng(42)
    X1 = rng.normal(loc=[0.0, 0.0], scale=0.25, size=(150, 2))
    X2 = rng.normal(loc=[3.0, 3.0], scale=0.35, size=(170, 2))
    X = np.vstack([X1, X2])

    res = fit_mclust(X, G_list=[1, 2, 3], models=[ModelName.VVV], random_state=1, max_iter=300)

    assert res.model_name == ModelName.VVV
    assert res.means.shape == (2, 2)
    assert res.converged is True
    # Should prefer G=2 typically; indirectly check via shape
    assert res.weights.shape[0] == 2
    # check predict_proba works (now a method on result)
    P = res.predict_proba(X[:5])
    assert P.shape == (5, 2)
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-6)
