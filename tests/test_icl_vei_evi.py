import numpy as np
from pymclust_native.api import fit_mclust
from pymclust_native.models import ModelName


def test_icl_selection_runs():
    rng = np.random.default_rng(0)
    X1 = rng.normal(loc=[0.0, 0.0], scale=0.25, size=(120, 2))
    X2 = rng.normal(loc=[3.0, 3.0], scale=0.35, size=(130, 2))
    X = np.vstack([X1, X2])

    res = fit_mclust(X, G_list=[1,2,3], models=[ModelName.VVV], selection_criterion="BIC", random_state=1)
    res2 = fit_mclust(X, G_list=[1,2,3], models=[ModelName.VVV], selection_criterion="ICL", random_state=1)
    assert res.bic is not None
    assert res2.bic is not None


def test_vei_evi_models():
    rng = np.random.default_rng(1)
    # Create axis-aligned clusters with shared shape or shared volume tendencies
    X1 = np.column_stack([rng.normal(0.0, 0.1, 150), rng.normal(0.0, 0.5, 150)])
    X2 = np.column_stack([rng.normal(3.0, 0.2, 140), rng.normal(3.0, 1.0, 140)])
    X = np.vstack([X1, X2])

    for model in (ModelName.VEI, ModelName.EVI):
        res = fit_mclust(X, G_list=[2], models=[model], random_state=2, max_iter=300)
        P = res.predict_proba(X[:10])
        assert np.allclose(P.sum(axis=1), 1.0, atol=1e-6)
        assert res.converged is True
