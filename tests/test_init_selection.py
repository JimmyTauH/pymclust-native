import numpy as np
from pymclust_native.api import fit_mclust
from pymclust_native.models import ModelName


def test_kmeans_init_and_n_init():
    rng = np.random.default_rng(123)
    X1 = rng.normal(loc=[0.0, 0.0], scale=0.3, size=(80, 2))
    X2 = rng.normal(loc=[3.0, 3.0], scale=0.3, size=(80, 2))
    X = np.vstack([X1, X2])

    # Random init single trial
    res_random = fit_mclust(
        X, G_list=[2], models=[ModelName.VVV], random_state=0,
        init_method="random", n_init=1, max_iter=300
    )
    # Kmeans with restarts
    res_km = fit_mclust(
        X, G_list=[2], models=[ModelName.VVV], random_state=0,
        init_method="kmeans", n_init=5, max_iter=300
    )

    # kmeans+restarts should be at least as good in loglik typically
    assert res_km.loglik >= res_random.loglik - 1e-6


def test_selection_criterion_bic_default():
    rng = np.random.default_rng(7)
    X1 = rng.normal(loc=[0.0, 0.0], scale=0.25, size=(100, 2))
    X2 = rng.normal(loc=[3.0, 3.0], scale=0.35, size=(130, 2))
    X = np.vstack([X1, X2])

    res = fit_mclust(X, G_list=[1,2,3], models=[ModelName.VVV], random_state=1)
    assert res.bic is not None
