import numpy as np
from pymclust_native.api import fit_mclust
from pymclust_native.models import ModelName


def test_full_jitter_config_affects_stability_vvv():
    # Construct nearly singular covariance by placing many points on a line
    rng = np.random.default_rng(0)
    n = 300
    x = rng.normal(0, 1e-3, size=n)
    X_line = np.column_stack([x, 10.0 * x])  # almost rank-1
    X = np.vstack([
        X_line + np.array([0.0, 0.0]),
        X_line + np.array([3.0, 3.0])
    ])

    # With a very small jitter, EM should still converge due to configurable jitter
    res_small = fit_mclust(
        X,
        G_list=[2],
        models=[ModelName.VVV],
        random_state=1,
        max_iter=200,
        full_jitter=1e-10,
    )
    assert res_small.converged is True

    # With larger jitter, loglik can differ but should still converge
    res_large = fit_mclust(
        X,
        G_list=[2],
        models=[ModelName.VVV],
        random_state=1,
        max_iter=200,
        full_jitter=1e-4,
    )
    assert res_large.converged is True


def test_var_floor_config_used_in_cov_updates():
    # Axis-aligned with tiny variance on one dimension; ensure floors are enforced
    rng = np.random.default_rng(123)
    X1 = np.column_stack([
        rng.normal(0.0, 1e-5, 120),  # tiny variance
        rng.normal(0.0, 0.2, 120),
    ])
    X2 = np.column_stack([
        rng.normal(3.0, 1e-5, 130),
        rng.normal(3.0, 0.3, 130),
    ])
    X = np.vstack([X1, X2])

    # Use VVI (per-component diagonal variances) to check floor application
    floor_val = 1e-4
    res = fit_mclust(
        X,
        G_list=[2],
        models=[ModelName.VVI],
        random_state=0,
        max_iter=300,
        var_floor=floor_val,
    )
    assert res.converged is True

    # Check learned diagonal variances respect the floor on the tiny-variance dimension
    diag_vars = res.covariances  # shape (G, d)
    assert diag_vars.shape[1] == 2
    assert np.all(diag_vars[:, 0] >= floor_val - 1e-12)
