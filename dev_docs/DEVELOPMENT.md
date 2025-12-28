pymclust-native Development Guide
=================================

Overview
- Goal: Native Python reimplementation of R package mclust for Gaussian mixture modelling (clustering, classification, density estimation).
- Status: EM with model selection and covariance parameterizations (VVV, EEE, EII, VII, EEI, VVI, VEI, EVI, VEE, EVE, VVE, EVV, EEV, VEV). K-means init + multi-restarts and ICL selection supported. predict_proba has been moved into GMMResult. APIs are stabilizing. EEI initialization and first E-step compatibility have been hardened.

Repository layout
- pyproject.toml: Build config (setuptools)
- README.md: Project summary
- dev_docs/
  - DEVELOPMENT.md: This document
- src/pymclust_native/
  - __init__.py: Public exports
  - api.py: fit_mclust top-level interface and model selection (BIC/ICL) accounting
  - models.py: Result dataclasses and enums
  - covariance.py: Covariance parameter initializations and constrained updates
  - em.py: EM algorithm core and log-density implementations (with stability floors/jitter)
- tests/
  - test_sanity.py: Basic import/fit smoke test (VVV)
  - test_api_basic.py: fit_mclust and predict_proba basic checks
  - test_models_parametric.py: Parametric covariance families tests
  - test_new_families_basic.py: Basic smoke tests for VEE/EVE/VVE/EVV/EEV/VEV (run + posterior normalization)
  - test_new_families_structure.py: Structure checks for VEE/EVE/VVE/EVV/EEV/VEV (det(A)=1, orthonormal D)
  - test_configurable_stability.py: Stability under configurable full_jitter/var_floor settings
  - test_regression_against_mclust.py: Regression vs R mclust (marked with slow/mclust; requires running bench scripts)

Environment & tooling
- Python: >= 3.9
- Dependencies: numpy, scipy
- Recommended: uv for environment and installs

Common commands (zsh)
- Create and use a virtual environment with uv
  - uv venv .venv
  - source .venv/bin/activate
- Install project and test dependencies
  - uv pip install -e ./pymclust-native
  - uv pip install pytest
- Run tests
  - pytest -q pymclust-native/tests

Defaults and options (current)
- EMOptions defaults:
  - tol: 1e-5
  - max_iter: 1000
  - init_method: "kmeans" (alternatives: "random")
  - n_init: 5
  - random_state: None
  - var_floor: 1e-8 (used in covariance floors via CovarianceModel)
  - full_jitter: 1e-6 (used in full-covariance Cholesky/jitter)
- fit_mclust options:
  - selection_criterion: "BIC" (default) or "ICL" (implemented)
  - models: subset supported: EII, VII, EEI, VVI, VEI, EVI, VVV, EEE, VEE, EVE, VVE, EVV, EEV, VEV
  - init_method/n_init/random_state propagated to EM
  - var_floor/full_jitter: override numerical stability parameters for EM/log-densities

Architecture
1) API layer (api.py)
   - fit_mclust(X, G_list, models, ...) runs a grid over component counts and model families, fits via EM, computes BIC, returns the best model.
   - BIC: 2 * loglik - m * log(n) with m=num free parameters.
   - Predict_proba is implemented as a method on GMMResult; no dynamic closures.

2) EM core (em.py)
   - E-step: compute component-wise log densities and responsibilities via log-sum-exp.
   - Accelerations:
     - Shared-orientation families (VEE/EVE/VVE): rotate once (X' = X D), cache rotated means and per-component diagonal variances, then use vectorized diagonal log-density for all components.
     - Full-covariance families (VVV/EVV/EEV/VEV): cache per-component Cholesky factors L_g and log-determinants; E-step computes Mahalanobis distance via G batched solves without recomputing Cholesky.
   - M-step: update weights, means, and covariance parameters respecting constraints per model.
   - Log-density helpers:
     - _log_gaussian_full (Cholesky-based), _log_gaussian_diag, _log_gaussian_spherical
     - Vectorized paths for shared D and cached full-cov families as described above

3) Covariance families (covariance.py)
   - Parameterization via Σ_g = λ_g D_g A_g D_g^T; in current increment we support simplified families and pooled-eigendecomposition updates for orientation/shared shape where applicable:
     - VVV: Full covariance per component
     - EII: Shared spherical variance λ
     - VII: Per-component spherical variance λ_g
     - EEI: Shared diagonal shape A (det=1) with global scale λ
     - VVI: Per-component diagonal variances diag(σ^2_{g,1..d})
     - VEI: Diagonal with variable volume and equal shape (shared A, det=1)
     - EVI: Diagonal with equal volume and variable shape (per-component A_g, det=1)
     - EEE: Full covariance shared across components
   - Update functions:
     - update_EII, update_VII, update_EEI, update_VVI, update_VEI, update_EVI (EEE shared covariance handled in EM)
   - Helper: _safe_det1_diagonal to enforce det(A)=1 for EEI.

4) Models and results (models.py)
   - ModelName enum lists supported families
   - GMMResult holds fitted parameters, log-likelihood, BIC, convergence info
   - predict_proba is implemented as a method on GMMResult

Parameter counting (for BIC)
- weights: (G - 1)
- means: G * d
- covariance params:
  - VVV: G * d * (d + 1) / 2
  - EII: 1
  - VII: G
  - EEI: 1 + (d - 1)  [diag shape with det=1]
  - VVI: G * d
  - VEI: G + (d - 1)
  - EVI: 1 + G * (d - 1)
  - EEE: d * (d + 1) / 2
  - VEE: G + (d - 1) + d(d - 1)/2
  - EVE: 1 + G*(d - 1) + d(d - 1)/2
  - VVE: G + G*(d - 1) + d(d - 1)/2
  - EVV: 1 + G*(d - 1) + G*d(d - 1)/2
  - EEV: 1 + (d - 1) + G*d(d - 1)/2
  - VEV: G + (d - 1) + G*d(d - 1)/2

Current limitations and TODOs
- Initialization: [DONE] k-means default with multiple restarts (n_init), deterministic seeds supported via random_state.
- Numerical stability: [IMPROVED] applied variance floor (1e-8) and full-cov jitter (1e-6); [DONE] configurable via EMOptions; [DONE] var_floor threaded via CovarianceModel; [DONE] EEI initialization set to (lambda, A) tuple to avoid first E-step issues; [DONE] _log_prob_components guards for placeholder covariances.
- Model coverage: [DONE] EVV/EVE/VEE/VVE/EEV/VEV implemented with pooled/per-component eigendecomposition as appropriate; next step is refine/validate against mclust via benchmarks.
- Predict_proba location: DONE. Method on GMMResult with model metadata.
- ICL criterion: DONE. ICL = BIC + 2 * sum_i sum_g z_ig log z_ig.
- Missing data: basic imputation/handling (future step).
- Validation: Cross-check against mclust R outputs on built-in datasets (bench scripts available).
- Performance: vectorize E-step across families; consider JIT (numba) later if needed.
- API parity: Provide densityMclust-like naming and convenience wrappers.

Backend design (planned)
- Goal: pluggable computational backends (NumPy, PyTorch, JIT) while keeping numerical behavior consistent.
- Minimal backend interface (draft):
  - array/tensor creation & dtype control
  - matmul / einsum
  - cholesky, triangular/linear solve, logsumexp
  - random sampling (for initialization)
- Integration strategy:
  - Encapsulate log-density computations to call backend ops only.
  - Begin with NumpyBackend, then prototype TorchBackend/JITBackend.
  - Keep data layout (n,d), (G,d), (G,d,d) consistent across backends.

Enhancements and options (stability)
- Configurable numerical stability parameters [DONE]
  - EMOptions.full_jitter controls Cholesky jitter on full covariances (default 1e-6). Also used in API ICL posterior when computing full-cov log densities.
  - EMOptions.var_floor controls covariance floors used in diagonal/spherical families via CovarianceModel(var_floor).
- Optional prediction-time safeguards [PLANNED]
  - For GMMResult.predict_proba on diagonal/spherical families, optionally clip extremely small variances before evaluating log densities to avoid inf/nan in extreme cases. This should be an opt-in flag to avoid deviating from trained parameters.

Testing additions [PLANNED/DONE]
- Added tests for configurable jitter and floors:
  - tests/test_configurable_stability.py::test_full_jitter_config_affects_stability_vvv ensures EM converges on nearly singular data with different jitter settings.
  - tests/test_configurable_stability.py::test_var_floor_config_used_in_cov_updates ensures learned diagonal variances respect configured var_floor.
- Add more assertions to compare log-likelihood monotonicity and sensitivity analysis under varying jitter/floor values. [PLANNED]

Contributing workflow
- Create feature branches per topic (e.g., feat/cov-EEI, fix/stability-diag)
- Add or update tests under tests/
- Ensure formatting and linting (optional tools can be added later)
- Run the full test suite before merging

Testing strategy
- Unit tests for:
  - Log-density functions: shapes, finite outputs, normalization via posterior
  - M-step updates: sanity monotonicity of log-likelihood across iterations (smoke)
  - BIC selection: simple synthetic mixtures prefer correct G under supported models
- Performance tests (manual/bench): scripts/bench with small datasets to confirm speedups from caching/rotation vs baseline.
- Regression tests: Match mclust BIC and (optionally) ICL within tolerances using tests/test_regression_against_mclust.py after running bench scripts.

Release plan (short-term)
- v0.0.2: ICL implemented; expanded to VEI/EVI; added EEE; stability floors/jitter applied.
- v0.0.3: Add EVV/EVE/VEE/VVE/EEV/VEV; docs polishing and examples; regression scaffold vs mclust.
- v0.0.4: Accelerated E-step (shared-D rotation + full-cov Cholesky caching), predict_proba fast paths; add ARI/NMI in compare; README/tutorials for release.

Examples
- Basic usage:
  - from pymclust_native.api import fit_mclust
  - res = fit_mclust(X, G_list=[1,2,3], models=[ModelName.VVV, ModelName.VVI])
  - labels = res.predict(X)
  - post = res.predict_proba(X)

Data conventions
- X is (n, d) float array
- Means shape: (G, d)
- Covariance representation stored in res.covariances depends on model:
  - VVV: (G, d, d)
  - EII: array([λ])
  - VII: (G,)
  - EEI: tuple (array([λ]), A_diag with det=1)
  - VVI: (G, d)
  - VEI: (lambda_g vector (G,), A_diag (d,))
  - EVI: (lambda array([λ]), A_g (G, d))
  - EEE: shared Sigma (d, d)
  - VEE: tuple (lambda_g (G,), A_diag (d,), D (d, d))
  - EVE: tuple (lambda (array([λ])), A_g (G, d), D (d, d))
  - VVE: tuple (lambda_g (G,), A_g (G, d), D (d, d))
  - EVV: tuple (lambda (array([λ])), A_g (G, d), D_g (G, d, d))
  - EEV: tuple (lambda (array([λ])), A (d,), D_g (G, d, d))
  - VEV: tuple (lambda_g (G,), A (d,), D_g (G, d, d))

License
- This project reimplements algorithms inspired by mclust (GPL-licensed). Ensure compatibility when distributing code.

Session Notes and Next Steps
- Completed:
  - Project scaffold with setuptools and tests
  - EM implementation with VVV full covariance
  - Top-level fit_mclust with BIC/ICL selection
  - Added covariance families: EII, VII, EEI, VVI, VEI, EVI, EEE, VEE, EVE, VVE, EVV
  - K-means initialization and multiple restarts (n_init)
  - Numerical stability: configurable variance floor and full-cov jitter; hardened EEI init and first E-step compatibility
  - Added ROADMAP and API mapping docs; pytest.ini marks registered (slow, mclust)
  - Bench pipeline: Python and R scripts runnable; compare_results produces ΔBIC/ΔICL summaries
- Next session priorities:
  - [P2] Extend compare_results with ARI/NMI and parameter alignment (Hungarian matching) for EVV/EVE/VEE/VVE/EEV/VEV
  - [P2] Performance improvements (vectorized E-step, caching of assembled covariances)
  - [P3] Packaging polish: README quickstart, tutorials, release notes
