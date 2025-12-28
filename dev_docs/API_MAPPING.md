API Mapping: R mclust -> pymclust-native
=======================================

Purpose
- Provide a quick reference mapping from common mclust R functions/arguments/returns to the Python-native counterparts in pymclust-native.

Legend
- R (mclust): Function/arg/slot names from the R package
- Python: Proposed/implemented names in pymclust-native
- Status: implemented / planned / partial

Top-level Fitting
- R: Mclust(x, G=1:9, modelNames=c("VVV", ...), initialization=list(...), control=list(...))
- Python: fit_mclust(X, G_list=[1..9], models=[ModelName.VVV, ...], random_state=None, max_iter=1000, tol=1e-5, init_method="kmeans", n_init=5, selection_criterion="BIC"|"ICL")
  - Status: implemented (broad set of models)
  - Notes: initialization uses k-means (default) or random; multi-restarts via n_init; ICL uses BIC + entropy penalty (2 * sum z log z). Numerical stability: var_floor (diag floors) and full_jitter (full-cov jitter) are configurable via EMOptions and fit_mclust. predict_proba is a method on GMMResult; EEI init hardened to (lambda, A) tuple.

Density Estimation
- R: densityMclust(data, G=..., modelNames=..., ...) -> mclustDensity object
- Python: fit_mclust(X, ...) returns GMMResult (used for clustering and density); density-specific helpers planned
  - Status: partial

Classification
- R: predict(object, newdata) returns classification, z (posterior), uncertainty
- Python: GMMResult.predict(X), GMMResult.predict_proba(X)
  - Status: implemented for posterior and hard labels; uncertainty to be added; predict_proba is a method on GMMResult (no closures)

Model Families (modelNames)
- R: "EII", "VII", "EEI", "VEI", "EVI", "VVI", "EEE", "EVV", "EVE", "EEV", "VEV", "VEE", "VVE", "VVV"
- Python: ModelName Enum: EII, VII, EEI, VVI, VEI, EVI, VVV, EEE, VEE, EVE, VVE, EVV, EEV, VEV
  - Status: implemented (full set in scope)

Model Selection
- R: BIC (internal), ICL available via icl() and options
- Python: BIC/ICL computed in fit_mclust
  - Status: BIC and ICL implemented (ICL = BIC + 2 * sum_i sum_g z_ig log z_ig). Parameter counting includes VEI/EVI/EEE/VEE/EVE/VVE/EVV. ICL posterior uses EM log-densities with full_jitter for stability.

Return Object
- R: Mclust object slots: modelName, G, BIC, loglik, parameters (pro, mean, variance{modelName, d, sigma/scale/shape/orientation}, z, classification, uncertainty)
- Python: GMMResult fields: model_name, means, covariances, weights, loglik, bic, n_iter, converged
  - Posterior z: via predict_proba
  - classification: via predict
  - uncertainty: planned
  - variance decomposition fields (scale/shape/orientation): stored implicitly in covariances; for VEE/EVE/VVE/EVV, covariances are tuples of (lambda/_g, A/_g, D or D_g). See DEVELOPMENT.md Data conventions for exact shapes.
  - JSON schema used in benchmarks (cov field):
    - EII: {type, lambda}
    - VII: {type, lambda}
    - EEI: {type, lambda, A}
    - VVI: {type, diag}
    - VEI: {type, lambda_g, A}
    - EVI: {type, lambda, A_g}
    - EEE: {type, Sigma}
    - VVV: {type, Sigmas}
    - VEE: {type, lambda_g, A, D}
    - EVE: {type, lambda, A_g, D}
    - VVE: {type, lambda_g, A_g, D}
    - EVV: {type, lambda, A_g, D_g}

Arguments and Options Mapping (selected)
- R: control = list(eps, tol, itmax, equalPro, ...)
- Python: EMOptions(tol, max_iter, init_method, n_init, random_state, var_floor, full_jitter) – exposed via fit_mclust parameters
  - equalPro: planned as an option (fit_mclust(..., equal_pro=False)) to constrain weights if True

Initialization
- R: initialization=list(hcPairs, subset, noise, ...)
- Python: options.init_method ("kmeans" or "random") – k-means planned; current uses random subset means

Predictions
- R: predict.Mclust(object, newdata) -> list(classification, z, uncertainty)
- Python: GMMResult.predict(X), GMMResult.predict_proba(X); uncertainty planned

Diagnostics
- R: icl(object) -> ICL value
- Python: ICL implemented as an option in fit_mclust (selection_criterion="BIC"|"ICL")

Data
- R: data can be vectors/matrices/data frames
- Python: X is numpy array of shape (n, d)

Vignettes and Visualization
- R: mclustV(), mclustD(), plots
- Python: minimal plotting helpers planned; focus remains on computational core

Examples
- R:
  - fit <- Mclust(x, G=1:9)
  - pred <- predict(fit, newdata=x)
- Python:
  - from pymclust_native.api import fit_mclust
  - res = fit_mclust(X, G_list=range(1,10), models=[ModelName.VVV, ModelName.VVI])
  - labels = res.predict(X)
  - z = res.predict_proba(X)
