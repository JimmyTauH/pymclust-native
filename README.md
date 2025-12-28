pymclust-native
=================

Native Python reimplementation of the R package mclust for Gaussian mixture modelling.

Note: This project is under active development. APIs are stabilizing; expect minor changes until v0.1.

Highlights
- Models: EII, VII, EEI, VEI, EVI, VVI, EEE, VEE, EVE, VVE, EVV, EEV, VEV, VVV
- Selection: BIC and ICL (mclust-style BIC = 2*loglik - m*log(n))
- Stability: configurable var_floor (diagonal floors) and full_jitter (full-cov jitter)
- Initialization: k-means with multiple restarts, deterministic seeds
- Performance: accelerated E-step via shared-orientation rotation (VEE/EVE/VVE) and per-component Cholesky caching (VVV/EVV/EEV/VEV); predict_proba uses the same fast paths

Install (editable)
- Python >= 3.9
- Dependencies: numpy, scipy
- From repo root:
  - pip install -e ./pymclust-native

Quickstart

BIC selection:
```python
from pymclust_native.api import fit_mclust
from pymclust_native.models import ModelName
import numpy as np

# synthetic 2D two-cluster data
data = np.vstack([
    np.random.default_rng(0).normal([0,0], 0.3, size=(150,2)),
    np.random.default_rng(1).normal([3,3], 0.4, size=(170,2)),
])

# fit VVV over G in {1,2,3} using BIC selection
res = fit_mclust(data, G_list=[1,2,3], models=[ModelName.VVV], selection_criterion="BIC")
print(res.model_name, res.means.shape, res.loglik)

# probabilities and labels
P = res.predict_proba(data)
labels = res.predict(data)
```

ICL selection (example):
```python
res = fit_mclust(data, G_list=[1,2,3], models=[ModelName.VVV], selection_criterion="ICL")
```

Tutorials
- See dev_docs/DEVELOPMENT.md for architecture and options
- See dev_docs/API_MAPPING.md for mapping vs R mclust
- Validation and benchmarks: dev_docs/VALIDATION.md and scripts/bench

Benchmarks vs R mclust (optional)
- ARI/NMI: compare_results.py will compute ARI/NMI if the JSON results include labels or posterior for both runs (posterior is argmax'ed to labels).
- Generate small datasets
  - python pymclust-native/scripts/bench/generate_reference_data.py --outdir pymclust-native/.bench_data --seeds 1 2 --sizes 300 --dims 2 3 --components 2 3
- Run Python benchmark
  - python pymclust-native/scripts/bench/run_pymclust_bench.py --manifest pymclust-native/.bench_data/manifest.jsonl --out pymclust-native/.bench_results/pymclust_results.jsonl --restarts 3 --var-floor 1e-6 --full-jitter 1e-9
- Run R benchmark
  - Rscript pymclust-native/scripts/bench/run_mclust_bench.R pymclust-native/.bench_data/manifest.jsonl pymclust-native/.bench_results/mclust_results.jsonl
- Compare summaries
  - python pymclust-native/scripts/bench/compare_results.py --r pymclust-native/.bench_results/mclust_results.jsonl --py pymclust-native/.bench_results/pymclust_results.jsonl --out pymclust-native/.bench_results/compare_summary.jsonl

Run tests
- pytest -q pymclust-native/tests
- Optional slow regression against R: pytest -q -m "slow and mclust" pymclust-native/tests/test_regression_against_mclust.py

Backend & integration (planned)
- Backend abstraction for NumPy/PyTorch/JIT will be introduced to ease integration with data science and deep learning frameworks.

License and Credits
- Reimplementation inspired by mclust (GPL-licensed)
- Original R package copyright (c) Chris Fraley, Adrian E. Raftery, Luca Scrucca (maintainer), Thomas Brendan Murphy, Michael Fop
- Python port copyright (c) Zhentao He

Citation
- Scrucca L, Fraley C, Murphy TB, Raftery AE (2023). Model-Based Clustering, Classification, and Density Estimation Using mclust in R. Chapman and Hall/CRC. ISBN 978-1032234953, doi:10.1201/9781003277965, https://mclust-org.github.io/book/.
