# Validation and Benchmarking vs R mclust

This document describes how to generate reference datasets, run both R mclust and pymclust-native on them, and compare metrics.

## Prerequisites
- Python >= 3.9 with numpy, sklearn
- R with packages: mclust, jsonlite, reticulate (for loading .npz)
- Ensure reticulate points to a Python with numpy installed
  - In R: `install.packages("reticulate"); library(reticulate); py_config()`
  - Optionally set `Sys.setenv(RETICULATE_PYTHON = "/path/to/python")` then restart R
- This repo checked out with folders:
  - pymclust-native/ (this package)
  - .mclust_R/ (optional reference repo; not required for benchmarking)

## Pipeline overview
1. Generate small, reproducible datasets (.npz) and a manifest.
2. Run pymclust-native on all datasets and selected models; dump JSON lines.
3. Run R mclust on the same datasets and models; dump JSON lines (uses reticulate to read .npz robustly).
4. Compare key metrics (loglik, BIC, ICL) and later align parameters and clustering scores.

## File layout
- scripts/bench/generate_reference_data.py
- scripts/bench/run_pymclust_bench.py
- scripts/bench/run_mclust_bench.R
- scripts/bench/compare_results.py
- .bench_data/ (generated)
- .bench_results/ (generated)

## How to run

### 1) Generate data
```bash
# from repo root
python pymclust-native/scripts/bench/generate_reference_data.py --outdir pymclust-native/.bench_data \
  --seeds 1 2 --sizes 300 --dims 2 3 --components 2 3
```

### 2) Python benchmark
```bash
python pymclust-native/scripts/bench/run_pymclust_bench.py \
  --manifest pymclust-native/.bench_data/manifest.jsonl \
  --out pymclust-native/.bench_results/pymclust_results.jsonl \
  --restarts 3 --var-floor 1e-6 --full-jitter 1e-9
```

### 3) R benchmark
```bash
Rscript pymclust-native/scripts/bench/run_mclust_bench.R \
  pymclust-native/.bench_data/manifest.jsonl \
  pymclust-native/.bench_results/mclust_results.jsonl
```

If reticulate cannot load numpy .npz on your R setup:
- Install reticulate: `install.packages("reticulate")`
- Ensure it points to a Python with numpy: set `RETICULATE_PYTHON` and check `py_config()`
- Our R loader uses `reticulate::py_get_item(npz, "X")` and falls back to `npz$files` if needed
- As a last resort, adapt data writer to CSV temporarily

### 4) Compare summaries
```bash
python pymclust-native/scripts/bench/compare_results.py \
  --r pymclust-native/.bench_results/mclust_results.jsonl \
  --py pymclust-native/.bench_results/pymclust_results.jsonl \
  --out pymclust-native/.bench_results/compare_summary.jsonl
```

The compare_summary.jsonl contains per-(dataset, model) diffs of loglik/BIC/ICL. If the JSON lines include labels or posterior for both R and Python runs, ARI and NMI will also be computed (posterior is argmax'ed to labels). Parameter alignment via component matching will be added next.

A quick sanity check can list worst |ΔBIC| entries against tolerance.

## Tolerances (initial draft)
- |ΔBIC| <= max(0.1 * |BIC|, 5.0)
- |ΔICL| similar to BIC
- Later: ARI >= 0.95 on clean synthetic datasets (after label matching)

## Serialization notes (R side)
- R script maps mclust variance to Python schema per family:
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
- Do not assume Sigma is 3D for shared-covariance families (e.g., EEE). Handle per-family branches.

## Next steps
- Extend serializers to export decomposed parameters (λ/λ_g, A/A_g, D/D_g) consistently and compare after component matching (Hungarian algorithm).
- Add pytest slow test to assert tolerances using precomputed R outputs (optional in CI).
- Use the differences report to drive refinements in EVV/EVE/VEE/VVE updates and to add caching/vectorization for performance.
