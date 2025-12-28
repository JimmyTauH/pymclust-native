pymclust-native Roadmap
=======================

Vision
- Provide a faithful, well-tested Python-native reimplementation of mclust for Gaussian mixture modelling with clear API, documentation, and performance sufficient for medium-sized datasets.

Milestones

M1: Usable Core (current)
- Status: DONE
- Deliverables:
  - EM for Gaussian mixtures
  - Model families: VVV, EII, VII, EEI, VVI
  - BIC-based model selection (fit_mclust)
  - Basic tests and documentation
- Gaps:
  - ICL criterion [DONE]

M2: Initialization, Stability, Selection (short term)
- Status: DONE
- Deliverables:
  - Deterministic k-means init + random restarts with best loglik pick [DONE]
  - ICL computation and selection option (BIC vs ICL) [DONE]
  - Covariance floors/jitter configurable via EMOptions.var_floor and EMOptions.full_jitter [DONE]
  - Predict_proba as method on GMMResult [DONE]
  - API parity wrappers (densityMclust-like names) [PLANNED]
- Tests:
  - Monotonicity checks of EM loglik across iterations [PLANNED]
  - Reproducible selections under seeds [PLANNED]

M3: Expanded Model Families
- Status: DONE
- Deliverables:
  - Implemented: VEE, EVE, VVE, EVV, EEV, VEV (EEE completed earlier)
  - Approach: pooled/per-class eigendecomposition for D and diagonal shape normalization (det=1)
  - Parameter counting: implemented in api._num_params for all families
- Validation:
  - Bench脚本端到端可用（Python/R），compare_results.py 生成对比摘要
  - 已添加慢测 tests/test_regression_against_mclust.py（需先运行脚本产出结果）

M4: Parity and Validation Against mclust (in progress)
- Deliverables:
  - Reference scripts对 BIC/ICL 做对比（已完成），后续加入聚类指标（ARI/NMI）
  - 容忍阈值与差异说明（VALIDATION.md 已初版）
  - 简单可视化（可选，后续）

M5: Usability & Packaging
- Deliverables:
  - README examples, API docs (docstrings + mkdocs or Sphinx)
  - Versioned releases, changelog
  - CI (GitHub Actions) with Linux/macOS
  - Pre-commit hooks (ruff/black, isort) and type hints pass (mypy optional)

M6: Advanced Features (post parity)
- Deliverables:
  - Semi-supervised classification (analog to mclustDA)
  - Dimension reduction for model-based clustering (analog to mclustDR)
  - Bootstrap utilities and uncertainty quantification helpers
  - Missing data strategies (EM with imputation)

M7: Backend Abstraction (planned)
- Deliverables:
  - Define minimal backend interface (NumPy first), then prototype Torch/JIT backends
  - Encapsulate log-density and linear algebra ops behind backend layer
  - Keep numerical behavior and data layouts consistent across backends

Task Backlog (updated)
- Core
  - [x] Move predict_proba into GMMResult with model metadata
  - [x] Add k-means init + multiple restarts; expose n_init in API
  - [x] Implement ICL
  - [x] Improve numerical stability and add covariance floors per family (configurable var_floor/full_jitter)
- Models
  - [x] VEI and EVI (diagonal with shared/variable shape-volume constraints)
  - [x] EEE (shared covariance across components)
  - [x] VEE, EVE, VVE, EVV
  - [x] Accurate parameter counting for all families (including EEV/VEV)
- Validation
  - [ ] Create reproducible comparison notebooks vs mclust
  - [x] Regression test scaffold: tests/test_regression_against_mclust.py（依赖预生成结果）
  - [x] Bench scripts runnable end-to-end (Python/R); serialize covariances per family; generate compare_summary
- Developer Experience
  - [ ] Add CONTRIBUTING.md
  - [ ] Add ruff/black configs and pre-commit
  - [ ] Set up GitHub Actions workflow for tests
  - [x] Register pytest marks (slow, mclust)
- Docs
  - [ ] API reference and examples with mkdocs/Sphinx
  - [ ] Explain mclust covariance parameterization and mapping
- Performance
  - [ ] Vectorize E-step across models fully; cache assembled covariances where possible
  - [ ] Optional numba acceleration flag

Near-term Priority Order
- [P2] 扩展 compare_results：加入 ARI/NMI 与参数对齐（λ/λ_g, A/A_g, D/D_g），使用匈牙利匹配
- [P2] 性能：E-step 向量化/缓存装配后的协方差，减少重复 Cholesky
- [P3] 文档与教程：完善 README、快速上手与基准对齐指南；准备发布
