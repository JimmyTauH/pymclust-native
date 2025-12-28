#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(jsonlite)
  library(mclust)
})

# Run R mclust over reference datasets and dump JSON lines comparable to Python outputs

model_list <- c(
  # diagonal/spherical sanity
  "EII", "VII", "EEI", "VVI", "VEI", "EVI",
  # new families of interest
  "VEE", "EVE", "VVE", "EVV", "EEV", "VEV"
)

# Utility: safe tolist for matrices/arrays
to_list <- function(x) {
  if (is.null(x)) return(NULL)
  # Keep 2D/3D structure if present
  if (length(dim(x)) == 2) return(unname(split(x, row(x)))) # list of rows
  if (length(dim(x)) == 3) {
    G <- dim(x)[3]
    return(lapply(seq_len(G), function(g) unname(x[,,g, drop = FALSE])))
  }
  return(unname(x))
}

serialize_result <- function(res, model, dataset_id) {
  out <- list(
    dataset_id = dataset_id,
    model = model,
    loglik = unname(res$loglik),
    bic = if (!is.null(res$bic)) unname(res$bic) else NULL,
    icl = if (!is.null(res$icl)) unname(res$icl) else NULL,
    weights = unname(res$parameters$pro),
    means = unname(t(res$parameters$mean))
  )
  pars <- res$parameters
  var <- pars$variance
  fam <- if (!is.null(var$modelName)) var$modelName else model

  cov <- list(type = fam)

  # Helpers to extract eigen-decomposition consistently
  # In mclust, variance$d 是特征值（可能是向量或矩阵），variance$D 是特征向量
  get_D_shared <- function() {
    if (!is.null(var$D)) return(unname(var$D))
    NULL
  }
  get_D_g <- function(G) {
    if (!is.null(var$D)) {
      # 若 D 是 3D 数组 (d,d,G)，直接返回
      if (length(dim(var$D)) == 3 && dim(var$D)[3] == G) {
        return(lapply(seq_len(G), function(g) unname(var$D[,,g, drop=FALSE][,,1])))
      }
      # 若 D 是共享矩阵，则返回重复
      return(replicate(G, unname(var$D), simplify = FALSE))
    }
    NULL
  }
  # d 的形状随模型而变，可能为长度 d 的向量（共享 A）或 G x d 矩阵（A_g）
  get_A_shared <- function() {
    if (!is.null(var$d)) {
      if (is.vector(var$d)) return(as.numeric(var$d))
      if (is.matrix(var$d) && nrow(var$d) == 1) return(as.numeric(var$d[1,]))
      # 如果是 G x d，我们无法作为共享 A 返回
    }
    NULL
  }
  get_A_g <- function(G) {
    if (!is.null(var$d)) {
      if (is.matrix(var$d) && nrow(var$d) == G) return(unname(var$d))
      if (is.vector(var$d)) return(matrix(rep(var$d, each = G), nrow = G, byrow = TRUE))
    }
    NULL
  }

  # Map mclust families to Python schema
  if (fam == "EII") {
    # spherical, equal volume: lambda scalar
    cov$lambda <- unname(var$sigmasq)
  } else if (fam == "VII") {
    # spherical, variable volume: lambda_g vector of length G
    # mclust exposes scale per component via 'scale' or derived; here attempt via sigma/sigmasq + pro
    # For reliability, read 'sigmasq' if vector, else fallback from Sigma diagonals
    if (!is.null(var$sigmasq) && length(var$sigmasq) > 1) {
      cov$lambda <- unname(as.numeric(var$sigmasq))
    } else if (!is.null(pars$variance$Sigma)) {
      S <- pars$variance$Sigma
      if (length(dim(S)) == 3) {
        G <- dim(S)[3]
        lam <- sapply(seq_len(G), function(g) mean(diag(S[,,g])))
        cov$lambda <- unname(as.numeric(lam))
      } else {
        cov$lambda <- unname(as.numeric(var$sigmasq))
      }
    } else {
      cov$lambda <- unname(as.numeric(var$sigmasq))
    }
  } else if (fam == "EEI") {
    # diagonal, equal volume and shape: lambda scalar + A (d,), det(A)=1 在 mclust 内部已处理为 d
    cov$lambda <- unname(var$sigmasq)
    A <- get_A_shared()
    if (is.null(A) && !is.null(var$diagonal)) {
      # 可能直接提供了对角（缩放），退化为 A（非严格 det=1，但足以对比）
      A <- as.numeric(var$diagonal)
      # 尽量归一到 det(A)=1
      if (length(A) > 0) {
        geo <- exp(mean(log(A)))
        if (is.finite(geo) && geo > 0) A <- A / geo
      }
    }
    cov$A <- A
  } else if (fam == "VVI") {
    # per-component diagonal: diag (G x d)
    if (!is.null(var$Sigma) && length(dim(var$Sigma)) == 3) {
      G <- dim(var$Sigma)[3]; d <- dim(var$Sigma)[1]
      diag_mat <- matrix(NA_real_, nrow = G, ncol = d)
      for (g in seq_len(G)) diag_mat[g,] <- diag(var$Sigma[,,g])
      cov$diag <- unname(diag_mat)
    } else if (!is.null(var$diagonal)) {
      # 有的版本直接提供 diagonal (d) 但不分组件；尽量从 parameters$mean 的聚类标签重建不现实
      cov$diag <- matrix(var$diagonal, nrow = length(pars$pro), ncol = length(var$diagonal), byrow = TRUE)
    }
  } else if (fam == "VEI") {
    # variable volume, equal shape: lambda_g (G) + A (d)
    lam_g <- NULL
    if (!is.null(var$sigmasq) && length(var$sigmasq) > 1) lam_g <- as.numeric(var$sigmasq)
    A <- get_A_shared()
    cov$lambda_g <- lam_g
    cov$A <- A
  } else if (fam == "EVI") {
    # equal volume, variable shape: lambda (scalar) + A_g (G x d)
    cov$lambda <- unname(var$sigmasq)
    G <- length(pars$pro)
    cov$A_g <- get_A_g(G)
  } else if (fam == "EEE") {
    # shared full covariance: Sigma (d x d)
    if (!is.null(var$Sigma) && length(dim(var$Sigma)) == 2) {
      cov$Sigma <- unname(var$Sigma)
    } else if (!is.null(var$Sigma) && length(dim(var$Sigma)) == 3) {
      # 某些版本可能有 3D，但 G=1
      cov$Sigma <- unname(var$Sigma[,,1])
    }
  } else if (fam == "VVV") {
    # full covariance per component: Sigmas list of (d x d)
    if (!is.null(var$Sigma) && length(dim(var$Sigma)) == 3) {
      G <- dim(var$Sigma)[3]
      cov$Sigmas <- lapply(seq_len(G), function(g) unname(var$Sigma[,,g]))
    }
  } else if (fam == "VEE") {
    # variable volume, equal shape, equal orientation: lambda_g + A + D
    G <- length(pars$pro)
    cov$lambda_g <- if (!is.null(var$sigmasq) && length(var$sigmasq) > 1) unname(as.numeric(var$sigmasq)) else NULL
    cov$A <- get_A_shared()
    cov$D <- get_D_shared()
  } else if (fam == "EVE") {
    # equal volume, variable shape, equal orientation: lambda + A_g + D
    cov$lambda <- unname(var$sigmasq)
    G <- length(pars$pro)
    cov$A_g <- get_A_g(G)
    cov$D <- get_D_shared()
  } else if (fam == "VVE") {
    # variable volume, variable shape, equal orientation: lambda_g + A_g + D
    G <- length(pars$pro)
    cov$lambda_g <- if (!is.null(var$sigmasq) && length(var$sigmasq) > 1) unname(as.numeric(var$sigmasq)) else NULL
    cov$A_g <- get_A_g(G)
    cov$D <- get_D_shared()
  } else if (fam == "EVV") {
    # equal volume, variable shape, variable orientation: lambda + A_g + D_g
    cov$lambda <- unname(var$sigmasq)
    G <- length(pars$pro)
    cov$A_g <- get_A_g(G)
    cov$D_g <- get_D_g(G)
  } else if (fam == "EEV") {
    # equal volume, equal shape, variable orientation: lambda + A + D_g
    cov$lambda <- unname(var$sigmasq)
    G <- length(pars$pro)
    cov$A <- get_A_shared()
    cov$D_g <- get_D_g(G)
  } else if (fam == "VEV") {
    # variable volume, equal shape, variable orientation: lambda_g + A + D_g
    cov$lambda_g <- if (!is.null(var$sigmasq) && length(var$sigmasq) > 1) unname(as.numeric(var$sigmasq)) else NULL
    G <- length(pars$pro)
    cov$A <- get_A_shared()
    cov$D_g <- get_D_g(G)
  } else {
    # Fallback: try Sigma 3D
    if (!is.null(var$Sigma) && length(dim(var$Sigma)) == 3) {
      G <- dim(var$Sigma)[3]
      cov$Sigmas <- lapply(seq_len(G), function(g) unname(var$Sigma[,,g]))
    } else if (!is.null(var$Sigma)) {
      cov$Sigma <- unname(var$Sigma)
    }
  }

  out$cov <- cov
  return(out)
}

run_one <- function(dataset_path, models) {
  data <- npzload(dataset_path)
  X <- data$X
  outputs <- list()
  for (model in models) {
    res <- tryCatch({
      fit <- Mclust(X, G = NULL, modelNames = model)
      fit
    }, error = function(e) e)
    if (inherits(res, "error")) {
      outputs[[length(outputs)+1]] <- list(dataset_id = tools::file_path_sans_ext(basename(dataset_path)), model = model, error = as.character(res$message))
    } else {
      outputs[[length(outputs)+1]] <- serialize_result(res, model, tools::file_path_sans_ext(basename(dataset_path)))
    }
  }
  return(outputs)
}

# npz loader via reticulate (robust to versions)
npzload <- function(path) {
  if (requireNamespace("reticulate", quietly = TRUE)) {
    np <- reticulate::import("numpy", convert = FALSE)
    npz <- np$load(path)
    # Prefer dict-like access via py_get_item
    X <- NULL
    # Try to get key "X"
    X <- tryCatch(reticulate::py_get_item(npz, "X"), error = function(e) NULL)
    # Fallback: use files list to find a key
    if (is.null(X)) {
      files <- tryCatch(reticulate::py_to_r(npz$files), error = function(e) NULL)
      if (!is.null(files) && length(files) > 0) {
        key <- if ("X" %in% files) "X" else files[[1]]
        X <- tryCatch(reticulate::py_get_item(npz, key), error = function(e) NULL)
      }
    }
    if (is.null(X)) {
      stop(paste("Failed to read 'X' from npz:", path))
    }
    X <- reticulate::py_to_r(X)
    return(list(X = X))
  } else if (requireNamespace("RcppCNPy", quietly = TRUE)) {
    stop("Please install reticulate to load npz for now.")
  } else {
    stop("Need reticulate or RcppCNPy to load .npz files.")
  }
}

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  manifest <- "pymclust-native/.bench_data/manifest.jsonl"
  out <- "pymclust-native/.bench_results/mclust_results.jsonl"
  if (length(args) >= 1) manifest <- args[1]
  if (length(args) >= 2) out <- args[2]

  outdir <- dirname(out)
  if (!dir.exists(outdir)) dir.create(outdir, recursive = TRUE)

  outputs <- list()
  con <- file(manifest, open = "r")
  on.exit(close(con))
  while (length(line <- readLines(con, n = 1, warn = FALSE)) > 0) {
    rec <- jsonlite::fromJSON(line)
    ds_path <- rec$path
    res_list <- run_one(ds_path, model_list)
    outputs <- c(outputs, res_list)
  }

  con_out <- file(out, open = "w")
  on.exit(close(con_out), add = TRUE)
  for (rec in outputs) {
    writeLines(jsonlite::toJSON(rec, auto_unbox = TRUE), con_out)
  }
  message(sprintf("Wrote %d results to %s", length(outputs), out))
}

if (identical(environment(), globalenv())) {
  main()
}