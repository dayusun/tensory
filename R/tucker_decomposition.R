#' @include tensor_class.R ttensor_class.R tensor_dense_methods.R tensor_ttm.R
NULL

.hosvd_rank_from_tol <- function(sv, energy_budget) {
  # Choose smallest r such that truncation tail energy is below the budget.
  tail_sq <- c(rev(cumsum(rev(sv^2))), 0)
  # tail_sq[r + 1] = sum(sv[(r+1):end]^2); tail_sq[length(sv)+1] = 0.
  r <- which(tail_sq[seq_len(length(sv) + 1L)] <= energy_budget)[1L]
  if (is.na(r)) {
    r <- length(sv)
  } else {
    r <- r - 1L
  }
  max(r, 1L)
}

.resolve_hosvd_ranks <- function(X, ranks, tol, dims, N) {
  if (!is.null(ranks)) {
    if (length(ranks) == 1L) ranks <- rep(as.integer(ranks), N)
    ranks <- as.integer(ranks)
    if (length(ranks) != N) {
      stop("ranks must have length 1 or ndims(X).")
    }
    if (any(ranks < 1L) || any(ranks > dims)) {
      stop("ranks must be between 1 and the corresponding mode size.")
    }
    return(ranks)
  }
  if (!is.null(tol)) {
    return(NULL)
  }
  pmin(dims, as.integer(dims))
}

#' Higher-Order Singular Value Decomposition
#'
#' Computes the (truncated) higher-order SVD of a tensor, returning a Tucker
#' representation whose factor matrices span the leading left singular spaces
#' of the mode-n unfoldings. Mirrors the MATLAB Tensor Toolbox `hosvd`.
#'
#' @param X A Tensor or array-like object.
#' @param ranks Optional integer vector of per-mode truncation ranks. Scalar
#'   inputs are replicated across modes.
#' @param tol Relative truncation tolerance. Ignored when `ranks` is supplied.
#'   Per-mode truncation keeps enough singular vectors to ensure the dropped
#'   energy is below `tol^2 / ndims(X)` of `fnorm(X)^2`.
#' @param dimorder Order in which modes are processed (matters only for
#'   `sequential = TRUE`).
#' @param sequential Logical; if `TRUE` (default) the ST-HOSVD variant is used,
#'   progressively contracting the core, which is typically more accurate and
#'   cheaper than the classical HOSVD.
#' @param verbosity Non-negative integer controlling diagnostic messages.
#' @return A `TTensor`.
#' @examples
#' X <- tensor(array(runif(60), dim = c(3, 4, 5)))
#' T <- hosvd(X, ranks = c(2, 3, 4))
#' @export
hosvd <- function(X,
                  ranks = NULL,
                  tol = NULL,
                  dimorder = NULL,
                  sequential = TRUE,
                  verbosity = 0L) {
  X <- as.tensor(X)
  if (!inherits(X, "Tensor")) {
    stop("X must be a Tensor or coercible to one.")
  }
  dims <- as.integer(X$dim())
  N <- length(dims)
  if (N < 1L) {
    stop("hosvd requires a tensor with at least 1 mode.")
  }

  if (is.null(dimorder)) {
    dimorder <- seq_len(N)
  }
  dimorder <- as.integer(dimorder)
  if (length(dimorder) != N || !setequal(dimorder, seq_len(N))) {
    stop("dimorder must be a permutation of 1:ndims(X).")
  }

  ranks <- .resolve_hosvd_ranks(X, ranks, tol, dims, N)
  energy_budget <- NULL
  if (is.null(ranks)) {
    normX2 <- sum(X$data^2)
    energy_budget <- (tol^2) * normX2 / N
  }

  U <- vector("list", N)
  G <- X

  for (n in dimorder) {
    src <- if (sequential) G else X
    Xn <- as.matrix(tenmat(src, rdims = n))
    sv <- svd(Xn)
    r_n <- if (!is.null(ranks)) ranks[n] else .hosvd_rank_from_tol(sv$d, energy_budget)
    if (r_n > length(sv$d)) r_n <- length(sv$d)
    U[[n]] <- sv$u[, seq_len(r_n), drop = FALSE]
    if (verbosity > 0L) {
      message(sprintf(" hosvd mode %d: keeping %d / %d singular vectors",
                      n, r_n, length(sv$d)))
    }
    if (sequential) {
      G <- ttm(G, t(U[[n]]), mode = n)
    }
  }

  if (!sequential) {
    Ut <- lapply(U, t)
    G <- ttm(X, Ut, mode = seq_len(N))
  }

  ttensor(G, U)
}

.init_tucker_factors <- function(X, ranks, dims, N, dimorder, init) {
  if (is.list(init)) {
    if (length(init) != N) {
      stop("init list must have length equal to ndims(X).")
    }
    U <- lapply(init, as.matrix)
    for (n in seq_len(N)) {
      if (!identical(dim(U[[n]]), c(as.integer(dims[n]), as.integer(ranks[n])))) {
        stop(sprintf("init[[%d]] must have dimensions %d x %d.",
                     n, dims[n], ranks[n]))
      }
    }
    return(U)
  }

  if (!is.character(init)) {
    stop("init must be a character string or list of matrices.")
  }
  init <- match.arg(init, c("nvecs", "random", "eigs"))

  if (init == "random") {
    U <- vector("list", N)
    for (n in seq_len(N)) {
      M <- matrix(stats::rnorm(dims[n] * ranks[n]), dims[n], ranks[n])
      qrm <- qr(M)
      U[[n]] <- qr.Q(qrm)[, seq_len(ranks[n]), drop = FALSE]
    }
    return(U)
  }

  # nvecs / eigs: leading singular vectors of the mode-n unfolding. The first
  # mode in dimorder is not used before it is overwritten, so skip it.
  U <- vector("list", N)
  skip_first <- dimorder[1L]
  for (n in seq_len(N)) {
    if (n == skip_first) {
      U[[n]] <- matrix(0, dims[n], ranks[n])
      next
    }
    if (dims[n] >= ranks[n]) {
      U[[n]] <- nvecs(X, mode = n, r = ranks[n])
    } else {
      M <- matrix(stats::rnorm(dims[n] * ranks[n]), dims[n], ranks[n])
      U[[n]] <- qr.Q(qr(M))[, seq_len(ranks[n]), drop = FALSE]
    }
  }
  U
}

#' Tucker Alternating Least Squares (HOOI)
#'
#' Computes a Tucker decomposition of a dense tensor with target multilinear
#' ranks via Higher-Order Orthogonal Iteration. Mirrors the MATLAB Tensor
#' Toolbox `tucker_als`.
#'
#' @param X A Tensor or array-like object.
#' @param ranks Integer vector of per-mode target ranks, or a scalar replicated
#'   across modes.
#' @param tol Convergence tolerance on change in fit (default `1e-4`).
#' @param maxiters Maximum number of HOOI sweeps (default `50`).
#' @param dimorder Integer permutation giving the order in which factor
#'   matrices are updated.
#' @param init Either `"nvecs"` (leading left singular vectors, default),
#'   `"random"` (random orthonormal), or a list of initial factor matrices.
#' @param printitn Print fit every `printitn` iterations (`0` to suppress).
#' @return A `TTensor`.
#' @examples
#' set.seed(1)
#' X <- tensor(array(runif(60), dim = c(3, 4, 5)))
#' T <- tucker_als(X, ranks = c(2, 3, 3), maxiters = 20)
#' @export
tucker_als <- function(X, ranks,
                       tol = 1e-4,
                       maxiters = 50L,
                       dimorder = NULL,
                       init = "nvecs",
                       printitn = 0L) {
  X <- as.tensor(X)
  if (!inherits(X, "Tensor")) {
    stop("X must be a Tensor or coercible to one.")
  }
  dims <- as.integer(X$dim())
  N <- length(dims)
  if (N < 2L) {
    stop("tucker_als requires a tensor with at least 2 modes.")
  }

  if (length(ranks) == 1L) ranks <- rep(as.integer(ranks), N)
  ranks <- as.integer(ranks)
  if (length(ranks) != N) {
    stop("ranks must have length 1 or ndims(X).")
  }
  if (any(ranks < 1L) || any(ranks > dims)) {
    stop("ranks must be between 1 and the corresponding mode size.")
  }

  maxiters <- as.integer(maxiters)
  printitn <- as.integer(printitn)

  if (is.null(dimorder)) {
    dimorder <- seq_len(N)
  }
  dimorder <- as.integer(dimorder)
  if (length(dimorder) != N || !setequal(dimorder, seq_len(N))) {
    stop("dimorder must be a permutation of 1:ndims(X).")
  }

  U <- .init_tucker_factors(X, ranks, dims, N, dimorder, init)

  normX <- fnorm(X)
  fit <- 0
  fit_prev <- 0
  core <- NULL

  for (iter in seq_len(maxiters)) {
    for (n in dimorder) {
      other_modes <- setdiff(seq_len(N), n)
      Utr <- lapply(other_modes, function(k) t(U[[k]]))
      Y <- ttm(X, Utr, mode = other_modes)
      Yn <- as.matrix(tenmat(Y, rdims = n))
      r_n <- ranks[n]
      max_r <- min(nrow(Yn), ncol(Yn))
      if (r_n > max_r) r_n <- max_r
      sv <- svd(Yn, nu = r_n, nv = 0)
      U[[n]] <- sv$u[, seq_len(r_n), drop = FALSE]
      core <- ttm(Y, t(U[[n]]), mode = n)
    }

    normG <- fnorm(core)
    normresidual <- sqrt(max(normX^2 - normG^2, 0))
    fit <- 1 - normresidual / normX
    fit_change <- abs(fit_prev - fit)

    if (printitn > 0L && (iter %% printitn == 0L || iter == 1L)) {
      message(sprintf(" Iter %2d: fit = %.6e, fitdelta = %.6e",
                      iter, fit, fit_change))
    }

    if (iter > 1L && fit_change < tol) {
      break
    }
    fit_prev <- fit
  }

  ttensor(core, U)
}
