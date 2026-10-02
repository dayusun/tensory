#' @include tensor_class.R ktensor_class.R tensor_dense_methods.R
NULL

.pinv <- function(M, tol = NULL) {
  s <- svd(M)
  if (is.null(tol)) {
    tol <- max(dim(M)) * .Machine$double.eps * max(s$d, 0)
  }
  pos <- s$d > tol
  if (!any(pos)) {
    return(matrix(0, ncol(M), nrow(M)))
  }
  s$v[, pos, drop = FALSE] %*%
    ((1 / s$d[pos]) * t(s$u[, pos, drop = FALSE]))
}

.fixsigns_cp <- function(K) {
  U <- K$U
  lambda <- K$lambda
  N <- length(U)
  R <- length(lambda)
  if (R == 0 || N == 0) {
    return(K)
  }

  for (r in seq_len(R)) {
    neg_modes <- integer(0)
    for (n in seq_len(N)) {
      col <- U[[n]][, r]
      if (length(col) == 0L) next
      idx <- which.max(abs(col))
      if (col[idx] < 0) {
        neg_modes <- c(neg_modes, n)
      }
    }
    for (n in neg_modes) {
      U[[n]][, r] <- -U[[n]][, r]
    }
    if (length(neg_modes) %% 2L == 1L) {
      lambda[r] <- -lambda[r]
    }
  }

  ktensor(lambda, U)
}

.init_cp_factors <- function(X, R, dims, N, init) {
  if (is.list(init)) {
    if (length(init) != N) {
      stop("init list must have length equal to ndims(X).")
    }
    U <- lapply(init, as.matrix)
    for (n in seq_len(N)) {
      if (!identical(dim(U[[n]]), c(as.integer(dims[n]), as.integer(R)))) {
        stop(sprintf("init[[%d]] must have dimensions %d x %d.", n, dims[n], R))
      }
    }
    return(U)
  }

  if (!is.character(init)) {
    stop("init must be a character string or list of matrices.")
  }
  init <- match.arg(init, c("random", "nvecs"))

  if (init == "random") {
    return(lapply(dims, function(d) matrix(stats::rnorm(d * R), d, R)))
  }

  U <- vector("list", N)
  for (n in seq_len(N)) {
    if (dims[n] >= R) {
      U[[n]] <- nvecs(X, mode = n, r = R)
    } else {
      U[[n]] <- matrix(stats::rnorm(dims[n] * R), dims[n], R)
    }
  }
  U
}

# Split point s (modes [1, s] / [s+1, N]) for cp_als's dimension tree, or
# NULL when it does not apply: sparse input, fewer than 3 modes (the tree
# saves nothing), an update order other than 1:N or N:1 (each group must be
# updated contiguously), no compiled kernels, or
# options(tensory.cp_dimtree = FALSE). Balances the two groups' sizes, like
# mttkrps_cpp.
.cp_dimtree_split <- function(X, dims, dimorder) {
  N <- length(dims)
  if (N < 3L || inherits(X, "Sptensor") ||
      !isTRUE(getOption("tensory.cp_dimtree", TRUE)) ||
      !exists("mttkrp_partial_cpp", mode = "function")) {
    return(NULL)
  }
  if (!identical(dimorder, seq_len(N)) && !identical(dimorder, rev(seq_len(N)))) {
    return(NULL)
  }
  worst <- vapply(seq_len(N - 1L), function(c) {
    max(prod(dims[seq_len(c)]), prod(dims[-seq_len(c)]))
  }, numeric(1))
  which.min(worst)
}

# MTTKRPs for an ALS-type sweep (cp_als, cp_nmu) that updates the factors one
# mode at a time in `dimorder`. Returns list(start, mttkrp): call start() at
# the beginning of every sweep, then mttkrp(U, n) for each mode in order with
# the current factors. With a dimension tree (.cp_dimtree_split) the partial
# for a group is computed on entering it and reused for the group's other
# modes -- valid because during a contiguous run of one group's updates the
# other group's factors do not change. Otherwise it is mttkrp() per mode.
.mttkrp_sweeper <- function(X, dims, dimorder) {
  split <- .cp_dimtree_split(X, dims, dimorder)
  if (is.null(split)) {
    return(list(start = function() invisible(NULL),
                mttkrp = function(U, n) mttkrp(X, U, mode = n)))
  }
  Xd <- .dense_data(X)
  state <- new.env(parent = emptyenv())
  start <- function() {
    state$partial <- NULL
    state$left <- NA
    invisible(NULL)
  }
  start()
  list(
    start = start,
    mttkrp = function(U, n) {
      left <- n <= split
      if (!identical(state$left, left)) {
        state$partial <- mttkrp_partial_cpp(Xd, U, split, left)
        state$left <- left
      }
      mttkrp_finish_cpp(state$partial, U, dims, split, n)
    }
  )
}

# mttkrp(X, U, n) for every mode with the same factors (gradient-based CP
# fits). A dense tensor of order >= 3 goes through mttkrps(), which reads it
# twice in total instead of once per mode; sparse tensors (mttkrps would
# densify them), order 2, or options(tensory.cp_dimtree = FALSE) use
# mttkrp() per mode.
.mttkrp_all <- function(X, U) {
  N <- length(U)
  if (N >= 3L && !inherits(X, "Sptensor") && inherits(X, "Tensor") &&
      isTRUE(getOption("tensory.cp_dimtree", TRUE)) &&
      exists("mttkrps_cpp", mode = "function")) {
    return(mttkrps(X, U))
  }
  lapply(seq_len(N), function(n) mttkrp(X, U, mode = n))
}

# optim()'s L-BFGS-B asks for fn and gr separately, usually at the same
# point; computing value and gradient together (fg(v) -> list(value,
# gradient)) and remembering the last point halves the work.
.fg_cached <- function(fg) {
  last_v <- NULL
  last <- NULL
  get <- function(v) {
    if (!identical(v, last_v)) {
      last <<- fg(v)
      last_v <<- v
    }
    last
  }
  list(fn = function(v) get(v)$value, gr = function(v) get(v)$gradient)
}

#' CP Alternating Least Squares Decomposition
#'
#' Computes a rank-R canonical polyadic (CP) decomposition of a dense tensor by
#' alternating least squares, mirroring the behavior of the MATLAB Tensor
#' Toolbox `cp_als`.
#'
#' @param X A Tensor or array-like object.
#' @param R Target CP rank (positive integer).
#' @param tol Convergence tolerance on change in fit (default `1e-4`).
#' @param maxiters Maximum number of ALS sweeps (default `50`).
#' @param dimorder Integer permutation of `1:ndims(X)` giving the order in which
#'   factor matrices are updated. Defaults to `1:ndims(X)`.
#' @param init Either `"random"` (i.i.d. normal), `"nvecs"` (leading left
#'   singular vectors of the mode-n unfolding; falls back to random for modes
#'   where the mode size is smaller than `R`), or a list of initial factor
#'   matrices.
#' @param printitn Print fit every `printitn` iterations (`0` to suppress).
#' @param fixsigns Logical; if `TRUE`, resolve sign ambiguity of the returned
#'   components.
#' @return A `KTensor` giving the rank-R CP decomposition of `X`.
#' @examples
#' set.seed(1)
#' X <- tensor(array(runif(24), dim = c(2, 3, 4)))
#' K <- cp_als(X, R = 2, maxiters = 20)
#' @export
cp_als <- function(X, R,
                   tol = 1e-4,
                   maxiters = 50L,
                   dimorder = NULL,
                   init = "random",
                   printitn = 0L,
                   fixsigns = TRUE) {
  # Sparse tensors are handled natively: fnorm, mttkrp, and nvecs all have
  # Sptensor methods, so the ALS loop below never densifies.
  if (!inherits(X, "Sptensor")) {
    X <- as.tensor(X)
    if (!inherits(X, "Tensor")) {
      stop("X must be a Tensor or coercible to one.")
    }
  }
  R <- as.integer(R)
  if (length(R) != 1L || R < 1L) {
    stop("R must be a positive integer.")
  }
  maxiters <- as.integer(maxiters)
  printitn <- as.integer(printitn)

  dims <- as.integer(X$dim())
  N <- length(dims)
  if (N < 2L) {
    stop("cp_als requires a tensor with at least 2 modes.")
  }

  if (is.null(dimorder)) {
    dimorder <- seq_len(N)
  }
  dimorder <- as.integer(dimorder)
  if (length(dimorder) != N || !setequal(dimorder, seq_len(N))) {
    stop("dimorder must be a permutation of 1:ndims(X).")
  }

  U <- .init_cp_factors(X, R, dims, N, init)
  lambda <- rep(1.0, R)
  gram <- lapply(U, crossprod)

  normX <- fnorm(X)
  fit <- 0
  fit_prev <- 0

  n_last <- dimorder[N]

  # Dimension tree (dense X): two passes over X per sweep instead of N, same
  # V as mttkrp() up to rounding (see .mttkrp_sweeper).
  sweep_mttkrp <- .mttkrp_sweeper(X, dims, dimorder)

  for (iter in seq_len(maxiters)) {
    V_last <- NULL
    sweep_mttkrp$start()
    for (n in dimorder) {
      V <- sweep_mttkrp$mttkrp(U, n)

      Y <- matrix(1, R, R)
      for (k in setdiff(seq_len(N), n)) {
        Y <- Y * gram[[k]]
      }

      Unew <- V %*% .pinv(Y)

      if (iter == 1L) {
        lambda <- sqrt(colSums(Unew^2))
      } else {
        lambda <- apply(Unew, 2L, function(col) max(max(abs(col)), 1))
      }
      lambda[lambda == 0] <- 1

      Unew <- sweep(Unew, 2L, lambda, "/")
      U[[n]] <- Unew
      gram[[n]] <- crossprod(Unew)

      if (n == n_last) {
        V_last <- V
      }
    }

    H <- Reduce(`*`, gram)
    normP_sq <- sum(tcrossprod(lambda) * H)

    Ulast_scaled <- sweep(U[[n_last]], 2L, lambda, "*")
    inner <- sum(Ulast_scaled * V_last)

    normresidual <- sqrt(max(normX^2 + normP_sq - 2 * inner, 0))
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

  P <- ktensor(lambda, U)
  if (fixsigns) {
    P <- .fixsigns_cp(P)
  }
  P
}
