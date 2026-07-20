#' @include tensor_class.R symktensor_class.R tensor_dense_methods.R cp_variants.R
NULL

#' Symmetric CP Decomposition via Direct Optimization
#'
#' Fits a symmetric CP model `sum_r lambda_r * u_r^(o m)` to a symmetric
#' tensor by minimizing the Frobenius residual over the weights and the shared
#' factor matrix with L-BFGS-B, mirroring the MATLAB Tensor Toolbox `cp_sym`.
#'
#' @details
#' With `t_r = <X, u_r^(o m)>` and `c_rs = u_r . u_s`, the objective is
#' `||X||^2 - 2 sum_r lambda_r t_r + sum_rs lambda_r lambda_s c_rs^m`, whose
#' gradients are evaluated exactly using [ttsv()].
#'
#' @param X A symmetric Tensor (all modes the same size).
#' @param R Number of symmetric rank-one components.
#' @param init `"random"` or an `n x R` matrix of initial factors.
#' @param maxiters Maximum optimizer iterations (default `500`).
#' @param factr `optim` L-BFGS-B `factr` convergence parameter.
#' @param symmetrize Logical; if `TRUE`, symmetrize `X` first instead of
#'   requiring exact symmetry.
#' @param printitn If positive, print the optimizer trace.
#' @return A `SymKTensor`.
#' @examples
#' set.seed(1)
#' u <- matrix(rnorm(6), 3, 2)
#' X <- as.tensor(symktensor(c(1, 2), u, m = 3))
#' S <- cp_sym(X, R = 2)
#' @export
cp_sym <- function(X, R,
                   init = "random",
                   maxiters = 500L,
                   factr = 1e7,
                   symmetrize = FALSE,
                   printitn = 0L) {
  X <- .tensor_as_dense(X)
  dims <- X$dim()
  m <- length(dims)
  if (m < 2 || !all(dims == dims[1])) {
    stop("cp_sym requires a tensor with all modes of the same size.")
  }
  n <- dims[1]
  if (isTRUE(symmetrize)) {
    X <- tensory::symmetrize(X)
  } else if (!issymmetric(X, tol = 1e-8)) {
    stop("X must be symmetric; call with symmetrize = TRUE to symmetrize it.")
  }
  R <- as.integer(R)
  if (length(R) != 1L || R < 1L) {
    stop("R must be a positive integer.")
  }

  if (is.matrix(init)) {
    if (!identical(dim(init), c(n, R))) {
      stop("init matrix must be n x R.")
    }
    U0 <- init
  } else {
    U0 <- matrix(stats::rnorm(n * R), n, R)
  }
  par0 <- c(rep(1, R), as.vector(U0))

  normX2 <- fnorm(X)^2

  fg <- function(par) {
    lambda <- par[seq_len(R)]
    U <- matrix(par[-seq_len(R)], n, R)

    tvals <- numeric(R)
    G <- matrix(0, n, R)
    for (r in seq_len(R)) {
      tvals[r] <- ttsv(X, U[, r], 0)
      G[, r] <- as.numeric(ttsv(X, U[, r], -1))
    }
    C <- crossprod(U)
    Cm <- C^m
    Cm1 <- C^(m - 1)

    f <- normX2 - 2 * sum(lambda * tvals) + sum(tcrossprod(lambda) * Cm)

    glambda <- -2 * tvals + 2 * as.vector(Cm %*% lambda)
    gU <- matrix(0, n, R)
    for (r in seq_len(R)) {
      gU[, r] <- -2 * lambda[r] * m * G[, r] +
        2 * m * lambda[r] * as.vector(U %*% (lambda * Cm1[, r]))
    }
    list(value = f, gradient = c(glambda, as.vector(gU)))
  }

  res <- stats::optim(
    par = par0,
    fn = function(p) fg(p)$value,
    gr = function(p) fg(p)$gradient,
    method = "L-BFGS-B",
    control = list(maxit = as.integer(maxiters), factr = factr,
                   trace = as.integer(printitn > 0))
  )

  lambda <- res$par[seq_len(R)]
  U <- matrix(res$par[-seq_len(R)], n, R)

  # Normalize columns, absorbing magnitudes (and sign parity) into lambda.
  nrm <- sqrt(colSums(U^2))
  nrm[nrm == 0] <- 1
  U <- sweep(U, 2L, nrm, `/`)
  lambda <- lambda * nrm^m
  ord <- order(abs(lambda), decreasing = TRUE)

  symktensor(lambda[ord], U[, ord, drop = FALSE], m)
}

#' Symmetric Tucker Decomposition
#'
#' Computes a Tucker decomposition with a single orthonormal factor matrix
#' shared by every mode via higher-order orthogonal iteration on the
#' symmetric subspace, mirroring the MATLAB Tensor Toolbox `tucker_sym`.
#'
#' @param X A symmetric Tensor (all modes the same size).
#' @param r Target subspace rank.
#' @param tol Convergence tolerance on change in core norm (default `1e-6`).
#' @param maxiters Maximum number of iterations (default `100`).
#' @param init `"nvecs"` (default) or an initial `n x r` matrix with
#'   orthonormal columns.
#' @param symmetrize Logical; if `TRUE`, symmetrize `X` first.
#' @return A `TTensor` whose factor matrices are all identical.
#' @examples
#' set.seed(1)
#' X <- symmetrize(tensor(array(rnorm(27), dim = c(3, 3, 3))))
#' T <- tucker_sym(X, r = 2)
#' @export
tucker_sym <- function(X, r,
                       tol = 1e-6,
                       maxiters = 100L,
                       init = "nvecs",
                       symmetrize = FALSE) {
  X <- .tensor_as_dense(X)
  dims <- X$dim()
  m <- length(dims)
  if (m < 2 || !all(dims == dims[1])) {
    stop("tucker_sym requires a tensor with all modes of the same size.")
  }
  n <- dims[1]
  if (isTRUE(symmetrize)) {
    X <- tensory::symmetrize(X)
  } else if (!issymmetric(X, tol = 1e-8)) {
    stop("X must be symmetric; call with symmetrize = TRUE to symmetrize it.")
  }
  r <- as.integer(r)
  if (length(r) != 1L || r < 1L || r > n) {
    stop("r must be a single integer between 1 and the mode size.")
  }

  if (is.matrix(init)) {
    if (!identical(dim(init), c(n, r))) {
      stop("init matrix must be n x r.")
    }
    U <- init
  } else {
    U <- nvecs(X, mode = 1L, r = r)
  }

  norm_prev <- -Inf
  for (iter in seq_len(maxiters)) {
    Y <- ttm(X, rep(list(t(U)), m - 1L), mode = seq_len(m)[-1L])
    Yn <- as.matrix(tenmat(Y, rdims = 1L))
    sv <- svd(Yn, nu = r, nv = 0)
    U <- sv$u[, seq_len(r), drop = FALSE]

    norm_core <- sqrt(sum(sv$d[seq_len(r)]^2))
    if (abs(norm_core - norm_prev) < tol) break
    norm_prev <- norm_core
  }

  core <- ttm(X, rep(list(t(U)), m), mode = seq_len(m))
  ttensor(core, rep(list(U), m))
}
