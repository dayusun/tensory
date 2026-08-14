#' @include tensor_class.R tensor_ttm.R tensor_dense_methods.R cp_decomposition.R
NULL

# Build the n x prod(dims) design matrix (row i = vec(X_i), mode-1 fastest)
# from either a list of same-shaped observations or a single order-(m+1)
# Tensor/array whose LAST mode indexes the observations.
.tepls_design <- function(X) {
  if (is.list(X) && !inherits(X, "Tensor")) {
    n <- length(X)
    if (n == 0) stop("X must contain at least one observation.")
    first <- .tensor_as_dense(X[[1]])$as_array()
    dims <- dim(as.array(first)); if (is.null(dims)) dims <- length(first)
    dims <- as.integer(dims)
    P <- prod(dims)
    Xmat <- t(matrix(vapply(X, function(xi) {
      a <- .tensor_as_dense(xi)$as_array()
      da <- dim(as.array(a)); if (is.null(da)) da <- length(a)
      if (!identical(as.integer(da), dims)) {
        stop("All observations in X must share the same dimensions.")
      }
      as.double(a)
    }, numeric(P)), nrow = P))
    return(list(Xmat = Xmat, dims = dims, n = n))
  }

  X <- .tensor_as_dense(X)
  full <- X$dim()
  if (length(full) < 2) {
    stop("A tensor X must have order >= 2 (last mode indexes observations).")
  }
  n <- full[length(full)]
  dims <- as.integer(full[-length(full)])
  # Data is stored mode-1-fastest with observations last, so reshaping the
  # flat vector to prod(dims) x n gives one column per observation.
  Xmat <- t(matrix(as.double(X$as_array()), nrow = prod(dims)))
  list(Xmat = Xmat, dims = dims, n = n)
}

# Per-mode marginal covariances, Zhang & Li (2017) eq. (1):
#   Sigma_k = (n prod_{j!=k} p_j)^{-1} sum_i X_{i(k)} X_{i(k)}^T,
# computed for every mode at once from the centered n x prod(p) design matrix.
.tepls_mode_covs <- function(Xc, dims, n) {
  m <- length(dims)
  Xarr <- array(t(Xc), dim = c(dims, n)) # p_1 x ... x p_m x n
  lapply(seq_len(m), function(k) {
    perm <- c(k, setdiff(seq_len(m), k), m + 1L)
    Ak <- matrix(aperm(Xarr, perm), nrow = dims[k]) # p_k x (rest * n)
    tcrossprod(Ak) / (n * prod(dims[-k]))
  })
}

# Inverse of a symmetric PD matrix, eigenvalues floored for stability.
.sym_inv <- function(M, ridge = 1e-10) .sym_pow(M, -1, ridge)

# SIMPLS envelope-basis estimation for one mode (Algorithm 4, steps 2-3).
# M is the (deflated) second-moment matrix C~ C~^T; Sig is Sigma_k. Returns a
# p_k x d matrix whose columns are the estimated factor directions w_{ks}.
.tepls_simpls_mode <- function(M, Sig, d) {
  pk <- nrow(M)
  W <- matrix(0, pk, d)
  Vbasis <- NULL # orthonormal basis of span(Sigma_k W_s)
  Q <- diag(pk)
  for (s in seq_len(d)) {
    Ms <- Q %*% M %*% Q
    Ms <- (Ms + t(Ms)) / 2
    w <- eigen(Ms, symmetric = TRUE)$vectors[, 1]
    W[, s] <- w

    a <- as.vector(Sig %*% w)
    if (!is.null(Vbasis)) {
      a <- a - Vbasis %*% (t(Vbasis) %*% a)
    }
    na <- sqrt(sum(a^2))
    if (na > 1e-10) {
      Vbasis <- cbind(Vbasis, a / na)
      Q <- diag(pk) - tcrossprod(Vbasis)
    }
  }
  W
}

#' Tensor Envelope Partial Least Squares Regression (TEPLS)
#'
#' Fits the tensor-predictor partial least squares regression of Zhang & Li
#' (2017). A tensor predictor `X` (order `m`) is reduced to a low-dimensional
#' latent tensor via per-mode envelope factor matrices estimated with a SIMPLS
#' iteration (their Algorithm 4), and the response is regressed on the latent
#' tensor. The coefficient tensor is reconstructed in the original predictor
#' space (their Lemma 2), giving the `B_PLS` estimator.
#'
#' @details
#' The predictor may be supplied either as a list of `n` `Tensor`/array
#' observations that share the same dimensions, or as a single
#' order-`(m + 1)` `Tensor`/array whose **last** mode indexes the `n`
#' observations. The response `Y` is a length-`n` numeric vector (scalar
#' response) or an `n x r` matrix (multivariate response).
#'
#' Each mode's factor matrix `W_k` has `u[k]` columns; `u` is the envelope
#' dimension per mode (recycled if a scalar). The mode-`k` marginal covariance
#' uses the moment estimator (their eq. 1), the mode-`k` cross-covariance is
#' standardized as in Algorithm 4, and the reduced regression of `Y` on the
#' latent tensor is fit by (regularized) least squares (their Step 6).
#'
#' The mode-`k` second-moment matrix maximized in the SIMPLS step,
#' `C_(k) (Sigma_Y^-1 kron ... kron Sigma_1^-1) C_(k)^T` (excluding
#' `Sigma_k`), is identical to the `U U^T` matrix in the reference
#' implementation `TEReg::TensPLS_fit`. Two deliberate divergences: (1) that
#' package estimates each mode's basis with an envelope (`EnvMU`) optimizer,
#' whereas this function uses the SIMPLS deflation of Algorithm 4 exactly;
#' (2) the estimated factor directions and the reduced-regression fit are both
#' invariant to the per-mode scale ambiguity of the separable covariance,
#' avoiding the need to pin the Kronecker scale.
#'
#' @param X Tensor predictor: a list of observations, or an order-`(m + 1)`
#'   tensor with observations in the last mode.
#' @param Y Response: numeric vector (length `n`) or `n x r` matrix.
#' @param u Envelope dimension per mode: an integer vector of length `m`, or a
#'   scalar recycled across modes. Each `u[k]` must satisfy
#'   `1 <= u[k] <= p_k`. The default `NULL` picks each `u[k]` from the largest
#'   consecutive eigenvalue ratio of that mode's signal matrix, the same rule
#'   [spgtr()] uses; supply `u` explicitly when you know the rank.
#' @param ridge Small ridge added when inverting covariance / normal-equation
#'   matrices for numerical stability (default `1e-8`).
#' @return An object of class `tepls`: a list with elements `coef` (the
#'   coefficient `Tensor`, order `m` for scalar response or order `m + 1` with
#'   a trailing response mode otherwise), `W` (list of factor matrices),
#'   `intercept`, `Xbar`, `dims`, `u`, `r`, and `fitted`.
#' @references Zhang, X. and Li, L. (2017). Tensor envelope partial
#'   least-squares regression. Technometrics 59(4), 426-436.
#' @seealso [predict.tepls()]
#' @examples
#' set.seed(1)
#' p <- c(8, 6)
#' B <- outer(c(1, rep(0, 7)), c(1, rep(0, 5))) # rank-1, envelope dim 1 per mode
#' X <- lapply(1:80, function(i) matrix(rnorm(prod(p)), p[1], p[2]))
#' y <- vapply(X, function(xi) sum(B * xi), numeric(1)) + rnorm(80, sd = 0.1)
#' fit <- tepls(X, y, u = c(1, 1))
#' tepls(X, y) # envelope dimensions chosen automatically
#' @export
tepls <- function(X, Y, u = NULL, ridge = 1e-8) {
  des <- .tepls_design(X)
  Xmat <- des$Xmat
  dims <- des$dims
  n <- des$n
  m <- length(dims)

  Y <- as.matrix(Y)
  if (nrow(Y) != n) {
    stop("Y must have one row (or element) per observation in X.")
  }
  r <- ncol(Y)

  # Center. Xt is prod(p) x n, one column per case: transposing first and
  # recycling Xbar down the columns centers in one pass, and it is the layout
  # the compiled covariance kernel wants.
  Xbar <- colMeans(Xmat)
  Xt <- t(Xmat) - Xbar
  Ybar <- colMeans(Y)
  Yc <- sweep(Y, 2L, Ybar, `-`)

  # Cross-covariance tensor C in R^{p_1 x ... x p_m x r}.
  Cmat <- Xt %*% Yc / n # prod(p) x r
  C <- Tensor$new(array(Cmat, dim = c(dims, r)), dims = as.integer(c(dims, r)),
                  fast = TRUE)

  # Mode covariances and response covariance (moment estimators).
  Sig <- .spgtr_mode_covs(Xt, dims, n)
  SigY <- crossprod(Yc) / n
  SigInv <- lapply(Sig, .sym_inv, ridge = ridge)
  SigYinv <- .sym_inv(SigY, ridge = ridge)

  # Per-mode signal matrices M_k = D_(k) C_(k)^T = C_(k) (Kron Sigma^-1) C_(k)^T,
  # where D is C standardized on every mode except k (and on the response mode).
  M <- lapply(seq_len(m), function(k) {
    other <- setdiff(seq_len(m), k)
    D <- ttm(C, c(SigInv[other], list(SigYinv)), mode = c(other, m + 1L))
    as.matrix(tenmat(D, rdims = k)) %*% t(as.matrix(tenmat(C, rdims = k)))
  })

  if (is.null(u)) u <- .spgtr_auto_u(M, dims, n, 0L)
  u <- as.integer(u)
  if (length(u) == 1L) u <- rep(u, m)
  if (length(u) != m) stop("u must have length 1 or ndims of the predictor.")
  if (any(u < 1L) || any(u > dims)) {
    stop("each u[k] must satisfy 1 <= u[k] <= p_k.")
  }

  # Per-mode SIMPLS envelope bases (Algorithm 4).
  W <- lapply(seq_len(m), function(k) .tepls_simpls_mode(M[[k]], Sig[[k]], u[k]))

  # Latent-space Kronecker map vec(T) = (W_m kron ... kron W_1)^T vec(X).
  Kfac <- W[[m]]
  if (m >= 2) {
    for (k in (m - 1):1) {
      Kfac <- kronecker(Kfac, W[[k]])
    }
  }

  # Reduce X to the latent scores and regress Y on them (Algorithm 4, Step 6).
  Tmat <- crossprod(Xt, Kfac) # n x prod(u)
  Psi <- .pinv(crossprod(Tmat) + diag(ridge, ncol(Tmat))) %*% crossprod(Tmat, Yc)

  # Map the latent coefficient back to predictor space.
  Bmat <- Kfac %*% Psi # prod(p) x r
  fitted <- crossprod(Xt, Bmat) + matrix(Ybar, n, r, byrow = TRUE)

  if (r == 1L) {
    Bt <- Tensor$new(array(Bmat, dim = dims), dims = dims, fast = TRUE)
    fitted <- as.vector(fitted)
  } else {
    Bt <- Tensor$new(array(Bmat, dim = c(dims, r)),
                     dims = as.integer(c(dims, r)), fast = TRUE)
  }

  structure(list(
    coef = Bt, W = W, intercept = Ybar, Xbar = Xbar,
    dims = dims, u = u, r = r, fitted = fitted
  ), class = "tepls")
}

#' Predict from a TEPLS Fit
#'
#' @param object A `tepls` object from [tepls()].
#' @param newX New predictors, in the same form accepted by [tepls()] (a list
#'   of observations or an order-`(m + 1)` tensor with observations in the last
#'   mode). If omitted, the fitted values are returned.
#' @param ... Unused.
#' @return A numeric vector of predictions (scalar response) or an
#'   `nnew x r` matrix (multivariate response).
#' @export
predict.tepls <- function(object, newX = NULL, ...) {
  if (is.null(newX)) {
    return(object$fitted)
  }
  des <- .tepls_design(newX)
  if (!identical(des$dims, object$dims)) {
    stop("newX dimensions must match the fitted predictor dimensions.")
  }
  Xc <- sweep(des$Xmat, 2L, object$Xbar, `-`)
  Bmat <- matrix(as.vector(object$coef$as_array()), ncol = object$r)
  pred <- Xc %*% Bmat + matrix(object$intercept, des$n, object$r, byrow = TRUE)
  if (object$r == 1L) as.vector(pred) else pred
}

#' @export
coef.tepls <- function(object, ...) {
  object$coef
}

#' @export
print.tepls <- function(x, ...) {
  cat("<tepls: tensor envelope PLS regression>\n")
  cat("Predictor dims: ", paste(x$dims, collapse = " x "), "\n")
  cat("Envelope dims (u): ", paste(x$u, collapse = " "), "\n")
  cat("Response dim (r): ", x$r, "\n")
  invisible(x)
}
