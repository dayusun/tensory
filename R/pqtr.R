#' @include tensor_class.R tepls.R spgtr.R ttensor_class.R
NULL

# ---------------------------------------------------------------------------
# Linear quantile regression by the MM algorithm of Hunter & Lange (2000): the
# check loss is majorized by a weighted least-squares surrogate with weights
# 1 / (eps + |residual|), and eps is annealed towards zero. The reference
# MATLAB code (quantreg_unc.m) runs `fminunc` on the non-smooth loss from an
# OLS start; MM is the standard smooth-surrogate alternative and keeps the
# package free of a linear-programming dependency. Accuracy is set by the
# final eps (1e-8 here), which is far below the noise in any tensor fit.
# ---------------------------------------------------------------------------

.rq_fit <- function(X, y, tau, maxit = 200L, tol = 1e-8) {
  X <- as.matrix(X)
  b <- as.vector(.pinv(crossprod(X)) %*% crossprod(X, y)) # OLS start
  s <- (2 * tau - 1) * colSums(X)
  for (eps in 10^-(1:8)) {
    for (it in seq_len(maxit)) {
      r <- as.vector(y - X %*% b)
      w <- 1 / (eps + abs(r))
      # Stationary point of the surrogate: (X' W X) b = X' W y + (2 tau - 1) X'1.
      bnew <- as.vector(.pinv(crossprod(X, X * w)) %*%
                          (as.vector(crossprod(X, w * y)) + s))
      step <- max(abs(bnew - b))
      b <- bnew
      if (step < tol * (1 + max(abs(b)))) break
    }
  }
  b
}

# Check loss sum_i rho_tau(r_i), the objective quantile regression minimizes.
.check_loss <- function(r, tau) sum(r * (tau - (r < 0)))

# ---------------------------------------------------------------------------
# Core fit, working from an n x prod(dims) design matrix so that pqtr_cv() can
# reuse row subsets without rebuilding the predictor.
# ---------------------------------------------------------------------------

.pqtr_fit <- function(Xmat, dims, y, Z, tau, u, ridge, maxit, tol) {
  n <- nrow(Xmat)
  m <- length(dims)
  q <- if (is.null(Z)) 0L else ncol(Z)

  # Step 1. Working residual of the quantile regression on the nuisance
  # covariates alone. With no covariates this reduces to the tau-th sample
  # quantile of y, so R is tau - 1{y below the fitted tau-th quantile}.
  if (is.null(Z)) {
    Zbar <- numeric(0)
    Z0 <- matrix(1, n, 1)
    fit0 <- rep(stats::quantile(y, tau, names = FALSE), n)
  } else {
    Zbar <- colMeans(Z)
    Z0 <- cbind(1, sweep(Z, 2L, Zbar, `-`))
    fit0 <- as.vector(Z0 %*% .rq_fit(Z0, y, tau, maxit, tol))
  }
  R <- tau - (y < fit0)

  # Step 2. Partial quantile covariance and the per-mode signal matrices.
  Xbar <- colMeans(Xmat)
  Xt <- t(Xmat) - Xbar # prod(p) x n, one column per subject
  sig <- .pls_signal(Xt, R, dims, n, ridge)

  # Step 3. Reduced dimension per mode, by eigenvalue ratio unless given.
  if (is.null(u)) u <- .spgtr_auto_u(sig$U, dims, n, q)
  u <- as.integer(u)
  if (length(u) == 1L) u <- rep(u, m)
  if (length(u) != m) stop("u must have length 1 or ndims of the predictor.")
  if (any(u < 1L) || any(u > dims)) {
    stop("each u[k] must satisfy 1 <= u[k] <= p_k.")
  }

  # Step 4. Per-mode weight matrices by the PQTR deflation (pls_tensor_core.m),
  # which keeps the weights themselves orthonormal.
  W <- lapply(seq_len(m), function(k) {
    .tepls_simpls_mode(sig$U[[k]], sig$Sig[[k]], u[k], orth = "weights")
  })

  # Step 5. Latent scores T_i = X_i x_1 W_1' ... x_m W_m', as one chain of
  # tensor-times-matrix products over the stacked data.
  Xten <- Tensor$new(array(Xt, dim = c(dims, n)),
                     dims = as.integer(c(dims, n)), fast = TRUE)
  Tten <- ttm(Xten, W, mode = seq_len(m), transpose = TRUE)
  Tmat <- t(matrix(as.vector(Tten$as_array()), nrow = prod(u)))

  # Step 6. Quantile regression of y on the covariates and the latent scores.
  design <- cbind(Z0, Tmat)
  cf <- .rq_fit(design, y, tau, maxit, tol)
  alpha <- cf[1L]
  gamma <- if (q > 0L) cf[seq_len(q) + 1L] else numeric(0)
  Dvec <- cf[-seq_len(q + 1L)]

  # Step 7. Map the latent coefficients back to the predictor's shape.
  core <- Tensor$new(array(Dvec, dim = u), dims = u, fast = TRUE)
  Bten <- ttm(core, W, mode = seq_len(m))
  eta <- as.vector(design %*% cf)

  structure(list(
    coef = ttensor(core$clone_tensor(), W), core = core, W = W,
    alpha = alpha, gamma = gamma, bvec = as.vector(Bten$as_array()),
    scores = Tmat, dims = dims, u = u, tau = tau, q = q, n = n,
    Xbar = Xbar, Zbar = Zbar, Sig = sig$Sig, U = sig$U,
    fitted = eta, residuals = y - eta, y = y,
    loss = .check_loss(y - eta, tau)
  ), class = "pqtr")
}

# ---------------------------------------------------------------------------
# User-facing API
# ---------------------------------------------------------------------------

#' Partial Quantile Tensor Regression (PQTR)
#'
#' Fits a quantile regression in which each subject contributes a whole array
#' of measurements -- an image, a connectivity matrix, a space-by-time grid --
#' together with an optional handful of ordinary covariates. Instead of the
#' mean of the outcome, `pqtr()` models a chosen quantile of it, so it answers
#' questions such as "which parts of the image go with an unusually *low*
#' score?" rather than only "which parts go with the average score?".
#'
#' Ordinary quantile regression cannot be run on the flattened array, because
#' one array holds far more numbers than there are subjects. `pqtr()` solves
#' that with a partial-least-squares construction: each dimension of the array
#' is compressed to a few directions chosen for their association with the
#' quantile of interest, the quantile regression is fit on those, and the
#' answer is mapped back to the original array shape.
#'
#' @section What you get back:
#' A coefficient array of exactly the same shape as one subject's data,
#' available through `coef(fit)`. A large positive entry means "a high value at
#' this position pushes the `tau`-th quantile of the outcome up". Unlike
#' [spgtr()] there is no sparsity penalty, so every entry is non-zero; read the
#' array by magnitude rather than by which entries are exactly zero.
#'
#' @section How to use it (short version):
#' 1. Put your data in a list: `X[[i]]` is subject `i`'s matrix or array, all
#'    the same shape, and `y` is a numeric vector with one value per subject.
#' 2. Run `fit <- pqtr(X, y, tau = 0.5)` for the median, or another `tau` in
#'    `(0, 1)` for a different part of the outcome distribution.
#' 3. Fit several quantiles and compare the coefficient arrays: if they differ,
#'    the array affects the spread or shape of the outcome, not just its
#'    centre.
#' 4. Use [pqtr_cv()] if you would rather have the number of directions chosen
#'    by cross-validation than by the built-in eigenvalue-ratio rule.
#'
#' @section How it works (technical):
#' Writing `Q(tau)` for the fitted `tau`-th quantile of the outcome given the
#' covariates `Z` alone, the outcome enters only through the working residual
#' `tau - 1{y < Q(tau)}`, the subgradient of the check loss at the
#' covariate-only fit. Let `C` be the cross-covariance tensor between the
#' centered predictor and that residual, and `Sigma_k` the mode-`k` marginal
#' covariance of the predictor. The mode-`k` signal matrix is
#' `U_k = C_(k) (kron_{j != k} Sigma_j^-1) C_(k)'`, and the weight matrix
#' `W_k` collects `u[k]` directions obtained by repeatedly taking the leading
#' eigenvector of `U_k` and deflating with the oblique projector
#' `I - Sigma_k W (W' Sigma_k W)^-1 W'`. That projector leaves `W_k` with
#' orthonormal columns, which is the deflation used in the reference MATLAB
#' implementation; it differs from the SIMPLS deflation in [tepls()] and
#' [spgtr()], which instead makes the latent scores uncorrelated. The two agree
#' whenever `U_k` has rank one and span different subspaces otherwise.
#'
#' The predictor is then reduced to latent scores
#' `T_i = X_i x_1 W_1' ... x_m W_m'`, the linear quantile regression of `y` on
#' `(Z, vec(T))` is fit, and the coefficient array is reconstructed as
#' `B = D x_1 W_1 ... x_m W_m` from the latent coefficients `D`. Nothing in
#' this chain inverts a `prod(p) x prod(p)` matrix, which is what makes the
#' method scale to large arrays.
#'
#' @section Divergences from the reference implementation:
#' Ported from the MATLAB code accompanying Sun et al. (2024)
#' (<https://github.com/dayusun/PQTR>), with four deliberate differences.
#' (1) The inner quantile regressions are solved by the MM algorithm of Hunter
#' & Lange (2000) rather than by MATLAB's `fminunc`, which is a smooth solver
#' applied to a non-smooth objective; the MM surrogate is exactly matched to
#' the check loss. (2) The predictor is centered before the latent scores are
#' formed, so `alpha` is the intercept at the training-sample means; the
#' coefficient array `B` is unchanged by this. (3) Cross-validated selection of
#' the reduced dimension lives in [pqtr_cv()], which takes an explicit grid of
#' candidate dimensions instead of enumerating every combination. (4) The
#' eigenvalue-ratio rule searches at most five candidate dimensions per mode,
#' rather than `sqrt(n - q)` of them: past the rank of the signal matrix the
#' eigenvalues are noise, and the largest ratio among them otherwise wins and
#' returns a dimension far too big to estimate. This matches the rule used by
#' [spgtr()] and [tepls()].
#'
#' @param X The tensor predictor, in either of two forms: a list of `n` equally
#'   shaped `Tensor` objects, matrices, or arrays (one per subject), or a
#'   single array/`Tensor` of order `m + 1` whose **last** dimension indexes
#'   the subjects.
#' @param y The outcome, a numeric vector with one value per subject.
#' @param tau The quantile to model, a single number strictly between 0 and 1.
#'   `0.5` (default) is the median.
#' @param Z Optional `n x q` matrix or data frame of ordinary (non-array)
#'   covariates such as age or sex. These are never reduced.
#' @param u Number of directions kept per dimension of the array: an integer
#'   vector of length `m`, or a single number used for every dimension. Leave
#'   as `NULL` (default) to have each `u[k]` chosen by the eigenvalue-ratio
#'   rule of the reference implementation.
#' @param ridge Relative floor applied to covariance eigenvalues for numerical
#'   stability (default `1e-8`).
#' @param maxit,tol Iteration cap and convergence tolerance of the quantile
#'   regression solver.
#' @return An object of class `pqtr`, a list whose most useful elements are:
#'   \describe{
#'     \item{`coef`}{coefficient array as a [TTensor] with core `D` and weight
#'       matrices `W`; `coef(fit)` returns it and `as.tensor(coef(fit))`
#'       expands it to a dense [Tensor].}
#'     \item{`alpha`, `gamma`}{intercept and coefficients of `Z`, on the
#'       centered scale.}
#'     \item{`u`, `W`, `core`, `scores`}{reduced dimensions, weight matrices,
#'       latent coefficients, and the `n x prod(u)` matrix of latent scores.}
#'     \item{`fitted`, `residuals`, `loss`}{fitted conditional quantiles, their
#'       residuals, and the attained check loss.}
#'   }
#' @references
#' Sun, D., Zhang, X. and Zhang, S. (2024). Partial quantile tensor regression.
#' Journal of the American Statistical Association 120(551).
#'
#' Hunter, D. R. and Lange, K. (2000). Quantile regression via an MM algorithm.
#' Journal of Computational and Graphical Statistics 9(1), 60-77.
#' @seealso [pqtr_cv()] to choose `u` by cross-validation, [predict.pqtr()],
#'   and [spgtr()] / [tepls()] for the mean-regression counterparts.
#' @examples
#' # 150 subjects measured on an 8 x 6 grid; only the top-left corner drives
#' # the outcome, and it does so more strongly in the upper tail.
#' set.seed(1)
#' B <- outer(c(1.5, rep(0, 7)), c(1, rep(0, 5)))
#' X <- lapply(1:150, function(i) matrix(rnorm(48), 8, 6))
#' signal <- vapply(X, function(xi) sum(B * xi), numeric(1))
#' y <- signal + (1 + 0.6 * signal) * rnorm(150)
#'
#' med <- pqtr(X, y, tau = 0.5)
#' upper <- pqtr(X, y, tau = 0.9)
#' med
#'
#' # Coefficient arrays, same shape as one subject's data.
#' round(as.tensor(coef(med))$as_array()[1:3, 1:3], 2)
#' round(as.tensor(coef(upper))$as_array()[1:3, 1:3], 2)
#'
#' # Predicted conditional quantiles for new subjects.
#' head(predict(med, X[1:5]))
#' @export
pqtr <- function(X, y, tau = 0.5, Z = NULL, u = NULL, ridge = 1e-8,
                 maxit = 200L, tol = 1e-8) {
  if (length(tau) != 1L || !is.finite(tau) || tau <= 0 || tau >= 1) {
    stop("tau must be a single number strictly between 0 and 1.")
  }
  des <- .tepls_design(X)
  Z <- .spgtr_check_z(Z, des$n)
  if (!is.numeric(y) || !is.null(dim(y))) {
    stop("y must be a numeric vector, one value per observation in X.")
  }
  if (length(y) != des$n) {
    stop("y must have one element per observation in X.")
  }
  .pqtr_fit(des$Xmat, des$dims, y, Z, tau, u, ridge, maxit, tol)
}

#' Choose the Reduced Dimension of a Quantile Tensor Regression
#'
#' Runs [pqtr()] over a grid of candidate reduced dimensions, scores each one
#' by `nfolds`-fold cross-validated check loss, and returns the model refitted
#' on all subjects at the best value. Use this when the eigenvalue-ratio rule
#' built into [pqtr()] is not obviously right, for instance when the
#' coefficient array is not close to rank one.
#'
#' @details
#' The default grid keeps the same number of directions in every mode,
#' `u = rep(d, m)` for `d` from 1 up to the largest value for which the reduced
#' quantile regression stays estimable, capped at 12 as in the reference
#' implementation. That is a deliberate simplification: enumerating every
#' combination of per-mode dimensions costs `d^m` fits. Pass `u_grid`
#' explicitly, for instance
#' `u_grid = apply(expand.grid(1:2, 1:3), 1, as.integer, simplify = FALSE)`,
#' when the modes should be allowed to differ.
#'
#' @param X,y,tau,Z,ridge,maxit,tol As in [pqtr()].
#' @param u_grid Optional list of candidate dimension vectors, each of length
#'   `m` (a plain integer vector is read as one candidate per element, used in
#'   every mode). Leave `NULL` (default) for the automatic equal-dimension
#'   grid.
#' @param nfolds Number of cross-validation folds (default `5`).
#' @return The `pqtr` fit at the selected dimension, with three extra elements:
#'   `u_grid` (the candidates tried), `cv` (a data frame of `u` and mean
#'   out-of-fold check `loss`), and `u_min` (the chosen dimension vector).
#' @seealso [pqtr()]
#' @examples
#' set.seed(2)
#' B <- outer(c(2, rep(0, 5)), c(2, rep(0, 4)))
#' X <- lapply(1:120, function(i) matrix(rnorm(30), 6, 5))
#' y <- vapply(X, function(xi) sum(B * xi), numeric(1)) + rnorm(120)
#'
#' fit <- pqtr_cv(X, y, tau = 0.5, nfolds = 3)
#' fit$u_min
#' fit$cv
#' @export
pqtr_cv <- function(X, y, tau = 0.5, Z = NULL, u_grid = NULL, nfolds = 5L,
                    ridge = 1e-8, maxit = 200L, tol = 1e-8) {
  if (length(tau) != 1L || !is.finite(tau) || tau <= 0 || tau >= 1) {
    stop("tau must be a single number strictly between 0 and 1.")
  }
  des <- .tepls_design(X)
  n <- des$n
  dims <- des$dims
  m <- length(dims)
  Z <- .spgtr_check_z(Z, n)
  if (!is.numeric(y) || length(y) != n) {
    stop("y must be a numeric vector with one value per observation in X.")
  }
  nfolds <- as.integer(nfolds)
  if (nfolds < 2L || nfolds > n) {
    stop("nfolds must be between 2 and the number of observations.")
  }
  q <- if (is.null(Z)) 0L else ncol(Z)

  if (is.null(u_grid)) {
    dmax <- min(floor((n - 1 - q)^(1 / m)) - 1L, 12L, min(dims))
    if (dmax < 1L) dmax <- 1L
    u_grid <- lapply(seq_len(dmax), function(d) rep.int(as.integer(d), m))
  } else if (!is.list(u_grid)) {
    u_grid <- lapply(as.integer(u_grid), function(d) rep.int(d, m))
  }
  u_grid <- lapply(u_grid, function(uu) {
    uu <- as.integer(uu)
    if (length(uu) == 1L) uu <- rep.int(uu, m)
    if (length(uu) != m || any(uu < 1L) || any(uu > dims)) {
      stop("each u_grid entry must satisfy 1 <= u[k] <= p_k in every mode.")
    }
    uu
  })

  folds <- sample(rep(seq_len(nfolds), length.out = n))
  loss <- matrix(NA_real_, length(u_grid), nfolds)
  for (f in seq_len(nfolds)) {
    tr <- folds != f
    for (j in seq_along(u_grid)) {
      fit <- try(.pqtr_fit(des$Xmat[tr, , drop = FALSE], dims, y[tr],
                           if (is.null(Z)) NULL else Z[tr, , drop = FALSE],
                           tau, u_grid[[j]], ridge, maxit, tol), silent = TRUE)
      if (inherits(fit, "try-error")) next
      eta <- .pqtr_eta(fit, des$Xmat[!tr, , drop = FALSE],
                       if (is.null(Z)) NULL else Z[!tr, , drop = FALSE])
      loss[j, f] <- .check_loss(y[!tr] - eta, tau)
    }
  }

  cvm <- rowMeans(loss, na.rm = TRUE)
  if (all(is.na(cvm))) stop("Every cross-validation fit failed.")
  best <- u_grid[[which.min(cvm)]]
  out <- .pqtr_fit(des$Xmat, dims, y, Z, tau, best, ridge, maxit, tol)
  out$cv <- data.frame(u = vapply(u_grid, paste, character(1), collapse = " x "),
                       loss = cvm)
  out$u_grid <- u_grid
  out$u_min <- best
  out
}

# Fitted conditional quantile for a raw n x prod(dims) design matrix.
.pqtr_eta <- function(object, Xmat, Z) {
  eta <- as.vector(sweep(Xmat, 2L, object$Xbar, `-`) %*% object$bvec) +
    object$alpha
  if (length(object$gamma)) {
    if (is.null(Z)) stop("newZ is required: the fit used nuisance covariates.")
    Z <- as.matrix(Z)
    if (ncol(Z) != length(object$gamma) || nrow(Z) != nrow(Xmat)) {
      stop("newZ must have one row per new subject and the fitted columns.")
    }
    eta <- eta + as.vector(sweep(Z, 2L, object$Zbar, `-`) %*% object$gamma)
  }
  eta
}

#' Predict from a Partial Quantile Tensor Regression Fit
#'
#' @param object A fit from [pqtr()] or [pqtr_cv()].
#' @param newX New subjects, in either form accepted by [pqtr()]. If omitted,
#'   the fitted values for the subjects used in fitting are returned.
#' @param newZ New ordinary covariates, required when the fit used `Z`.
#' @param ... Unused.
#' @return A numeric vector of predicted conditional `tau`-th quantiles, one
#'   per subject.
#' @examples
#' set.seed(3)
#' X <- lapply(1:80, function(i) matrix(rnorm(20), 5, 4))
#' y <- vapply(X, function(xi) xi[1, 1], numeric(1)) + rnorm(80)
#' fit <- pqtr(X, y, tau = 0.5, u = c(1, 1))
#' predict(fit, X[1:5])
#' @export
predict.pqtr <- function(object, newX = NULL, newZ = NULL, ...) {
  if (is.null(newX)) return(object$fitted)
  des <- .tepls_design(newX)
  if (!identical(des$dims, object$dims)) {
    stop("newX dimensions must match the fitted predictor dimensions.")
  }
  .pqtr_eta(object, des$Xmat, newZ)
}

#' @rdname pqtr
#' @param object A fit from [pqtr()] or [pqtr_cv()].
#' @param ... Unused.
#' @export
coef.pqtr <- function(object, ...) {
  object$coef
}

#' @rdname pqtr
#' @param x A fit from [pqtr()] or [pqtr_cv()].
#' @export
print.pqtr <- function(x, ...) {
  cat("<pqtr: partial quantile tensor regression>\n")
  cat("Quantile (tau): ", x$tau, "\n")
  cat("Subjects:       ", x$n, "\n")
  cat("Array shape:    ", paste(x$dims, collapse = " x "), "\n")
  cat("Directions (u): ", paste(x$u, collapse = " "),
      sprintf("(%d latent %s)", prod(x$u),
              if (prod(x$u) == 1L) "score" else "scores"), "\n")
  cat("Covariates (Z): ", x$q, "\n")
  cat("Check loss:     ", format(x$loss, digits = 6), "\n")
  invisible(x)
}
