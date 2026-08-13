#' @include tensor_class.R tepls.R ttensor_class.R cp_decomposition.R
NULL

# ---------------------------------------------------------------------------
# Numerical helpers. The two hot kernels (mode covariances, manifold solver)
# have C++ implementations in src/tensor_spgtr.cpp; the R versions below are
# the reference fallbacks and must stay behaviorally identical.
# ---------------------------------------------------------------------------

# Symmetric matrix power via eigendecomposition, eigenvalues floored at
# `ridge` * max eigenvalue so near-singular covariances stay invertible.
.sym_pow <- function(M, pow, ridge = 1e-10) {
  M <- (M + t(M)) / 2
  e <- eigen(M, symmetric = TRUE)
  d <- pmax(e$values, ridge * max(e$values, 1))
  e$vectors %*% (d^pow * t(e$vectors))
}

# Mode-wise marginal covariances from the P x n (predictor-by-observation)
# centered data matrix.
.spgtr_mode_covs <- function(Xt, dims, n) {
  if (exists("spgtr_mode_covs_cpp", mode = "function")) {
    return(spgtr_mode_covs_cpp(Xt, as.integer(dims)))
  }
  .tepls_mode_covs(t(Xt), dims, n)
}

# Envelope objective of Cook & Zhang (2016) and its Euclidean gradient:
#   f(G) = log|G' M G| + log|G' (M + U)^-1 G|,  G a p x d semi-orthogonal basis.
.env_fg <- function(G, M, MUinv) {
  A <- crossprod(G, M %*% G)
  A <- (A + t(A)) / 2
  B <- crossprod(G, MUinv %*% G)
  B <- (B + t(B)) / 2
  ea <- eigen(A, symmetric = TRUE, only.values = TRUE)$values
  eb <- eigen(B, symmetric = TRUE, only.values = TRUE)$values
  list(
    f = sum(log(ea[ea > 0])) + sum(log(eb[eb > 0])),
    grad = 2 * (M %*% G %*% .pinv(A)) + 2 * (MUinv %*% G %*% .pinv(B))
  )
}

# Gradient projected onto the Stiefel tangent space, including the multiplier
# contribution of the row-wise L2,1 penalty (Xiao, Liu & Yuan 2021).
.env_pgrad <- function(G, grad, gam) {
  lam <- crossprod(G, grad)
  lam <- (lam + t(lam)) / 2
  if (any(gam > 0)) {
    s <- sqrt(gam) / (1e-14 + sqrt(sqrt(rowSums(G^2))))
    lam <- lam + crossprod(G * s)
  }
  grad - G %*% lam
}

# Sequential linearized proximal gradient on the Stiefel manifold with
# alternating BB step sizes; `gam` is the per-row penalty weight vector.
# Retraction is the polar factor G (G'G)^-1/2, a right multiplication, so rows
# zeroed by the proximal step stay zero.
.env_slpg_r <- function(G, M, MUinv, gam, maxit = 500L, tol = 1e-8,
                        ridge = 1e-10) {
  p <- nrow(G)
  fg <- .env_fg(G, M, MUinv)
  beta <- 0.01 * norm(fg$grad, "F")
  Gr <- .env_pgrad(G, fg$grad, gam)
  step <- 1 / (norm(G, "F") / 2 * norm(Gr, "F") + p * beta)
  if (!is.finite(step) || step <= 0) step <- 1e-3
  stalled <- 0L

  for (it in seq_len(maxit)) {
    Gp <- G
    Grp <- Gr
    G <- .prox21(G - step * Gr, gam * step)
    GG <- crossprod(G)
    # A rank-deficient iterate cannot be retracted: back off instead of
    # refilling the rows the penalty just removed.
    if (min(eigen(GG, symmetric = TRUE, only.values = TRUE)$values) <= 1e-10) {
      G <- Gp
      step <- step / 2
      stalled <- stalled + 1L
      if (stalled > 5L) break
      next
    }
    stalled <- 0L
    G <- G %*% .sym_pow(GG, -0.5, ridge)

    fg <- .env_fg(G, M, MUinv)
    Gr <- .env_pgrad(G, fg$grad, gam)
    D <- G - Gp
    if (norm(D, "F") / step < tol) break
    Yd <- Gr - Grp
    sdy <- sum(Yd * D)
    step <- if (it %% 2 == 0) norm(D, "F")^2 / sdy else sdy / norm(Yd, "F")^2
    step <- if (is.finite(step)) min(max(abs(step), 1e-12), 1000) else 1e-3
  }
  G
}

.env_slpg <- function(G, M, MUinv, gam, maxit = 500L, tol = 1e-8,
                      ridge = 1e-10) {
  if (exists("spgtr_slpg_cpp", mode = "function")) {
    return(spgtr_slpg_cpp(G, M, MUinv, gam, as.integer(maxit), tol, ridge))
  }
  .env_slpg_r(G, M, MUinv, gam, maxit, tol, ridge)
}

# Row-wise group soft threshold (proximal operator of the L2,1 norm).
.prox21 <- function(G, thr) {
  rn <- sqrt(rowSums(G^2))
  (1 - thr / pmax(rn, thr, .Machine$double.xmin)) * G
}

# Best of the SIMPLS basis and four eigenvector starts (Cook & Zhang's MU-init).
.env_init <- function(M, U, d, gam, Wpls) {
  MU <- M + U
  cands <- list(Wpls)
  for (A in list(MU, M)) {
    V <- eigen((A + t(A)) / 2, symmetric = TRUE)$vectors
    S <- .sym_pow(A, -0.5)
    for (B in list(U, S %*% U %*% S)) {
      sc <- colSums(V * (B %*% V))
      keep <- order(sc, decreasing = TRUE)[seq_len(d)]
      cands[[length(cands) + 1L]] <- V[, keep, drop = FALSE]
    }
  }
  MUinv <- .sym_pow(MU, -1)
  obj <- vapply(cands, function(G) {
    .env_fg(G, M, MUinv)$f + sum(sqrt(rowSums(G^2)) * gam)
  }, numeric(1))
  cands[[which.min(obj)]]
}

# Eigenvalue-ratio rule for the per-mode envelope dimension (MATLAB
# pls_tensor_core_auto): u_k maximizes lambda_s / lambda_{s+1} of U_k.
.spgtr_auto_u <- function(U, dims, n, q) {
  m <- length(dims)
  upper <- max(2L, min(floor((n - 1 - q)^(1 / m)), 5L))
  vapply(seq_len(m), function(k) {
    K <- min(upper, dims[k])
    if (K < 2L) return(1L)
    ev <- eigen(U[[k]], symmetric = TRUE, only.values = TRUE)$values[seq_len(K)]
    ev <- pmax(ev, .Machine$double.eps)
    which.max(ev[-K] / ev[-1L])
  }, integer(1))
}

# ---------------------------------------------------------------------------
# Stage 1: everything that does not depend on the penalty. Splitting the fit
# here lets a whole lambda path reuse one set of covariances and SIMPLS bases.
# ---------------------------------------------------------------------------

.spgtr_prepare <- function(Xmat, dims, y, Z, family, u, ridge) {
  n <- nrow(Xmat)
  m <- length(dims)
  q <- if (is.null(Z)) 0L else ncol(Z)

  # Working residual from the GLM of y on Z alone (intercept only when Z is
  # NULL, which reduces to y - mean(y) for canonical links).
  Z0 <- if (is.null(Z)) matrix(1, n, 1) else cbind(1, Z)
  R <- y - stats::glm.fit(Z0, y, family = family)$fitted.values
  sdR <- sqrt(mean((R - mean(R))^2))
  if (sdR < .Machine$double.eps) stop("The working residual has zero variance.")

  # prod(p) x n, one column per case. Transposing first and recycling Xbar down
  # the columns centers in one pass, half the cost of sweep-then-transpose.
  Xbar <- colMeans(Xmat)
  Xt <- t(Xmat) - Xbar
  Sig <- .spgtr_mode_covs(Xt, dims, n)

  # Cross-covariance tensor between the centered predictor and the residual.
  C <- Tensor$new(array(as.vector(Xt %*% R) / n / sdR, dim = dims),
                  dims = dims, fast = TRUE)
  Sig_isqrt <- lapply(Sig, .sym_pow, pow = -0.5, ridge = ridge)

  # U_k = C_(k) (kron_{j != k} Sigma_j^-1) C_(k)', formed as C_0k C_0k' so it
  # is positive semi-definite by construction.
  U <- vector("list", m)
  for (k in seq_len(m)) {
    other <- setdiff(seq_len(m), k)
    Ck <- if (m == 1L) {
      as.matrix(C$as_array())
    } else {
      as.matrix(tenmat(ttm(C, Sig_isqrt[other], mode = other), rdims = k))
    }
    U[[k]] <- tcrossprod(Ck)
  }

  if (is.null(u)) u <- .spgtr_auto_u(U, dims, n, q)
  u <- as.integer(u)
  if (length(u) == 1L) u <- rep(u, m)
  if (length(u) != m) stop("u must have length 1 or ndims of the predictor.")
  if (any(u < 1L) || any(u > dims)) {
    stop("each u[k] must satisfy 1 <= u[k] <= p_k.")
  }

  Wpls <- lapply(seq_len(m), function(k) {
    .tepls_simpls_mode(U[[k]], Sig[[k]], u[k])
  })
  list(
    Xt = Xt, Xbar = Xbar, dims = dims, n = n, m = m, q = q, y = y, Z = Z,
    Z0 = Z0, family = family, Sig = Sig, U = U, Wpls = Wpls, u = u,
    ridge = ridge,
    # Adaptive weights from the unpenalized SIMPLS basis, as in the MATLAB code.
    weights = lapply(Wpls, function(W) pmin(1 / sqrt(rowSums(W^2)), 1e6))
  )
}

# Stage 2: estimate the bases at one penalty value and fit the score-level GLM.
.spgtr_solve <- function(prep, lambda, basis, maxit, tol, Winit = NULL) {
  m <- prep$m
  u <- prep$u
  W <- vector("list", m)
  for (k in seq_len(m)) {
    if (basis == "simpls") {
      W[[k]] <- prep$Wpls[[k]]
      next
    }
    gam <- lambda * prep$weights[[k]]
    MUinv <- .sym_pow(prep$Sig[[k]] + prep$U[[k]], -1, ridge = prep$ridge)
    G0 <- if (is.null(Winit)) {
      .env_init(prep$Sig[[k]], prep$U[[k]], u[k], gam, prep$Wpls[[k]])
    } else {
      Winit[[k]]
    }
    W[[k]] <- .env_slpg(G0, prep$Sig[[k]], MUinv, gam, maxit, tol, prep$ridge)
  }

  # Latent scores T_i = X_i x_1 W_1' ... x_m W_m', done as one tensor-times-
  # matrix chain over the stacked data rather than through a Kronecker product.
  Xten <- Tensor$new(array(prep$Xt, dim = c(prep$dims, prep$n)),
                     dims = as.integer(c(prep$dims, prep$n)), fast = TRUE)
  Tten <- ttm(Xten, W, mode = seq_len(m), transpose = TRUE)
  Tmat <- t(matrix(as.vector(Tten$as_array()), nrow = prod(u)))

  fit <- stats::glm.fit(cbind(prep$Z0, Tmat), prep$y, family = prep$family)
  cf <- fit$coefficients
  cf[is.na(cf)] <- 0
  alpha <- unname(cf[1L])
  gamma <- if (prep$q > 0L) unname(cf[seq_len(prep$q) + 1L]) else numeric(0)
  Dvec <- unname(cf[-seq_len(prep$q + 1L)])

  core <- Tensor$new(array(Dvec, dim = u), dims = u, fast = TRUE)
  Bten <- ttm(core, W, mode = seq_len(m))
  keep <- lapply(W, function(w) which(rowSums(w^2) > 0))

  structure(list(
    coef = ttensor(core$clone_tensor(), W), core = core, W = W,
    weights = prep$weights, Sig = prep$Sig, U = prep$U,
    alpha = alpha, gamma = gamma, bvec = as.vector(Bten$as_array()),
    scores = Tmat, dims = prep$dims, u = u, lambda = lambda, basis = basis,
    family = prep$family, Xbar = prep$Xbar, n = prep$n,
    selected = keep, nonzero = vapply(keep, length, integer(1)),
    linear.predictors = unname(fit$linear.predictors),
    fitted = unname(fit$fitted.values), y = prep$y,
    deviance = fit$deviance, null.deviance = fit$null.deviance,
    converged = fit$converged
  ), class = "spgtr")
}

# ---------------------------------------------------------------------------
# User-facing API
# ---------------------------------------------------------------------------

#' Sparse Penalized Generalized Tensor Regression (SPGTR)
#'
#' Fits a regression model in which each subject contributes a whole array of
#' measurements -- an image, a connectivity matrix, a spectrogram, a
#' space-by-time grid -- and the outcome is a single number per subject, such
#' as a yes/no diagnosis, a count, or a continuous score. Ordinary regression
#' cannot be used directly here because one array holds far more numbers than
#' there are subjects. `spgtr()` solves that by compressing every dimension of
#' the array down to a handful of informative directions and fitting the
#' generalized linear model on those, then translating the answer back to the
#' original array shape so it can be read off like a picture.
#'
#' @section What you get back:
#' The main output is a coefficient array of exactly the same shape as one
#' subject's data, available through `coef(fit)`. A large positive entry means
#' "a high value at this position pushes the outcome up"; a zero entry means
#' the position was not used. With `lambda > 0` whole rows, columns, or slices
#' are set to zero, so the fit also tells you which parts of the array matter
#' at all (see `fit$selected` and [summary.spgtr()]).
#'
#' @section How to use it (short version):
#' 1. Put your data in a list: `X[[i]]` is subject `i`'s matrix or array, all
#'    the same shape. `y` is a vector with one entry per subject.
#' 2. Run `fit <- spgtr(X, y)` for a yes/no outcome, or add
#'    `family = poisson()` / `family = gaussian()` for counts / continuous
#'    outcomes.
#' 3. Look at `summary(fit)`, view `coef(fit)`, and predict new subjects with
#'    `predict(fit, newX)`.
#' 4. To also *select* which parts of the array matter, use [spgtr_cv()],
#'    which picks the amount of sparsity for you by cross-validation.
#'
#' @section How it works (technical):
#' The outcome enters through the working residual `r = y - mu_0`, where
#' `mu_0` is the fitted mean of the GLM of `y` on the nuisance covariates `Z`
#' alone (the intercept only, when `Z` is `NULL`). Writing `Sigma_k` for the
#' mode-`k` marginal covariance of the centered predictor and `C` for its
#' cross-covariance with `r`, the mode-`k` signal matrix is
#' `U_k = C_(k) (kron_{j != k} Sigma_j^-1) C_(k)'`.
#'
#' With `basis = "simpls"` the factor matrix `W_k` collects the first `u[k]`
#' SIMPLS directions of `(U_k, Sigma_k)`, exactly as in [tepls()]. With
#' `basis = "envelope"` (the default) that basis is refined by minimizing the
#' envelope objective `log|W' Sigma_k W| + log|W' (Sigma_k + U_k)^-1 W|` over
#' the Stiefel manifold. A positive `lambda` adds the adaptively weighted
#' row-wise penalty `lambda * sum_i w_ki ||W_k[i, ]||_2`, whose proximal
#' operator zeroes entire rows of `W_k` and therefore drops the corresponding
#' mode-`k` slices of the predictor from the model. Weights `w_ki` are the
#' inverse row norms of the unpenalized SIMPLS basis.
#'
#' The penalized problem is solved by the sequential linearized proximal
#' gradient method of Xiao, Liu & Yuan (2021) with alternating Barzilai-Borwein
#' step sizes. Its retraction is the polar factor `W (W'W)^-1/2`, a
#' right-multiplication, which is what allows the zero rows created by the
#' proximal step to survive orthonormalization.
#'
#' Finally the predictor is reduced to latent scores `T_i = X_i x_1 W_1' ...
#' x_m W_m'`, the GLM of `y` on `(Z, vec(T))` is fit by [stats::glm.fit()], and
#' the coefficient array is reconstructed as `B = D x_1 W_1 ... x_m W_m` from
#' the latent coefficients `D`.
#'
#' @section Speed:
#' The mode-wise covariances and the manifold solver are compiled kernels
#' (`spgtr_mode_covs_cpp`, `spgtr_slpg_cpp`); the score computation reuses the
#' package's compiled [ttm()]. Reference implementations in R are used
#' automatically if the package was installed without compilation, and the two
#' paths agree to numerical tolerance.
#'
#' @param X The tensor predictor, in either of two forms: a list of `n`
#'   equally shaped `Tensor` objects, matrices, or arrays (one per subject), or
#'   a single array/`Tensor` of order `m + 1` whose **last** dimension indexes
#'   the subjects.
#' @param y The outcome, with one entry per subject. For `binomial()` this may
#'   be a 0/1 numeric vector, a logical vector, or a two-level factor (the
#'   first level is treated as the reference).
#' @param u Number of directions kept per dimension of the array: an integer
#'   vector of length `m`, or a single number used for every dimension. Leave
#'   as `NULL` (default) to have each `u[k]` chosen automatically by an
#'   eigenvalue-ratio rule. Larger values fit more flexible models; `1` or `2`
#'   per mode is typical.
#' @param Z Optional `n x q` matrix or data frame of ordinary (non-array)
#'   covariates such as age or sex. These are never penalized or reduced.
#' @param family The outcome type, as a name, a family function, or a family
#'   object: `binomial()` (default) for yes/no, `poisson()` for counts,
#'   `gaussian()` for continuous outcomes.
#' @param basis `"envelope"` (default) refines the SIMPLS directions by
#'   envelope optimization and is required for sparsity; `"simpls"` uses the
#'   closed-form SIMPLS directions alone, which is faster and needs no
#'   iteration.
#' @param lambda Amount of sparsity, `>= 0`. `0` (default) keeps every row of
#'   every factor matrix; larger values remove more slices of the array. Use
#'   [spgtr_cv()] if you do not want to pick this by hand. Ignored when
#'   `basis = "simpls"`.
#' @param maxit,tol Iteration cap and stationarity tolerance of the manifold
#'   solver.
#' @param ridge Relative floor applied to covariance eigenvalues for numerical
#'   stability (default `1e-8`).
#' @return An object of class `spgtr`, a list whose most useful elements are:
#'   \describe{
#'     \item{`coef`}{coefficient array as a [TTensor] with core `D` and factor
#'       matrices `W`; `coef(fit)` returns it and `as.tensor(coef(fit))`
#'       expands it to a dense [Tensor].}
#'     \item{`alpha`, `gamma`}{intercept and coefficients of `Z`.}
#'     \item{`selected`, `nonzero`}{indices, and counts, of the retained rows
#'       in each dimension.}
#'     \item{`W`, `core`, `scores`}{factor matrices, latent coefficients, and
#'       the `n x prod(u)` matrix of latent scores.}
#'     \item{`fitted`, `linear.predictors`, `deviance`, `null.deviance`}{as in
#'       a [stats::glm()] fit.}
#'   }
#' @references
#' Zhang, X. and Li, L. (2017). Tensor envelope partial least-squares
#' regression. Technometrics 59(4), 426-436.
#'
#' Cook, R. D. and Zhang, X. (2016). Algorithms for envelope estimation.
#' Journal of Computational and Graphical Statistics 25(1), 284-300.
#'
#' Xiao, N., Liu, X. and Yuan, Y. (2021). Exact penalty function for L21 norm
#' minimization over the Stiefel manifold. SIAM Journal on Optimization 31(4),
#' 3097-3126.
#' @seealso [spgtr_cv()] to choose `lambda`, [predict.spgtr()],
#'   [summary.spgtr()], and [tepls()] for the continuous-response version.
#' @examples
#' # 120 subjects, each measured on an 8 x 6 grid; only the top-left corner
#' # of the grid actually drives the yes/no outcome.
#' set.seed(1)
#' B <- outer(c(1.5, rep(0, 7)), c(1.5, rep(0, 5)))
#' X <- lapply(1:120, function(i) matrix(rnorm(48), 8, 6))
#' eta <- vapply(X, function(xi) sum(B * xi), numeric(1))
#' y <- rbinom(120, 1, 1 / (1 + exp(-eta)))
#'
#' fit <- spgtr(X, y, u = c(1, 1))
#' summary(fit)
#'
#' # Coefficient array, same shape as one subject's data.
#' round(as.tensor(coef(fit))$as_array(), 2)
#'
#' # Predicted probabilities and labels.
#' head(predict(fit, type = "response"))
#' head(predict(fit, type = "class"))
#'
#' # With covariates, and with a count outcome.
#' Z <- cbind(age = rnorm(120))
#' fit_z <- spgtr(X, y, u = c(1, 1), Z = Z)
#' counts <- rpois(120, exp(eta / 2))
#' fit_p <- spgtr(X, counts, u = c(1, 1), family = poisson())
#' @export
spgtr <- function(X, y, u = NULL, Z = NULL, family = stats::binomial(),
                  basis = c("envelope", "simpls"), lambda = 0,
                  maxit = 500L, tol = 1e-8, ridge = 1e-8) {
  basis <- match.arg(basis)
  family <- .spgtr_family(family)
  if (length(lambda) != 1L || !is.finite(lambda) || lambda < 0) {
    stop("lambda must be a single non-negative number.")
  }
  des <- .tepls_design(X)
  Z <- .spgtr_check_z(Z, des$n)
  y <- .spgtr_check_y(y, des$n)

  prep <- .spgtr_prepare(des$Xmat, des$dims, y, Z, family, u, ridge)
  .spgtr_solve(prep, lambda, basis, maxit, tol)
}

#' Choose the Sparsity of a Tensor Regression by Cross-Validation
#'
#' Runs [spgtr()] over a range of sparsity levels, scores each one by
#' `nfolds`-fold cross-validation, and returns the model refitted at the best
#' value. Use this when you want the method to decide by itself how much of the
#' array to keep.
#'
#' @details
#' Folds are drawn at random once; within each fold the covariances, SIMPLS
#' bases, and adaptive weights are computed a single time and reused across the
#' whole `lambda` path, and each fit is warm-started from the previous (less
#' sparse) solution. Held-out fits are scored by the deviance of the chosen
#' `family`, which is the residual sum of squares for `gaussian()` and twice
#' the negative log-likelihood (up to a constant) otherwise; smaller is better.
#'
#' If `lambda` is not supplied, the grid runs from `lambda_max / lambda_ratio`
#' up to `lambda_max`, the value at which the first proximal step would zero
#' every row of every factor matrix. That endpoint is a heuristic, so inspect
#' `fit$cv` and widen `lambda_ratio` if the selected value sits at either end
#' of the grid.
#'
#' @param X,y,u,Z,family,maxit,tol,ridge As in [spgtr()].
#' @param lambda Optional vector of sparsity levels to try. Leave `NULL`
#'   (default) to use an automatic log-spaced grid.
#' @param nfolds Number of cross-validation folds (default `5`).
#' @param nlambda,lambda_ratio Length of the automatic grid and the factor
#'   between its smallest and largest value (defaults `20` and `100`).
#' @return The `spgtr` fit at the selected sparsity, with three extra elements:
#'   `lambda_min` (the chosen value), `lambda_seq` (the grid), and `cv` (a data
#'   frame of `lambda` and mean out-of-fold `deviance`).
#' @seealso [spgtr()], [summary.spgtr()]
#' @examples
#' set.seed(2)
#' B <- outer(c(2, rep(0, 5)), c(2, rep(0, 4)))
#' X <- lapply(1:100, function(i) matrix(rnorm(30), 6, 5))
#' eta <- vapply(X, function(xi) sum(B * xi), numeric(1))
#' y <- rbinom(100, 1, 1 / (1 + exp(-eta)))
#'
#' fit <- spgtr_cv(X, y, u = c(1, 1), nfolds = 3, nlambda = 6)
#' fit$lambda_min
#' fit$cv
#' fit$selected # rows kept in each dimension
#' @export
spgtr_cv <- function(X, y, u = NULL, Z = NULL, family = stats::binomial(),
                     lambda = NULL, nfolds = 5L, nlambda = 20L,
                     lambda_ratio = 100, maxit = 500L, tol = 1e-8,
                     ridge = 1e-8) {
  family <- .spgtr_family(family)
  des <- .tepls_design(X)
  Z <- .spgtr_check_z(Z, des$n)
  y <- .spgtr_check_y(y, des$n)
  nfolds <- as.integer(nfolds)
  if (nfolds < 2L || nfolds > des$n) {
    stop("nfolds must be between 2 and the number of observations.")
  }

  prep_rows <- function(rows, uu) {
    .spgtr_prepare(des$Xmat[rows, , drop = FALSE], des$dims, y[rows],
                   if (is.null(Z)) NULL else Z[rows, , drop = FALSE],
                   family, uu, ridge)
  }

  full <- prep_rows(rep(TRUE, des$n), u)
  if (is.null(lambda)) {
    lam_max <- .spgtr_lambda_max(.spgtr_solve(full, 0, "envelope", maxit, tol))
    lambda <- exp(seq(log(lam_max / lambda_ratio), log(lam_max),
                      length.out = nlambda))
  }
  lambda <- sort(unique(as.numeric(lambda)))

  folds <- sample(rep(seq_len(nfolds), length.out = des$n))
  dev <- matrix(NA_real_, length(lambda), nfolds)
  for (f in seq_len(nfolds)) {
    tr <- folds != f
    prep <- try(prep_rows(tr, full$u), silent = TRUE)
    if (inherits(prep, "try-error")) next
    warm <- NULL
    for (l in seq_along(lambda)) {
      fit <- try(.spgtr_solve(prep, lambda[l], "envelope", maxit, tol, warm),
                 silent = TRUE)
      if (inherits(fit, "try-error")) next
      # Warm start the next (sparser) fit unless this one collapsed.
      if (all(fit$nonzero >= fit$u)) warm <- fit$W
      eta <- .spgtr_eta(fit, des$Xmat[!tr, , drop = FALSE],
                        if (is.null(Z)) NULL else Z[!tr, , drop = FALSE])
      mu <- family$linkinv(eta)
      dev[l, f] <- sum(family$dev.resids(y[!tr], mu, rep(1, sum(!tr))))
    }
  }

  cvm <- rowMeans(dev, na.rm = TRUE)
  if (all(is.na(cvm))) stop("Every cross-validation fit failed.")
  best <- lambda[which.min(cvm)]
  out <- .spgtr_solve(full, best, "envelope", maxit, tol)
  out$cv <- data.frame(lambda = lambda, deviance = cvm)
  out$lambda_seq <- lambda
  out$lambda_min <- best
  out
}

.spgtr_family <- function(family) {
  if (is.character(family)) family <- get(family, mode = "function")()
  if (is.function(family)) family <- family()
  if (!inherits(family, "family")) stop("family must be a GLM family.")
  family
}

.spgtr_check_z <- function(Z, n) {
  if (is.null(Z)) return(NULL)
  Z <- as.matrix(Z)
  if (nrow(Z) != n) stop("Z must have one row per observation in X.")
  if (!is.numeric(Z)) stop("Z must be numeric; encode factors as dummies.")
  Z
}

.spgtr_check_y <- function(y, n) {
  if (is.factor(y)) {
    if (nlevels(y) != 2L) stop("A factor y must have exactly two levels.")
    y <- as.integer(y) - 1L
  }
  if (is.logical(y)) y <- as.integer(y)
  if (!is.numeric(y)) stop("y must be numeric, logical, or a two-level factor.")
  if (length(y) != n) stop("y must have one element per observation in X.")
  y
}

# Largest useful penalty: the lambda whose first proximal step would zero every
# row of every mode's basis. Heuristic grid endpoint, not a KKT bound.
.spgtr_lambda_max <- function(fit) {
  min(vapply(seq_along(fit$W), function(k) {
    G <- fit$W[[k]]
    M <- fit$Sig[[k]]
    MUinv <- .sym_pow(M + fit$U[[k]], -1)
    fg <- .env_fg(G, M, MUinv)
    Gr <- .env_pgrad(G, fg$grad, rep(0, nrow(G)))
    step <- 1 / (norm(G, "F") / 2 * norm(Gr, "F") +
                   nrow(G) * 0.01 * norm(fg$grad, "F"))
    max(sqrt(rowSums((G - step * Gr)^2)) / (fit$weights[[k]] * step))
  }, numeric(1)))
}

# Linear predictor for a raw n x prod(dims) design matrix.
.spgtr_eta <- function(object, Xmat, Z) {
  eta <- as.vector(sweep(Xmat, 2L, object$Xbar, `-`) %*% object$bvec) +
    object$alpha
  if (!is.null(Z) && length(object$gamma)) {
    eta <- eta + as.vector(as.matrix(Z) %*% object$gamma)
  }
  eta
}

#' Predict from a Tensor Regression Fit
#'
#' @param object A fit from [spgtr()] or [spgtr_cv()].
#' @param newX New subjects, in either form accepted by [spgtr()]. If omitted,
#'   predictions for the subjects used in fitting are returned.
#' @param newZ New ordinary covariates, required when the fit used `Z`.
#' @param type `"response"` (default) for the outcome scale -- a probability
#'   for `binomial()`, an expected count for `poisson()`; `"link"` for the
#'   linear predictor; `"class"` for a 0/1 label thresholded at `0.5`
#'   (binomial fits only).
#' @param ... Unused.
#' @return A numeric vector with one prediction per subject.
#' @examples
#' set.seed(3)
#' X <- lapply(1:80, function(i) matrix(rnorm(20), 5, 4))
#' y <- rbinom(80, 1, 0.5)
#' fit <- spgtr(X, y, u = c(1, 1))
#' predict(fit, X[1:5], type = "response")
#' @export
predict.spgtr <- function(object, newX = NULL, newZ = NULL,
                          type = c("response", "link", "class"), ...) {
  type <- match.arg(type)
  if (is.null(newX)) {
    eta <- object$linear.predictors
  } else {
    des <- .tepls_design(newX)
    if (!identical(des$dims, object$dims)) {
      stop("newX dimensions must match the fitted predictor dimensions.")
    }
    if (length(object$gamma) && is.null(newZ)) {
      stop("newZ is required: the fit used nuisance covariates.")
    }
    eta <- .spgtr_eta(object, des$Xmat, newZ)
  }
  switch(type,
    link = eta,
    response = object$family$linkinv(eta),
    class = {
      if (object$family$family != "binomial") {
        stop("type = \"class\" is only defined for binomial fits.")
      }
      as.integer(object$family$linkinv(eta) > 0.5)
    }
  )
}

#' @rdname spgtr
#' @param object A fit from [spgtr()] or [spgtr_cv()].
#' @export
coef.spgtr <- function(object, ...) {
  object$coef
}

#' @rdname spgtr
#' @param x A fit from [spgtr()] or [spgtr_cv()].
#' @param ... Unused.
#' @export
print.spgtr <- function(x, ...) {
  cat("<spgtr: sparse penalized generalized tensor regression>\n")
  cat("Outcome:        ", x$family$family, "with", x$family$link, "link\n")
  cat("Subjects:       ", x$n, "\n")
  cat("Array shape:    ", paste(x$dims, collapse = " x "), "\n")
  cat("Directions (u): ", paste(x$u, collapse = " "), "\n")
  cat("Basis:          ", x$basis,
      if (x$basis == "envelope") sprintf("(lambda = %g)", x$lambda) else "",
      "\n")
  cat("Slices kept:    ",
      paste(sprintf("%d/%d", x$nonzero, x$dims), collapse = "  "), "\n")
  cat("Deviance:       ", format(x$deviance, digits = 6), "\n")
  invisible(x)
}

#' Summarize a Tensor Regression Fit
#'
#' Prints a plain-language report: how much of the array was kept in each
#' dimension and which slices those are, how well the model fits, and how far
#' the coefficient array is from zero. For yes/no outcomes it also reports the
#' in-sample accuracy and AUC.
#'
#' In-sample fit statistics are optimistic. For an honest estimate, hold out
#' subjects or read the cross-validated deviance in `fit$cv` after
#' [spgtr_cv()].
#'
#' @param object A fit from [spgtr()] or [spgtr_cv()].
#' @param ... Unused.
#' @return Invisibly, a list with the quantities printed: `pseudo_r2`,
#'   `deviance`, `null_deviance`, `selected`, `bnorm`, and, for binomial fits,
#'   `accuracy` and `auc`.
#' @examples
#' set.seed(4)
#' B <- outer(c(2, rep(0, 5)), c(2, rep(0, 4)))
#' X <- lapply(1:100, function(i) matrix(rnorm(30), 6, 5))
#' eta <- vapply(X, function(xi) sum(B * xi), numeric(1))
#' y <- rbinom(100, 1, 1 / (1 + exp(-eta)))
#' summary(spgtr(X, y, u = c(1, 1)))
#' @export
summary.spgtr <- function(object, ...) {
  print(object)
  out <- list(
    deviance = object$deviance, null_deviance = object$null.deviance,
    pseudo_r2 = 1 - object$deviance / object$null.deviance,
    selected = object$selected, bnorm = sqrt(sum(object$bvec^2))
  )
  cat("\nSlices used, by dimension:\n")
  for (k in seq_along(object$selected)) {
    idx <- object$selected[[k]]
    cat(sprintf("  dim %d (%d): %s\n", k, object$dims[k],
                if (length(idx) == object$dims[k]) {
                  "all"
                } else {
                  paste(idx, collapse = ", ")
                }))
  }
  cat(sprintf("\nDeviance explained: %.1f%% (in-sample)\n",
              100 * out$pseudo_r2))
  cat(sprintf("Coefficient array norm: %.4g\n", out$bnorm))
  if (object$family$family == "binomial") {
    yy <- object$y
    out$accuracy <- mean((object$fitted > 0.5) == (yy > 0.5))
    n1 <- sum(yy > 0.5)
    n0 <- length(yy) - n1
    out$auc <- if (n1 == 0 || n0 == 0) {
      NA_real_
    } else {
      (sum(rank(object$fitted)[yy > 0.5]) - n1 * (n1 + 1) / 2) / (n1 * n0)
    }
    cat(sprintf("Accuracy: %.3f    AUC: %.3f  (in-sample)\n",
                out$accuracy, out$auc))
  }
  if (!is.null(object$lambda_min)) {
    cat(sprintf("Sparsity chosen by cross-validation: lambda = %.4g\n",
                object$lambda_min))
  }
  invisible(out)
}
