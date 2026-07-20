#' @include tensor_class.R ktensor_class.R tensor_dense_methods.R cp_decomposition.R ktensor_methods.R
NULL

# Khatri-Rao product of all factors except mode n, with rows ordered to match
# the columns of unfold(x, rdims = n) (smallest remaining mode fastest).
.kr_others <- function(U, n) {
  others <- setdiff(seq_along(U), n)
  mats <- U[rev(others)]
  if (length(mats) == 1) {
    return(mats[[1]])
  }
  khatri_rao(mats)
}

# Shared validation for the CP family. Returns list(X, dims, N).
.cp_check_input <- function(X, R, fname) {
  if (!inherits(X, "Sptensor")) {
    X <- as.tensor(X)
  }
  R <- as.integer(R)
  if (length(R) != 1L || R < 1L) {
    stop("R must be a positive integer.")
  }
  dims <- as.integer(X$dim())
  N <- length(dims)
  if (N < 2L) {
    stop(sprintf("%s requires a tensor with at least 2 modes.", fname))
  }
  list(X = X, R = R, dims = dims, N = N)
}

# Relative fit 1 - ||X - K|| / ||X|| computed from CP structure (no dense
# residual): ||K||^2 via the Gram Hadamard identity, <X, K> via one MTTKRP.
.cp_fit <- function(X, lambda, U, normX) {
  H <- .kt_gram_hadamard(U)
  normP_sq <- sum(tcrossprod(lambda) * H)
  V <- mttkrp(X, U, mode = 1L)
  inner <- sum(lambda * colSums(U[[1L]] * V))
  normresidual <- sqrt(max(normX^2 + normP_sq - 2 * inner, 0))
  1 - normresidual / normX
}

#' Nonnegative CP Decomposition via Multiplicative Updates
#'
#' Computes a nonnegative CP decomposition with Lee-Seung style multiplicative
#' updates, mirroring the MATLAB Tensor Toolbox `cp_nmu`. All entries of `X`
#' must be nonnegative; factors stay nonnegative throughout.
#'
#' @param X A nonnegative Tensor or array-like object.
#' @param R Target CP rank.
#' @param tol Convergence tolerance on change in fit (default `1e-4`).
#' @param maxiters Maximum number of sweeps (default `100`).
#' @param init `"random"` (uniform) or a list of nonnegative factor matrices.
#' @param printitn Print fit every `printitn` iterations (`0` to suppress).
#' @return A `KTensor` with nonnegative factors.
#' @examples
#' set.seed(1)
#' X <- tensor(array(runif(24), dim = c(2, 3, 4)))
#' K <- cp_nmu(X, R = 2, maxiters = 20)
#' @export
cp_nmu <- function(X, R,
                   tol = 1e-4,
                   maxiters = 100L,
                   init = "random",
                   printitn = 0L) {
  chk <- .cp_check_input(X, R, "cp_nmu")
  X <- chk$X; R <- chk$R; dims <- chk$dims; N <- chk$N
  if (any(.tensor_as_dense(X)$data < 0)) {
    stop("cp_nmu requires a nonnegative tensor.")
  }

  if (is.list(init)) {
    U <- lapply(init, as.matrix)
    if (length(U) != N) stop("init list must have length ndims(X).")
  } else {
    U <- lapply(dims, function(d) matrix(stats::runif(d * R), d, R))
  }

  epsilon <- .Machine$double.eps
  normX <- fnorm(X)
  fit_prev <- 0

  for (iter in seq_len(maxiters)) {
    for (n in seq_len(N)) {
      Vn <- mttkrp(X, U, mode = n)
      Gamma <- .kt_gram_hadamard(U, modes = setdiff(seq_len(N), n))
      denom <- U[[n]] %*% Gamma + epsilon
      U[[n]] <- U[[n]] * (Vn / denom)
    }

    fit <- .cp_fit(X, rep(1, R), U, normX)
    fit_change <- abs(fit - fit_prev)
    if (printitn > 0L && (iter %% printitn == 0L || iter == 1L)) {
      message(sprintf(" Iter %2d: fit = %.6e, fitdelta = %.6e",
                      iter, fit, fit_change))
    }
    if (iter > 1L && fit_change < tol) break
    fit_prev <- fit
  }

  fixsigns(arrange(ktensor(rep(1, R), U)))
}

#' Poisson CP Decomposition (CP-APR) via Multiplicative Updates
#'
#' Fits a CP model to nonnegative (count) data by maximizing the Poisson
#' log-likelihood with the multiplicative-update algorithm of Chi & Kolda,
#' mirroring the MATLAB Tensor Toolbox `cp_apr` (`'mu'` method).
#'
#' @param X A nonnegative Tensor or array-like object (typically counts).
#' @param R Target CP rank.
#' @param tol KKT-violation stopping tolerance (default `1e-4`).
#' @param maxiters Maximum number of outer iterations (default `200`).
#' @param maxinner Maximum inner updates per mode per outer iteration
#'   (default `10`).
#' @param epsDivZero Safeguard added before divisions (default `1e-10`).
#' @param init `"random"` or a list of initial nonnegative factor matrices.
#' @param printitn Print progress every `printitn` outer iterations.
#' @return A `KTensor` with nonnegative factors and weights.
#' @references Chi, E. C. and Kolda, T. G. (2012). On tensors, sparsity, and
#'   nonnegative factorizations. SIAM J. Matrix Anal. Appl. 33(4).
#' @examples
#' set.seed(1)
#' X <- tensor(array(rpois(24, 3), dim = c(2, 3, 4)))
#' K <- cp_apr(X, R = 2, maxiters = 20)
#' @export
cp_apr <- function(X, R,
                   tol = 1e-4,
                   maxiters = 200L,
                   maxinner = 10L,
                   epsDivZero = 1e-10,
                   init = "random",
                   printitn = 0L) {
  chk <- .cp_check_input(X, R, "cp_apr")
  X <- chk$X; R <- chk$R; dims <- chk$dims; N <- chk$N
  Xd <- .tensor_as_dense(X)
  if (any(Xd$data < 0)) {
    stop("cp_apr requires a nonnegative tensor.")
  }

  if (is.list(init)) {
    U <- lapply(init, as.matrix)
    if (length(U) != N) stop("init list must have length ndims(X).")
  } else {
    U <- lapply(dims, function(d) matrix(stats::runif(d * R), d, R))
  }
  lambda <- rep(1, R)

  unfoldings <- lapply(seq_len(N), function(n) unfold(Xd, rdims = n))

  kkt_viol <- Inf
  for (iter in seq_len(maxiters)) {
    is_converged <- TRUE
    kkt_viol <- 0

    for (n in seq_len(N)) {
      Xn <- unfoldings[[n]]
      B <- sweep(U[[n]], 2L, lambda, `*`)
      Pi <- .kr_others(U, n)

      for (inner in seq_len(maxinner)) {
        Mn <- B %*% t(Pi)
        Phi <- (Xn / pmax(Mn, epsDivZero)) %*% Pi
        viol <- max(abs(pmin(B, 1 - Phi)))
        kkt_viol <- max(kkt_viol, viol)
        if (viol < tol) break
        is_converged <- FALSE
        B <- B * Phi
      }

      lambda <- colSums(B)
      lam_safe <- ifelse(lambda == 0, 1, lambda)
      U[[n]] <- sweep(B, 2L, lam_safe, `/`)
    }

    if (printitn > 0L && (iter %% printitn == 0L || iter == 1L)) {
      message(sprintf(" Iter %2d: max KKT violation = %.6e", iter, kkt_viol))
    }
    if (is_converged) break
  }

  arrange(ktensor(lambda, U))
}

# Pack / unpack a factor list into a flat parameter vector for optim().
.factors_to_vec <- function(U) {
  unlist(lapply(U, as.vector), use.names = FALSE)
}

.vec_to_factors <- function(v, dims, R) {
  out <- vector("list", length(dims))
  offset <- 0L
  for (n in seq_along(dims)) {
    len <- dims[n] * R
    out[[n]] <- matrix(v[offset + seq_len(len)], dims[n], R)
    offset <- offset + len
  }
  out
}

#' CP Decomposition via Direct Optimization
#'
#' Fits a CP model by minimizing `||X - K||^2` over all factor matrices
#' simultaneously with L-BFGS-B, mirroring the MATLAB Tensor Toolbox `cp_opt`.
#'
#' @param X A Tensor or array-like object.
#' @param R Target CP rank.
#' @param init `"random"` (scaled normal) or a list of initial factor
#'   matrices.
#' @param lower Optional lower bound for all factor entries (e.g. `0` for a
#'   nonnegative model); default unbounded.
#' @param maxiters Maximum optimizer iterations (default `500`).
#' @param factr `optim` L-BFGS-B `factr` convergence parameter.
#' @param printitn If positive, print the optimizer trace.
#' @return A `KTensor`.
#' @examples
#' set.seed(1)
#' X <- tensor(array(rnorm(24), dim = c(2, 3, 4)))
#' K <- cp_opt(X, R = 2)
#' @export
cp_opt <- function(X, R,
                   init = "random",
                   lower = -Inf,
                   maxiters = 500L,
                   factr = 1e7,
                   printitn = 0L) {
  chk <- .cp_check_input(X, R, "cp_opt")
  X <- chk$X; R <- chk$R; dims <- chk$dims; N <- chk$N

  if (is.list(init)) {
    U0 <- lapply(init, as.matrix)
    if (length(U0) != N) stop("init list must have length ndims(X).")
  } else {
    U0 <- .init_cp_factors(X, R, dims, N, init)
    scale0 <- (fnorm(X) / sqrt(R))^(1 / N)
    U0 <- lapply(U0, function(M) scale0 * M / sqrt(mean(M^2)))
    if (is.finite(lower)) {
      U0 <- lapply(U0, function(M) pmax(abs(M), lower))
    }
  }

  normX2 <- fnorm(X)^2

  fg <- function(v) {
    U <- .vec_to_factors(v, dims, R)
    H <- .kt_gram_hadamard(U)
    V1 <- mttkrp(X, U, mode = 1L)
    inner <- sum(U[[1L]] * V1)
    f <- normX2 - 2 * inner + sum(H)

    grad <- vector("list", N)
    for (n in seq_len(N)) {
      Vn <- if (n == 1L) V1 else mttkrp(X, U, mode = n)
      Gamma <- .kt_gram_hadamard(U, modes = setdiff(seq_len(N), n))
      grad[[n]] <- 2 * (U[[n]] %*% Gamma - Vn)
    }
    list(value = f, gradient = .factors_to_vec(grad))
  }

  res <- stats::optim(
    par = .factors_to_vec(U0),
    fn = function(v) fg(v)$value,
    gr = function(v) fg(v)$gradient,
    method = "L-BFGS-B",
    lower = lower,
    control = list(maxit = as.integer(maxiters), factr = factr,
                   trace = as.integer(printitn > 0))
  )

  U <- .vec_to_factors(res$par, dims, R)
  fixsigns(arrange(ktensor(rep(1, R), U)))
}

#' Weighted CP Decomposition via Direct Optimization
#'
#' Fits a CP model to data with a weight (indicator) tensor by minimizing
#' `||W * (X - K)||^2`, mirroring the MATLAB Tensor Toolbox `cp_wopt`. Use a
#' 0/1 weight tensor to fit in the presence of missing entries.
#'
#' @param X A Tensor or array-like object (missing entries may hold any value,
#'   typically 0).
#' @param W A weight tensor of the same dimensions as `X` (commonly 0/1).
#' @param R Target CP rank.
#' @param init `"random"` or a list of initial factor matrices.
#' @param maxiters Maximum optimizer iterations (default `500`).
#' @param factr `optim` L-BFGS-B `factr` convergence parameter.
#' @param printitn If positive, print the optimizer trace.
#' @return A `KTensor`.
#' @examples
#' set.seed(1)
#' X <- tensor(array(rnorm(24), dim = c(2, 3, 4)))
#' W <- tensor(array(rbinom(24, 1, 0.8), dim = c(2, 3, 4)))
#' K <- cp_wopt(X, W, R = 2)
#' @export
cp_wopt <- function(X, W, R,
                    init = "random",
                    maxiters = 500L,
                    factr = 1e7,
                    printitn = 0L) {
  chk <- .cp_check_input(X, R, "cp_wopt")
  X <- .tensor_as_dense(chk$X); R <- chk$R; dims <- chk$dims; N <- chk$N
  W <- .tensor_as_dense(W)
  if (!identical(as.integer(W$dim()), dims)) {
    stop("W must have the same dimensions as X.")
  }

  W2 <- W$data * W$data
  Y <- W2 * X$data # pre-weighted data

  if (is.list(init)) {
    U0 <- lapply(init, as.matrix)
    if (length(U0) != N) stop("init list must have length ndims(X).")
  } else {
    U0 <- .init_cp_factors(X, R, dims, N, init)
    scale0 <- (sqrt(sum(Y * X$data)) / sqrt(R) + .Machine$double.eps)^(1 / N)
    U0 <- lapply(U0, function(M) scale0 * M / sqrt(mean(M^2)))
  }

  fg <- function(v) {
    U <- .vec_to_factors(v, dims, R)
    M <- as.tensor(ktensor(rep(1, R), U))$data
    D <- W2 * M - Y # = W^2 * (M - X)
    f <- sum(D * (M - X$data))
    Dt <- Tensor$new(D, dims = dims, fast = TRUE)
    grad <- vector("list", N)
    for (n in seq_len(N)) {
      grad[[n]] <- 2 * mttkrp(Dt, U, mode = n)
    }
    list(value = f, gradient = .factors_to_vec(grad))
  }

  res <- stats::optim(
    par = .factors_to_vec(U0),
    fn = function(v) fg(v)$value,
    gr = function(v) fg(v)$gradient,
    method = "L-BFGS-B",
    control = list(maxit = as.integer(maxiters), factr = factr,
                   trace = as.integer(printitn > 0))
  )

  U <- .vec_to_factors(res$par, dims, R)
  fixsigns(arrange(ktensor(rep(1, R), U)))
}

#' CP Decomposition via Randomized (Sampled) ALS
#'
#' Alternating least squares in which each subproblem is solved from a uniform
#' sample of tensor fibers rather than the full unfolding, in the spirit of
#' the MATLAB Tensor Toolbox `cp_arls`. Unlike `cp_arls`, no FFT-based mixing
#' is applied before sampling (a documented divergence); sampling is plain
#' uniform with replacement.
#'
#' @param X A Tensor or array-like object.
#' @param R Target CP rank.
#' @param tol Convergence tolerance on change in (exact) fit.
#' @param maxiters Maximum number of sweeps (default `50`).
#' @param nsamples Number of sampled fibers per solve. Defaults to
#'   `max(ceiling(10 * R * log2(R + 1)), 4 * R)` capped at the full count.
#' @param ridge Tikhonov regularizer added to the sampled normal equations
#'   (default `1e-10`).
#' @param init `"random"` or a list of initial factor matrices.
#' @param printitn Print fit every `printitn` iterations.
#' @return A `KTensor`.
#' @examples
#' set.seed(1)
#' X <- tensor(array(rnorm(60), dim = c(3, 4, 5)))
#' K <- cp_arls(X, R = 2, maxiters = 20)
#' @export
cp_arls <- function(X, R,
                    tol = 1e-4,
                    maxiters = 50L,
                    nsamples = NULL,
                    ridge = 1e-10,
                    init = "random",
                    printitn = 0L) {
  chk <- .cp_check_input(X, R, "cp_arls")
  X <- .tensor_as_dense(chk$X); R <- chk$R; dims <- chk$dims; N <- chk$N

  if (is.list(init)) {
    U <- lapply(init, as.matrix)
    if (length(U) != N) stop("init list must have length ndims(X).")
  } else {
    U <- .init_cp_factors(X, R, dims, N, init)
  }

  normX <- fnorm(X)
  fit_prev <- 0
  lambda <- rep(1, R)

  for (iter in seq_len(maxiters)) {
    for (n in seq_len(N)) {
      others <- setdiff(seq_len(N), n)
      P <- prod(dims[others])
      s <- if (is.null(nsamples)) {
        min(P, max(ceiling(10 * R * log2(R + 1)), 4L * R))
      } else {
        min(P, as.integer(nsamples))
      }

      midx <- vapply(others, function(k) sample.int(dims[k], s, replace = TRUE),
                     integer(s))
      midx <- matrix(midx, nrow = s)

      Xs <- fibers(X, mode = n, midx = midx) # I_n x s
      Zs <- matrix(1, s, R)
      for (j in seq_along(others)) {
        Zs <- Zs * U[[others[j]]][midx[, j], , drop = FALSE]
      }

      G <- crossprod(Zs) + diag(ridge, R)
      U[[n]] <- t(solve(G, t(Zs) %*% t(Xs)))

      scale_n <- sqrt(colSums(U[[n]]^2))
      scale_n[scale_n == 0] <- 1
      U[[n]] <- sweep(U[[n]], 2L, scale_n, `/`)
      lambda <- scale_n
    }

    Ul <- U
    Ul[[N]] <- sweep(Ul[[N]], 2L, lambda, `*`)
    fit <- .cp_fit(X, rep(1, R), Ul, normX)
    fit_change <- abs(fit - fit_prev)
    if (printitn > 0L && (iter %% printitn == 0L || iter == 1L)) {
      message(sprintf(" Iter %2d: fit = %.6e, fitdelta = %.6e",
                      iter, fit, fit_change))
    }
    if (iter > 1L && fit_change < tol) break
    fit_prev <- fit
  }

  U[[N]] <- sweep(U[[N]], 2L, lambda, `*`)
  fixsigns(arrange(ktensor(rep(1, R), U)))
}
