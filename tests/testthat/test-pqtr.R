# Rank-one-per-mode coefficient with iid standard normal predictors, so the
# mode-k signal matrix is rank one and one direction per mode suffices.
make_qreg <- function(seed, p = c(8, 6), n = 400, sd = 1, hetero = 0) {
  set.seed(seed)
  X <- lapply(seq_len(n), function(i) array(rnorm(prod(p)), dim = p))
  B <- outer(c(1.5, rep(0, p[1] - 1)), c(1, rep(0, p[2] - 1)))
  signal <- vapply(X, function(xi) sum(B * xi), numeric(1))
  scale <- 1 + hetero * (signal - min(signal))
  list(X = X, y = signal + sd * scale * rnorm(n), B = B, p = p, n = n)
}

cosine <- function(a, b) sum(a * b) / sqrt(sum(a^2) * sum(b^2))

test_that("the quantile regression solver reproduces the sample quantile", {
  set.seed(11)
  y <- rnorm(101)
  ones <- matrix(1, 101, 1)
  expect_equal(tensory:::.rq_fit(ones, y, 0.5), median(y), tolerance = 1e-6)
  for (tau in c(0.1, 0.25, 0.75, 0.9)) {
    b <- tensory:::.rq_fit(ones, y, tau)
    # The sample quantile is a minimizer of the check loss, so the solver may
    # not tie it exactly (the minimizer is an interval) but must not lose to it.
    expect_lte(tensory:::.check_loss(y - b, tau),
               tensory:::.check_loss(y - quantile(y, tau, names = FALSE), tau) +
                 1e-6)
  }
})

test_that("the quantile regression solver is not beaten by a generic optimizer", {
  set.seed(12)
  X <- cbind(1, matrix(rnorm(200 * 3), 200, 3))
  y <- X %*% c(1, -2, 0.5, 3) + rt(200, df = 3)
  for (tau in c(0.3, 0.5, 0.8)) {
    b <- tensory:::.rq_fit(X, y, tau)
    obj <- function(g) tensory:::.check_loss(y - X %*% g, tau)
    ref <- stats::optim(qr.coef(qr(X), y), obj, method = "BFGS",
                        control = list(maxit = 2000, reltol = 1e-12))
    expect_lte(obj(b), ref$value + 1e-6)
  }
})

test_that("pqtr recovers a rank-one signal and calibrates the quantile", {
  d <- make_qreg(1)
  fit <- pqtr(d$X, d$y, tau = 0.5, u = c(1, 1))
  expect_s3_class(fit, "pqtr")
  expect_equal(dim(as.tensor(coef(fit))$as_array()), d$p)
  expect_gt(abs(cosine(as.tensor(coef(fit))$as_array(), d$B)), 0.9)
  # A fitted quantile regression puts (about) a tau fraction of the sample
  # below the fitted line.
  expect_equal(mean(d$y < predict(fit)), 0.5, tolerance = 0.05)

  upper <- pqtr(d$X, d$y, tau = 0.9, u = c(1, 1))
  expect_equal(mean(d$y < predict(upper)), 0.9, tolerance = 0.05)
  expect_gt(mean(predict(upper) > predict(fit)), 0.95)
})

test_that("pqtr tracks a coefficient that changes across quantiles", {
  # Noise scale grows with the signal, so the upper quantile depends on the
  # predictor more strongly than the median does.
  d <- make_qreg(2, n = 800, hetero = 0.8)
  b5 <- pqtr(d$X, d$y, tau = 0.5, u = c(1, 1))$bvec
  b9 <- pqtr(d$X, d$y, tau = 0.9, u = c(1, 1))$bvec
  expect_gt(sqrt(sum(b9^2)), sqrt(sum(b5^2)))
  expect_gt(cosine(b9, as.vector(d$B)), 0.9)
})

test_that("pqtr accepts an order-(m+1) tensor with observations in the last mode", {
  d <- make_qreg(3, n = 200)
  arr <- array(0, dim = c(d$p, d$n))
  for (i in seq_len(d$n)) arr[, , i] <- d$X[[i]]

  from_tensor <- pqtr(tensor(arr), d$y, tau = 0.5, u = c(1, 1))
  from_list <- pqtr(d$X, d$y, tau = 0.5, u = c(1, 1))
  expect_equal(from_tensor$bvec, from_list$bvec, tolerance = 1e-10)
})

test_that("pqtr chooses the reduced dimension automatically", {
  d <- make_qreg(4)
  fit <- pqtr(d$X, d$y, tau = 0.5)
  expect_equal(fit$u, c(1L, 1L))
  expect_gt(abs(cosine(as.tensor(coef(fit))$as_array(), d$B)), 0.9)
})

test_that("pqtr handles nuisance covariates and predicts new subjects", {
  d <- make_qreg(5, n = 300)
  Z <- cbind(age = rnorm(d$n), sex = rbinom(d$n, 1, 0.5))
  y <- d$y + Z %*% c(2, -1)

  fit <- pqtr(d$X, as.vector(y), tau = 0.5, Z = Z, u = c(1, 1))
  expect_equal(fit$gamma, c(2, -1), tolerance = 0.3)
  expect_equal(predict(fit, d$X[1:5], Z[1:5, , drop = FALSE]),
               predict(fit)[1:5], tolerance = 1e-10)
  expect_error(predict(fit, d$X[1:5]), "newZ is required")
})

test_that("pqtr_cv scores a grid of reduced dimensions", {
  d <- make_qreg(6, n = 200)
  fit <- pqtr_cv(d$X, d$y, tau = 0.5, u_grid = 1:3, nfolds = 3)
  expect_s3_class(fit, "pqtr")
  expect_equal(nrow(fit$cv), 3L)
  expect_true(all(is.finite(fit$cv$loss)))
  expect_equal(fit$u, fit$u_min)
  expect_true(list(fit$u_min) %in% fit$u_grid)
})

test_that("the PQTR deflation leaves the weights orthonormal", {
  set.seed(7)
  p <- 8
  C0 <- matrix(rnorm(p * 5), p, 5)
  M <- tcrossprod(C0)
  A <- matrix(rnorm(p * p), p, p)
  Sig <- crossprod(A) + diag(p)

  W <- tensory:::.tepls_simpls_mode(M, Sig, 3, orth = "weights")
  expect_equal(crossprod(W), diag(3), tolerance = 1e-10)

  # The SIMPLS deflation used by tepls()/spgtr() instead makes the latent
  # scores uncorrelated.
  V <- tensory:::.tepls_simpls_mode(M, Sig, 3, orth = "scores")
  off <- crossprod(V, Sig %*% V)
  expect_equal(off[upper.tri(off)], rep(0, 3), tolerance = 1e-8)

  # With a rank-one signal matrix the two deflations build the same Krylov
  # subspace, so they can only differ where the signal matrix has higher rank.
  M1 <- tcrossprod(C0[, 1])
  proj <- function(A) tcrossprod(qr.Q(qr(A)))
  expect_equal(proj(tensory:::.tepls_simpls_mode(M1, Sig, 3, orth = "weights")),
               proj(tensory:::.tepls_simpls_mode(M1, Sig, 3, orth = "scores")),
               tolerance = 1e-8)
})

test_that("pqtr validates its arguments", {
  d <- make_qreg(8, n = 40)
  expect_error(pqtr(d$X, d$y, tau = 0), "strictly between 0 and 1")
  expect_error(pqtr(d$X, d$y, tau = c(0.2, 0.5)), "strictly between 0 and 1")
  expect_error(pqtr(d$X, d$y[1:10], tau = 0.5), "one element per observation")
  expect_error(pqtr(d$X, d$y, tau = 0.5, u = c(1, 1, 1)), "length 1 or ndims")
  expect_error(pqtr(d$X, d$y, tau = 0.5, u = c(1, 99)), "1 <= u")
  expect_error(pqtr(d$X, d$y, tau = 0.5, Z = matrix(0, 3, 1)), "one row")
  expect_error(pqtr_cv(d$X, d$y, tau = 0.5, nfolds = 1), "nfolds must be")
})
