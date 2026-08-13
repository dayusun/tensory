sim_logistic <- function(n = 200, p = c(8, 6), scale = 2, seed = 1) {
  set.seed(seed)
  B <- outer(c(scale, rep(0, p[1] - 1)), c(1, rep(0, p[2] - 1)))
  X <- lapply(seq_len(n), function(i) matrix(rnorm(prod(p)), p[1], p[2]))
  eta <- vapply(X, function(xi) sum(B * xi), numeric(1))
  list(X = X, y = rbinom(n, 1, 1 / (1 + exp(-eta))), B = B, eta = eta)
}

test_that("spgtr recovers a rank-1 logistic signal", {
  d <- sim_logistic()
  fit <- spgtr(d$X, d$y, u = c(1, 1))

  bhat <- as.vector(as.tensor(coef(fit))$as_array())
  expect_gt(abs(cor(bhat, as.vector(d$B))), 0.8)
  # The Bayes rule itself only reaches ~0.72 here, so beat the majority rule.
  expect_gt(mean(predict(fit, type = "class") == d$y),
            max(mean(d$y), 1 - mean(d$y)) + 0.1)
  expect_equal(fit$dims, c(8L, 6L))
  expect_equal(dim(fit$scores), c(200L, 1L))
})

test_that("predict agrees on the training data and honours type", {
  d <- sim_logistic(n = 120)
  fit <- spgtr(d$X, d$y, u = c(1, 1))

  expect_equal(predict(fit, d$X, type = "link"), fit$linear.predictors)
  expect_equal(predict(fit, d$X), fit$fitted)
  expect_equal(predict(fit, type = "class"), as.integer(fit$fitted > 0.5))
  # Same data supplied as an order-3 tensor with observations last.
  Xt <- array(unlist(d$X), dim = c(8, 6, 120))
  expect_equal(predict(fit, Xt, type = "link"), fit$linear.predictors)
})

test_that("the coefficient TTensor matches the vectorized coefficient", {
  d <- sim_logistic(n = 100)
  fit <- spgtr(d$X, d$y, u = c(2, 2))
  expect_s3_class(fit$coef, "TTensor")
  expect_equal(as.vector(as.tensor(fit$coef)$as_array()), fit$bvec)
})

test_that("the L2,1 penalty zeroes rows while keeping W semi-orthogonal", {
  d <- sim_logistic(n = 150, scale = 3)
  dense <- spgtr(d$X, d$y, u = c(1, 1), lambda = 0)
  sparse <- spgtr(d$X, d$y, u = c(1, 1), lambda = 5)

  expect_true(all(dense$nonzero == dense$dims))
  expect_true(any(sparse$nonzero < sparse$dims))
  for (W in sparse$W) {
    expect_equal(crossprod(W), diag(ncol(W)), tolerance = 1e-6)
  }
  # The first row carries the whole signal, so it must survive.
  expect_true(1 %in% sparse$selected[[1]])
})

test_that("nuisance covariates enter the model linearly", {
  d <- sim_logistic(n = 200)
  Z <- cbind(z1 = rnorm(200), z2 = rbinom(200, 1, 0.5))
  y <- rbinom(200, 1, 1 / (1 + exp(-(d$eta + 1.5 * Z[, 1]))))

  fit <- spgtr(d$X, y, u = c(1, 1), Z = Z)
  expect_length(fit$gamma, 2L)
  expect_gt(fit$gamma[1], 0)
  expect_error(predict(fit, d$X), "newZ is required")
  expect_equal(predict(fit, d$X, Z, type = "link"), fit$linear.predictors)
})

test_that("order-3 predictors and other families work", {
  set.seed(7)
  p <- c(5, 4, 3)
  B <- array(0, p)
  B[1, 1, 1] <- 3
  X <- lapply(1:150, function(i) array(rnorm(prod(p)), p))
  eta <- vapply(X, function(xi) sum(B * xi), numeric(1))

  fit <- spgtr(X, rbinom(150, 1, 1 / (1 + exp(-eta))), u = c(1, 1, 1))
  expect_length(fit$u, 3L)
  bhat <- as.tensor(coef(fit))$as_array()
  expect_equal(which.max(abs(bhat)), 1L)

  pois <- spgtr(X, rpois(150, exp(eta / 3)), u = c(1, 1, 1), family = poisson())
  expect_equal(pois$family$family, "poisson")
  expect_true(all(predict(pois, type = "response") > 0))
  expect_error(predict(pois, type = "class"), "binomial")
})

test_that("keeping every direction reproduces the full GLM exactly", {
  # With u = p each W_k is a full orthonormal basis, so the scores are just a
  # rotation of vec(X) and the GLM fit must be identical to glm.fit() on the
  # flattened predictor. Catches any error in centering, the score map, or the
  # reconstruction B = D x_1 W_1 ... x_m W_m.
  set.seed(9)
  p <- c(3, 3)
  n <- 200
  X <- lapply(seq_len(n), function(i) matrix(rnorm(prod(p)), p[1], p[2]))
  V <- t(vapply(X, as.vector, numeric(prod(p))))
  Vc <- V - rep(colMeans(V), each = n)
  eta <- as.vector(V %*% rnorm(prod(p))) / 2

  for (fam in list(binomial(), poisson(), gaussian())) {
    yv <- switch(fam$family,
      binomial = rbinom(n, 1, 1 / (1 + exp(-eta))),
      poisson = rpois(n, exp(eta / 2)),
      gaussian = eta + rnorm(n))
    ref <- stats::glm.fit(cbind(1, Vc), yv, family = fam)
    for (bs in c("simpls", "envelope")) {
      fit <- spgtr(X, yv, u = p, basis = bs, family = fam)
      expect_equal(fit$fitted, unname(ref$fitted.values), tolerance = 1e-8)
      expect_equal(fit$bvec, unname(ref$coefficients[-1]), tolerance = 1e-7)
    }
  }
})

test_that("the SIMPLS basis reproduces tepls on a gaussian response", {
  set.seed(3)
  p <- c(7, 5)
  B <- outer(c(1, rep(0, 6)), c(1, rep(0, 4)))
  X <- lapply(1:90, function(i) matrix(rnorm(prod(p)), p[1], p[2]))
  y <- vapply(X, function(xi) sum(B * xi), numeric(1)) + rnorm(90, sd = 0.2)

  a <- tepls(X, y, u = c(2, 2))
  b <- spgtr(X, y, u = c(2, 2), family = gaussian(), basis = "simpls")
  expect_equal(as.vector(a$coef$as_array()), b$bvec, tolerance = 1e-6)
  expect_equal(a$fitted, b$fitted, tolerance = 1e-6)
})

test_that("envelope dimensions can be selected automatically", {
  d <- sim_logistic(n = 150)
  fit <- spgtr(d$X, d$y)
  expect_length(fit$u, 2L)
  expect_true(all(fit$u >= 1 & fit$u <= fit$dims))
})

test_that("spgtr_cv selects a penalty from its grid", {
  d <- sim_logistic(n = 120, p = c(6, 5), scale = 3)
  set.seed(11)
  fit <- spgtr_cv(d$X, d$y, u = c(1, 1), nfolds = 3, nlambda = 5)

  expect_true(fit$lambda_min %in% fit$lambda_seq)
  expect_equal(fit$lambda, fit$lambda_min)
  expect_equal(nrow(fit$cv), 5L)
  expect_true(all(is.finite(fit$cv$deviance)))
})

test_that("summary reports the selected slices and in-sample fit", {
  d <- sim_logistic(n = 120, scale = 3)
  fit <- spgtr(d$X, d$y, u = c(1, 1), lambda = 5)
  expect_output(summary(fit), "Slices used")
  s <- utils::capture.output(res <- summary(fit))
  s <- res
  expect_equal(s$selected, fit$selected)
  expect_gte(s$auc, 0)
  expect_lte(s$auc, 1)
  expect_equal(s$accuracy, mean((fit$fitted > 0.5) == (d$y > 0.5)))
})

test_that("y accepts factors and logicals", {
  d <- sim_logistic(n = 100)
  num <- spgtr(d$X, d$y, u = c(1, 1))
  fac <- spgtr(d$X, factor(d$y), u = c(1, 1))
  lgl <- spgtr(d$X, d$y == 1, u = c(1, 1))
  expect_equal(num$bvec, fac$bvec)
  expect_equal(num$bvec, lgl$bvec)
  expect_error(spgtr(d$X, d$y[1:10], u = c(1, 1)), "one element per")
  expect_error(spgtr(d$X, d$y, u = c(1, 1), lambda = -1), "non-negative")
})

# --- compiled kernels vs. their R fallbacks ---------------------------------

test_that("spgtr_mode_covs_cpp matches the R fallback", {
  skip_if_not(exists("spgtr_mode_covs_cpp", mode = "function"))
  set.seed(5)
  dims <- c(4L, 3L, 5L)
  n <- 20L
  Xt <- matrix(rnorm(prod(dims) * n), prod(dims), n)
  cpp <- spgtr_mode_covs_cpp(Xt, dims)
  ref <- tensory:::.tepls_mode_covs(t(Xt), dims, n)
  expect_equal(cpp, ref, tolerance = 1e-10)
})

test_that("spgtr_slpg_cpp matches the R fallback", {
  skip_if_not(exists("spgtr_slpg_cpp", mode = "function"))
  set.seed(6)
  p <- 12L
  A <- matrix(rnorm(p * p), p)
  M <- crossprod(A) / p + diag(p)
  V <- qr.Q(qr(matrix(rnorm(p * 2), p)))
  U <- tcrossprod(V %*% diag(c(4, 2)), V)
  MUinv <- tensory:::.sym_pow(M + U, -1)
  G0 <- qr.Q(qr(matrix(rnorm(p * 2), p)))

  for (lam in c(0, 0.5)) {
    gam <- rep(lam, p)
    a <- spgtr_slpg_cpp(G0, M, MUinv, gam, 500L, 1e-8, 1e-10)
    b <- tensory:::.env_slpg_r(G0, M, MUinv, gam, 500L, 1e-8, 1e-10)
    # Compare the subspaces, which are what the algorithm identifies.
    expect_equal(tcrossprod(a), tcrossprod(b), tolerance = 1e-5)
    expect_equal(rowSums(a^2) > 0, rowSums(b^2) > 0)
    expect_equal(crossprod(a), diag(2), tolerance = 1e-8)
  }
})
