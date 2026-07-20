test_that("cp_sym recovers a symmetric CP model", {
  set.seed(1)
  u <- qr.Q(qr(matrix(rnorm(8), 4, 2)))
  S0 <- symktensor(c(3, 1.5), u, m = 3)
  X <- as.tensor(S0)

  set.seed(2)
  S <- cp_sym(X, R = 2, maxiters = 2000L, factr = 1e1)
  expect_s3_class(S, "SymKTensor")
  expect_equal(as.tensor(S)$data, X$data, tolerance = 1e-4)

  # Non-symmetric input rejected unless symmetrize = TRUE
  Y <- tensor(array(rnorm(64), dim = c(4, 4, 4)))
  expect_error(cp_sym(Y, R = 2), "symmetric")
  expect_s3_class(cp_sym(Y, R = 2, maxiters = 10, symmetrize = TRUE),
                  "SymKTensor")
})

test_that("tucker_sym computes a shared-subspace Tucker decomposition", {
  set.seed(3)
  # Construct a symmetric tensor with an exact rank-2 symmetric structure.
  u <- qr.Q(qr(matrix(rnorm(10), 5, 2)))
  core <- symmetrize(tensor(array(rnorm(8), dim = c(2, 2, 2))))
  X <- ttm(core, list(u, u, u), mode = 1:3)

  T1 <- tucker_sym(X, r = 2)
  expect_s3_class(T1, "TTensor")
  # All factors identical and orthonormal.
  expect_identical(T1$U[[1]], T1$U[[2]])
  expect_identical(T1$U[[1]], T1$U[[3]])
  expect_equal(crossprod(T1$U[[1]]), diag(2), tolerance = 1e-10)
  # Exact reconstruction for an exactly low-rank symmetric tensor.
  expect_equal(as.tensor(T1)$data, X$data, tolerance = 1e-8)

  expect_error(tucker_sym(tensor(array(rnorm(125), dim = c(5, 5, 5))), 2),
               "symmetric")
})

test_that("gcp_opt with gaussian loss matches cp_opt behavior", {
  set.seed(4)
  U <- lapply(c(5, 4, 3), function(d) matrix(rnorm(d * 2), d, 2))
  X <- as.tensor(ktensor(c(1, 1), U))

  res <- gcp_opt(X, R = 2, type = "gaussian", init = "nvecs",
                 maxiters = 1000L, factr = 1e1)
  fit <- 1 - fnorm(as.tensor(res$K) - X) / fnorm(X)
  expect_gt(fit, 0.999)
  expect_lt(res$objective, 1e-4)
})

test_that("gcp_opt poisson loss fits count data with nonneg factors", {
  set.seed(5)
  U <- lapply(c(6, 5, 4), function(d) matrix(runif(d * 2, 0.2, 1.5), d, 2))
  M <- as.tensor(ktensor(c(8, 4), U))
  X <- tensor(array(rpois(120, as.vector(M$data)), dim = c(6, 5, 4)))

  res <- gcp_opt(X, R = 2, type = "poisson", maxiters = 500L)
  expect_true(all(unlist(res$K$U) >= 0))
  expect_gt(cor(as.vector(as.tensor(res$K)$data), as.vector(M$data)), 0.85)
})

test_that("gcp_opt bernoulli-logit runs on binary data", {
  set.seed(6)
  U <- lapply(c(8, 7, 6), function(d) matrix(rnorm(d * 2), d, 2))
  logits <- as.tensor(ktensor(c(2, 1), U))
  P <- 1 / (1 + exp(-logits$data))
  X <- tensor(array(rbinom(length(P), 1, as.vector(P)), dim = dim(P)))

  res <- gcp_opt(X, R = 2, type = "bernoulli-logit", maxiters = 300L)
  # Objective must be below the trivial all-zero-logits model.
  trivial <- sum(log1p(exp(0)) - X$data * 0)
  expect_lt(res$objective, trivial)
})

test_that("gcp_opt accepts a custom loss", {
  set.seed(7)
  X <- tensor(array(rnorm(24), dim = c(2, 3, 4)))
  myloss <- list(
    f = function(x, m) abs(x - m),
    g = function(x, m) -sign(x - m),
    lower = -Inf
  )
  res <- gcp_opt(X, R = 1, type = myloss, maxiters = 50L)
  expect_s3_class(res$K, "KTensor")
  expect_error(gcp_opt(X, R = 1, type = list(f = identity)), "custom loss")
})
