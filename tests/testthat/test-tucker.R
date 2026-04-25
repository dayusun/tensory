test_that("hosvd recovers a low-multilinear-rank tensor exactly", {
  set.seed(11)
  ranks <- c(2L, 3L, 2L)
  dims <- c(5L, 6L, 4L)
  core <- tensor(array(stats::rnorm(prod(ranks)), dim = ranks))
  U <- lapply(seq_along(dims), function(n) {
    qr.Q(qr(matrix(stats::rnorm(dims[n] * ranks[n]), dims[n], ranks[n])))
  })
  truth <- ttensor(core, U)
  X <- as.tensor(truth)

  T <- hosvd(X, ranks = ranks)
  expect_s3_class(T, "TTensor")
  expect_equal(T$dim(), dims)
  diff <- as.tensor(T)$as_array() - X$as_array()
  expect_lt(max(abs(diff)), 1e-8)

  for (n in seq_along(U)) {
    expect_equal(crossprod(T$U[[n]]), diag(ranks[n]), tolerance = 1e-10)
  }
})

test_that("hosvd tol-based truncation picks sensible ranks", {
  set.seed(2)
  dims <- c(6L, 5L, 4L)
  core_dims <- c(2L, 2L, 2L)
  core <- tensor(array(stats::rnorm(prod(core_dims)), dim = core_dims))
  U <- lapply(seq_along(dims), function(n) {
    qr.Q(qr(matrix(stats::rnorm(dims[n] * core_dims[n]), dims[n], core_dims[n])))
  })
  X <- as.tensor(ttensor(core, U))

  T <- hosvd(X, tol = 1e-6)
  for (n in seq_along(dims)) {
    expect_lte(ncol(T$U[[n]]), core_dims[n])
  }
  diff <- as.tensor(T)$as_array() - X$as_array()
  expect_lt(max(abs(diff)), 1e-4)
})

test_that("tucker_als recovers a low-multilinear-rank tensor", {
  set.seed(21)
  ranks <- c(2L, 2L, 2L)
  dims <- c(5L, 6L, 4L)
  core <- tensor(array(stats::rnorm(prod(ranks)), dim = ranks))
  U <- lapply(seq_along(dims), function(n) {
    qr.Q(qr(matrix(stats::rnorm(dims[n] * ranks[n]), dims[n], ranks[n])))
  })
  X <- as.tensor(ttensor(core, U))

  T <- tucker_als(X, ranks = ranks, maxiters = 50L, tol = 1e-10,
                  init = "nvecs")

  expect_s3_class(T, "TTensor")
  expect_equal(T$dim(), dims)
  expect_equal(T$core$dim(), ranks)
  for (n in seq_along(U)) {
    expect_equal(crossprod(T$U[[n]]), diag(ranks[n]), tolerance = 1e-8)
  }
  diff <- as.tensor(T)$as_array() - X$as_array()
  expect_lt(max(abs(diff)), 1e-6)
})

test_that("tucker_als accepts scalar ranks and random init", {
  set.seed(3)
  X <- tensor(array(stats::rnorm(4 * 4 * 4), dim = c(4, 4, 4)))
  T <- tucker_als(X, ranks = 2L, maxiters = 20L, init = "random")
  expect_equal(T$core$dim(), c(2L, 2L, 2L))
})

test_that("tucker_als validates inputs", {
  X <- tensor(array(1:24, dim = c(2, 3, 4)))
  expect_error(tucker_als(X, ranks = c(0L, 1L, 1L)), "between 1")
  expect_error(tucker_als(X, ranks = c(3L, 4L, 5L)), "between 1")
  expect_error(tucker_als(X, ranks = 2L, dimorder = c(1L, 1L, 2L)), "permutation")
})
