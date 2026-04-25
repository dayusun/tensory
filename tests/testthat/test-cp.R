test_that("cp_als recovers a rank-R ktensor up to sign/permutation", {
  set.seed(42)
  R <- 3L
  true_U <- list(
    matrix(stats::rnorm(8 * R), 8, R),
    matrix(stats::rnorm(7 * R), 7, R),
    matrix(stats::rnorm(6 * R), 6, R)
  )
  true_lambda <- c(5, 4, 3)
  truth <- ktensor(true_lambda, true_U)
  X <- as.tensor(truth)

  K <- cp_als(X, R = R, maxiters = 500L, tol = 1e-12,
              init = "random", fixsigns = TRUE)

  expect_s3_class(K, "KTensor")
  expect_equal(K$dim(), c(8L, 7L, 6L))
  diff <- as.tensor(K)$as_array() - X$as_array()
  expect_lt(max(abs(diff)), 1e-6)
})

test_that("cp_als fit improves monotonically across iterations for random data", {
  set.seed(7)
  X <- tensor(array(stats::runif(4 * 5 * 3), dim = c(4, 5, 3)))
  normX <- fnorm(X)

  K1 <- cp_als(X, R = 2L, maxiters = 1L, init = "random", fixsigns = FALSE)
  K2 <- cp_als(X, R = 2L, maxiters = 50L, tol = 1e-10,
               init = "random", fixsigns = FALSE)

  res1 <- fnorm(X - as.tensor(K1))
  res2 <- fnorm(X - as.tensor(K2))
  expect_lte(res2, res1 + 1e-8)
  expect_lt(res2 / normX, 1)
})

test_that("cp_als validates inputs", {
  X <- tensor(array(1:24, dim = c(2, 3, 4)))
  expect_error(cp_als(X, R = 0L), "positive integer")
  expect_error(cp_als(X, R = 2L, dimorder = c(1L, 1L, 2L)), "permutation")
  expect_error(cp_als(X, R = 2L, init = "garbage"))
})

test_that("cp_als accepts an explicit init list", {
  set.seed(1)
  dims <- c(3L, 4L, 5L)
  X <- tensor(array(stats::rnorm(prod(dims)), dim = dims))
  init <- list(
    matrix(stats::rnorm(3 * 2), 3, 2),
    matrix(stats::rnorm(4 * 2), 4, 2),
    matrix(stats::rnorm(5 * 2), 5, 2)
  )
  K <- cp_als(X, R = 2L, init = init, maxiters = 5L, fixsigns = FALSE)
  expect_equal(K$dim(), dims)
})

test_that("mttkrp_blas_cpp matches the R reference", {
  set.seed(123)
  X <- tensor(array(stats::rnorm(4 * 3 * 5), dim = c(4, 3, 5)))
  U <- list(
    matrix(stats::rnorm(4 * 2), 4, 2),
    matrix(stats::rnorm(3 * 2), 3, 2),
    matrix(stats::rnorm(5 * 2), 5, 2)
  )
  for (n in 1:3) {
    V_fast <- mttkrp(X, U, mode = n)
    V_ref <- matrix(0, nrow = X$dim()[n], ncol = 2L)
    others <- setdiff(1:3, n)
    for (r in 1:2) {
      vecs <- lapply(others, function(k) U[[k]][, r])
      contracted <- ttv(X, vecs, mode = others)
      V_ref[, r] <- as.vector(contracted$as_array())
    }
    expect_equal(V_fast, V_ref, tolerance = 1e-10)
  }
})
