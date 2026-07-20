test_that("symktensor represents symmetric CP models", {
  set.seed(1)
  u <- matrix(rnorm(6), 3, 2)
  S <- symktensor(c(2, -1), u, m = 3)

  expect_equal(S$dim(), rep(3L, 3))
  expect_equal(S$ndims(), 3L)
  expect_equal(ncomponents(S), 2L)

  K <- S$as_ktensor()
  expect_equal(as.tensor(S)$data, as.tensor(K)$data, tolerance = 1e-12)
  expect_true(issymmetric(as.tensor(S), tol = 1e-12))
  expect_equal(fnorm(S), fnorm(as.tensor(S)), tolerance = 1e-10)
})

test_that("symtensor round-trips symmetric tensors compactly", {
  set.seed(2)
  X <- symmetrize(tensor(array(rnorm(81), dim = c(3, 3, 3, 3))))
  S <- symtensor(X)

  expect_equal(length(S$vals), choose(3 + 4 - 1, 4))
  expect_equal(S$full()$data, X$data, tolerance = 1e-12)
  expect_true(issymmetric(S))
  expect_equal(fnorm(S), fnorm(X), tolerance = 1e-12)

  # ttsv passes through to the dense equivalent
  v <- rnorm(3)
  expect_equal(ttsv(S, v, 0), ttsv(X, v, 0), tolerance = 1e-12)

  # Non-symmetric input is rejected unless symmetrize = TRUE
  Y <- tensor(array(rnorm(27), dim = c(3, 3, 3)))
  expect_error(symtensor(Y), "symmetric")
  S2 <- symtensor(Y, symmetrize = TRUE)
  expect_equal(S2$full()$data, symmetrize(Y)$data, tolerance = 1e-12)
})

test_that("sumtensor distributes operations over parts", {
  set.seed(3)
  X <- tensor(array(rnorm(24), dim = c(2, 3, 4)))
  K <- ktensor(c(1, 2), lapply(c(2, 3, 4), function(d) matrix(rnorm(d * 2), d, 2)))
  S <- sumtensor(X, K)

  dense <- X + as.tensor(K)
  expect_equal(S$dim(), c(2L, 3L, 4L))
  expect_equal(as.tensor(S)$data, dense$data, tolerance = 1e-12)

  Y <- tensor(array(rnorm(24), dim = c(2, 3, 4)))
  expect_equal(innerprod(S, Y), innerprod(dense, Y), tolerance = 1e-10)
  expect_equal(fnorm(S), fnorm(dense), tolerance = 1e-10)

  # ttm / ttv distribute over parts
  M <- matrix(rnorm(6), 2, 3)
  expect_equal(ttm(S, M, mode = 2)$data, ttm(dense, M, mode = 2)$data,
               tolerance = 1e-10)
  v <- rnorm(3)
  expect_equal(ttv(S, v, mode = 2)$data, ttv(dense, v, mode = 2)$data,
               tolerance = 1e-10)

  # appending parts
  S3 <- S + Y
  expect_equal(length(S3$parts), 3L)
  expect_equal(as.tensor(S3)$data, (dense + Y)$data, tolerance = 1e-12)
})
