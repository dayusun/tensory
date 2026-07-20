test_that("tenrand creates a tensor with entries in [0, 1]", {
  set.seed(1)
  X <- tenrand(c(2, 3, 4))
  expect_s3_class(X, "Tensor")
  expect_equal(X$dim(), c(2L, 3L, 4L))
  expect_true(all(X$data >= 0 & X$data <= 1))
})

test_that("teneye satisfies the identity property on unit vectors", {
  E <- teneye(4, 3)
  expect_true(issymmetric(E))
  set.seed(2)
  for (i in 1:3) {
    x <- rnorm(3)
    x <- x / sqrt(sum(x^2))
    expect_equal(as.numeric(ttsv(E, x, -1)), x, tolerance = 1e-12)
  }
  expect_error(teneye(3, 2), "even")
})

test_that("tendiag places values on the superdiagonal", {
  D <- tendiag(1:3, c(3, 3, 3))
  expect_equal(D$data[1, 1, 1], 1)
  expect_equal(D$data[2, 2, 2], 2)
  expect_equal(D$data[3, 3, 3], 3)
  expect_equal(sum(D$data != 0), 3)

  M <- tendiag(c(5, 7))
  expect_equal(M$dim(), c(2L, 2L))
  expect_equal(M$data[cbind(1:2, 1:2)], c(5, 7))

  expect_error(tendiag(1:3, c(2, 3)), "length")
})

test_that("export_data / import_data round-trips tensors, matrices, ktensors", {
  tmp <- tempfile(fileext = ".txt")
  on.exit(unlink(tmp))

  X <- tensor(array(rnorm(24), dim = c(2, 3, 4)))
  export_data(X, tmp)
  Y <- import_data(tmp)
  expect_equal(Y$dim(), X$dim())
  expect_equal(Y$data, X$data, tolerance = 1e-14)

  M <- matrix(rnorm(6), 2, 3)
  export_data(M, tmp)
  expect_equal(import_data(tmp), M, tolerance = 1e-14)

  K <- ktensor(c(2, 3), list(matrix(rnorm(8), 4), matrix(rnorm(6), 3)))
  export_data(K, tmp)
  K2 <- import_data(tmp)
  expect_s3_class(K2, "KTensor")
  expect_equal(K2$lambda, K$lambda, tolerance = 1e-14)
  expect_equal(K2$U, K$U, tolerance = 1e-14)
})
