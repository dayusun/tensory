library(tensory)

test_that("ttm rejects mode below 1 or above tensor order", {
  t3d <- tensor(array(1:24, dim = c(4, 3, 2)))
  m <- matrix(1:8, nrow = 2, ncol = 4)

  expect_error(ttm(t3d, m, mode = 0), "Mode")
  expect_error(ttm(t3d, m, mode = 4), "Mode must be between 1 and number of tensor dimensions")
})

test_that("ttm rejects mixed-sign mode vectors", {
  t3d <- tensor(array(1:24, dim = c(4, 3, 2)))
  m1 <- matrix(1:8, nrow = 2, ncol = 4)
  m2 <- matrix(1:6, nrow = 2, ncol = 3)

  expect_error(
    ttm(t3d, list(m1, m2), mode = c(1, -2)),
    "all positive or all negative"
  )
})

test_that("ttt outer product with two scalar tensors stays scalar", {
  a <- Tensor$new(2, integer(0), fast = TRUE)
  b <- Tensor$new(3, integer(0), fast = TRUE)

  res <- ttt(a, b)

  expect_s3_class(res, "Tensor")
  expect_identical(res$dim(), integer(0))
  expect_equal(as.numeric(res$data), 6)
})

test_that("ttt scalar times tensor preserves the other operand's shape", {
  a <- Tensor$new(2, integer(0), fast = TRUE)
  b <- tensor(array(1:6, dim = c(2, 3)))

  res <- ttt(a, b)

  expect_identical(res$dim(), c(2L, 3L))
  expect_equal(as.numeric(res$data), 2 * as.numeric(b$data))
})

test_that("ktensor full() returns a scalar tensor for empty dims", {
  # Empty-arg initializer leaves U = NULL / lambda = NULL, which dim() reports
  # as integer(0). full() in that branch must honor the scalar-tensor convention.
  k <- KTensor$new()
  full_t <- k$full()

  expect_s3_class(full_t, "Tensor")
  expect_identical(full_t$dim(), integer(0))
})

test_that("Tensor constructor rejects mismatched dims and data length", {
  expect_error(
    Tensor$new(1:6, dims = c(2L, 4L)),
    "Product of specified dimensions must match"
  )
})
