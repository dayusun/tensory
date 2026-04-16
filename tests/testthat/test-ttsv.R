library(tensory)

test_that("ttsv with n = 0 contracts every mode and returns a numeric scalar", {
  # Symmetric-shape tensor: all modes equal length 3
  x <- tensor(array(1:27, dim = c(3, 3, 3)))
  v <- c(1, 2, 3)

  result <- ttsv(x, v, n = 0)

  expect_type(result, "double")
  expect_length(result, 1L)

  # Reference: contract every mode with v sequentially
  expected <- ttv(x, list(v, v, v), mode = c(1, 2, 3))
  expect_equal(result, as.numeric(expected$as_array()))
})

test_that("ttsv with n = -1 keeps one leading mode and returns a vector", {
  x <- tensor(array(1:27, dim = c(3, 3, 3)))
  v <- c(1, 2, 3)

  result <- ttsv(x, v, n = -1)

  expect_true(is.numeric(result) && !is.matrix(result))
  expect_length(result, 3L)

  expected <- ttv(x, list(v, v), mode = c(2, 3))
  expect_equal(as.numeric(result), as.numeric(expected$as_array()))
})

test_that("ttsv with n = -2 keeps two leading modes and returns a matrix", {
  x <- tensor(array(1:27, dim = c(3, 3, 3)))
  v <- c(1, 2, 3)

  result <- ttsv(x, v, n = -2)

  expect_true(is.matrix(result))
  expect_equal(dim(result), c(3L, 3L))
})

test_that("ttsv rejects non-symmetric mode lengths", {
  x <- tensor(array(1:24, dim = c(4, 3, 2)))
  expect_error(ttsv(x, c(1, 2, 3, 4)), "same length")
})

test_that("ttsv rejects mismatched vector length", {
  x <- tensor(array(1:27, dim = c(3, 3, 3)))
  expect_error(ttsv(x, c(1, 2)), "Vector length must match")
})

test_that("ttsv rejects positive n", {
  x <- tensor(array(1:27, dim = c(3, 3, 3)))
  expect_error(ttsv(x, c(1, 2, 3), n = 1), "non-positive")
})

test_that("ttsv rejects n that exceeds tensor order", {
  x <- tensor(array(1:27, dim = c(3, 3, 3)))
  expect_error(ttsv(x, c(1, 2, 3), n = -4), "Cannot keep more modes")
})
