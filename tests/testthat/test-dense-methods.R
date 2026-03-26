library(tensory)

test_that("ttv is an alias for vector ttm", {
  x <- tensor(array(1:24, dim = c(4, 3, 2)))
  v <- 1:4

  expect_equal(ttv(x, v, mode = 1)$as_array(), ttm(x, v, mode = 1)$as_array())
})

test_that("innerprod computes Frobenius inner products", {
  x <- tensor(array(1:8, dim = c(2, 2, 2)))
  y <- tensor(array(2:9, dim = c(2, 2, 2)))

  expect_equal(innerprod(x, y), sum(as.vector(x$as_array()) * as.vector(y$as_array())))
})

test_that("nnz and find operate on dense tensors", {
  x <- tensor(array(c(1, 0, 2, 0, 0, 3), dim = c(2, 3)))

  expect_equal(nnz(x), 3)

  subs <- find(x)
  expect_equal(subs, cbind(mode1 = c(1L, 1L, 2L), mode2 = c(1L, 2L, 3L)))

  found <- find(x, values = TRUE)
  expect_equal(found$subs, subs)
  expect_equal(found$vals, c(1, 2, 3))
})

test_that("permute reshapes axes without changing values", {
  x <- tensor(array(1:24, dim = c(2, 3, 4)))
  y <- permute(x, c(3, 1, 2))

  expect_equal(y$dim(), c(4L, 2L, 3L))
  expect_equal(y$as_array(), aperm(x$as_array(), c(3, 1, 2)))
})

test_that("reshape and squeeze are exposed as dense functions", {
  x <- tensor(array(1:24, dim = c(2, 3, 4)))
  y <- reshape(x, c(6, 4))

  expect_equal(y$dim(), c(6L, 4L))
  expect_equal(as.vector(y$as_array()), as.vector(x$as_array()))

  base_array <- array(1:24, dim = c(2, 3, 4))
  reshaped_array <- reshape(base_array, c(4, 6))
  expect_equal(dim(reshaped_array), c(4L, 6L))
  expect_equal(as.vector(reshaped_array), as.vector(base_array))

  z <- tensor(array(1:6, dim = c(1, 2, 1, 3)))
  expect_equal(squeeze(z)$dim(), c(2L, 3L))
})

test_that("unfold and vec match tensor toolbox conventions", {
  x <- tensor(array(1:24, dim = c(2, 3, 4)))

  expect_equal(vec(x), as.vector(x$as_array()))
  expect_equal(unfold(x), as.vector(x$as_array()))
  expect_equal(unfold(x, 2), as.matrix(tenmat(x, rdims = 2)))
  expect_equal(unfold(x, c(1, 3), 2), as.matrix(tenmat(x, rdims = c(1, 3), cdims = 2)))
})

test_that("nvecs returns leading left singular vectors of a mode unfolding", {
  x <- tensor(array(c(3, 0, 0, 1, 0, 0, 0, 2), dim = c(2, 2, 2)))

  actual <- nvecs(x, mode = 1, r = 1)
  X1 <- unfold(x, 1)
  expected <- svd(X1, nu = 1, nv = 0)$u

  if (actual[which.max(abs(actual[, 1])), 1] < 0) {
    actual[, 1] <- -actual[, 1]
  }
  if (expected[which.max(abs(expected[, 1])), 1] < 0) {
    expected[, 1] <- -expected[, 1]
  }

  expect_equal(actual, expected)
})

test_that("mttkrp matches a direct column-wise contraction", {
  x <- tensor(array(1:8, dim = c(2, 2, 2)))
  U <- list(
    matrix(c(1, 0, 0, 1), nrow = 2),
    matrix(c(2, 1, 1, 2), nrow = 2),
    matrix(c(1, 3, 2, 4), nrow = 2)
  )

  actual <- mttkrp(x, U, mode = 2)
  expected <- cbind(
    as.vector(ttv(x, list(U[[1]][, 1], U[[3]][, 1]), mode = c(1, 3))$as_array()),
    as.vector(ttv(x, list(U[[1]][, 2], U[[3]][, 2]), mode = c(1, 3))$as_array())
  )

  expect_equal(actual, expected)
})

test_that("mttkrps returns the full sequence of mode products", {
  x <- tensor(array(1:8, dim = c(2, 2, 2)))
  U <- list(
    matrix(c(1, 0, 0, 1), nrow = 2),
    matrix(c(2, 1, 1, 2), nrow = 2),
    matrix(c(1, 3, 2, 4), nrow = 2)
  )

  actual <- mttkrps(x, U)
  expect_length(actual, 3)
  expect_equal(actual[[1]], mttkrp(x, U, mode = 1))
  expect_equal(actual[[2]], mttkrp(x, U, mode = 2))
  expect_equal(actual[[3]], mttkrp(x, U, mode = 3))
})

test_that("contract traces over two tensor modes", {
  x <- tensor(array(1:8, dim = c(2, 2, 2)))
  actual <- contract(x, 1, 2)
  expected <- tensor(c(5, 13), dims = 2)

  expect_equal(actual$as_array(), expected$as_array())
})

test_that("mask extracts values at nonzero mask positions", {
  x <- tensor(array(1:8, dim = c(2, 2, 2)))
  w <- tensor(array(c(1, 0, 0, 1, 0, 0, 1, 0), dim = c(2, 2, 2)))

  expect_equal(mask(x, w), c(1, 4, 7))
})

test_that("fibers extracts requested mode-k fibers", {
  x <- tensor(array(1:8, dim = c(2, 2, 2)))
  midx <- rbind(c(1L, 1L), c(2L, 2L))

  actual <- fibers(x, mode = 2, midx = midx)
  expected <- cbind(x$as_array()[1, , 1], x$as_array()[2, , 2])

  expect_equal(actual, expected)
})

test_that("tenfun applies elementwise tensor functions", {
  x <- tensor(array(1:8, dim = c(2, 2, 2)))
  y <- tensor(array(8:1, dim = c(2, 2, 2)))

  expect_equal(tenfun(`+`, x, y)$as_array(), array(rep(9, 8), dim = c(2, 2, 2)))
  expect_equal(tenfun(function(a) a + 1, x)$as_array(), x$as_array() + 1)
})

test_that("ttsv contracts with the same vector in multiple modes", {
  x <- tensor(array(1:8, dim = c(2, 2, 2)))
  v <- c(1, 2)

  expect_equal(ttsv(x, v), as.numeric(ttv(x, list(v, v, v), mode = c(1, 2, 3))$as_array()))
  expect_equal(ttsv(x, v, n = -1), ttv(x, list(v, v), mode = c(2, 3))$as_array())
})

test_that("symmetrize averages over permutation classes", {
  x <- tensor(array(c(1, 2, 3, 4, 5, 6, 7, 8), dim = c(2, 2, 2)))
  y <- symmetrize(x)

  expect_equal(y$as_array()[1, 2, 1], y$as_array()[2, 1, 1])
  expect_equal(y$as_array()[1, 2, 2], y$as_array()[2, 1, 2])
  expect_equal(y$as_array()[2, 1, 2], y$as_array()[2, 2, 1])
})

test_that("issymmetric detects symmetric tensors and grouped symmetry", {
  x <- tensor(array(1:8, dim = c(2, 2, 2)))
  y <- symmetrize(x)

  expect_true(issymmetric(y))
  expect_true(issymmetric(y, grps = c(1, 2)))
  expect_false(issymmetric(x))
})

test_that("full, double, isequal, isscalar, scale, and transpose wrappers work", {
  x <- tensor(array(1:4, dim = c(2, 2)))
  y <- tensor(array(1:4, dim = c(2, 2)))
  z <- tensor(array(1:4, dim = c(4)))

  expect_equal(full(x), x$as_array())
  expect_equal(double.Tensor(x), x$as_array())
  expect_equal(as.double(x), as.double(x$as_array()))
  expect_true(isequal(x, y))
  expect_false(isequal(x, z))
  expect_true(isscalar(tensor(5)))
  expect_true(isscalar(ttt(x, x, dimsA = c(1, 2), dimsB = c(1, 2))))
  expect_equal(max(x)$dim(), integer(0))
  expect_equal(scale(x, c(10, 20), dims = 1)$as_array(), t_scale(x, c(10, 20), dims = 1)$as_array())
  expect_error(transpose(x), "not defined")
})
