make_sp <- function(seed = 1, dims = c(4, 3, 5), nnz = 10) {
  set.seed(seed)
  sptenrand(dims, nnz)
}

test_that("sptensor aggregates duplicates and drops zeros", {
  subs <- rbind(c(1, 1, 1), c(1, 1, 1), c(2, 2, 2), c(3, 1, 4))
  S <- sptensor(subs, c(1, 2, 5, 0), c(3, 3, 4))
  expect_equal(nnz(S), 2L)
  expect_equal(S$full()$data[1, 1, 1], 3)
  expect_equal(S$full()$data[2, 2, 2], 5)
  expect_equal(S$full()$data[3, 1, 4], 0)

  expect_error(sptensor(rbind(c(5, 1, 1)), 1, c(3, 3, 4)), "out of bounds")
})

test_that("sptensor converts to and from dense", {
  set.seed(1)
  X <- tensor(array(rbinom(24, 1, 0.4) * rnorm(24), dim = c(2, 3, 4)))
  S <- sptensor(X)
  expect_equal(S$full()$data, X$data, tolerance = 1e-14)
  expect_equal(as.tensor(S)$data, X$data, tolerance = 1e-14)
  expect_equal(nnz(S), sum(X$data != 0))
  expect_equal(fnorm(S), fnorm(X), tolerance = 1e-12)
})

test_that("sptenrand hits the requested nonzero count and bounds", {
  S <- make_sp(2, c(10, 10, 10), 25)
  expect_equal(nnz(S), 25L)
  expect_true(all(S$vals >= 0 & S$vals <= 1))
  expect_true(all(S$subs >= 1 & S$subs <= 10))
})

test_that("permute / find / collapse work on sparse tensors", {
  S <- make_sp(3)
  D <- as.tensor(S)

  P <- permute(S, c(3, 1, 2))
  expect_equal(as.tensor(P)$data, aperm(D$data, c(3, 1, 2)), tolerance = 1e-14)

  f <- find(S, values = TRUE)
  expect_equal(nrow(f$subs), nnz(S))

  C <- collapse(S, dims = 3)
  expect_equal(as.tensor(C)$data, collapse(D, dims = 3)$data, tolerance = 1e-12)

  Call <- collapse(S, dims = 1:3)
  expect_true(isscalar(Call))
  expect_equal(as.numeric(Call$data), sum(S$vals), tolerance = 1e-12)
})

test_that("sparse innerprod matches dense for all pairings", {
  S <- make_sp(4)
  D <- as.tensor(S)
  set.seed(5)
  Y <- tensor(array(rnorm(60), dim = c(4, 3, 5)))
  expect_equal(innerprod(S, Y), innerprod(D, Y), tolerance = 1e-12)

  S2 <- make_sp(6)
  expect_equal(innerprod(S, S2), innerprod(D, as.tensor(S2)), tolerance = 1e-12)

  K <- ktensor(c(1.5, -0.5),
               lapply(c(4, 3, 5), function(d) matrix(rnorm(d * 2), d, 2)))
  expect_equal(innerprod(S, K), innerprod(D, as.tensor(K)), tolerance = 1e-10)
})

test_that("sparse mttkrp matches the dense kernel", {
  S <- make_sp(7)
  D <- as.tensor(S)
  set.seed(8)
  U <- lapply(c(4, 3, 5), function(d) matrix(rnorm(d * 2), d, 2))
  for (n in 1:3) {
    expect_equal(mttkrp(S, U, mode = n), mttkrp(D, U, mode = n),
                 tolerance = 1e-10)
  }
})

test_that("sparse ttv and ttm match their dense counterparts", {
  S <- make_sp(9)
  D <- as.tensor(S)
  set.seed(10)

  v <- rnorm(3)
  expect_equal(as.tensor(ttv(S, v, mode = 2))$data, ttv(D, v, mode = 2)$data,
               tolerance = 1e-12)

  vs <- list(rnorm(4), rnorm(5))
  r_sp <- ttv(S, vs, mode = c(1, 3))
  expect_equal(as.tensor(r_sp)$data, ttv(D, vs, mode = c(1, 3))$data,
               tolerance = 1e-12)

  # Full contraction returns a scalar Tensor, matching the dense convention.
  vall <- list(rnorm(4), rnorm(3), rnorm(5))
  r_all <- ttv(S, vall, mode = 1:3)
  expect_true(isscalar(r_all))
  expect_equal(as.numeric(r_all$data),
               as.numeric(ttv(D, vall, mode = 1:3)$data), tolerance = 1e-12)

  M <- matrix(rnorm(9), 3, 3)
  expect_equal(ttm(S, M, mode = 2)$data, ttm(D, M, mode = 2)$data,
               tolerance = 1e-12)
  expect_equal(ttm(S, M, mode = 2, transpose = TRUE)$data,
               ttm(D, M, mode = 2, transpose = TRUE)$data, tolerance = 1e-12)

  Ms <- list(matrix(rnorm(8), 2, 4), matrix(rnorm(10), 2, 5))
  expect_equal(ttm(S, Ms, mode = c(1, 3))$data,
               ttm(D, Ms, mode = c(1, 3))$data, tolerance = 1e-12)
})

test_that("sparse nvecs matches dense nvecs", {
  S <- make_sp(11, nnz = 20)
  D <- as.tensor(S)
  for (n in 1:3) {
    expect_equal(abs(nvecs(S, n, r = 2)), abs(nvecs(D, n, r = 2)),
                 tolerance = 1e-8)
  }
})

test_that("sparse arithmetic stays sparse where possible", {
  S1 <- make_sp(12)
  S2 <- make_sp(13)
  D1 <- as.tensor(S1)
  D2 <- as.tensor(S2)

  expect_s3_class(S1 + S2, "Sptensor")
  expect_equal(as.tensor(S1 + S2)$data, (D1 + D2)$data, tolerance = 1e-12)
  expect_equal(as.tensor(S1 - S2)$data, (D1 - D2)$data, tolerance = 1e-12)
  expect_equal(as.tensor(S1 * S2)$data, (D1 * D2)$data, tolerance = 1e-12)

  expect_s3_class(S1 * 3, "Sptensor")
  expect_equal(as.tensor(S1 * 3)$data, (D1 * 3)$data, tolerance = 1e-12)
  expect_equal(as.tensor(S1 / 2)$data, (D1 / 2)$data, tolerance = 1e-12)
  expect_equal(as.tensor(-S1)$data, (-D1)$data, tolerance = 1e-12)

  # Mixed sparse-dense densifies silently (no incompatible-methods warning).
  expect_warning(mix <- S1 + D2, NA)
  expect_s3_class(mix, "Tensor")
  expect_equal(mix$data, (D1 + D2)$data, tolerance = 1e-12)

  # Scalar addition fills in zeros, so it densifies.
  expect_s3_class(S1 + 1, "Tensor")
  expect_equal((S1 + 1)$data, (D1 + 1)$data, tolerance = 1e-12)
})

test_that("sptenmat unfolds sparse tensors correctly", {
  S <- make_sp(14)
  D <- as.tensor(S)
  A <- sptenmat(S, rdims = 2)
  expect_equal(A$dim(), c(3, 20))
  expect_equal(as.matrix(A), as.matrix(tenmat(D, rdims = 2)), tolerance = 1e-12)

  A2 <- sptenmat(S, rdims = c(1, 3))
  expect_equal(as.matrix(A2), as.matrix(tenmat(D, rdims = c(1, 3))),
               tolerance = 1e-12)
})

test_that("cp_als on a sparse tensor matches cp_als on its dense equivalent", {
  set.seed(15)
  S <- sptenrand(c(12, 10, 8), 40)
  D <- as.tensor(S)
  U0 <- lapply(c(12, 10, 8), function(d) matrix(rnorm(d * 2), d, 2))

  Ks <- cp_als(S, R = 2, maxiters = 10L, tol = 0, init = U0)
  Kd <- cp_als(D, R = 2, maxiters = 10L, tol = 0, init = U0)

  expect_equal(Ks$lambda, Kd$lambda, tolerance = 1e-8)
  for (n in 1:3) {
    expect_equal(Ks$U[[n]], Kd$U[[n]], tolerance = 1e-8)
  }
})

test_that("export_data / import_data round-trips sparse tensors", {
  tmp <- tempfile(fileext = ".txt")
  on.exit(unlink(tmp))
  S <- make_sp(16)
  export_data(S, tmp)
  S2 <- import_data(tmp)
  expect_s3_class(S2, "Sptensor")
  expect_equal(S2$dims, S$dims)
  expect_equal(as.tensor(S2)$data, as.tensor(S)$data, tolerance = 1e-12)
})
