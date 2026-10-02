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

test_that("two-step mttkrp kernels match the elementwise kernel on every branch", {
  skip_if_not(exists("mttkrp_blas_cpp", mode = "function"))
  set.seed(7)
  # Mode 1 / last mode (single dgemm), middle modes with M1 <= M2 and
  # M1 > M2, orders 2-6, rank 1, and singleton modes.
  shapes <- list(c(6, 5), c(4, 3, 5), c(7, 3, 2), c(2, 3, 9), c(3, 4, 2, 5),
                 c(2, 3, 2, 3, 2), c(5, 1, 4), c(1, 6, 1), c(1, 1, 5, 3),
                 c(4, 2, 1, 1), c(1, 3, 1, 4, 1), c(2, 1, 3, 2, 1, 2),
                 c(3, 2, 2, 2, 2, 3))
  for (d in shapes) {
    for (R in c(1L, 3L)) {
      X <- array(stats::rnorm(prod(d)), dim = d)
      U <- lapply(d, function(n) matrix(stats::rnorm(n * R), n, R))
      ref <- lapply(seq_along(d), function(n) mttkrp_cpp(X, U, n))
      for (n in seq_along(d)) {
        expect_equal(mttkrp_blas_cpp(X, U, n), ref[[n]], tolerance = 1e-12)
      }
      expect_equal(mttkrps_cpp(X, U), ref, tolerance = 1e-12)
    }
  }
})

test_that("mttkrp kernels handle empty extents and ignore the skipped factor", {
  skip_if_not(exists("mttkrp_blas_cpp", mode = "function"))
  X <- array(stats::rnorm(24), dim = c(2, 3, 4))
  U <- list(matrix(1, 2, 2), matrix(2, 3, 2), matrix(3, 4, 2))
  # The mode-n factor is never read, so it may have any shape.
  U_bad <- U
  U_bad[[2]] <- matrix(0, 1, 1)
  expect_equal(mttkrp_blas_cpp(X, U_bad, 2L), mttkrp_blas_cpp(X, U, 2L))

  Z <- array(numeric(0), dim = c(0, 3, 4))
  UZ <- list(matrix(1, 0, 2), matrix(2, 3, 2), matrix(3, 4, 2))
  expect_equal(mttkrp_blas_cpp(Z, UZ, 2L), matrix(0, 3, 2))
  expect_equal(mttkrp_blas_cpp(Z, UZ, 1L), matrix(0, 0, 2))
  expect_equal(mttkrps_cpp(Z, UZ),
               list(matrix(0, 0, 2), matrix(0, 3, 2), matrix(0, 4, 2)))
  expect_error(mttkrps_cpp(X, U[1:2]), "same length")
  expect_error(mttkrp_blas_cpp(X, list(U[[1]], U[[2]], matrix(1, 4, 3)), 1L),
               "same number of columns")
})

test_that("dimension-tree partials reproduce mttkrp for every split and mode", {
  skip_if_not(exists("mttkrp_partial_cpp", mode = "function"))
  set.seed(11)
  for (d in list(c(4, 3, 5), c(2, 3, 4, 5), c(3, 1, 2, 4, 2), c(1, 5, 1, 3))) {
    N <- length(d)
    X <- array(stats::rnorm(prod(d)), dim = d)
    U <- lapply(d, function(n) matrix(stats::rnorm(n * 3), n, 3))
    for (s in seq_len(N - 1L)) {
      PL <- mttkrp_partial_cpp(X, U, s, TRUE)
      PR <- mttkrp_partial_cpp(X, U, s, FALSE)
      expect_identical(dim(PL), c(as.integer(prod(d[seq_len(s)])), 3L))
      for (n in seq_len(N)) {
        P <- if (n <= s) PL else PR
        expect_equal(mttkrp_finish_cpp(P, U, as.integer(d), s, n),
                     mttkrp_cpp(X, U, n), tolerance = 1e-12)
      }
    }
  }
  expect_error(mttkrp_partial_cpp(array(1, c(2, 2, 2)), rep(list(matrix(1, 2, 1)), 3), 3L, TRUE),
               "non-empty groups")
  expect_error(mttkrp_finish_cpp(matrix(1, 3, 1), rep(list(matrix(1, 2, 1)), 3),
                                 c(2L, 2L, 2L), 1L, 1L), "does not match")
})

test_that("cp_als with the dimension tree matches per-mode mttkrp", {
  skip_if_not(exists("mttkrp_partial_cpp", mode = "function"))
  set.seed(12)
  run <- function(X, dimorder, tree) {
    old <- options(tensory.cp_dimtree = tree)
    on.exit(options(old))
    set.seed(99)
    cp_als(X, R = 3L, maxiters = 15L, tol = 0, dimorder = dimorder,
           fixsigns = FALSE)
  }
  for (d in list(c(6, 5, 4), c(4, 3, 5, 2), c(3, 2, 2, 3, 2))) {
    X <- tensor(array(stats::rnorm(prod(d)), dim = d))
    N <- length(d)
    for (ord in list(seq_len(N), rev(seq_len(N)))) {
      a <- run(X, ord, TRUE)
      b <- run(X, ord, FALSE)
      expect_equal(a$lambda, b$lambda, tolerance = 1e-8)
      for (k in seq_len(N)) expect_equal(a$U[[k]], b$U[[k]], tolerance = 1e-8)
    }
  }
  # any other update order falls back to per-mode mttkrp exactly
  X <- tensor(array(stats::rnorm(60), dim = c(3, 4, 5)))
  a <- run(X, c(2L, 1L, 3L), TRUE)
  b <- run(X, c(2L, 1L, 3L), FALSE)
  expect_identical(a$lambda, b$lambda)
})

test_that("cp_nmu, cp_opt, cp_wopt and gcp_opt match their per-mode paths", {
  skip_if_not(exists("mttkrp_partial_cpp", mode = "function"))
  with_tree <- function(tree, expr) {
    old <- options(tensory.cp_dimtree = tree)
    on.exit(options(old))
    set.seed(5)
    force(expr)
  }
  same_ktensor <- function(a, b, tol = 1e-7) {
    expect_equal(a$lambda, b$lambda, tolerance = tol)
    for (k in seq_along(a$U)) expect_equal(a$U[[k]], b$U[[k]], tolerance = tol)
  }
  set.seed(21)
  for (d in list(c(6, 5, 4), c(4, 3, 3, 2))) {
    X <- tensor(array(stats::runif(prod(d)), dim = d))
    U0 <- lapply(d, function(n) matrix(stats::runif(n * 2), n, 2))
    same_ktensor(with_tree(TRUE, cp_nmu(X, 2, maxiters = 20L, tol = 0, init = U0)),
                 with_tree(FALSE, cp_nmu(X, 2, maxiters = 20L, tol = 0, init = U0)))
    same_ktensor(with_tree(TRUE, cp_opt(X, 2, init = U0, maxiters = 30L)),
                 with_tree(FALSE, cp_opt(X, 2, init = U0, maxiters = 30L)))
    W <- tensor(array(as.numeric(stats::runif(prod(d)) > 0.2), dim = d))
    same_ktensor(with_tree(TRUE, cp_wopt(X, W, 2, init = U0, maxiters = 30L)),
                 with_tree(FALSE, cp_wopt(X, W, 2, init = U0, maxiters = 30L)))
    a <- with_tree(TRUE, gcp_opt(X, 2, type = "gaussian", init = U0, maxiters = 30L))
    b <- with_tree(FALSE, gcp_opt(X, 2, type = "gaussian", init = U0, maxiters = 30L))
    expect_equal(a$objective, b$objective, tolerance = 1e-8)
    same_ktensor(a$K, b$K)
  }
  # sparse input keeps the per-mode path and agrees with the dense fit
  X <- tensor(array(stats::rnorm(60) * (stats::runif(60) > 0.5), dim = c(3, 4, 5)))
  U0 <- lapply(c(3, 4, 5), function(n) matrix(stats::rnorm(n * 2), n, 2))
  same_ktensor(cp_opt(X, 2, init = U0, maxiters = 30L),
               cp_opt(sptensor(X), 2, init = U0, maxiters = 30L))
})

test_that(".cp_fit gives the same fit from any mode's MTTKRP", {
  set.seed(22)
  X <- tensor(array(stats::rnorm(60), dim = c(3, 4, 5)))
  U <- lapply(c(3, 4, 5), function(n) matrix(stats::rnorm(n * 2), n, 2))
  lambda <- c(1.5, -0.5)
  ref <- .cp_fit(X, lambda, U, fnorm(X))
  for (n in 1:3) {
    expect_equal(.cp_fit(X, lambda, U, fnorm(X), V = mttkrp(X, U, n), n = n),
                 ref, tolerance = 1e-12)
  }
})

test_that(".fg_cached evaluates each point once for fn and gr", {
  calls <- 0
  fg <- function(v) {
    calls <<- calls + 1
    list(value = sum((v - 1)^2), gradient = 2 * (v - 1))
  }
  obj <- .fg_cached(fg)
  expect_equal(obj$fn(c(0, 0)), 2)
  expect_equal(obj$gr(c(0, 0)), c(-2, -2))
  expect_equal(calls, 1)
  obj$gr(c(1, 0))
  expect_equal(calls, 2)
  res <- stats::optim(c(0, 0), obj$fn, obj$gr, method = "L-BFGS-B")
  expect_equal(res$par, c(1, 1), tolerance = 1e-6)
})
