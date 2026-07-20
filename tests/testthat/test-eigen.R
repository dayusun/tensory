# Kofidis & Regalia example tensor (m = 4, n = 3), the standard SS-HOPM
# benchmark; values listed in lexicographic order of the sorted index classes.
kofidis_regalia <- function() {
  s <- SymTensor$new()
  s$m <- 4L
  s$n <- 3L
  s$vals <- c(0.2883, -0.0031, 0.1973, -0.2485, -0.2939, 0.3847,
              0.2972, 0.1862, 0.0919, -0.3619, 0.1241, -0.3420,
              0.2127, 0.2727, -0.3054)
  s$full()
}

test_that("eig_sshopm reduces to the matrix power method for m = 2", {
  set.seed(1)
  M <- crossprod(matrix(rnorm(25), 5, 5)) # SPD, so lambda_max is dominant
  A <- tensor(M)
  res <- eig_sshopm(A, tol = 1e-12)
  expect_true(res$converged)
  expect_equal(res$lambda, max(eigen(M)$values), tolerance = 1e-8)
})

test_that("eig_sshopm returns genuine Z-eigenpairs", {
  set.seed(2)
  A <- symmetrize(tensor(array(rnorm(81), dim = c(3, 3, 3, 3))))
  res <- eig_sshopm(A, tol = 1e-12)
  expect_true(res$converged)
  resid <- as.numeric(ttsv(A, res$x, -1)) - res$lambda * res$x
  expect_lt(sqrt(sum(resid^2)), 1e-6)
  expect_equal(sum(res$x^2), 1, tolerance = 1e-12)
})

test_that("eig_sshopm finds the known Kofidis-Regalia eigenvalues", {
  A <- kofidis_regalia()

  set.seed(3)
  maxima <- replicate(20, eig_sshopm(A, maxiters = 1000L)$lambda)
  expect_lt(min(abs(maxima - 0.8893)), 1e-3)

  set.seed(4)
  minima <- replicate(20, eig_sshopm(A, maximize = FALSE,
                                     maxiters = 1000L)$lambda)
  expect_lt(min(abs(minima - (-1.0954))), 1e-3)

  # Fixed shift (alpha = 2) also converges, as in the original SS-HOPM.
  set.seed(5)
  res <- eig_sshopm(A, shift = 2, tol = 1e-13, maxiters = 2000L)
  expect_true(res$converged)
  resid <- as.numeric(ttsv(A, res$x, -1)) - res$lambda * res$x
  expect_lt(sqrt(sum(resid^2)), 1e-5)
})

test_that("eig_geap with B = teneye matches eig_sshopm", {
  A <- kofidis_regalia()
  E <- teneye(4, 3)

  set.seed(6)
  x0 <- rnorm(3)
  res_g <- eig_geap(A, E, start = x0, tol = 1e-12, maxiters = 1000L)
  res_s <- eig_sshopm(A, start = x0, tol = 1e-12, maxiters = 1000L)
  expect_true(res_g$converged)
  expect_equal(res_g$lambda, res_s$lambda, tolerance = 1e-6)

  # Generalized residual A x^(m-1) = lambda B x^(m-1)
  resid <- as.numeric(ttsv(A, res_g$x, -1)) -
    res_g$lambda * as.numeric(ttsv(E, res_g$x, -1))
  expect_lt(sqrt(sum(resid^2)), 1e-6)
})

test_that("eig_geap computes H-eigenpairs with a diagonal B", {
  A <- kofidis_regalia()
  B <- tendiag(rep(1, 3), rep(3, 4)) # B x^(m-1) = x^[m-1] elementwise

  set.seed(7)
  res <- eig_geap(A, B, tol = 1e-13, maxiters = 2000L)
  expect_true(res$converged)
  resid <- as.numeric(ttsv(A, res$x, -1)) - res$lambda * res$x^3
  expect_lt(sqrt(sum(resid^2)), 1e-5)
})

test_that("eigensolvers validate their inputs", {
  X <- tensor(array(rnorm(27), dim = c(3, 3, 3)))
  expect_error(eig_sshopm(X), "symmetric")
  expect_error(eig_sshopm(tensor(array(rnorm(12), dim = c(3, 4)))), "same size")
  A <- symmetrize(X)
  B <- symmetrize(tensor(array(rnorm(16), dim = c(4, 4))))
  expect_error(eig_geap(A, B), "same dimensions|same size")
})
