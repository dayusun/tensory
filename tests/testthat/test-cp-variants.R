# A noiseless (or near-noiseless) low-rank problem every variant should nail.
make_lowrank <- function(seed, dims = c(6, 5, 4), R = 2, nonneg = FALSE,
                         noise = 0) {
  set.seed(seed)
  gen <- if (nonneg) function(n) runif(n, 0.1, 1) else rnorm
  U <- lapply(dims, function(d) matrix(gen(d * R), d, R))
  K <- ktensor(rep(1, R), U)
  X <- as.tensor(K)
  if (noise > 0) {
    X <- X + tensor(array(rnorm(prod(dims), sd = noise), dim = dims))
  }
  list(X = X, K = K)
}

fit_of <- function(K, X) {
  1 - fnorm(as.tensor(K) - X) / fnorm(X)
}

test_that("the .kr_others row ordering matches the mode-n unfolding", {
  set.seed(1)
  X <- tensor(array(rnorm(60), dim = c(3, 4, 5)))
  U <- lapply(c(3, 4, 5), function(d) matrix(rnorm(d * 2), d, 2))
  for (n in 1:3) {
    direct <- unfold(X, rdims = n) %*% tensory:::.kr_others(U, n)
    expect_equal(direct, mttkrp(X, U, mode = n), tolerance = 1e-10)
  }
})

test_that("cp_nmu recovers a nonnegative low-rank tensor", {
  prob <- make_lowrank(1, nonneg = TRUE)
  K <- cp_nmu(prob$X, R = 2, maxiters = 500L, tol = 1e-10)
  expect_gt(fit_of(K, prob$X), 0.99)
  expect_true(all(unlist(K$U) >= 0))
  expect_error(cp_nmu(prob$X - 10, R = 2), "nonnegative")
})

test_that("cp_apr fits Poisson count data", {
  set.seed(2)
  dims <- c(6, 5, 4)
  U <- lapply(dims, function(d) matrix(runif(d * 2, 0.2, 1.5), d, 2))
  M <- as.tensor(ktensor(c(20, 10), U))
  X <- tensor(array(rpois(prod(dims), as.vector(M$data)), dim = dims))

  K <- cp_apr(X, R = 2, maxiters = 100L)
  expect_true(all(K$lambda >= 0))
  expect_true(all(unlist(K$U) >= 0))
  # The Poisson model mean should correlate strongly with the truth.
  expect_gt(cor(as.vector(as.tensor(K)$data), as.vector(M$data)), 0.9)
  expect_error(cp_apr(X - 100, R = 2), "nonnegative")
})

test_that("cp_opt matches cp_als quality on a noiseless problem", {
  prob <- make_lowrank(3)
  set.seed(4)
  K <- cp_opt(prob$X, R = 2, init = "nvecs", maxiters = 1000L, factr = 1e1)
  expect_gt(fit_of(K, prob$X), 0.9999)
})

test_that("cp_opt supports nonnegativity via lower bound", {
  prob <- make_lowrank(5, nonneg = TRUE)
  set.seed(6)
  K <- cp_opt(prob$X, R = 2, lower = 0, maxiters = 1000L, factr = 1e1)
  expect_true(all(unlist(K$U) >= 0))
  expect_gt(fit_of(K, prob$X), 0.99)
})

test_that("cp_wopt recovers a low-rank tensor with missing entries", {
  prob <- make_lowrank(7, dims = c(8, 7, 6), R = 2)
  set.seed(8)
  W <- tensor(array(rbinom(prod(c(8, 7, 6)), 1, 0.75), dim = c(8, 7, 6)))
  Xobs <- tensor(W$data * prob$X$data)

  K <- cp_wopt(Xobs, W, R = 2, init = "nvecs", maxiters = 2000L, factr = 1e1)
  # Fit measured on ALL entries, including the unobserved ones.
  expect_gt(fit_of(K, prob$X), 0.98)
})

test_that("cp_arls converges on a low-rank problem", {
  prob <- make_lowrank(9, dims = c(10, 9, 8), R = 2)
  set.seed(10)
  K <- cp_arls(prob$X, R = 2, maxiters = 100L, tol = 1e-8)
  expect_gt(fit_of(K, prob$X), 0.99)
})

test_that("variants accept explicit initial factors", {
  prob <- make_lowrank(11, nonneg = TRUE)
  U0 <- lapply(c(6, 5, 4), function(d) matrix(runif(d * 2), d, 2))
  expect_s3_class(cp_nmu(prob$X, 2, maxiters = 3, init = U0), "KTensor")
  expect_s3_class(cp_apr(prob$X, 2, maxiters = 3, init = U0), "KTensor")
  expect_s3_class(cp_opt(prob$X, 2, maxiters = 5, init = U0), "KTensor")
  W <- tensor(array(1, dim = c(6, 5, 4)))
  expect_s3_class(cp_wopt(prob$X, W, 2, maxiters = 5, init = U0), "KTensor")
  expect_s3_class(cp_arls(prob$X, 2, maxiters = 3, init = U0), "KTensor")
})
