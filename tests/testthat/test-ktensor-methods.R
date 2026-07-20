make_k <- function(seed = 1, dims = c(4, 3, 5), R = 3) {
  set.seed(seed)
  ktensor(runif(R, 0.5, 2), lapply(dims, function(d) matrix(rnorm(d * R), d, R)))
}

test_that("ncomponents / extract / redistribute behave like TTB", {
  K <- make_k()
  expect_equal(ncomponents(K), 3L)

  K2 <- extract(K, c(3, 1))
  expect_equal(ncomponents(K2), 2L)
  expect_equal(K2$lambda, K$lambda[c(3, 1)])
  expect_equal(K2$U[[2]], K$U[[2]][, c(3, 1)])

  K3 <- redistribute(K, 2)
  expect_equal(K3$lambda, rep(1, 3))
  expect_equal(as.tensor(K3)$data, as.tensor(K)$data, tolerance = 1e-12)
})

test_that("normalize produces unit-norm columns and preserves the tensor", {
  K <- make_k(2)
  Kn <- normalize(K)
  for (n in 1:3) {
    expect_equal(colSums(Kn$U[[n]]^2), rep(1, 3), tolerance = 1e-12)
  }
  expect_true(all(Kn$lambda >= 0))
  expect_equal(as.tensor(Kn)$data, as.tensor(K)$data, tolerance = 1e-12)

  K0 <- normalize(K, mode = 0)
  expect_equal(K0$lambda, rep(1, 3))
  expect_equal(as.tensor(K0)$data, as.tensor(K)$data, tolerance = 1e-12)

  K1 <- normalize(K, mode = 1)
  expect_equal(K1$lambda, rep(1, 3))
  expect_equal(as.tensor(K1)$data, as.tensor(K)$data, tolerance = 1e-12)
})

test_that("arrange sorts components by weight", {
  K <- make_k(3)
  Ka <- arrange(K)
  expect_true(all(diff(Ka$lambda) <= 0))
  expect_equal(as.tensor(Ka)$data, as.tensor(K)$data, tolerance = 1e-12)

  Kp <- arrange(K, perm = c(2, 3, 1))
  expect_equal(Kp$lambda, K$lambda[c(2, 3, 1)])
})

test_that("fixsigns flips signs without changing the tensor", {
  K <- make_k(4)
  K$U[[1]][, 1] <- -abs(K$U[[1]][, 1])
  Kf <- fixsigns(K)
  expect_equal(as.tensor(Kf)$data, as.tensor(K)$data, tolerance = 1e-12)
  for (r in 1:3) {
    col <- Kf$U[[1]][, r]
    expect_gte(col[which.max(abs(col))], 0)
  }
})

test_that("tovec stacks lambda and factors", {
  K <- make_k(5, dims = c(2, 3), R = 2)
  v <- tovec(K)
  expect_length(v, 2 + (2 + 3) * 2)
  expect_equal(v[1:2], K$lambda)

  v2 <- tovec(K, lambda = FALSE)
  expect_length(v2, (2 + 3) * 2)
})

test_that("score recovers a permuted copy with score 1", {
  K <- make_k(6)
  perm <- c(3, 1, 2)
  Kp <- ktensor(K$lambda[perm], lapply(K$U, function(M) M[, perm]))
  s <- score(Kp, K)
  expect_equal(s$score, 1, tolerance = 1e-10)
  # The matched arrangement must represent the same tensor as the reference
  # ordering up to component order.
  expect_equal(as.tensor(s$x)$data, as.tensor(K)$data, tolerance = 1e-10)

  # Score of unrelated random ktensors is below 1.
  K2 <- make_k(7)
  expect_lt(score(K2, K)$score, 0.99)
})

test_that("permute.KTensor reorders modes", {
  K <- make_k(8)
  Kp <- permute(K, c(2, 3, 1))
  expect_equal(Kp$dim(), K$dim()[c(2, 3, 1)])
  expect_equal(as.tensor(Kp)$data, aperm(as.tensor(K)$data, c(2, 3, 1)),
               tolerance = 1e-12)
})

test_that("fnorm.KTensor and innerprod.KTensor match dense computation", {
  K <- make_k(9)
  Xd <- as.tensor(K)
  expect_equal(fnorm(K), fnorm(Xd), tolerance = 1e-10)

  K2 <- make_k(10)
  expect_equal(innerprod(K, K2), innerprod(as.tensor(K), as.tensor(K2)),
               tolerance = 1e-10)

  set.seed(11)
  Y <- tensor(array(rnorm(60), dim = c(4, 3, 5)))
  expect_equal(innerprod(K, Y), innerprod(Xd, Y), tolerance = 1e-10)
})

test_that("nvecs.KTensor matches dense nvecs", {
  K <- make_k(12)
  Xd <- as.tensor(K)
  for (n in 1:3) {
    expect_equal(abs(nvecs(K, n, r = 2)), abs(nvecs(Xd, n, r = 2)),
                 tolerance = 1e-8)
  }
})
