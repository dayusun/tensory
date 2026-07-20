make_t <- function(seed = 1, dims = c(4, 3, 5), ranks = c(2, 2, 3)) {
  set.seed(seed)
  core <- tensor(array(rnorm(prod(ranks)), dim = ranks))
  U <- Map(function(d, r) qr.Q(qr(matrix(rnorm(d * r), d, r))), dims, ranks)
  ttensor(core, U)
}

test_that("permute.TTensor reorders modes", {
  T1 <- make_t()
  Tp <- permute(T1, c(3, 1, 2))
  expect_equal(Tp$dim(), T1$dim()[c(3, 1, 2)])
  expect_equal(as.tensor(Tp)$data, aperm(as.tensor(T1)$data, c(3, 1, 2)),
               tolerance = 1e-12)
})

test_that("fnorm.TTensor matches dense and the orthonormal shortcut", {
  T1 <- make_t(2)
  expect_equal(fnorm(T1), fnorm(as.tensor(T1)), tolerance = 1e-10)
  # Orthonormal factors: equals the core norm.
  expect_equal(fnorm(T1), fnorm(T1$core), tolerance = 1e-10)

  # Non-orthonormal factors still match dense.
  set.seed(3)
  U <- lapply(c(4, 3, 5), function(d) matrix(rnorm(d * 2), d, 2))
  core <- tensor(array(rnorm(8), dim = c(2, 2, 2)))
  T2 <- ttensor(core, U)
  expect_equal(fnorm(T2), fnorm(as.tensor(T2)), tolerance = 1e-10)
})

test_that("innerprod.TTensor matches dense for all pairings", {
  T1 <- make_t(4)
  T2 <- make_t(5, ranks = c(3, 2, 2))
  expect_equal(innerprod(T1, T2),
               innerprod(as.tensor(T1), as.tensor(T2)), tolerance = 1e-10)

  set.seed(6)
  Y <- tensor(array(rnorm(60), dim = c(4, 3, 5)))
  expect_equal(innerprod(T1, Y), innerprod(as.tensor(T1), Y),
               tolerance = 1e-10)

  K <- ktensor(c(1.5, 0.5),
               lapply(c(4, 3, 5), function(d) matrix(rnorm(d * 2), d, 2)))
  expect_equal(innerprod(T1, K), innerprod(as.tensor(T1), as.tensor(K)),
               tolerance = 1e-10)
  expect_equal(innerprod(K, T1), innerprod(T1, K), tolerance = 1e-12)
})

test_that("nvecs.TTensor matches dense nvecs", {
  T1 <- make_t(7)
  Xd <- as.tensor(T1)
  for (n in 1:3) {
    expect_equal(abs(nvecs(T1, n, r = 2)), abs(nvecs(Xd, n, r = 2)),
                 tolerance = 1e-8)
  }
})
