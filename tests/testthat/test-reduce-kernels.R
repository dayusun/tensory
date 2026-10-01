test_that("comparison and logical operators keep the dim attribute", {
  X <- tensor(array(1:8, c(2, 2, 2)))
  Y <- tensor(array(c(1, 0), c(2, 2, 2)))
  for (r in list(X == Y, X != Y, X < Y, X <= Y, X > Y, X >= Y, X & Y, X | Y,
                 !X, X == 3)) {
    expect_identical(dim(r$data), c(2L, 2L, 2L))
    expect_identical(r$dims, c(2L, 2L, 2L))
    expect_true(is.double(r$data))
  }
  expect_equal(as.vector((X > 4)$data), as.numeric(1:8 > 4))
  # a compiled kernel reads the shape from the dim attribute
  out <- ttm(X > 4, matrix(1, 3, 2), mode = 2)
  expect_identical(out$dim(), c(2L, 3L, 2L))
})

test_that("Tensor$new keeps plain double arrays and matches array() otherwise", {
  a <- array(as.double(1:24), c(2, 3, 4))
  t1 <- Tensor$new(a)
  expect_identical(t1$data, a)
  # copy-on-modify still protects the caller's array
  a[1] <- 100
  expect_identical(t1$data[1], 1)

  v <- as.double(1:24)
  t2 <- Tensor$new(v, c(2, 3, 4))
  expect_identical(t2$data, array(v, dim = c(2L, 3L, 4L)))
  named <- array(as.double(1:6), c(2, 3), dimnames = list(c("a", "b"), NULL))
  expect_identical(Tensor$new(named)$data, array(as.double(1:6), dim = c(2L, 3L)))
  expect_identical(Tensor$new(1:6, c(2, 3))$data, array(as.double(1:6), c(2L, 3L)))
})

test_that("fnorm, innerprod and ttt inner products match sum()", {
  set.seed(1)
  X <- tensor(array(rnorm(60), c(3, 4, 5)))
  Y <- tensor(array(rnorm(60), c(3, 4, 5)))
  expect_equal(fnorm(X), sqrt(sum(X$data^2)), tolerance = 1e-14)
  expect_equal(X$fnorm(), sqrt(sum(X$data^2)), tolerance = 1e-14)
  expect_equal(innerprod(X, Y), sum(X$data * Y$data), tolerance = 1e-14)
  expect_equal(as.numeric(ttt(X, Y, 1:3, 1:3)$data), sum(X$data * Y$data),
               tolerance = 1e-14)
  # NA/NaN keep base R semantics
  Z <- X$clone_tensor()
  Z$data[2] <- NA
  expect_identical(innerprod(Z, Y), sum(Z$data * Y$data))
  expect_true(is.na(fnorm(Z)))
  Z$data[2] <- NaN
  expect_identical(innerprod(Z, Y), sum(Z$data * Y$data))
})

test_that("isequal compares values without regard to attribute copies", {
  a <- array(as.double(1:6), c(2, 3))
  expect_true(isequal(tensor(a), tensor(a)))
  expect_false(isequal(tensor(a), tensor(a + 1)))
  expect_false(isequal(tensor(a), tensor(array(a, c(3, 2)))))
})

test_that("collapse with sum matches apply() in every keep order", {
  set.seed(2)
  X <- tensor(array(rnorm(2 * 3 * 4 * 5), c(2, 3, 4, 5)))
  ref <- function(keep) {
    r <- apply(X$data, keep, sum)
    if (is.null(dim(r))) r <- array(r, dim = length(r))
    r
  }
  for (dims in list(1, 2, c(1, 3), c(2, 4), c(1, 2, 3), -c(3, 1), -c(4, 2, 1), -2)) {
    keep <- if (all(dims < 0)) abs(dims) else setdiff(1:4, dims)
    got <- collapse(X, dims)
    expect_equal(got$data, ref(keep), tolerance = 1e-12, ignore_attr = FALSE)
    expect_identical(as.integer(got$dim()), as.integer(dim(ref(keep))))
  }
  # singleton kept modes survive
  S <- tensor(array(rnorm(15), c(1, 5, 3)))
  expect_identical(collapse(S, 3)$dim(), c(1L, 5L))
  # NA falls back to apply(), and other functions still use apply()
  Z <- X$clone_tensor()
  Z$data[1] <- NA
  expect_identical(collapse(Z, 2)$data, apply(Z$data, c(1, 3, 4), sum))
  expect_equal(collapse(X, 2, fun = max)$data, apply(X$data, c(1, 3, 4), max))
  expect_equal(collapse(Z, 2, na.rm = TRUE)$data,
               apply(Z$data, c(1, 3, 4), sum, na.rm = TRUE))
})

test_that("t_scale matches sweep() for single, unsorted and all modes", {
  set.seed(3)
  X <- tensor(array(rnorm(2 * 3 * 4), c(2, 3, 4)))
  for (dims in list(1, 2, 3, c(1, 3), c(3, 1), c(2, 3, 1), 1:3, -2)) {
    used <- if (all(dims < 0)) setdiff(1:3, abs(dims)) else dims
    s <- rnorm(prod(X$dim()[used]))
    ref <- sweep(X$data, used, array(s, dim = X$dim()[used]), "*")
    expect_equal(t_scale(X, s, dims)$data, ref, tolerance = 1e-15)
  }
})

test_that("symmetrize matches the permutation average and is exactly symmetric", {
  set.seed(4)
  perm_mean <- function(a, grp) {
    n <- length(dim(a))
    perms <- function(v) {
      if (length(v) <= 1) return(list(v))
      do.call(c, lapply(seq_along(v), function(i)
        lapply(perms(v[-i]), function(p) c(v[i], p))))
    }
    ps <- perms(grp)
    acc <- 0
    for (p in ps) {
      full <- seq_len(n)
      full[grp] <- p
      acc <- acc + aperm(a, full)
    }
    acc / length(ps)
  }
  a <- array(rnorm(4 * 4 * 3 * 4), c(4, 4, 3, 4))
  X <- tensor(a)
  for (grp in list(c(1, 2), c(1, 2, 4), c(2, 4))) {
    got <- symmetrize(X, list(grp))
    expect_equal(got$data, perm_mean(a, grp), tolerance = 1e-13)
    expect_true(issymmetric(got, list(grp)))
  }
  # two groups at once
  b <- array(rnorm(3^4), c(3, 3, 3, 3))
  got <- symmetrize(tensor(b), list(c(1, 2), c(3, 4)))
  expect_equal(got$data, perm_mean(perm_mean(b, c(1, 2)), c(3, 4)),
               tolerance = 1e-13)
  expect_true(issymmetric(got, list(c(1, 2), c(3, 4))))
})

test_that("gram_cpp matches tcrossprod of the unfolding on every branch", {
  skip_if_not(exists("gram_cpp", mode = "function"))
  set.seed(5)
  # mode 1 (single dsyrk), M1 in {2, 3} (gathered), M1 >= 4 (per slice)
  for (d in list(c(5, 4, 3), c(2, 6, 3), c(3, 6, 2), c(4, 6, 3), c(2, 3, 4, 5),
                 c(7, 1, 6))) {
    X <- tensor(array(rnorm(prod(d)), d))
    for (m in seq_along(d)) {
      ref <- tcrossprod(unfold(X, rdims = m))
      expect_equal(gram_cpp(X$data, m), ref, tolerance = 1e-12,
                   ignore_attr = TRUE)
    }
  }
})

test_that("nvecs returns the leading left singular vectors", {
  set.seed(6)
  X <- tensor(array(rnorm(6 * 5 * 4), c(6, 5, 4)))
  for (m in 1:3) {
    u <- nvecs(X, m, 2)
    sv <- svd(unfold(X, rdims = m))$u[, 1:2]
    # same vectors up to sign; nvecs flips each to make its largest entry > 0
    for (k in 1:2) {
      s <- sign(sv[which.max(abs(sv[, k])), k])
      expect_equal(u[, k], s * sv[, k], tolerance = 1e-8)
    }
    expect_equal(crossprod(u), diag(2), tolerance = 1e-12)
  }
  expect_error(nvecs(X, 3, 5), "rank bound")
})

test_that("KTensor full() matches the sum of rank-one outer products", {
  set.seed(7)
  brute <- function(lambda, U) {
    acc <- 0
    for (r in seq_along(lambda)) {
      op <- U[[1]][, r]
      for (k in seq_along(U)[-1]) op <- outer(op, U[[k]][, r])
      acc <- acc + lambda[r] * op
    }
    array(acc, dim = vapply(U, nrow, integer(1)))
  }
  # mode 1 larger, mode N larger, rank above every extent, order 2-4
  for (d in list(c(6, 3), c(3, 7), c(5, 4, 3), c(2, 3, 6), c(2, 2, 2, 3))) {
    for (R in c(1L, 3L, 9L)) {
      U <- lapply(d, function(n) matrix(rnorm(n * R), n, R))
      lambda <- rnorm(R)
      got <- as.tensor(ktensor(lambda, U))
      expect_equal(got$data, brute(lambda, U), tolerance = 1e-12,
                   ignore_attr = FALSE)
      expect_identical(got$dim(), as.integer(d))
    }
  }
  # order 1
  U1 <- list(matrix(rnorm(8), 4, 2))
  expect_equal(as.vector(as.tensor(ktensor(c(2, -1), U1))$data),
               as.vector(U1[[1]] %*% c(2, -1)))
})

test_that("hosvd factors span the leading singular subspaces", {
  set.seed(8)
  X <- tensor(array(rnorm(6 * 5 * 4), c(6, 5, 4)))
  T1 <- hosvd(X, ranks = c(3, 2, 2), sequential = FALSE)
  for (n in 1:3) {
    u_svd <- svd(unfold(X, rdims = n))$u[, seq_len(ncol(T1$U[[n]])), drop = FALSE]
    # same subspace: projectors agree
    expect_equal(tcrossprod(T1$U[[n]]), tcrossprod(u_svd), tolerance = 1e-8)
  }
  # full rank reconstructs exactly; tol-based ranks still respect the bound
  Tfull <- hosvd(X, ranks = c(6, 5, 4))
  expect_equal(as.tensor(Tfull)$data, X$data, tolerance = 1e-10)
  Ttol <- hosvd(X, tol = 0.5)
  rel <- fnorm(as.tensor(Ttol) - X) / fnorm(X)
  expect_lte(rel, 0.5 + 1e-8)
})
