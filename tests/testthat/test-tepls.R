# A rank-one-per-mode coefficient tensor with iid standard-normal predictors:
# Sigma_k = I, so the mode-k envelope is exactly 1-dimensional and TEPLS with
# u = 1 per mode should recover the signal and predict near-perfectly.
make_reg <- function(seed, p = c(8, 6), n = 1000, sd = 0.03, r = 1) {
  set.seed(seed)
  X <- lapply(seq_len(n), function(i) array(rnorm(prod(p)), dim = p))
  a <- lapply(p, function(pk) { v <- rnorm(pk); v / sqrt(sum(v^2)) })
  Blist <- lapply(seq_len(r), function(j) {
    outer(a[[1]] * (1 + 0.3 * j), a[[2]])
  })
  Y <- sapply(seq_len(r), function(j) {
    vapply(X, function(xi) sum(Blist[[j]] * xi), numeric(1)) + rnorm(n, sd = sd)
  })
  list(X = X, Y = if (r == 1) as.vector(Y) else Y, B = Blist, p = p, n = n, r = r)
}

r2 <- function(pred, truth) {
  1 - sum((truth - pred)^2) / sum((truth - mean(truth))^2)
}

test_that("tepls recovers a rank-one signal and predicts a scalar response", {
  d <- make_reg(1)
  fit <- tepls(d$X, d$Y, u = c(1, 1))
  expect_s3_class(fit, "tepls")
  expect_equal(dim(fit$coef$as_array()), d$p)

  # In-sample fit is near-perfect for a 1-dim envelope with tiny noise
  # (R^2 tracks cosine^2 of the estimated vs. true coefficient direction).
  expect_gt(r2(predict(fit), d$Y), 0.98)

  # Coefficient direction aligns with the truth (sign-free).
  cosine <- sum(fit$coef$as_array() * d$B[[1]]) /
    (fnorm(fit$coef) * sqrt(sum(d$B[[1]]^2)))
  expect_gt(abs(cosine), 0.98)
})

test_that("tepls predicts held-out observations", {
  d <- make_reg(2, n = 1200)
  tr <- 1:1000
  te <- 1001:1200
  fit <- tepls(d$X[tr], d$Y[tr], u = c(1, 1))
  pred <- predict(fit, d$X[te])
  expect_length(pred, length(te))
  expect_gt(r2(pred, d$Y[te]), 0.98)
})

test_that("tepls accepts an order-(m+1) tensor with observations in the last mode", {
  d <- make_reg(3)
  arr <- array(0, dim = c(d$p, d$n))
  for (i in seq_len(d$n)) arr[, , i] <- d$X[[i]]
  Xt <- tensor(arr)

  fit_tensor <- tepls(Xt, d$Y, u = c(1, 1))
  fit_list <- tepls(d$X, d$Y, u = c(1, 1))
  expect_equal(fit_tensor$coef$as_array(), fit_list$coef$as_array(),
               tolerance = 1e-10)
})

test_that("tepls handles a multivariate response", {
  d <- make_reg(4, r = 3)
  fit <- tepls(d$X, d$Y, u = c(1, 1))
  expect_equal(fit$r, 3L)
  expect_equal(dim(fit$coef$as_array()), c(d$p, 3L))

  pred <- predict(fit)
  expect_equal(dim(pred), c(d$n, 3L))
  for (j in 1:3) expect_gt(r2(pred[, j], d$Y[, j]), 0.98)
})

test_that("tepls with full envelope dims reproduces the OLS fit", {
  # Small problem where n > prod(p) so full-rank OLS is well-defined.
  set.seed(5)
  p <- c(3, 2)
  n <- 200
  X <- lapply(seq_len(n), function(i) array(rnorm(prod(p)), dim = p))
  Btrue <- array(rnorm(prod(p)), dim = p)
  y <- vapply(X, function(xi) sum(Btrue * xi), numeric(1)) + rnorm(n, sd = 0.1)

  fit <- tepls(X, y, u = p) # full envelope = no reduction
  Xmat <- t(vapply(X, as.vector, numeric(prod(p))))
  ols <- lm(y ~ Xmat)
  expect_equal(as.vector(fit$coef$as_array()), unname(coef(ols)[-1]),
               tolerance = 1e-6)
})

test_that("tepls validates its arguments", {
  d <- make_reg(6, n = 40)
  expect_error(tepls(d$X, d$Y[1:10], u = c(1, 1)), "one row")
  expect_error(tepls(d$X, d$Y, u = c(1, 1, 1)), "length 1 or ndims")
  expect_error(tepls(d$X, d$Y, u = c(1, 99)), "1 <= u")
})
