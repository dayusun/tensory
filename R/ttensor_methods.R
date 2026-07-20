#' @include tensor_class.R ttensor_class.R ktensor_class.R tensor_dense_methods.R
NULL

#' @export
permute.TTensor <- function(x, order, ...) {
  order <- as.integer(order)
  if (length(order) != x$ndims() || !setequal(order, seq_len(x$ndims()))) {
    stop("order must be a permutation of 1:ndims(x).")
  }
  ttensor(permute(x$core, order), x$U[order])
}

#' @export
fnorm.TTensor <- function(x, ...) {
  # ||T||^2 = <core, core x_n (Un' Un)>; reduces to ||core|| for orthonormal U.
  V <- lapply(x$U, crossprod)
  W <- ttm(x$core, V, mode = seq_along(V))
  sqrt(max(innerprod(x$core, W), 0))
}

#' @export
innerprod.TTensor <- function(x, y, ...) {
  if (inherits(y, "TTensor")) {
    if (!identical(x$dim(), y$dim())) {
      stop("Tensors must have the same dimensions.")
    }
    # <T1, T2> = <core1, core2 x_n (U1n' U2n)>.
    W <- Map(function(a, b) crossprod(a, b), x$U, y$U)
    Z <- ttm(y$core, W, mode = seq_along(W))
    return(innerprod(x$core, Z))
  }
  if (inherits(y, "KTensor")) {
    if (!identical(x$dim(), y$dim())) {
      stop("Tensors must have the same dimensions.")
    }
    # <T, K> = sum_r lambda_r * core x_n (Un' u_nr).
    W <- Map(function(u, v) crossprod(u, v), x$U, y$U)
    total <- 0
    for (r in seq_along(y$lambda)) {
      vecs <- lapply(W, function(M) M[, r])
      contracted <- ttv(x$core, vecs, mode = seq_along(vecs))
      total <- total + y$lambda[r] * as.numeric(contracted$as_array())
    }
    return(total)
  }
  y <- .tensor_as_dense(y)
  if (!identical(x$dim(), y$dim())) {
    stop("Tensors must have the same dimensions.")
  }
  # <T, X> = <core, X x_n Un'>.
  Z <- ttm(y, lapply(x$U, t), mode = seq_len(x$ndims()))
  innerprod(x$core, Z)
}

#' @export
nvecs.TTensor <- function(x, mode, r = 1, flipsign = TRUE, ...) {
  mode <- as.integer(mode)
  N <- x$ndims()
  if (length(mode) != 1 || mode < 1 || mode > N) {
    stop("mode must be a single valid tensor mode.")
  }

  # Xn Xn' = Un Bn W Bn' Un', where Bn is the mode-n core unfolding and W the
  # Kronecker product of the other-mode Grams matching Bn's column ordering
  # (smaller modes vary fastest, so larger modes go first in the kron).
  Bn <- as.matrix(tenmat(x$core, rdims = mode))
  others <- setdiff(seq_len(N), mode)
  W <- NULL
  for (k in rev(others)) {
    G <- crossprod(x$U[[k]])
    W <- if (is.null(W)) G else W %x% G
  }
  mid <- if (is.null(W)) tcrossprod(Bn) else Bn %*% W %*% t(Bn)
  M <- x$U[[mode]] %*% mid %*% t(x$U[[mode]])
  M <- (M + t(M)) / 2
  eig <- eigen(M, symmetric = TRUE)
  u <- eig$vectors[, seq_len(r), drop = FALSE]

  if (flipsign) {
    loc <- apply(abs(u), 2, which.max)
    for (i in seq_len(ncol(u))) {
      if (u[loc[i], i] < 0) {
        u[, i] <- -u[, i]
      }
    }
  }
  u
}
