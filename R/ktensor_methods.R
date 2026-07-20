#' @include tensor_class.R ktensor_class.R tensor_dense_methods.R tensor_operations.R
NULL

# Hadamard product of crossprod(U[[k]], V[[k]]) over the given modes.
.kt_gram_hadamard <- function(U, V = U, modes = seq_along(U)) {
  H <- NULL
  for (k in modes) {
    G <- crossprod(U[[k]], V[[k]])
    H <- if (is.null(H)) G else H * G
  }
  H
}

#' Number of Components of a Kruskal Tensor
#'
#' @param x A `KTensor`.
#' @return Integer number of rank-one components.
#' @param ... Additional arguments passed to methods.
#' @export
ncomponents <- function(x, ...) {
  UseMethod("ncomponents")
}

#' @export
ncomponents.KTensor <- function(x, ...) {
  length(x$lambda)
}

#' Extract Components of a Kruskal Tensor
#'
#' Returns a new `KTensor` containing only the selected rank-one components,
#' mirroring the MATLAB Tensor Toolbox `extract`.
#'
#' @param x A `KTensor`.
#' @param idx Integer vector of component indices to keep.
#' @return A `KTensor` with `length(idx)` components.
#' @param ... Additional arguments passed to methods.
#' @export
extract <- function(x, idx, ...) {
  UseMethod("extract")
}

#' @export
extract.KTensor <- function(x, idx, ...) {
  idx <- as.integer(idx)
  R <- length(x$lambda)
  if (length(idx) == 0 || any(idx < 1) || any(idx > R)) {
    stop("idx must select valid components.")
  }
  ktensor(x$lambda[idx], lapply(x$U, function(M) M[, idx, drop = FALSE]))
}

#' Redistribute Kruskal Weights into a Mode
#'
#' Absorbs the lambda weights into the factor matrix of the given mode and
#' resets the weights to one, mirroring the MATLAB Tensor Toolbox
#' `redistribute`.
#'
#' @param x A `KTensor`.
#' @param mode Mode that absorbs the weights.
#' @return A `KTensor` with unit weights.
#' @param ... Additional arguments passed to methods.
#' @export
redistribute <- function(x, mode, ...) {
  UseMethod("redistribute")
}

#' @export
redistribute.KTensor <- function(x, mode, ...) {
  mode <- as.integer(mode)
  if (length(mode) != 1 || mode < 1 || mode > x$ndims()) {
    stop("mode must be a single valid tensor mode.")
  }
  U <- x$U
  U[[mode]] <- sweep(U[[mode]], 2L, x$lambda, `*`)
  ktensor(rep(1, length(x$lambda)), U)
}

#' Normalize a Kruskal Tensor
#'
#' Normalizes the columns of every factor matrix, absorbing the magnitudes
#' into the weight vector `lambda`. Mirrors the MATLAB Tensor Toolbox
#' `normalize(K, N)` semantics.
#'
#' @param x A `KTensor`.
#' @param mode `NULL` (default) keeps all weight in `lambda`; a mode number
#'   absorbs the weights into that factor matrix; `0` distributes the weights
#'   evenly across all modes (each factor absorbs `lambda^(1/N)`).
#' @param sort Logical; if `TRUE`, sort the components by weight, descending.
#'   Only allowed when the weights remain in `lambda` (i.e. `mode` is `NULL`).
#' @param normtype Norm to use for the factor columns (passed as the `p` in
#'   the vector p-norm; default `2`).
#' @return A normalized `KTensor`.
#' @param ... Additional arguments passed to methods.
#' @export
normalize <- function(x, ...) {
  UseMethod("normalize")
}

#' @rdname normalize
#' @export
normalize.KTensor <- function(x, mode = NULL, sort = FALSE, normtype = 2, ...) {
  lambda <- x$lambda
  U <- x$U
  N <- length(U)
  R <- length(lambda)

  for (n in seq_len(N)) {
    scale_n <- apply(U[[n]], 2L, function(col) {
      if (normtype == 2) sqrt(sum(col^2)) else sum(abs(col)^normtype)^(1 / normtype)
    })
    fix <- scale_n == 0
    scale_n[fix] <- 1
    U[[n]] <- sweep(U[[n]], 2L, scale_n, `/`)
    lambda <- lambda * scale_n
  }

  # Flip components with negative weight so lambda >= 0 (absorb sign in mode 1).
  neg <- lambda < 0
  if (any(neg)) {
    U[[1L]][, neg] <- -U[[1L]][, neg, drop = FALSE]
    lambda[neg] <- -lambda[neg]
  }

  if (!is.null(mode)) {
    mode <- as.integer(mode)
    if (isTRUE(sort)) {
      stop("sort is only supported when the weights remain in lambda.")
    }
    if (mode == 0L) {
      d <- lambda^(1 / N)
      for (n in seq_len(N)) {
        U[[n]] <- sweep(U[[n]], 2L, d, `*`)
      }
      lambda <- rep(1, R)
    } else if (mode >= 1L && mode <= N) {
      U[[mode]] <- sweep(U[[mode]], 2L, lambda, `*`)
      lambda <- rep(1, R)
    } else {
      stop("mode must be NULL, 0, or a valid tensor mode.")
    }
  } else if (isTRUE(sort)) {
    ord <- order(lambda, decreasing = TRUE)
    lambda <- lambda[ord]
    U <- lapply(U, function(M) M[, ord, drop = FALSE])
  }

  ktensor(lambda, U)
}

#' Arrange the Components of a Kruskal Tensor
#'
#' Normalizes the columns of each factor matrix and sorts the components by
#' weight in descending order, mirroring the MATLAB Tensor Toolbox `arrange`.
#' When `perm` is given, the components are permuted without normalization.
#'
#' @param x A `KTensor`.
#' @param perm Optional explicit permutation of `1:ncomponents(x)`.
#' @return A rearranged `KTensor`.
#' @param ... Additional arguments passed to methods.
#' @export
arrange <- function(x, ...) {
  UseMethod("arrange")
}

#' @rdname arrange
#' @export
arrange.KTensor <- function(x, perm = NULL, ...) {
  if (!is.null(perm)) {
    perm <- as.integer(perm)
    if (!setequal(perm, seq_along(x$lambda)) || length(perm) != length(x$lambda)) {
      stop("perm must be a permutation of 1:ncomponents(x).")
    }
    return(ktensor(x$lambda[perm], lapply(x$U, function(M) M[, perm, drop = FALSE])))
  }
  normalize(x, sort = TRUE)
}

#' Fix Sign Ambiguity of a Kruskal Tensor
#'
#' Flips the signs of factor-matrix columns so that the largest-magnitude
#' entry of each column is positive, compensating in the weights so the
#' represented tensor is unchanged. Mirrors the MATLAB Tensor Toolbox
#' `fixsigns`.
#'
#' @param x A `KTensor`.
#' @return A sign-fixed `KTensor`.
#' @param ... Additional arguments passed to methods.
#' @export
fixsigns <- function(x, ...) {
  UseMethod("fixsigns")
}

#' @export
fixsigns.KTensor <- function(x, ...) {
  .fixsigns_cp(x)
}

#' Kruskal Tensor to Vector
#'
#' Stacks a `KTensor` into a single numeric vector, mirroring the MATLAB
#' Tensor Toolbox `tovec`.
#'
#' @param x A `KTensor`.
#' @param lambda Logical; if `TRUE` (default) the weight vector is included as
#'   the leading block. If `FALSE`, the weights are first absorbed into mode 1.
#' @return A numeric vector of length `R + sum(dims) * R` (with weights) or
#'   `sum(dims) * R` (without).
#' @param ... Additional arguments passed to methods.
#' @export
tovec <- function(x, ...) {
  UseMethod("tovec")
}

#' @rdname tovec
#' @export
tovec.KTensor <- function(x, lambda = TRUE, ...) {
  if (!lambda) {
    x <- redistribute(x, 1L)
    return(unlist(lapply(x$U, as.vector), use.names = FALSE))
  }
  c(x$lambda, unlist(lapply(x$U, as.vector), use.names = FALSE))
}

#' Score the Similarity of Two Kruskal Tensors
#'
#' Computes the factor match score between two `KTensor` objects with the
#' same number of components, mirroring the MATLAB Tensor Toolbox `score`.
#' Each candidate component pairing is scored by the product of the absolute
#' cosine similarities of the matched factor columns, optionally discounted by
#' the relative difference of the component weights; components are matched
#' greedily.
#'
#' @param x A `KTensor` (the estimate).
#' @param y A `KTensor` with the same dimensions and number of components
#'   (the reference).
#' @param lambda_penalty Logical; if `TRUE` (default) each pair score is
#'   multiplied by `1 - |la - lb| / max(la, lb)` computed from the normalized
#'   weights.
#' @param greedy Logical; must be `TRUE` (greedy matching is the only
#'   implemented strategy, as it is the MATLAB default).
#' @return A list with elements `score` (mean of the matched component
#'   scores), `perm` (the permutation of `y`'s components matched to `x`), and
#'   `x` (a normalized copy of `x` arranged to match `y`).
#' @param ... Additional arguments passed to methods.
#' @export
score <- function(x, ...) {
  UseMethod("score")
}

#' @rdname score
#' @export
score.KTensor <- function(x, y, lambda_penalty = TRUE, greedy = TRUE, ...) {
  if (!inherits(y, "KTensor")) {
    stop("y must be a KTensor.")
  }
  if (!identical(x$dim(), y$dim())) {
    stop("x and y must have the same dimensions.")
  }
  RA <- length(x$lambda)
  RB <- length(y$lambda)
  if (RA < RB) {
    stop("x must have at least as many components as y.")
  }
  if (!isTRUE(greedy)) {
    stop("Only greedy matching is implemented.")
  }

  A <- normalize(x)
  B <- normalize(y)
  N <- A$ndims()

  # Pairwise |cosine| congruence per mode; columns are already unit norm.
  C <- matrix(1, RA, RB)
  for (n in seq_len(N)) {
    C <- C * abs(crossprod(A$U[[n]], B$U[[n]]))
  }

  if (lambda_penalty) {
    la <- abs(A$lambda)
    lb <- abs(B$lambda)
    penalty <- 1 - abs(outer(la, lb, `-`)) / pmax(outer(la, lb, pmax), .Machine$double.eps)
    C <- C * penalty
  }

  perm <- integer(RB)
  used <- rep(FALSE, RA)
  scores <- numeric(RB)
  work <- C
  for (i in seq_len(RB)) {
    best <- which(work == max(work), arr.ind = TRUE)[1, ]
    scores[i] <- work[best[1], best[2]]
    perm[best[2]] <- best[1]
    used[best[1]] <- TRUE
    work[best[1], ] <- -Inf
    work[, best[2]] <- -Inf
  }

  full_perm <- c(perm, which(!used))
  list(
    score = mean(scores),
    perm = perm,
    x = arrange(A, perm = full_perm)
  )
}

#' Visualize a Kruskal Tensor
#'
#' Plots each factor-matrix column as a line plot in an `ndims x ncomponents`
#' grid, a lightweight analogue of the MATLAB Tensor Toolbox `viz`.
#'
#' @param x A `KTensor`.
#' @param normalize Logical; if `TRUE` (default) plot the normalized factors.
#' @return Invisibly, `x`.
#' @param ... Additional arguments passed to methods.
#' @export
viz <- function(x, ...) {
  UseMethod("viz")
}

#' @rdname viz
#' @export
viz.KTensor <- function(x, normalize = TRUE, ...) {
  K <- if (normalize) normalize.KTensor(x) else x
  N <- K$ndims()
  R <- length(K$lambda)
  old <- graphics::par(mfrow = c(N, R), mar = c(2, 2, 1.5, 0.5))
  on.exit(graphics::par(old))
  for (n in seq_len(N)) {
    for (r in seq_len(R)) {
      graphics::plot(K$U[[n]][, r], type = "l",
                     main = sprintf("mode %d, comp %d", n, r),
                     xlab = "", ylab = "", cex.main = 0.8)
    }
  }
  invisible(x)
}

#' @export
permute.KTensor <- function(x, order, ...) {
  order <- as.integer(order)
  if (length(order) != x$ndims() || !setequal(order, seq_len(x$ndims()))) {
    stop("order must be a permutation of 1:ndims(x).")
  }
  ktensor(x$lambda, x$U[order])
}

#' @export
fnorm.KTensor <- function(x, ...) {
  H <- .kt_gram_hadamard(x$U)
  val <- sum(tcrossprod(x$lambda) * H)
  sqrt(max(val, 0))
}

#' @export
innerprod.KTensor <- function(x, y, ...) {
  if (inherits(y, "KTensor")) {
    if (!identical(x$dim(), y$dim())) {
      stop("Tensors must have the same dimensions.")
    }
    H <- .kt_gram_hadamard(x$U, y$U)
    return(sum(outer(x$lambda, y$lambda) * H))
  }
  if (inherits(y, "TTensor")) {
    return(innerprod(y, x))
  }
  y <- .tensor_as_dense(y)
  if (!identical(x$dim(), y$dim())) {
    stop("Tensors must have the same dimensions.")
  }
  # <X, K> = sum_r lambda_r * <X, u1r o ... o uNr> via one MTTKRP.
  V <- mttkrp(y, x$U, mode = 1L)
  sum(x$lambda * colSums(x$U[[1L]] * V))
}

#' @export
nvecs.KTensor <- function(x, mode, r = 1, flipsign = TRUE, ...) {
  mode <- as.integer(mode)
  N <- x$ndims()
  if (length(mode) != 1 || mode < 1 || mode > N) {
    stop("mode must be a single valid tensor mode.")
  }
  # Xn Xn' = Un L H L Un' with H the skip-mode Hadamard of Grams.
  H <- .kt_gram_hadamard(x$U, modes = setdiff(seq_len(N), mode))
  L <- diag(x$lambda, nrow = length(x$lambda))
  M <- x$U[[mode]] %*% L %*% H %*% L %*% t(x$U[[mode]])
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
