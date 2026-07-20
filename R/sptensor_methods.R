#' @include sptensor_class.R tensor_dense_methods.R tensor_operations.R
NULL

#' @export
nnz.Sptensor <- function(x, ...) {
  length(x$vals)
}

#' @export
find.Sptensor <- function(x, values = FALSE, ...) {
  if (values) {
    return(list(subs = x$subs, vals = x$vals))
  }
  x$subs
}

#' @export
fnorm.Sptensor <- function(x, ...) {
  sqrt(sum(x$vals^2))
}

#' @export
isscalar.Sptensor <- function(x, ...) {
  length(x$dims) == 0
}

#' @export
permute.Sptensor <- function(x, order, ...) {
  order <- as.integer(order)
  if (length(order) != x$ndims() || !setequal(order, seq_len(x$ndims()))) {
    stop("order must be a permutation of 1:ndims(x).")
  }
  sptensor(x$subs[, order, drop = FALSE], x$vals, x$dims[order])
}

#' @export
innerprod.Sptensor <- function(x, y, ...) {
  if (inherits(y, "Sptensor")) {
    if (!identical(x$dims, y$dims)) {
      stop("Tensors must have the same dimensions.")
    }
    lx <- .sp_linear_index(x$subs, x$dims)
    ly <- .sp_linear_index(y$subs, y$dims)
    hit <- match(lx, ly)
    ok <- !is.na(hit)
    return(sum(x$vals[ok] * y$vals[hit[ok]]))
  }
  if (inherits(y, "KTensor")) {
    if (!identical(x$dims, as.integer(y$dim()))) {
      stop("Tensors must have the same dimensions.")
    }
    # sum_r lambda_r sum_e val_e prod_n U_n[i_en, r], using only nonzeros.
    total <- 0
    R <- length(y$lambda)
    if (nrow(x$subs) == 0 || R == 0) return(0)
    W <- matrix(1, nrow(x$subs), R)
    for (n in seq_len(x$ndims())) {
      W <- W * y$U[[n]][x$subs[, n], , drop = FALSE]
    }
    return(sum(y$lambda * colSums(x$vals * W)))
  }
  y <- .tensor_as_dense(y)
  if (!identical(x$dims, as.integer(y$dim()))) {
    stop("Tensors must have the same dimensions.")
  }
  if (nrow(x$subs) == 0) return(0)
  sum(x$vals * y$data[x$subs])
}

#' @export
mttkrp.Sptensor <- function(x, U, mode, ...) {
  if (inherits(U, "KTensor")) {
    lambda <- U$lambda
    U <- U$U
    absorb <- if (mode == 1L) 2L else 1L
    U[[absorb]] <- sweep(U[[absorb]], 2L, lambda, `*`)
  }
  mode <- as.integer(mode)
  N <- x$ndims()
  if (length(mode) != 1 || mode < 1 || mode > N) {
    stop("mode must be a single valid tensor mode.")
  }
  if (!is.list(U) || length(U) != N) {
    stop("U must be a list of factor matrices of matching order.")
  }
  R <- ncol(U[[if (mode == 1L) 2L else 1L]])
  V <- matrix(0, x$dims[mode], R)
  if (nrow(x$subs) == 0) return(V)

  # W[e, r] = val_e * prod_{n != mode} U_n[i_en, r]; accumulate rows by i_e,mode.
  W <- matrix(x$vals, nrow(x$subs), R)
  for (n in setdiff(seq_len(N), mode)) {
    W <- W * U[[n]][x$subs[, n], , drop = FALSE]
  }
  agg <- rowsum(W, group = x$subs[, mode])
  V[as.integer(rownames(agg)), ] <- agg
  V
}

#' @export
nvecs.Sptensor <- function(x, mode, r = 1, flipsign = TRUE, ...) {
  mode <- as.integer(mode)
  N <- x$ndims()
  if (length(mode) != 1 || mode < 1 || mode > N) {
    stop("mode must be a single valid tensor mode.")
  }
  n_sz <- x$dims[mode]
  M <- matrix(0, n_sz, n_sz)
  if (nrow(x$subs) > 0) {
    # Group nonzeros by their fiber (all subscripts except `mode`); each
    # group contributes an outer product of its sparse column.
    others <- setdiff(seq_len(N), mode)
    key <- .sp_linear_index(x$subs[, others, drop = FALSE], x$dims[others])
    groups <- split(seq_along(key), key)
    for (g in groups) {
      idx <- x$subs[g, mode]
      v <- x$vals[g]
      M[idx, idx] <- M[idx, idx] + tcrossprod(v)
    }
  }
  eig <- eigen((M + t(M)) / 2, symmetric = TRUE)
  u <- eig$vectors[, seq_len(r), drop = FALSE]
  if (flipsign) {
    loc <- apply(abs(u), 2, which.max)
    for (i in seq_len(ncol(u))) {
      if (u[loc[i], i] < 0) u[, i] <- -u[, i]
    }
  }
  u
}

#' @export
collapse.Sptensor <- function(x, dims, fun = sum, ...) {
  dims <- as.integer(dims)
  N <- x$ndims()
  if (any(dims < 1) || any(dims > N)) {
    stop("dims must contain valid tensor modes to reduce.")
  }
  remdims <- setdiff(seq_len(N), dims)
  if (length(remdims) == 0) {
    value <- if (nrow(x$subs) == 0) fun(numeric(0)) else fun(x$vals)
    return(Tensor$new(as.double(value), integer(0), fast = TRUE))
  }
  if (!identical(fun, sum)) {
    # Non-additive collapses need the implicit zeros; go dense.
    return(collapse(x$full(), dims = dims, fun = fun, ...))
  }
  key_subs <- x$subs[, remdims, drop = FALSE]
  agg <- rowsum(x$vals, group = .sp_linear_index(key_subs, x$dims[remdims]))
  new_subs <- .sp_decode(as.numeric(rownames(agg)), x$dims[remdims])
  sptensor(new_subs, as.numeric(agg), x$dims[remdims])
}

# ttm/ttv kernel for sparse tensors, dispatched from the ttm() generic.
# Vectors contract to a smaller sparse tensor; matrices produce a dense
# Tensor (the result of a sparse-times-dense-matrix product is dense).
.ttm_sparse <- function(x, mat, mode = NULL, transpose = FALSE) {
  if (is.list(mat)) {
    if (is.null(mode)) mode <- seq_along(mat)
    if (length(mode) != length(mat)) stop("Length of mode and mat must match")
    ord <- order(mode, decreasing = TRUE)
    mat <- mat[ord]
    mode <- mode[ord]
    res <- x
    for (i in seq_along(mat)) {
      res <- ttm(res, mat[[i]], mode = mode[i], transpose = transpose)
    }
    return(res)
  }

  if (is.null(mode)) mode <- 1L
  mode <- as.integer(mode)
  N <- x$ndims()
  if (length(mode) != 1 || mode < 1 || mode > N) {
    stop("Mode out of bounds")
  }

  is_vec <- is.vector(mat) && !is.list(mat) && !is.matrix(mat)
  if (is_vec) {
    v <- as.double(mat)
    if (length(v) != x$dims[mode]) {
      stop("Vector length must match the tensor dimension in the given mode.")
    }
    vals <- x$vals * v[x$subs[, mode]]
    remdims <- setdiff(seq_len(N), mode)
    if (length(remdims) == 0) {
      return(Tensor$new(sum(vals), integer(0), fast = TRUE))
    }
    return(sptensor(x$subs[, remdims, drop = FALSE], vals, x$dims[remdims]))
  }

  mat <- as.matrix(mat)
  M <- if (transpose) t(mat) else mat
  if (ncol(M) != x$dims[mode]) {
    stop("Matrix dimensions must match the tensor dimension in the given mode.")
  }
  J <- nrow(M)

  others <- setdiff(seq_len(N), mode)
  other_dims <- x$dims[others]
  new_dims <- x$dims
  new_dims[mode] <- J

  out <- array(0, dim = c(J, other_dims))
  if (nrow(x$subs) > 0) {
    # contrib[e, ] = val_e * M[, i_e]; accumulate rows sharing a fiber index.
    contrib <- x$vals * t(M[, x$subs[, mode], drop = FALSE])
    p <- .sp_linear_index(x$subs[, others, drop = FALSE], other_dims)
    agg <- rowsum(contrib, group = p)
    cols <- as.numeric(rownames(agg))
    flat <- matrix(out, nrow = J)
    flat[, cols] <- t(agg)
    out <- array(flat, dim = c(J, other_dims))
  }

  # Move the new mode-J axis back into position `mode`.
  perm <- integer(N)
  perm[mode] <- 1L
  perm[others] <- seq_len(N)[-1L]
  res <- aperm(out, perm)
  Tensor$new(res, dims = as.integer(new_dims), fast = TRUE)
}

# Sparse fast paths for the arithmetic operators. Called from the Tensor
# operator methods (Sptensor inherits "Tensor", so those methods receive the
# sparse operands); returns NULL when no sparse-preserving path applies and
# the caller should densify instead.
.sptensor_arith <- function(e1, e2, op) {
  both_sparse <- inherits(e1, "Sptensor") && inherits(e2, "Sptensor")
  if (both_sparse) {
    if (!identical(e1$dims, e2$dims)) {
      stop("Tensors must have the same dimensions.")
    }
    if (op == "+") {
      return(sptensor(rbind(e1$subs, e2$subs), c(e1$vals, e2$vals), e1$dims))
    }
    if (op == "-") {
      return(sptensor(rbind(e1$subs, e2$subs), c(e1$vals, -e2$vals), e1$dims))
    }
    if (op == "*") {
      l1 <- .sp_linear_index(e1$subs, e1$dims)
      l2 <- .sp_linear_index(e2$subs, e2$dims)
      hit <- match(l1, l2)
      ok <- !is.na(hit)
      return(sptensor(e1$subs[ok, , drop = FALSE],
                      e1$vals[ok] * e2$vals[hit[ok]], e1$dims))
    }
    return(NULL)
  }

  sp <- if (inherits(e1, "Sptensor")) e1 else e2
  other <- if (inherits(e1, "Sptensor")) e2 else e1
  sp_first <- inherits(e1, "Sptensor")

  if (is.numeric(other) && length(other) == 1) {
    if (op == "*") {
      return(sptensor(sp$subs, sp$vals * other, sp$dims))
    }
    if (op == "/" && sp_first) {
      return(sptensor(sp$subs, sp$vals / other, sp$dims))
    }
    if (op == "^" && sp_first && other > 0) {
      return(sptensor(sp$subs, sp$vals^other, sp$dims))
    }
  }
  NULL
}
