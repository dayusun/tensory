# Dense tensor methods modeled after MATLAB Tensor Toolbox.

.tensor_as_dense <- function(x) {
  if (inherits(x, c("KTensor", "TTensor"))) {
    return(as.tensor(x))
  }
  if (inherits(x, "Tensor")) {
    return(x)
  }
  as.tensor(x)
}

.tensor_sub2ind <- function(subs, dims) {
  if (nrow(subs) == 0) {
    return(integer(0))
  }
  multipliers <- cumprod(c(1L, head(as.integer(dims), -1L)))
  as.integer(1 + (subs - 1L) %*% multipliers)
}

.normalize_factor_list <- function(x, U, mode = NULL) {
  x <- .tensor_as_dense(x)
  N <- x$ndims()

  if (!is.null(mode)) {
    mode <- as.integer(mode)
    if (length(mode) != 1 || mode < 1 || mode > N) {
      stop("mode must be a single valid tensor mode.")
    }
  }

  if (inherits(U, "KTensor")) {
    U_list <- lapply(U$U, function(mat) as.matrix(mat))
    if (!is.null(mode)) {
      absorb_mode <- if (mode == 1L) 2L else 1L
      U_list[[absorb_mode]] <- sweep(U_list[[absorb_mode]], 2, U$lambda, `*`)
    } else {
      U_list[[1L]] <- sweep(U_list[[1L]], 2, U$lambda, `*`)
    }
    U <- U_list
  }

  if (!is.list(U) || length(U) != N) {
    stop("U must be a list of factor matrices or a KTensor of matching order.")
  }

  rank <- NULL
  for (i in seq_len(N)) {
    Ui <- as.matrix(U[[i]])
    if (nrow(Ui) != x$dim()[i]) {
      stop(sprintf("Factor matrix U[[%d]] has incompatible row dimension.", i))
    }
    if (!is.null(mode) && i == mode) {
      U[[i]] <- Ui
      next
    }
    if (is.null(rank)) {
      rank <- ncol(Ui)
    } else if (ncol(Ui) != rank) {
      stop("All factor matrices must have the same number of columns.")
    }
    U[[i]] <- Ui
  }

  list(x = x, U = U, rank = rank)
}

#' Tensor Times Vector
#'
#' Convenience wrapper around [ttm()] for vector contractions.
#'
#' @param tensor A Tensor-like object.
#' @param vector A numeric vector or list of numeric vectors.
#' @param mode Integer mode or modes to contract.
#' @return A Tensor object.
#' @export
ttv <- function(tensor, vector, mode = NULL) {
  ttm(tensor, vector, mode = mode, transpose = FALSE)
}

#' Inner Product
#'
#' Computes the Frobenius inner product between two tensors.
#'
#' @param x First tensor-like object.
#' @param y Second tensor-like object.
#' @return Numeric scalar inner product.
#' @export
innerprod <- function(x, y, ...) {
  UseMethod("innerprod", x)
}

#' @export
innerprod.Tensor <- function(x, y, ...) {
  if (inherits(y, c("KTensor", "TTensor"))) {
    y <- as.tensor(y)
  }
  if (!inherits(y, "Tensor")) {
    stop("innerprod is only implemented for Tensor, KTensor, and TTensor inputs.")
  }
  if (!identical(x$dim(), y$dim())) {
    stop("x and y must have the same dimensions.")
  }
  sum(as.double(x$data) * as.double(y$data))
}

#' Number of Nonzeros
#'
#' @param x A tensor-like object.
#' @return Integer count of nonzero entries.
#' @export
nnz <- function(x, ...) {
  UseMethod("nnz")
}

#' @export
nnz.Tensor <- function(x, ...) {
  sum(x$data != 0)
}

#' @export
nnz.default <- function(x, ...) {
  sum(as.vector(x) != 0)
}

#' Find Nonzero Entries
#'
#' Returns subscripts of nonzero entries and optionally their values.
#'
#' @param x A tensor-like object.
#' @param values Logical; if `TRUE`, include values in the result.
#' @return If `values = FALSE`, a matrix of subscripts. Otherwise a list with
#'   `subs` and `vals`.
#' @export
find <- function(x, ...) {
  UseMethod("find")
}

#' @export
find.Tensor <- function(x, values = FALSE, ...) {
  idx <- which(as.vector(x$data) != 0)
  dims <- x$dim()

  if (length(idx) == 0) {
    subs <- matrix(integer(0), nrow = 0, ncol = length(dims))
    colnames(subs) <- paste0("mode", seq_along(dims))
    if (values) {
      return(list(subs = subs, vals = numeric(0)))
    }
    return(subs)
  }

  subs <- arrayInd(idx, .dim = dims)
  colnames(subs) <- paste0("mode", seq_along(dims))

  if (!values) {
    return(subs)
  }

  vals <- as.vector(x$data)[idx]
  list(subs = subs, vals = vals)
}

#' @export
find.default <- function(x, mode = "function", numeric = FALSE, simple.words = TRUE, ...) {
  utils::find(
    what = x,
    mode = mode,
    numeric = numeric,
    simple.words = simple.words
  )
}

#' Permute Tensor Dimensions
#'
#' @param x A tensor-like object.
#' @param order Permutation of dimensions.
#' @return A permuted Tensor object.
#' @export
permute <- function(x, order, ...) {
  UseMethod("permute")
}

#' @export
permute.Tensor <- function(x, order, ...) {
  order <- as.integer(order)
  if (length(order) != x$ndims() || !setequal(order, seq_len(x$ndims()))) {
    stop("Invalid permutation order.")
  }
  if (identical(order, seq_len(x$ndims()))) {
    return(x$clone_tensor())
  }

  permuted <- aperm(x$data, order)
  Tensor$new(permuted, dims = x$dim()[order], fast = TRUE)
}

#' @export
permute.default <- function(x, order, ...) {
  if (is.array(x)) {
    return(aperm(x, order))
  }
  stop("permute is not implemented for this object type.")
}

#' Reshape Tensor
#'
#' @param x A tensor-like object.
#' @param new_dims New dimensions.
#' @return A reshaped object.
#' @export
reshape <- function(x, new_dims, ...) {
  UseMethod("reshape")
}

#' @export
reshape.Tensor <- function(x, new_dims, ...) {
  x$clone_tensor()$reshape(new_dims)
}

#' @export
reshape.default <- function(x, new_dims, ...) {
  new_dims <- as.integer(new_dims)
  if (anyNA(new_dims) || any(new_dims < 0L)) {
    stop("new_dims must be a non-negative integer vector.")
  }
  if (prod(new_dims) != length(x)) {
    stop("Product of new_dims must match the number of elements in x.")
  }
  array(as.vector(x), dim = new_dims)
}

#' Squeeze Tensor
#'
#' @param x A tensor-like object.
#' @return Object with singleton dimensions removed.
#' @export
squeeze <- function(x, ...) {
  UseMethod("squeeze")
}

#' @export
squeeze.Tensor <- function(x, ...) {
  x$squeeze()
}

#' @export
squeeze.default <- function(x, ...) {
  drop(x)
}

#' Unfold Tensor
#'
#' @param x A tensor-like object.
#' @param rdims Row modes.
#' @param cdims Column modes. If omitted, all remaining modes are used.
#' @return A matrix.
#' @export
unfold <- function(x, rdims = NULL, cdims = NULL, ...) {
  UseMethod("unfold")
}

#' @export
unfold.Tensor <- function(x, rdims = NULL, cdims = NULL, ...) {
  if (is.null(rdims)) {
    return(as.vector(x$data))
  }
  as.matrix(tenmat(x, rdims = rdims, cdims = cdims))
}

#' @export
unfold.default <- function(x, rdims = NULL, cdims = NULL, ...) {
  if (is.null(rdims)) {
    return(as.vector(x))
  }
  stop("unfold is not implemented for this object type.")
}

#' Vectorize Tensor
#'
#' @param x A tensor-like object.
#' @return A vector.
#' @export
vec <- function(x, ...) {
  UseMethod("vec")
}

#' @export
vec.Tensor <- function(x, ...) {
  as.vector(x$data)
}

#' @export
vec.default <- function(x, ...) {
  as.vector(x)
}

#' Leading Mode-n Vectors
#'
#' Computes the leading left singular vectors of the mode-n unfolding.
#'
#' @param x A Tensor object.
#' @param mode Mode along which to compute vectors.
#' @param r Number of vectors to return.
#' @param flipsign Logical; if `TRUE`, flip signs so the largest-magnitude
#'   entry in each column is positive.
#' @return A matrix whose columns are the leading mode-n vectors.
#' @export
nvecs <- function(x, mode, r = 1, flipsign = TRUE, ...) {
  UseMethod("nvecs")
}

#' @export
nvecs.Tensor <- function(x, mode, r = 1, flipsign = TRUE, ...) {
  mode <- as.integer(mode)
  if (length(mode) != 1 || mode < 1 || mode > x$ndims()) {
    stop("mode must be a single valid tensor mode.")
  }
  r <- as.integer(r)
  if (length(r) != 1 || r < 1) {
    stop("r must be a positive integer.")
  }

  Xn <- unfold(x, rdims = mode)
  max_rank <- min(nrow(Xn), ncol(Xn))
  if (r > max_rank) {
    stop("r cannot exceed the rank bound of the mode unfolding.")
  }

  sv <- svd(Xn, nu = r, nv = 0)
  u <- sv$u[, seq_len(r), drop = FALSE]

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

#' @export
nvecs.default <- function(x, mode, r = 1, flipsign = TRUE, ...) {
  nvecs(as.tensor(x), mode = mode, r = r, flipsign = flipsign, ...)
}

#' Matricized Tensor Times Khatri-Rao Product
#'
#' Computes the mode-n unfolding multiplied by the Khatri-Rao product of all
#' factor matrices except the skipped mode.
#'
#' @param x A Tensor object.
#' @param U A list of factor matrices or a `KTensor`.
#' @param mode Mode to skip.
#' @return A matrix of size `dim(x)[mode] x R`.
#' @export
mttkrp <- function(x, U, mode, ...) {
  UseMethod("mttkrp")
}

#' @export
mttkrp.Tensor <- function(x, U, mode, ...) {
  normalized <- .normalize_factor_list(x, U, mode = mode)
  x <- normalized$x
  U <- normalized$U
  N <- x$ndims()

  if (N < 2) {
    stop("mttkrp is invalid for tensors with fewer than 2 dimensions.")
  }

  if (exists("mttkrp_blas_cpp", mode = "function")) {
    return(mttkrp_blas_cpp(x$data, U, as.integer(mode)))
  }

  if (exists("mttkrp_cpp", mode = "function")) {
    return(mttkrp_cpp(x$data, U, as.integer(mode)))
  }

  other_modes <- setdiff(seq_len(N), mode)
  V <- matrix(0, nrow = x$dim()[mode], ncol = normalized$rank)
  for (r in seq_len(normalized$rank)) {
    vectors <- lapply(other_modes, function(i) U[[i]][, r])
    contracted <- ttv(x, vectors, mode = other_modes)
    V[, r] <- as.vector(contracted$as_array())
  }
  V
}

#' @export
mttkrp.default <- function(x, U, mode, ...) {
  mttkrp(.tensor_as_dense(x), U = U, mode = mode, ...)
}

#' Sequence of MTTKRP Calculations
#'
#' Computes `mttkrp(x, U, k)` for all tensor modes.
#'
#' @param x A Tensor object.
#' @param U A list of factor matrices or a `KTensor`.
#' @return A list of matrices, one per mode.
#' @export
mttkrps <- function(x, U, ...) {
  UseMethod("mttkrps")
}

#' @export
mttkrps.Tensor <- function(x, U, ...) {
  if (inherits(U, "KTensor")) {
    x <- .tensor_as_dense(x)
    return(lapply(seq_len(x$ndims()), function(mode) mttkrp(x, U, mode = mode)))
  }

  normalized <- .normalize_factor_list(x, U)
  x <- normalized$x
  U <- normalized$U

  if (exists("mttkrps_cpp", mode = "function")) {
    return(unname(mttkrps_cpp(x$data, U)))
  }

  lapply(seq_len(x$ndims()), function(mode) mttkrp(x, U, mode = mode))
}

#' @export
mttkrps.default <- function(x, U, ...) {
  mttkrps(.tensor_as_dense(x), U = U, ...)
}

#' Contract Tensor Dimensions
#'
#' Contracts a tensor along two equally sized dimensions.
#'
#' @param x A Tensor object.
#' @param i First mode.
#' @param j Second mode.
#' @return A Tensor object, potentially scalar.
#' @export
contract <- function(x, i, j, ...) {
  UseMethod("contract")
}

#' @export
contract.Tensor <- function(x, i, j, ...) {
  x <- .tensor_as_dense(x)
  dims <- x$dim()
  i <- as.integer(i)
  j <- as.integer(j)

  if (length(i) != 1 || length(j) != 1 || i < 1 || j < 1 || i > length(dims) || j > length(dims)) {
    stop("i and j must be valid tensor modes.")
  }
  if (i == j) {
    stop("i and j must be different modes.")
  }
  if (dims[i] != dims[j]) {
    stop("Must contract along equally sized dimensions.")
  }

  if (exists("contract_cpp", mode = "function")) {
    result <- contract_cpp(x$data, i, j)
    result_dims <- as.integer(result$dims)
    if (length(result_dims) == 0) {
      return(Tensor$new(as.numeric(result$data), dims = integer(0), fast = TRUE))
    }
    return(Tensor$new(array(as.numeric(result$data), dim = result_dims), dims = result_dims, fast = TRUE))
  }

  remdims <- setdiff(seq_along(dims), c(i, j))
  new_dims <- dims[remdims]
  n <- dims[i]
  m <- if (length(new_dims) == 0) 1L else as.integer(prod(new_dims))

  permuted <- aperm(x$data, c(remdims, i, j))
  data3 <- array(permuted, dim = c(m, n, n))
  new_data <- numeric(m)
  for (k in seq_len(n)) {
    new_data <- new_data + data3[, k, k]
  }

  if (length(new_dims) == 0) {
    return(Tensor$new(new_data, dims = integer(0), fast = TRUE))
  }

  Tensor$new(array(new_data, dim = new_dims), dims = new_dims, fast = TRUE)
}

#' @export
contract.default <- function(x, i, j, ...) {
  contract(.tensor_as_dense(x), i = i, j = j, ...)
}

#' Mask Tensor Values
#'
#' Extracts values from a tensor at positions corresponding to nonzero entries
#' of a mask tensor.
#'
#' @param x A Tensor object.
#' @param w A Tensor-like mask.
#' @return A vector of extracted values.
#' @export
mask <- function(x, w, ...) {
  UseMethod("mask")
}

#' @export
mask.Tensor <- function(x, w, ...) {
  x <- .tensor_as_dense(x)
  w <- .tensor_as_dense(w)

  x_dims <- x$dim()
  w_dims <- w$dim()
  if (length(w_dims) < length(x_dims)) {
    w_dims <- c(w_dims, rep.int(1L, length(x_dims) - length(w_dims)))
  }

  if (length(w_dims) != length(x_dims) || any(w_dims > x_dims)) {
    stop("Mask cannot be bigger than the data tensor.")
  }

  w_data <- array(as.double(w$data), dim = w_dims)
  if (exists("mask_cpp", mode = "function")) {
    return(mask_cpp(x$data, w_data))
  }

  w_subs <- find(tensor(w_data, dims = w_dims))
  if (nrow(w_subs) == 0) {
    return(numeric(0))
  }

  idx <- .tensor_sub2ind(w_subs, x_dims)
  as.vector(x$data)[idx]
}

#' @export
mask.default <- function(x, w, ...) {
  mask(.tensor_as_dense(x), w = w, ...)
}

#' Apply Elementwise Function to Tensor Arguments
#'
#' @param fun A function or function-like object.
#' @param ... Tensor-like objects or scalars.
#' @return A Tensor object.
#' @export
tenfun <- function(fun, ...) {
  args <- list(...)
  if (length(args) == 0) {
    stop("tenfun requires at least one tensor argument.")
  }

  if (!is.function(fun)) {
    stop("fun must be a function.")
  }

  normalized <- lapply(args, function(arg) {
    if (inherits(arg, c("Tensor", "KTensor", "TTensor"))) {
      return(.tensor_as_dense(arg))
    }
    if (is.atomic(arg) && length(arg) == 1) {
      return(arg)
    }
    if (is.numeric(arg) || is.logical(arg) || is.matrix(arg) || is.array(arg)) {
      return(as.tensor(arg))
    }
    stop("Invalid input.")
  })

  tensor_args <- Filter(function(arg) inherits(arg, "Tensor"), normalized)
  if (length(tensor_args) == 0) {
    stop("tenfun requires at least one tensor-like input.")
  }

  target_dims <- tensor_args[[1]]$dim()
  for (arg in tensor_args[-1]) {
    if (!identical(arg$dim(), target_dims)) {
      stop("All tensor inputs must have the same dimensions.")
    }
  }

  raw_args <- lapply(normalized, function(arg) {
    if (inherits(arg, "Tensor")) arg$data else arg
  })

  result <- do.call(fun, raw_args)
  Tensor$new(result, dims = target_dims)
}

#' Extract Tensor Fibers
#'
#' Extracts mode-k fibers specified by a matrix of sample indices.
#'
#' @param x A Tensor object.
#' @param mode Mode of the fibers to extract.
#' @param midx Matrix with one row per requested fiber and `ndims(x) - 1`
#'   columns containing the fixed indices for the other modes.
#' @return A matrix of size `dim(x)[mode] x nrow(midx)`.
#' @export
fibers <- function(x, mode, midx, ...) {
  UseMethod("fibers")
}

#' @export
fibers.Tensor <- function(x, mode, midx, ...) {
  x <- .tensor_as_dense(x)
  mode <- as.integer(mode)
  if (length(mode) != 1 || mode < 1 || mode > x$ndims()) {
    stop("mode must be a single valid tensor mode.")
  }
  midx <- as.matrix(midx)
  storage.mode(midx) <- "integer"

  if (ncol(midx) != x$ndims() - 1L) {
    stop("midx must have ndims(x) - 1 columns.")
  }

  if (exists("fibers_cpp", mode = "function")) {
    return(fibers_cpp(x$data, mode, midx))
  }

  other_modes <- setdiff(seq_len(x$ndims()), mode)
  out <- matrix(0, nrow = x$dim()[mode], ncol = nrow(midx))
  for (s in seq_len(nrow(midx))) {
    coords <- vector("list", x$ndims())
    coords[[mode]] <- seq_len(x$dim()[mode])
    for (k in seq_along(other_modes)) {
      coords[[other_modes[k]]] <- midx[s, k]
    }
    out[, s] <- do.call(`[`, c(list(x$data), coords, list(drop = TRUE)))
  }
  out
}

#' @export
fibers.default <- function(x, mode, midx, ...) {
  fibers(.tensor_as_dense(x), mode = mode, midx = midx, ...)
}

#' Tensor Times Same Vector
#'
#' Multiplies a tensor by the same vector in all contracted modes.
#'
#' @param x A Tensor object with equal mode sizes.
#' @param v A vector whose length matches each contracted mode.
#' @param n Non-positive integer controlling how many leading modes remain.
#' @return A scalar, vector, matrix, or Tensor depending on the number of
#'   remaining modes.
#' @export
ttsv <- function(x, v, n = 0, ...) {
  UseMethod("ttsv")
}

#' @export
ttsv.Tensor <- function(x, v, n = 0, ...) {
  x <- .tensor_as_dense(x)
  dims <- x$dim()
  if (!all(dims == dims[1])) {
    stop("ttsv requires all tensor modes to have the same length.")
  }
  if (length(v) != dims[1]) {
    stop("Vector length must match the common tensor dimension.")
  }
  n <- as.integer(n)
  if (length(n) != 1 || n > 0) {
    stop("n must be a single non-positive integer.")
  }

  keep_count <- -n
  if (keep_count > x$ndims()) {
    stop("Cannot keep more modes than the tensor order.")
  }

  contract_modes <- setdiff(seq_len(x$ndims()), seq_len(keep_count))
  if (length(contract_modes) == 0) {
    result <- x
  } else {
    result <- ttv(x, rep(list(as.double(v)), length(contract_modes)), mode = contract_modes)
  }

  if (inherits(result, "Tensor")) {
    result_dims <- result$dim()
    if (length(result_dims) == 0) {
      return(as.numeric(result$as_array()))
    }
    if (length(result_dims) <= 2) {
      return(result$as_array())
    }
  }

  result
}

#' @export
ttsv.default <- function(x, v, n = 0, ...) {
  ttsv(.tensor_as_dense(x), v = v, n = n, ...)
}

#' Check Tensor Symmetry
#'
#' Checks whether a tensor is symmetric with respect to one or more groups of
#' modes.
#'
#' @param x A Tensor object.
#' @param grps A vector of modes or list of mode vectors.
#' @return Logical scalar.
#' @export
issymmetric <- function(x, grps = NULL, ...) {
  UseMethod("issymmetric")
}

.normalize_symmetry_groups <- function(x, grps = NULL) {
  n <- x$ndims()
  if (is.null(grps)) {
    grps <- list(seq_len(n))
  } else if (!is.list(grps)) {
    grps <- list(as.integer(grps))
  } else {
    grps <- lapply(grps, as.integer)
  }

  seen <- integer(0)
  for (grp in grps) {
    if (length(grp) <= 1) next
    if (any(grp < 1) || any(grp > n)) {
      stop("All symmetry groups must reference valid modes.")
    }
    if (length(intersect(seen, grp)) > 0) {
      stop("Cannot have overlapping symmetries.")
    }
    seen <- c(seen, grp)
  }

  grps
}

#' @export
issymmetric.Tensor <- function(x, grps = NULL, ...) {
  x <- .tensor_as_dense(x)
  grps <- .normalize_symmetry_groups(x, grps)

  if (exists("issymmetric_cpp", mode = "function")) {
    return(issymmetric_cpp(x$data, grps))
  }

  for (grp in grps) {
    if (length(grp) <= 1) next
    if (!all(x$dim()[grp] == x$dim()[grp[1]])) {
      return(FALSE)
    }
    subs <- arrayInd(seq_len(length(x$data)), .dim = x$dim())
    class_subs <- subs
    class_subs[, grp] <- t(apply(subs[, grp, drop = FALSE], 1, sort))
    class_idx <- .tensor_sub2ind(class_subs, x$dim())
    if (any(as.vector(x$data) != as.vector(x$data)[class_idx])) {
      return(FALSE)
    }
  }

  TRUE
}

#' @export
issymmetric.default <- function(x, grps = NULL, ...) {
  issymmetric(.tensor_as_dense(x), grps = grps, ...)
}

#' Symmetrize Tensor
#'
#' Symmetrizes a tensor with respect to one or more mode groups.
#'
#' @param x A Tensor object.
#' @param grps A vector of modes or list of mode vectors.
#' @return A symmetrized Tensor object.
#' @export
symmetrize <- function(x, grps = NULL, ...) {
  UseMethod("symmetrize")
}

#' @export
symmetrize.Tensor <- function(x, grps = NULL, ...) {
  x <- .tensor_as_dense(x)
  dims <- x$dim()
  n <- x$ndims()
  grps <- .normalize_symmetry_groups(x, grps)

  seen <- integer(0)
  current <- x$clone_tensor()

  for (grp in grps) {
    if (length(grp) <= 1) {
      next
    }
    if (any(grp < 1) || any(grp > n)) {
      stop("All symmetrization groups must reference valid modes.")
    }
    if (length(intersect(seen, grp)) > 0) {
      stop("Cannot have overlapping symmetries.")
    }
    seen <- c(seen, grp)
    if (!all(dims[grp] == dims[grp[1]])) {
      stop("Dimension mismatch for symmetrization.")
    }

    subs <- arrayInd(seq_len(length(current$data)), .dim = dims)
    class_subs <- subs
    class_subs[, grp] <- t(apply(subs[, grp, drop = FALSE], 1, sort))
    class_idx <- .tensor_sub2ind(class_subs, dims)

    class_sum <- tapply(as.vector(current$data), class_idx, sum)
    class_n <- tapply(rep.int(1, length(class_idx)), class_idx, sum)
    class_avg <- class_sum / class_n
    values <- as.numeric(class_avg[as.character(class_idx)])

    current <- Tensor$new(array(values, dim = dims), dims = dims, fast = TRUE)
  }

  current
}

#' @export
symmetrize.default <- function(x, grps = NULL, ...) {
  symmetrize(.tensor_as_dense(x), grps = grps, ...)
}

#' Dense Array Representation
#'
#' Returns the dense base-array representation of a tensor.
#'
#' @param x A Tensor object.
#' @return A base R array.
#' @export
full <- function(x, ...) {
  UseMethod("full")
}

#' @export
full.Tensor <- function(x, ...) {
  x$as_array()
}

#' @export
full.default <- function(x, ...) {
  x
}

#' MATLAB-Style Double Conversion
#'
#' Returns the dense array data stored by a tensor.
#'
#' @param x A Tensor object.
#' @return A base R array.
#' @export
double.Tensor <- function(x) {
  x$as_array()
}

#' @export
as.double.Tensor <- function(x, ...) {
  as.double(x$as_array())
}

#' Equality Test for Tensors
#'
#' @param x First object.
#' @param y Second object.
#' @return Logical scalar.
#' @export
isequal <- function(x, y, ...) {
  UseMethod("isequal", x)
}

#' @export
isequal.Tensor <- function(x, y, ...) {
  if (inherits(y, c("KTensor", "TTensor"))) {
    y <- as.tensor(y)
  }
  inherits(y, "Tensor") &&
    identical(x$dim(), y$dim()) &&
    identical(as.vector(x$data), as.vector(y$data))
}

#' @export
isequal.default <- function(x, y, ...) {
  identical(x, y)
}

#' Scalar Tensor Predicate
#'
#' @param x An object.
#' @return Logical scalar.
#' @export
isscalar <- function(x, ...) {
  UseMethod("isscalar")
}

#' @export
isscalar.Tensor <- function(x, ...) {
  length(x$as_array()) == 1L
}

#' @export
isscalar.default <- function(x, ...) {
  length(x) == 1L
}

#' Tensor Scaling
#'
#' Alias for [t_scale()] with Tensor Toolbox naming.
#'
#' @param x A Tensor object.
#' @param ... For tensors, pass `s =` and `dims =`. For non-tensors, arguments
#'   are forwarded to [base::scale()].
#' @return A scaled Tensor.
#' @export
scale <- function(x, ...) {
  UseMethod("scale")
}

#' @export
scale.Tensor <- function(x, s, dims, ...) {
  t_scale(x, s = s, dims = dims, ...)
}

#' @export
scale.default <- function(x, ...) {
  base::scale(x, ...)
}

#' Transpose Tensor
#'
#' Tensor transpose is not defined; use [permute()] instead.
#'
#' @param x A Tensor object.
#' @return This function always errors for tensors.
#' @export
transpose <- function(x, ...) {
  UseMethod("transpose")
}

#' @export
transpose.Tensor <- function(x, ...) {
  stop("Transpose on Tensor is not defined. Use permute() instead.")
}

#' @export
transpose.default <- function(x, ...) {
  base::t(x)
}
