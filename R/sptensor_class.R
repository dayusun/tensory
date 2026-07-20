#' @include tensor_class.R tensor_dense_methods.R
NULL

# Linear index (1-based, double to avoid integer overflow) of subscript rows.
.sp_linear_index <- function(subs, dims) {
  mult <- cumprod(c(1, as.numeric(dims[-length(dims)])))
  as.vector(1 + (subs - 1) %*% mult)
}

# Decode 1-based linear indices back to subscript rows.
.sp_decode <- function(lin, dims) {
  N <- length(dims)
  out <- matrix(0L, length(lin), N)
  rem <- lin - 1
  for (k in seq_len(N)) {
    out[, k] <- as.integer(rem %% dims[k]) + 1L
    rem <- rem %/% dims[k]
  }
  out
}

#' R6 Class for Sparse Tensors (Sptensor)
#'
#' @description
#' Stores a tensor as a list of subscripts and nonzero values, mirroring the
#' MATLAB Tensor Toolbox `sptensor`. Duplicate subscripts are summed and
#' explicit zeros dropped on construction.
#'
#' @export
Sptensor <- R6::R6Class("Sptensor",
  public = list(
    #' @field subs Integer matrix of subscripts, one row per nonzero
    subs = NULL,

    #' @field vals Numeric vector of nonzero values
    vals = NULL,

    #' @field dims Integer vector of dimensions
    dims = NULL,

    #' @description Initialize a new Sptensor
    #' @param subs Integer matrix of subscripts (`nnz x ndims`).
    #' @param vals Numeric vector of values, one per subscript row (a scalar
    #'   is recycled).
    #' @param dims Integer vector of tensor dimensions.
    #' @return A new Sptensor object
    initialize = function(subs = NULL, vals = NULL, dims = NULL) {
      if (is.null(subs) && is.null(dims)) {
        return(invisible(self))
      }
      dims <- as.integer(dims)
      if (length(dims) == 0 || anyNA(dims) || any(dims < 0)) {
        stop("dims must be a vector of non-negative integers.")
      }
      if (is.null(subs) || (is.matrix(subs) && nrow(subs) == 0)) {
        self$subs <- matrix(0L, 0, length(dims))
        self$vals <- numeric(0)
        self$dims <- dims
        return(invisible(self))
      }
      subs <- as.matrix(subs)
      storage.mode(subs) <- "integer"
      if (ncol(subs) != length(dims)) {
        stop("subs must have one column per tensor dimension.")
      }
      vals <- as.double(vals)
      if (length(vals) == 1L) {
        vals <- rep(vals, nrow(subs))
      }
      if (length(vals) != nrow(subs)) {
        stop("vals must have one entry per row of subs.")
      }
      for (k in seq_along(dims)) {
        if (any(subs[, k] < 1L) || any(subs[, k] > dims[k])) {
          stop("Subscripts are out of bounds.")
        }
      }

      # Aggregate duplicates and drop explicit zeros; store sorted by linear
      # index so comparisons and merges are deterministic.
      lin <- .sp_linear_index(subs, dims)
      agg <- rowsum(vals, group = lin)
      ulin <- as.numeric(rownames(agg))
      v <- as.numeric(agg)
      keep <- v != 0
      self$subs <- .sp_decode(ulin[keep], dims)
      self$vals <- v[keep]
      self$dims <- dims
      invisible(self)
    },

    #' @description Get dimensions of the tensor
    #' @return Integer vector of dimensions
    dim = function() {
      self$dims
    },

    #' @description Get number of dimensions
    #' @return Integer tensor order
    ndims = function() {
      length(self$dims)
    },

    #' @description Print the Sptensor object
    #' @param ... Additional arguments
    #' @return Invisible self
    print = function(...) {
      cat("<Sptensor object>\n")
      cat("Dimensions: ", paste(self$dims, collapse = " x "), "\n")
      cat("Nonzeros: ", length(self$vals), "\n")
      k <- min(length(self$vals), 5L)
      for (i in seq_len(k)) {
        cat("  (", paste(self$subs[i, ], collapse = ","), ") = ",
            self$vals[i], "\n", sep = "")
      }
      if (length(self$vals) > k) cat("  ...\n")
      invisible(self)
    },

    #' @description Convert to a dense Tensor
    #' @return A dense `Tensor`
    full = function() {
      data <- array(0, dim = self$dims)
      if (nrow(self$subs) > 0) {
        data[self$subs] <- self$vals
      }
      Tensor$new(data, dims = self$dims, fast = TRUE)
    }
  )
)

#' Create a Sparse Tensor (Sptensor)
#'
#' @param subs Integer matrix of subscripts (`nnz x ndims`), or a dense
#'   Tensor/array to convert.
#' @param vals Numeric vector of values, one per subscript row.
#' @param dims Integer vector of tensor dimensions.
#' @return An `Sptensor` object.
#' @examples
#' S <- sptensor(rbind(c(1, 1, 1), c(2, 3, 4)), c(5, 7), c(2, 3, 4))
#' @rdname Sptensor
#' @export
sptensor <- function(subs = NULL, vals = NULL, dims = NULL) {
  # A bare Tensor/array with no vals/dims is a dense object to convert; a
  # matrix accompanied by vals/dims is the subscript form (a matrix is an
  # array, so the check must not rely on is.array alone).
  if (inherits(subs, "Tensor") || (is.array(subs) && is.null(vals) && is.null(dims))) {
    x <- .tensor_as_dense(subs)
    nz <- which(as.vector(x$data) != 0)
    d <- x$dim()
    sub_mat <- if (length(nz)) arrayInd(nz, .dim = d) else matrix(0L, 0, length(d))
    res <- Sptensor$new(sub_mat, as.vector(x$data)[nz], d)
  } else {
    res <- Sptensor$new(subs, vals, dims)
  }
  # Inherit "Tensor" so mixed sparse-dense arithmetic dispatches to the
  # (Sptensor-aware) Tensor operators instead of raising an
  # incompatible-methods error; Sptensor methods still win where defined.
  class(res) <- c("Sptensor", "Tensor", class(res))
  res
}

#' Random Sparse Tensor
#'
#' Creates a sparse tensor with approximately `nnz` uniformly random nonzeros
#' at uniformly random positions, mirroring the MATLAB Tensor Toolbox
#' `sptenrand`.
#'
#' @param dims Integer vector of dimensions.
#' @param nnz Number of nonzeros, or (if less than 1) the target density.
#' @return An `Sptensor`.
#' @examples
#' S <- sptenrand(c(10, 10, 10), 20)
#' @export
sptenrand <- function(dims, nnz) {
  dims <- as.integer(dims)
  total <- prod(as.numeric(dims))
  if (nnz < 1) {
    nnz <- max(1, round(nnz * total))
  }
  nnz <- min(as.integer(nnz), total)

  # Draw distinct positions by per-mode sampling with rejection of duplicates.
  subs <- matrix(0L, 0, length(dims))
  while (nrow(subs) < nnz) {
    draw <- vapply(dims, function(d) sample.int(d, 2L * nnz, replace = TRUE),
                   integer(2L * nnz))
    draw <- matrix(draw, ncol = length(dims))
    subs <- unique(rbind(subs, draw))
  }
  subs <- subs[seq_len(nnz), , drop = FALSE]
  sptensor(subs, stats::runif(nnz), dims)
}

#' @export
print.Sptensor <- function(x, ...) {
  x$print(...)
}

#' @export
as.tensor.Sptensor <- function(x, ...) {
  x$full()
}
