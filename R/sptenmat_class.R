#' @include sptensor_class.R
NULL

#' R6 Class for Sparse Matricized Tensors (Sptenmat)
#'
#' @description
#' A sparse matricization of an `Sptensor`: the nonzeros of the tensor mapped
#' to (row, column) coordinates of the unfolding defined by `rdims`/`cdims`,
#' mirroring the MATLAB Tensor Toolbox `sptenmat`.
#'
#' @export
Sptenmat <- R6::R6Class("Sptenmat",
  public = list(
    #' @field subs Two-column integer matrix of (row, col) coordinates
    subs = NULL,

    #' @field vals Numeric vector of nonzero values
    vals = NULL,

    #' @field rdims Tensor modes mapped to matrix rows
    rdims = NULL,

    #' @field cdims Tensor modes mapped to matrix columns
    cdims = NULL,

    #' @field tsize Dimensions of the original tensor
    tsize = NULL,

    #' @description Initialize a new Sptenmat from a sparse tensor
    #' @param x An `Sptensor`.
    #' @param rdims Modes mapped to rows.
    #' @param cdims Modes mapped to columns (defaults to the remaining modes
    #'   in ascending order).
    #' @return A new Sptenmat object
    initialize = function(x = NULL, rdims = NULL, cdims = NULL) {
      if (is.null(x)) {
        return(invisible(self))
      }
      if (!inherits(x, "Sptensor")) {
        stop("x must be an Sptensor.")
      }
      N <- x$ndims()
      if (is.null(rdims) && is.null(cdims)) {
        stop("At least one of rdims and cdims must be given.")
      }
      if (is.null(rdims)) rdims <- setdiff(seq_len(N), cdims)
      if (is.null(cdims)) cdims <- setdiff(seq_len(N), rdims)
      rdims <- as.integer(rdims)
      cdims <- as.integer(cdims)
      if (!setequal(c(rdims, cdims), seq_len(N)) ||
          length(c(rdims, cdims)) != N) {
        stop("rdims and cdims must partition the tensor modes.")
      }

      row_idx <- if (length(rdims)) {
        .sp_linear_index(x$subs[, rdims, drop = FALSE], x$dims[rdims])
      } else {
        rep(1, nrow(x$subs))
      }
      col_idx <- if (length(cdims)) {
        .sp_linear_index(x$subs[, cdims, drop = FALSE], x$dims[cdims])
      } else {
        rep(1, nrow(x$subs))
      }

      self$subs <- cbind(row = as.integer(row_idx), col = as.integer(col_idx))
      self$vals <- x$vals
      self$rdims <- rdims
      self$cdims <- cdims
      self$tsize <- x$dims
      invisible(self)
    },

    #' @description Matrix dimensions of the unfolding
    #' @return Integer vector `c(nrow, ncol)`
    dim = function() {
      c(prod(as.numeric(self$tsize[self$rdims])),
        prod(as.numeric(self$tsize[self$cdims])))
    },

    #' @description Print the Sptenmat object
    #' @param ... Additional arguments
    #' @return Invisible self
    print = function(...) {
      d <- self$dim()
      cat("<Sptenmat object>\n")
      cat("Matrix size: ", d[1], " x ", d[2], "\n", sep = "")
      cat("Row modes: ", paste(self$rdims, collapse = " "), "\n")
      cat("Col modes: ", paste(self$cdims, collapse = " "), "\n")
      cat("Nonzeros: ", length(self$vals), "\n")
      invisible(self)
    },

    #' @description Convert to a dense matrix
    #' @return A dense matrix of the unfolding
    as_matrix = function() {
      d <- self$dim()
      M <- matrix(0, d[1], d[2])
      if (nrow(self$subs) > 0) {
        M[self$subs] <- self$vals
      }
      M
    }
  )
)

#' Create a Sparse Matricized Tensor (Sptenmat)
#'
#' @param x An `Sptensor`.
#' @param rdims Modes mapped to matrix rows.
#' @param cdims Modes mapped to matrix columns.
#' @return An `Sptenmat` object.
#' @examples
#' S <- sptenrand(c(4, 3, 2), 5)
#' A <- sptenmat(S, rdims = 1)
#' @rdname Sptenmat
#' @export
sptenmat <- function(x, rdims = NULL, cdims = NULL) {
  res <- Sptenmat$new(x, rdims = rdims, cdims = cdims)
  class(res) <- c("Sptenmat", class(res))
  res
}

#' @export
print.Sptenmat <- function(x, ...) {
  x$print(...)
}

#' @export
as.matrix.Sptenmat <- function(x, ...) {
  x$as_matrix()
}
