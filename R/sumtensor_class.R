#' @include tensor_class.R ktensor_class.R ttensor_class.R
NULL

#' R6 Class for Implicit Sums of Tensors (SumTensor)
#'
#' @description
#' Represents a tensor that is the sum of several parts (dense tensors,
#' Kruskal tensors, or Tucker tensors) without materializing the sum,
#' mirroring the MATLAB Tensor Toolbox `sumtensor`. Operations distribute
#' over the parts where a structured implementation exists.
#'
#' @export
SumTensor <- R6::R6Class("SumTensor",
  public = list(
    #' @field parts List of tensor-like parts
    parts = NULL,

    #' @description Initialize a new SumTensor
    #' @param parts List of tensor-like objects with identical dimensions.
    #' @return A new SumTensor object
    initialize = function(parts = NULL) {
      if (is.null(parts)) {
        return(invisible(self))
      }
      if (!is.list(parts) || length(parts) == 0) {
        stop("parts must be a non-empty list of tensor-like objects.")
      }
      dims <- NULL
      for (p in parts) {
        if (!inherits(p, c("Tensor", "KTensor", "TTensor"))) {
          stop("Each part must be a Tensor, KTensor, or TTensor.")
        }
        d <- p$dim()
        if (is.null(dims)) {
          dims <- d
        } else if (!identical(d, dims)) {
          stop("All parts must have the same dimensions.")
        }
      }
      self$parts <- parts
      invisible(self)
    },

    #' @description Get dimensions of the tensor
    #' @return Integer vector of dimensions
    dim = function() {
      self$parts[[1]]$dim()
    },

    #' @description Get number of dimensions
    #' @return Integer tensor order
    ndims = function() {
      length(self$dim())
    },

    #' @description Print the SumTensor object
    #' @param ... Additional arguments
    #' @return Invisible self
    print = function(...) {
      cat("<SumTensor object>\n")
      cat("Dimensions: ", paste(self$dim(), collapse = " x "), "\n")
      cat("Parts: ", length(self$parts), "\n")
      for (i in seq_along(self$parts)) {
        cat("  [[", i, "]] ", class(self$parts[[i]])[1], "\n", sep = "")
      }
      invisible(self)
    },

    #' @description Convert to a dense Tensor by materializing the sum
    #' @return A dense `Tensor`
    full = function() {
      total <- .tensor_as_dense(self$parts[[1]])$clone_tensor()
      for (p in self$parts[-1]) {
        total <- total + .tensor_as_dense(p)
      }
      total
    }
  )
)

#' Create an Implicit Sum of Tensors (SumTensor)
#'
#' @param ... Tensor-like parts (or a single list of parts) with identical
#'   dimensions.
#' @return A `SumTensor` object.
#' @examples
#' X <- tensor(array(rnorm(24), dim = c(2, 3, 4)))
#' K <- ktensor(1, list(matrix(1, 2), matrix(1, 3), matrix(1, 4)))
#' S <- sumtensor(X, K)
#' @rdname SumTensor
#' @export
sumtensor <- function(...) {
  parts <- list(...)
  if (length(parts) == 1 && is.list(parts[[1]]) && !inherits(parts[[1]], "R6")) {
    parts <- parts[[1]]
  }
  res <- SumTensor$new(parts)
  # Inherit "Tensor" so mixed arithmetic dispatches to the (SumTensor-aware)
  # Tensor operators instead of raising an incompatible-methods error.
  class(res) <- c("SumTensor", "Tensor", class(res))
  res
}

#' @export
print.SumTensor <- function(x, ...) {
  x$print(...)
}

#' @export
as.tensor.SumTensor <- function(x, ...) {
  x$full()
}

#' @export
innerprod.SumTensor <- function(x, y, ...) {
  if (inherits(y, "SumTensor")) {
    return(sum(vapply(y$parts, function(p) innerprod(x, p), numeric(1))))
  }
  sum(vapply(x$parts, function(p) innerprod(p, y), numeric(1)))
}

#' @export
fnorm.SumTensor <- function(x, ...) {
  sqrt(max(innerprod(x, x), 0))
}

.sumtensor_plus <- function(e1, e2) {
  as_parts <- function(e) {
    if (inherits(e, "SumTensor")) {
      return(e$parts)
    }
    if (inherits(e, c("Tensor", "KTensor", "TTensor"))) {
      return(list(e))
    }
    stop("Can only add tensor-like objects to a SumTensor.")
  }
  sumtensor(c(as_parts(e1), as_parts(e2)))
}
