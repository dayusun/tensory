#' @include tensor_class.R ktensor_class.R
NULL

#' R6 Class for Symmetric Kruskal Tensors (SymKTensor)
#'
#' @description
#' A symmetric Kruskal tensor is a sum of symmetric rank-one terms
#' `lambda_r * u_r o u_r o ... o u_r` (`m` copies of the same vector),
#' mirroring the MATLAB Tensor Toolbox `symktensor`. It is represented by a
#' weight vector, a single factor matrix shared by all modes, and the order.
#'
#' @export
SymKTensor <- R6::R6Class("SymKTensor",
  public = list(
    #' @field lambda Vector of weights
    lambda = NULL,

    #' @field u Shared factor matrix (`n x R`)
    u = NULL,

    #' @field m Tensor order
    m = NULL,

    #' @description Initialize a new SymKTensor
    #' @param lambda Numeric vector of weights.
    #' @param u Shared factor matrix with one column per component.
    #' @param m Tensor order (positive integer).
    #' @return A new SymKTensor object
    initialize = function(lambda = NULL, u = NULL, m = NULL) {
      if (is.null(lambda) && is.null(u)) {
        return(invisible(self))
      }
      u <- as.matrix(u)
      m <- as.integer(m)
      if (length(m) != 1 || m < 1) {
        stop("m must be a single positive integer.")
      }
      if (length(lambda) != ncol(u)) {
        stop("Length of lambda must match the number of columns of u.")
      }
      self$lambda <- as.numeric(lambda)
      self$u <- u
      self$m <- m
      invisible(self)
    },

    #' @description Get dimensions of the tensor
    #' @return Integer vector of dimensions (`m` copies of `nrow(u)`)
    dim = function() {
      rep(nrow(self$u), self$m)
    },

    #' @description Get number of dimensions
    #' @return Integer tensor order
    ndims = function() {
      self$m
    },

    #' @description Print the SymKTensor object
    #' @param ... Additional arguments
    #' @return Invisible self
    print = function(...) {
      cat("<SymKTensor object>\n")
      cat("Order (m): ", self$m, "\n")
      cat("Size (n): ", nrow(self$u), "\n")
      cat("Weights (lambda): ", paste(round(self$lambda, 4), collapse = " "), "\n")
      invisible(self)
    },

    #' @description Convert to an ordinary Kruskal tensor
    #' @return A `KTensor` with the shared factor repeated in every mode
    as_ktensor = function() {
      ktensor(self$lambda, rep(list(self$u), self$m))
    },

    #' @description Convert to a dense Tensor
    #' @return A dense `Tensor`
    full = function() {
      self$as_ktensor()$full()
    }
  )
)

#' Create a Symmetric Kruskal Tensor (SymKTensor)
#'
#' @param lambda Numeric vector of weights.
#' @param u Shared factor matrix with one column per component.
#' @param m Tensor order.
#' @return A `SymKTensor` object.
#' @examples
#' S <- symktensor(c(1, 2), matrix(rnorm(6), 3, 2), m = 3)
#' @rdname SymKTensor
#' @export
symktensor <- function(lambda = NULL, u = NULL, m = NULL) {
  res <- SymKTensor$new(lambda, u, m)
  class(res) <- c("SymKTensor", class(res))
  res
}

#' @export
print.SymKTensor <- function(x, ...) {
  x$print(...)
}

#' @export
as.tensor.SymKTensor <- function(x, ...) {
  x$full()
}

#' @export
fnorm.SymKTensor <- function(x, ...) {
  fnorm(x$as_ktensor())
}

#' @export
ncomponents.SymKTensor <- function(x, ...) {
  length(x$lambda)
}
