#' @include tensor_class.R tensor_dense_methods.R
NULL

# Nondecreasing index tuples of length m over 1..n, one per column, in
# lexicographic order (the standard combination-with-repetition bijection).
.sym_index_tuples <- function(n, m) {
  comb <- utils::combn(n + m - 1L, m)
  comb - (seq_len(m) - 1L)
}

.sym_key <- function(subs) {
  apply(subs, 2L, paste, collapse = ",")
}

#' R6 Class for Symmetric Tensors (SymTensor)
#'
#' @description
#' Stores a symmetric tensor compactly by keeping only one value per distinct
#' (sorted) index class, mirroring the MATLAB Tensor Toolbox `symtensor`.
#' An order-`m`, size-`n` symmetric tensor stores `choose(n + m - 1, m)`
#' values instead of `n^m`.
#'
#' @export
SymTensor <- R6::R6Class("SymTensor",
  public = list(
    #' @field vals Values, one per distinct sorted index class
    vals = NULL,

    #' @field m Tensor order
    m = NULL,

    #' @field n Size of each mode
    n = NULL,

    #' @description Initialize a new SymTensor from a dense symmetric tensor
    #' @param x A symmetric `Tensor` (all modes the same size).
    #' @return A new SymTensor object
    initialize = function(x = NULL) {
      if (is.null(x)) {
        return(invisible(self))
      }
      x <- .tensor_as_dense(x)
      dims <- x$dim()
      if (length(dims) < 1 || !all(dims == dims[1])) {
        stop("x must have all modes of the same size.")
      }
      if (!issymmetric(x)) {
        stop("x must be symmetric; use symmetrize() first if needed.")
      }
      self$m <- length(dims)
      self$n <- dims[1]
      tuples <- .sym_index_tuples(self$n, self$m)
      self$vals <- x$data[t(tuples)]
      invisible(self)
    },

    #' @description Get dimensions of the tensor
    #' @return Integer vector of dimensions
    dim = function() {
      rep(self$n, self$m)
    },

    #' @description Get number of dimensions
    #' @return Integer tensor order
    ndims = function() {
      self$m
    },

    #' @description Print the SymTensor object
    #' @param ... Additional arguments
    #' @return Invisible self
    print = function(...) {
      cat("<SymTensor object>\n")
      cat("Order (m): ", self$m, "\n")
      cat("Size (n): ", self$n, "\n")
      cat("Stored values: ", length(self$vals), " (dense would be ",
          self$n^self$m, ")\n", sep = "")
      invisible(self)
    },

    #' @description Convert to a dense Tensor
    #' @return A dense `Tensor`
    full = function() {
      dims <- rep(self$n, self$m)
      tuples <- .sym_index_tuples(self$n, self$m)
      lookup <- stats::setNames(self$vals, .sym_key(tuples))

      all_subs <- arrayInd(seq_len(prod(dims)), .dim = dims)
      sorted <- t(apply(all_subs, 1L, sort))
      keys <- apply(sorted, 1L, paste, collapse = ",")
      data <- array(as.numeric(lookup[keys]), dim = dims)
      Tensor$new(data, dims = as.integer(dims), fast = TRUE)
    }
  )
)

#' Create a Symmetric Tensor (SymTensor)
#'
#' Compactly stores a symmetric tensor using one value per distinct index
#' class, mirroring the MATLAB Tensor Toolbox `symtensor`.
#'
#' @param x A symmetric `Tensor`, or any tensor if `symmetrize = TRUE`.
#' @param symmetrize Logical; if `TRUE` the input is symmetrized first.
#' @return A `SymTensor` object.
#' @examples
#' X <- symmetrize(tensor(array(rnorm(27), dim = c(3, 3, 3))))
#' S <- symtensor(X)
#' @rdname SymTensor
#' @export
symtensor <- function(x, symmetrize = FALSE) {
  if (isTRUE(symmetrize)) {
    x <- tensory::symmetrize(.tensor_as_dense(x))
  }
  res <- SymTensor$new(x)
  class(res) <- c("SymTensor", class(res))
  res
}

#' @export
print.SymTensor <- function(x, ...) {
  x$print(...)
}

#' @export
as.tensor.SymTensor <- function(x, ...) {
  x$full()
}

#' @export
issymmetric.SymTensor <- function(x, grps = NULL, ...) {
  TRUE
}

#' @export
fnorm.SymTensor <- function(x, ...) {
  fnorm(x$full())
}

#' @export
ttsv.SymTensor <- function(x, v, n = 0, ...) {
  ttsv(x$full(), v = v, n = n, ...)
}
