#' @include tensor_class.R
NULL

# ------------------------------------------------------------------------------
# Khatri-Rao Product
# ------------------------------------------------------------------------------

#' Khatri-Rao Product
#'
#' Computes the Khatri-Rao product (column-wise Kronecker product) of two matrices,
#' or a list of matrices.
#'
#' @param x A matrix or list of matrices.
#' @param y A matrix (if x is a matrix).
#' @param reverse Logical. If TRUE, computes y \eqn{\odot} x instead of x \eqn{\odot} y.
#' @param ... Additional arguments.
#' @return A matrix representing the Khatri-Rao product.
#'
#' @export
khatri_rao <- function(x, ...) {
  UseMethod("khatri_rao")
}

#' @export
khatri_rao.matrix <- function(x, y, reverse = FALSE, ...) {
  if (!is.matrix(x) || !is.matrix(y)) {
    stop("x and y must be matrices.")
  }
  if (ncol(x) != ncol(y)) {
    stop("Matrices must have the same number of columns.")
  }

  J <- ncol(x)
  I <- nrow(x)
  K <- nrow(y)

  if (reverse) {
    # y (K x J) Khatri-Rao x (I x J) = (K*I) x J
    # For each column j, kron(y[,j], x[,j])
    # x is duplicated K times, y elements are repeated I times
    ret <- matrix(0, nrow = I * K, ncol = J)
    # R subset recycling happens efficiently
    ret[] <- x[rep(1:I, times = K), ] * y[rep(1:K, each = I), ]
  } else {
    # x (I x J) Khatri-Rao y (K x J) = (I*K) x J
    # x elements repeated K times, y is duplicated I times
    ret <- matrix(0, nrow = I * K, ncol = J)
    ret[] <- y[rep(1:K, times = I), ] * x[rep(1:I, each = K), ]
  }

  return(ret)
}

#' @export
khatri_rao.list <- function(x, reverse = FALSE, ...) {
  if (length(x) < 2) {
    stop("List must contain at least two matrices.")
  }

  ret <- x[[1]]
  for (i in 2:length(x)) {
    ret <- khatri_rao(ret, x[[i]], reverse = reverse)
  }
  return(ret)
}

#' @export
khatri_rao.default <- function(x, ...) {
  stop("khatri_rao not implemented for this class.")
}


# ------------------------------------------------------------------------------
# Kronecker Product (extends base::kronecker)
# ------------------------------------------------------------------------------

#' Kronecker Product
#'
#' Extends base::kronecker to support list of matrices.
#'
#' @param X A matrix, Tensor, or list of matrices.
#' @param Y A matrix or Tensor (optional if X is a list).
#' @param FUN The function to use (default "*").
#' @param make.dimnames Logical.
#' @param ... Additional arguments.
#' @return The Kronecker product.
#'
#' @export
kronecker <- function(X, Y = NULL, FUN = "*", make.dimnames = FALSE, ...) {
  UseMethod("kronecker")
}

#' @export
kronecker.default <- function(X, Y = NULL, FUN = "*", make.dimnames = FALSE, ...) {
  if (is.null(Y)) {
    stop("Y is missing, with no default")
  }
  base::kronecker(X, Y, FUN = FUN, make.dimnames = make.dimnames, ...)
}

#' @export
kronecker.list <- function(X, Y = NULL, FUN = "*", make.dimnames = FALSE, ...) {
  if (length(X) < 2) {
    stop("List must contain at least two items.")
  }

  ret <- X[[1]]
  for (i in 2:length(X)) {
    # use base R kronecker for internal
    ret <- base::kronecker(ret, X[[i]], FUN = FUN, make.dimnames = make.dimnames, ...)
  }
  return(ret)
}

#' @export
kronecker.Tensor <- function(X, Y = NULL, FUN = "*", make.dimnames = FALSE, ...) {
  stop("Kronecker product for Tensors operates differently. Use tensor specific methods.")
}


# ------------------------------------------------------------------------------
# Hadamard Product
# ------------------------------------------------------------------------------

#' Hadamard Product
#'
#' Element-wise multiplication of matrices or list of matrices.
#'
#' @param x A matrix or list of matrices.
#' @param y A matrix (if x is a matrix).
#' @param ... Additional arguments.
#' @return A matrix representing the Hadamard product.
#'
#' @export
hadamard <- function(x, ...) {
  UseMethod("hadamard")
}

#' @export
hadamard.matrix <- function(x, y, ...) {
  if (any(dim(x) != dim(y))) {
    stop("Matrices must have the same dimensions for Hadamard product.")
  }
  return(x * y)
}

#' @export
hadamard.list <- function(x, ...) {
  if (length(x) < 2) {
    stop("List must contain at least two items.")
  }

  ret <- x[[1]]
  for (i in 2:length(x)) {
    ret <- hadamard.matrix(ret, x[[i]])
  }
  return(ret)
}

#' @export
hadamard.default <- function(x, ...) {
  stop("hadamard not implemented for this class.")
}


# ------------------------------------------------------------------------------
# Frobenius Norm
# ------------------------------------------------------------------------------

#' Frobenius Norm
#'
#' Calculates the Frobenius norm of a tensor.
#'
#' @param x A tensor or matrix.
#' @param ... Additional arguments.
#' @return A scalar value representing the Frobenius norm.
#'
#' @export
fnorm <- function(x, ...) {
  UseMethod("fnorm")
}

#' @export
fnorm.Tensor <- function(x, ...) {
  return(sqrt(sum((x$data)^2)))
}

#' @export
fnorm.matrix <- function(x, ...) {
  return(sqrt(sum(x^2)))
}

#' @export
fnorm.default <- function(x, ...) {
  stop("fnorm not implemented for this class.")
}


# ------------------------------------------------------------------------------
# Collapse Tensor
# ------------------------------------------------------------------------------

#' Collapse Tensor
#'
#' Collapses a tensor over specified dimensions using an accumulation function
#' (e.g., sum, mean, max).
#'
#' @param x A Tensor object.
#' @param dims Integer vector specifying the dimensions to collapse along.
#'             Negative dimensions exclude those dimensions.
#' @param fun The function to apply (default is sum).
#' @param ... Additional arguments passed to \code{fun}.
#' @return A collapsed Tensor.
#'
#' @export
collapse <- function(x, ...) {
  UseMethod("collapse")
}

#' @export
collapse.Tensor <- function(x, dims, fun = sum, ...) {
  tensor_dims <- x$dim()
  num_dims <- length(tensor_dims)

  if (all(dims < 0)) {
    # Keep the negative dimensions, collapse the others
    keep_dims <- abs(dims)
  } else if (all(dims > 0)) {
    # Collapse positive dimensions
    keep_dims <- setdiff(1:num_dims, dims)
  } else {
    stop("dims must be all positive or all negative integers.")
  }

  if (length(keep_dims) == 0) {
    # Collapse all dimensions
    res <- fun(x$data, ...)
    return(tensor(res, 1))
  }

  # apply collapses the dimensions and leaves the keep_dims
  # R apply will output a matrix/array with dimensions in order of keep_dims
  res <- base::apply(x$data, keep_dims, fun, ...)

  # base::apply drops dimension sizes of 1 if there is only 1 keep_dim
  if (is.null(dim(res))) {
    # It returned a vector
    res <- array(res, dim = length(res))
  }

  return(tensor(res))
}

#' @export
collapse.default <- function(x, ...) {
  stop("collapse not implemented for this class.")
}


# ------------------------------------------------------------------------------
# Scale Tensor
# ------------------------------------------------------------------------------

#' Scale Tensor
#'
#' Scales a tensor along specified dimensions. Equivalent to broadcasting
#' a scaling vector or tensor for multiplication.
#'
#' @param x A Tensor.
#' @param s A vector or Tensor containing scaling factors.
#' @param dims Dimensions to scale.
#' @param ... Additional arguments.
#' @return A scaled Tensor.
#'
#' @export
t_scale <- function(x, ...) {
  UseMethod("t_scale")
}

#' @export
t_scale.Tensor <- function(x, s, dims, ...) {
  # Input shapes
  tensor_dims <- x$dim()
  num_dims <- length(tensor_dims)

  if (all(dims < 0)) {
    dims <- setdiff(1:num_dims, abs(dims))
  } else if (any(dims < 0)) {
    stop("dims must be all positive or all negative integers.")
  }

  # Validate scaling factor
  if (inherits(s, "Tensor")) {
    s_data <- s$as_array()
  } else {
    s_data <- s
  }

  if (length(s_data) != prod(tensor_dims[dims])) {
    stop("Number of elements in 's' must match the product of the scaled dimensions.")
  }

  # Use R's sweep and margin capabilities.
  # Note: sweep applies a function along margins. Array broadcasting is done essentially
  # by sweeping over the required dimensions.

  s_arr <- array(s_data, dim = tensor_dims[dims])

  # If we scale all dimensions, it is just elementwise
  if (length(dims) == length(tensor_dims)) {
    return(tensor(x$data * s_arr))
  }

  # For sweeping an array of sizes, we need 'sweep' with FUN="*".
  # However, base 'sweep' expects STATS to conform exactly to MARGIN.
  # Let's use it.
  res_data <- base::sweep(x$data, MARGIN = dims, STATS = s_arr, FUN = "*")

  return(tensor(res_data))
}

#' @export
t_scale.default <- function(x, ...) {
  stop("t_scale not implemented for this class.")
}
